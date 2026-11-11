from __future__ import annotations

import hashlib
import os
import re
import shlex
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from pygent.agent.pygent_v1.core.io import plain_data
from pygent.agent.pygent_v1.core.redaction import redact_secrets
from pygent.agent.pygent_v1.tracing import (
    DiffRecorder,
    EventStore,
    read_snapshot_content,
)
from pygent.runtime.codec import tool_task_to_dict
from pygent.tool import ToolResult as PygentToolResult

from .file_effect_models import DeferredFileEffectJob, FileEffect, FileSnapshot

MAX_BASH_RESULT_CHARS = 20_000
MAX_BASH_RESULT_LINES = 200
BASH_RESULT_PREVIEW_LINES = 120
BASH_RESULT_PREVIEW_CHARS = 12_000
SPOOLED_TEXT_TOOL_NAMES = frozenset({"bash", "grep"})


class SnapshotBudgetExceeded(RuntimeError):
    """The scan stopped cooperatively; no partial baseline may be committed."""


class FileEffectTracker:
    IGNORED_DIRS = frozenset(
        {".git", ".pygent_v1", ".venv", "__pycache__", ".pytest_cache", "sessions",
         ".tmp", "node_modules", ".npm-cache", ".ruff_cache", ".vite",
         "build", "dist"}
    )
    PATH_ARG_NAMES = ("file_path",)
    WRITE_TYPES = frozenset({"file.write", "file.edit", "file.delete"})

    def __init__(
        self,
        workspace_root: str | Path,
        store: EventStore,
        *,
        allow_read_outside_workspace: bool = True,
    ):
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.store = store
        self.allow_read_outside_workspace = allow_read_outside_workspace
        self._appended_keys: set[tuple[Any, ...]] = set()

    SNAPSHOT_MAX_SECONDS = 5.0
    SNAPSHOT_MAX_FILES = 20_000
    SNAPSHOT_MAX_BYTES = 64 * 1024 * 1024

    def _workspace_files(self, deadline: float) -> Iterable[Path]:
        # Prune before descending: filtering rglob results still traverses caches.
        def on_error(error: OSError) -> None:
            raise SnapshotBudgetExceeded("directory unavailable during scan") from error

        for root, directories, files in os.walk(self.workspace_root, followlinks=False, onerror=on_error):
            if time.monotonic() >= deadline:
                raise SnapshotBudgetExceeded("time budget exceeded")
            directories[:] = [
                name for name in directories
                if not self._is_ignored(Path(root) / name)
                and not (Path(root) / name).is_symlink()
                and not _path_is_junction(Path(root) / name)
            ]
            for name in files:
                yield Path(root) / name

    def snapshot_workspace(
        self, paths: Iterable[Path] | None = None,
    ) -> dict[str, FileSnapshot]:
        snapshots: dict[str, FileSnapshot] = {}
        deadline = time.monotonic() + self.SNAPSHOT_MAX_SECONDS
        remaining = self.SNAPSHOT_MAX_BYTES
        count = 0

        def check_budget() -> None:
            if time.monotonic() >= deadline:
                raise SnapshotBudgetExceeded("time budget exceeded")

        candidates = self._workspace_files(deadline) if paths is None else paths
        for path in candidates:
            check_budget()
            if paths is None and self._is_ignored(path):
                continue
            if path.is_symlink():
                continue
            try:
                stat = path.stat()
                if not path.is_file():
                    continue
                count += 1
                if count > self.SNAPSHOT_MAX_FILES:
                    raise SnapshotBudgetExceeded("file count budget exceeded")
                if stat.st_size > remaining:
                    raise SnapshotBudgetExceeded("byte budget exceeded")
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    while True:
                        check_budget()
                        chunk = handle.read(min(1024 * 1024, remaining + 1))
                        remaining -= len(chunk)
                        if remaining < 0:
                            raise SnapshotBudgetExceeded("byte budget exceeded")
                        if not chunk:
                            break
                        digest.update(chunk)
                check_budget()
                # Content capture performs another read, so reserve its bytes too.
                if stat.st_size > remaining:
                    raise SnapshotBudgetExceeded("byte budget exceeded")
                remaining -= stat.st_size
                content = read_snapshot_content(path)
                check_budget()
            except FileNotFoundError as exc:
                if paths is not None:
                    continue
                raise SnapshotBudgetExceeded("file disappeared during scan") from exc
            except OSError as exc:
                raise SnapshotBudgetExceeded("file unavailable during scan") from exc
            normalized = str(path.resolve())
            snapshots[normalized] = FileSnapshot(
                path=normalized, exists=True, kind="file", size=stat.st_size,
                mtime_ns=stat.st_mtime_ns, content_hash=digest.hexdigest(),
                content=content.content, content_available=content.available,
                content_unavailable_reason=content.reason,
            )
        return snapshots

    def declared_effects(
        self,
        tool_name: str,
        args: dict[str, Any],
        tool_call_id: str,
    ) -> list[FileEffect]:
        if tool_name == "bash":
            return self._bash_read_effects(args, tool_call_id)
        path = self._path_from_args(
            args,
            allow_outside_workspace=tool_name in {"write", "edit"}
            or (tool_name == "read" and self.allow_read_outside_workspace),
        )
        if path is None:
            return []
        if tool_name == "read":
            return [
                FileEffect(
                    type="file.read",
                    path=path,
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    detected_by=["tool_args"],
                    confidence="declared",
                )
            ]
        if tool_name == "write":
            return [
                FileEffect(
                    type="file.write",
                    path=path,
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    detected_by=["tool_args"],
                    confidence="declared",
                )
            ]
        if tool_name == "edit":
            return [
                FileEffect(
                    type="file.edit",
                    path=path,
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    detected_by=["tool_args"],
                    confidence="declared",
                )
            ]
        return []

    def observed_effects(
        self,
        before: dict[str, FileSnapshot],
        after: dict[str, FileSnapshot],
        *,
        tool_name: str,
        tool_call_id: str,
    ) -> list[FileEffect]:
        effects: list[FileEffect] = []
        for path in sorted(set(before) | set(after)):
            before_snapshot = before.get(path)
            after_snapshot = after.get(path)
            if before_snapshot is None and after_snapshot is not None:
                effects.append(
                    FileEffect(
                        type="file.write",
                        path=path,
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        detected_by=["snapshot_diff"],
                        confidence="observed",
                        after_hash=after_snapshot.content_hash,
                        before_exists=False,
                        after_exists=True,
                        after_content=after_snapshot.content,
                        after_content_available=after_snapshot.content_available,
                        after_content_unavailable_reason=after_snapshot.content_unavailable_reason,
                    )
                )
            elif before_snapshot is not None and after_snapshot is None:
                effects.append(
                    FileEffect(
                        type="file.delete",
                        path=path,
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        detected_by=["snapshot_diff"],
                        confidence="observed",
                        before_hash=before_snapshot.content_hash,
                        before_exists=True,
                        after_exists=False,
                        before_content=before_snapshot.content,
                        before_content_available=before_snapshot.content_available,
                        before_content_unavailable_reason=before_snapshot.content_unavailable_reason,
                    )
                )
            elif (
                before_snapshot is not None
                and after_snapshot is not None
                and before_snapshot.content_hash != after_snapshot.content_hash
            ):
                effects.append(
                    FileEffect(
                        type="file.edit",
                        path=path,
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        detected_by=["snapshot_diff"],
                        confidence="observed",
                        before_hash=before_snapshot.content_hash,
                        after_hash=after_snapshot.content_hash,
                        before_exists=True,
                        after_exists=True,
                        before_content=before_snapshot.content,
                        after_content=after_snapshot.content,
                        before_content_available=before_snapshot.content_available,
                        after_content_available=after_snapshot.content_available,
                        before_content_unavailable_reason=before_snapshot.content_unavailable_reason,
                        after_content_unavailable_reason=after_snapshot.content_unavailable_reason,
                    )
                )
        return sorted(
            effects, key=lambda effect: (_effect_type_order(effect.type), effect.path)
        )

    def merge_effects(
        self,
        declared: list[FileEffect],
        observed: list[FileEffect],
    ) -> list[FileEffect]:
        observed_by_path = {effect.path: effect for effect in observed}
        declared_by_path: dict[str, FileEffect] = {}
        for effect in declared:
            declared_by_path.setdefault(effect.path, effect)

        merged: list[FileEffect] = []
        for path in sorted(set(declared_by_path) | set(observed_by_path)):
            declared_effect = declared_by_path.get(path)
            observed_effect = observed_by_path.get(path)
            if observed_effect is None:
                if declared_effect is not None:
                    merged.append(declared_effect)
                continue

            if declared_effect is None:
                merged.append(observed_effect)
                continue

            detected_by = _merge_detected_by(
                declared_effect.detected_by, observed_effect.detected_by
            )
            merged.append(
                FileEffect(
                    type=observed_effect.type,
                    path=observed_effect.path,
                    tool_call_id=declared_effect.tool_call_id,
                    tool_name=declared_effect.tool_name,
                    detected_by=detected_by,
                    confidence=observed_effect.confidence,
                    before_hash=observed_effect.before_hash,
                    after_hash=observed_effect.after_hash,
                    before_exists=observed_effect.before_exists,
                    after_exists=observed_effect.after_exists,
                    before_content=observed_effect.before_content,
                    after_content=observed_effect.after_content,
                    before_content_available=observed_effect.before_content_available,
                    after_content_available=observed_effect.after_content_available,
                    before_content_unavailable_reason=observed_effect.before_content_unavailable_reason,
                    after_content_unavailable_reason=observed_effect.after_content_unavailable_reason,
                )
            )

        write_seen: set[str] = set()
        deduped: list[FileEffect] = []
        for effect in merged:
            if effect.type in self.WRITE_TYPES:
                if effect.path in write_seen:
                    continue
                write_seen.add(effect.path)
            deduped.append(effect)
        return sorted(
            deduped, key=lambda effect: (_effect_type_order(effect.type), effect.path)
        )

    def append_effects(
        self,
        effects: list[FileEffect],
        *,
        turn_id: str | None,
    ) -> None:
        diff_recorder = DiffRecorder(self.store, self.workspace_root)
        for effect in effects:
            key = effect.key()
            if key in self._appended_keys:
                continue
            self._appended_keys.add(key)
            self.store.append(
                effect.type,
                actor="tool",
                payload={
                    "path": effect.path,
                    "tool_call_id": effect.tool_call_id,
                    "tool_name": effect.tool_name,
                    "detected_by": effect.detected_by,
                    "confidence": effect.confidence,
                    "before_exists": effect.before_exists,
                    "after_exists": effect.after_exists,
                    "before_hash": effect.before_hash,
                    "after_hash": effect.after_hash,
                    "content_hash": effect.after_hash or effect.before_hash,
                },
                turn_id=turn_id,
            )
            diff_recorder.record_effect(effect, turn_id=turn_id)

    def _path_from_args(
        self, args: dict[str, Any], *, allow_outside_workspace: bool = False
    ) -> str | None:
        for name in self.PATH_ARG_NAMES:
            value = args.get(name)
            if isinstance(value, str) and value:
                return self._resolve_workspace_path(
                    value, allow_outside_workspace=allow_outside_workspace
                )
        return None

    def _resolve_workspace_path(
        self, path: str | Path, *, allow_outside_workspace: bool = False
    ) -> str:
        candidate = Path(_normalize_user_path(path)).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace_root / candidate
        resolved = candidate.resolve()
        if not _is_relative_to(resolved, self.workspace_root) and not (
            allow_outside_workspace
        ):
            raise ValueError(f"Path is outside workspace: {path}")
        return str(resolved)

    def _is_ignored(self, path: Path) -> bool:
        try:
            relative = path.relative_to(self.workspace_root)
        except ValueError:
            return True
        return any(part in self.IGNORED_DIRS or part.startswith(".venv-") for part in relative.parts)

    def _bash_read_effects(
        self, args: dict[str, Any], tool_call_id: str
    ) -> list[FileEffect]:
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            return []
        paths = self._parse_bash_read_paths(command)
        effects: list[FileEffect] = []
        seen: set[str] = set()
        for raw_path in paths:
            try:
                path = self._resolve_workspace_path(
                    raw_path,
                    allow_outside_workspace=self.allow_read_outside_workspace,
                )
            except ValueError:
                # Bash has already executed by the time audit runs. An
                # inferred external read (for example a configured CLI
                # executable) must not retroactively fail an otherwise valid
                # tool call when outside-workspace read tracking is disabled.
                # Explicit read tool paths remain fail-closed above.
                continue
            if path in seen:
                continue
            seen.add(path)
            effects.append(
                FileEffect(
                    type="file.read",
                    path=path,
                    tool_call_id=tool_call_id,
                    tool_name="bash",
                    detected_by=["bash_command_parse"],
                    confidence="inferred",
                )
            )
        return effects

    @staticmethod
    def _parse_bash_read_paths(command: str) -> list[str]:
        paths: list[str] = []
        separators = ("&&", "||", ";", "|")
        normalized = command
        for separator in separators:
            normalized = normalized.replace(separator, "\n")
        for line in normalized.splitlines():
            try:
                tokens = shlex.split(line, posix=True)
            except ValueError:
                try:
                    tokens = shlex.split(line, posix=False)
                except ValueError:
                    continue
            if not tokens:
                continue
            executable = Path(tokens[0]).name.lower()
            if executable in {"cat", "type"}:
                paths.extend(_non_option_tokens(tokens[1:]))
            elif executable == "grep":
                candidates = _non_option_tokens(tokens[1:])
                if len(candidates) >= 2:
                    paths.extend(candidates[1:])
            elif executable in {"get-content", "gc"}:
                paths.extend(_powershell_path_args(tokens[1:]))
        return paths


class ToolObserver:
    def __init__(
        self,
        store: EventStore,
        *,
        workspace_root: str | Path | None = None,
        track_file_effects: bool = False,
        defer_file_effects: bool = False,
        allow_read_outside_workspace: bool = True,
        bash_full_output_allowlist: list[str] | None = None,
    ):
        self.store = store
        self.bash_full_output_allowlist = list(bash_full_output_allowlist or [])
        self.defer_file_effects = defer_file_effects
        if track_file_effects and workspace_root is None:
            raise ValueError("workspace_root is required when track_file_effects=True")
        self.file_effect_tracker = (
            FileEffectTracker(
                workspace_root=workspace_root,
                store=store,
                allow_read_outside_workspace=allow_read_outside_workspace,
            )
            if track_file_effects and workspace_root is not None
            else None
        )

    def record_framework_result(
        self,
        name: str,
        args: dict[str, Any],
        turn_id: str | None,
        result: PygentToolResult,
        *,
        available_tools: tuple[str, ...] = (),
        audit_call_id: str | None = None,
    ) -> tuple[dict[str, Any], DeferredFileEffectJob | None]:
        """Persist pygent_v1 audit facts after Pygent has executed a tool."""

        call_id = audit_call_id or self.store.append(
            "tool.call",
            actor="assistant",
            payload={
                "tool_name": name,
                "args": args,
                "model_tool_call_id": result.call_id,
            },
            turn_id=turn_id,
        )
        declared = (
            self.file_effect_tracker.declared_effects(name, args, call_id)
            if self.file_effect_tracker is not None
            else []
        )
        deferred_job: DeferredFileEffectJob | None = None
        if self.defer_file_effects and result.status != "detached":
            requires_snapshot = _tool_requires_deferred_snapshot(name, args)
            if (
                result.status == "succeeded"
                and declared
                and not requires_snapshot
                and self.file_effect_tracker is not None
            ):
                self.file_effect_tracker.append_effects(declared, turn_id=turn_id)
            else:
                deferred_job = self._deferred_file_effect_job(
                    tool_call_id=call_id,
                    tool_name=name,
                    args=args,
                    turn_id=turn_id,
                    declared=declared,
                    include_declared=result.status == "succeeded",
                )

        if result.status == "detached":
            payload = {
                "tool_call_id": call_id,
                "status": "running",
                "framework_status": "detached",
                "result": self._spool_output(name, args, result.output, call_id),
                "task": tool_task_to_dict(result.task),
                "next_action": "Use tool_task_get(task_id) to read output or tool_task_stop(task_id) to request cancellation.",
            }
        elif result.status == "succeeded":
            value = self._spool_output(name, args, result.output, call_id)
            payload = {"tool_call_id": call_id, "status": "success", "result": value}
        else:
            payload = {
                "tool_call_id": call_id,
                "status": "error",
                "error": result.error or f"tool {result.status}",
                "error_type": result.error_kind or "ToolError",
                "details": {
                    "framework_status": result.status,
                    "error_code": result.error_code,
                    "retryable": result.retryable,
                    "side_effect_committed": result.side_effect_committed,
                    "available_tools": list(available_tools),
                },
            }
        payload["model_tool_call_id"] = result.call_id
        if result.status not in {"succeeded", "detached"} and result.output is not None:
            payload["result"] = self._spool_output(name, args, result.output, call_id)
        if audit_call_id is not None:
            payload["task"] = tool_task_to_dict(result.task)
            payload["framework_status"] = result.status
        self.store.append("tool.result", actor="tool", payload=payload, turn_id=turn_id)
        return payload, deferred_job

    def _spool_output(self, name: str, args: dict[str, Any], output: Any, call_id: str) -> Any:
        def spool(value: Any, suffix: str = "") -> Any:
            return _spool_large_tool_result(
                tool_name="bash" if name in {"tool_task_get", "tool_task_stop"} else name,
                args=args, result=value, tool_call_id=f"{call_id}{suffix}",
                run_dir=self.store.run_dir,
                bash_full_output_allowlist=self.bash_full_output_allowlist,
            )

        value = plain_data(output)
        if name in {"tool_task_get", "tool_task_stop"} and isinstance(value, dict):
            value = dict(value)
            if "output" in value:
                value["output"] = spool(value["output"], "-snapshot")
            if isinstance(value.get("result"), dict):
                final = dict(value["result"])
                if "output" in final:
                    final["output"] = spool(final["output"], "-final")
                value["result"] = final
            return value
        return spool(value)

    def _deferred_file_effect_job(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        args: dict[str, Any],
        turn_id: str | None,
        declared: list[FileEffect],
        include_declared: bool = True,
    ) -> DeferredFileEffectJob | None:
        requires_snapshot = _tool_requires_deferred_snapshot(tool_name, args)
        if not requires_snapshot and (not include_declared or not declared):
            return None
        return DeferredFileEffectJob(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            args=dict(args),
            turn_id=turn_id,
            declared=declared,
            include_declared=include_declared,
            requires_snapshot=requires_snapshot,
        )


def _normalize_user_path(path: str | Path) -> str | Path:
    if os.name != "nt" or not isinstance(path, str):
        return path
    value = path.replace("\\", "/")
    parts = value.split("/")
    drive: str | None = None
    tail: list[str] = []
    if len(parts) >= 2 and parts[0] == "" and len(parts[1]) == 1 and parts[1].isalpha():
        drive = parts[1]
        tail = parts[2:]
    elif (
        len(parts) >= 3
        and parts[0] == ""
        and parts[1].lower() in {"mnt", "cygdrive"}
        and len(parts[2]) == 1
        and parts[2].isalpha()
    ):
        drive = parts[2]
        tail = parts[3:]
    if drive is None:
        return path
    suffix = "/".join(part for part in tail if part)
    return f"{drive.upper()}:/{suffix}" if suffix else f"{drive.upper()}:/"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _tool_requires_deferred_snapshot(tool_name: str, args: dict[str, Any]) -> bool:
    if tool_name in {"write", "edit"}:
        return True
    if tool_name != "bash":
        return False
    command = args.get("command")
    return isinstance(command, str) and _bash_command_may_write(command)


def _shell_syntax(command: str) -> str:
    """Keep shell syntax, masking quoted strings, comments and heredoc bodies.

    This is an observation heuristic, not a shell security boundary.
    """
    lines: list[str] = []
    delimiters: list[tuple[str, bool]] = []
    for line in command.splitlines(keepends=True):
        if delimiters:
            delimiter, strip_tabs = delimiters[0]
            candidate = line.lstrip("\t") if strip_tabs else line
            if candidate.rstrip("\r\n") == delimiter:
                delimiters.pop(0)
            continue
        # Preserve the command line (including real output redirects).
        lines.append(line)
        for match in re.finditer(
            r"'(?:[^']*)'|\"(?:\\.|[^\"\\])*\"|\\.|#[^\n]*|"
            r"<<(-?)[ \t]*(?:'([^']+)'|\"([^\"]+)\"|([A-Za-z_][\w]*))", line
        ):
            if match[1] is None:
                continue
            delimiters.append((next(x for x in match.groups()[1:] if x is not None), bool(match[1])))
    text = "".join(lines)
    return re.sub(r"'(?:[^']*)'|\"(?:\\.|[^\"\\])*\"|\\.|#[^\n]*", " ", text)


def _path_is_junction(path: Path) -> bool:
    """Junctions only exist on Windows; other runtimes have nothing to skip."""
    is_junction = getattr(path, "is_junction", None)
    if is_junction is None:
        return False
    return bool(is_junction())


def _bash_command_may_write(command: str) -> bool:
    # Quoted shell programs are executable code, unlike quoted echo/print data.
    try:
        words = shlex.split(command)
    except ValueError:
        words = []
    for index, word in enumerate(words[:-2]):
        if (
            word.rsplit("/", 1)[-1] in {"sh", "bash", "zsh"}
            and words[index + 1] == "-c"
            and _bash_command_may_write(words[index + 2])
        ):
            return True
    command = _shell_syntax(command)
    lowered = command.lower()
    if re.search(r"(?<!>)>{1,2}(?![>&])", command):
        return True
    if re.search(
        r"\b(cat\s+>|tee|touch|mkdir|rm|rmdir|del|move|mv|copy|cp)\b", lowered
    ):
        return True
    if re.search(r"\bsed\b[^;&|]*\s-i\b", lowered):
        return True
    if re.search(r"\b(?:npm|pnpm|yarn)\s+(?:install|add)\b", lowered):
        return not re.search(r"(?:^|\s)(?:-g|--global)(?:\s|$)", lowered)
    return False


def _merge_detected_by(
    left: list[Literal["tool_args", "snapshot_diff", "bash_command_parse"]],
    right: list[Literal["tool_args", "snapshot_diff", "bash_command_parse"]],
) -> list[Literal["tool_args", "snapshot_diff", "bash_command_parse"]]:
    order: tuple[Literal["tool_args", "snapshot_diff", "bash_command_parse"], ...] = (
        "tool_args",
        "snapshot_diff",
        "bash_command_parse",
    )
    values = {*left, *right}
    return [value for value in order if value in values]


def _effect_type_order(effect_type: str) -> int:
    return {"file.write": 0, "file.edit": 1, "file.delete": 2, "file.read": 3}.get(
        effect_type, 99
    )


def _non_option_tokens(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token and not token.startswith("-")]


def _powershell_path_args(tokens: list[str]) -> list[str]:
    paths: list[str] = []
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        lower = token.lower()
        if lower in {"-path", "-literalpath"}:
            skip_next = False
            continue
        if lower.startswith("-"):
            skip_next = lower in {"-encoding", "-totalcount", "-tail", "-readcount"}
            continue
        paths.append(token)
    return paths


def _spool_large_tool_result(
    *,
    tool_name: str,
    args: dict[str, Any],
    result: Any,
    tool_call_id: str,
    run_dir: Path,
    bash_full_output_allowlist: list[str],
) -> Any:
    if tool_name not in SPOOLED_TEXT_TOOL_NAMES or not isinstance(result, str):
        return result
    if tool_name == "bash":
        command = args.get("command")
        if isinstance(command, str) and _bash_command_matches_allowlist(
            command, bash_full_output_allowlist
        ):
            return result
    lines = result.splitlines()
    if len(result) <= MAX_BASH_RESULT_CHARS and len(lines) <= MAX_BASH_RESULT_LINES:
        return result

    output_dir = run_dir / "tool_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{tool_call_id}.txt"
    spooled_text = redact_secrets(result)
    output_path.write_text(spooled_text, encoding="utf-8")

    preview = "\n".join(lines[:BASH_RESULT_PREVIEW_LINES])
    if len(preview) > BASH_RESULT_PREVIEW_CHARS:
        preview = preview[:BASH_RESULT_PREVIEW_CHARS].rstrip()

    return {
        "status": "truncated",
        "truncated": True,
        "reason": f"{tool_name} output exceeded the model-visible result limit",
        "char_count": len(result),
        "line_count": len(lines),
        "preview_line_count": len(preview.splitlines()),
        "preview": preview,
        "full_output_path": str(output_path.resolve()),
        "next_step": (
            f"Full {tool_name} output was written to full_output_path. "
            "Use the read tool with file_path=full_output_path and offset/limit "
            "to inspect additional chunks only if needed."
        ),
    }


def _bash_command_matches_allowlist(command: str, allowlist: list[str]) -> bool:
    command = command.strip()
    if not command:
        return False
    for raw_entry in allowlist:
        entry = raw_entry.strip()
        if not entry:
            continue
        if command == entry or command.startswith(f"{entry} "):
            return True
    return False
