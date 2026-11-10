from __future__ import annotations

import hashlib
import json
import subprocess
import time
from html import escape
from pathlib import Path
from typing import Any

from pygent.agent.lara.core.prompt import Prompt
from pygent.agent.lara.graph.common import _now

from .models import ReminderSection

GIT_STATUS_TIMEOUT_SECONDS = 0.5
GIT_STATUS_MAX_LINES = 100
GIT_STATUS_MAX_CHARS = 4096
GIT_CHECK_MIN_INTERVAL_SECONDS = 10.0
GIT_CHECK_MAX_INTERVAL_SECONDS = 60.0
GIT_CHECK_DURATION_MULTIPLIER = 20.0


def _run_git(workspace_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(workspace_root), *args],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=GIT_STATUS_TIMEOUT_SECONDS,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def capture_git_snapshot(
    workspace_root: Path,
    *,
    repository_root: str = "",
) -> dict[str, Any] | None:
    resolved_workspace = workspace_root.expanduser().resolve()
    resolved_repository = repository_root.strip()
    if not resolved_repository:
        discovered = _run_git(resolved_workspace, "rev-parse", "--show-toplevel")
        if not discovered:
            return None
        resolved_repository = str(Path(discovered).expanduser().resolve())
    status = _run_git(
        Path(resolved_repository),
        "status",
        "--short",
        "--branch",
        "--no-ahead-behind",
        "--untracked-files=normal",
        "--",
        ".",
        ":(exclude).lara",
    )
    if status is None:
        return None
    source_lines = status.splitlines()
    branch_line = next((line for line in source_lines if line.startswith("## ")), "")
    entries: dict[str, dict[str, str | int]] = {}
    for line in source_lines:
        if line.startswith("## ") or len(line) < 4:
            continue
        path_text = line[3:]
        stat_path_text = path_text.rsplit(" -> ", 1)[-1]
        signature = "unavailable"
        if not stat_path_text.startswith('"'):
            candidate = Path(resolved_repository) / stat_path_text
            try:
                file_stat = candidate.stat()
                signature = f"{file_stat.st_size}:{file_stat.st_mtime_ns}"
            except OSError:
                signature = "missing"
        entries[path_text] = {"status": line[:2], "signature": signature}
        if len(entries) >= GIT_STATUS_MAX_LINES:
            break
    fingerprint = hashlib.sha256(
        json.dumps(
            {"status": status, "entries": entries}, ensure_ascii=False, sort_keys=True
        ).encode("utf-8")
    ).hexdigest()
    rendered_lines: list[str] = []
    rendered_chars = 0
    truncated = False
    for line in source_lines:
        candidate_size = len(line) + (1 if rendered_lines else 0)
        if (
            len(rendered_lines) >= GIT_STATUS_MAX_LINES
            or rendered_chars + candidate_size > GIT_STATUS_MAX_CHARS
        ):
            truncated = True
            break
        rendered_lines.append(line)
        rendered_chars += candidate_size
    return {
        "repository_root": resolved_repository,
        "captured_at": _now(),
        "fingerprint": fingerprint,
        "branch_line": branch_line,
        "entries": entries,
        "status_lines": rendered_lines,
        "truncated": truncated,
    }


def snapshot_state(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "branch_line": str(snapshot.get("branch_line") or ""),
        "entries": dict(snapshot.get("entries") or {}),
    }


def adaptive_check_interval(state: dict[str, Any]) -> float:
    duration_seconds = float(state.get("last_check_duration_ms") or 0.0) / 1000
    return min(
        GIT_CHECK_MAX_INTERVAL_SECONDS,
        max(
            GIT_CHECK_MIN_INTERVAL_SECONDS,
            duration_seconds * GIT_CHECK_DURATION_MULTIPLIER,
        ),
    )


def render_git_section(
    snapshot: dict[str, Any],
    *,
    reason: str,
    previous_snapshot: dict[str, Any] | None = None,
) -> ReminderSection:
    delta_lines, delta_truncated = _delta_lines(previous_snapshot or {}, snapshot)
    is_delta = previous_snapshot is not None
    truncated = str(bool(snapshot.get("truncated")) or delta_truncated).lower()
    lines: list[str] = []
    if is_delta:
        lines.append("  <changes>")
        lines.extend(delta_lines or ['    <change kind="metadata-only" />'])
        lines.append("  </changes>")
        lines.append(
            f'  <summary dirty-total="{len(dict(snapshot.get("entries") or {}))}" '
            f'change-count="{max(1, len(delta_lines))}" />'
        )
    else:
        lines.append('  <status format="porcelain-v1">')
        status_lines = list(snapshot.get("status_lines") or [])
        lines.extend(
            f"    {escape(str(line), quote=False)}"
            for line in status_lines or ["clean"]
        )
        lines.append("  </status>")
    lines.append(
        "  <notice>This is a point-in-time snapshot. Run git status before staging, committing, or reporting the current repository state.</notice>"
    )
    wrapped = Prompt("\n".join(lines)).wrap(
        "git-context",
        **{
            "repository-root": str(snapshot["repository_root"]),
            "captured-at": str(snapshot["captured_at"]),
            "reason": reason,
            "mode": "delta" if is_delta else "baseline",
            "truncated": truncated,
        },
    )
    return ReminderSection(
        source="git.context",
        order=30,
        lines=tuple(str(wrapped).splitlines()),
        include_time=True,
    )


def detect_git_change(
    workspace_root: Path, state: dict[str, Any]
) -> ReminderSection | None:
    now_epoch = time.time()
    if now_epoch - float(
        state.get("last_checked_epoch") or 0.0
    ) < adaptive_check_interval(state):
        return None
    started = time.perf_counter()
    snapshot = capture_git_snapshot(
        workspace_root,
        repository_root=str(state.get("repository_root") or ""),
    )
    state["last_checked_at"] = _now()
    state["last_checked_epoch"] = now_epoch
    state["last_check_duration_ms"] = (time.perf_counter() - started) * 1000
    if snapshot is None:
        return None
    previous_hash = str(state.get("last_observed_hash") or "")
    previous_snapshot = dict(state.get("last_snapshot") or {})
    fingerprint = str(snapshot["fingerprint"])
    state.update(
        {
            "repository_root": str(snapshot["repository_root"]),
            "last_observed_hash": fingerprint,
            "last_snapshot": snapshot_state(snapshot),
        }
    )
    if fingerprint == previous_hash:
        return None
    state["last_injected_hash"] = fingerprint
    return render_git_section(
        snapshot, reason="tool-change", previous_snapshot=previous_snapshot
    )


def _delta_lines(
    previous_snapshot: dict[str, Any], snapshot: dict[str, Any]
) -> tuple[list[str], bool]:
    previous_branch = str(previous_snapshot.get("branch_line") or "")
    current_branch = str(snapshot.get("branch_line") or "")
    previous_entries = dict(previous_snapshot.get("entries") or {})
    current_entries = dict(snapshot.get("entries") or {})
    changes: list[str] = []
    if previous_branch != current_branch:
        changes.append(
            "    <branch-change "
            f'before="{escape(previous_branch, quote=True)}" after="{escape(current_branch, quote=True)}" />'
        )
    for path_text in sorted(set(previous_entries) | set(current_entries)):
        previous = previous_entries.get(path_text)
        current = current_entries.get(path_text)
        escaped_path = escape(path_text, quote=True)
        if previous is None and isinstance(current, dict):
            changes.append(
                f'    <change kind="added" status="{escape(str(current.get("status") or ""), quote=True)}" path="{escaped_path}" />'
            )
        elif current is None and isinstance(previous, dict):
            changes.append(
                f'    <change kind="removed" previous-status="{escape(str(previous.get("status") or ""), quote=True)}" path="{escaped_path}" />'
            )
        elif isinstance(previous, dict) and isinstance(current, dict):
            previous_status = str(previous.get("status") or "")
            current_status = str(current.get("status") or "")
            if previous_status != current_status:
                changes.append(
                    f'    <change kind="status-changed" previous-status="{escape(previous_status, quote=True)}" '
                    f'status="{escape(current_status, quote=True)}" path="{escaped_path}" />'
                )
            elif str(previous.get("signature") or "") != str(
                current.get("signature") or ""
            ):
                changes.append(
                    f'    <change kind="content-changed" status="{escape(current_status, quote=True)}" path="{escaped_path}" />'
                )
    selected: list[str] = []
    rendered_chars = 0
    for line in changes:
        if (
            len(selected) >= GIT_STATUS_MAX_LINES
            or rendered_chars + len(line) > GIT_STATUS_MAX_CHARS
        ):
            return selected, True
        selected.append(line)
        rendered_chars += len(line)
    return selected, False
