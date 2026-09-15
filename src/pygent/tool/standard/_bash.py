"""Workspace-scoped bash adapter with bounded process and output handling."""

from __future__ import annotations

import asyncio
import locale
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Annotated, Any, cast

from pydantic import Field

from pygent.tool.executors import ToolExecutionError
from pygent.tool.functional import tool
from pygent.tool.types import IdempotencyPolicy, ToolSideEffect

from ._paths import ToolPathContext, resolve_dir_path

_DEFAULT_TIMEOUT_MS = 30_000
_MAX_TIMEOUT_MS = 600_000
_MAX_OUTPUT_BYTES = 512 * 1024
_MAX_FULL_OUTPUT_BYTES = 16 * 1024 * 1024
_PROCESS_KILL_GRACE_SECONDS = 1.0
_PROCESS_CLEANUP_SECONDS = 2.0
_OUTPUT_COPY_CHUNK_BYTES = 1024 * 1024


def _looks_like_utf16(data: bytes) -> bool:
    if not data:
        return False
    sample = data[:256]
    return sample.count(b"\x00") >= max(2, len(sample) // 5)


def _guess_utf16_encoding(data: bytes) -> str | None:
    if data.startswith(b"\xff\xfe"):
        return "utf-16-le"
    if data.startswith(b"\xfe\xff"):
        return "utf-16-be"
    sample = data[:256]
    if len(sample) < 4:
        return None
    even_nulls = sample[0::2].count(0)
    odd_nulls = sample[1::2].count(0)
    threshold = max(2, len(sample) // 20)
    if odd_nulls >= threshold and odd_nulls > even_nulls * 2:
        return "utf-16-le"
    if even_nulls >= threshold and even_nulls > odd_nulls * 2:
        return "utf-16-be"
    return None


def _decode_mixed_utf16_prefix(data: bytes) -> str | None:
    encoding = _guess_utf16_encoding(data)
    if encoding not in {"utf-16-le", "utf-16-be"}:
        return None
    newline = b"\n\x00" if encoding == "utf-16-le" else b"\x00\n"
    search_from = 0
    while True:
        newline_at = data.find(newline, search_from)
        if newline_at < 0:
            return None
        prefix_end = newline_at + len(newline)
        suffix = data[prefix_end:]
        if not suffix:
            return None
        if _guess_utf16_encoding(suffix) or _looks_like_utf16(suffix):
            search_from = prefix_end
            continue
        prefix = data[:prefix_end]
        try:
            decoded_prefix = prefix.decode(encoding, errors="strict")
        except UnicodeDecodeError:
            decoded_prefix = prefix.decode(encoding, errors="replace")
        return decoded_prefix + _decode_output(suffix, max_bytes=len(suffix))


def _unique_encodings(*encodings: str | None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for encoding in encodings:
        if not encoding:
            continue
        normalized = encoding.lower().replace("_", "-")
        if normalized not in seen:
            seen.add(normalized)
            result.append(encoding)
    return result


def _decode_output(data: bytes, max_bytes: int = _MAX_OUTPUT_BYTES) -> str:
    if not data:
        return ""
    data = data[:max_bytes]
    mixed_output = _decode_mixed_utf16_prefix(data)
    if mixed_output is not None:
        return mixed_output
    guessed_utf16 = _guess_utf16_encoding(data)
    utf16_candidates: list[str | None] = []
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        utf16_candidates = ["utf-16", guessed_utf16]
    elif guessed_utf16:
        utf16_candidates = [guessed_utf16]
    elif _looks_like_utf16(data):
        utf16_candidates = ["utf-16", "utf-16-le", "utf-16-be"]
    for encoding in _unique_encodings(
        *utf16_candidates,
        "utf-8-sig",
        "utf-8",
        # Try the common multibyte Windows encoding before locale-dependent
        # single-byte codecs.  Code pages such as cp1252 accept every byte, so
        # placing the host locale first can silently turn cp936 output into
        # mojibake on an English Windows runner.
        "gb18030",
        "cp936",
        locale.getpreferredencoding(False),
        getattr(sys.stdout, "encoding", None),
        "cp1252",
        "latin-1",
    ):
        candidate = data
        if (
            encoding.lower().replace("_", "-").startswith("utf-16")
            and len(candidate) % 2
        ):
            candidate = candidate[:-1]
        try:
            return candidate.decode(encoding, errors="strict")
        except UnicodeDecodeError as exc:
            normalized = encoding.lower().replace("_", "-")
            if (
                normalized in {"utf-8", "utf-8-sig"}
                and exc.reason == "unexpected end of data"
                and exc.start >= len(candidate) - 4
            ):
                return candidate.decode(encoding, errors="replace")
        except LookupError:
            continue
    return data.decode("utf-8", errors="replace")


def _normalize_timeout_seconds(timeout: float | None) -> float:
    timeout_ms = _DEFAULT_TIMEOUT_MS if timeout is None else int(timeout)
    if timeout_ms <= 0:
        timeout_ms = _DEFAULT_TIMEOUT_MS
    return min(timeout_ms, _MAX_TIMEOUT_MS) / 1000.0


def _append_unique(paths: list[str], candidate: str | None) -> None:
    if not candidate:
        return
    normalized = os.path.normcase(
        os.path.abspath(os.path.expandvars(os.path.expanduser(candidate)))
    )
    existing = {os.path.normcase(os.path.abspath(item)) for item in paths}
    if normalized not in existing:
        paths.append(candidate)


def _windows_bash_candidates() -> list[str]:
    candidates: list[str] = []
    for path_entry in os.environ.get("PATH", "").split(os.pathsep):
        if not path_entry:
            continue
        path = Path(path_entry)
        if path.name.lower() == "cmd":
            _append_unique(candidates, str(path.parent / "bin" / "bash.exe"))
            _append_unique(candidates, str(path.parent / "usr" / "bin" / "bash.exe"))
    for root in (
        os.environ.get("ProgramFiles"),
        os.environ.get("ProgramFiles(x86)"),
        os.environ.get("LocalAppData"),
    ):
        if root:
            _append_unique(candidates, str(Path(root) / "Git" / "bin" / "bash.exe"))
            _append_unique(
                candidates, str(Path(root) / "Git" / "usr" / "bin" / "bash.exe")
            )
    for drive in ("C:", "D:"):
        drive_root = Path(drive + os.sep)
        _append_unique(candidates, str(drive_root / "Git" / "bin" / "bash.exe"))
        _append_unique(candidates, str(drive_root / "Git" / "usr" / "bin" / "bash.exe"))
        _append_unique(
            candidates, str(drive_root / "msys64" / "usr" / "bin" / "bash.exe")
        )
    return candidates


def _bash_candidates() -> list[str]:
    candidates: list[str] = []
    if sys.platform == "win32":
        for candidate in _windows_bash_candidates():
            _append_unique(candidates, candidate)
        _append_unique(candidates, shutil.which("bash"))
    else:
        _append_unique(candidates, shutil.which("bash"))
        _append_unique(candidates, "/bin/bash")
        _append_unique(candidates, "/usr/bin/bash")
    return candidates


def _is_functional_bash(executable: str) -> bool:
    probe_command = "printf ok"
    if sys.platform == "win32":
        drive = (os.environ.get("SystemDrive") or "C:").rstrip(":").lower()
        probe_command = f"test -d /{drive} && printf ok"
    try:
        process = subprocess.run(
            [executable, "-lc", probe_command],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=3,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False
    return process.returncode == 0 and process.stdout == b"ok"


def _find_bash_executable() -> str:
    configured = os.environ.get("PYGENT_BASH_PATH")
    if configured:
        return configured
    candidates = _bash_candidates()
    for candidate in candidates:
        if _is_functional_bash(candidate):
            return candidate
    return candidates[0] if candidates else "bash"


def _read_limited_output(
    output_file: Any, max_bytes: int = _MAX_OUTPUT_BYTES
) -> tuple[bytes, bool]:
    output_file.seek(0)
    data = output_file.read(max_bytes + 1)
    return data[:max_bytes], len(data) > max_bytes


def _save_full_output(
    output_file: Any, cwd: str, pid: int | None = None
) -> tuple[str | None, str | None]:
    timestamp_ms = int(time.time() * 1000)
    pid_part = f"_{pid}" if pid is not None else ""
    for attempt in range(100):
        attempt_part = f"_{attempt}" if attempt else ""
        path = Path(cwd) / (
            f".pygent_bash_output_{timestamp_ms}{pid_part}{attempt_part}.log"
        )
        try:
            output_file.seek(0)
            with path.open("xb") as saved:
                shutil.copyfileobj(output_file, saved, length=_OUTPUT_COPY_CHUNK_BYTES)
            return str(path.resolve()), None
        except FileExistsError:
            continue
        except OSError as exc:
            return None, str(exc)
    return None, "could not allocate a unique output file name"


def _format_result(
    exit_code: int | str,
    output: str,
    *,
    truncated: bool = False,
    full_output_path: str | None = None,
    full_output_error: str | None = None,
    timed_out_after: float | None = None,
    capture_truncated: bool = False,
    cleanup_complete: bool = True,
) -> str:
    result = f"exit_code: {exit_code}\noutput:\n{output}"
    notices = []
    if timed_out_after is not None:
        notices.append(f"timed out after {timed_out_after:.3g} seconds")
    if not cleanup_complete:
        notices.append(
            "process cleanup incomplete; output capture closed; descendants may still be running"
        )
    if truncated:
        notices.append(f"output truncated to the first {_MAX_OUTPUT_BYTES} bytes")
        if full_output_path:
            notices.append(f"full output saved to: {full_output_path}")
        elif full_output_error:
            notices.append(f"failed to save full output: {full_output_error}")
    if capture_truncated:
        notices.append(
            f"captured output capped at {_MAX_FULL_OUTPUT_BYTES} bytes while the process stream was drained"
        )
    if notices:
        if output and not output.endswith("\n"):
            result += "\n"
        result += "".join(f"[{notice}]\n" for notice in notices).rstrip("\n")
    return result


class _BashProcess(asyncio.SubprocessProtocol):
    """Own a foreground transport and bounded capture until exit AND pipe EOF.

    Capturing through the public protocol API lets cleanup close the read end
    without waiting for inherited write handles or leaving a reader task behind.
    """

    def __init__(self, output_file: Any = None) -> None:
        self.transport: asyncio.SubprocessTransport | None = None
        self.output_file = output_file
        self.outcome_ready: asyncio.Future[None] = (
            asyncio.get_running_loop().create_future()
        )
        self.connection_closed = False
        self.captured = 0
        self.truncated = False
        self.error: Exception | None = None
        self.close_requested = False
        self.startup: (
            asyncio.Task[tuple[asyncio.SubprocessTransport, _BashProcess]] | None
        ) = None

    async def start(self, *args: str, **kwargs: Any) -> None:
        self.startup = asyncio.create_task(
            asyncio.get_running_loop().subprocess_exec(lambda: self, *args, **kwargs),
            name="pygent-bash-startup",
        )
        # A late startup failure must be observed even after the caller's deadline.
        self.startup.add_done_callback(
            lambda task: None if task.cancelled() else task.exception()
        )
        await asyncio.shield(self.startup)

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = cast(asyncio.SubprocessTransport, transport)
        if self.close_requested:
            self.transport.close()

    def pipe_data_received(self, fd: int, data: bytes) -> None:
        if self.output_file is None:
            return
        remaining = _MAX_FULL_OUTPUT_BYTES - self.captured
        try:
            saved = data[:remaining]
            self.output_file.write(saved)
            self.captured += len(saved)
            self.truncated |= len(data) > remaining
        except OSError as exc:
            self._capture_failed(exc)

    def pipe_connection_lost(self, fd: int, exc: Exception | None) -> None:
        if exc is not None:
            self._capture_failed(exc)

    def _capture_failed(self, exc: Exception) -> None:
        self.error = exc
        self.output_file = None
        if not self.outcome_ready.done():
            self.outcome_ready.set_result(None)

    def connection_lost(self, exc: Exception | None) -> None:
        self.connection_closed = True
        if exc is not None and self.error is None:
            self.error = exc
        if not self.outcome_ready.done():
            self.outcome_ready.set_result(None)

    def close(self) -> None:
        # Disable writes before the caller closes its temporary output file.
        self.close_requested = True
        self.output_file = None
        if self.transport is not None:
            self.transport.close()
            if self.startup is not None and not self.startup.done():
                self.startup.cancel()

    async def aclose(self, is_windows: bool) -> bool:
        """Spend one cleanup budget, then close local handles unconditionally."""
        try:
            async with asyncio.timeout(_PROCESS_CLEANUP_SECONDS):
                if self.transport is None and self.startup is not None:
                    # Let pipe connection finish without cancelling asyncio's
                    # own startup waiter, which may otherwise wait for EOF.
                    await asyncio.shield(self.startup)
                if self.connection_closed or self.transport is None:
                    return True
                terminated = await self._terminate_tree(is_windows)
                await asyncio.shield(self.outcome_ready)
                return terminated and self.connection_closed
        except (TimeoutError, OSError):
            return False
        finally:
            self.close()

    async def _terminate_tree(self, is_windows: bool) -> bool:
        assert self.transport is not None
        pid = self.transport.get_pid()
        if is_windows:
            # taskkill cannot reliably identify descendants of an exited parent.
            # Do not target a potentially reused PID; report incomplete cleanup.
            if self.transport.get_returncode() is not None:
                return False
            return await _terminate_windows_process_tree(pid)

        # A POSIX process group can outlive its leader. Always signal the group.
        kill_process_group = os.killpg  # type: ignore[attr-defined]
        try:
            kill_process_group(pid, signal.SIGTERM)
        except ProcessLookupError:
            return True
        try:
            async with asyncio.timeout(_PROCESS_KILL_GRACE_SECONDS):
                await asyncio.shield(self.outcome_ready)
        except TimeoutError:
            pass
        # Remaining descendants may have closed stdout but ignored SIGTERM.
        try:
            kill_process_group(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
        except ProcessLookupError:
            pass
        return True


class BashTools:
    """Deployment-local bash process adapter with workspace confinement."""

    def __init__(
        self,
        *,
        workspace_root: str | Path,
        bash_executable: str | None = None,
        restrict_to_workspace: bool = True,
    ) -> None:
        self.path_context = ToolPathContext.from_workspace_root(
            workspace_root, restrict_to_workspace=restrict_to_workspace
        )
        self.workspace_root = self.path_context.workspace_root
        self.bash_executable = bash_executable or _find_bash_executable()
        self._is_windows = sys.platform == "win32"

    @tool(
        tool_id="standard.shell.bash",
        version="2.0.0",
        side_effect=ToolSideEffect.EXTERNAL,
        idempotency=IdempotencyPolicy.NOT_IDEMPOTENT,
        timeout=610,
        resource_key="shell",
        sandbox_profile="workspace",
        required_permissions=("shell:execute",),
    )
    async def bash(
        self,
        command: str,
        working_directory: str | None = None,
        timeout: Annotated[float | None, Field(gt=0, le=_MAX_TIMEOUT_MS)] = None,
        description: str | None = None,
        is_background: bool = False,
    ) -> str:
        """Run one bash command in the configured workspace.

        Args:
            command: Complete command string passed to ``bash -lc``.
            working_directory: Directory resolved from workspace_root.
            timeout: Execution and output timeout in milliseconds, capped at
                600000, followed by at most two seconds of process cleanup.
            description: Optional caller-facing description; not executed.
            is_background: Start an independent process and return its PID.
        """

        del description
        cwd = self._resolve_working_directory(working_directory)
        if is_background:
            return await self._run_background(command or "", cwd)
        return await self._run_foreground(
            command or "", cwd, _normalize_timeout_seconds(timeout)
        )

    def _resolve_working_directory(self, working_directory: str | None) -> str:
        path = resolve_dir_path(working_directory, self.path_context)
        if not path.is_dir():
            raise ToolExecutionError(
                f"working directory does not exist or is not a directory: {path}",
                kind="filesystem_error",
                code="not_a_directory",
                side_effect_committed=False,
            )
        return str(path)

    def _command_args(self, command: str) -> list[str]:
        return [self.bash_executable, "-lc", command]

    def _process_kwargs(self, cwd: str, output: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "cwd": cwd,
            "stdin": subprocess.DEVNULL,
            "stdout": output,
            "stderr": subprocess.STDOUT,
        }
        if self._is_windows:
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            kwargs["start_new_session"] = True
        return kwargs

    async def _run_background(self, command: str, cwd: str) -> str:
        try:
            process = await asyncio.create_subprocess_exec(
                *self._command_args(command),
                **self._process_kwargs(cwd, subprocess.DEVNULL),
            )
        except FileNotFoundError as exc:
            raise ToolExecutionError(
                "bash executable was not found",
                kind="process_error",
                code="executable_not_found",
                side_effect_committed=False,
            ) from exc
        except OSError as exc:
            raise ToolExecutionError(
                "background command could not be started",
                kind="process_error",
                code="process_start_failed",
                side_effect_committed=False,
            ) from exc
        return f"started background process PID={process.pid}; output is not captured"

    async def _run_foreground(
        self, command: str, cwd: str, timeout_seconds: float
    ) -> str:
        with tempfile.TemporaryFile() as output_file:
            process = _BashProcess(output_file)
            timed_out = False
            cancelled = False
            try:
                async with asyncio.timeout(timeout_seconds):
                    try:
                        await process.start(
                            *self._command_args(command),
                            **self._process_kwargs(cwd, asyncio.subprocess.PIPE),
                        )
                    except FileNotFoundError as exc:
                        raise ToolExecutionError(
                            "bash executable was not found",
                            kind="process_error",
                            code="executable_not_found",
                            side_effect_committed=False,
                        ) from exc
                    except OSError as exc:
                        raise ToolExecutionError(
                            "command could not be started",
                            kind="process_error",
                            code="process_start_failed",
                            side_effect_committed=False,
                        ) from exc
                    await asyncio.shield(process.outcome_ready)
            except TimeoutError:
                timed_out = True
            except asyncio.CancelledError:
                cancelled = True
            finally:
                cleanup = asyncio.create_task(
                    process.aclose(self._is_windows),
                    name="pygent-bash-cleanup",
                )
                # Repeated caller cancellation must neither restart the budget
                # nor abandon the task that owns the transport and output file.
                while True:
                    try:
                        cleanup_complete = await asyncio.shield(cleanup)
                        break
                    except asyncio.CancelledError:
                        if cleanup.cancelled():
                            raise
                        cancelled = True
                if cancelled:
                    raise asyncio.CancelledError

            if process.error is not None and not timed_out:
                raise process.error
            transport = process.transport
            output_file.flush()
            data, truncated = _read_limited_output(output_file)
            full_path = None
            full_error = None
            if truncated:
                full_path, full_error = _save_full_output(
                    output_file, cwd, transport.get_pid() if transport else None
                )
            returncode = transport.get_returncode() if transport else None
            exit_code: int | str = (
                "timeout" if timed_out else returncode if returncode is not None else -1
            )
            formatted = _format_result(
                exit_code,
                _decode_output(data),
                truncated=truncated,
                full_output_path=full_path,
                full_output_error=full_error,
                timed_out_after=timeout_seconds if timed_out else None,
                capture_truncated=process.truncated,
                cleanup_complete=cleanup_complete,
            )
            if timed_out:
                raise ToolExecutionError(
                    formatted,
                    kind="timeout",
                    code="command_timeout",
                    retryable=False,
                    side_effect_committed=None,
                )
            return formatted


async def _terminate_windows_process_tree(pid: int) -> bool:
    """Run taskkill within the caller's cleanup deadline and own its transport."""
    terminator = _BashProcess()
    try:
        await terminator.start(
            "taskkill",
            "/PID",
            str(pid),
            "/T",
            "/F",
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        await asyncio.shield(terminator.outcome_ready)
        assert terminator.transport is not None
        return terminator.transport.get_returncode() == 0
    finally:
        terminator.close()


__all__ = ["BashTools"]
