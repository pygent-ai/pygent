import locale
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from pygent.common import PygentString
from pygent.module.tool import ToolErrorResult
from pygent.module.tool.utils import ToolClassBase, tool_class, tool_method
from pygent.toolkits.path_utils import normalize_tool_path


_DEFAULT_TIMEOUT_MS = 30000
_MAX_TIMEOUT_MS = 600000
_MAX_OUTPUT_BYTES = 512 * 1024
_PROCESS_KILL_GRACE_SECONDS = 1.0
_OUTPUT_COPY_CHUNK_BYTES = 1024 * 1024


def _looks_like_utf16(data: bytes) -> bool:
    if not data:
        return False
    sample = data[:256]
    return sample.count(b"\x00") >= max(2, len(sample) // 5)


def _unique_encodings(*encodings: Optional[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for encoding in encodings:
        if not encoding:
            continue
        normalized = encoding.lower().replace("_", "-")
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(encoding)
    return result


def _decode_output(data: bytes, max_bytes: int = _MAX_OUTPUT_BYTES) -> str:
    """Decode process output without raising and without exceeding max_bytes."""
    if not data:
        return ""

    data = data[:max_bytes]
    preferred = locale.getpreferredencoding(False)
    console_encoding = getattr(sys.stdout, "encoding", None)

    utf16_candidates: list[str] = []
    if data.startswith((b"\xff\xfe", b"\xfe\xff")) or _looks_like_utf16(data):
        utf16_candidates = ["utf-16", "utf-16-le", "utf-16-be"]

    for encoding in _unique_encodings(
        *utf16_candidates,
        "utf-8-sig",
        "utf-8",
        preferred,
        console_encoding,
        "gb18030",
        "cp936",
        "cp1252",
        "latin-1",
    ):
        candidate = data
        if encoding.lower().replace("_", "-").startswith("utf-16") and len(candidate) % 2:
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
            continue
        except LookupError:
            continue

    return data.decode("utf-8", errors="replace")


def _normalize_timeout_seconds(timeout: Optional[float]) -> float:
    timeout_ms = timeout if timeout is not None else _DEFAULT_TIMEOUT_MS
    try:
        timeout_ms = int(timeout_ms)
    except (TypeError, ValueError):
        timeout_ms = _DEFAULT_TIMEOUT_MS

    if timeout_ms <= 0:
        timeout_ms = _DEFAULT_TIMEOUT_MS
    timeout_ms = min(timeout_ms, _MAX_TIMEOUT_MS)
    return timeout_ms / 1000.0


def _append_unique(paths: list[str], candidate: Optional[str]) -> None:
    if not candidate:
        return
    normalized = os.path.normcase(os.path.abspath(os.path.expandvars(os.path.expanduser(candidate))))
    if normalized not in {os.path.normcase(os.path.abspath(p)) for p in paths}:
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

    program_files = [
        os.environ.get("ProgramFiles"),
        os.environ.get("ProgramFiles(x86)"),
        os.environ.get("LocalAppData"),
    ]
    for root in program_files:
        if not root:
            continue
        _append_unique(candidates, str(Path(root) / "Git" / "bin" / "bash.exe"))
        _append_unique(candidates, str(Path(root) / "Git" / "usr" / "bin" / "bash.exe"))

    for drive in ("C:", "D:"):
        _append_unique(candidates, str(Path(drive + os.sep) / "Git" / "bin" / "bash.exe"))
        _append_unique(candidates, str(Path(drive + os.sep) / "Git" / "usr" / "bin" / "bash.exe"))
        _append_unique(candidates, str(Path(drive + os.sep) / "msys64" / "usr" / "bin" / "bash.exe"))
    return candidates


def _bash_candidates() -> list[str]:
    candidates: list[str] = []
    _append_unique(candidates, shutil.which("bash"))
    if sys.platform == "win32":
        for candidate in _windows_bash_candidates():
            _append_unique(candidates, candidate)
    else:
        _append_unique(candidates, "/bin/bash")
        _append_unique(candidates, "/usr/bin/bash")
    return candidates


def _is_functional_bash(executable: str) -> bool:
    try:
        proc = subprocess.run(
            [executable, "-lc", "printf ok"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=3,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0 and proc.stdout == b"ok"


def _find_bash_executable() -> str:
    configured = os.environ.get("PYGENT_BASH_PATH")
    if configured:
        return configured

    candidates = _bash_candidates()
    for candidate in candidates:
        if _is_functional_bash(candidate):
            return candidate
    return candidates[0] if candidates else "bash"


def _read_limited_output(output_file: Any, max_bytes: int = _MAX_OUTPUT_BYTES) -> tuple[bytes, bool]:
    output_file.seek(0)
    data = output_file.read(max_bytes + 1)
    truncated = len(data) > max_bytes
    if truncated:
        data = data[:max_bytes]
    return data, truncated


def _save_full_output(output_file: Any, cwd: str, pid: Optional[int] = None) -> tuple[Optional[str], Optional[str]]:
    timestamp_ms = int(time.time() * 1000)
    pid_part = f"_{pid}" if pid is not None else ""
    output_dir = Path(cwd)

    for attempt in range(100):
        attempt_part = f"_{attempt}" if attempt else ""
        output_path = output_dir / f".pygent_bash_output_{timestamp_ms}{pid_part}{attempt_part}.log"
        try:
            output_file.seek(0)
            with output_path.open("xb") as saved_output:
                shutil.copyfileobj(output_file, saved_output, length=_OUTPUT_COPY_CHUNK_BYTES)
            return str(output_path.resolve()), None
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
    full_output_path: Optional[str] = None,
    full_output_error: Optional[str] = None,
    timed_out_after: Optional[float] = None,
) -> str:
    result = f"exit_code: {exit_code}\noutput:\n{output}"
    notices: list[str] = []
    if timed_out_after is not None:
        notices.append(f"timed out after {timed_out_after:.3g} seconds; process terminated")
    if truncated:
        notices.append(f"output truncated to the first {_MAX_OUTPUT_BYTES} bytes")
        if full_output_path:
            notices.append(f"full output saved to: {full_output_path}")
        elif full_output_error:
            notices.append(f"failed to save full output: {full_output_error}")
    if notices:
        if output and not output.endswith("\n"):
            result += "\n"
        result += "".join(f"[{notice}]\n" for notice in notices).rstrip("\n")
    return result


def _tool_error(
    message: str,
    error_type: str = "ToolExecutionError",
    **details: Any,
) -> ToolErrorResult:
    input_path = details.pop("input_path", details.pop("input", None))
    path = details.pop("path", details.pop("resolved_path", None))
    normalized_details = dict(details)
    if input_path is not None:
        normalized_details["input_path"] = str(input_path)
    if path is not None:
        normalized_details["path"] = str(path)
    return ToolErrorResult(message, error_type=error_type, details=normalized_details or None)


@tool_class(description="Bash command toolkit with timeout, cwd, encoding, and large-output handling.")
class BashToolkits(ToolClassBase):
    """Run commands with bash in the configured workspace."""

    def __init__(
        self,
        session_id: str,
        workspace_root: Optional[str] = None,
        bash_executable: Optional[str] = None,
    ):
        super().__init__()
        self.session_id = PygentString(session_id)
        self.workspace_root = str(Path(workspace_root or os.getcwd()).expanduser().resolve())
        self.bash_executable = bash_executable or _find_bash_executable()
        self._is_windows = sys.platform == "win32"

    @tool_method(
        name="bash",
        description="Run a command with bash. Output is the combined terminal-visible stdout/stderr stream. working_directory accepts Windows, Windows slash, Git Bash/MSYS drive paths on Windows, and relative paths resolved from workspace_root.",
    )
    def bash(
        self,
        command: str,
        working_directory: Optional[str] = None,
        timeout: Optional[float] = None,
        description: Optional[str] = None,
        is_background: bool = False,
    ) -> str:
        """
        Run a bash command in the workspace.

        Args:
            command: Complete bash command string to run.
            working_directory: Directory for the command. Accepts Windows paths such as E:\\Projects\\repo, Windows slash paths such as E:/Projects/repo, and on Windows Git Bash/MSYS drive paths such as /e/Projects/repo. Relative paths are resolved from workspace_root.
            timeout: Timeout in milliseconds. Defaults to 30000 and is capped at 600000.
            description: Optional human-readable command description for callers.
            is_background: If true, start the command in the background and do not capture output.
        """
        del description
        command = "" if command is None else str(command)
        timeout_sec = _normalize_timeout_seconds(timeout)

        try:
            cwd = self._resolve_working_directory(working_directory)
        except ValueError as exc:
            details = getattr(exc, "details", {})
            return _tool_error(f"error: {exc}", "NotADirectoryError", **details)

        if is_background:
            return self._run_background(command, cwd)
        return self._run_foreground(command, cwd, timeout_sec)

    def _resolve_working_directory(self, working_directory: Optional[str]) -> str:
        if working_directory is None or not str(working_directory).strip():
            path = Path(self.workspace_root)
            input_path = None
        else:
            input_path = str(working_directory).strip()
            path = normalize_tool_path(input_path, self.workspace_root)

        resolved = path.resolve()
        if not resolved.is_dir():
            exc = ValueError(f"working directory does not exist or is not a directory: {resolved}")
            exc.details = {
                "input_path": input_path or self.workspace_root,
                "path": str(resolved),
            }
            raise exc
        return str(resolved)

    def _bash_args(self, command: str) -> list[str]:
        return [self.bash_executable, "-lc", command]

    def _popen_kwargs(self, cwd: str, output_file: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "cwd": cwd,
            "stdin": subprocess.DEVNULL,
            "stdout": output_file,
            "stderr": subprocess.STDOUT,
        }
        if self._is_windows:
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            kwargs["start_new_session"] = True
        return kwargs

    def _run_background(self, command: str, cwd: str) -> str:
        try:
            kwargs: dict[str, Any] = {
                "cwd": cwd,
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
            }
            if self._is_windows:
                kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            else:
                kwargs["start_new_session"] = True
            proc = subprocess.Popen(self._bash_args(command), **kwargs)
            return f"started background process PID={proc.pid}; output is not captured"
        except FileNotFoundError as exc:
            return f"error: bash executable not found: {exc}"
        except OSError as exc:
            return f"error: failed to start bash command: {exc}"

    def _run_foreground(self, command: str, cwd: str, timeout_sec: float) -> str:
        with tempfile.TemporaryFile() as output_file:
            try:
                proc = subprocess.Popen(
                    self._bash_args(command),
                    **self._popen_kwargs(cwd, output_file),
                )
            except FileNotFoundError as exc:
                return f"error: bash executable not found: {exc}"
            except OSError as exc:
                return f"error: failed to run bash command: {exc}"

            try:
                exit_code = proc.wait(timeout=timeout_sec)
                data, truncated = _read_limited_output(output_file)
                full_output_path = None
                full_output_error = None
                if truncated:
                    full_output_path, full_output_error = _save_full_output(output_file, cwd, proc.pid)
                return _format_result(
                    exit_code,
                    _decode_output(data),
                    truncated=truncated,
                    full_output_path=full_output_path,
                    full_output_error=full_output_error,
                )
            except subprocess.TimeoutExpired:
                self._terminate_process_tree(proc)
                try:
                    proc.wait(timeout=_PROCESS_KILL_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    self._kill_process_tree(proc)
                    try:
                        proc.wait(timeout=_PROCESS_KILL_GRACE_SECONDS)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                data, truncated = _read_limited_output(output_file)
                full_output_path = None
                full_output_error = None
                if truncated:
                    full_output_path, full_output_error = _save_full_output(output_file, cwd, proc.pid)
                return _format_result(
                    "timeout",
                    _decode_output(data),
                    truncated=truncated,
                    full_output_path=full_output_path,
                    full_output_error=full_output_error,
                    timed_out_after=timeout_sec,
                )

    def _terminate_process_tree(self, proc: subprocess.Popen[Any]) -> None:
        if proc.poll() is not None:
            return

        if self._is_windows:
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                    check=False,
                )
                return
            except (OSError, subprocess.TimeoutExpired):
                proc.kill()
                return

        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except OSError:
            proc.terminate()

    def _kill_process_tree(self, proc: subprocess.Popen[Any]) -> None:
        if proc.poll() is not None:
            return

        if self._is_windows:
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                    check=False,
                )
                return
            except (OSError, subprocess.TimeoutExpired):
                proc.kill()
                return

        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except OSError:
            proc.kill()


__all__ = [
    "BashToolkits",
    "_DEFAULT_TIMEOUT_MS",
    "_MAX_OUTPUT_BYTES",
    "_MAX_TIMEOUT_MS",
    "_decode_output",
]
