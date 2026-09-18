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
from typing import Annotated, Any

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
) -> str:
    result = f"exit_code: {exit_code}\noutput:\n{output}"
    notices = []
    if timed_out_after is not None:
        notices.append(
            f"timed out after {timed_out_after:.3g} seconds; process terminated"
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
            timeout: Process timeout in milliseconds, capped at 600000.
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
        # Ensure the subprocess sees a proxy configuration consistent with the host system
        # across Windows, macOS, and Linux where feasible. We only augment the environment
        # when the standard proxy variables are absent, to avoid overriding explicit user
        # settings provided by the caller or parent environment.
        try:
            env = os.environ.copy()
            proxy_env = self._detect_system_proxy_env(env)
            if proxy_env:
                # Only set variables that are not already present (case-insensitive check)
                existing = {k.lower() for k in env.keys()}
                for k, v in proxy_env.items():
                    if k.lower() not in existing and v:
                        env[k] = v
                # Also mirror to common lowercase/uppercase variants for broader tool compatibility
                def _mirror(key: str):
                    if key.upper() in env and key.lower() not in env:
                        env[key.lower()] = env[key.upper()]
                    if key.lower() in env and key.upper() not in env:
                        env[key.upper()] = env[key.lower()]
                for key in ("http_proxy", "https_proxy", "all_proxy", "no_proxy"):
                    _mirror(key)
            kwargs["env"] = env
        except Exception:
            # If detection fails for any reason, fall back to inheriting the current environment
            kwargs["env"] = os.environ.copy()
        if self._is_windows:
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            kwargs["start_new_session"] = True
        return kwargs

    def _detect_system_proxy_env(self, base_env: dict[str, str]) -> dict[str, str]:
        # Do not override explicit env provided upstream; we only suggest defaults.
        result: dict[str, str] = {}
        try:
            # If any proxy env is already present, still compute suggestions but caller will avoid overriding.
            if sys.platform == "win32":
                # WinINET user-level proxy settings
                try:
                    import winreg  # type: ignore

                    with winreg.OpenKey(
                        winreg.HKEY_CURRENT_USER,
                        r"Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings",
                    ) as key:
                        try:
                            proxy_enable, _ = winreg.QueryValueEx(key, "ProxyEnable")
                        except FileNotFoundError:
                            proxy_enable = 0
                        try:
                            proxy_server, _ = winreg.QueryValueEx(key, "ProxyServer")
                        except FileNotFoundError:
                            proxy_server = ""
                        try:
                            proxy_override, _ = winreg.QueryValueEx(key, "ProxyOverride")
                        except FileNotFoundError:
                            proxy_override = ""
                    if proxy_enable and proxy_server:
                        mapping: dict[str, str] = {}
                        # ProxyServer formats:
                        #  - "host:port" (applies to all HTTP(S))
                        #  - "http=host:port;https=host:port;socks=host:port;..."
                        parts = [p.strip() for p in proxy_server.split(";") if p.strip()]
                        kv = {}
                        bare = None
                        for p in parts:
                            if "=" in p:
                                k, v = p.split("=", 1)
                                kv[k.strip().lower()] = v.strip()
                            else:
                                bare = p.strip()
                        if bare:
                            mapping["http"] = bare
                            mapping["https"] = bare
                        mapping.update(kv)
                        http = mapping.get("http")
                        https = mapping.get("https")
                        socks = mapping.get("socks") or mapping.get("socks5") or mapping.get("socks4")
                        if http:
                            result["HTTP_PROXY"] = f"http://{http}"
                        if https:
                            result["HTTPS_PROXY"] = f"http://{https}"
                        if socks:
                            # Many tools honor ALL_PROXY for SOCKS
                            if not socks.startswith("socks"):
                                result["ALL_PROXY"] = f"socks5://{socks}"
                            else:
                                result["ALL_PROXY"] = socks
                        if proxy_override:
                            # Windows uses semicolons; replace with comma list
                            no_proxy = ",".join(
                                [h.strip() for h in proxy_override.split(";") if h.strip()]
                            )
                            if no_proxy:
                                result["NO_PROXY"] = no_proxy
                except Exception:
                    pass
                # WinHTTP fallback (system-wide). Only parse simple static proxies.
                try:
                    proc = subprocess.run(
                        ["netsh", "winhttp", "show", "proxy"],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        text=True,
                        timeout=1.5,
                        check=False,
                    )
                    out = proc.stdout or ""
                    if "Direct access" not in out and out.strip():
                        # Lines like: "Proxy Server(s) : http=host:port;https=host:port" and "Bypass List : <...>"
                        for line in out.splitlines():
                            line = line.strip()
                            if line.lower().startswith("proxy server") and ":" in line:
                                rhs = line.split(":", 1)[1]
                                parts = [p.strip() for p in rhs.split(";") if p.strip()]
                                for p in parts:
                                    if "=" in p:
                                        k, v = p.split("=", 1)
                                        k = k.strip().lower()
                                        v = v.strip()
                                        if k == "http" and "HTTP_PROXY" not in result:
                                            result["HTTP_PROXY"] = f"http://{v}"
                                        if k == "https" and "HTTPS_PROXY" not in result:
                                            result["HTTPS_PROXY"] = f"http://{v}"
                                        if k.startswith("socks") and "ALL_PROXY" not in result:
                                            result["ALL_PROXY"] = f"socks5://{v}"
                            if line.lower().startswith("bypass list") and ":" in line and "NO_PROXY" not in result:
                                rhs = line.split(":", 1)[1]
                                items = [i.strip() for i in rhs.split(",") if i.strip()]
                                if items:
                                    result["NO_PROXY"] = ",".join(items)
                except Exception:
                    pass
            elif sys.platform == "darwin":
                try:
                    proc = subprocess.run(
                        ["scutil", "--proxy"],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        text=True,
                        timeout=1.5,
                        check=False,
                    )
                    out = proc.stdout or ""
                    if out:
                        def get_val(name: str) -> str | None:
                            for line in out.splitlines():
                                line = line.strip()
                                if line.startswith(name + " :"):
                                    return line.split(":", 1)[1].strip()
                            return None
                        def enabled(name: str) -> bool:
                            return get_val(name) in {"1", "true", "TRUE"}
                        if enabled("HTTPEnable"):
                            host = get_val("HTTPProxy")
                            port = get_val("HTTPPort")
                            if host and port:
                                result["HTTP_PROXY"] = f"http://{host}:{port}"
                        if enabled("HTTPSEnable"):
                            host = get_val("HTTPSProxy")
                            port = get_val("HTTPSPort")
                            if host and port:
                                result["HTTPS_PROXY"] = f"http://{host}:{port}"
                        if enabled("SOCKSEnable"):
                            host = get_val("SOCKSProxy")
                            port = get_val("SOCKSPort")
                            if host and port:
                                result["ALL_PROXY"] = f"socks5://{host}:{port}"
                        # ExceptionsList appears as a multiline array
                        if "ExceptionsList" in out and "NO_PROXY" not in result:
                            items: list[str] = []
                            capture = False
                            for line in out.splitlines():
                                s = line.strip()
                                if s.startswith("ExceptionsList") and s.endswith("("):
                                    capture = True
                                    continue
                                if capture:
                                    if s == ")":
                                        break
                                    s = s.strip('" ,')
                                    if s:
                                        items.append(s)
                            if items:
                                result["NO_PROXY"] = ",".join(items)
                except Exception:
                    pass
            else:
                # Linux: prefer existing env; otherwise try GNOME gsettings if available
                try:
                    if shutil.which("gsettings"):
                        mode_proc = subprocess.run(
                            ["gsettings", "get", "org.gnome.system.proxy", "mode"],
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL,
                            text=True,
                            timeout=1.0,
                            check=False,
                        )
                        mode = (mode_proc.stdout or "").strip().strip("'\"")
                        if mode == "manual":
                            def get_schema(schema: str, key: str) -> str | None:
                                p = subprocess.run(
                                    ["gsettings", "get", schema, key],
                                    stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL,
                                    text=True,
                                    timeout=1.0,
                                    check=False,
                                )
                                return (p.stdout or "").strip()
                            http_host = get_schema("org.gnome.system.proxy.http", "host")
                            http_port = get_schema("org.gnome.system.proxy.http", "port")
                            https_host = get_schema("org.gnome.system.proxy.https", "host")
                            https_port = get_schema("org.gnome.system.proxy.https", "port")
                            socks_host = get_schema("org.gnome.system.proxy.socks", "host")
                            socks_port = get_schema("org.gnome.system.proxy.socks", "port")
                            if http_host and http_host != "''" and http_port and http_port.isdigit():
                                result["HTTP_PROXY"] = f"http://{http_host.strip('\'\"')}:" f"{http_port}"
                            if https_host and https_host != "''" and https_port and https_port.isdigit():
                                result["HTTPS_PROXY"] = f"http://{https_host.strip('\'\"')}:" f"{https_port}"
                            if socks_host and socks_host != "''" and socks_port and socks_port.isdigit():
                                result["ALL_PROXY"] = f"socks5://{socks_host.strip('\'\"')}:" f"{socks_port}"
                            ignore = get_schema("org.gnome.system.proxy", "ignore-hosts")
                            if ignore and ignore.startswith("["):
                                # Example: "['localhost', '127.0.0.1']"
                                items = [
                                    i.strip().strip("'\"")
                                    for i in ignore.strip("[]").split(",")
                                    if i.strip()
                                ]
                                if items:
                                    result["NO_PROXY"] = ",".join(items)
                except Exception:
                    pass
        except Exception:
            return {}
        return result

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
            try:
                process = await asyncio.create_subprocess_exec(
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

            if process.stdout is None:  # pragma: no cover - PIPE invariant
                raise RuntimeError("bash output pipe was not created")
            capture_task = asyncio.create_task(
                self._capture_output(process.stdout, output_file),
                name=f"pygent-bash-capture-{process.pid}",
            )
            timed_out = False
            try:
                await asyncio.wait_for(process.wait(), timeout_seconds)
            except TimeoutError:
                timed_out = True
                await self._terminate_process_tree(process)
            except asyncio.CancelledError:
                await self._terminate_process_tree(process)
                await capture_task
                raise

            capture_truncated = await capture_task
            data, truncated = _read_limited_output(output_file)
            full_path = None
            full_error = None
            if truncated:
                full_path, full_error = _save_full_output(output_file, cwd, process.pid)
            exit_code: int | str = (
                "timeout"
                if timed_out
                else process.returncode
                if process.returncode is not None
                else -1
            )
            formatted = _format_result(
                exit_code,
                _decode_output(data),
                truncated=truncated,
                full_output_path=full_path,
                full_output_error=full_error,
                timed_out_after=timeout_seconds if timed_out else None,
                capture_truncated=capture_truncated,
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

    async def _capture_output(
        self, stream: asyncio.StreamReader, output_file: Any
    ) -> bool:
        captured = 0
        truncated = False
        while chunk := await stream.read(64 * 1024):
            remaining = _MAX_FULL_OUTPUT_BYTES - captured
            if remaining > 0:
                saved = chunk[:remaining]
                output_file.write(saved)
                captured += len(saved)
            if len(chunk) > remaining:
                truncated = True
        output_file.flush()
        return truncated

    async def _terminate_process_tree(
        self, process: asyncio.subprocess.Process
    ) -> None:
        if process.returncode is not None:
            return
        try:
            if self._is_windows:
                process.terminate()
            else:
                kill_process_group = os.killpg  # type: ignore[attr-defined]
                kill_process_group(process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            await asyncio.wait_for(process.wait(), _PROCESS_KILL_GRACE_SECONDS)
            return
        except TimeoutError:
            pass
        try:
            if self._is_windows:
                process.kill()
            else:
                kill_process_group = os.killpg  # type: ignore[attr-defined]
                kill_process_group(
                    process.pid, getattr(signal, "SIGKILL", signal.SIGTERM)
                )
        except (ProcessLookupError, PermissionError):
            pass
        await process.wait()


__all__ = ["BashTools"]
