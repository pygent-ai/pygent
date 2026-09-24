"""Workspace-scoped bash adapter with bounded process and output handling."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated, Any, ClassVar, cast

from pydantic import Field

from pygent.core import (
    JsonValue,
    active_infrastructure,
    current_infrastructure,
    thaw_json,
)
from pygent.core._tool_values import (
    ToolResult,
    ToolTask,
    _tool_result_content_to_value,
)
from pygent.tool.executors import ToolEventEmitter, ToolTaskManager
from pygent.tool.functional import tool
from pygent.tool.task_handle import ToolTaskHandle
from pygent.tool.types import (
    IdempotencyPolicy,
    ToolSideEffect,
)

from . import _process
from ._paths import resolve_workspace_directory
from ._shell import ShellIdentity, ShellResolver, append_unique_path
from ._shell_tools import NativeShellTools

_MAX_OUTPUT_BYTES = _process.MAX_OUTPUT_BYTES
_MAX_FULL_OUTPUT_BYTES = _process.MAX_FULL_OUTPUT_BYTES
_BASH_VERSION_COMMAND = 'printf %s "$BASH_VERSION"'
# Bounded probe timeouts (seconds) for bash discovery and system proxy
# detection on each platform.
_BASH_FUNCTIONAL_PROBE_TIMEOUT_SECONDS = 3.0
_PROXY_DETECT_TIMEOUT_SECONDS = 1.5
_GSETTINGS_TIMEOUT_SECONDS = 1.0
# gsettings reports its values as quoted strings, for example ``'127.0.0.1'``.
# Keeping the quotes out of the f-string expression keeps this module runnable on
# Python 3.11, which forbids backslashes inside them.
_GSETTINGS_QUOTES = "'\""


def _windows_bash_candidates() -> list[str]:
    candidates: list[str] = []
    for path_entry in os.environ.get("PATH", "").split(os.pathsep):
        if not path_entry:
            continue
        path = Path(path_entry)
        if path.name.lower() == "cmd":
            append_unique_path(candidates, str(path.parent / "bin" / "bash.exe"))
            append_unique_path(candidates, str(path.parent / "usr" / "bin" / "bash.exe"))
    for root in (
        os.environ.get("ProgramFiles"),
        os.environ.get("ProgramFiles(x86)"),
        os.environ.get("LocalAppData"),
    ):
        if root:
            append_unique_path(candidates, str(Path(root) / "Git" / "bin" / "bash.exe"))
            append_unique_path(
                candidates, str(Path(root) / "Git" / "usr" / "bin" / "bash.exe")
            )
    for drive in ("C:", "D:"):
        drive_root = Path(drive + os.sep)
        append_unique_path(candidates, str(drive_root / "Git" / "bin" / "bash.exe"))
        append_unique_path(candidates, str(drive_root / "Git" / "usr" / "bin" / "bash.exe"))
        append_unique_path(
            candidates, str(drive_root / "msys64" / "usr" / "bin" / "bash.exe")
        )
    return candidates


def _bash_candidates() -> list[str]:
    candidates: list[str] = []
    if sys.platform == "win32":
        for candidate in _windows_bash_candidates():
            append_unique_path(candidates, candidate)
        append_unique_path(candidates, shutil.which("bash"))
    else:
        append_unique_path(candidates, shutil.which("bash"))
        append_unique_path(candidates, "/bin/bash")
        append_unique_path(candidates, "/usr/bin/bash")
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
            timeout=_BASH_FUNCTIONAL_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False
    return process.returncode == 0 and process.stdout == b"ok"


def _bash_shell_resolver(*, executable: str | None = None) -> ShellResolver:
    """Return the bash resolver for the configured or detected executable."""

    return ShellResolver(
        "bash",
        candidates=_bash_candidates,
        probe=_is_functional_bash,
        executable=executable,
        env_override=os.environ.get("PYGENT_BASH_PATH"),
        version_command=_BASH_VERSION_COMMAND,
        fallback="bash",
    )


def _find_bash_executable() -> str:
    return _bash_shell_resolver().resolve().executable


def bash_shell_identity(*, executable: str | None = None) -> ShellIdentity:
    """Resolve the bash identity used by the shell adapters."""

    return _bash_shell_resolver(executable=executable).resolve()


def _decode_output(data: bytes, max_bytes: int = _MAX_OUTPUT_BYTES) -> str:
    """Decode bounded bash output with the shared shell output policy."""

    return _process.decode_output(data, max_bytes)


class BashTools(NativeShellTools):
    """Deployment-local bash process adapter with workspace confinement."""

    tool_name: ClassVar[str] = "bash"
    shell_name: ClassVar[str] = "bash"

    def __init__(
        self,
        *,
        workspace_root: str | Path,
        bash_executable: str | None = None,
        restrict_to_workspace: bool = True,
        timeout: float = 600,
        task_manager: ToolTaskManager | None = None,
        task_event_sink: ToolEventEmitter | None = None,
    ) -> None:
        identity = _bash_shell_resolver(executable=bash_executable).resolve()
        super().__init__(
            workspace_root=workspace_root,
            shell_identity=identity,
            restrict_to_workspace=restrict_to_workspace,
            timeout=timeout,
            task_manager=task_manager,
            task_event_sink=task_event_sink,
        )
        self.bash_executable = identity.executable

    def _temporary_output_file(self) -> Any:
        # Kept on this module so deployments and tests can substitute the capture file.
        return tempfile.TemporaryFile()

    def _max_capture_bytes(self) -> int:
        return _MAX_FULL_OUTPUT_BYTES

    @tool(
        tool_id="standard.shell.bash",
        version="3.1.0",
        side_effect=ToolSideEffect.EXTERNAL,
        idempotency=IdempotencyPolicy.NOT_IDEMPOTENT,
        wait_timeout=600,
        wait_timeout_parameter="timeout",
        resource_key="shell",
        sandbox_profile="workspace",
        required_permissions=("shell:execute",),
    )
    async def bash(
        self,
        command: str,
        working_directory: str | None = None,
        description: str | None = None,
        is_background: bool = False,
        timeout: Annotated[float | None, Field(ge=0, allow_inf_nan=False)] = None,
    ) -> str | ToolTaskHandle:
        """Run one bash command in the configured workspace.

        Args:
            command: Complete command string passed to ``bash -lc``.
            working_directory: Directory resolved from workspace_root.
            description: Optional caller-facing description; not executed.
            is_background: Immediately return a reference to the managed task.
            timeout: Foreground wait in seconds; overrides the configured default.
                Expiry returns a background task reference without killing the command.
        """

        del description
        return await self._run_tool_call(
            command,
            working_directory,
            is_background=is_background,
            timeout=timeout,
        )

    @property
    def toolkit(self):
        from pygent.tool.functional import ToolKit

        return ToolKit(
            self.bash,
            self.tool_task_get,
            self.tool_task_stop,
            wait_timeouts={"bash": self.timeout},
        )

    def _control_manager(self) -> ToolTaskManager:
        if active_infrastructure() is not None:
            manager = current_infrastructure().resolve_tool_task_manager()
            if manager is not None:
                return cast(ToolTaskManager, manager)
        return self.task_manager

    @tool(
        tool_id="standard.shell.task_get",
        version="3.0.0",
        side_effect=ToolSideEffect.READ,
        task_control=True,
    )
    async def tool_task_get(self, task_id: str) -> dict[str, Any]:
        """Get a task snapshot, captured output and any final result without waiting."""
        manager = self._control_manager()
        result = await manager.get_result(task_id)
        task = result.task if result is not None else await manager.get_task(task_id)
        return {
            "task": _task_json(task),
            "output": thaw_json(await manager.get_output(task_id)),
            "result": _result_json(result),
        }

    @tool(
        tool_id="standard.shell.task_stop",
        version="3.0.0",
        side_effect=ToolSideEffect.WRITE,
        idempotency=IdempotencyPolicy.INHERENT,
        task_control=True,
    )
    async def tool_task_stop(self, task_id: str) -> dict[str, Any]:
        """Request task cancellation; the returned snapshot records confirmed state."""
        requested = await self._control_manager().cancel(task_id)
        snapshot = await self.tool_task_get(task_id)
        return {"cancel_requested": requested, **snapshot}

    def _resolve_working_directory(self, working_directory: str | None) -> str:
        return resolve_workspace_directory(working_directory, self.path_context)

    def _command_args(self, command: str) -> list[str]:
        return [self.bash_executable, "-lc", command]

    def _process_kwargs(self, cwd: str, output: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "cwd": cwd,
            "stdin": subprocess.DEVNULL,
            "stdout": output,
            "stderr": subprocess.STDOUT,
        }
        # Inject system proxy environment so subprocess networking follows host settings
        # without overriding explicit user-provided variables.
        try:
            env = os.environ.copy()
            proxy_env = getattr(self, "_system_proxy_env", None)
            if proxy_env is None:
                proxy_env = self._detect_system_proxy_env(env)
                self._system_proxy_env = proxy_env
            if proxy_env:
                existing = {k.lower() for k in env}
                for k, v in proxy_env.items():
                    if k.lower() not in existing and v:
                        env[k] = v
                # Mirror case variants for broader tool compatibility
                def _mirror(key: str) -> None:
                    if key.upper() in env and key.lower() not in env:
                        env[key.lower()] = env[key.upper()]
                    if key.lower() in env and key.upper() not in env:
                        env[key.upper()] = env[key.lower()]
                for k in ("http_proxy", "https_proxy", "all_proxy", "no_proxy"):
                    _mirror(k)
            kwargs["env"] = env
        # Proxy detection is best-effort: any failure must fall back to the
        # inherited environment instead of breaking the shell invocation.
        except Exception:  # noqa: BLE001
            kwargs["env"] = os.environ.copy()
        if self._is_windows:
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            kwargs["start_new_session"] = True
        return kwargs

    def _detect_system_proxy_env(self, base_env: dict[str, str]) -> dict[str, str]:
        result: dict[str, str] = {}
        try:
            if sys.platform == "win32":
                # WinINET user-level proxy
                try:
                    import winreg  # type: ignore

                    with winreg.OpenKey(
                        winreg.HKEY_CURRENT_USER,
                        r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
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
                        parts = [p.strip() for p in proxy_server.split(";") if p.strip()]
                        kv: dict[str, str] = {}
                        bare: str | None = None
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
                            result["ALL_PROXY"] = (
                                socks
                                if socks.startswith(("socks://", "socks5://", "socks4://"))
                                else f"socks5://{socks}"
                            )
                        if proxy_override:
                            no_proxy = ",".join([h.strip() for h in proxy_override.split(";") if h.strip()])
                            if no_proxy:
                                result["NO_PROXY"] = no_proxy
                # Registry probe is best-effort; an absent key is normal.
                except Exception:  # noqa: BLE001, S110
                    pass
                # WinHTTP fallback
                try:
                    proc = subprocess.run(
                        ["netsh", "winhttp", "show", "proxy"],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        text=True,
                        errors="replace",
                        timeout=_PROXY_DETECT_TIMEOUT_SECONDS,
                        check=False,
                    )
                    out = proc.stdout or ""
                    if "Direct access" not in out and out.strip():
                        for line in out.splitlines():
                            s = line.strip()
                            if s.lower().startswith("proxy server") and ":" in s:
                                rhs = s.split(":", 1)[1]
                                parts = [p.strip() for p in rhs.split(";") if p.strip()]
                                bare_netsh: str | None = None
                                for p in parts:
                                    if "=" in p:
                                        k, v = p.split("=", 1)
                                        k = k.strip().lower(); v = v.strip()
                                        if k == "http" and "HTTP_PROXY" not in result:
                                            result["HTTP_PROXY"] = f"http://{v}"
                                        if k == "https" and "HTTPS_PROXY" not in result:
                                            result["HTTPS_PROXY"] = f"http://{v}"
                                        if k.startswith("socks") and "ALL_PROXY" not in result:
                                            result["ALL_PROXY"] = (
                                                v
                                                if v.startswith(("socks://", "socks5://", "socks4://"))
                                                else f"socks5://{v}"
                                            )
                                    else:
                                        bare_netsh = p
                                if bare_netsh and "HTTP_PROXY" not in result:
                                    result["HTTP_PROXY"] = f"http://{bare_netsh}"
                                    result["HTTPS_PROXY"] = f"http://{bare_netsh}"
                            if s.lower().startswith("bypass list") and ":" in s and "NO_PROXY" not in result:
                                rhs = s.split(":", 1)[1]
                                items = [i.strip() for i in rhs.replace(",", ";").split(";") if i.strip()]
                                if items:
                                    result["NO_PROXY"] = ",".join(items)
                # WinHTTP/netsh parsing is best-effort; any failure just
                # leaves the result without proxy hints.
                except Exception:  # noqa: BLE001, S110
                    pass
            elif sys.platform == "darwin":
                try:
                    proc = subprocess.run(
                        ["scutil", "--proxy"],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        text=True,
                        errors="replace",
                        timeout=_PROXY_DETECT_TIMEOUT_SECONDS,
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
                            host = get_val("HTTPProxy"); port = get_val("HTTPPort")
                            if host and port:
                                result["HTTP_PROXY"] = f"http://{host}:{port}"
                        if enabled("HTTPSEnable"):
                            host = get_val("HTTPSProxy"); port = get_val("HTTPSPort")
                            if host and port:
                                result["HTTPS_PROXY"] = f"http://{host}:{port}"
                        if enabled("SOCKSEnable"):
                            host = get_val("SOCKSProxy"); port = get_val("SOCKSPort")
                            if host and port:
                                result["ALL_PROXY"] = f"socks5://{host}:{port}"
                        if "ExceptionsList" in out and "NO_PROXY" not in result:
                            items: list[str] = []
                            capture = False
                            for line in out.splitlines():
                                s = line.strip()
                                if s.startswith("ExceptionsList") and s.endswith("("):
                                    capture = True; continue
                                if capture:
                                    if s == ")":
                                        break
                                    s = s.strip("'\" ,")
                                    if s:
                                        items.append(s)
                            if items:
                                result["NO_PROXY"] = ",".join(items)
                # macOS proxy probe is best-effort; failure means no hints.
                except Exception:  # noqa: BLE001, S110
                    pass
            else:
                # Linux: try GNOME gsettings if available
                try:
                    if shutil.which("gsettings"):
                        mode_proc = subprocess.run(
                            ["gsettings", "get", "org.gnome.system.proxy", "mode"],
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL,
                            text=True,
                            errors="replace",
                            timeout=_GSETTINGS_TIMEOUT_SECONDS,
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
                                    errors="replace",
                                    timeout=_GSETTINGS_TIMEOUT_SECONDS,
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
                                result["HTTP_PROXY"] = f"http://{http_host.strip(_GSETTINGS_QUOTES)}:{http_port}"
                            if https_host and https_host != "''" and https_port and https_port.isdigit():
                                result["HTTPS_PROXY"] = f"http://{https_host.strip(_GSETTINGS_QUOTES)}:{https_port}"
                            if socks_host and socks_host != "''" and socks_port and socks_port.isdigit():
                                result["ALL_PROXY"] = f"socks5://{socks_host.strip(_GSETTINGS_QUOTES)}:{socks_port}"
                            ignore = get_schema("org.gnome.system.proxy", "ignore-hosts")
                            if ignore and ignore.startswith("["):
                                items = [i.strip().strip("'\"") for i in ignore.strip("[]").split(",") if i.strip()]
                                if items:
                                    result["NO_PROXY"] = ",".join(items)
                # gsettings probe is best-effort; failure just means no
                # proxy hints from the desktop environment.
                except Exception:  # noqa: BLE001, S110
                    pass
        # The whole environment probe is best-effort: callers fall back to the
        # inherited environment when this returns {}.
        except Exception:  # noqa: BLE001
            return {}
        return result

def _task_json(task: ToolTask | None) -> dict[str, Any] | None:
    if task is None:
        return None
    return {
        "task_id": task.task_id,
        "call_id": task.call_id,
        "tool_id": task.tool_id,
        "version": task.version,
        "state": task.state.value,
        "job_id": task.job_id,
        "metadata": thaw_json(cast(JsonValue, task.metadata)),
    }


def _result_json(result: ToolResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "call_id": result.call_id,
        "name": result.name,
        "status": result.status,
        "task": _task_json(result.task),
        "output": thaw_json(result.output),
        "content": [
            _tool_result_content_to_value(item) for item in result.content
        ],
        "error": result.error,
        "error_kind": result.error_kind,
        "error_code": result.error_code,
        "retryable": result.retryable,
        "side_effect_committed": result.side_effect_committed,
        "tool_id": result.tool_id,
        "tool_version": result.tool_version,
        "missing_capabilities": list(result.missing_capabilities),
    }


__all__ = ["BashTools", "bash_shell_identity"]
