"""Workspace-scoped native PowerShell adapter for Windows and PowerShell 7 hosts.

The adapter supplies the executable, argv and environment for native PowerShell
and shares confinement, wait and process handling with the other shell adapters.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Any, ClassVar

from pydantic import Field

from pygent.tool.executors import ToolEventEmitter, ToolTaskManager
from pygent.tool.functional import tool
from pygent.tool.task_handle import ToolTaskHandle
from pygent.tool.types import IdempotencyPolicy, ToolSideEffect

from ._shell import ShellIdentity, ShellResolver, append_unique_path, run_probe
from ._shell_tools import NativeShellTools

_POWERSHELL_VERSION_COMMAND = "$PSVersionTable.PSVersion.ToString()"
_POWERSHELL_ENV_OVERRIDE = "PYGENT_POWERSHELL_PATH"
_FUNCTIONAL_PROBE = "Write-Output ok"


def _powershell_candidates() -> list[str]:
    candidates: list[str] = []
    append_unique_path(candidates, shutil.which("pwsh"))
    if sys.platform == "win32":
        for root in (
            os.environ.get("ProgramFiles"),
            os.environ.get("ProgramFiles(x86)"),
            os.environ.get("LocalAppData"),
        ):
            if root:
                for version in ("7", "6"):
                    append_unique_path(
                        candidates, str(Path(root) / "PowerShell" / version / "pwsh.exe")
                    )
        system_root = os.environ.get("SystemRoot") or os.environ.get("WINDIR")
        if system_root:
            append_unique_path(
                candidates,
                str(
                    Path(system_root)
                    / "System32"
                    / "WindowsPowerShell"
                    / "v1.0"
                    / "powershell.exe"
                ),
            )
    else:
        for candidate in (
            "/usr/bin/pwsh",
            "/usr/local/bin/pwsh",
            "/opt/microsoft/powershell/7/pwsh",
            "/snap/bin/pwsh",
        ):
            append_unique_path(candidates, candidate)
    append_unique_path(candidates, shutil.which("powershell"))
    return candidates


def _is_functional_powershell(executable: str) -> bool:
    probed = run_probe("powershell", executable, _FUNCTIONAL_PROBE)
    if probed is None:
        return False
    returncode, output = probed
    return returncode == 0 and output == "ok"


def powershell_shell_identity(*, executable: str | None = None) -> ShellIdentity:
    """Resolve the PowerShell identity used by the shell adapters."""

    return ShellResolver(
        "powershell",
        candidates=_powershell_candidates,
        probe=_is_functional_powershell,
        executable=executable,
        env_override=os.environ.get(_POWERSHELL_ENV_OVERRIDE),
        version_command=_POWERSHELL_VERSION_COMMAND,
    ).resolve()


def _find_powershell_executable() -> str:
    return (
        ShellResolver(
            "powershell",
            candidates=_powershell_candidates,
            probe=_is_functional_powershell,
            env_override=os.environ.get(_POWERSHELL_ENV_OVERRIDE),
            version_command=_POWERSHELL_VERSION_COMMAND,
        )
        .resolve()
        .executable
    )


class PowerShellTools(NativeShellTools):
    """Deployment-local native PowerShell adapter with workspace confinement."""

    tool_name: ClassVar[str] = "powershell"
    shell_name: ClassVar[str] = "powershell"

    def __init__(
        self,
        *,
        workspace_root: str | Path,
        powershell_executable: str | None = None,
        restrict_to_workspace: bool = True,
        timeout: float = 600,
        task_manager: ToolTaskManager | None = None,
        task_event_sink: ToolEventEmitter | None = None,
    ) -> None:
        identity = ShellResolver(
            "powershell",
            candidates=_powershell_candidates,
            probe=_is_functional_powershell,
            executable=powershell_executable,
            env_override=os.environ.get(_POWERSHELL_ENV_OVERRIDE),
            version_command=_POWERSHELL_VERSION_COMMAND,
        ).resolve()
        super().__init__(
            workspace_root=workspace_root,
            shell_identity=identity,
            restrict_to_workspace=restrict_to_workspace,
            timeout=timeout,
            task_manager=task_manager,
            task_event_sink=task_event_sink,
        )
        self.powershell_executable = identity.executable

    @tool(
        tool_id="standard.shell.powershell",
        version="1.0.0",
        side_effect=ToolSideEffect.EXTERNAL,
        idempotency=IdempotencyPolicy.NOT_IDEMPOTENT,
        wait_timeout=600,
        wait_timeout_parameter="timeout",
        resource_key="shell",
        sandbox_profile="workspace",
        required_permissions=("shell:execute",),
    )
    async def powershell(
        self,
        command: str,
        working_directory: str | None = None,
        description: str | None = None,
        is_background: bool = False,
        timeout: Annotated[float | None, Field(ge=0, allow_inf_nan=False)] = None,
    ) -> str | ToolTaskHandle:
        """Run one native PowerShell command in the configured workspace.

        The command runs in a fresh, non-interactive PowerShell process started
        with ``-NoLogo -NoProfile -NonInteractive -Command``; working directory
        and environment variables do not persist between calls. The reported
        exit code is the process exit code, so cmdlets that report a
        non-terminating error can still exit successfully.

        Args:
            command: Complete PowerShell command string.
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

    def _command_args(self, command: str) -> list[str]:
        return [
            self.shell_executable,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            command,
        ]

    def _process_kwargs(self, cwd: str, output: Any) -> dict[str, Any]:
        # PowerShell inherits the host environment; the adapter does not rewrite
        # proxy variables, because .NET already follows the system proxy.
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


__all__ = ["PowerShellTools", "powershell_shell_identity"]
