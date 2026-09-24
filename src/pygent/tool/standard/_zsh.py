"""Workspace-scoped native Zsh adapter for macOS, Linux and other POSIX hosts.

The adapter supplies the executable, argv and environment for native Zsh and
shares confinement, wait and process handling with the other shell adapters.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Annotated, Any, ClassVar

from pydantic import Field

from pygent.tool.executors import ToolEventEmitter, ToolTaskManager
from pygent.tool.functional import tool
from pygent.tool.task_handle import ToolTaskHandle
from pygent.tool.types import IdempotencyPolicy, ToolSideEffect

from ._shell import ShellIdentity, ShellResolver, append_unique_path, run_probe
from ._shell_tools import NativeShellTools

_ZSH_VERSION_COMMAND = 'printf %s "$ZSH_VERSION"'
_ZSH_ENV_OVERRIDE = "PYGENT_ZSH_PATH"
_FUNCTIONAL_PROBE = "printf ok"


def _zsh_candidates() -> list[str]:
    candidates: list[str] = []
    for candidate in (
        shutil.which("zsh"),
        "/bin/zsh",
        "/usr/bin/zsh",
        "/usr/local/bin/zsh",
        "/opt/homebrew/bin/zsh",
        "/opt/local/bin/zsh",
    ):
        append_unique_path(candidates, candidate)
    return candidates


def _is_functional_zsh(executable: str) -> bool:
    probed = run_probe("zsh", executable, _FUNCTIONAL_PROBE)
    if probed is None:
        return False
    returncode, output = probed
    return returncode == 0 and output == "ok"


def zsh_shell_identity(*, executable: str | None = None) -> ShellIdentity:
    """Resolve the Zsh identity used by the shell adapters."""

    return ShellResolver(
        "zsh",
        candidates=_zsh_candidates,
        probe=_is_functional_zsh,
        executable=executable,
        env_override=os.environ.get(_ZSH_ENV_OVERRIDE),
        version_command=_ZSH_VERSION_COMMAND,
    ).resolve()


def _find_zsh_executable() -> str:
    return (
        ShellResolver(
            "zsh",
            candidates=_zsh_candidates,
            probe=_is_functional_zsh,
            env_override=os.environ.get(_ZSH_ENV_OVERRIDE),
            version_command=_ZSH_VERSION_COMMAND,
        )
        .resolve()
        .executable
    )


class ZshTools(NativeShellTools):
    """Deployment-local native Zsh adapter with workspace confinement."""

    tool_name: ClassVar[str] = "zsh"
    shell_name: ClassVar[str] = "zsh"

    def __init__(
        self,
        *,
        workspace_root: str | Path,
        zsh_executable: str | None = None,
        restrict_to_workspace: bool = True,
        timeout: float = 600,
        task_manager: ToolTaskManager | None = None,
        task_event_sink: ToolEventEmitter | None = None,
    ) -> None:
        identity = ShellResolver(
            "zsh",
            candidates=_zsh_candidates,
            probe=_is_functional_zsh,
            executable=zsh_executable,
            env_override=os.environ.get(_ZSH_ENV_OVERRIDE),
            version_command=_ZSH_VERSION_COMMAND,
        ).resolve()
        super().__init__(
            workspace_root=workspace_root,
            shell_identity=identity,
            restrict_to_workspace=restrict_to_workspace,
            timeout=timeout,
            task_manager=task_manager,
            task_event_sink=task_event_sink,
        )
        self.zsh_executable = identity.executable

    @tool(
        tool_id="standard.shell.zsh",
        version="1.0.0",
        side_effect=ToolSideEffect.EXTERNAL,
        idempotency=IdempotencyPolicy.NOT_IDEMPOTENT,
        wait_timeout=600,
        wait_timeout_parameter="timeout",
        resource_key="shell",
        sandbox_profile="workspace",
        required_permissions=("shell:execute",),
    )
    async def zsh(
        self,
        command: str,
        working_directory: str | None = None,
        description: str | None = None,
        is_background: bool = False,
        timeout: Annotated[float | None, Field(ge=0, allow_inf_nan=False)] = None,
    ) -> str | ToolTaskHandle:
        """Run one native Zsh command in the configured workspace.

        The command runs in a fresh login Zsh process (``zsh -lc``); working
        directory and shell state do not persist between calls.

        Args:
            command: Complete Zsh command string.
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
        return [self.shell_executable, "-lc", command]

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


__all__ = ["ZshTools", "zsh_shell_identity"]
