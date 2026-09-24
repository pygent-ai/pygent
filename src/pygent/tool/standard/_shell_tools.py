"""Shared assembly for the workspace-scoped native shell adapters.

Every native shell adapter declares its own tool identity and supplies the
executable, argv and environment; workspace confinement, foreground wait, detach
admission, process execution and output projection are identical, so they live
here. No adapter in this package translates or reinterprets a Shell Language.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path
from typing import Any, ClassVar, cast
from uuid import uuid4

from pygent.core._tool_values import ToolCall
from pygent.tool._waiting import resolve_wait_timeout
from pygent.tool.executors import ToolEventEmitter, ToolExecutionError, ToolTaskManager
from pygent.tool.task_handle import ToolTaskHandle

from . import _process
from ._paths import ToolPathContext, resolve_workspace_directory
from ._shell import ShellIdentity


class NativeShellTools:
    """Assembly for one native shell executable and its managed task facility."""

    tool_name: ClassVar[str] = ""
    shell_name: ClassVar[str] = ""

    def __init__(
        self,
        *,
        workspace_root: str | Path,
        shell_identity: ShellIdentity,
        restrict_to_workspace: bool = True,
        timeout: float = 600,
        task_manager: ToolTaskManager | None = None,
        task_event_sink: ToolEventEmitter | None = None,
        max_output_bytes: int = _process.MAX_OUTPUT_BYTES,
        max_capture_bytes: int = _process.MAX_FULL_OUTPUT_BYTES,
    ) -> None:
        if not isinstance(shell_identity, ShellIdentity):
            raise TypeError("shell_identity must be a ShellIdentity")
        if min(max_output_bytes, max_capture_bytes) <= 0:
            raise ValueError("max_output_bytes and max_capture_bytes must be positive")
        configured_timeout = resolve_wait_timeout(timeout)
        if configured_timeout is None:
            raise ValueError("configured timeout must be a number of seconds")
        self.timeout = configured_timeout
        self._output_byte_limit = max_output_bytes
        self._capture_byte_limit = max_capture_bytes
        self._task_manager = task_manager
        self._owns_task_manager = task_manager is None
        self._task_event_sink = task_event_sink
        self._closed = False
        self.path_context = ToolPathContext.from_workspace_root(
            workspace_root, restrict_to_workspace=restrict_to_workspace
        )
        self.workspace_root = self.path_context.workspace_root
        self.shell_identity = shell_identity
        self.shell_executable = shell_identity.executable
        self._is_windows = sys.platform == "win32"

    async def _run_tool_call(
        self,
        command: str,
        working_directory: str | None,
        *,
        is_background: bool,
        timeout: float | None,
    ) -> str | ToolTaskHandle:
        """Run one call directly or through the managed task facility."""

        from pygent.tool.executors import current_tool_execution

        wait_timeout = resolve_wait_timeout(
            self.timeout, timeout, is_background=is_background
        )
        cwd = self._resolve_working_directory(working_directory)
        context = current_tool_execution()
        if context is not None:
            return await self._run_process(command or "", cwd)
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed")
        kit = self.toolkit
        spec = kit.specs[0]
        registry = kit.build_registry()
        task = await self.task_manager.submit(
            spec,
            ToolCall(
                call_id=f"{self.tool_name}-{uuid4()}",
                name=self.tool_name,
                arguments={
                    "command": command,
                    "working_directory": working_directory,
                },
            ),
            execution=registry.execute,
        )
        handle = ToolTaskHandle(self.task_manager, task.task_id)
        if is_background:
            return handle
        result = await handle.wait(wait_timeout)
        if result is None:
            return handle
        if result.status != "succeeded":
            raise ToolExecutionError(
                result.error or f"{self.tool_name} task did not succeed",
                kind=result.error_kind or "executor_error",
                code=result.error_code,
                retryable=result.retryable,
                side_effect_committed=result.side_effect_committed,
                missing_capabilities=result.missing_capabilities,
            )
        return cast(str, result.output)

    async def _run_process(self, command: str, cwd: str) -> str:
        with self._temporary_output_file() as output_file:
            return await _process.run_shell_command(
                _process.ShellProcess(
                    output_file,
                    max_capture_bytes=self._max_capture_bytes(),
                    task_prefix=self._task_prefix(),
                ),
                output_file,
                argv=self._command_args(command),
                process_kwargs=self._process_kwargs(cwd, asyncio.subprocess.PIPE),
                cwd=cwd,
                shell_name=self.shell_name,
                is_windows=self._is_windows,
                max_output_bytes=self._max_output_bytes(),
                max_capture_bytes=self._max_capture_bytes(),
                output_prefix=self._output_prefix(),
            )

    def _resolve_working_directory(self, working_directory: str | None) -> str:
        return resolve_workspace_directory(working_directory, self.path_context)

    def _temporary_output_file(self) -> Any:
        """Return the temporary capture file; adapters may keep it patchable."""

        return tempfile.TemporaryFile()

    def _max_output_bytes(self) -> int:
        return self._output_byte_limit

    def _max_capture_bytes(self) -> int:
        return self._capture_byte_limit

    def _task_prefix(self) -> str:
        return f"pygent-{self.shell_name}"

    def _output_prefix(self) -> str:
        return f"pygent_{self.shell_name}_output"

    def _command_args(self, command: str) -> list[str]:
        raise NotImplementedError

    def _process_kwargs(self, cwd: str, output: Any) -> dict[str, Any]:
        raise NotImplementedError

    @property
    def toolkit(self):
        from pygent.tool.functional import ToolKit

        return ToolKit(
            getattr(self, self.tool_name),
            wait_timeouts={self.tool_name: self.timeout},
        )

    @property
    def task_manager(self) -> ToolTaskManager:
        from pygent.tool.executors import InMemoryToolTaskManager

        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed")
        if self._task_manager is None:
            self._task_manager = InMemoryToolTaskManager(
                self.toolkit.build_registry(), emit=self._task_event_sink,
            )
        return self._task_manager

    async def aclose(self) -> None:
        self._closed = True
        if self._owns_task_manager and self._task_manager is not None:
            await self._task_manager.close(cancel=True)

    async def close(self) -> None:
        await self.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


__all__ = ["NativeShellTools"]
