"""Interactive terminal sessions: explicit ToolTask ownership and bounded I/O.

An interactive terminal is one admitted ToolTask that owns a live native shell
process. Nothing about it is hidden: the session outlives the call that started
it only because that call was admitted as an independent task, input reaches the
session through the trusted task-control family the queries and stops already
use, and output is published as the task output snapshot. There is no second tool
API and no implicit "current terminal".

Rebuild note: this module was reconstructed from
``tests/tool/standard/test_terminal_tool.py`` after a concurrent edit overwrote
the previous revision. See ``docs/tool/NATIVE_SHELL_NEXT_STEPS.md``.
"""

from __future__ import annotations

import asyncio
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, ClassVar, cast
from uuid import uuid4

from pydantic import Field

from pygent.core import active_infrastructure, current_infrastructure
from pygent.core._tool_values import ToolCall
from pygent.tool._waiting import resolve_wait_timeout
from pygent.tool.executors import (
    LiveToolTaskRegistry,
    ToolExecutionError,
    ToolTaskManager,
    current_tool_execution,
)
from pygent.tool.functional import tool
from pygent.tool.task_handle import ToolTaskHandle
from pygent.tool.types import IdempotencyPolicy, ToolSideEffect

from . import _process
from ._bash import bash_shell_identity
from ._powershell import powershell_shell_identity
from ._shell import ShellIdentity
from ._shell_tools import NativeShellTools
from ._zsh import zsh_shell_identity

_MAX_OUTPUT_BYTES = _process.MAX_OUTPUT_BYTES
_TERMINAL_CAPTURE_BYTES = 4 * 1024 * 1024
_TERMINAL_WINDOW_BYTES = 64 * 1024
_TERMINAL_PUBLISH_INTERVAL_SECONDS = 0.25
_TERMINAL_READ_TIMEOUT_SECONDS = 1.5
_TERMINAL_WAIT_TIMEOUT_SECONDS = 10.0
_MAX_INPUT_BYTES = 64 * 1024

_SHELL_IDENTITIES = {
    "bash": bash_shell_identity,
    "zsh": zsh_shell_identity,
    "powershell": powershell_shell_identity,
}


def _resolve_terminal_identity(
    shell: str, *, shell_executable: str | None = None
) -> ShellIdentity:
    resolver = _SHELL_IDENTITIES.get(shell)
    if resolver is None:
        supported = ", ".join(sorted(_SHELL_IDENTITIES))
        raise ValueError(
            f"unsupported terminal shell: {shell!r}; supported: {supported}"
        )
    return resolver(executable=shell_executable)


def _interactive_command_args(identity: ShellIdentity) -> list[str]:
    """Return the argv that starts one interactive session for ``identity``."""

    if identity.name == "powershell":
        return [
            identity.executable,
            *identity.args,
            "-NoLogo",
            "-NoProfile",
            "-Command",
            "-",
        ]
    return [identity.executable, *identity.args, "-l"]


def _tail_window(
    output_file: Any, process: _process.ShellProcess, limit: int
) -> bytes:
    end = process.captured
    start = max(0, end - limit)
    position = output_file.tell()
    output_file.seek(start)
    data = output_file.read(end - start)
    output_file.seek(position)
    return data


@dataclass(slots=True)
class TerminalSession:
    """One live interactive session owned by an admitted terminal ToolTask."""

    task_id: str
    identity: ShellIdentity
    process: _process.ShellProcess
    output_file: Any
    cwd: str
    initial_command: str | None = None
    backend: str = "pipe"
    closed: bool = False
    final_output: str | None = None
    published_bytes: int = field(default=0)
    # The owning task's bounded output channel; an input call publishes through
    # it so the task snapshot reflects the answer to that input immediately.
    publish: Any = None
    # Ends the session bookkeeping when the admitted task stops, including a
    # task cancelled before its driver ever started.
    watcher: Any = None
    # The task facility that admitted this session; kept so bookkeeping still
    # works while the assembly itself is closing.
    manager: Any = None

    def tail(self, limit: int = _TERMINAL_WINDOW_BYTES) -> str:
        """Return the newest bounded window of session output."""

        return _process.decode_output(
            _tail_window(self.output_file, self.process, limit), limit
        )

    async def write(self, payload: bytes) -> bool:
        if self.closed:
            return False
        return await self.process.write_stdin(payload)

    def finalize(self, *, cleanup_complete: bool) -> str:
        """Freeze the bounded final projection; later reads never touch the file."""

        if self.final_output is None:
            transport = self.process.transport
            returncode = transport.get_returncode() if transport else None
            exit_code: int | str = returncode if returncode is not None else -1
            self.final_output = _process.format_result(
                exit_code,
                self.tail(_MAX_OUTPUT_BYTES),
                max_output_bytes=_MAX_OUTPUT_BYTES,
                max_capture_bytes=_TERMINAL_CAPTURE_BYTES,
                capture_truncated=self.process.truncated,
                cleanup_complete=cleanup_complete,
            )
        return self.final_output

    def release(self) -> None:
        """Close the capture file after the projection has been frozen."""

        if not self.output_file.closed:
            self.output_file.close()

    def close(self, *, cleanup_complete: bool = False) -> str:
        """Freeze this session exactly once and return its final projection."""

        if not self.closed:
            self.closed = True
            self.finalize(cleanup_complete=cleanup_complete)
            self.release()
        return self.final_output or ""

    async def wait(self) -> str:
        """Wait until the session ends and return its bounded final projection."""

        if self.final_output is not None:
            return self.final_output
        try:
            await asyncio.wait_for(
                asyncio.shield(self.process.outcome_ready),
                _TERMINAL_WAIT_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            pass
        return self.finalize(cleanup_complete=True)


class TerminalTools(NativeShellTools):
    """Deployment-local interactive terminal adapter for one native shell."""

    tool_name: ClassVar[str] = "terminal"
    shell_name: ClassVar[str] = "terminal"

    def __init__(
        self,
        *,
        workspace_root: str | Path,
        shell: str = "bash",
        shell_executable: str | None = None,
        restrict_to_workspace: bool = True,
        timeout: float = 0,
        task_manager: ToolTaskManager | None = None,
    ) -> None:
        # A session is an explicit ToolTask, so the foreground wait only decides
        # how long the starting call observes startup before returning the task
        # reference; it never bounds the session itself.
        super().__init__(
            workspace_root=workspace_root,
            shell_identity=_resolve_terminal_identity(
                shell, shell_executable=shell_executable
            ),
            restrict_to_workspace=restrict_to_workspace,
            timeout=timeout,
            task_manager=task_manager,
        )
        self.sessions: dict[str, TerminalSession] = {}

    @tool(
        tool_id="standard.shell.terminal",
        version="1.0.0",
        side_effect=ToolSideEffect.EXTERNAL,
        idempotency=IdempotencyPolicy.NOT_IDEMPOTENT,
        wait_timeout=0,
        wait_timeout_parameter="timeout",
        resource_key="shell",
        required_permissions=("shell:execute",),
    )
    async def terminal(
        self,
        command: str | None = None,
        working_directory: str | None = None,
        description: str | None = None,
        timeout: Annotated[float | None, Field(ge=0, allow_inf_nan=False)] = None,
    ) -> str | ToolTaskHandle:
        """Start one interactive native shell session and return its task reference.

        The session owns its process until the task ends, so ``task_id`` is the
        only handle to it: ``terminal_input`` writes to it, the task output
        snapshot carries its bounded output, and ``tool_task_stop`` or the task
        handle cancel it. A session is not confined beyond its initial working
        directory and is never recovered after a Runtime restart.

        Args:
            command: Optional first command line executed after startup.
            working_directory: Directory resolved from workspace_root.
            description: Optional caller-facing description; not executed.
            timeout: Foreground wait in seconds before returning the reference.
        """

        del description
        context = current_tool_execution()
        if context is not None and context.admitted:
            # This invocation *is* the admitted session owner: drive the session
            # that the starting call registered for this task.
            session = await self._find_session(context.task_id or "")
            if session is None:
                raise ToolExecutionError(
                    "terminal session is no longer registered",
                    kind="executor_error",
                    code="terminal_session_lost",
                    side_effect_committed=False,
                )
            return await self._drive_session(session, context)
        cwd = self._resolve_working_directory(working_directory)
        wait_timeout = resolve_wait_timeout(self.timeout, timeout)
        if self._closed:
            raise RuntimeError("TerminalTools is closed")
        kit = self.toolkit
        spec = kit.specs[0]
        registry = kit.build_registry()
        snapshot = await self.task_manager.prepare(
            spec,
            ToolCall(
                call_id=f"terminal-{uuid4()}",
                name="terminal",
                arguments={
                    "command": command,
                    "working_directory": working_directory,
                },
            ),
            execution=registry.execute,
        )
        await self._open_session(snapshot.task_id, cwd, command)
        await self.task_manager.start(snapshot.task_id)
        session = self.sessions[snapshot.task_id]
        session.watcher = asyncio.create_task(
            self._watch_session(session), name="pygent-terminal-watch"
        )
        handle = ToolTaskHandle(self.task_manager, snapshot.task_id)
        if wait_timeout > 0:
            result = await handle.wait(wait_timeout)
            if result is not None:
                if result.status != "succeeded":
                    raise ToolExecutionError(
                        result.error or "terminal task did not succeed",
                        kind=result.error_kind or "executor_error",
                        code=result.error_code,
                        retryable=result.retryable,
                        side_effect_committed=result.side_effect_committed,
                        missing_capabilities=result.missing_capabilities,
                    )
                return cast(str, result.output)
        return handle

    @tool(
        tool_id="standard.shell.terminal_input",
        version="1.0.0",
        side_effect=ToolSideEffect.EXTERNAL,
        idempotency=IdempotencyPolicy.NOT_IDEMPOTENT,
        task_control=True,
        resource_key="terminal",
        required_permissions=("shell:execute",),
    )
    async def terminal_input(self, task_id: str, input: str) -> dict[str, Any]:
        """Write one line to a running terminal session and report its newest output.

        The call never creates, adopts or restarts a session: an unknown, ended
        or foreign ``task_id`` is refused instead of silently starting a process.
        """

        if not task_id or not task_id.strip():
            raise ToolExecutionError(
                "task_id must be a non-empty string",
                kind="validation_error",
                code="unknown_terminal_task",
                side_effect_committed=False,
            )
        payload = (input + "\n").encode("utf-8")
        if len(payload) > _MAX_INPUT_BYTES:
            raise ToolExecutionError(
                f"input exceeds the {_MAX_INPUT_BYTES} byte terminal input budget",
                kind="validation_error",
                code="input_too_large",
                side_effect_committed=False,
            )
        session = await self._find_session(task_id)
        if session is None or session.closed:
            raise ToolExecutionError(
                f"unknown terminal task: {task_id}",
                kind="validation_error",
                code="unknown_terminal_task",
                side_effect_committed=False,
            )
        before = session.process.captured
        if not await self._await_process(session):
            raise ToolExecutionError(
                f"input was not confirmed for terminal task {task_id}",
                kind="executor_error",
                code="input_not_confirmed",
                side_effect_committed=None,
            )
        if not await session.write(payload):
            raise ToolExecutionError(
                f"input was not confirmed for terminal task {task_id}",
                kind="executor_error",
                code="input_not_confirmed",
                side_effect_committed=None,
            )
        await self._await_new_output(session, before)
        if session.publish is not None:
            await self._publish(session, session.publish)
        return {
            "task_id": task_id,
            "written": True,
            "state": "closed" if session.closed else "running",
            "backend": session.backend,
            "output": session.tail(),
        }

    @property
    def toolkit(self):
        from pygent.tool.functional import ToolKit

        return ToolKit(
            self.terminal,
            self.terminal_input,
            wait_timeouts={"terminal": self.timeout},
        )

    async def aclose(self) -> None:
        self._closed = True
        if self._owns_task_manager and self._task_manager is not None:
            await self._task_manager.close(cancel=True)
        watchers = []
        for session in tuple(self.sessions.values()):
            session.close(cleanup_complete=False)
            watcher = session.watcher
            if isinstance(watcher, asyncio.Task) and not watcher.done():
                watcher.cancel()
                watchers.append(watcher)
        self.sessions.clear()
        if watchers:
            await asyncio.gather(*watchers, return_exceptions=True)

    async def _watch_session(self, session: TerminalSession) -> None:
        """Release one session as soon as its admitted task reaches a terminal state."""

        manager = session.manager or self._active_manager()
        if manager is None:
            return
        await ToolTaskHandle(manager, session.task_id).wait(None)
        if session.closed:
            return
        session.close(cleanup_complete=False)
        self.sessions.pop(session.task_id, None)
        await self._unregister_session(session)

    async def _open_session(
        self, task_id: str, cwd: str, initial_command: str | None
    ) -> TerminalSession:
        # The capture file belongs to the session, not to this call, so it stays
        # open until the session is released.
        output_file = tempfile.TemporaryFile()  # noqa: SIM115
        session = TerminalSession(
            task_id=task_id,
            identity=self.shell_identity,
            process=_process.ShellProcess(
                output_file,
                max_capture_bytes=_TERMINAL_CAPTURE_BYTES,
                task_prefix="pygent-terminal",
            ),
            output_file=output_file,
            cwd=cwd,
            initial_command=initial_command,
        )
        self.sessions[task_id] = session
        manager = self._control_manager()
        session.manager = manager
        if isinstance(manager, LiveToolTaskRegistry):
            # A trusted control call issued from another assembly resolves this
            # same live session through the task manager, not a private dict.
            await manager.register_live_resource(task_id, session)
        return session

    async def _find_session(self, task_id: str) -> TerminalSession | None:
        session = self.sessions.get(task_id)
        if session is not None:
            return session
        manager = self._active_manager()
        if isinstance(manager, LiveToolTaskRegistry):
            resource = await manager.get_live_resource(task_id)
            if isinstance(resource, TerminalSession):
                return resource
        return None

    def _active_manager(self) -> ToolTaskManager | None:
        if active_infrastructure() is not None:
            manager = current_infrastructure().resolve_tool_task_manager()
            if manager is not None:
                return cast(ToolTaskManager, manager)
        if self._closed:
            return self._task_manager
        return self.task_manager

    def _control_manager(self) -> ToolTaskManager:
        manager = self._active_manager()
        if manager is None:
            raise RuntimeError("TerminalTools is closed")
        return manager

    async def _await_new_output(self, session: TerminalSession, before: int) -> None:
        """Give the shell a bounded chance to answer before reporting output."""

        loop = asyncio.get_running_loop()
        deadline = loop.time() + _TERMINAL_READ_TIMEOUT_SECONDS
        while (
            session.process.captured == before
            and not session.closed
            and loop.time() < deadline
        ):
            await asyncio.sleep(0.02)

    async def _await_process(self, session: TerminalSession) -> bool:
        """Wait for a just-admitted session to accept its stdin pipe."""

        loop = asyncio.get_running_loop()
        deadline = loop.time() + _TERMINAL_READ_TIMEOUT_SECONDS
        while (
            session.process.transport is None
            and not session.closed
            and loop.time() < deadline
        ):
            await asyncio.sleep(0.02)
        return session.process.transport is not None and not session.closed

    async def _drive_session(self, session: TerminalSession, context: Any) -> str:
        process = session.process
        publish_output = context.publish_output
        session.publish = publish_output
        cancelled = False
        try:
            try:
                await process.start(
                    *_interactive_command_args(session.identity),
                    **self._process_kwargs(session.cwd),
                )
            except FileNotFoundError as exc:
                raise ToolExecutionError(
                    f"{session.identity.name} executable was not found",
                    kind="process_error",
                    code="executable_not_found",
                    side_effect_committed=False,
                ) from exc
            except OSError as exc:
                raise ToolExecutionError(
                    "terminal could not be started",
                    kind="process_error",
                    code="process_start_failed",
                    side_effect_committed=False,
                ) from exc
            if session.initial_command is not None and not await session.write(
                (session.initial_command + "\n").encode("utf-8")
            ):
                raise ToolExecutionError(
                    "terminal stdin is not writable",
                    kind="process_error",
                    code="stdin_unavailable",
                    side_effect_committed=False,
                )
            while not process.outcome_ready.done():
                done, _ = await asyncio.wait(
                    {process.outcome_ready},
                    timeout=_TERMINAL_PUBLISH_INTERVAL_SECONDS,
                )
                await self._publish(session, publish_output)
                if done:
                    break
        except asyncio.CancelledError:
            cancelled = True
        finally:
            cleanup = asyncio.create_task(
                process.aclose(self._is_windows),
                name="pygent-terminal-cleanup",
            )
            # Repeated caller cancellation must neither restart the budget nor
            # abandon the task that owns the transport and output file.
            while True:
                try:
                    cleanup_complete = await asyncio.shield(cleanup)
                    break
                except asyncio.CancelledError:
                    if cleanup.cancelled():
                        raise
                    cancelled = True
            await self._publish(session, publish_output)
            self.sessions.pop(session.task_id, None)
            await self._unregister_session(session)
            final = session.close(cleanup_complete=cleanup_complete)
        if cancelled:
            raise asyncio.CancelledError
        return final

    async def _unregister_session(self, session: TerminalSession) -> None:
        manager = session.manager or self._active_manager()
        if isinstance(manager, LiveToolTaskRegistry):
            await manager.unregister_live_resource(session.task_id, session)

    async def _publish(self, session: TerminalSession, publish_output: Any) -> None:
        if publish_output is None or session.process.captured == session.published_bytes:
            return
        await publish_output(session.tail())
        session.published_bytes = session.process.captured

    def _process_kwargs(self, cwd: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "cwd": cwd,
            "stdin": subprocess.PIPE,
            "stdout": asyncio.subprocess.PIPE,
            "stderr": subprocess.STDOUT,
        }
        if self._is_windows:
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            kwargs["start_new_session"] = True
        return kwargs

    def _command_args(self, command: str) -> list[str]:
        raise NotImplementedError("interactive sessions start without arguments")


__all__ = ["TerminalSession", "TerminalTools"]
