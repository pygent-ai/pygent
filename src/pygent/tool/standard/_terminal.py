"""Interactive terminal sessions: explicit ToolTask ownership and bounded I/O.

An interactive terminal is one admitted ToolTask that owns a live native shell
process. Nothing about it is hidden: the session outlives the call that started
it only because that call was admitted as an independent task, input reaches the
session through the trusted task-control family the queries and stops already
use, and output is published as the task output snapshot. There is no second tool
API and no implicit "current terminal".

Design contract: ``docs/tool/NATIVE_SHELL_AND_INTERACTIVE_TERMINAL_PROPOSAL.md``
freezes the boundary this module implements — a session is an explicit admitted
ToolTask, no second resource API, and the ``sandbox_profile="workspace"`` claim
is made only when the isolation is actually applied to the process tree.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Annotated, Any, ClassVar, cast
from uuid import uuid4

from pydantic import Field

from pygent.core import active_infrastructure, current_infrastructure
from pygent.core._tool_values import ToolCall
from pygent.tool._waiting import resolve_wait_timeout
from pygent.tool.executors import (
    ToolExecutionContext,
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
    output_file: Any, process: Any, limit: int
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
    """One live interactive session owned by an admitted terminal ToolTask.

    The owning task's executor creates the session and its driver releases it;
    ``publish_output`` and ``published_bytes`` ride along because only the
    owning task's publish callback may write the task output snapshot.
    ``input_lock`` serializes the write-observe cycles of ``terminal_input``
    so one session carries one ordered input stream.
    """

    task_id: str
    identity: ShellIdentity
    process: Any
    output_file: Any
    cwd: str
    initial_command: str | None = None
    backend: str = "pipe"
    closed: bool = False
    final_output: str | None = None
    publish_output: Any = None
    published_bytes: int = 0
    input_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

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
            future = self.process.outcome_ready
        except RuntimeError:
            # Process was never started (task cancelled before start).
            return self.finalize(cleanup_complete=True)
        try:
            await asyncio.wait_for(
                asyncio.shield(future),
                _TERMINAL_WAIT_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            pass
        return self.finalize(cleanup_complete=True)


class TerminalSessionStore:
    """Deployment adapter holding one live session per admitted terminal task.

    A session's process, capture file and streaming bookkeeping are invocation
    state, so they never live on the shared assembly (Pygent principle 3):
    this store is the explicit deployment resource that owns them between
    admission and release. It is assembly-local and never enters ``ToolSpec``,
    ``Context`` or an ``ExecutionPlan`` (Tool principle 12).

    Scope: process-local, single event loop, non-durable. One store may be
    shared by several ``TerminalTools`` assemblies on the same loop; it never
    spans threads or processes, and a session is never recovered after a
    Runtime restart.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, TerminalSession] = {}

    def __contains__(self, task_id: object) -> bool:
        return task_id in self._sessions

    def create(self, session: TerminalSession) -> None:
        """Register one session; a task id admits at most one live session."""

        if session.task_id in self._sessions:
            raise ToolExecutionError(
                f"terminal session already exists: {session.task_id}",
                kind="executor_error",
                code="terminal_session_already_exists",
                side_effect_committed=False,
            )
        self._sessions[session.task_id] = session

    def get(self, task_id: str) -> TerminalSession | None:
        return self._sessions.get(task_id)

    def release(self, task_id: str) -> None:
        self._sessions.pop(task_id, None)

    def task_ids(self) -> tuple[str, ...]:
        return tuple(self._sessions)

    async def aclose(self) -> None:
        """Close and drop any session its driver could not release."""

        for session in tuple(self._sessions.values()):
            session.close(cleanup_complete=False)
        self._sessions.clear()


class TerminalTools(NativeShellTools):
    """Deployment-local interactive terminal adapter for one native shell.

    ``sandbox=True`` (PTY + Linux Landlock) is a workspace-write confinement,
    not a full sandbox: the child may read and execute the whole filesystem
    (shell binary, loader, libraries, ``/etc``) and keeps network, IPC, ``/dev``
    and ``/proc`` access; only writes outside the workspace are denied. The
    ``workspace`` sandbox profile is claimed only after the child confirms the
    ruleset was applied.
    """

    tool_name: ClassVar[str] = "terminal"
    shell_name: ClassVar[str] = "terminal"

    def __init__(
        self,
        *,
        workspace_root: str | Path,
        shell: str = "bash",
        shell_executable: str | None = None,
        backend: str = "pipe",
        sandbox: bool = False,
        restrict_to_workspace: bool = True,
        timeout: float = 0,
        task_manager: ToolTaskManager | None = None,
        session_store: TerminalSessionStore | None = None,
    ) -> None:
        # A session is an explicit ToolTask, so the foreground wait only decides
        # how long the starting call observes startup before returning the task
        # reference; it never bounds the session itself.
        if backend not in ("pipe", "pty", "conpty"):
            supported = ", ".join(sorted(("conpty", "pipe", "pty")))
            raise ValueError(
                f"unsupported terminal backend: {backend!r}; supported: {supported}"
            )
        if backend == "pty" and sys.platform == "win32":
            raise ValueError(
                "PTY backend is not supported on Windows; use backend='pipe' or 'conpty'"
            )
        if backend == "conpty" and sys.platform != "win32":
            raise ValueError(
                "ConPTY backend is only supported on Windows; use backend='pty' or 'pipe'"
            )
        if backend == "conpty":
            # EXPERIMENTAL: the ConPTY output pipe does not forward child
            # output in every environment yet (verified on Windows Server
            # 2022 builds).  Prefer backend="pipe" for reliable interactive
            # sessions on Windows until the ConPTY capture path is fixed.
            pass
        if sandbox and backend != "pty":
            raise ValueError(
                "sandbox=True requires backend='pty' (process-tree isolation is "
                "implemented only by the PTY backend)"
            )
        if sandbox and sys.platform == "win32":
            raise ValueError(
                "sandbox=True is not supported on Windows (Landlock requires Linux)"
            )
        self.backend = backend
        self.sandbox = sandbox
        self._sandbox_profile: str | None
        super().__init__(
            workspace_root=workspace_root,
            shell_identity=_resolve_terminal_identity(
                shell, shell_executable=shell_executable
            ),
            restrict_to_workspace=restrict_to_workspace,
            timeout=timeout,
            task_manager=task_manager,
        )
        if sandbox:
            # Phase 4 contract: never declare "workspace" unless the isolation
            # can actually be applied.  Probe the Landlock ruleset creation in
            # the parent; refuse instead of silently claiming a sandbox.
            from ._confinement import _probe_confine_supported

            if not _probe_confine_supported(str(self.workspace_root)):
                raise ValueError(
                    "sandbox=True requires Linux 5.13+ with Landlock support for "
                    "the workspace root"
                )
            self._sandbox_profile = "workspace"
        else:
            self._sandbox_profile = None
        self.session_store = (
            session_store if session_store is not None else TerminalSessionStore()
        )
        self._owns_session_store = session_store is None

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
            # This invocation is the admitted session owner: the session is
            # created here, exactly once per task, and the driver releases it.
            # The framework invariant guarantees an admitted context carries a
            # stable task_id.
            assert context.task_id is not None
            session = self._open_session(
                context.task_id,
                self._resolve_working_directory(working_directory),
                command,
            )
            return await self._drive_session(session, context)
        if context is not None:
            # The model path reached us with a sync lifecycle.  An interactive
            # session is inherently detached, so promoting it to an independent
            # task here would silently override the application's lifecycle
            # decision.  Refuse instead (Tool principle 11).
            raise ToolExecutionError(
                "terminal requires a detach lifecycle",
                kind="capability_error",
                code="terminal_requires_detach",
                side_effect_committed=False,
            )
        # The starting call validates the working directory before admission;
        # the admitted task re-resolves it when it creates the session.
        self._resolve_working_directory(working_directory)
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
        await self.task_manager.start(snapshot.task_id)
        # The session is created by the task's executor; wait until it is
        # registered, the task terminates (propagating a failed start), or
        # bounded observation expires, before returning the task reference.
        await self._await_session(snapshot.task_id)
        handle = ToolTaskHandle(self.task_manager, snapshot.task_id)
        if wait_timeout is not None and wait_timeout > 0:
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
        session = self.session_store.get(task_id)
        if session is None or session.closed:
            raise ToolExecutionError(
                f"unknown terminal task: {task_id}",
                kind="validation_error",
                code="unknown_terminal_task",
                side_effect_committed=False,
            )
        # ToolCallLayer may run several inputs to one terminal concurrently;
        # the per-session lock keeps one ordered input stream, so each call's
        # observation window covers exactly its own write.
        async with session.input_lock:
            if session.closed:
                # The session ended while this call waited on the input lock;
                # the write never happened, so the absent side effect is
                # determinable rather than unknown.
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
            await self._publish(session)
            # The session may end while this call observes it; the frozen final
            # projection is then the only safe output source (the capture file
            # is already closed by finalize).
            output = session.final_output if session.closed else session.tail()
            return {
                "task_id": task_id,
                "written": True,
                # Whether this call observed new bytes; the output below is a
                # stream window, never a per-command result (proposal §7.7).
                "new_output": session.process.captured != before,
                "state": "closed" if session.closed else "running",
                "backend": session.backend,
                "output": output,
                "observation": {
                    # Seconds since the transport last captured output bytes:
                    # output observation, not execution readiness (proposal
                    # §7.7). Foreground and stdin-wait state are not observable
                    # through pipe/pty today and stay null.
                    "output_quiet_seconds": round(
                        time.monotonic() - session.process.last_output_at, 3
                    ),
                    "foreground_active": None,
                    "waiting_for_input": None,
                },
            }

    @property
    def toolkit(self):
        from pygent.tool.functional import ToolKit

        kit = ToolKit(
            self.terminal,
            self.terminal_input,
            wait_timeouts={"terminal": self.timeout},
        )
        if self._sandbox_profile is not None:
            specs = tuple(
                replace(spec, sandbox_profile=self._sandbox_profile)
                if spec.tool_id == "standard.shell.terminal"
                else spec
                for spec in kit._specs
            )
            kit._specs = specs
        return kit

    async def aclose(self) -> None:
        self._closed = True
        if self._owns_task_manager and self._task_manager is not None:
            # Owned manager: full close cancels every admitted task and joins
            # it, so each driver's finally has released its session.
            await self._task_manager.close(cancel=True)
        else:
            # Externally-owned manager: TerminalTools must not silently destroy
            # a session whose ToolTask still owns the process.  Request
            # cancellation so every admitted task reaches a terminal state and
            # its driver releases the session; cancel joins each task.
            manager = self._active_manager()
            if manager is not None:
                for task_id in self.session_store.task_ids():
                    await manager.cancel(task_id)
        if self._owns_session_store:
            await self.session_store.aclose()

    def _new_process(self, output_file: Any) -> Any:
        """Return a process object compatible with the configured backend."""
        if self.backend == "pty":
            from ._pty import PtyProcess

            return PtyProcess(
                output_file,
                max_capture_bytes=_TERMINAL_CAPTURE_BYTES,
                task_prefix="pygent-terminal",
                workspace_root=(
                    str(self.workspace_root) if self.sandbox else None
                ),
            )
        if self.backend == "conpty":
            from ._conpty import ConPtyProcess

            return ConPtyProcess(
                output_file,
                max_capture_bytes=_TERMINAL_CAPTURE_BYTES,
                task_prefix="pygent-terminal",
            )
        return _process.ShellProcess(
            output_file,
            max_capture_bytes=_TERMINAL_CAPTURE_BYTES,
            task_prefix="pygent-terminal",
        )

    def _open_session(
        self, task_id: str, cwd: str, initial_command: str | None
    ) -> TerminalSession:
        # The capture file belongs to the session, not to this call, so it stays
        # open until the session is released.
        output_file = tempfile.TemporaryFile()  # noqa: SIM115
        process = self._new_process(output_file)
        session = TerminalSession(
            task_id=task_id,
            identity=self.shell_identity,
            process=process,
            output_file=output_file,
            cwd=cwd,
            initial_command=initial_command,
            backend=self.backend,
        )
        try:
            self.session_store.create(session)
        except BaseException:
            # Registration failed: this session never reaches the driver's
            # finally, so its locally allocated resources are rolled back here.
            session.close(cleanup_complete=True)
            raise
        return session

    def _active_manager(self) -> ToolTaskManager | None:
        if active_infrastructure() is not None:
            manager = current_infrastructure().resolve_tool_task_manager()
            if manager is not None:
                return cast(ToolTaskManager, manager)
        if self._closed:
            return self._task_manager
        return self.task_manager

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

    async def _await_session(self, task_id: str) -> None:
        """Wait for the admitted task to register its session or terminate.

        A task that reaches a terminal state before its session appears either
        failed to start — its ToolResult is then propagated as the starting
        error — or already finished outright.  Bounded observation keeps a
        task that is still starting on its handle.
        """

        loop = asyncio.get_running_loop()
        deadline = loop.time() + _TERMINAL_READ_TIMEOUT_SECONDS
        handle = ToolTaskHandle(self.task_manager, task_id)
        while loop.time() < deadline:
            if task_id in self.session_store:
                return
            result = await handle.wait(0)
            if result is not None:
                if result.status == "succeeded":
                    # The task finished (and released its session) outright;
                    # the caller's foreground wait reports its output.
                    return
                raise ToolExecutionError(
                    result.error or "terminal task did not start",
                    kind=result.error_kind or "executor_error",
                    code=result.error_code,
                    retryable=result.retryable,
                    side_effect_committed=result.side_effect_committed,
                    missing_capabilities=result.missing_capabilities,
                )
            await asyncio.sleep(0.02)

    async def _drive_session(
        self, session: TerminalSession, context: ToolExecutionContext
    ) -> str:
        """Drive one admitted session as its executor, from create to release.

        The session was created by this admitted invocation, so this driver is
        the session's only lifecycle owner: its finally stops the process,
        freezes the output and releases the session exactly once.
        """

        process = session.process
        session.publish_output = context.publish_output
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
            if self.sandbox and not getattr(process, "confinement_applied", True):
                # The workspace profile is claimed only when the child really
                # applied the confinement; a silently unconfined process must
                # not keep running under a declared sandbox (proposal §8).
                raise ToolExecutionError(
                    "workspace confinement could not be applied to the terminal process",
                    kind="capability_error",
                    code="sandbox_not_applied",
                    side_effect_committed=False,
                )
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
                await self._publish(session)
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
            await self._publish(session)
            self.session_store.release(session.task_id)
            final = session.close(cleanup_complete=cleanup_complete)
        if cancelled:
            raise asyncio.CancelledError
        return final

    async def _publish(self, session: TerminalSession) -> None:
        publish_output = session.publish_output
        if publish_output is None or session.process.captured == session.published_bytes:
            return
        await publish_output(session.tail())
        session.published_bytes = session.process.captured

    def _process_kwargs(self, cwd: str, output: Any = None) -> dict[str, Any]:
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


__all__ = ["TerminalSession", "TerminalSessionStore", "TerminalTools"]
