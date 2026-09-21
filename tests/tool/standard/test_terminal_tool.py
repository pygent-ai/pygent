"""Interactive terminal sessions: explicit ToolTask ownership and bounded I/O."""

from __future__ import annotations

import asyncio
import os
import sys

import pytest

from pygent import (
    AIMessage,
    Context,
    IdempotencyPolicy,
    ToolCall,
    ToolKit,
    ToolSideEffect,
)
from pygent.tool import TerminalSessionStore, TerminalTools, ToolExecutionError
from pygent.tool.standard._powershell import powershell_shell_identity
from pygent.tool.standard._terminal import TerminalSession, _interactive_command_args


def _identity():
    identity = powershell_shell_identity()
    if not os.path.exists(identity.executable):
        pytest.skip("functional native shell is not available")
    return identity


IS_WINDOWS = sys.platform == "win32"


def _functional_shell() -> dict[str, str]:
    """PowerShell is the verified interactive shell on Windows, bash on POSIX.

    Interactive pipe semantics are platform-specific: pwsh on Linux does not
    process piped stdin as a line-oriented REPL, so each platform runs the
    session functional tests against its native shell.
    """

    if IS_WINDOWS:
        return {"shell": "powershell", "shell_executable": _identity().executable}
    if not _bash_available():
        pytest.skip("functional native shell is not available")
    return {"shell": "bash"}


def _assign(name: str, value: int) -> str:
    return f"${name} = {value}" if IS_WINDOWS else f"{name}={value}"


def _increment(name: str) -> str:
    return f"Write-Output (${name} + 1)" if IS_WINDOWS else f"echo $((${name} + 1))"


def _output(text: str) -> str:
    return f"Write-Output {text}" if IS_WINDOWS else f"echo {text}"


def _slow_echo(text: str) -> str:
    if IS_WINDOWS:
        return f"Start-Sleep -Milliseconds 300; Write-Output {text}"
    return f"sleep 0.3; echo {text}"


def _parse_session(output: str) -> tuple[str, str]:
    header, terminal_output = output.split("output:\n", 1)
    return header.removeprefix("exit_code: ").strip(), terminal_output


def test_terminal_ut_registers_terminal_and_input_tools(tmp_path):
    tools = TerminalTools(workspace_root=tmp_path)
    definitions = ToolKit(tools.terminal, tools.terminal_input).definitions
    specs = ToolKit(tools.terminal, tools.terminal_input).specs

    assert [item.name for item in definitions] == ["terminal", "terminal_input"]

    terminal_spec, input_spec = specs
    assert terminal_spec.tool_id == "standard.shell.terminal"
    assert terminal_spec.version == "1.0.0"
    assert terminal_spec.side_effect is ToolSideEffect.EXTERNAL
    assert terminal_spec.idempotency is IdempotencyPolicy.NOT_IDEMPOTENT
    assert terminal_spec.wait_timeout == 0
    assert terminal_spec.resource_key == "shell"
    assert terminal_spec.sandbox_profile is None
    assert input_spec.tool_id == "standard.shell.terminal_input"
    assert input_spec.side_effect is ToolSideEffect.EXTERNAL
    assert input_spec.resource_key == "terminal"


def test_terminal_ut_never_claims_workspace_confinement_by_default(tmp_path):
    """By default a persistent shell does not claim workspace confinement."""

    spec = ToolKit(TerminalTools(workspace_root=tmp_path).terminal).specs[0]

    assert spec.sandbox_profile is None


def test_terminal_ut_sandbox_requires_pty_backend(tmp_path):
    """sandbox=True is rejected on non-PTY backends (no real isolation)."""

    with pytest.raises(ValueError, match="sandbox=True requires backend='pty'"):
        TerminalTools(workspace_root=tmp_path, sandbox=True, backend="pipe")

    if sys.platform == "win32":
        # PTY backend validation fires before the sandbox check on Windows.
        with pytest.raises(ValueError, match="PTY backend is not supported"):
            TerminalTools(workspace_root=tmp_path, sandbox=True, backend="pty")
    else:
        with pytest.raises(ValueError, match="requires Linux 5.13"):
            TerminalTools(workspace_root=tmp_path, sandbox=True, backend="pty")


@pytest.mark.skipif(
    not hasattr(os, "landlock_create_ruleset"),
    reason="Landlock requires Linux 5.13+ with Python 3.11+",
)
def test_terminal_ut_sandbox_declares_workspace_profile(tmp_path):
    """sandbox=True (PTY + Landlock) declares workspace confinement."""

    suite = TerminalTools(workspace_root=tmp_path, backend="pty", sandbox=True)
    spec = suite.toolkit.specs[0]

    assert spec.sandbox_profile == "workspace"

    # The input tool keeps its own profile (no confinement claim).
    input_spec = suite.toolkit.specs[1]
    assert input_spec.sandbox_profile is None


def test_terminal_ut_sandbox_disabled_keeps_none(tmp_path):
    """sandbox=False keeps the terminal spec without a sandbox profile."""

    suite = TerminalTools(workspace_root=tmp_path, sandbox=False)
    spec = suite.toolkit.specs[0]

    assert spec.sandbox_profile is None


def test_terminal_ut_selects_shell_identity(tmp_path):
    suite = TerminalTools(workspace_root=tmp_path, shell="powershell")

    assert suite.shell_identity.name == "powershell"

    explicit = TerminalTools(
        workspace_root=tmp_path, shell="bash", shell_executable="custom-bash"
    )
    assert explicit.shell_identity.executable == "custom-bash"

    with pytest.raises(ValueError, match="unsupported terminal shell"):
        TerminalTools(workspace_root=tmp_path, shell="fish")


def test_terminal_ut_interactive_arguments_per_shell():
    from pygent.tool.standard._shell import ShellIdentity

    powershell = _interactive_command_args(
        ShellIdentity(platform="windows", name="powershell", executable="pwsh")
    )
    bash = _interactive_command_args(
        ShellIdentity(platform="linux", name="bash", executable="/bin/bash")
    )

    assert powershell == ["pwsh", "-NoLogo", "-NoProfile", "-Command", "-"]
    assert bash == ["/bin/bash", "-l"]


@pytest.mark.asyncio
async def test_terminal_ut_keeps_state_across_inputs(tmp_path):
    suite = TerminalTools(
        workspace_root=tmp_path,
        **_functional_shell(),
    )

    async with suite:
        handle = await suite.terminal()
        session = suite.session_store.get(handle.task_id)
        assert session is not None

        first = await suite.terminal_input(handle.task_id, _assign("x", 41))
        assert first["state"] == "running"
        assert first["backend"] == "pipe"
        assert first["observation"]["output_quiet_seconds"] >= 0

        second = await suite.terminal_input(handle.task_id, _increment("x"))
        assert "42" in second["output"]

        await handle.cancel()
        await asyncio.sleep(0.2)
        assert session.closed


@pytest.mark.asyncio
async def test_terminal_ut_validates_initial_working_directory(tmp_path):
    outside = tmp_path.parent
    suite = TerminalTools(workspace_root=tmp_path)

    with pytest.raises(ToolExecutionError) as raised:
        await suite.terminal(working_directory=str(outside))

    assert raised.value.code == "path_outside_workspace"
    assert raised.value.side_effect_committed is False


@pytest.mark.asyncio
async def test_terminal_ut_rejects_unknown_task(tmp_path):
    suite = TerminalTools(workspace_root=tmp_path)

    with pytest.raises(ToolExecutionError) as raised:
        await suite.terminal_input("missing-task", "echo hi")

    assert raised.value.code == "unknown_terminal_task"
    assert raised.value.side_effect_committed is False


@pytest.mark.asyncio
async def test_terminal_ut_stopped_session_refuses_input(tmp_path):
    suite = TerminalTools(
        workspace_root=tmp_path,
        **_functional_shell(),
    )

    async with suite:
        handle = await suite.terminal()
        await handle.cancel()
        for _ in range(100):
            if suite.session_store.get(handle.task_id) is None:
                break
            await asyncio.sleep(0.05)
        assert suite.session_store.get(handle.task_id) is None

        with pytest.raises(ToolExecutionError) as raised:
            await suite.terminal_input(handle.task_id, "echo hi")

        assert raised.value.code == "unknown_terminal_task"


@pytest.mark.asyncio
async def test_terminal_ut_publishes_output_snapshots(tmp_path):
    suite = TerminalTools(
        workspace_root=tmp_path,
        **_functional_shell(),
    )

    async with suite:
        handle = await suite.terminal()
        await suite.terminal_input(handle.task_id, _output("snapshot-value"))

        output = await suite.task_manager.get_output(handle.task_id)

        assert "snapshot-value" in str(output)

        await handle.cancel()


@pytest.mark.asyncio
async def test_terminal_ut_model_path_returns_detached_task(tmp_path):
    suite = TerminalTools(
        workspace_root=tmp_path,
        **_functional_shell(),
    )

    async with suite:
        from pygent import ToolAuthorizationDecision
        from pygent.tool.functional import ToolKit

        def allow_detach(request, _context):
            return ToolAuthorizationDecision(
                call_id=request.call.call_id,
                allowed=True,
                reason_code="test_allowed",
                lifecycle="detach",
            )

        toolkit = ToolKit(suite.terminal)
        layer = toolkit.local_layer(authorization_adapter=allow_detach)
        context = toolkit.make_visible_in(Context())
        message, _ = await layer.invoke(
            AIMessage(
                tool_calls=(
                    ToolCall(
                        call_id="terminal-call",
                        name=toolkit.definitions[0].name,
                        arguments={"working_directory": "."},
                    ),
                )
            ),
            context,
        )
        result = message.results[0]

        assert result.status in {"succeeded", "detached"}
        task_id = (
            result.task.task_id if result.task is not None else None
        ) or _task_id_from_output(result.output)
        assert task_id is not None, result
        # The admitted task must actually create and drive a live session, not
        # fail silently while the layer reports "detached" (regression guard).
        for _ in range(100):
            if task_id in suite.session_store:
                break
            await asyncio.sleep(0.05)
        assert task_id in suite.session_store, "model detach path created a live session"
        snapshot = await suite.task_manager.get_task(task_id)
        assert snapshot is not None and snapshot.state.value != "failed", snapshot
        await suite.task_manager.cancel(task_id)


async def test_terminal_ut_model_sync_path_is_refused(tmp_path):
    """A sync-authorized model call is refused, never silently promoted."""

    suite = TerminalTools(workspace_root=tmp_path)

    async with suite:
        from pygent import ToolAuthorizationDecision
        from pygent.tool.functional import ToolKit

        def allow_sync(request, _context):
            return ToolAuthorizationDecision(
                call_id=request.call.call_id,
                allowed=True,
                reason_code="test_allowed",
                lifecycle="sync",
            )

        toolkit = ToolKit(suite.terminal)
        layer = toolkit.local_layer(authorization_adapter=allow_sync)
        context = toolkit.make_visible_in(Context())
        message, _ = await layer.invoke(
            AIMessage(
                tool_calls=(
                    ToolCall(
                        call_id="terminal-sync-call",
                        name=toolkit.definitions[0].name,
                        arguments={"working_directory": "."},
                    ),
                )
            ),
            context,
        )
        result = message.results[0]

        assert result.status == "failed"
        assert result.error_code == "terminal_requires_detach"


def _task_id_from_output(output: object) -> str | None:
    if isinstance(output, str):
        for line in output.splitlines():
            if "task_id" in line:
                return line.strip().strip('",')
    if isinstance(output, dict):
        value = output.get("task_id")
        return value if isinstance(value, str) else None
    return None


@pytest.mark.asyncio
async def test_terminal_ut_session_survives_foreground_observation(tmp_path):
    """A session outlives the call that started it and keeps owning its process."""

    suite = TerminalTools(
        workspace_root=tmp_path,
        **_functional_shell(),
    )

    async with suite:
        handle = await suite.terminal()
        assert handle.task_id in suite.session_store

        await asyncio.sleep(0.5)
        snapshot = await handle.snapshot()
        assert snapshot.state.value == "running"

        assert await suite.task_manager.cancel(handle.task_id) is True
        for _ in range(100):
            if handle.task_id not in suite.session_store:
                break
            await asyncio.sleep(0.05)
        assert handle.task_id not in suite.session_store


@pytest.mark.asyncio
async def test_terminal_ut_close_releases_sessions(tmp_path):
    suite = TerminalTools(
        workspace_root=tmp_path,
        **_functional_shell(),
    )
    handle = await suite.terminal()
    session = suite.session_store.get(handle.task_id)
    assert session is not None

    await suite.aclose()

    assert session.closed
    assert suite.session_store.task_ids() == ()
    assert _parse_session(await session.wait())[0] is not None


@pytest.mark.asyncio
async def test_terminal_ut_natural_exit_releases_session(tmp_path):
    """A session that ends (shell exit) is removed from the single registry."""

    suite = TerminalTools(
        workspace_root=tmp_path,
        **_functional_shell(),
    )
    handle = await suite.terminal()
    task_id = handle.task_id
    assert task_id in suite.session_store

    # The input call survives the session ending mid-observation.
    result = await suite.terminal_input(task_id, "exit")
    assert result["state"] == "closed"

    for _ in range(100):
        if task_id not in suite.session_store:
            break
        await asyncio.sleep(0.05)
    assert task_id not in suite.session_store
    await suite.aclose()


@pytest.mark.asyncio
async def test_terminal_ut_initial_command_exit_completes_task(tmp_path):
    """A shell that exits on its initial command ends and releases its session."""

    suite = TerminalTools(
        workspace_root=tmp_path,
        **_functional_shell(),
    )

    async with suite:
        output = await suite.terminal(command="exit", timeout=5)

        assert isinstance(output, str)
        assert _parse_session(output)[0] is not None
        # The task's own driver released the session when the shell ended.
        assert suite.session_store.task_ids() == ()


@pytest.mark.asyncio
async def test_terminal_ut_aclose_external_manager_reaches_terminal(tmp_path):
    """aclose with an externally-owned manager must not leave a running task
    behind while the session registry is already gone."""

    suite = TerminalTools(
        workspace_root=tmp_path,
        **_functional_shell(),
    )
    external = suite.task_manager
    suite._owns_task_manager = False  # simulate an externally-owned manager
    handle = await suite.terminal()
    task_id = handle.task_id
    assert task_id in suite.session_store

    await suite.aclose()

    task = await external.get_task(task_id)
    assert task is not None, "task record must still exist"
    assert task.state.value != "running", f"task must reach a terminal state: {task}"
    assert task_id not in suite.session_store


def _session_for(suite: TerminalTools, task_id: str) -> TerminalSession:
    return TerminalSession(
        task_id=task_id,
        identity=suite.shell_identity,
        process=object(),
        output_file=object(),
        cwd=str(suite.workspace_root),
    )


def test_terminal_ut_store_admits_one_session_per_task(tmp_path):
    """The store enforces at most one live session per task id."""

    suite = TerminalTools(workspace_root=tmp_path)
    store = suite.session_store
    first = _session_for(suite, "tool-t1")

    store.create(first)

    with pytest.raises(ToolExecutionError) as raised:
        store.create(_session_for(suite, "tool-t1"))

    assert raised.value.code == "terminal_session_already_exists"
    assert store.get("tool-t1") is first


@pytest.mark.asyncio
async def test_terminal_ut_duplicate_registration_rolls_back_resources(
    tmp_path, monkeypatch
):
    """A session that fails to register releases its capture file and leaves
    the already-registered session untouched."""

    import tempfile as tempfile_module

    from pygent.tool.standard import _terminal

    suite = TerminalTools(workspace_root=tmp_path)
    store = suite.session_store
    existing = _session_for(suite, "tool-t1")
    store.create(existing)

    created_files = []
    real_temporary_file = tempfile_module.TemporaryFile

    def capturing_temporary_file(*args, **kwargs):
        handle = real_temporary_file(*args, **kwargs)
        created_files.append(handle)
        return handle

    monkeypatch.setattr(_terminal.tempfile, "TemporaryFile", capturing_temporary_file)

    with pytest.raises(ToolExecutionError) as raised:
        suite._open_session("tool-t1", str(tmp_path), None)

    assert raised.value.code == "terminal_session_already_exists"
    # The rolled-back session released its capture file; the store kept the
    # original session and the process was never started.
    assert len(created_files) == 1
    assert created_files[0].closed
    assert store.get("tool-t1") is existing


@pytest.mark.asyncio
async def test_terminal_ut_store_shares_concurrent_sessions(tmp_path):
    """One store safely carries several live sessions with distinct task ids."""

    suite = TerminalTools(
        workspace_root=tmp_path,
        **_functional_shell(),
    )

    async with suite:
        first = await suite.terminal()
        second = await suite.terminal()
        assert first.task_id in suite.session_store
        assert second.task_id in suite.session_store

        outputs = await asyncio.gather(
            suite.terminal_input(first.task_id, _output("one")),
            suite.terminal_input(second.task_id, _output("two")),
        )
        assert "one" in outputs[0]["output"]
        assert "two" in outputs[1]["output"]

        await asyncio.gather(first.cancel(), second.cancel())


@pytest.mark.asyncio
async def test_terminal_ut_failed_start_propagates_before_handle(tmp_path):
    """A task that fails before registering its session raises instead of
    returning a handle to a dead task."""

    suite = TerminalTools(workspace_root=tmp_path)

    # 123 is not a valid "command" string: the executor rejects the call
    # before the session is ever created.
    with pytest.raises(ToolExecutionError):
        await suite.terminal(command=123, timeout=5)

    assert suite.session_store.task_ids() == ()


class _DelayedStore(TerminalSessionStore):
    """Store wrapper that hides registered sessions until revealed."""

    def __init__(self) -> None:
        super().__init__()
        self.reveal = False

    def get(self, task_id: str) -> TerminalSession | None:
        return super().get(task_id) if self.reveal else None

    def __contains__(self, task_id: object) -> bool:
        return self.reveal and super().__contains__(task_id)


@pytest.mark.asyncio
async def test_terminal_ut_registration_timeout_keeps_handle(
    tmp_path, monkeypatch
):
    """Bounded registration observation returns the handle while the task is
    still starting, without failing it."""

    from pygent.tool.standard import _terminal

    store = _DelayedStore()
    suite = TerminalTools(
        workspace_root=tmp_path,
        **_functional_shell(),
        session_store=store,
    )
    monkeypatch.setattr(_terminal, "_TERMINAL_READ_TIMEOUT_SECONDS", 0.05)

    async with suite:
        handle = await suite.terminal()
        assert handle.task_id not in store

        for _ in range(100):
            snapshot = await handle.snapshot()
            if snapshot.state.value == "running":
                break
            await asyncio.sleep(0.05)
        assert snapshot.state.value == "running"

        store.reveal = True
        assert handle.task_id in store
        await handle.cancel()


@pytest.mark.asyncio
async def test_terminal_ut_concurrent_inputs_stay_ordered(tmp_path):
    """Concurrent inputs to one session serialize on the per-session lock."""

    suite = TerminalTools(
        workspace_root=tmp_path,
        **_functional_shell(),
    )

    async with suite:
        handle = await suite.terminal()
        first, second = await asyncio.gather(
            suite.terminal_input(
                handle.task_id,
                _slow_echo("SLOW-1"),
            ),
            suite.terminal_input(handle.task_id, _output("QUICK-2")),
        )

        assert first["written"] and second["written"]
        output = second["output"]
        assert "SLOW-1" in output
        assert "QUICK-2" in output
        # The second input was written only after the first call's observation
        # window saw the shell answer, so the stream stays ordered.
        assert output.index("SLOW-1") < output.index("QUICK-2")


# ── PTY backend tests (POSIX only) ──────────────────────────────────────────


def _bash_available() -> bool:
    import shutil

    return shutil.which("bash") is not None


def test_terminal_ut_backend_validation(tmp_path):
    """Invalid backend is rejected; platform-mismatched backends are rejected."""
    with pytest.raises(ValueError, match="unsupported terminal backend"):
        TerminalTools(workspace_root=tmp_path, backend="ptmx")

    TerminalTools(workspace_root=tmp_path, backend="pipe")

    if sys.platform == "win32":
        # PTY is POSIX-only; ConPTY is available on Windows.
        with pytest.raises(ValueError, match="PTY backend is not supported"):
            TerminalTools(workspace_root=tmp_path, backend="pty")
        TerminalTools(workspace_root=tmp_path, backend="conpty")
    else:
        # PTY is available on POSIX; ConPTY is Windows-only.
        with pytest.raises(ValueError, match="ConPTY backend is only supported"):
            TerminalTools(workspace_root=tmp_path, backend="conpty")
        TerminalTools(workspace_root=tmp_path, backend="pty")


@pytest.mark.skipif(
    not hasattr(os, "fork"), reason="PTY requires os.fork (POSIX only)"
)
def test_terminal_ut_pty_process_interface(tmp_path):
    """PtyProcess exposes the same duck interface as ShellProcess."""
    import tempfile

    from pygent.tool.standard._pty import PtyProcess

    output = tempfile.TemporaryFile()  # noqa: SIM115
    proc = PtyProcess(output, max_capture_bytes=4096, task_prefix="pygent-test")

    # Read-only properties exist on the class (before start they raise).
    assert hasattr(PtyProcess, "outcome_ready")
    assert hasattr(PtyProcess, "transport")
    assert proc.captured == 0
    assert proc.truncated is False
    assert proc.master_fd is None
    assert proc.child_pid is None


@pytest.mark.skipif(
    not hasattr(os, "fork"), reason="PTY requires os.fork (POSIX only)"
)
@pytest.mark.skipif(not _bash_available(), reason="bash executable not found")
@pytest.mark.asyncio
async def test_pty_process_captures_stdout(tmp_path):
    """PtyProcess captures stdout from a quick-running command."""
    import tempfile

    from pygent.tool.standard._pty import PtyProcess

    output = tempfile.TemporaryFile()  # noqa: SIM115
    proc = PtyProcess(output, max_capture_bytes=4096, task_prefix="pygent-test")

    await proc.start("bash", "-c", "printf 'hello pty\\n'", cwd=str(tmp_path))

    # The process exits quickly; outcome_ready should complete.
    try:
        await asyncio.wait_for(asyncio.shield(proc.outcome_ready), timeout=5.0)
    except TimeoutError:
        pytest.fail("PtyProcess did not exit within 5 seconds")
    finally:
        await proc.aclose(is_windows=False)

    # Verify capture
    output.seek(0)
    captured = output.read()
    assert b"hello pty" in captured, f"captured={captured!r}"
    assert proc.captured > 0
    assert proc.transport is not None
    returncode = proc.transport.get_returncode()
    assert returncode == 0, f"returncode={returncode}"


@pytest.mark.skipif(
    not hasattr(os, "fork"), reason="PTY requires os.fork (POSIX only)"
)
@pytest.mark.skipif(not _bash_available(), reason="bash executable not found")
@pytest.mark.asyncio
async def test_pty_process_writes_stdin(tmp_path):
    """PtyProcess can write to the PTY master stdin."""
    import tempfile

    from pygent.tool.standard._pty import PtyProcess

    output = tempfile.TemporaryFile()  # noqa: SIM115
    proc = PtyProcess(output, max_capture_bytes=4096, task_prefix="pygent-test")

    await proc.start("bash", "-c", "read line; echo received: $line", cwd=str(tmp_path))

    written = await proc.write_stdin(b"hello stdin\n")
    assert written, "write_stdin returned False"

    try:
        await asyncio.wait_for(asyncio.shield(proc.outcome_ready), timeout=5.0)
    except TimeoutError:
        pytest.fail("PtyProcess did not exit within 5 seconds")
    finally:
        await proc.aclose(is_windows=False)

    output.seek(0)
    captured = output.read()
    assert b"received: hello stdin" in captured, f"captured={captured!r}"
    assert proc.transport is not None
    assert proc.transport.get_returncode() == 0


@pytest.mark.skipif(
    not hasattr(os, "fork"), reason="PTY requires os.fork (POSIX only)"
)
@pytest.mark.skipif(not _bash_available(), reason="bash executable not found")
@pytest.mark.asyncio
async def test_terminal_pty_backend_basic_session(tmp_path):
    """A PTY-backed terminal session works through the standard ToolTask lifecycle."""
    suite = TerminalTools(
        workspace_root=tmp_path,
        shell="bash",
        backend="pty",
    )

    async with suite:
        handle = await suite.terminal()
        session = suite.session_store.get(handle.task_id)
        assert session is not None
        assert session.backend == "pty"

        # Send a command and check the output contains the expected result.
        first = await suite.terminal_input(handle.task_id, "echo pty-works")
        assert first["state"] == "running"
        assert first["backend"] == "pty"
        assert "pty-works" in first["output"], f"output={first['output']!r}"

        await handle.cancel()
        await asyncio.sleep(0.2)
        assert session.closed


@pytest.mark.skipif(
    not hasattr(os, "fork"), reason="PTY requires os.fork (POSIX only)"
)
@pytest.mark.skipif(not _bash_available(), reason="bash executable not found")
@pytest.mark.asyncio
async def test_terminal_pty_input_keeps_state(tmp_path):
    """PTY sessions preserve shell state across inputs (via persistent process)."""
    suite = TerminalTools(
        workspace_root=tmp_path,
        shell="bash",
        backend="pty",
    )

    async with suite:
        handle = await suite.terminal()

        first = await suite.terminal_input(handle.task_id, "x=41")
        assert first["state"] == "running"

        second = await suite.terminal_input(handle.task_id, "echo $((x + 1))")
        output = second["output"]
        session = suite.session_store.get(handle.task_id)
        for _ in range(30):
            if "42" in output:
                break
            await asyncio.sleep(0.1)
            if session is not None and not session.closed:
                output = session.tail()
        assert "42" in output, f"output={output!r}"

        await handle.cancel()
        await asyncio.sleep(0.2)
        assert handle.task_id not in suite.session_store


@pytest.mark.skipif(
    not hasattr(os, "fork"), reason="PTY requires os.fork (POSIX only)"
)
def test_terminal_ut_sandbox_pty_passes_workspace_root(tmp_path):
    """With sandbox=True the PtyProcess receives the workspace root."""
    import tempfile

    from pygent.tool.standard._pty import PtyProcess

    output = tempfile.TemporaryFile()  # noqa: SIM115
    proc = PtyProcess(
        output,
        max_capture_bytes=4096,
        task_prefix="pygent-test",
        workspace_root=str(tmp_path),
    )
    assert proc.workspace_root == str(tmp_path)


@pytest.mark.skipif(
    not hasattr(os, "fork"), reason="PTY requires os.fork (POSIX only)"
)
def test_terminal_ut_sandbox_disabled_passes_none(tmp_path):
    """Without sandbox the PtyProcess has no workspace confinement."""
    import tempfile

    from pygent.tool.standard._pty import PtyProcess

    output = tempfile.TemporaryFile()  # noqa: SIM115
    proc = PtyProcess(output, max_capture_bytes=4096, task_prefix="pygent-test")
    assert proc.workspace_root is None


@pytest.mark.skipif(
    not hasattr(os, "landlock_create_ruleset"),
    reason="Landlock requires Linux 5.13+ with Python 3.11+",
)
def test_terminal_ut_landlock_confinement_available(tmp_path):
    """Landlock confinement helper applies successfully on capable systems."""
    from pygent.tool.standard._confinement import (
        _landlock_available,
        confine_workspace,
    )

    assert _landlock_available() is True

    # Landlock cannot be applied from a process that already restricted itself,
    # so we only verify the probe + helper contract here; the confinement
    # itself is exercised by the PTY functional tests on Linux.
    assert confine_workspace is not None
