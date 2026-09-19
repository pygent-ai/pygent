"""Interactive terminal sessions: explicit ToolTask ownership and bounded I/O."""

from __future__ import annotations

import asyncio
import os

import pytest

from pygent import IdempotencyPolicy, ToolKit, ToolSideEffect
from pygent.tool import TerminalTools, ToolExecutionError
from pygent.tool.standard._terminal import _interactive_command_args
from pygent.tool.standard._powershell import powershell_shell_identity

from ._helpers import invoke_tool


def _identity():
    identity = powershell_shell_identity()
    if not os.path.exists(identity.executable):
        pytest.skip("functional native shell is not available")
    return identity


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


def test_terminal_ut_never_claims_workspace_confinement(tmp_path):
    """A persistent shell is not confined beyond its initial directory."""

    spec = ToolKit(TerminalTools(workspace_root=tmp_path).terminal).specs[0]

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
        shell="powershell",
        shell_executable=_identity().executable,
    )

    async with suite:
        handle = await suite.terminal()
        session = suite.sessions[handle.task_id]

        first = await suite.terminal_input(handle.task_id, "$x = 41")
        assert first["state"] == "running"
        assert first["backend"] == "pipe"

        second = await suite.terminal_input(handle.task_id, "Write-Output ($x + 1)")
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
        shell="powershell",
        shell_executable=_identity().executable,
    )

    async with suite:
        handle = await suite.terminal()
        await handle.cancel()
        for _ in range(100):
            if suite.sessions.get(handle.task_id) is None:
                break
            await asyncio.sleep(0.05)
        assert suite.sessions.get(handle.task_id) is None

        with pytest.raises(ToolExecutionError) as raised:
            await suite.terminal_input(handle.task_id, "echo hi")

        assert raised.value.code == "unknown_terminal_task"


@pytest.mark.asyncio
async def test_terminal_ut_publishes_output_snapshots(tmp_path):
    suite = TerminalTools(
        workspace_root=tmp_path,
        shell="powershell",
        shell_executable=_identity().executable,
    )

    async with suite:
        handle = await suite.terminal()
        await suite.terminal_input(handle.task_id, "Write-Output 'snapshot-value'")

        output = await suite.task_manager.get_output(handle.task_id)

        assert "snapshot-value" in str(output)

        await handle.cancel()


@pytest.mark.asyncio
async def test_terminal_ut_model_path_returns_detached_task(tmp_path):
    suite = TerminalTools(
        workspace_root=tmp_path,
        shell="powershell",
        shell_executable=_identity().executable,
    )

    async with suite:
        result = await invoke_tool(
            suite.terminal, {"working_directory": "."}, call_id="terminal-call"
        )

        assert result.status in {"succeeded", "detached"}
        task_id = (
            result.task.task_id if result.task is not None else None
        ) or _task_id_from_output(result.output)
        assert task_id is not None, result
        await suite.task_manager.cancel(task_id)


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
        shell="powershell",
        shell_executable=_identity().executable,
    )

    async with suite:
        handle = await suite.terminal()
        assert handle.task_id in suite.sessions

        await asyncio.sleep(0.5)
        snapshot = await handle.snapshot()
        assert snapshot.state.value == "running"

        assert await suite.task_manager.cancel(handle.task_id) is True
        for _ in range(100):
            if handle.task_id not in suite.sessions:
                break
            await asyncio.sleep(0.05)
        assert handle.task_id not in suite.sessions


@pytest.mark.asyncio
async def test_terminal_ut_close_releases_sessions(tmp_path):
    suite = TerminalTools(
        workspace_root=tmp_path,
        shell="powershell",
        shell_executable=_identity().executable,
    )
    handle = await suite.terminal()
    session = suite.sessions[handle.task_id]

    await suite.aclose()

    assert session.closed
    assert suite.sessions == {}
    assert _parse_session(await session.wait())[0] is not None
