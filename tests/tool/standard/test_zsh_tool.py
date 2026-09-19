"""Native Zsh adapter: definition-time resolution and workspace confinement."""

from __future__ import annotations

import os

import pytest

from pygent import IdempotencyPolicy, ToolKit, ToolSideEffect
from pygent.tool import ToolExecutionError, ZshTools
from pygent.tool.standard._zsh import (
    _find_zsh_executable,
    _is_functional_zsh,
    _zsh_candidates,
)

from ._helpers import invoke_tool


def _real_zsh_executable() -> str:
    executable = (
        os.environ.get("PYGENT_TEST_ZSH")
        or os.environ.get("PYGENT_ZSH_PATH")
        or _find_zsh_executable()
    )
    if not executable or not _is_functional_zsh(executable):
        pytest.skip("functional zsh is not available")
    return executable


def test_zsh_ut_registers_zsh_tool_name_only(tmp_path):
    tools = ZshTools(workspace_root=tmp_path)
    definitions = ToolKit(tools.zsh).definitions

    assert [item.name for item in definitions] == ["zsh"]

    spec = ToolKit(tools.zsh).specs[0]
    assert spec.tool_id == "standard.shell.zsh"
    assert spec.version == "1.0.0"
    assert spec.side_effect is ToolSideEffect.EXTERNAL
    assert spec.idempotency is IdempotencyPolicy.NOT_IDEMPOTENT
    assert spec.resource_key == "shell"
    assert spec.sandbox_profile == "workspace"
    assert spec.required_permissions == ("shell:execute",)
    assert spec.wait_timeout == 600
    assert spec.timeout is None


def test_zsh_ut_keeps_explicit_executable_override(tmp_path):
    tools = ZshTools(workspace_root=tmp_path, zsh_executable="/bin/zsh")

    assert tools.zsh_executable == "/bin/zsh"
    assert tools.shell_identity.name == "zsh"
    assert tools.shell_identity.executable == "/bin/zsh"


def test_zsh_ut_honors_environment_override(monkeypatch):
    monkeypatch.setenv("PYGENT_ZSH_PATH", "env-zsh")

    assert _find_zsh_executable() == "env-zsh"


def test_zsh_ut_skips_nonfunctional_candidates(monkeypatch):
    from pygent.tool.standard import _zsh as zsh_module

    monkeypatch.delenv("PYGENT_ZSH_PATH", raising=False)
    monkeypatch.setattr(zsh_module, "_zsh_candidates", lambda: ["broken-zsh", "good-zsh"])
    monkeypatch.setattr(
        zsh_module, "_is_functional_zsh", lambda executable: executable == "good-zsh"
    )

    assert _find_zsh_executable() == "good-zsh"


def test_zsh_ut_candidates_include_portable_locations(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)

    candidates = _zsh_candidates()

    assert "/bin/zsh" in candidates
    assert "/usr/bin/zsh" in candidates


@pytest.mark.asyncio
async def test_zsh_ut_restricts_working_directory_to_workspace(tmp_path):
    outside = tmp_path.parent
    tools = ZshTools(workspace_root=tmp_path)

    result = await invoke_tool(
        tools.zsh, {"command": "pwd", "working_directory": str(outside)}
    )

    assert result.status == "failed"
    assert result.error_code == "path_outside_workspace"
    assert result.side_effect_committed is False


@pytest.mark.asyncio
async def test_zsh_ut_rejects_missing_working_directory(tmp_path):
    tools = ZshTools(workspace_root=tmp_path)

    with pytest.raises(ToolExecutionError) as raised:
        await tools.zsh(command="pwd", working_directory="missing")

    assert raised.value.code == "not_a_directory"


@pytest.mark.asyncio
async def test_zsh_ut_runs_in_requested_working_directory(tmp_path):
    executable = _real_zsh_executable()
    nested = tmp_path / "nested"
    nested.mkdir()
    tools = ZshTools(workspace_root=tmp_path, zsh_executable=executable)

    output = await tools.zsh(command="basename \"$PWD\"", working_directory="nested")

    assert isinstance(output, str)
    exit_code, terminal_output = output.split("output:\n", 1)
    assert exit_code.removeprefix("exit_code: ").strip() == "0"
    assert terminal_output.strip() == "nested"


@pytest.mark.asyncio
async def test_zsh_ut_does_not_share_state_between_calls(tmp_path):
    executable = _real_zsh_executable()
    tools = ZshTools(workspace_root=tmp_path, zsh_executable=executable)

    await tools.zsh(command="export PYGENT_STATE=set; cd ..")
    output = await tools.zsh(command="printf '[%s]%s' \"$PYGENT_STATE\" \"$PWD\"")

    assert isinstance(output, str)
    _, terminal_output = output.split("output:\n", 1)
    assert terminal_output.strip() == f"[]{tools.workspace_root}"


@pytest.mark.asyncio
async def test_zsh_ut_background_submission_returns_handle(tmp_path):
    executable = _real_zsh_executable()
    tools = ZshTools(workspace_root=tmp_path, zsh_executable=executable, timeout=0)

    async with tools:
        handle = await tools.zsh(command="sleep 30", is_background=True)

        assert handle.task_id
        await handle.cancel()
