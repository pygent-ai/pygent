"""Native PowerShell adapter: real process behavior and workspace confinement."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from pygent import IdempotencyPolicy, ToolKit, ToolSideEffect
from pygent.tool import PowerShellTools, ToolExecutionError
from pygent.tool.standard._powershell import (
    _find_powershell_executable,
    _is_functional_powershell,
)

from ._helpers import invoke_tool, succeeded


def _real_powershell_executable() -> str:
    executable = (
        os.environ.get("PYGENT_TEST_POWERSHELL")
        or os.environ.get("PYGENT_POWERSHELL_PATH")
        or _find_powershell_executable()
    )
    if not executable or not _is_functional_powershell(executable):
        pytest.skip("functional PowerShell is not available")
    return executable


def _parse_result(output: str) -> tuple[str, str]:
    header, terminal_output = output.split("output:\n", 1)
    return header.removeprefix("exit_code: ").strip(), terminal_output


def test_powershell_ut_registers_powershell_tool_name_only(tmp_path):
    tools = PowerShellTools(workspace_root=tmp_path)
    definitions = ToolKit(tools.powershell).definitions

    assert [item.name for item in definitions] == ["powershell"]

    spec = ToolKit(tools.powershell).specs[0]
    assert spec.tool_id == "standard.shell.powershell"
    assert spec.version == "1.0.0"
    assert spec.side_effect is ToolSideEffect.EXTERNAL
    assert spec.idempotency is IdempotencyPolicy.NOT_IDEMPOTENT
    assert spec.resource_key == "shell"
    assert spec.sandbox_profile == "workspace"
    assert spec.required_permissions == ("shell:execute",)
    assert spec.wait_timeout == 600
    assert spec.timeout is None


def test_powershell_ut_keeps_explicit_executable_override(tmp_path):
    tools = PowerShellTools(workspace_root=tmp_path, powershell_executable="pwsh")

    assert tools.powershell_executable == "pwsh"
    assert tools.shell_identity.executable == "pwsh"
    assert tools.shell_identity.name == "powershell"


def test_powershell_ut_honors_environment_override(monkeypatch):
    monkeypatch.setenv("PYGENT_POWERSHELL_PATH", "env-pwsh")

    assert _find_powershell_executable() == "env-pwsh"


def test_powershell_ut_skips_nonfunctional_candidates(monkeypatch):
    from pygent.tool.standard import _powershell as powershell_module

    monkeypatch.delenv("PYGENT_POWERSHELL_PATH", raising=False)
    monkeypatch.setattr(
        powershell_module,
        "_powershell_candidates",
        lambda: ["broken-pwsh", "working-pwsh"],
    )
    monkeypatch.setattr(
        powershell_module,
        "_is_functional_powershell",
        lambda executable: executable == "working-pwsh",
    )

    assert _find_powershell_executable() == "working-pwsh"


@pytest.mark.asyncio
async def test_powershell_ut_restricts_working_directory_to_workspace(tmp_path):
    outside = tmp_path.parent
    tools = PowerShellTools(workspace_root=tmp_path)

    result = await invoke_tool(
        tools.powershell, {"command": "Get-Location", "working_directory": str(outside)}
    )

    assert result.status == "failed"
    assert result.error_code == "path_outside_workspace"
    assert result.side_effect_committed is False


@pytest.mark.asyncio
async def test_powershell_ut_rejects_missing_working_directory(tmp_path):
    tools = PowerShellTools(workspace_root=tmp_path)

    with pytest.raises(ToolExecutionError) as raised:
        await tools.powershell(command="Get-Location", working_directory="missing")

    assert raised.value.code == "not_a_directory"
    assert raised.value.side_effect_committed is False


@pytest.mark.asyncio
async def test_powershell_ut_runs_in_requested_working_directory(tmp_path):
    executable = _real_powershell_executable()
    nested = tmp_path / "nested"
    nested.mkdir()
    tools = PowerShellTools(workspace_root=tmp_path, powershell_executable=executable)

    output = await succeeded(
        tools.powershell, command="Split-Path -Leaf (Get-Location)", working_directory="nested"
    )

    exit_code, terminal_output = _parse_result(output)
    assert exit_code == "0"
    assert terminal_output.strip() == "nested"


@pytest.mark.asyncio
async def test_powershell_ut_reports_native_exit_code(tmp_path):
    executable = _real_powershell_executable()
    tools = PowerShellTools(workspace_root=tmp_path, powershell_executable=executable)

    output = await succeeded(tools.powershell, command="exit 3")

    assert _parse_result(output)[0] == "3"


@pytest.mark.asyncio
async def test_powershell_ut_does_not_share_state_between_calls(tmp_path):
    executable = _real_powershell_executable()
    tools = PowerShellTools(workspace_root=tmp_path, powershell_executable=executable)

    await succeeded(tools.powershell, command="$env:PYGENT_STATE = 'set'; Set-Location ..")
    output = await succeeded(
        tools.powershell,
        command="Write-Output \"[$($env:PYGENT_STATE)]$(Get-Location)\"",
    )

    exit_code, terminal_output = _parse_result(output)
    assert exit_code == "0"
    assert terminal_output.strip() == f"[]{tools.workspace_root}"


@pytest.mark.asyncio
async def test_powershell_ut_decodes_utf16_cmdlet_output(tmp_path):
    executable = _real_powershell_executable()
    tools = PowerShellTools(workspace_root=tmp_path, powershell_executable=executable)

    output = await succeeded(tools.powershell, command="Write-Output '中文输出'")

    exit_code, terminal_output = _parse_result(output)
    assert exit_code == "0"
    assert "中文输出" in terminal_output


@pytest.mark.asyncio
async def test_powershell_ut_foreground_wait_expires_into_background_task(tmp_path):
    executable = _real_powershell_executable()
    tools = PowerShellTools(
        workspace_root=tmp_path, powershell_executable=executable, timeout=0.2
    )

    async with tools:
        handle = await tools.powershell(command="Start-Sleep -Seconds 5")

        assert handle.task_id
        snapshot = await handle.snapshot()
        assert snapshot.state.value in {"pending", "running"}
        await handle.cancel()


@pytest.mark.asyncio
async def test_powershell_ut_background_submission_returns_immediately(tmp_path):
    executable = _real_powershell_executable()
    tools = PowerShellTools(
        workspace_root=tmp_path, powershell_executable=executable, timeout=0
    )

    started = time.monotonic()
    async with tools:
        handle = await tools.powershell(command="Start-Sleep -Seconds 5", is_background=True)

        assert time.monotonic() - started < 3
        assert handle.task_id
        await handle.cancel()


@pytest.mark.asyncio
async def test_powershell_ut_rejects_missing_executable(tmp_path):
    tools = PowerShellTools(
        workspace_root=tmp_path, powershell_executable="missing-pwsh"
    )

    result = await invoke_tool(tools.powershell, {"command": "Write-Output ok"})

    assert result.status == "failed"
    assert result.error_code == "executable_not_found"
    assert result.side_effect_committed is False


@pytest.mark.asyncio
async def test_powershell_ut_caps_captured_output(tmp_path, monkeypatch):
    executable = _real_powershell_executable()
    monkeypatch.setattr(PowerShellTools, "_max_capture_bytes", lambda self: 1024)
    tools = PowerShellTools(workspace_root=tmp_path, powershell_executable=executable)

    output = await succeeded(
        tools.powershell, command="Write-Output ('x' * 4000)"
    )

    exit_code, terminal_output = _parse_result(output)
    assert exit_code == "0"
    assert "captured output capped at 1024 bytes" in terminal_output


@pytest.mark.asyncio
async def test_powershell_ut_cancellation_leaves_no_shell_task(tmp_path):
    executable = _real_powershell_executable()
    tools = PowerShellTools(workspace_root=tmp_path, powershell_executable=executable)

    task = asyncio.create_task(tools.powershell(command="Start-Sleep -Seconds 30"))
    await asyncio.sleep(0.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    for _ in range(200):
        if not any(
            item.get_name().startswith("pygent-powershell-") and not item.done()
            for item in asyncio.all_tasks()
        ):
            break
        await asyncio.sleep(0.01)
    assert not any(
        item.get_name().startswith("pygent-powershell-") and not item.done()
        for item in asyncio.all_tasks()
    )


@pytest.mark.asyncio
async def test_powershell_ut_shares_an_injected_task_manager(tmp_path):
    executable = _real_powershell_executable()
    owner = PowerShellTools(workspace_root=tmp_path, powershell_executable=executable)
    async with owner:
        shared = owner.task_manager
        assert shared is not None

        user = PowerShellTools(
            workspace_root=tmp_path,
            powershell_executable=executable,
            timeout=0,
            task_manager=shared,
        )
        handle = await user.powershell(command="Start-Sleep -Seconds 5")
        assert await shared.get_task(handle.task_id) is not None
        await handle.cancel()


def test_powershell_ut_is_only_assembled_on_request(tmp_path):
    """The standard toolkit keeps its existing model-visible tool set."""

    from pygent.tool.standard import StandardTools

    suite = StandardTools(workspace_root=tmp_path)
    names = [definition.name for definition in suite.toolkit.definitions]

    assert "powershell" not in names
    assert "bash" in names


def test_powershell_ut_platform_candidates_are_absolute_paths(monkeypatch):
    from pygent.tool.standard import _powershell as powershell_module

    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(powershell_module.sys, "platform", "win32")
    monkeypatch.setenv("SystemRoot", r"C:\Windows")

    candidates = powershell_module._powershell_candidates()

    assert all(Path(candidate).is_absolute() for candidate in candidates)
    assert any(candidate.endswith("powershell.exe") for candidate in candidates)


def test_powershell_ut_probe_reports_unavailable_shell():
    assert _is_functional_powershell("missing-pwsh") is False


def test_powershell_ut_real_executable_is_functional():
    executable = _real_powershell_executable()
    probe = subprocess.run(
        [executable, "-NoLogo", "-NoProfile", "-Command", "Write-Output ok"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert probe.returncode == 0
    assert b"ok" in probe.stdout
