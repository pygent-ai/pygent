from __future__ import annotations

import asyncio
import importlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from pygent import IdempotencyPolicy, ToolKit, ToolSideEffect
from pygent.tool import ToolExecutionError
from pygent.tool.standard._bash import (
    _MAX_OUTPUT_BYTES,
    BashTools,
    _decode_output,
    _find_bash_executable,
)

from ._helpers import invoke_tool, succeeded

bash_module = importlib.import_module("pygent.tool.standard._bash")


class PythonCommandTools(BashTools):
    """Keep process behavior while substituting Python for bash in unit tests."""

    def _command_args(self, command: str) -> list[str]:
        return [sys.executable, "-c", command]


def _parse_result(output: str) -> tuple[str, str]:
    header, terminal_output = output.split("output:\n", 1)
    return header.removeprefix("exit_code: ").strip(), terminal_output


def _full_output_path_from_result(terminal_output: str) -> Path:
    prefix = "[full output saved to: "
    for line in terminal_output.splitlines():
        if line.startswith(prefix) and line.endswith("]"):
            return Path(line[len(prefix) : -1])
    raise AssertionError("full output path notice was not found")


def _real_bash_executable() -> str:
    executable = (
        os.environ.get("PYGENT_TEST_BASH")
        or os.environ.get("PYGENT_BASH_PATH")
        or _find_bash_executable()
        or shutil.which("bash")
    )
    if not executable:
        pytest.skip("functional bash is not available")
    probe = subprocess.run(
        [executable, "-lc", "printf ok"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if probe.returncode != 0 or probe.stdout != b"ok":
        pytest.skip("functional bash is not available")
    return executable


def _run_real_bash(executable: str, command: str, cwd: str) -> tuple[int, str]:
    proc = subprocess.run(
        [executable, "-lc", command],
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return proc.returncode, _decode_output(proc.stdout)


def _prepare_bash_workspace(path: Path) -> None:
    (path / "a.txt").write_text("a", encoding="utf-8")
    (path / "b.txt").write_text("b", encoding="utf-8")
    (path / "sub").mkdir(exist_ok=True)
    (path / "sub" / "note.md").write_text("note", encoding="utf-8")


def _to_msys_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    rest = resolved.as_posix()[3:]
    return f"/{drive}/{rest}"


def _npm_available(executable: str) -> bool:
    proc = subprocess.run(
        [
            executable,
            "-lc",
            "command -v npm >/dev/null 2>&1 && npm --version >/dev/null",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=30,
    )
    return proc.returncode == 0


def _bash_python_command(executable: str) -> str:
    probe = "print('ok')"
    for candidate in ("python3", "python"):
        proc = subprocess.run(
            [
                executable,
                "-lc",
                (
                    f"command -v {candidate} >/dev/null 2>&1 && "
                    f"{candidate} -c {shlex.quote(probe)}"
                ),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=30,
        )
        if proc.returncode == 0 and b"ok" in proc.stdout:
            return candidate
    pytest.skip("python is not available from bash")


def _prepare_npm_workspace(path: Path) -> None:
    package = {
        "name": "npm-probe",
        "version": "1.2.3",
        "description": "local npm probe package",
        "main": "index.js",
        "scripts": {
            "echo": "node -e \"console.log('stdout-line'); console.error('stderr-line')\"",
            "args": "node -e \"console.log(process.argv.slice(1).join('|'))\"",
            "fail": "node -e \"console.error('boom-line'); process.exit(7)\"",
            "unicode": "node -e \"console.log('中文'); console.error('错误')\"",
        },
        "files": ["index.js", "README.md"],
        "license": "MIT",
    }
    (path / "package.json").write_text(
        json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (path / "index.js").write_text("module.exports = 'probe';\n", encoding="utf-8")
    (path / "README.md").write_text("# npm probe\n", encoding="utf-8")


def test_bash_ut_decode_output_truncates_and_handles_common_encodings():
    assert _decode_output(b"hello") == "hello"
    assert _decode_output("中文".encode("cp936")) == "中文"
    assert _decode_output("hello".encode("utf-16-le")) == "hello"
    assert len(_decode_output(b"x" * (_MAX_OUTPUT_BYTES + 10))) == _MAX_OUTPUT_BYTES


def test_bash_ut_decode_output_handles_mixed_utf16_prefix_and_utf8_tail():
    warning = ("wsl: 检测到 localhost 代理配置，但未镜像到 WSL。\r\n").encode(
        "utf-16-le"
    )
    bash_error = b"/bin/bash: line 1: cd: /e/Projects/lora: No such file or directory\n"

    output = _decode_output(warning + bash_error)

    assert "wsl: 检测到 localhost" in output
    assert "/bin/bash: line 1: cd: /e/Projects/lora" in output
    assert "戏温戏獡" not in output


def test_bash_ut_decode_output_truncated_utf8_boundary_keeps_valid_prefix():
    text = "中" * ((_MAX_OUTPUT_BYTES // 3) + 10)

    output = _decode_output(text.encode())

    assert output.startswith("中" * 100)
    assert "涓" not in output[:100]
    assert "�" in output


def test_bash_ut_registers_bash_tool_name_only(tmp_path):
    tools = BashTools(workspace_root=tmp_path)
    definitions = ToolKit(tools.bash).definitions

    assert [item.name for item in definitions] == ["bash"]
    assert definitions[0].name != "run_terminal_cmd"


def test_bash_ut_find_bash_honors_explicit_override(monkeypatch):
    monkeypatch.setenv("PYGENT_BASH_PATH", "custom-bash")
    assert _find_bash_executable() == "custom-bash"


def test_bash_ut_find_bash_skips_nonfunctional_path_bash(monkeypatch):
    monkeypatch.delenv("PYGENT_BASH_PATH", raising=False)
    monkeypatch.setattr(bash_module.sys, "platform", "win32")
    monkeypatch.setattr(bash_module.shutil, "which", lambda name: "wsl-bash")
    monkeypatch.setattr(
        bash_module, "_windows_bash_candidates", lambda: ["wsl-bash", "git-bash"]
    )
    monkeypatch.setattr(
        bash_module, "_is_functional_bash", lambda executable: executable == "git-bash"
    )
    assert _find_bash_executable() == "git-bash"


def test_bash_ut_functional_bash_on_windows_requires_msys_drive_mount(monkeypatch):
    calls: list[list[str]] = []

    class Proc:
        returncode = 1
        stdout = b""

    def fake_run(args, **kwargs):
        calls.append(args)
        return Proc()

    monkeypatch.setattr(bash_module.sys, "platform", "win32")
    monkeypatch.setenv("SystemDrive", "C:")
    monkeypatch.setattr(bash_module.subprocess, "run", fake_run)

    assert bash_module._is_functional_bash("bash") is False
    assert calls[0][2] == "test -d /c && printf ok"


def test_bash_ut_find_bash_prefers_git_bash_over_windows_system_wsl_bash(monkeypatch):
    monkeypatch.delenv("PYGENT_BASH_PATH", raising=False)
    monkeypatch.setattr(bash_module.sys, "platform", "win32")
    monkeypatch.setenv("SystemRoot", r"C:\Windows")
    monkeypatch.setattr(
        bash_module.shutil, "which", lambda name: r"C:\Windows\system32\bash.EXE"
    )
    monkeypatch.setattr(
        bash_module, "_windows_bash_candidates", lambda: [r"D:\Git\bin\bash.exe"]
    )
    monkeypatch.setattr(bash_module, "_is_functional_bash", lambda executable: True)
    assert _find_bash_executable() == r"D:\Git\bin\bash.exe"


@pytest.mark.asyncio
async def test_bash_ut_executes_in_requested_working_directory(tmp_path):
    tools = PythonCommandTools(workspace_root=tmp_path)
    nested = tmp_path / "nested"
    nested.mkdir()

    output = await succeeded(
        tools.bash,
        command="import pathlib; print(pathlib.Path.cwd().name)",
        working_directory="nested",
        timeout=5000,
    )

    exit_code, terminal_output = _parse_result(output)
    assert exit_code == "0"
    assert terminal_output == f"nested{os.linesep}"


@pytest.mark.asyncio
async def test_bash_ut_restricts_working_directory_to_workspace_by_default(tmp_path):
    outside = tmp_path.parent
    tools = PythonCommandTools(workspace_root=tmp_path)

    result = await invoke_tool(
        tools.bash, {"command": "print('hi')", "working_directory": str(outside)}
    )

    assert result.status == "failed"
    assert result.error_code == "path_outside_workspace"
    assert str(outside.resolve()) in (result.error or "")
    assert result.side_effect_committed is False


@pytest.mark.asyncio
async def test_bash_ut_can_disable_workspace_restriction(tmp_path):
    outside = tmp_path.parent
    tools = PythonCommandTools(workspace_root=tmp_path, restrict_to_workspace=False)

    output = await succeeded(
        tools.bash,
        command="import pathlib; print(pathlib.Path.cwd())",
        working_directory=str(outside),
        timeout=5000,
    )

    exit_code, terminal_output = _parse_result(output)
    assert exit_code == "0"
    assert terminal_output.strip() == str(outside.resolve())


@pytest.mark.asyncio
async def test_bash_ut_accepts_git_bash_msys_working_directory_on_windows(tmp_path):
    if os.name != "nt":
        pytest.skip("MSYS drive path compatibility is Windows-specific")
    tools = PythonCommandTools(workspace_root=tmp_path)
    nested = tmp_path / "nested"
    nested.mkdir()

    output = await succeeded(
        tools.bash,
        command="import pathlib; print(pathlib.Path.cwd().name)",
        working_directory=_to_msys_path(nested),
        timeout=5000,
    )
    assert _parse_result(output) == ("0", f"nested{os.linesep}")


@pytest.mark.asyncio
async def test_bash_ut_empty_command_matches_shell_success(tmp_path):
    output = await succeeded(
        PythonCommandTools(workspace_root=tmp_path).bash, command=""
    )
    assert _parse_result(output) == ("0", "")


@pytest.mark.asyncio
async def test_bash_ut_rejects_missing_working_directory(tmp_path):
    tools = PythonCommandTools(workspace_root=tmp_path)

    with pytest.raises(ToolExecutionError) as raised:
        await tools.bash(command="print('hi')", working_directory="missing")

    assert raised.value.code == "not_a_directory"
    assert raised.value.side_effect_committed is False


@pytest.mark.asyncio
async def test_bash_ut_missing_working_directory_is_structured_tool_error(tmp_path):
    tools = PythonCommandTools(workspace_root=tmp_path)

    result = await invoke_tool(
        tools.bash, {"command": "print('hi')", "working_directory": "missing"}
    )

    assert result.status == "failed"
    assert result.error_code == "not_a_directory"
    assert "working directory" in (result.error or "")
    assert str(tmp_path / "missing") in (result.error or "")
    assert result.side_effect_committed is False


@pytest.mark.asyncio
async def test_bash_cancellation_joins_process_before_returning(tmp_path):
    tools = PythonCommandTools(workspace_root=tmp_path)
    started = tmp_path / "started.txt"
    should_not_exist = tmp_path / "after-cancel.txt"
    command = (
        "import pathlib,time; "
        f"pathlib.Path({str(started)!r}).write_text('started'); "
        "time.sleep(1); "
        f"pathlib.Path({str(should_not_exist)!r}).write_text('late')"
    )
    invocation = asyncio.create_task(invoke_tool(tools.bash, {"command": command}))
    for _ in range(100):
        if started.exists():
            break
        await asyncio.sleep(0.01)
    assert started.exists()

    invocation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await invocation
    await asyncio.sleep(1.1)

    assert not should_not_exist.exists()


@pytest.mark.asyncio
async def test_bash_ut_times_out_and_preserves_partial_terminal_output(tmp_path):
    tools = PythonCommandTools(workspace_root=tmp_path)

    result = await invoke_tool(
        tools.bash,
        {
            "command": "import time; print('before', flush=True); time.sleep(1)",
            "timeout": 50,
        },
    )

    assert result.status == "unknown"
    assert result.error_kind == "timeout"
    assert result.error_code == "command_timeout"
    assert result.side_effect_committed is None
    assert f"before{os.linesep}" in (result.error or "")
    assert "timed out after" in (result.error or "")


@pytest.mark.asyncio
async def test_bash_ut_truncates_large_output_without_losing_exit_code(tmp_path):
    tools = PythonCommandTools(workspace_root=tmp_path)
    output = await succeeded(
        tools.bash,
        command=f"import sys; sys.stdout.write('x' * ({_MAX_OUTPUT_BYTES} + 10))",
        timeout=5000,
    )

    exit_code, terminal_output = _parse_result(output)
    assert exit_code == "0"
    assert terminal_output.startswith("x" * 100)
    assert "output truncated" in terminal_output
    full_output_path = _full_output_path_from_result(terminal_output)
    assert full_output_path.parent == tmp_path.resolve()
    assert full_output_path.read_bytes() == b"x" * (_MAX_OUTPUT_BYTES + 10)


@pytest.mark.asyncio
async def test_bash_ut_truncates_large_utf8_output_without_mojibake(tmp_path):
    tools = PythonCommandTools(workspace_root=tmp_path)
    output = await succeeded(
        tools.bash,
        command=(
            "import sys; "
            f"sys.stdout.buffer.write(('\\u4e2d' * (({_MAX_OUTPUT_BYTES} // 3) + 10)).encode('utf-8'))"
        ),
        timeout=5000,
    )

    exit_code, terminal_output = _parse_result(output)
    assert exit_code == "0"
    assert terminal_output.startswith("中" * 100)
    assert "涓" not in terminal_output[:100]
    assert "output truncated" in terminal_output
    full_output_path = _full_output_path_from_result(terminal_output)
    assert full_output_path.read_bytes().decode() == "中" * (
        (_MAX_OUTPUT_BYTES // 3) + 10
    )


@pytest.mark.asyncio
async def test_bash_ut_background_returns_pid(tmp_path):
    output = await succeeded(
        PythonCommandTools(workspace_root=tmp_path).bash,
        command="import time; time.sleep(0.2)",
        is_background=True,
    )
    assert "PID=" in output
    assert "output is not captured" in output


@pytest.mark.parametrize(
    "command",
    [
        "",
        "printf 'hello\\n'",
        "printf 'no-newline'",
        "printf 'line1\\nline2\\n'",
        "printf 'out'; printf 'err' >&2; printf 'done\\n'",
        "printf 'out1\\n'; printf 'err1\\n' >&2; printf 'out2\\n'; printf 'err2\\n' >&2",
        "printf '%s\\n' \"a b\" '$HOME' \"$HOME\"",
        "set -o pipefail; false | true",
        "set -e; echo before; false; echo after",
        "printf '%s\\n' *.txt",
        "printf '%s\\n' *.missing",
        "for i in 1 2 3; do echo item:$i; done",
        "FOO=bar; export FOO; printf '%s\\n' \"$FOO\"",
        'x=$(printf abc); echo "x=$x"',
        "printf $'\\u4e2d\\u6587\\n'",
        "node -e \"console.log('\\u4e2d\\u6587'); console.error('\\u9519\\u8bef')\"",
        "printf 'a\\0b'",
        "read -r value || true; printf '<%s>' \"$value\"",
        "cd sub && pwd && printf '%s\\n' *.md",
        "if then",
        "exit 42",
    ],
)
@pytest.mark.asyncio
async def test_bash_st_output_and_exit_code_match_real_bash(tmp_path, command):
    executable = _real_bash_executable()
    _prepare_bash_workspace(tmp_path)
    tools = BashTools(workspace_root=tmp_path, bash_executable=executable)

    expected = _run_real_bash(executable, command, str(tmp_path))
    output = await succeeded(tools.bash, command=command, timeout=5000)
    actual_exit_code, actual_output = _parse_result(output)

    assert actual_exit_code == str(expected[0])
    assert actual_output == expected[1]


@pytest.mark.asyncio
async def test_bash_st_working_directory_matches_real_bash(tmp_path):
    executable = _real_bash_executable()
    _prepare_bash_workspace(tmp_path)
    nested = tmp_path / "nested"
    nested.mkdir()
    tools = BashTools(workspace_root=tmp_path, bash_executable=executable)

    expected = _run_real_bash(executable, "pwd", str(nested))
    output = await succeeded(
        tools.bash, command="pwd", working_directory="nested", timeout=5000
    )
    actual_exit_code, actual_output = _parse_result(output)
    assert actual_exit_code == str(expected[0])
    assert actual_output == expected[1]


@pytest.mark.asyncio
async def test_bash_st_glob_and_nonzero_exit_match_real_bash(tmp_path):
    executable = _real_bash_executable()
    _prepare_bash_workspace(tmp_path)
    tools = BashTools(workspace_root=tmp_path, bash_executable=executable)
    command = "printf '%s\\n' *.txt; exit 7"

    expected = _run_real_bash(executable, command, str(tmp_path))
    output = await succeeded(tools.bash, command=command, timeout=5000)
    actual_exit_code, actual_output = _parse_result(output)
    assert actual_exit_code == str(expected[0])
    assert actual_output == expected[1]


@pytest.mark.asyncio
async def test_bash_st_large_utf8_output_truncates_without_mojibake(tmp_path):
    executable = _real_bash_executable()
    python_cmd = _bash_python_command(executable)
    tools = BashTools(workspace_root=tmp_path, bash_executable=executable)
    script = (
        "import sys; "
        f"sys.stdout.buffer.write(('\\u4e2d' * (({_MAX_OUTPUT_BYTES} // 3) + 10)).encode('utf-8'))"
    )

    output = await succeeded(
        tools.bash,
        command=f"{python_cmd} -c {shlex.quote(script)}",
        timeout=10000,
    )
    actual_exit_code, actual_output = _parse_result(output)

    assert actual_exit_code == "0"
    assert actual_output.startswith("中" * 100)
    assert "涓" not in actual_output[:100]
    assert "output truncated" in actual_output
    full_output_path = _full_output_path_from_result(actual_output)
    assert full_output_path.read_bytes().decode() == "中" * (
        (_MAX_OUTPUT_BYTES // 3) + 10
    )


@pytest.mark.parametrize(
    "command",
    [
        "npm --version",
        "node --version",
        "npm config get registry",
        "npm prefix",
        "npm pkg get name",
        "npm pkg get version",
        "npm pkg get scripts.echo",
        "npm run echo --silent",
        "npm run echo",
        "npm run args --silent -- alpha \"two words\" '$HOME'",
        "npm run fail --silent",
        "npm run missing --silent",
        "npm run unicode --silent",
        "npm pack --dry-run --json",
    ],
)
@pytest.mark.asyncio
async def test_bash_st_npm_commands_match_real_bash(tmp_path, command):
    executable = _real_bash_executable()
    if not _npm_available(executable):
        pytest.skip("npm is not available from bash")
    _prepare_npm_workspace(tmp_path)
    tools = BashTools(workspace_root=tmp_path, bash_executable=executable)

    expected = _run_real_bash(executable, command, str(tmp_path))
    output = await succeeded(tools.bash, command=command, timeout=30000)
    actual_exit_code, actual_output = _parse_result(output)
    assert actual_exit_code == str(expected[0])
    assert actual_output == expected[1]


def test_bash_publishes_explicit_02_external_policy(tmp_path):
    spec = ToolKit(BashTools(workspace_root=tmp_path).bash).specs[0]

    assert (spec.tool_id, spec.version) == ("standard.shell.bash", "2.0.0")
    assert spec.side_effect is ToolSideEffect.EXTERNAL
    assert spec.idempotency is IdempotencyPolicy.NOT_IDEMPOTENT
    assert spec.timeout == 610
    assert spec.resource_key == "shell"
    assert spec.sandbox_profile == "workspace"
    assert spec.required_permissions == ("shell:execute",)
