from __future__ import annotations

import asyncio
import importlib
import json
import locale
import os
import shlex
import shutil
import subprocess
import sys
import time
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


def test_bash_ut_decode_output_truncates_and_handles_common_encodings(monkeypatch):
    monkeypatch.setattr(locale, "getpreferredencoding", lambda _do_setlocale: "cp1252")
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


@pytest.mark.skipif(sys.platform != "win32", reason="Windows process-tree behavior")
@pytest.mark.asyncio
async def test_bash_cancellation_terminates_windows_descendants(tmp_path):
    tools = PythonCommandTools(workspace_root=tmp_path)
    child_started = tmp_path / "child-started.txt"
    child_late = tmp_path / "child-late.txt"
    child = (
        "import pathlib,time; "
        f"pathlib.Path({str(child_started)!r}).write_text('started'); "
        "time.sleep(1); "
        f"pathlib.Path({str(child_late)!r}).write_text('late')"
    )
    parent = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}]); "
        "time.sleep(30)"
    )
    invocation = asyncio.create_task(invoke_tool(tools.bash, {"command": parent}))
    for _ in range(200):
        if child_started.exists():
            break
        await asyncio.sleep(0.01)
    assert child_started.exists()

    invocation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(invocation, timeout=5)
    await asyncio.sleep(1.1)

    assert not child_late.exists()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows process-tree behavior")
@pytest.mark.asyncio
async def test_bash_timeout_terminates_windows_descendants(tmp_path):
    tools = PythonCommandTools(workspace_root=tmp_path)
    child_late = tmp_path / "timeout-child-late.txt"
    child = (
        "import pathlib,time; time.sleep(1); "
        f"pathlib.Path({str(child_late)!r}).write_text('late')"
    )
    parent = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}]); "
        "print('started', flush=True); time.sleep(30)"
    )

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(tools._run_process(parent, str(tmp_path)), 0.25)
    await asyncio.sleep(1.1)

    assert not child_late.exists()


@pytest.mark.asyncio
async def test_bash_ut_times_out_and_preserves_partial_terminal_output(tmp_path):
    async with PythonCommandTools(workspace_root=tmp_path, timeout=0.25) as tools:
        handle = await tools.bash(
            "import time; print('before',flush=True); time.sleep(1)"
        )
        assert handle.task_id
        snapshot = await tools.tool_task_get(handle.task_id)
        assert "before" in snapshot["output"]
        assert (await handle.result()).status == "succeeded"


@pytest.mark.asyncio
async def test_bash_timeout_bounds_output_held_by_exited_parents_child(tmp_path):
    tools = PythonCommandTools(workspace_root=tmp_path)
    child = "import time; time.sleep(4)"
    command = (
        "import subprocess,sys; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}]); "
        "print('parent-exiting', flush=True)"
    )
    started = time.monotonic()

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(tools._run_process(command, str(tmp_path)), 0.5)

    elapsed = time.monotonic() - started
    assert elapsed < 3.5, f"output EOF held the invocation open for {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_bash_cancellation_bounds_output_held_by_exited_parents_child(tmp_path):
    tools = PythonCommandTools(workspace_root=tmp_path)
    started_file = tmp_path / "parent-exiting.txt"
    command = (
        "import subprocess,sys,pathlib; "
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(4)']); "
        f"pathlib.Path({str(started_file)!r}).write_text('ready')"
    )
    task = asyncio.create_task(tools._run_process(command, str(tmp_path)))
    for _ in range(200):
        if started_file.exists():
            break
        await asyncio.sleep(0.01)
    assert started_file.exists()
    await asyncio.sleep(0.2)
    started = time.monotonic()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    elapsed = time.monotonic() - started
    assert elapsed < 3, f"cancellation waited for output EOF for {elapsed:.3f}s"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows taskkill behavior")
@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.asyncio
async def test_bash_bounds_stalled_taskkill_and_releases_transports(
    tmp_path, monkeypatch, cancel
):
    tools = PythonCommandTools(workspace_root=tmp_path)
    loop = asyncio.get_running_loop()
    spawn = loop.subprocess_exec
    transports = []
    terminator_started = asyncio.Event()

    async def spawn_with_stalled_taskkill(factory, executable, *args, **kwargs):
        is_terminator = executable == "taskkill"
        if is_terminator:
            executable = sys.executable
            args = ("-c", "import time; time.sleep(10)")
        transport, protocol = await spawn(factory, executable, *args, **kwargs)
        transports.append(transport)
        if is_terminator:
            terminator_started.set()
        return transport, protocol

    monkeypatch.setattr(loop, "subprocess_exec", spawn_with_stalled_taskkill)
    invocation = asyncio.create_task(
        tools._run_process(
            "import time; print('before', flush=True); time.sleep(10)", str(tmp_path)
        )
    )
    await asyncio.sleep(0.25)
    invocation.cancel()
    await asyncio.wait_for(terminator_started.wait(), 3)
    started = time.monotonic()
    if cancel:
        # Both cancellations arrive during cleanup; neither may abandon it.
        invocation.cancel()
        await asyncio.sleep(0.1)
        invocation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await invocation
    else:
        with pytest.raises(asyncio.CancelledError):
            await invocation
    assert time.monotonic() - started < 3
    assert all(transport.is_closing() for transport in transports)
    for _ in range(100):
        if all(transport.get_returncode() is not None for transport in transports):
            break
        await asyncio.sleep(0.01)
    assert all(transport.get_returncode() is not None for transport in transports)
    assert not any(
        task.get_name().startswith("pygent-bash-") and not task.done()
        for task in asyncio.all_tasks()
    )


@pytest.mark.asyncio
async def test_bash_drains_output_beyond_capture_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(bash_module, "_MAX_FULL_OUTPUT_BYTES", 1024)
    output = await PythonCommandTools(workspace_root=tmp_path).bash(
        command="import sys; sys.stdout.write('x' * 1000000)"
    )
    exit_code, terminal_output = _parse_result(output)
    assert exit_code == "0"
    assert terminal_output.startswith("x" * 1024 + "\n[")
    assert "captured output capped at 1024 bytes" in terminal_output


@pytest.mark.asyncio
async def test_bash_output_write_failure_terminates_process(tmp_path, monkeypatch):
    real_temporary_file = bash_module.tempfile.TemporaryFile
    files = []

    def failing_output_file():
        output = real_temporary_file()
        files.append(output)

        def fail_write(data):
            raise OSError("capture disk full")

        output.write = fail_write
        return output

    monkeypatch.setattr(bash_module.tempfile, "TemporaryFile", failing_output_file)
    loop = asyncio.get_running_loop()
    spawn = loop.subprocess_exec
    transports = []

    async def record_spawn(*args, **kwargs):
        transport, protocol = await spawn(*args, **kwargs)
        transports.append(transport)
        return transport, protocol

    monkeypatch.setattr(loop, "subprocess_exec", record_spawn)
    started = time.monotonic()
    with pytest.raises(OSError, match="capture disk full"):
        await PythonCommandTools(workspace_root=tmp_path)._run_process(
            "import time; print('before', flush=True); time.sleep(10)", str(tmp_path)
        )
    assert time.monotonic() - started < 3
    assert all(output.closed for output in files)
    assert all(transport.is_closing() for transport in transports)


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.asyncio
async def test_bash_owns_process_during_startup(tmp_path, monkeypatch, cancel):
    loop = asyncio.get_running_loop()
    spawn = loop.subprocess_exec
    transports = []
    started = asyncio.Event()

    async def delayed_spawn(factory, executable, *args, **kwargs):
        transport, protocol = await spawn(factory, executable, *args, **kwargs)
        transports.append(transport)
        if executable != "taskkill":
            started.set()
            await asyncio.sleep(4)
        return transport, protocol

    monkeypatch.setattr(loop, "subprocess_exec", delayed_spawn)
    task = asyncio.create_task(
        PythonCommandTools(workspace_root=tmp_path)._run_process(
            "import time; time.sleep(10)", str(tmp_path)
        )
    )
    await asyncio.wait_for(started.wait(), 3)
    begin = time.monotonic()
    try:
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(task, 0.25)
        assert time.monotonic() - begin < 3
        assert all(transport.is_closing() for transport in transports)
    finally:
        for transport in transports:
            transport.close()


@pytest.mark.parametrize("error_type", [OSError, TimeoutError])
@pytest.mark.asyncio
async def test_bash_pipe_read_error_is_not_reported_as_timeout(
    tmp_path, monkeypatch, error_type
):
    loop = asyncio.get_running_loop()
    spawn = loop.subprocess_exec

    async def spawn_with_read_error(factory, *args, **kwargs):
        transport, protocol = await spawn(factory, *args, **kwargs)
        if args[0] != "taskkill":
            loop.call_later(
                0.02, protocol.pipe_connection_lost, 1, error_type("pipe read failed")
            )
        return transport, protocol

    monkeypatch.setattr(loop, "subprocess_exec", spawn_with_read_error)
    with pytest.raises(error_type, match="pipe read failed"):
        await PythonCommandTools(workspace_root=tmp_path)._run_process(
            "import time; time.sleep(10)", str(tmp_path)
        )


@pytest.mark.parametrize("connection_delay", [0.3, 3.0])
@pytest.mark.asyncio
async def test_bash_cancellation_during_native_pipe_connection_is_bounded(
    tmp_path, monkeypatch, connection_delay
):
    loop = asyncio.get_running_loop()
    connect = loop.connect_read_pipe
    connecting = asyncio.Event()
    pipes = []

    async def slow_connect(*args, **kwargs):
        result = await connect(*args, **kwargs)
        pipes.append(result[0])
        connecting.set()
        await asyncio.sleep(connection_delay)
        return result

    monkeypatch.setattr(loop, "connect_read_pipe", slow_connect)
    task = asyncio.create_task(
        PythonCommandTools(workspace_root=tmp_path)._run_process(
            (
                "import subprocess,sys,time; "
                "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(4)']); "
                "time.sleep(10)"
            ),
            str(tmp_path),
        )
    )
    await asyncio.wait_for(connecting.wait(), 3)
    begin = time.monotonic()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert time.monotonic() - begin < 3
    # Even startup delivered after the cleanup deadline retains a closing owner.
    for _ in range(200):
        pending = [
            task
            for task in asyncio.all_tasks()
            if task.get_name().startswith("pygent-bash-") and not task.done()
        ]
        if all(pipe.is_closing() for pipe in pipes) and not pending:
            break
        await asyncio.sleep(0.01)
    assert all(pipe.is_closing() for pipe in pipes)
    assert not pending


@pytest.mark.asyncio
async def test_bash_ut_truncates_large_output_without_losing_exit_code(tmp_path):
    tools = PythonCommandTools(workspace_root=tmp_path)
    output = await succeeded(
        tools.bash,
        command=f"import sys; sys.stdout.write('x' * ({_MAX_OUTPUT_BYTES} + 10))",
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
async def test_bash_ut_background_returns_managed_handle(tmp_path):
    async with PythonCommandTools(workspace_root=tmp_path) as tools:
        handle = await tools.bash(
            command="import time; time.sleep(.2)", is_background=True
        )
        assert handle.task_id
        assert (await handle.result()).status == "succeeded"


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
    output = await succeeded(tools.bash, command=command)
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
    output = await succeeded(tools.bash, command="pwd", working_directory="nested")
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
    output = await succeeded(tools.bash, command=command)
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
    output = await succeeded(tools.bash, command=command)
    actual_exit_code, actual_output = _parse_result(output)
    assert actual_exit_code == str(expected[0])
    assert actual_output == expected[1]


def test_bash_publishes_explicit_02_external_policy(tmp_path):
    spec = ToolKit(BashTools(workspace_root=tmp_path).bash).specs[0]

    assert (spec.tool_id, spec.version) == ("standard.shell.bash", "3.1.0")
    assert spec.side_effect is ToolSideEffect.EXTERNAL
    assert spec.idempotency is IdempotencyPolicy.NOT_IDEMPOTENT
    assert spec.timeout is None
    assert spec.wait_timeout == 600
    assert "timeout" in spec.definition.parameters["properties"]
    assert spec.wait_timeout_parameter == "timeout"
    assert spec.resource_key == "shell"
    assert spec.sandbox_profile == "workspace"
    assert spec.required_permissions == ("shell:execute",)
