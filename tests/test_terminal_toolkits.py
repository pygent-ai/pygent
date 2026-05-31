import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import pygent.toolkits.bash as bash_module
from pygent.toolkits.bash import BashToolkits, _MAX_OUTPUT_BYTES, _decode_output, _find_bash_executable


class PythonCommandToolkits(BashToolkits):
    """Test double that keeps the process runner but swaps bash for Python."""

    def _bash_args(self, command: str) -> list[str]:
        return [sys.executable, "-c", command]


def _parse_result(output: str) -> tuple[str, str]:
    header, terminal_output = output.split("output:\n", 1)
    return header.removeprefix("exit_code: ").strip(), terminal_output


def _full_output_path_from_result(terminal_output: str) -> Path:
    prefix = "[full output saved to: "
    for line in terminal_output.splitlines():
        if line.startswith(prefix) and line.endswith("]"):
            return Path(line[len(prefix):-1])
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


def _prepare_bash_workspace(path):
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
        [executable, "-lc", "command -v npm >/dev/null 2>&1 && npm --version >/dev/null"],
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
                f"command -v {candidate} >/dev/null 2>&1 && {candidate} -c {shlex.quote(probe)}",
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


def _prepare_npm_workspace(path):
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
    (path / "package.json").write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")
    (path / "index.js").write_text("module.exports = 'probe';\n", encoding="utf-8")
    (path / "README.md").write_text("# npm probe\n", encoding="utf-8")


def test_bash_ut_decode_output_truncates_and_handles_common_encodings():
    assert _decode_output("hello".encode("utf-8")) == "hello"
    assert _decode_output("中文".encode("cp936")) == "中文"
    assert _decode_output("hello".encode("utf-16-le")) == "hello"
    assert len(_decode_output(b"x" * (_MAX_OUTPUT_BYTES + 10))) == _MAX_OUTPUT_BYTES


def test_bash_ut_decode_output_truncated_utf8_boundary_keeps_valid_prefix():
    text = "\u4e2d" * ((_MAX_OUTPUT_BYTES // 3) + 10)

    output = _decode_output(text.encode("utf-8"))

    assert output.startswith("\u4e2d" * 100)
    assert "\u6d93" not in output[:100]
    assert "\ufffd" in output


def test_bash_ut_registers_bash_tool_name_only(tmp_path):
    tools = BashToolkits(session_id="s", workspace_root=str(tmp_path))

    assert tools.get_tool("bash") is not None
    assert tools.get_tool("run_terminal_cmd") is None
    assert [tool.metadata.data["name"] for tool in tools.get_all_tools()] == ["bash"]


def test_bash_ut_find_bash_honors_explicit_override(monkeypatch):
    monkeypatch.setenv("PYGENT_BASH_PATH", "custom-bash")

    assert _find_bash_executable() == "custom-bash"


def test_bash_ut_find_bash_skips_nonfunctional_path_bash(monkeypatch):
    monkeypatch.delenv("PYGENT_BASH_PATH", raising=False)
    monkeypatch.setattr(bash_module.sys, "platform", "win32")
    monkeypatch.setattr(bash_module.shutil, "which", lambda name: "wsl-bash")
    monkeypatch.setattr(bash_module, "_windows_bash_candidates", lambda: ["wsl-bash", "git-bash"])
    monkeypatch.setattr(bash_module, "_is_functional_bash", lambda executable: executable == "git-bash")

    assert _find_bash_executable() == "git-bash"


def test_bash_ut_executes_in_requested_working_directory(tmp_path):
    tools = PythonCommandToolkits(session_id="s", workspace_root=str(tmp_path))
    nested = tmp_path / "nested"
    nested.mkdir()

    output = tools.bash(
        "import pathlib; print(pathlib.Path.cwd().name)",
        working_directory="nested",
        timeout=5000,
    )

    exit_code, terminal_output = _parse_result(output)
    assert exit_code == "0"
    assert terminal_output == f"nested{os.linesep}"


def test_bash_ut_accepts_git_bash_msys_working_directory_on_windows(tmp_path):
    if os.name != "nt":
        pytest.skip("MSYS drive path compatibility is Windows-specific")

    tools = PythonCommandToolkits(session_id="s", workspace_root=str(tmp_path))
    nested = tmp_path / "nested"
    nested.mkdir()

    output = tools.bash(
        "import pathlib; print(pathlib.Path.cwd().name)",
        working_directory=_to_msys_path(nested),
        timeout=5000,
    )

    exit_code, terminal_output = _parse_result(output)
    assert exit_code == "0"
    assert terminal_output == f"nested{os.linesep}"


def test_bash_ut_empty_command_matches_shell_success(tmp_path):
    tools = PythonCommandToolkits(session_id="s", workspace_root=str(tmp_path))

    exit_code, terminal_output = _parse_result(tools.bash(""))

    assert exit_code == "0"
    assert terminal_output == ""


def test_bash_ut_rejects_missing_working_directory(tmp_path):
    tools = PythonCommandToolkits(session_id="s", workspace_root=str(tmp_path))

    output = tools.bash("print('hi')", working_directory="missing")

    assert output.startswith("error:")
    assert str(tmp_path / "missing") in output


def test_bash_ut_missing_working_directory_is_structured_tool_error(tmp_path):
    tools = PythonCommandToolkits(session_id="s", workspace_root=str(tmp_path))

    result = tools.call_tool("bash", command="print('hi')", working_directory="missing")

    assert result["success"] is False
    assert result["error_type"] == "NotADirectoryError"
    assert "working directory" in result["error"]
    assert result["details"]["input_path"] == "missing"
    assert result["details"]["path"] == str(tmp_path / "missing")
    assert "result" not in result


def test_bash_ut_times_out_and_preserves_partial_terminal_output(tmp_path):
    tools = PythonCommandToolkits(session_id="s", workspace_root=str(tmp_path))

    output = tools.bash(
        "import time; print('before', flush=True); time.sleep(1)",
        timeout=50,
    )

    exit_code, terminal_output = _parse_result(output)
    assert exit_code == "timeout"
    assert terminal_output.startswith(f"before{os.linesep}")
    assert "timed out after" in terminal_output


def test_bash_ut_truncates_large_output_without_losing_exit_code(tmp_path):
    tools = PythonCommandToolkits(session_id="s", workspace_root=str(tmp_path))

    output = tools.bash(
        f"import sys; sys.stdout.write('x' * ({_MAX_OUTPUT_BYTES} + 10))",
        timeout=5000,
    )

    exit_code, terminal_output = _parse_result(output)
    assert exit_code == "0"
    assert terminal_output.startswith("x" * 100)
    assert "output truncated" in terminal_output
    full_output_path = _full_output_path_from_result(terminal_output)
    assert full_output_path.parent == tmp_path.resolve()
    assert full_output_path.read_bytes() == b"x" * (_MAX_OUTPUT_BYTES + 10)


def test_bash_ut_truncates_large_utf8_output_without_mojibake(tmp_path):
    tools = PythonCommandToolkits(session_id="s", workspace_root=str(tmp_path))

    output = tools.bash(
        (
            "import sys; "
            f"sys.stdout.buffer.write(('\\u4e2d' * (({_MAX_OUTPUT_BYTES} // 3) + 10)).encode('utf-8'))"
        ),
        timeout=5000,
    )

    exit_code, terminal_output = _parse_result(output)
    assert exit_code == "0"
    assert terminal_output.startswith("\u4e2d" * 100)
    assert "\u6d93" not in terminal_output[:100]
    assert "output truncated" in terminal_output
    full_output_path = _full_output_path_from_result(terminal_output)
    assert full_output_path.parent == tmp_path.resolve()
    assert full_output_path.read_bytes().decode("utf-8") == "\u4e2d" * ((_MAX_OUTPUT_BYTES // 3) + 10)


def test_bash_ut_background_returns_pid(tmp_path):
    tools = PythonCommandToolkits(session_id="s", workspace_root=str(tmp_path))

    output = tools.bash("import time; time.sleep(0.2)", is_background=True)

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
        "x=$(printf abc); echo \"x=$x\"",
        "printf $'\\u4e2d\\u6587\\n'",
        "node -e \"console.log('\\u4e2d\\u6587'); console.error('\\u9519\\u8bef')\"",
        "printf 'a\\0b'",
        "read -r value || true; printf '<%s>' \"$value\"",
        "cd sub && pwd && printf '%s\\n' *.md",
        "if then",
        "exit 42",
    ],
)
def test_bash_st_output_and_exit_code_match_real_bash(tmp_path, command):
    executable = _real_bash_executable()
    _prepare_bash_workspace(tmp_path)
    tools = BashToolkits(session_id="s", workspace_root=str(tmp_path), bash_executable=executable)

    expected_exit_code, expected_output = _run_real_bash(executable, command, str(tmp_path))
    actual_exit_code, actual_output = _parse_result(tools.bash(command, timeout=5000))

    assert actual_exit_code == str(expected_exit_code)
    assert actual_output == expected_output


def test_bash_st_working_directory_matches_real_bash(tmp_path):
    executable = _real_bash_executable()
    _prepare_bash_workspace(tmp_path)
    nested = tmp_path / "nested"
    nested.mkdir()
    tools = BashToolkits(session_id="s", workspace_root=str(tmp_path), bash_executable=executable)

    expected_exit_code, expected_output = _run_real_bash(executable, "pwd", str(nested))
    actual_exit_code, actual_output = _parse_result(
        tools.bash("pwd", working_directory="nested", timeout=5000)
    )

    assert actual_exit_code == str(expected_exit_code)
    assert actual_output == expected_output


def test_bash_st_glob_and_nonzero_exit_match_real_bash(tmp_path):
    executable = _real_bash_executable()
    _prepare_bash_workspace(tmp_path)
    tools = BashToolkits(session_id="s", workspace_root=str(tmp_path), bash_executable=executable)
    command = "printf '%s\\n' *.txt; exit 7"

    expected_exit_code, expected_output = _run_real_bash(executable, command, str(tmp_path))
    actual_exit_code, actual_output = _parse_result(tools.bash(command, timeout=5000))

    assert actual_exit_code == str(expected_exit_code)
    assert actual_output == expected_output


def test_bash_st_large_utf8_output_truncates_without_mojibake(tmp_path):
    executable = _real_bash_executable()
    python_cmd = _bash_python_command(executable)
    tools = BashToolkits(session_id="s", workspace_root=str(tmp_path), bash_executable=executable)
    script = (
        "import sys; "
        f"sys.stdout.buffer.write(('\\u4e2d' * (({_MAX_OUTPUT_BYTES} // 3) + 10)).encode('utf-8'))"
    )

    actual_exit_code, actual_output = _parse_result(
        tools.bash(f"{python_cmd} -c {shlex.quote(script)}", timeout=10000)
    )

    assert actual_exit_code == "0"
    assert actual_output.startswith("\u4e2d" * 100)
    assert "\u6d93" not in actual_output[:100]
    assert "output truncated" in actual_output
    full_output_path = _full_output_path_from_result(actual_output)
    assert full_output_path.parent == tmp_path.resolve()
    assert full_output_path.read_bytes().decode("utf-8") == "\u4e2d" * ((_MAX_OUTPUT_BYTES // 3) + 10)


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
def test_bash_st_npm_commands_match_real_bash(tmp_path, command):
    executable = _real_bash_executable()
    if not _npm_available(executable):
        pytest.skip("npm is not available from bash")
    _prepare_npm_workspace(tmp_path)
    tools = BashToolkits(session_id="s", workspace_root=str(tmp_path), bash_executable=executable)

    expected_exit_code, expected_output = _run_real_bash(executable, command, str(tmp_path))
    actual_exit_code, actual_output = _parse_result(tools.bash(command, timeout=30000))

    assert actual_exit_code == str(expected_exit_code)
    assert actual_output == expected_output
