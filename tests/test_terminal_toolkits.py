import sys

from pygent.toolkits.run_terminal_cmd import TerminalToolkits, _MAX_OUTPUT_BYTES, _decode_output


def test_decode_output_truncates_and_handles_legacy_encoding():
    assert _decode_output("hello".encode("utf-8")) == "hello"
    assert _decode_output("中文".encode("cp936")) == "中文"
    assert len(_decode_output(b"x" * (_MAX_OUTPUT_BYTES + 10))) == _MAX_OUTPUT_BYTES


def test_run_terminal_cmd_executes_in_requested_working_directory(tmp_path):
    tools = TerminalToolkits(session_id="s", workspace_root=str(tmp_path))
    script = "import pathlib; print(pathlib.Path.cwd().name)"

    output = tools.run_terminal_cmd(
        f'{sys.executable} -c "{script}"',
        working_directory=str(tmp_path),
        timeout=5000,
    )

    assert "exit_code: 0" in output
    assert tmp_path.name in output


def test_run_terminal_cmd_rejects_empty_and_missing_working_directory(tmp_path):
    tools = TerminalToolkits(session_id="s", workspace_root=str(tmp_path))

    assert "命令不能为空" in tools.run_terminal_cmd("")
    missing_output = tools.run_terminal_cmd("echo hi", working_directory="missing")
    assert str(tmp_path / "missing") in missing_output


def test_run_terminal_cmd_times_out(tmp_path):
    tools = TerminalToolkits(session_id="s", workspace_root=str(tmp_path))
    output = tools.run_terminal_cmd(
        f'{sys.executable} -c "import time; time.sleep(1)"',
        timeout=1,
    )

    assert "超时" in output


def test_run_terminal_cmd_background_returns_pid(tmp_path):
    tools = TerminalToolkits(session_id="s", workspace_root=str(tmp_path))
    output = tools.run_terminal_cmd(
        f'{sys.executable} -c "print(123)"',
        is_background=True,
    )

    assert "PID=" in output
