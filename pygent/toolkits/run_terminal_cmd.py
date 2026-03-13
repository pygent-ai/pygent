import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Any

from pygent.common import PygentString
from pygent.module.tool.utils import ToolClassBase, tool_method, tool_class


# 终端命令：超时与输出上限（避免长时间阻塞或内存溢出）
_DEFAULT_TIMEOUT_MS = 30000
_MAX_TIMEOUT_MS = 600000  # 10 分钟上限
_MAX_OUTPUT_BYTES = 512 * 1024  # 单路 stdout/stderr 最多保留 512KB


def _decode_output(data: bytes, max_bytes: int = _MAX_OUTPUT_BYTES) -> str:
    """将子进程输出解码为 str，控制长度并避免解码崩溃。"""
    if not data:
        return ""
    if len(data) > max_bytes:
        data = data[:max_bytes]
    for enc in ("utf-8", "utf-8", "cp936", "cp1252", "latin-1"):
        try:
            return data.decode(enc, errors="strict")
        except (LookupError, UnicodeDecodeError):
            continue
    return data.decode("utf-8", errors="replace")


@tool_class(description="终端命令工具：带超时、工作目录、编码与错误处理的可靠执行。")
class TerminalToolkits(ToolClassBase):
    """终端命令工具：带超时、工作目录、编码与错误处理的可靠执行。"""

    def __init__(self, session_id: str, workspace_root: Optional[str] = None):
        self.session_id = PygentString(session_id)
        self.workspace_root = (workspace_root or os.getcwd()).rstrip(os.sep)
        self._is_windows = sys.platform == "win32"

    @tool_method(
        name="run_terminal_cmd",
        description="在终端执行给定命令，可指定工作目录、超时与是否后台运行。",
    )
    def run_terminal_cmd(
        self,
        command: str,
        working_directory: Optional[str] = None,
        timeout: Optional[float] = None,
        description: Optional[str] = None,
        is_background: bool = False,
    ) -> str:
        """
        在终端执行命令。强制超时、安全的工作目录与编码处理，适合关键场景。

        Args:
            command: 要执行的完整命令字符串。
            working_directory: 执行命令的工作目录绝对路径，不传则使用当前目录。
            timeout: 超时时间（毫秒），不传默认 30000；最大 600000。
            description: 命令的简短描述（约 5–10 个词），仅用于日志。
            is_background: 是否在后台运行该命令；后台时不等待、不捕获输出。
        """
        command = (command or "").strip()
        if not command:
            return "错误：命令不能为空"

        # 超时：毫秒 -> 秒，并限制上限
        timeout_ms = timeout if timeout is not None else _DEFAULT_TIMEOUT_MS
        try:
            timeout_ms = min(max(0, int(timeout_ms)), _MAX_TIMEOUT_MS)
        except (TypeError, ValueError):
            timeout_ms = _DEFAULT_TIMEOUT_MS
        timeout_sec = timeout_ms / 1000.0

        # 工作目录：解析并校验
        cwd: Optional[str] = None
        if working_directory is not None and working_directory.strip():
            try:
                p = Path(working_directory.strip())
                if not p.is_absolute():
                    p = Path(self.workspace_root) / p
                cwd = str(p.resolve())
                if not os.path.isdir(cwd):
                    return f"错误：工作目录不存在或不是目录: {cwd}"
            except (OSError, RuntimeError) as e:
                return f"错误：工作目录无效: {e}"
        else:
            cwd = self.workspace_root

        if is_background:
            return self._run_background(command, cwd)

        return self._run_foreground(command, cwd, timeout_sec)

    def _run_background(self, command: str, cwd: Optional[str]) -> str:
        """后台运行：不等待、不捕获输出，仅返回 PID 与说明。"""
        try:
            kwargs: dict[str, Any] = {
                "shell": True,
                "cwd": cwd,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
                "stdin": subprocess.DEVNULL,
            }
            if not self._is_windows:
                kwargs["start_new_session"] = True
            proc = subprocess.Popen(command, **kwargs)
            pid = proc.pid
            return f"已在后台启动进程 PID={pid}。输出未捕获，请通过日志或文件查看结果。"
        except FileNotFoundError as e:
            return f"错误：未找到可执行程序或 shell: {e}"
        except OSError as e:
            return f"错误：启动失败: {e}"

    def _run_foreground(self, command: str, cwd: Optional[str], timeout_sec: float) -> str:
        """前台运行：等待结束，强制超时，捕获并解码输出。"""
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                timeout=timeout_sec,
                stdin=subprocess.DEVNULL,
            )
            stdout = _decode_output(proc.stdout or b"")
            stderr = _decode_output(proc.stderr or b"")
            exit_code = proc.returncode
            parts = [
                f"exit_code: {exit_code}",
                "",
                "stdout:",
                stdout or "(无)",
                "",
                "stderr:",
                stderr or "(无)",
            ]
            if len((proc.stdout or b"")) > _MAX_OUTPUT_BYTES:
                parts.append(f"\n[stdout 已截断，仅保留前 {_MAX_OUTPUT_BYTES} 字节]")
            if len((proc.stderr or b"")) > _MAX_OUTPUT_BYTES:
                parts.append(f"\n[stderr 已截断，仅保留前 {_MAX_OUTPUT_BYTES} 字节]")
            return "\n".join(parts)
        except subprocess.TimeoutExpired:
            return (
                f"超时（{timeout_sec:.1f} 秒）。进程已被终止。"
                "请考虑缩短执行时间或增大 timeout 参数。"
            )
        except FileNotFoundError as e:
            return f"错误：未找到可执行程序或 shell: {e}"
        except OSError as e:
            return f"错误：执行失败: {e}"
