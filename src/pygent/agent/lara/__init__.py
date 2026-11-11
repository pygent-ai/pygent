"""lara — 从 lara 项目抽取的完整 agent 实现（独立仓 lara）。

本仓只保留 agent 运行时相关子系统：agents（ReAct 脚手架）、runtime（前台
harness、管线、永续会话、实时对话）、sessions、orchestration、schema、
scheduler、tracing、core。环境层（配置文件机制、用户凭据、CLI）
位于 :mod:`lara_shell`，对外契约位于 :mod:`lara_sdk`。

依赖方向固定为 ``lara_sdk -> lara_shell -> lara``：本包不得导入
``lara_shell`` 或 ``lara_sdk``。
"""

from pygent.agent.lara.runtime.tools import ToolObserver

from .graph.core import ForegroundAgent, PygentAgent
from .sessions import SessionManager

__all__ = [
    "ForegroundAgent",
    "PygentAgent",
    "SessionManager",
    "ToolObserver",
]
