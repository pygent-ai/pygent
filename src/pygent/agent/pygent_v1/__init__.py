"""pygent_v1 — 完整的 Agent 个体实现。

本包承载一个自包含的 Agent 个体：agents（main/fork/child 脚手架与
AgentRunner）、graph（PygentAgent 与执行图）、runtime（前台 harness、
管线、永续会话、实时对话）、sessions、orchestration、schema、scheduler、
tracing、core、config、clis。

依赖方向：本包依赖 pygent 的 core、llm、tool、runtime 子包与
:mod:`pygent.agent.react_agent`（内置 ReAct 实现）；这些基础包不得反向
导入本包。``pygent.agent`` 通过延迟导出暴露本包的 ``PygentAgent``。
"""

from pygent.agent.pygent_v1.runtime.tools import ToolObserver

from .graph.core import ForegroundAgent, PygentAgent
from .sessions import SessionManager

__all__ = [
    "ForegroundAgent",
    "PygentAgent",
    "SessionManager",
    "ToolObserver",
]
