"""
多 Session 示例：基于 react_agent 实现多会话管理。
每个 Session 拥有独立 session_id、Context 与持久化目录，互不干扰。
"""
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from pygent.agent import BaseAgent
from pygent.context import BaseContext
from pygent.llm import AsyncRequestsClient
from pygent.message import UserMessage, ToolMessage
from pygent.module.tool import ToolManager
from pygent.session import Session
from pygent.toolkits.file_operations import FileToolkits
from pygent.toolkits.run_terminal_cmd import TerminalToolkits
from pygent.toolkits.web_search import WebSearchToolkits
from pygent.toolkits.web_fetch import WebFetchToolkits

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("multi_session_agent")


class SessionReactAgent(BaseAgent):
    """支持 session_id 的 ReactAgent，与 Session 配合使用。"""

    def __init__(self, session_id: str, root_dir: str = "."):
        super().__init__()
        self.session_id = session_id
        self.root_dir = os.path.abspath(root_dir)
        self.llm = AsyncRequestsClient(
            base_url="https://api.deepseek.com",
            api_key=os.environ["DEEPSEEK_API_KEY"],
            model_name="deepseek-chat",
        )
        self.tool_manager = ToolManager()
        for tool in FileToolkits(session_id=session_id, workspace_root=self.root_dir).get_all_tools():
            self.tool_manager.register_tool(tool)
        for tool in TerminalToolkits(session_id=session_id, workspace_root=self.root_dir).get_all_tools():
            self.tool_manager.register_tool(tool)
        for tool in WebSearchToolkits(session_id=session_id, workspace_root=self.root_dir).get_all_tools():
            self.tool_manager.register_tool(tool)
        for tool in WebFetchToolkits(session_id=session_id, workspace_root=self.root_dir).get_all_tools():
            self.tool_manager.register_tool(tool)

    def _tools_param(self):
        funcs = self.tool_manager.get_openai_functions()
        return [{"type": "function", "function": f} for f in funcs]

    def _normalize_tool_kwargs(self, name: str, kwargs: dict) -> dict:
        kwargs = dict(kwargs)
        if name == "run_terminal_cmd":
            if "command" not in kwargs and "cmd" in kwargs:
                kwargs["command"] = kwargs.pop("cmd")
            if "command" not in kwargs and "args" in kwargs:
                args_val = kwargs.pop("args")
                kwargs["command"] = args_val if isinstance(args_val, str) else " ".join(str(x) for x in args_val)
        elif name == "read_file":
            if "path" not in kwargs and "file_path" in kwargs:
                kwargs["path"] = kwargs.pop("file_path")
        elif name == "write":
            if "path" not in kwargs and "file_path" in kwargs:
                kwargs["path"] = kwargs.pop("file_path")
            if "contents" not in kwargs and "content" in kwargs:
                kwargs["contents"] = kwargs.pop("content")
            if "contents" not in kwargs and "text" in kwargs:
                kwargs["contents"] = kwargs.pop("text")
        return kwargs

    async def forward(self, context: BaseContext):
        """使用传入的 context，支持多轮对话与 Session 持久化。"""
        result = await self.llm.forward(context, tools=self._tools_param())
        while getattr(context.last_message, "tool_calls", None) and context.last_message.tool_calls.data:
            for tool_call in context.last_message.tool_calls.data:
                name = tool_call.tool_name.data
                kwargs = self._normalize_tool_kwargs(name, dict(tool_call.arguments.data))
                result = self.tool_manager.call_tool(name, **kwargs)
                content = json.dumps(result, ensure_ascii=False)
                context.add_message(ToolMessage(content=content, tool_call_id=tool_call.tool_call_id.data))
            result = await self.llm.forward(context, tools=self._tools_param())
        return context.last_message.content


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 创建两个 Session
    session_a = Session(
        session_id="user_a_001",
        workspace_root=root_dir,
        system_prompt="你是会话 A 的助手。",
        metadata={"user": "A"},
    )
    session_b = Session(
        session_id="user_b_001",
        workspace_root=root_dir,
        system_prompt="你是会话 B 的助手。",
        metadata={"user": "B"},
    )

    # 为每个 Session 创建 Agent（绑定 session_id）
    agent_a = SessionReactAgent(session_id=session_a.session_id, root_dir=root_dir)
    agent_b = SessionReactAgent(session_id=session_b.session_id, root_dir=root_dir)

    # Session A：使用 read_file 工具，验证 session 会存储 tool_calls 和 ToolMessage
    session_a.context.add_message(UserMessage(content="读取项目根目录的 README.md 文件，告诉我前 100 个字。"))
    await agent_a.forward(session_a.context)
    session_a.save()
    logger.info("Session A reply (with tools): %s", str(session_a.context.last_message.content.data)[:120] if session_a.context.last_message.content else "")

    # 检查 history 中是否有 tool 相关消息
    history = session_a.context.history.data
    tool_msgs = [m for m in history if getattr(m.role, "data", m.role) == "tool"]
    assistant_with_tools = [m for m in history if getattr(m.role, "data", m.role) == "assistant" and getattr(m, "tool_calls", None)]
    logger.info("Session A history: %d messages, %d tool messages, %d assistant with tool_calls", len(history), len(tool_msgs), len(assistant_with_tools))

    # Session B：简单对话
    session_b.context.add_message(UserMessage(content="你好，请记住：我是用户 B。"))
    await agent_b.forward(session_b.context)
    session_b.save()
    logger.info("Session B reply: %s", str(session_b.context.last_message.content.data)[:80] if session_b.context.last_message.content else "")

    # 从磁盘恢复 Session A，验证 tool_calls / ToolMessage 能正确反序列化
    restored_a = Session.load(root_dir, "user_a_001")
    restored_a.context.add_message(UserMessage(content="根据刚才读的内容，用一句话总结这个项目是干什么的。"))
    agent_a_restore = SessionReactAgent(session_id=restored_a.session_id, root_dir=root_dir)
    await agent_a_restore.forward(restored_a.context)
    restored_a.save()
    logger.info("Session A (restored, with tool history): %s", str(restored_a.context.last_message.content.data)[:120] if restored_a.context.last_message.content else "")

    print("Session A (含工具调用) 已保存到:", restored_a.session_dir)
    print("Session 文件:", restored_a.session_dir / "session.json")


if __name__ == "__main__":
    asyncio.run(main())
