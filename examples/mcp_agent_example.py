"""
MCP 工具使用示例：Agent 通过 LLM 决策调用 MCP 工具。

类似 react_agent，使用 BaseAgent + LLM + ToolManager，仅注册 MCP 工具。
LLM 根据用户输入决定调用哪些 MCP 工具，验证端到端工具调用流程。

运行：python examples/mcp_agent_example.py
需要 .env 中配置 DEEPSEEK_API_KEY。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from pygent.agent import BaseAgent
from pygent.context import BaseContext
from pygent.llm import AsyncRequestsClient
from pygent.message import UserMessage, ToolMessage
from pygent.module.tool import ToolManager

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mcp_agent_example")


class MCPAgent(BaseAgent):
    """仅使用 MCP 工具的 Agent，用于验证 MCP 端到端调用。"""

    def __init__(self, root_dir: str = "."):
        super().__init__()
        self.root_dir = os.path.abspath(root_dir)
        self.llm = AsyncRequestsClient(
            base_url="https://api.deepseek.com",
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            model_name="deepseek-chat",
        )
        self.tool_manager = ToolManager()

        server_script = root / "tests" / "mcp_test_server.py"
        if not server_script.is_file():
            raise FileNotFoundError(f"未找到 MCP 测试服务器 {server_script}")

        tools = self.tool_manager.add_mcp_server_stdio(
            server_id="mcp_demo",
            command=sys.executable,
            args=[str(server_script)],
            cwd=str(root),
        )
        logger.info("已注册 %d 个 MCP 工具: %s", len(tools), [t.metadata.data["name"] for t in tools])

    def _tools_param(self):
        funcs = self.tool_manager.get_openai_functions()
        return [{"type": "function", "function": f} for f in funcs]

    def _normalize_tool_kwargs(self, name: str, kwargs: dict) -> dict:
        kwargs = dict(kwargs)
        if name == "write_raw_conversation" and "content" not in kwargs and "text" in kwargs:
            kwargs["content"] = kwargs.pop("text")
        return kwargs

    async def forward(self, user_input: str, max_steps: int = 10):
        context = BaseContext(system_prompt="""你是一个小助手，可以使用以下 MCP 工具：
- get_memory_statistics: 获取内存统计信息
- list_memory_files: 列出内存文件
- write_raw_conversation: 将对话内容写入内存，需要 content 参数

根据用户请求选择合适的工具调用，完成后直接回复结论。""")
        context.add_message(UserMessage(content=user_input))

        result = await self.llm.forward(context, tools=self._tools_param())
        step = 0
        while step < max_steps and getattr(context.last_message, "tool_calls", None) and context.last_message.tool_calls.data:
            for tool_call in context.last_message.tool_calls.data:
                name = tool_call.tool_name.data
                kwargs = self._normalize_tool_kwargs(name, dict(tool_call.arguments.data))
                logger.info("agent tool_call name=%s kwargs=%s", name, kwargs)
                result = self.tool_manager.call_tool(name, **kwargs)
                logger.info("agent tool_result name=%s success=%s result=%s", name, result.get("success"), result.get("result"))
                if not result.get("success", True):
                    logger.warning("tool call failed: %s", result)
                content = json.dumps(result, ensure_ascii=False)
                context.add_message(ToolMessage(content=content, tool_call_id=tool_call.tool_call_id.data))
            result = await self.llm.forward(context, tools=self._tools_param())
            step += 1

        return context.last_message.content


async def main():
    agent = MCPAgent(root_dir=str(root))
    user_input = "请调用 get_memory_statistics 获取内存统计，然后调用 write_raw_conversation 把 '测试对话' 写入内存，最后用一句话总结结果。"
    logger.info("user_input: %s", user_input)
    content = await agent.forward(user_input)
    reply = content.data if hasattr(content, "data") else content
    logger.info("agent final reply: %s", reply)
    print("Agent 回复:", reply)


if __name__ == "__main__":
    asyncio.run(main())
