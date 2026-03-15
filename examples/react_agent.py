import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import AsyncIterator, Union

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from pygent.agent import BaseAgent
from pygent.context import BaseContext
from pygent.llm import AsyncRequestsClient
from pygent.message import UserMessage, ToolMessage, BaseMessage, BaseMessageChunk, SystemMessage
from pygent.module.tool import ToolManager
from pygent.toolkits.file_operations import FileToolkits
from pygent.toolkits.run_terminal_cmd import TerminalToolkits
from pygent.toolkits.web_search import WebSearchToolkits
from pygent.toolkits.web_fetch import WebFetchToolkits

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger("react_stream_agent")

# 用于 debug 的 context 持久化目录（与 Session 同结构，便于排查 400 等请求问题）
REACT_DEBUG_SESSION_DIR = Path(__file__).resolve().parent.parent / "sessions" / "react_debug"


def save_context_for_debug(context: BaseContext, step_label: str = "") -> str:
    """
    将当前 context 按 Session 格式保存到 sessions/react_debug，便于排查请求体与消息格式问题。
    返回保存的文件路径。
    """
    REACT_DEBUG_SESSION_DIR.mkdir(parents=True, exist_ok=True)
    history = getattr(context, "history", None)
    history_data = []
    if history is not None:
        for msg in (history.data if hasattr(history, "data") else history):
            history_data.append(msg.to_dict())
    system_prompt = getattr(context, "system_prompt", None)
    system_prompt_str = system_prompt.data if hasattr(system_prompt, "data") else (system_prompt or "")
    payload = {
        "version": "1.0",
        "session_id": "react_debug",
        "step": step_label,
        "system_prompt": system_prompt_str,
        "history": history_data,
        "history_count": len(history_data),
    }
    path = REACT_DEBUG_SESSION_DIR / "session.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("Context saved for debug: %s", path)
    return str(path)


class ReactAgent(BaseAgent):
    def __init__(self, root_dir: str = "."):
        super(ReactAgent, self).__init__()
        self.root_dir = os.path.abspath(root_dir)
        self.llm = AsyncRequestsClient(
            base_url="https://api.deepseek.com",
            api_key=os.environ["DEEPSEEK_API_KEY"],
            model_name="deepseek-chat",
        )

        self.tool_manager = ToolManager()
        # 注册文件操作、终端命令与网络工具（绑定到当前 workspace）
        file_toolkits = FileToolkits(session_id="react", workspace_root=self.root_dir)
        for tool in file_toolkits.get_all_tools():
            self.tool_manager.register_tool(tool)
        terminal_toolkits = TerminalToolkits(session_id="react", workspace_root=self.root_dir)
        for tool in terminal_toolkits.get_all_tools():
            self.tool_manager.register_tool(tool)
        web_search_toolkits = WebSearchToolkits(session_id="react", workspace_root=self.root_dir)
        for tool in web_search_toolkits.get_all_tools():
            self.tool_manager.register_tool(tool)
        web_fetch_toolkits = WebFetchToolkits(session_id="react", workspace_root=self.root_dir)
        for tool in web_fetch_toolkits.get_all_tools():
            self.tool_manager.register_tool(tool)

    def _tools_param(self):
        """OpenAI-compatible tools format."""
        funcs = self.tool_manager.get_openai_functions()
        return [{"type": "function", "function": f} for f in funcs]

    def _normalize_tool_kwargs(self, name: str, kwargs: dict) -> dict:
        """将 LLM 常用错误参数名映射到工具实际参数名，避免调用失败。"""
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

    async def forward(self, user_input: str):
        context = BaseContext(system_prompt="""
你是一个小助手。
""")
        context.add_message(UserMessage(content=user_input))

        result = await self.llm.forward(context, tools=self._tools_param())
        logger.info(result)
        while getattr(context.last_message, "tool_calls", None):
            for tool_call in context.last_message.tool_calls:
                name = tool_call.tool_name.data
                kwargs = self._normalize_tool_kwargs(name, dict(tool_call.arguments.data))
                result = self.tool_manager.call_tool(name, **kwargs)
                content = json.dumps(result, ensure_ascii=False)
                context.add_message(
                    ToolMessage(content=content, tool_call_id=tool_call.tool_call_id.data)
                )
            result = await self.llm.forward(context, tools=self._tools_param())
            logger.info(result)

        return context.last_message.content

    async def stream(self, context: BaseContext, max_steps: int = 20) -> AsyncIterator[Union[BaseMessage, BaseMessageChunk]]:
        """与 forward 逻辑一致，但按步 yield 每条助手消息与工具结果消息，便于流式展示或日志。"""
        tools_param = self._tools_param()

        for step in range(max_steps):
            async for chunk in self.llm.stream_forward(context, tools=tools_param):
                yield chunk
            last = context.last_message
            last_type = type(last).__name__
            has_tool_calls = getattr(last, "tool_calls", None)
            has_tool_call_chunks = getattr(last, "tool_call_chunks", None)
            tool_calls_list = list(has_tool_calls.data) if has_tool_calls and hasattr(has_tool_calls, "data") else []
            chunks_list = list(has_tool_call_chunks.data) if has_tool_call_chunks and hasattr(has_tool_call_chunks, "data") else []
            logger.info(
                "stream step=%s: last_message type=%s, has_tool_calls=%s (len=%s), has_tool_call_chunks=%s (len=%s)",
                step + 1, last_type, bool(has_tool_calls), len(tool_calls_list), bool(has_tool_call_chunks), len(chunks_list),
            )
            if tool_calls_list:
                for i, tc in enumerate(tool_calls_list):
                    name = getattr(getattr(tc, "tool_name", None), "data", None)
                    tid = getattr(getattr(tc, "tool_call_id", None), "data", None)
                    logger.info("stream step=%s: tool_calls[%s] name=%s tool_call_id=%s", step + 1, i, name, tid)
            yield context.last_message

            if not getattr(context.last_message, "tool_calls", None) or not context.last_message.tool_calls.data:
                logger.info("stream done: no tool_calls on last_message")
                return

            for tool_call in context.last_message.tool_calls.data:
                name = tool_call.tool_name.data
                kwargs = self._normalize_tool_kwargs(name, dict(tool_call.arguments.data))
                logger.info("stream: calling tool name=%s kwargs=%s", name, kwargs)
                result = self.tool_manager.call_tool(name, **kwargs)
                content = json.dumps(result, ensure_ascii=False)
                tool_msg = ToolMessage(content=content, tool_call_id=tool_call.tool_call_id.data)
                context.add_message(tool_msg)
                yield tool_msg
                if not result.get("success", True):
                    logger.warning("stream: tool call failed result=%s", result)
        logger.warning("stream stopped: max_steps=%s reached", max_steps)


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    agent = ReactAgent(root_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    input_str = "你好"
    context = BaseContext()
    context.add_message(SystemMessage(content="你是一个小助手"))
    context.add_message(UserMessage(content=input_str))

    # 保存 context 便于排查 400 等请求问题（格式与 Session 一致）
    save_context_for_debug(context, step_label="before_first_stream")

    async for message in agent.stream(context, max_steps=200):
        if isinstance(message, BaseMessageChunk):
            pass
        else:
            role = getattr(message, "role", None) and getattr(message.role, "data", message.role)
            content = getattr(message, "content", None) and getattr(message.content, "data", message.content) or ""
            logger.info("received message: role=%s content_len=%s", role, len(content))
            if content:
                logger.info("agent reply content: %s", content)

    context.add_message(UserMessage(content="你当前在哪个目录下"))
    save_context_for_debug(context, step_label="before_second_stream")
    async for message in agent.stream(context, max_steps=200):
        if isinstance(message, BaseMessageChunk):
            pass
        else:
            role = getattr(message, "role", None) and getattr(message.role, "data", message.role)
            content = getattr(message, "content", None) and getattr(message.content, "data", message.content) or ""
            logger.info("received message: role=%s content_len=%s", role, len(content))
            if content:
                logger.info("agent reply content: %s", content)

if __name__ == "__main__":
    asyncio.run(main())