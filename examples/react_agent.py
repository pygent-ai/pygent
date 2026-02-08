import json
import os

from pygent.agent import BaseAgent
from pygent.context import BaseContext
from pygent.llm import AsyncOpenAIClient
from pygent.message import UserMessage, ToolMessage
from pygent.module.tool import ToolManager
from pygent.toolkits import RestrictedTerminal


class ReactAgent(BaseAgent):
    def __init__(self, root_dir: str = "."):
        super(ReactAgent, self).__init__()
        self.llm = AsyncOpenAIClient(
            base_url="https://api.deepseek.com",
            api_key=os.environ["DEEPSEEK_API_KEY"],
            model_name="deepseek-chat",
        )

        self.tool_manager = ToolManager()
        terminal = RestrictedTerminal(root_dir=root_dir)
        self.tool_manager.add_module("terminal", terminal)
        self.tool_manager.register_tools(terminal.get_tools())

    def _tools_param(self):
        """OpenAI-compatible tools format."""
        funcs = self.tool_manager.get_openai_functions()
        return [{"type": "function", "function": f} for f in funcs]

    async def forward(self, user_input: str):
        context = BaseContext(system_prompt="""
你是一个小助手。
""")
        context.add_message(UserMessage(content=user_input))

        context = await self.llm.forward(context, tools=self._tools_param())

        while getattr(context.last_message, "tool_calls", None):
            for tool_call in context.last_message.tool_calls:
                name = tool_call.tool_name.data
                kwargs = dict(tool_call.arguments.data)
                result = self.tool_manager.call_tool(name, **kwargs)
                content = json.dumps(result, ensure_ascii=False)
                context.add_message(
                    ToolMessage(content=content, tool_call_id=tool_call.tool_call_id.data)
                )
            context = await self.llm.forward(context, tools=self._tools_param())

        return context.last_message.content


