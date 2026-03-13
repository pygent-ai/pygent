import asyncio
import json
import os
from collections import AsyncIterable

from pygent.agent import BaseAgent
from pygent.context import BaseContext
from pygent.llm import AsyncOpenAIClient
from pygent.message import UserMessage, ToolMessage, BaseMessage
from pygent.module.tool import ToolManager

from dotenv import load_dotenv
load_dotenv()
import logging

logger = logging.getLogger("react_stream_agent")


class ReactAgent(BaseAgent):
    def __init__(self, root_dir: str = "."):
        super(ReactAgent, self).__init__()
        self.llm = AsyncOpenAIClient(
            base_url="https://api.deepseek.com",
            api_key=os.environ["DEEPSEEK_API_KEY"],
            model_name="deepseek-chat",
        )

        self.tool_manager = ToolManager()
        # todo add file operations and run_terminal_cmd

    def _tools_param(self):
        """OpenAI-compatible tools format."""
        funcs = self.tool_manager.get_openai_functions()
        return [{"type": "function", "function": f} for f in funcs]

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
                kwargs = dict(tool_call.arguments.data)
                result = self.tool_manager.call_tool(name, **kwargs)
                content = json.dumps(result, ensure_ascii=False)
                context.add_message(
                    ToolMessage(content=content, tool_call_id=tool_call.tool_call_id.data)
                )
            result = await self.llm.forward(context, tools=self._tools_param())
            logger.info(result)

        return context.last_message.content

    async def stream(self, user_input: str) -> AsyncIterable[BaseMessage]:
        # todo finish and test end to end
        pass


async def main():
    agent = ReactAgent()
    # todo
    input_str = ""

    async for message in agent.stream(input_str):
        logger.info(message)


if __name__ == "__main__":
    asyncio.run(main())