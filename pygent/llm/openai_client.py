from typing import Optional

from openai import AsyncClient

from pygent.context import BaseContext
from pygent.llm import BaseAsyncClient
from pygent.message import AssistantMessage, ToolCall


class AsyncOpenAIClient(BaseAsyncClient):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        timeout: int = 30,
        max_retries: int = 3,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ):
        super().__init__(base_url, api_key, model_name, timeout, max_retries, temperature, max_tokens, stream, **kwargs)

        self.client = AsyncClient(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)

    async def forward(self, context: BaseContext, **kwargs) -> BaseContext:
        request_params = {
            "model": self.model_name.data,
            "messages": [m.to_openai_format() for m in context.history],
            "temperature": self.temperature.data,
            **kwargs
        }
        if self.max_tokens.data:
            request_params["max_tokens"] = self.max_tokens.data

        response = await self.client.chat.completions.create(**request_params)

        choice = response.choices[0]
        message = choice.message
        content = message.content or ""
        tool_calls = getattr(message, "tool_calls", None)

        if tool_calls:
            tool_calls_list = [
                ToolCall.from_dict(tc.model_dump() if hasattr(tc, "model_dump") else dict(tc))
                for tc in tool_calls
            ]
            context.add_message(AssistantMessage(content=content, tool_calls=tool_calls_list))
        else:
            context.add_message(AssistantMessage(content=content))
        return context