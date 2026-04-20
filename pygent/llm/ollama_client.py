import json
from typing import Any, AsyncGenerator, Optional

from pygent.context import BaseContext
from pygent.llm import BaseAsyncClient
from pygent.message import AssistantMessage, AssistantMessageChunk, ToolCallChunk


class OllamaAsyncClient(BaseAsyncClient):
    """Ollama 本地模型异步客户端。"""

    def __init__(
        self,
        model_name: str,
        timeout: int = 30,
        max_retries: int = 3,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ):
        super().__init__(
            base_url="",
            api_key="",
            model_name=model_name,
            timeout=timeout,
            max_retries=max_retries,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            **kwargs
        )

    @staticmethod
    def _get_ollama_module():
        try:
            import ollama  # type: ignore
        except Exception as ex:
            raise RuntimeError("[Pygent] `ollama` package is required. Install with: pip install ollama") from ex
        return ollama

    @staticmethod
    def _get_attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    @classmethod
    def _get_chunk(cls, chunk: Any) -> AssistantMessageChunk:
        message = cls._get_attr_or_key(chunk, "message", {})
        content = cls._get_attr_or_key(message, "content", "") or ""
        role = cls._get_attr_or_key(message, "role", "") or ""
        tool_calls_raw = cls._get_attr_or_key(message, "tool_calls", []) or []

        tool_call_chunks = []
        for i, tool_call_chunk in enumerate(tool_calls_raw):
            function_obj = cls._get_attr_or_key(tool_call_chunk, "function", {})
            function_name = cls._get_attr_or_key(function_obj, "name", "") or ""
            arguments_obj = cls._get_attr_or_key(function_obj, "arguments", {})
            tool_call_id = cls._get_attr_or_key(tool_call_chunk, "id", None)
            arguments_str = json.dumps(arguments_obj, ensure_ascii=False) if isinstance(arguments_obj, dict) else (
                arguments_obj if isinstance(arguments_obj, str) else "{}"
            )
            tool_call_chunks.append(
                ToolCallChunk(
                    index=i,
                    tool_call_id=tool_call_id,
                    tool_name=function_name,
                    arguments=arguments_str,
                )
            )

        return AssistantMessageChunk(
            content=content,
            name=role,
            tool_call_chunks=tool_call_chunks,
        )

    async def forward(self, context: BaseContext, **kwargs) -> AssistantMessage:
        ollama = self._get_ollama_module()

        history = getattr(context, "history", None)
        iterable = (history.data if hasattr(history, "data") else history) if history is not None else []
        messages = [m.to_ollama_format() for m in iterable]

        request_params = {
            "model": self.model_name.data,
            "messages": messages,
            "stream": False,
            **kwargs,
        }
        if "think" not in request_params:
            request_params["think"] = False
        if "options" not in request_params:
            request_params["options"] = {"temperature": self.temperature.data}

        response = ollama.chat(**request_params)
        chunk = self._get_chunk(response)
        full_message = AssistantMessage(content="") + chunk
        context.add_message(full_message)
        return context.last_message

    async def stream_forward(
        self,
        context: BaseContext,
        **kwargs
    ) -> AsyncGenerator[AssistantMessageChunk, Any]:
        ollama = self._get_ollama_module()

        history = getattr(context, "history", None)
        iterable = (history.data if hasattr(history, "data") else history) if history is not None else []
        messages = [m.to_ollama_format() for m in iterable]

        request_params = {
            "model": self.model_name.data,
            "messages": messages,
            "stream": True,
            **kwargs,
        }
        if "think" not in request_params:
            request_params["think"] = False
        if "options" not in request_params:
            request_params["options"] = {"temperature": self.temperature.data}

        accumulated = None
        response = ollama.chat(**request_params)

        for raw_chunk in response:
            chunk = self._get_chunk(raw_chunk)
            yield chunk
            accumulated = chunk if accumulated is None else accumulated + chunk

        if accumulated is not None:
            full_message = AssistantMessage(content="") + accumulated
            context.add_message(full_message)
        else:
            raise RuntimeError("[Pygent] No output from stream")
