"""
基于 urllib 的异步大模型客户端。
仅使用 Python 内置库（urllib、json、asyncio），支持流式与非流式调用。
"""

import asyncio
import json
import logging
import time
from typing import Callable, Optional, Any, AsyncGenerator

from pygent.context import BaseContext
from pygent.llm import BaseAsyncClient
from pygent.message import AssistantMessage, AssistantMessageChunk, ToolCall, ToolCallChunk

logger = logging.getLogger(__name__)


def _http_post(url: str, headers: dict, body: bytes, timeout: int) -> tuple[int, bytes]:
    """使用 urllib 同步 POST 请求"""
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError, URLError

    req = Request(url, data=body, headers=headers, method="POST")
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except (HTTPError, URLError) as e:
        if hasattr(e, "code"):
            raise RuntimeError(f"HTTP {e.code}: {e.reason}") from e
        raise RuntimeError(str(e)) from e


def _http_post_stream(url: str, headers: dict, body: bytes, timeout: int):
    """使用 urllib 同步 POST 流式请求，返回可迭代的 (status, line_iterator)"""
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError, URLError

    req = Request(url, data=body, headers=headers, method="POST")
    try:
        resp = urlopen(req, timeout=timeout)
        status = resp.status

        def _iter_lines():
            buf = b""
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf or b"\r" in buf:
                    line, _, buf = buf.partition(b"\n")
                    if b"\r" in line:
                        line = line.split(b"\r")[0]
                    if line.strip():
                        yield line.decode("utf-8", errors="replace")
            if buf.strip():
                yield buf.decode("utf-8", errors="replace")

        return status, _iter_lines()
    except (HTTPError, URLError) as e:
        if hasattr(e, "code"):
            raise RuntimeError(f"HTTP {e.code}: {e.reason}") from e
        raise RuntimeError(str(e)) from e


class AsyncRequestsClient(BaseAsyncClient):
    """
    使用 Python 内置 urllib 请求大模型的异步客户端。
    兼容 OpenAI Chat Completions API 格式，支持流式与非流式。
    """

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
        super().__init__(
            base_url, api_key, model_name,
            timeout, max_retries, temperature, max_tokens, stream,
            **kwargs
        )

    def _get_chat_url(self) -> str:
        """获取 chat completions 接口 URL"""
        base = self.base_url.data.rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    def _do_request(self, payload: dict) -> dict:
        """同步执行 HTTP 请求（非流式）"""
        url = self._get_chat_url()
        headers = {
            "Authorization": f"Bearer {self.api_key.data}",
            "Content-Type": "application/json",
        }
        body = json.dumps(payload).encode("utf-8")
        for attempt in range(self.max_retries.data + 1):
            try:
                status, data = _http_post(url, headers, body, self.timeout.data)
                if status != 200:
                    raise RuntimeError(f"HTTP {status}: {data.decode('utf-8', errors='replace')[:500]}")
                return json.loads(data.decode("utf-8"))
            except Exception:
                if attempt == self.max_retries.data:
                    raise
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError("Unreachable")

    def _parse_response(self, data: dict) -> AssistantMessage:
        """解析 API 响应为 AssistantMessage"""
        choices = data.get("choices", [])
        if not choices:
            return AssistantMessage(content="")
        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content") or ""
        tool_calls_raw = message.get("tool_calls")
        if tool_calls_raw:
            tool_calls_list = [
                ToolCall.from_dict(
                    tc if isinstance(tc, dict) else (tc.model_dump() if hasattr(tc, "model_dump") else dict(tc))
                )
                for tc in tool_calls_raw
            ]
            return AssistantMessage(content=content, tool_calls=tool_calls_list)
        return AssistantMessage(content=content)

    def _parse_sse_delta(self, line: str) -> Optional[AssistantMessageChunk]:
        """解析 SSE 行，提取 delta 并转为 AssistantMessageChunk"""
        if not line.startswith("data: "):
            return None
        payload = line[6:].strip()
        if payload == "[DONE]":
            return None
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return None
        choices = data.get("choices", [])
        if not choices:
            return None
        choice = choices[0]
        delta = choice.get("delta", {})
        content = delta.get("content") or ""
        tool_calls_delta = delta.get("tool_calls") or []
        tool_call_chunks = []
        for tc in tool_calls_delta:
            idx = tc.get("index", 0)
            tid = tc.get("id") or ""
            fn = tc.get("function", {})
            name = fn.get("name") or ""
            args = fn.get("arguments") or ""
            tool_call_chunks.append(
                ToolCallChunk(index=idx, tool_call_id=tid, tool_name=name, arguments=args)
            )
        if not content and not tool_call_chunks:
            return None
        return AssistantMessageChunk(content=content, tool_call_chunks=tool_call_chunks or None)

    def _stream_worker_sync(self, payload: dict, queue: asyncio.Queue) -> None:
        """在后台线程中执行：同步流式请求，将 AssistantMessageChunk 放入 queue，结束时 put(None)。"""
        url = self._get_chat_url()
        headers = {
            "Authorization": f"Bearer {self.api_key.data}",
            "Content-Type": "application/json",
        }
        body = json.dumps(payload).encode("utf-8")
        try:
            status, line_iter = _http_post_stream(url, headers, body, self.timeout.data)
            if status != 200:
                queue.put_nowait(RuntimeError(f"HTTP {status}"))
                return
            for line in line_iter:
                chunk = self._parse_sse_delta(line)
                if chunk is not None:
                    queue.put_nowait(chunk)
        except Exception as e:
            queue.put_nowait(e)
        finally:
            queue.put_nowait(None)

    async def _do_request_stream(self, payload: dict) -> AsyncGenerator[AssistantMessageChunk, Any]:
        """异步流式请求：在线程中执行同步 I/O，通过 queue 产出 AssistantMessageChunk，不阻塞事件循环。"""
        queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def worker():
            self._stream_worker_sync(payload, queue)

        future = loop.run_in_executor(None, worker)
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item
        await future

    async def forward(
        self,
        context: BaseContext,
        **kwargs
    ) -> AssistantMessage:
        history = getattr(context, "history", None)
        messages = []
        if history is not None:
            messages = [m.to_openai_format() for m in history]
        request_params = {
            "model": self.model_name.data,
            "messages": messages,
            "temperature": self.temperature.data,
            **kwargs
        }
        if self.max_tokens.data:
            request_params["max_tokens"] = self.max_tokens.data
        request_params["stream"] = False
        data = await asyncio.to_thread(self._do_request, request_params)
        assistant_msg = self._parse_response(data)
        context.add_message(assistant_msg)
        return context.last_message

    async def stream_forward(
        self,
        context: BaseContext,
        **kwargs
    ) -> AsyncGenerator[AssistantMessageChunk, Any]:
        history = context.history
        messages = []
        if history is not None:
            messages = [m.to_openai_format() for m in history]
        request_params = {
            "model": self.model_name.data,
            "messages": messages,
            "temperature": self.temperature.data,
            **kwargs
        }
        if self.max_tokens.data:
            request_params["max_tokens"] = self.max_tokens.data
        request_params["stream"] = True

        tools = request_params.get("tools")
        tools_count = len(tools) if tools else 0
        logger.info("[stream_forward] request: messages=%s, tools_count=%s", len(messages), tools_count)

        accumulated = None
        chunk_index = 0
        async for chunk in self._do_request_stream(request_params):
            yield chunk
            chunk_index += 1
            if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks and chunk.tool_call_chunks.data:
                for i, tc in enumerate(chunk.tool_call_chunks.data):
                    logger.info(
                        "[stream_forward] chunk#%s tool_call_chunk: index=%s id=%s name=%s args_len=%s",
                        chunk_index, getattr(tc, "index", None),
                        getattr(getattr(tc, "tool_call_id", None), "data", tc) if hasattr(tc, "tool_call_id") else None,
                        getattr(getattr(tc, "tool_name", None), "data", "") if hasattr(tc, "tool_name") else None,
                        len(getattr(getattr(tc, "arguments", None), "data", "") or "") if hasattr(tc, "arguments") else 0,
                    )
            accumulated = chunk if accumulated is None else accumulated + chunk

        if accumulated is not None:
            has_chunks = getattr(accumulated, "tool_call_chunks", None) and accumulated.tool_call_chunks.data
            logger.info(
                "[stream_forward] accumulated: type=%s, content_len=%s, tool_call_chunks_count=%s",
                type(accumulated).__name__,
                len(getattr(accumulated.content, "data", accumulated.content) or ""),
                len(accumulated.tool_call_chunks.data) if has_chunks else 0,
            )
            # 关键：流式得到的是 AssistantMessageChunk（只有 tool_call_chunks），
            # agent 检查的是 tool_calls。必须合并为完整 AssistantMessage 再写入 context。
            full_message = AssistantMessage(content="") + accumulated
            tool_calls_count = len(full_message.tool_calls.data) if getattr(full_message, "tool_calls", None) else 0
            logger.info("[stream_forward] full_message: tool_calls_count=%s", tool_calls_count)
            context.add_message(full_message)
        else:
            raise RuntimeError("[Pygent] No output from stream")