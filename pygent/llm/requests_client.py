"""
基于 urllib 的异步大模型客户端。
仅使用 Python 内置库（urllib、json、asyncio），支持流式与非流式调用。
"""

import asyncio
import json
import time
from typing import Callable, Optional

from pygent.context import BaseContext
from pygent.llm import BaseAsyncClient
from pygent.message import AssistantMessage, AssistantMessageChunk, ToolCall, ToolCallChunk


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

    def _do_request_stream(self, payload: dict):
        """同步执行流式请求，返回生成器产出 AssistantMessageChunk"""
        url = self._get_chat_url()
        headers = {
            "Authorization": f"Bearer {self.api_key.data}",
            "Content-Type": "application/json",
        }
        body = json.dumps(payload).encode("utf-8")
        status, line_iter = _http_post_stream(url, headers, body, self.timeout.data)
        if status != 200:
            raise RuntimeError(f"HTTP {status}")
        for line in line_iter:
            chunk = self._parse_sse_delta(line)
            if chunk is not None:
                yield chunk

    async def forward(
        self,
        context: BaseContext,
        on_chunk: Optional[Callable[[AssistantMessageChunk], None]] = None,
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
        stream = self.stream.data or kwargs.get("stream", False)
        request_params["stream"] = stream

        if not stream:
            data = await asyncio.to_thread(self._do_request, request_params)
            assistant_msg = self._parse_response(data)
            context.add_message(assistant_msg)
            return context.last_message

        # 流式：在线程中收集 chunks，主协程累积并回调
        def _collect_chunks():
            chunks = []
            for c in self._do_request_stream(request_params):
                chunks.append(c)
            return chunks

        chunks = await asyncio.to_thread(_collect_chunks)
        accumulated = None
        for c in chunks:
            if on_chunk is not None:
                on_chunk(c)
            accumulated = c if accumulated is None else accumulated + c
        if accumulated is not None:
            assistant_msg = accumulated._merge_into_message(AssistantMessage(content=""))
        else:
            assistant_msg = AssistantMessage(content="")
        context.add_message(assistant_msg)
        return context.last_message
