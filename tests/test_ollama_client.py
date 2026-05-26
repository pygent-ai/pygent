import asyncio
import builtins
import sys
from types import SimpleNamespace

import pytest

from pygent.context import BaseContext
from pygent.llm import OllamaAsyncClient
from pygent.message import AssistantMessage, ToolCall, ToolMessage, UserMessage


def test_to_ollama_format_tool_call_arguments_are_dict():
    msg = AssistantMessage(
        content="",
        tool_calls=[
            ToolCall(
                tool_call_id="call_1",
                tool_name="get_weather",
                arguments={"city": "Beijing"},
            )
        ],
    )

    payload = msg.to_ollama_format()
    assert payload["role"] == "assistant"
    assert payload["tool_calls"][0]["function"]["name"] == "get_weather"
    assert isinstance(payload["tool_calls"][0]["function"]["arguments"], dict)
    assert payload["tool_calls"][0]["function"]["arguments"]["city"] == "Beijing"


def test_tool_message_to_ollama_format_contains_tool_call_id():
    msg = ToolMessage(content='{"temp": 20}', tool_call_id="call_1")
    payload = msg.to_ollama_format()
    assert payload["role"] == "tool"
    assert payload["tool_call_id"] == "call_1"


def test_stream_forward_writes_accumulated_message_and_tool_calls(monkeypatch):
    captured = {}

    def fake_chat(**kwargs):
        captured["kwargs"] = kwargs
        return iter(
            [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": {"city": "Beijing"},
                                },
                            }
                        ],
                    }
                },
                {"message": {"role": "assistant", "content": "北京今天晴。", "tool_calls": []}},
            ]
        )

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))

    context = BaseContext()
    context.add_message(UserMessage("今天天气如何？"))
    context.add_message(
        AssistantMessage(
            content="",
            tool_calls=[ToolCall("call_1", "get_weather", {"city": "Beijing"})],
        )
    )
    context.add_message(ToolMessage(content='{"temp": 20}', tool_call_id="call_1"))

    client = OllamaAsyncClient(model_name="qwen2.5")

    async def run():
        chunks = []
        async for chunk in client.stream_forward(context, tools=[{"type": "function"}]):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(run())

    assert len(chunks) == 2
    assert context.last_message.content.data == "北京今天晴。"
    assert len(context.last_message.tool_calls.data) == 1
    assert context.last_message.tool_calls.data[0].tool_name.data == "get_weather"
    assert context.last_message.tool_calls.data[0].arguments.data["city"] == "Beijing"

    sent_messages = captured["kwargs"]["messages"]
    assert sent_messages[0]["role"] == "user"
    assert sent_messages[1]["role"] == "assistant"
    assert sent_messages[1]["tool_calls"][0]["function"]["arguments"]["city"] == "Beijing"
    assert sent_messages[2]["role"] == "tool"
    assert sent_messages[2]["tool_call_id"] == "call_1"


def test_stream_forward_raises_when_no_output(monkeypatch):
    def fake_chat(**kwargs):
        return iter([])

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))
    context = BaseContext()
    context.add_message(UserMessage("hello"))
    client = OllamaAsyncClient(model_name="qwen2.5")

    async def run():
        async for _ in client.stream_forward(context):
            pass

    with pytest.raises(RuntimeError, match="No output from stream"):
        asyncio.run(run())


def test_missing_ollama_dependency_raises_clear_error(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ollama":
            raise ModuleNotFoundError("No module named ollama")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "ollama", raising=False)

    client = OllamaAsyncClient(model_name="qwen2.5")
    with pytest.raises(RuntimeError, match="pip install ollama"):
        client._get_ollama_module()


def test_real_model_tripolskypetr_qwen35_works():
    """真实模型连通性测试（依赖本地 Ollama 服务与已拉取模型）。"""
    pytest.importorskip("ollama")
    client = OllamaAsyncClient(model_name="tripolskypetr/qwen3.5-uncensored-aggressive:27b")

    # 非流式
    context = BaseContext()
    context.add_message(UserMessage("Reply with: REAL_MODEL_OK"))
    msg = asyncio.run(client.forward(context))
    assert isinstance(msg.content.data, str)
    assert len(msg.content.data.strip()) > 0

    # 流式：至少拿到一个非空 chunk
    context_stream = BaseContext()
    context_stream.add_message(UserMessage("Reply with: STREAM_MODEL_OK"))

    async def run_stream():
        chunks = []
        async for chunk in client.stream_forward(context_stream):
            text = (chunk.content.data or "").strip()
            if text:
                chunks.append(text)
                break
        return chunks

    streamed = asyncio.run(run_stream())
    assert len(streamed) >= 1
