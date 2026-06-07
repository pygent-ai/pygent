import asyncio
import json

from pygent.context import BaseContext
from pygent.common import PygentDict, PygentList, PygentSet
from pygent.llm.requests_client import AsyncRequestsClient
from pygent.message import AssistantMessage, ToolCall, UserMessage
from pygent.message.adapters import OllamaMessageAdapter, OpenAIMessageAdapter


def _client():
    return AsyncRequestsClient(
        base_url="https://api.deepseek.com",
        api_key="test-key",
        model_name="deepseek-v4-flash",
    )


def test_parse_response_preserves_reasoning_usage_and_tool_call_details():
    data = {
        "id": "chatcmpl-test",
        "model": "deepseek-v4-flash",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "done",
                    "reasoning_content": "I should call the tool.",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\"city\":\"Beijing\"}",
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "completion_tokens_details": {"reasoning_tokens": 7},
        },
    }

    msg = _client()._parse_response(data)

    assert msg.content.data == "done"
    assert msg.reasoning_content.data == "I should call the tool."
    assert msg.usage.data["completion_tokens_details"]["reasoning_tokens"] == 7
    assert msg.tool_calls.data[0].tool_call_id.data == "call_1"
    assert msg.tool_calls.data[0].tool_type.data == "function"
    assert msg.tool_calls.data[0].data["index"] == 0
    assert msg.tool_calls.data[0].arguments.data == {"city": "Beijing"}


def test_openai_adapter_keeps_reasoning_content_with_tool_calls():
    msg = _client()._parse_response(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "private chain",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": "{\"city\":\"Beijing\"}",
                                },
                            }
                        ],
                    }
                }
            ],
            "usage": {"total_tokens": 1},
        }
    )
    context = BaseContext()
    context.add_message(UserMessage("hello"))
    context.add_message(msg)

    payload = OpenAIMessageAdapter.messages_from_context(context)

    assert payload[1]["role"] == "assistant"
    assert payload[1]["reasoning_content"] == "private chain"
    assert "usage" not in payload[1]
    assert "metadata" not in payload[1]
    assert "index" not in payload[1]["tool_calls"][0]
    assert payload[1]["tool_calls"][0]["function"]["name"] == "get_weather"


def test_openai_adapter_keeps_reasoning_content_without_tool_calls():
    msg = _client()._parse_response(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "final answer",
                        "reasoning_content": "private chain",
                    }
                }
            ],
            "usage": {"total_tokens": 1},
        }
    )
    context = BaseContext()
    context.add_message(UserMessage("hello"))
    context.add_message(msg)

    payload = OpenAIMessageAdapter.messages_from_context(context)

    assert payload[1] == {
        "role": "assistant",
        "content": "final answer",
        "reasoning_content": "private chain",
    }


def test_ollama_adapter_does_not_forward_reasoning_content():
    msg = _client()._parse_response(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "private chain",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": "{\"city\":\"Beijing\"}",
                                },
                            }
                        ],
                    }
                }
            ]
        }
    )

    payload = OllamaMessageAdapter.to_message_dict(msg)

    assert "reasoning_content" not in payload
    assert payload["tool_calls"][0]["function"]["arguments"] == {"city": "Beijing"}


def test_tool_call_from_dict_accepts_dict_arguments():
    tool_call = ToolCall.from_dict(
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": {"city": "Beijing"},
            },
        }
    )

    assert tool_call.arguments.data == {"city": "Beijing"}


def test_streaming_delta_preserves_reasoning_usage_and_tool_type():
    client = _client()
    chunks = [
        client._parse_sse_delta(
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {"delta": {"role": "assistant", "reasoning_content": "think "}}
                    ]
                }
            )
        ),
        client._parse_sse_delta(
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "lookup",
                                            "arguments": "{\"q\":",
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                }
            )
        ),
        client._parse_sse_delta(
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "reasoning_content": "again",
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": "\"x\"}"},
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {"total_tokens": 42},
                }
            )
        ),
    ]

    accumulated = chunks[0] + chunks[1] + chunks[2]
    msg = AssistantMessage("") + accumulated

    assert msg.reasoning_content.data == "think again"
    assert msg.usage.data["total_tokens"] == 42
    assert msg.tool_calls.data[0].tool_type.data == "function"
    assert msg.tool_calls.data[0].arguments.data == {"q": "x"}

    context = BaseContext()
    context.add_message(UserMessage("hello"))
    context.add_message(msg)
    payload = OpenAIMessageAdapter.messages_from_context(context)
    assert payload[1]["reasoning_content"] == "think again"


def test_usage_only_sse_chunk_is_not_dropped():
    chunk = _client()._parse_sse_delta(
        "data: "
        + json.dumps(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "total_tokens": 3,
                },
            }
        )
    )

    assert chunk.usage.data["total_tokens"] == 3


def test_forward_omits_max_tokens_when_not_provided():
    client = _client()
    captured = {}

    def fake_do_request(payload):
        captured["payload"] = payload
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "ok",
                    }
                }
            ]
        }

    client._do_request = fake_do_request

    async def run():
        context = BaseContext()
        context.add_message(UserMessage("hello"))
        await client.forward(context)

    asyncio.run(run())

    assert client.max_tokens.data is None
    assert "max_tokens" not in captured["payload"]


def test_forward_omits_temperature_when_not_provided():
    client = AsyncRequestsClient(
        base_url="https://api.deepseek.com",
        api_key="test-key",
        model_name="deepseek-v4-flash",
        temperature=None,
    )
    captured = {}

    def fake_do_request(payload):
        captured["payload"] = payload
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "ok",
                    }
                }
            ]
        }

    client._do_request = fake_do_request

    async def run():
        context = BaseContext()
        context.add_message(UserMessage("hello"))
        await client.forward(context)

    asyncio.run(run())

    assert client.temperature.data is None
    assert "temperature" not in captured["payload"]


def test_stream_request_retries_transient_error_before_first_chunk(monkeypatch):
    client = AsyncRequestsClient(
        base_url="https://api.deepseek.com",
        api_key="test-key",
        model_name="deepseek-v4-flash",
        max_retries=1,
    )
    attempts = 0

    def fake_http_post_stream(url, headers, body, timeout, debug_body=None):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("TLS handshake failed")
        return 200, iter(
            [
                "data: "
                + json.dumps(
                    {
                        "choices": [
                            {"delta": {"content": "ok"}}
                        ]
                    }
                )
            ]
        )

    monkeypatch.setattr("pygent.llm.requests_client.time.sleep", lambda _: None)
    monkeypatch.setattr("pygent.llm.requests_client._http_post_stream", fake_http_post_stream)

    async def run():
        return [chunk async for chunk in client._do_request_stream({"stream": True})]

    chunks = asyncio.run(run())

    assert attempts == 2
    assert [chunk.content.data for chunk in chunks] == ["ok"]


def test_stream_request_does_not_retry_after_chunk_is_emitted(monkeypatch):
    client = AsyncRequestsClient(
        base_url="https://api.deepseek.com",
        api_key="test-key",
        model_name="deepseek-v4-flash",
        max_retries=1,
    )
    attempts = 0

    def broken_stream():
        yield "data: " + json.dumps({"choices": [{"delta": {"content": "partial"}}]})
        raise RuntimeError("connection reset")

    def fake_http_post_stream(url, headers, body, timeout, debug_body=None):
        nonlocal attempts
        attempts += 1
        return 200, broken_stream()

    monkeypatch.setattr("pygent.llm.requests_client.time.sleep", lambda _: None)
    monkeypatch.setattr("pygent.llm.requests_client._http_post_stream", fake_http_post_stream)

    async def run():
        chunks = []
        try:
            async for chunk in client._do_request_stream({"stream": True}):
                chunks.append(chunk)
        except RuntimeError as exc:
            return chunks, exc
        raise AssertionError("stream error was not raised")

    chunks, exc = asyncio.run(run())

    assert attempts == 1
    assert [chunk.content.data for chunk in chunks] == ["partial"]
    assert str(exc) == "connection reset"


def test_pygent_container_str_does_not_recurse_for_usage_debugging():
    assert str(PygentDict({"total_tokens": 3})) == "{'total_tokens': 3}"
    assert str(PygentList([1, 2])) == "[1, 2]"
    assert str(PygentSet({1})) == "{1}"
