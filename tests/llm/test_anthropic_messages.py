from __future__ import annotations

import json
from dataclasses import replace

import httpx
import pytest

from pygent import (
    AIMessage,
    Context,
    GenerationConfig,
    ModelContinuation,
    ToolCall,
    ToolDefinition,
    ToolMessage,
    ToolResult,
    UserMessage,
)
from pygent.core import freeze_json_object
from pygent.llm import (
    AnthropicMessagesAdapter,
    AnthropicMessagesClient,
    DefaultModelInvoker,
    ModelEntry,
    ModelErrorKind,
    ModelFailureReason,
    ModelGroup,
    ModelProviderError,
    ModelProviderRequest,
    RetryPolicy,
)
from tests.support.model_specs import model_entry


def anthropic_entry(
    *, provider: str = "anthropic", options: dict[str, object] | None = None
) -> ModelEntry:
    source = model_entry("main", provider, "claude-opus-5")
    return ModelEntry(
        source.name,
        replace(
            source.spec,
            protocol="anthropic_messages",
            provider_options=options or {},
        ),
    )


def request(
    *,
    entry: ModelEntry | None = None,
    message=None,
    context: Context | None = None,
    generation: GenerationConfig | None = None,
    tools: tuple[ToolDefinition, ...] = (),
) -> ModelProviderRequest:
    value = entry or anthropic_entry()
    return ModelProviderRequest(
        model_key=value.name,
        model=value.spec,
        message=message or UserMessage(content="question"),
        context=context or Context(),
        generation=generation or GenerationConfig(max_output_tokens=4096),
        tools=tools,
    )


def test_anthropic_request_requires_explicit_max_tokens() -> None:
    with pytest.raises(ModelProviderError) as raised:
        AnthropicMessagesAdapter().build_request(
            request(generation=GenerationConfig())
        )
    assert raised.value.kind is ModelErrorKind.INVALID_REQUEST


def test_anthropic_request_projects_system_tools_choice_and_schema() -> None:
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    tool = ToolDefinition(
        name="lookup", description="Lookup", parameters={"type": "object"}
    )
    payload = AnthropicMessagesAdapter().build_request(
        request(
            context=Context(system_prompt="system"),
            tools=(tool,),
            generation=GenerationConfig(
                max_output_tokens=4096,
                tool_choice="required",
                response_schema=schema,
            ),
        )
    ).to_dict()

    assert payload["model"] == "claude-opus-5"
    assert payload["max_tokens"] == 4096
    assert payload["system"] == "system"
    assert payload["tools"][0]["input_schema"]["type"] == "object"
    assert payload["tool_choice"] == {"type": "any"}
    assert payload["output_config"]["format"] == {
        "type": "json_schema",
        "schema": schema,
    }


def test_anthropic_request_projects_valid_provider_options() -> None:
    payload = AnthropicMessagesAdapter().build_request(
        request(
            entry=anthropic_entry(
                options={
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": 1024,
                        "display": "summarized",
                    },
                    "output_config": {"effort": "high"},
                    "service_tier": "standard_only",
                    "stop_sequences": ["END"],
                }
            ),
            generation=GenerationConfig(
                max_output_tokens=4096,
                temperature=1,
            ),
        )
    ).to_dict()

    assert payload["thinking"] == {
        "type": "enabled",
        "budget_tokens": 1024,
        "display": "summarized",
    }
    assert payload["output_config"] == {"effort": "high"}
    assert payload["service_tier"] == "standard_only"
    assert payload["stop_sequences"] == ["END"]


def test_anthropic_request_projects_assistant_and_tool_results() -> None:
    call = ToolCall(call_id="call-1", name="lookup", arguments={"id": 1})
    context = Context(
        messages=(AIMessage(content="checking", tool_calls=(call,)),)
    )
    payload = AnthropicMessagesAdapter().build_request(
        request(
            message=ToolMessage(
                results=(
                    ToolResult(
                        call_id="call-1",
                        name="lookup",
                        status="failed",
                        error="not found",
                    ),
                )
            ),
            context=context,
        )
    ).to_dict()

    assert payload["messages"][0] == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "checking"},
            {"type": "tool_use", "id": "call-1", "name": "lookup", "input": {"id": 1}},
        ],
    }
    result = payload["messages"][1]["content"][0]
    assert result["type"] == "tool_result"
    assert result["tool_use_id"] == "call-1"
    assert result["is_error"] is True


def test_anthropic_non_stream_response_decodes_blocks_usage_and_continuation() -> None:
    response = AnthropicMessagesAdapter().parse_response(
        request(),
        freeze_json_object(
            {
                "id": "msg-1",
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "reasoning", "signature": "sig"},
                    {"type": "text", "text": "answer"},
                    {"type": "tool_use", "id": "call-1", "name": "lookup", "input": {"id": 1}},
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 10, "output_tokens": 4},
            }
        ),
    )

    assert response.message.content == "answer"
    assert response.message.tool_calls[0].arguments["id"] == 1
    assert response.usage["input_tokens"] == 10
    assert response.provider_request_id == "msg-1"
    assert response.finish_reason == "tool_calls"
    assert response.message.continuation == ModelContinuation(
        provider="anthropic",
        protocol="anthropic_messages",
        data={
            "version": 1,
            "blocks": [
                {"type": "thinking", "thinking": "reasoning", "signature": "sig"},
                {"type": "text", "start": 0, "end": 6},
                {"type": "tool_use", "index": 0},
            ],
        },
    )


@pytest.mark.parametrize("invalid", [True, -1, "1"])
def test_anthropic_response_rejects_invalid_cache_usage_counters(
    invalid: object,
) -> None:
    with pytest.raises(ModelProviderError) as raised:
        AnthropicMessagesAdapter().parse_response(
            request(),
            freeze_json_object(
                {
                    "id": "msg-1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "answer"}],
                    "stop_reason": "end_turn",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 4,
                        "cache_read_input_tokens": invalid,
                    },
                }
            ),
        )

    assert raised.value.kind is ModelErrorKind.INVALID_RESPONSE


@pytest.mark.asyncio
async def test_anthropic_client_joins_endpoint_sets_headers_and_lists_models() -> None:
    seen: list[httpx.Request] = []

    async def handler(raw: httpx.Request) -> httpx.Response:
        seen.append(raw)
        if raw.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={"data": [{"id": "claude-opus-5", "created_at": "2026-01-01T00:00:00Z"}]},
            )
        return httpx.Response(200, json={"type": "message"})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AnthropicMessagesClient(
        base_url="https://api.deepseek.com/anthropic/",
        api_key="secret",
        headers={"x-app": "test"},
        client=http,
    )
    try:
        await client.invoke(anthropic_entry(provider="deepseek").spec, freeze_json_object({}))
        models = await client.models.list()
    finally:
        await client.aclose()

    assert seen[0].url.path == "/anthropic/v1/messages"
    assert seen[1].url.path == "/anthropic/v1/models"
    assert seen[0].headers["x-api-key"] == "secret"
    assert seen[0].headers["anthropic-version"] == "2023-06-01"
    assert seen[0].headers["x-app"] == "test"
    assert models[0].id == "claude-opus-5"
    assert models[0].created is None
    assert models[0].owned_by == "anthropic"
    assert not http.is_closed
    await http.aclose()


@pytest.mark.parametrize(
    "options",
    [
        {"thinking": {"type": "enabled", "budget_tokens": 512}},
        {"thinking": {"type": "enabled", "budget_tokens": 4096}},
        {"thinking": {"type": "adaptive", "display": "full"}},
        {"output_config": {"format": {}}},
        {"service_tier": "priority"},
        {"stop_sequences": [""]},
        {"headers": {"authorization": "secret"}},
        {"unknown": True},
    ],
)
def test_anthropic_provider_options_fail_closed(options: dict[str, object]) -> None:
    with pytest.raises(ModelProviderError) as raised:
        AnthropicMessagesAdapter().build_request(
            request(entry=anthropic_entry(options=options))
        )
    assert raised.value.kind is ModelErrorKind.INVALID_REQUEST


def test_anthropic_http_error_mapping_is_closed() -> None:
    error = AnthropicMessagesAdapter().normalize_error(
        ModelProviderError(
            ModelErrorKind.UNAVAILABLE,
            "unavailable",
            reason_code=ModelFailureReason.PROVIDER_UNAVAILABLE,
            http_status=529,
        )
    )
    assert error is ModelErrorKind.UNAVAILABLE


def test_anthropic_stream_decoder_reduces_indexed_blocks_and_continuation() -> None:
    decoder = AnthropicMessagesAdapter().create_stream_decoder(request())
    payloads = [
        {"type": "message_start", "message": {"id": "msg-1", "usage": {"input_tokens": 10}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "reasoning"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "answer"}},
        {"type": "content_block_stop", "index": 1},
        {"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "call-1", "name": "lookup", "input": {}}},
        {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": '{"id":1}'}},
        {"type": "content_block_stop", "index": 2},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 4}},
        {"type": "message_stop"},
    ]

    parts = [
        part
        for payload in payloads
        for part in decoder.feed(freeze_json_object(payload))
    ]

    assert decoder.finish() == ()
    assert [part.kind for part in parts] == [
        "reasoning",
        "text",
        "tool_call",
        "tool_call",
        "usage",
        "continuation",
        "finish",
    ]
    continuation = parts[-2].data
    assert continuation["provider"] == "anthropic"
    assert continuation["data"]["blocks"] == (
        freeze_json_object({"type": "thinking", "thinking": "reasoning", "signature": "sig"}),
        freeze_json_object({"type": "text", "start": 0, "end": 6}),
        freeze_json_object({"type": "tool_use", "index": 0}),
    )
    assert parts[-1].data["finish_reason"] == "tool_calls"
    assert parts[-1].data["provider_request_id"] == "msg-1"


def test_anthropic_stream_rejects_missing_signature_and_premature_eof() -> None:
    decoder = AnthropicMessagesAdapter().create_stream_decoder(request())
    decoder.feed(
        freeze_json_object(
            {"type": "message_start", "message": {"id": "msg-1", "usage": {}}}
        )
    )
    decoder.feed(
        freeze_json_object(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": "reasoning"},
            }
        )
    )

    with pytest.raises(ModelProviderError) as missing_signature:
        decoder.feed(
            freeze_json_object({"type": "content_block_stop", "index": 0})
        )
    assert missing_signature.value.reason_code is ModelFailureReason.STREAM_EVENT_INVALID

    incomplete = AnthropicMessagesAdapter().create_stream_decoder(request())
    with pytest.raises(ModelProviderError) as eof:
        incomplete.finish()
    assert eof.value.reason_code is ModelFailureReason.STREAM_INCOMPLETE


def test_anthropic_compatible_stream_preserves_unsigned_thinking() -> None:
    compatible_request = request(entry=anthropic_entry(provider="deepseek"))
    decoder = AnthropicMessagesAdapter().create_stream_decoder(compatible_request)
    payloads = [
        {"type": "message_start", "message": {"id": "msg-1", "usage": {}}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "reasoning"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "text", "text": "answer"},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {},
        },
        {"type": "message_stop"},
    ]

    parts = [
        part
        for payload in payloads
        for part in decoder.feed(freeze_json_object(payload))
    ]

    continuation = next(part for part in parts if part.kind == "continuation")
    assert continuation.data["data"]["blocks"][0] == freeze_json_object(
        {"type": "thinking", "thinking": "reasoning"}
    )


def test_anthropic_compatible_non_stream_replays_unsigned_thinking() -> None:
    compatible_request = request(entry=anthropic_entry(provider="deepseek"))
    adapter = AnthropicMessagesAdapter()
    response = adapter.parse_response(
        compatible_request,
        freeze_json_object(
            {
                "id": "msg-1",
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "reasoning"},
                    {"type": "text", "text": "answer"},
                ],
                "stop_reason": "end_turn",
                "usage": {},
            }
        ),
    )

    replay = adapter.build_request(
        request(entry=anthropic_entry(provider="deepseek"), message=response.message)
    ).to_dict()

    assert replay["messages"][0]["content"][0] == {
        "type": "thinking",
        "thinking": "reasoning",
    }


def test_anthropic_matching_continuation_rebuilds_exact_assistant_layout() -> None:
    adapter = AnthropicMessagesAdapter()
    response = adapter.parse_response(
        request(),
        freeze_json_object(
            {
                "id": "msg-1",
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "reasoning", "signature": "sig"},
                    {"type": "text", "text": "answer"},
                    {"type": "tool_use", "id": "call-1", "name": "lookup", "input": {"id": 1}},
                ],
                "stop_reason": "tool_use",
                "usage": {},
            }
        ),
    )

    payload = adapter.build_request(request(message=response.message)).to_dict()

    assert payload["messages"][0]["content"] == [
        {"type": "thinking", "thinking": "reasoning", "signature": "sig"},
        {"type": "text", "text": "answer"},
        {"type": "tool_use", "id": "call-1", "name": "lookup", "input": {"id": 1}},
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "x"}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "image"}},
        {"type": "message_delta", "delta": {"stop_reason": "pause_turn"}, "usage": {}},
    ],
)
def test_anthropic_stream_rejects_unknown_or_out_of_order_semantics(
    payload: dict[str, object],
) -> None:
    decoder = AnthropicMessagesAdapter().create_stream_decoder(request())
    with pytest.raises(ModelProviderError) as raised:
        decoder.feed(freeze_json_object(payload))
    assert raised.value.reason_code is ModelFailureReason.STREAM_EVENT_INVALID


@pytest.mark.asyncio
async def test_anthropic_http_stream_reaches_common_invoker_reducer() -> None:
    events = [
        {"type": "message_start", "message": {"id": "msg-1", "usage": {"input_tokens": 2}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "reason"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "answer"}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    ]

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text="".join(f"data: {json.dumps(event)}\n\n" for event in events),
            headers={"content-type": "text/event-stream"},
        )

    entry = anthropic_entry()
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AnthropicMessagesClient(base_url="https://api.anthropic.com", client=http)
    invoker = DefaultModelInvoker(
        adapters={entry.spec.protocol: AnthropicMessagesAdapter()},
        clients={entry.name: client},
    )
    execution = invoker.execute(
        model_group=ModelGroup("assistant", (entry,)),
        retry_policy=RetryPolicy(),
        generation=GenerationConfig(max_output_tokens=64),
        message=UserMessage(content="question"),
        context=Context(),
    )

    result = await execution.result()
    async with execution.subscribe() as subscription:
        public_events = [event async for event in subscription]

    assert result.message.content == "answer"
    assert result.message.continuation is not None
    assert result.usage["total_tokens"] == 3
    assert all("continuation" not in event.kind for event in public_events)
    await invoker.aclose()
    await http.aclose()
