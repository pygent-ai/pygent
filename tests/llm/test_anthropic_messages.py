from __future__ import annotations

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
    ModelEntry,
    ModelErrorKind,
    ModelFailureReason,
    ModelProviderError,
    ModelProviderRequest,
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
