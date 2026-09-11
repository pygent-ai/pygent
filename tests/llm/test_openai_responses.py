from __future__ import annotations

import asyncio
from dataclasses import replace

import httpx
import pytest

from pygent import (
    AIMessage,
    BuiltinModelProtocol,
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
    ModelErrorKind,
    ModelFailureReason,
    ModelProviderError,
    ModelProviderRequest,
    OpenAIResponsesAdapter,
    OpenAIResponsesClient,
)
from tests.support.model_specs import model_entry


def _request(
    *,
    message=None,
    context: Context | None = None,
    generation: GenerationConfig | None = None,
    tools: tuple[ToolDefinition, ...] = (),
    options: dict[str, object] | None = None,
) -> ModelProviderRequest:
    entry = model_entry("main", "openai", "gpt-5.4")
    spec = replace(
        entry.spec,
        protocol="openai_responses",
        provider_options=options or {},
    )
    return ModelProviderRequest(
        model_key=entry.name,
        model=spec,
        message=message or UserMessage(content="question"),
        context=context or Context(),
        generation=generation or GenerationConfig(max_output_tokens=64),
        tools=tools,
    )


def test_builtin_protocol_has_precise_responses_identifier() -> None:
    assert BuiltinModelProtocol.OPENAI_RESPONSES == "openai_responses"


def test_responses_request_projects_input_tools_schema_and_reasoning() -> None:
    tool = ToolDefinition(
        name="lookup", description="Lookup", parameters={"type": "object"}
    )
    call = ToolCall(call_id="call-1", name="lookup", arguments={"id": 1})
    request = _request(
        message=ToolMessage(
            results=(
                ToolResult(
                    call_id="call-1", name="lookup", status="succeeded", output="ok"
                ),
            )
        ),
        context=Context(
            system_prompt="system",
            messages=(AIMessage(content="checking", tool_calls=(call,)),),
        ),
        generation=GenerationConfig(
            max_output_tokens=128,
            tool_choice="lookup",
            response_schema={"type": "object", "properties": {}},
            response_schema_name="answer",
        ),
        tools=(tool,),
        options={"reasoning": {"effort": "low", "summary": "auto"}},
    )

    payload = OpenAIResponsesAdapter().build_request(request).to_dict()
    assert payload["model"] == "gpt-5.4"
    assert payload["instructions"] == "system"
    assert payload["input"][0]["role"] == "assistant"
    assert payload["input"][1]["type"] == "function_call"
    assert payload["input"][2] == {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": "ok",
    }
    assert payload["tools"][0]["name"] == "lookup"
    assert payload["tool_choice"] == {"type": "function", "name": "lookup"}
    assert payload["text"]["format"]["type"] == "json_schema"
    assert payload["reasoning"] == {"effort": "low", "summary": "auto"}


@pytest.mark.parametrize("field", ["api_key", "headers", "base_url"])
def test_responses_provider_options_reject_connection_fields(field: str) -> None:
    with pytest.raises(ModelProviderError) as raised:
        OpenAIResponsesAdapter().build_request(
            _request(options={field: "must-not-be-persisted"})
        )
    assert raised.value.kind is ModelErrorKind.INVALID_REQUEST


def test_responses_parse_preserves_reasoning_and_function_call() -> None:
    response = OpenAIResponsesAdapter().parse_response(
        _request(),
        freeze_json_object(
            {
                "id": "resp-1",
                "status": "completed",
                "output": [
                    {
                        "id": "rs-1",
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": "reason"}],
                    },
                    {
                        "id": "msg-1",
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "answer", "annotations": []}
                        ],
                    },
                    {
                        "id": "fc-1",
                        "type": "function_call",
                        "call_id": "call-1",
                        "name": "lookup",
                        "arguments": '{"id":1}',
                    },
                ],
                "usage": {
                    "input_tokens": 3,
                    "output_tokens": 2,
                    "total_tokens": 5,
                    "output_tokens_details": {"reasoning_tokens": 1},
                },
            }
        ),
    )
    assert response.message.content == "answer"
    assert response.message.tool_calls[0].arguments["id"] == 1
    assert response.finish_reason == "tool_calls"
    assert response.usage["reasoning_tokens"] == 1
    assert response.message.continuation == ModelContinuation(
        provider="openai",
        protocol="openai_responses",
        data={
            "version": 1,
            "items": [
                {
                    "id": "rs-1",
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "reason"}],
                }
            ],
        },
    )


def test_responses_structured_output_rejects_schema_mismatch() -> None:
    request = _request(
        generation=GenerationConfig(
            response_schema={
                "type": "object",
                "required": ["answer"],
                "properties": {"answer": {"type": "string"}},
            }
        )
    )
    with pytest.raises(ModelProviderError) as raised:
        OpenAIResponsesAdapter().parse_response(
            request,
            freeze_json_object(
                {
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": '{"wrong":1}'}],
                        }
                    ],
                }
            ),
        )
    assert raised.value.reason_code is ModelFailureReason.GENERATION_SCHEMA_INVALID


def test_responses_continuation_is_replayed_before_visible_assistant_items() -> None:
    continuation = ModelContinuation(
        provider="openai",
        protocol="openai_responses",
        data={"version": 1, "items": [{"id": "rs-1", "type": "reasoning"}]},
    )
    payload = OpenAIResponsesAdapter().build_request(
        _request(
            context=Context(
                messages=(AIMessage(content="answer", continuation=continuation),)
            )
        )
    ).to_dict()
    assert payload["input"][0] == {"id": "rs-1", "type": "reasoning"}
    assert payload["input"][1]["role"] == "assistant"


def test_responses_stream_decodes_text_reasoning_usage_and_finish() -> None:
    decoder = OpenAIResponsesAdapter().create_stream_decoder(_request())
    parts = []
    for event in (
        {"type": "response.reasoning_summary_text.delta", "delta": "reason"},
        {"type": "response.output_text.delta", "delta": "answer"},
        {
            "type": "response.completed",
            "response": {
                "id": "resp-1",
                "status": "completed",
                "output": [],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            },
        },
    ):
        parts.extend(decoder.feed(freeze_json_object(event)))
    decoder.finish()
    assert [part.kind for part in parts] == ["reasoning", "text", "usage", "finish"]


@pytest.mark.asyncio
async def test_responses_client_uses_exact_endpoint_and_secret_safe_state() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"status": "completed", "output": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = OpenAIResponsesClient(
            base_url="https://api.example/v1",
            api_key="very-secret",
            client=http,
        )
        await client.invoke(_request().model, freeze_json_object({"model": "gpt"}))
        assert requests[0].url == httpx.URL("https://api.example/v1/responses")
        assert requests[0].headers["authorization"] == "Bearer very-secret"
        assert "very-secret" not in repr(client)


@pytest.mark.asyncio
async def test_responses_client_maps_status_and_propagates_cancellation() -> None:
    async def unavailable(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"type": "server_error"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(unavailable)) as http:
        client = OpenAIResponsesClient(base_url="https://api.example/v1", client=http)
        with pytest.raises(ModelProviderError) as raised:
            await client.invoke(_request().model, freeze_json_object({}))
        assert raised.value.kind is ModelErrorKind.UNAVAILABLE

    async def cancelled(request: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError

    async with httpx.AsyncClient(transport=httpx.MockTransport(cancelled)) as http:
        client = OpenAIResponsesClient(base_url="https://api.example/v1", client=http)
        with pytest.raises(asyncio.CancelledError):
            await client.invoke(_request().model, freeze_json_object({}))
