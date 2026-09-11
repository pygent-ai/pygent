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
    GeminiGenerateContentAdapter,
    GeminiGenerateContentClient,
    ModelErrorKind,
    ModelFailureReason,
    ModelProviderError,
    ModelProviderRequest,
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
    entry = model_entry("main", "google", "gemini-3.7-flash")
    spec = replace(
        entry.spec,
        protocol="gemini_generate_content",
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


def test_builtin_protocol_has_precise_gemini_identifier() -> None:
    assert BuiltinModelProtocol.GEMINI_GENERATE_CONTENT == "gemini_generate_content"


def test_gemini_request_projects_roles_tools_results_schema_and_thinking() -> None:
    tool = ToolDefinition(
        name="lookup", description="Lookup", parameters={"type": "object"}
    )
    call = ToolCall(call_id="gemini-call-0", name="lookup", arguments={"id": 1})
    request = _request(
        message=ToolMessage(
            results=(
                ToolResult(
                    call_id="gemini-call-0",
                    name="lookup",
                    status="succeeded",
                    output={"value": "ok"},
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
        ),
        tools=(tool,),
        options={
            "thinking_config": {"include_thoughts": True, "thinking_budget": 64}
        },
    )
    payload = GeminiGenerateContentAdapter().build_request(request).to_dict()
    assert payload["systemInstruction"] == {"parts": [{"text": "system"}]}
    assert payload["contents"][0]["role"] == "model"
    assert payload["contents"][0]["parts"][1]["functionCall"]["name"] == "lookup"
    assert payload["contents"][1]["parts"][0]["functionResponse"] == {
        "name": "lookup",
        "response": {"value": "ok"},
    }
    assert payload["tools"][0]["functionDeclarations"][0]["name"] == "lookup"
    assert payload["toolConfig"]["functionCallingConfig"] == {
        "mode": "ANY",
        "allowedFunctionNames": ["lookup"],
    }
    generation = payload["generationConfig"]
    assert generation["responseMimeType"] == "application/json"
    assert generation["responseJsonSchema"] == {"type": "object", "properties": {}}
    assert generation["thinkingConfig"] == {
        "includeThoughts": True,
        "thinkingBudget": 64,
    }


def test_gemini_parse_preserves_thought_signature_and_function_call() -> None:
    response = GeminiGenerateContentAdapter().parse_response(
        _request(),
        freeze_json_object(
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {
                                    "text": "reason",
                                    "thought": True,
                                    "thoughtSignature": "sig",
                                },
                                {"text": "answer"},
                                {"functionCall": {"name": "lookup", "args": {"id": 1}}},
                            ],
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 3,
                    "candidatesTokenCount": 2,
                    "totalTokenCount": 5,
                    "thoughtsTokenCount": 1,
                },
            }
        ),
    )
    assert response.message.content == "answer"
    assert response.message.tool_calls[0] == ToolCall(
        call_id="gemini-call-0", name="lookup", arguments={"id": 1}
    )
    assert response.usage["reasoning_tokens"] == 1
    assert response.finish_reason == "tool_calls"
    assert response.message.continuation == ModelContinuation(
        provider="google",
        protocol="gemini_generate_content",
        data={
            "version": 1,
            "parts": [
                {"text": "reason", "thought": True, "thoughtSignature": "sig"}
            ],
        },
    )


def test_gemini_structured_output_rejects_schema_mismatch() -> None:
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
        GeminiGenerateContentAdapter().parse_response(
            request,
            freeze_json_object(
                {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [{"text": '{"wrong":1}'}],
                            },
                            "finishReason": "STOP",
                        }
                    ]
                }
            ),
        )
    assert raised.value.reason_code is ModelFailureReason.GENERATION_SCHEMA_INVALID


def test_gemini_continuation_is_replayed_with_visible_assistant_text() -> None:
    continuation = ModelContinuation(
        provider="google",
        protocol="gemini_generate_content",
        data={
            "version": 1,
            "parts": [{"thought": True, "thoughtSignature": "sig", "text": "reason"}],
        },
    )
    payload = GeminiGenerateContentAdapter().build_request(
        _request(
            context=Context(
                messages=(AIMessage(content="answer", continuation=continuation),)
            )
        )
    ).to_dict()
    assert payload["contents"][0]["parts"] == [
        {"thought": True, "thoughtSignature": "sig", "text": "reason"},
        {"text": "answer"},
    ]


def test_gemini_stream_decodes_thinking_text_usage_and_finish() -> None:
    decoder = GeminiGenerateContentAdapter().create_stream_decoder(_request())
    parts = []
    for chunk in (
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "reason", "thought": True}],
                    }
                }
            ]
        },
        {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": "answer"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 1,
                "candidatesTokenCount": 1,
                "totalTokenCount": 2,
            },
        },
    ):
        parts.extend(decoder.feed(freeze_json_object(chunk)))
    decoder.finish()
    assert [part.kind for part in parts] == ["reasoning", "text", "usage", "finish"]


def test_gemini_safety_finish_is_content_policy_error() -> None:
    with pytest.raises(ModelProviderError) as raised:
        GeminiGenerateContentAdapter().parse_response(
            _request(),
            freeze_json_object(
                {
                    "candidates": [
                        {
                            "content": {"role": "model", "parts": []},
                            "finishReason": "SAFETY",
                        }
                    ]
                }
            ),
        )
    assert raised.value.reason_code is ModelFailureReason.CONTENT_POLICY_REJECTED


@pytest.mark.asyncio
async def test_gemini_client_uses_model_endpoint_and_header_auth() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"candidates": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = GeminiGenerateContentClient(
            base_url="https://generativelanguage.example/v1beta",
            api_key="very-secret",
            client=http,
        )
        await client.invoke(_request().model, freeze_json_object({"contents": []}))
        assert requests[0].url == httpx.URL(
            "https://generativelanguage.example/v1beta/models/"
            "gemini-3.7-flash:generateContent"
        )
        assert requests[0].headers["x-goog-api-key"] == "very-secret"
        assert "very-secret" not in str(requests[0].url)
        assert "very-secret" not in repr(client)


@pytest.mark.asyncio
async def test_gemini_client_maps_status_and_propagates_cancellation() -> None:
    async def unavailable(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"status": "UNAVAILABLE"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(unavailable)) as http:
        client = GeminiGenerateContentClient(base_url="https://api.example/v1beta", client=http)
        with pytest.raises(ModelProviderError) as raised:
            await client.invoke(_request().model, freeze_json_object({}))
        assert raised.value.kind is ModelErrorKind.UNAVAILABLE

    async def cancelled(request: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError

    async with httpx.AsyncClient(transport=httpx.MockTransport(cancelled)) as http:
        client = GeminiGenerateContentClient(base_url="https://api.example/v1beta", client=http)
        with pytest.raises(asyncio.CancelledError):
            await client.invoke(_request().model, freeze_json_object({}))
