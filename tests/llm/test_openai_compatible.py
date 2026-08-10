from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from pygent import (
    Context,
    ToolDefinition,
    ToolMessage,
    ToolResult,
    UserMessage,
)
from pygent.core import freeze_json_object
from pygent.llm import (
    GenerationConfig,
    ModelErrorKind,
    ModelInfo,
    ModelProviderError,
    ModelProviderRequest,
    ModelRoute,
    OpenAICompatibleAdapter,
    OpenAICompatibleClient,
)
from pygent.llm import openai_compatible as openai_compatible_module


def _request(
    *,
    generation: GenerationConfig | None = None,
    tools: tuple[ToolDefinition, ...] = (),
):
    return ModelProviderRequest(
        route=ModelRoute("main", "openai", "gpt-test"),
        message=UserMessage(content="hello"),
        context=Context(system_prompt="be brief"),
        generation=generation or GenerationConfig(),
        tools=tools,
    )


def test_openai_codec_parses_usage_tools_and_structured_output():
    adapter = OpenAICompatibleAdapter()
    request = _request(
        generation=GenerationConfig(
            response_schema={
                "type": "object",
                "required": ["answer"],
                "properties": {"answer": {"type": "string"}},
            }
        )
    )
    payload = adapter.build_request(request).to_dict()
    assert payload["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello"},
    ]
    response = adapter.parse_response(
        request,
        # Tool calls are legal alongside structured text on compatible APIs.
        adapter_request_payload(
            content='{"answer":"ok"}',
            usage={"prompt_tokens": 3, "completion_tokens": 2},
        ),
    )
    assert response.message.content == '{"answer":"ok"}'
    assert response.usage["input_tokens"] == 3
    assert response.usage["output_tokens"] == 2


def test_tool_result_error_classification_is_visible_to_the_model() -> None:
    request = ModelProviderRequest(
        route=ModelRoute("main", "openai", "gpt-test"),
        message=ToolMessage(
            results=(
                ToolResult(
                    call_id="grep-1",
                    name="grep",
                    status="rejected",
                    error="Additional properties are not allowed ('glob' was unexpected)",
                    error_kind="validation_error",
                    error_code="invalid_arguments",
                    side_effect_committed=False,
                    tool_id="standard.files.grep",
                    tool_version="3.0.0",
                ),
                ToolResult(
                    call_id="read-1",
                    name="read",
                    status="succeeded",
                    output="contents",
                    side_effect_committed=True,
                ),
            )
        ),
        context=Context(),
        generation=GenerationConfig(),
    )

    payload = OpenAICompatibleAdapter().build_request(request).to_dict()
    content = json.loads(payload["messages"][0]["content"])

    assert content == {
        "status": "rejected",
        "output": None,
        "error": "Additional properties are not allowed ('glob' was unexpected)",
        "error_kind": "validation_error",
        "error_code": "invalid_arguments",
    }
    success_content = json.loads(payload["messages"][1]["content"])
    assert success_content == {
        "status": "succeeded",
        "output": "contents",
        "error": None,
    }


def test_provider_raw_diagnostics_are_not_projected_through_usage() -> None:
    canaries = (
        "synthetic-api-key-canary",
        "https://synthetic-endpoint.invalid/private",
        "provider-internal-stack-canary",
    )
    adapter = OpenAICompatibleAdapter()
    request = _request()
    response = adapter.parse_response(
        request,
        freeze_json_object(
            {
                "id": "safe-request-id",
                "choices": [{"message": {"content": "safe answer"}}],
                "provider_debug": {"secret": canaries[0], "trace": canaries[2]},
                "usage": {
                    "prompt_tokens": 3,
                    "total_tokens": 4,
                    "debug": canaries[0],
                    "endpoint": canaries[1],
                    "internal": {"trace": canaries[2]},
                },
            }
        ),
    )
    stream_parts = adapter.parse_stream_events(
        request,
        freeze_json_object(
            {
                "usage": {
                    "completion_tokens": 1,
                    "debug": canaries[0],
                    "endpoint": canaries[1],
                    "internal": canaries[2],
                }
            }
        ),
    )

    assert response.message.content == "safe answer"
    assert dict(response.usage) == {"input_tokens": 3, "total_tokens": 4}
    assert len(stream_parts) == 1
    assert dict(stream_parts[0].data) == {"output_tokens": 1}
    public = repr((response.message, response.usage, stream_parts))
    assert all(canary not in public for canary in canaries)


def test_stream_accepts_openai_usage_only_chunk_with_empty_choices() -> None:
    parts = OpenAICompatibleAdapter().parse_stream_events(
        _request(),
        freeze_json_object(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            }
        ),
    )

    assert len(parts) == 1
    assert parts[0].kind == "usage"
    assert dict(parts[0].data) == {
        "input_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
    }


def test_optional_generation_fields_are_omitted_but_zero_is_preserved():
    adapter = OpenAICompatibleAdapter()
    default_payload = adapter.build_request(_request()).to_dict()
    assert "temperature" not in default_payload
    assert "max_tokens" not in default_payload

    explicit_payload = adapter.build_request(
        _request(generation=GenerationConfig(temperature=0, max_output_tokens=1))
    ).to_dict()
    assert explicit_payload["temperature"] == 0
    assert explicit_payload["max_tokens"] == 1


def test_deployment_static_projections_are_reused_without_sharing_public_mutation():
    tool = ToolDefinition(
        name="weather.lookup",
        description="weather",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}},
    )
    generation = GenerationConfig(
        response_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
        }
    )
    request = _request(generation=generation, tools=(tool,))
    adapter = OpenAICompatibleAdapter()
    openai_compatible_module._tool_projection.cache_clear()
    openai_compatible_module._response_format_projection.cache_clear()

    first = adapter.build_request(request).to_dict()
    first["tools"][0]["function"]["description"] = "mutated"
    first["response_format"]["json_schema"]["name"] = "mutated"
    second = adapter.build_request(request).to_dict()

    assert second["tools"][0]["function"]["description"] == "weather"
    assert second["response_format"]["json_schema"]["name"] == "response"
    assert openai_compatible_module._tool_projection.cache_info().hits == 1
    assert openai_compatible_module._response_format_projection.cache_info().hits == 1


def test_portable_tool_name_and_named_choice_round_trip_through_wire_mapping():
    tool = ToolDefinition(
        name="weather.lookup",
        description="weather",
        parameters={"type": "object"},
    )
    adapter = OpenAICompatibleAdapter()
    request = _request(
        generation=GenerationConfig(tool_choice="weather.lookup"),
        tools=(tool,),
    )
    payload = adapter.build_request(request).to_dict()
    wire_name = payload["tools"][0]["function"]["name"]

    assert wire_name != tool.name
    assert len(wire_name) <= 64
    assert payload["tool_choice"] == {
        "type": "function",
        "function": {"name": wire_name},
    }
    response = adapter.parse_response(
        request,
        freeze_json_object(
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call-weather",
                                    "type": "function",
                                    "function": {
                                        "name": wire_name,
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        ),
    )
    assert response.message.tool_calls[0].name == tool.name


def test_tool_choice_cannot_reference_a_non_visible_tool():
    adapter = OpenAICompatibleAdapter()
    with pytest.raises(ModelProviderError) as raised:
        adapter.build_request(
            _request(generation=GenerationConfig(tool_choice="missing"))
        )
    assert raised.value.kind is ModelErrorKind.INVALID_REQUEST


@pytest.mark.parametrize("arguments", [{"value": 2}, '{"value":2}'])
def test_tool_call_arguments_accept_compatible_object_or_json_string(arguments):
    from pygent.core import freeze_json_object

    adapter = OpenAICompatibleAdapter()
    response = adapter.parse_response(
        _request(),
        freeze_json_object(
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "double",
                                        "arguments": arguments,
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        ),
    )
    assert response.message.tool_calls[0].arguments.to_dict() == {"value": 2}


def test_structured_output_is_validated():
    adapter = OpenAICompatibleAdapter()
    request = _request(
        generation=GenerationConfig(
            response_schema={"type": "object", "required": ["answer"]}
        )
    )
    with pytest.raises(ModelProviderError) as raised:
        adapter.parse_response(request, adapter_request_payload(content="[]"))
    assert raised.value.kind is ModelErrorKind.INVALID_RESPONSE


@pytest.mark.asyncio
async def test_http_and_sse_transport_use_openai_compatible_endpoint():
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = await request.aread()
        if b'"stream":true' in body:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=(
                    b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
                    b"data: [DONE]\n\n"
                ),
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "hi"}}]},
        )

    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(
        base_url="https://models.example/v1", transport=transport
    )
    client = OpenAICompatibleClient(
        base_url="https://models.example/v1", client=http_client
    )
    route = ModelRoute("main", "openai", "gpt-test")
    payload = OpenAICompatibleAdapter().build_request(_request())
    full = await client.invoke(route, payload)
    streamed = [item async for item in client.stream(route, payload)]
    assert full["choices"]
    assert streamed[-1]["done"] is True
    assert all(request.url.path == "/v1/chat/completions" for request in requests)
    await client.aclose()
    # An injected client remains owned by the caller.
    assert not http_client.is_closed
    await http_client.aclose()


@pytest.mark.asyncio
async def test_owned_client_close_is_idempotent_and_blocks_reuse():
    client = OpenAICompatibleClient(base_url="https://models.example/v1")
    owned_clients = client._clients
    assert len(owned_clients) == 8
    assert [client._next_http_client() for _ in range(9)] == [
        *owned_clients,
        owned_clients[0],
    ]
    await client.aclose()
    await client.aclose()
    assert all(http_client.is_closed for http_client in owned_clients)
    with pytest.raises(RuntimeError, match="closed"):
        await client.invoke(
            ModelRoute("main", "openai", "gpt-test"),
            OpenAICompatibleAdapter().build_request(_request()),
        )


@pytest.mark.asyncio
async def test_injected_http_client_is_not_sharded_or_closed():
    http_client = httpx.AsyncClient()
    client = OpenAICompatibleClient(
        base_url="https://models.example/v1", client=http_client
    )

    assert client._clients == (http_client,)
    assert client._next_http_client() is http_client
    assert client._next_http_client() is http_client
    await client.aclose()
    assert not http_client.is_closed
    await http_client.aclose()


@pytest.mark.asyncio
async def test_model_catalog_lists_credential_visible_models_in_provider_order():
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "id": "model-b",
                        "object": "model",
                        "created": 20,
                        "owned_by": "provider",
                    },
                    {"id": "model-a"},
                ],
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleClient(
        base_url="https://models.example/v1/",
        api_key="catalog-token",
        client=http_client,
    )

    models = await client.models.list()

    assert models == (
        ModelInfo("model-b", created=20, owned_by="provider"),
        ModelInfo("model-a"),
    )
    assert requests[0].url.path == "/v1/models"
    assert requests[0].headers["authorization"] == "Bearer catalog-token"
    await client.aclose()
    assert not http_client.is_closed
    await http_client.aclose()


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"object": "model", "data": []},
        {"object": "list", "data": {}},
        {"object": "list", "data": ["model-a"]},
        {"object": "list", "data": [{"id": ""}]},
        {"object": "list", "data": [{"id": "model-a", "created": True}]},
        {"object": "list", "data": [{"id": "model-a", "owned_by": ""}]},
        {"object": "list", "data": [{"id": "model-a", "object": "list"}]},
        {
            "object": "list",
            "data": [{"id": "model-a"}, {"id": "model-a"}],
        },
    ],
)
@pytest.mark.asyncio
async def test_model_catalog_rejects_invalid_or_ambiguous_responses(payload):
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json=payload)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleClient(
        base_url="https://models.example/v1", client=http_client
    )
    with pytest.raises(ModelProviderError) as raised:
        await client.models.list()
    assert raised.value.kind is ModelErrorKind.INVALID_RESPONSE
    await http_client.aclose()


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (401, ModelErrorKind.AUTHENTICATION),
        (429, ModelErrorKind.RATE_LIMIT),
        (500, ModelErrorKind.UNAVAILABLE),
    ],
)
@pytest.mark.asyncio
async def test_model_catalog_normalizes_http_errors_without_response_leakage(
    status, kind
):
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(status, text="synthetic-private-provider-canary")

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleClient(
        base_url="https://models.example/v1", client=http_client
    )
    with pytest.raises(ModelProviderError) as raised:
        await client.models.list()
    assert raised.value.kind is kind
    assert "synthetic-private-provider-canary" not in str(raised.value)
    await http_client.aclose()


@pytest.mark.parametrize("timeout", (0, -1, float("inf"), float("nan"), True))
@pytest.mark.asyncio
async def test_model_catalog_rejects_invalid_timeouts(timeout):
    client = OpenAICompatibleClient(base_url="https://models.example/v1")
    with pytest.raises(ValueError, match="timeout"):
        await client.models.list(timeout=timeout)
    await client.aclose()


@pytest.mark.asyncio
async def test_model_catalog_normalizes_transport_timeout_and_propagates_cancel():
    waiting = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.headers.get("x-mode") == "timeout":
            raise httpx.ReadTimeout("private timeout details", request=request)
        waiting.set()
        await asyncio.Event().wait()
        raise AssertionError("cancelled catalog request resumed")

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    timeout_client = OpenAICompatibleClient(
        base_url="https://models.example/v1",
        headers={"x-mode": "timeout"},
        client=http_client,
    )
    with pytest.raises(ModelProviderError) as raised:
        await timeout_client.models.list()
    assert raised.value.kind is ModelErrorKind.TIMEOUT
    assert "private timeout details" not in str(raised.value)

    cancel_client = OpenAICompatibleClient(
        base_url="https://models.example/v1", client=http_client
    )
    task = asyncio.create_task(cancel_client.models.list(timeout=None))
    await waiting.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await http_client.aclose()


@pytest.mark.asyncio
async def test_closed_client_rejects_model_catalog_queries():
    client = OpenAICompatibleClient(base_url="https://models.example/v1")
    catalog = client.models
    await client.aclose()
    with pytest.raises(RuntimeError, match="closed"):
        await catalog.list()


def adapter_request_payload(*, content: str, usage: dict[str, int] | None = None):
    from pygent.core import freeze_json_object

    return freeze_json_object(
        {
            "id": "req-safe",
            "choices": [{"message": {"content": content}}],
            "usage": usage or {},
        }
    )
