from __future__ import annotations

import asyncio
import json
import ssl

import httpx
import pytest

from pygent import (
    Context,
    ToolDefinition,
    ToolMessage,
    ToolResult,
    UserMessage,
)
from pygent.core import FrozenJsonObject, freeze_json_object
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
async def test_completed_sse_response_reuses_http1_connection(
) -> None:
    connection_count = 0

    async def serve_connection(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        nonlocal connection_count
        connection_count += 1
        try:
            while True:
                try:
                    head = await reader.readuntil(b"\r\n\r\n")
                except (asyncio.IncompleteReadError, ConnectionError):
                    return
                headers = {}
                for raw_line in head.split(b"\r\n")[1:]:
                    if b":" in raw_line:
                        name, value = raw_line.split(b":", 1)
                        headers[name.lower()] = value.strip()
                content_length = int(headers.get(b"content-length", b"0"))
                if content_length:
                    await reader.readexactly(content_length)
                body = (
                    b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
                    b"data: [DONE]\n\n"
                )
                writer.write(
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: text/event-stream\r\n"
                    + f"Content-Length: {len(body)}\r\n".encode()
                    + b"\r\n"
                    + body
                )
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(serve_connection, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    http_client = httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{port}/v1", trust_env=False
    )
    client = OpenAICompatibleClient(
        base_url=f"http://127.0.0.1:{port}/v1", client=http_client
    )
    route = ModelRoute("main", "openai", "gpt-test")
    payload = OpenAICompatibleAdapter().build_request(_request())

    try:
        for _ in range(2):
            streamed = [item async for item in client.stream(route, payload)]
            assert streamed[-1]["done"] is True
        assert connection_count == 1
    finally:
        await client.aclose()
        await http_client.aclose()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_done_does_not_wait_for_unknown_length_sse_eof() -> None:
    release = asyncio.Event()

    async def serve_connection(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            content_length = next(
                (
                    int(line.split(b":", 1)[1].strip())
                    for line in head.split(b"\r\n")
                    if line.lower().startswith(b"content-length:")
                ),
                0,
            )
            if content_length:
                await reader.readexactly(content_length)
            body = b"data: [DONE]\n\n"
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/event-stream\r\n"
                b"Transfer-Encoding: chunked\r\n\r\n"
                + f"{len(body):x}\r\n".encode()
                + body
                + b"\r\n"
            )
            await writer.drain()
            await release.wait()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(serve_connection, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    http_client = httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{port}/v1", trust_env=False
    )
    client = OpenAICompatibleClient(
        base_url=f"http://127.0.0.1:{port}/v1", client=http_client
    )

    async def collect() -> list[FrozenJsonObject]:
        return [
            item
            async for item in client.stream(
                ModelRoute("main", "openai", "gpt-test"), freeze_json_object({})
            )
        ]

    try:
        started = asyncio.get_running_loop().time()
        streamed = await asyncio.wait_for(collect(), timeout=1)
        elapsed = asyncio.get_running_loop().time() - started
        assert streamed[-1]["done"] is True
        assert elapsed < 0.25
    finally:
        release.set()
        await client.aclose()
        await http_client.aclose()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_delay", [0.0, 0.01])
async def test_completed_chunked_sse_response_reuses_http1_connection(
    terminal_delay: float,
) -> None:
    connection_count = 0

    async def serve_connection(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        nonlocal connection_count
        connection_count += 1
        try:
            while True:
                try:
                    head = await reader.readuntil(b"\r\n\r\n")
                except (asyncio.IncompleteReadError, ConnectionError):
                    return
                content_length = next(
                    (
                        int(line.split(b":", 1)[1].strip())
                        for line in head.split(b"\r\n")
                        if line.lower().startswith(b"content-length:")
                    ),
                    0,
                )
                if content_length:
                    await reader.readexactly(content_length)
                body = (
                    b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
                    b"data: [DONE]\n\n"
                )
                writer.write(
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: text/event-stream\r\n"
                    b"Transfer-Encoding: chunked\r\n\r\n"
                    + f"{len(body):x}\r\n".encode()
                    + body
                    + b"\r\n"
                )
                await writer.drain()
                if terminal_delay:
                    await asyncio.sleep(terminal_delay)
                writer.write(b"0\r\n\r\n")
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(serve_connection, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    http_client = httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{port}/v1", trust_env=False
    )
    client = OpenAICompatibleClient(
        base_url=f"http://127.0.0.1:{port}/v1", client=http_client
    )
    route = ModelRoute("main", "openai", "gpt-test")
    payload = OpenAICompatibleAdapter().build_request(_request())

    try:
        for _ in range(2):
            streamed = [item async for item in client.stream(route, payload)]
            assert streamed[-1]["done"] is True
        assert connection_count == 1
    finally:
        await client.aclose()
        await http_client.aclose()
        server.close()
        await server.wait_closed()


@pytest.mark.parametrize("bypass", [False, True])
@pytest.mark.asyncio
async def test_owned_client_honors_system_proxy_bypass(
    monkeypatch, bypass: bool
) -> None:
    trust_environment = []
    original_async_client = httpx.AsyncClient

    monkeypatch.setattr(
        "pygent.llm.openai_compatible.urllib.request.proxy_bypass",
        lambda _: bypass,
    )

    def recording_async_client(*args, **kwargs):
        trust_environment.append(kwargs.get("trust_env"))
        return original_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", recording_async_client)
    client = OpenAICompatibleClient(base_url="https://models.example/v1")

    assert trust_environment == [not bypass] * 8
    await client.aclose()


@pytest.mark.asyncio
async def test_owned_client_keeps_environment_when_proxy_bypass_fails(
    monkeypatch,
) -> None:
    trust_environment = []
    original_async_client = httpx.AsyncClient

    def fail_bypass(_: str) -> bool:
        raise OSError("proxy bypass lookup failed")

    monkeypatch.setattr(
        "pygent.llm.openai_compatible.urllib.request.proxy_bypass",
        fail_bypass,
    )

    def recording_async_client(*args, **kwargs):
        trust_environment.append(kwargs.get("trust_env"))
        return original_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", recording_async_client)
    client = OpenAICompatibleClient(base_url="https://models.example/v1")

    assert trust_environment == [True] * 8
    await client.aclose()


@pytest.mark.asyncio
async def test_owned_client_close_is_idempotent_and_blocks_reuse():
    client = OpenAICompatibleClient(base_url="https://models.example/v1")
    owned_clients = client._clients
    assert len(owned_clients) == 8
    assert [client._next_http_slot()[0] for _ in range(9)] == [
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
async def test_owned_shard_admission_bounds_work_before_httpcore(monkeypatch):
    active = 0
    maximum_active = 0
    saturated = asyncio.Event()
    release = asyncio.Event()
    original_async_client = httpx.AsyncClient

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        if active == 56:
            saturated.set()
        try:
            await release.wait()
            return httpx.Response(200, json={"ok": True})
        finally:
            active -= 1

    def mock_async_client(*args, **kwargs):
        return original_async_client(
            *args,
            **kwargs,
            transport=httpx.MockTransport(handler),
        )

    monkeypatch.setattr(httpx, "AsyncClient", mock_async_client)
    client = OpenAICompatibleClient(base_url="https://models.example/v1")
    route = ModelRoute("main", "openai", "gpt-test")
    payload = OpenAICompatibleAdapter().build_request(_request())
    tasks = [asyncio.create_task(client.invoke(route, payload)) for _ in range(64)]
    try:
        await asyncio.wait_for(saturated.wait(), timeout=1)
        await asyncio.sleep(0)
        assert maximum_active == 56
        assert active == 56

        for task in tasks[-8:]:
            task.cancel()
        cancelled = await asyncio.gather(*tasks[-8:], return_exceptions=True)
        assert all(isinstance(value, asyncio.CancelledError) for value in cancelled)

        release.set()
        results = await asyncio.gather(*tasks[:-8])
        assert all(result["ok"] is True for result in results)
        assert maximum_active == 56
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await client.aclose()


@pytest.mark.asyncio
async def test_owned_close_rejects_queued_shard_requests(monkeypatch):
    active = 0
    saturated = asyncio.Event()
    release = asyncio.Event()
    original_async_client = httpx.AsyncClient

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal active
        active += 1
        if active == 56:
            saturated.set()
        try:
            await release.wait()
            return httpx.Response(200, json={"ok": True})
        finally:
            active -= 1

    def mock_async_client(*args, **kwargs):
        return original_async_client(
            *args,
            **kwargs,
            transport=httpx.MockTransport(handler),
        )

    monkeypatch.setattr(httpx, "AsyncClient", mock_async_client)
    client = OpenAICompatibleClient(base_url="https://models.example/v1")
    route = ModelRoute("main", "openai", "gpt-test")
    payload = OpenAICompatibleAdapter().build_request(_request())
    tasks = [asyncio.create_task(client.invoke(route, payload)) for _ in range(64)]

    await asyncio.wait_for(saturated.wait(), timeout=1)
    close_task = asyncio.create_task(client.aclose())
    await asyncio.sleep(0)
    assert not close_task.done()
    release.set()
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)
    await asyncio.wait_for(close_task, timeout=1)

    assert sum(isinstance(value, RuntimeError) for value in outcomes) == 8
    assert sum(isinstance(value, FrozenJsonObject) for value in outcomes) == 56


@pytest.mark.asyncio
async def test_owned_http_shards_share_one_strict_ssl_context(monkeypatch):
    contexts = []
    shard_contexts = []
    shard_limits = []
    original_create_ssl_context = httpx.create_ssl_context
    original_async_client = httpx.AsyncClient

    def recording_create_ssl_context(*args, **kwargs):
        context = original_create_ssl_context(*args, **kwargs)
        contexts.append(context)
        return context

    def recording_async_client(*args, **kwargs):
        shard_contexts.append(kwargs.get("verify"))
        shard_limits.append(kwargs.get("limits"))
        return original_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "create_ssl_context", recording_create_ssl_context)
    monkeypatch.setattr(httpx, "AsyncClient", recording_async_client)
    client = OpenAICompatibleClient(
        base_url="https://models.example/v1",
        api_key="secret",
    )
    try:
        assert len(contexts) == 1
        assert len(shard_contexts) == 8
        assert all(context is contexts[0] for context in shard_contexts)
        assert all(limit.max_connections == 7 for limit in shard_limits)
        assert all(limit.max_keepalive_connections == 7 for limit in shard_limits)
        assert contexts[0].check_hostname is True
        assert contexts[0].verify_mode == ssl.CERT_REQUIRED
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_injected_http_client_does_not_create_or_take_over_ssl_context(
    monkeypatch,
):
    injected = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200))
    )

    def fail_create_ssl_context(*args, **kwargs):
        raise AssertionError("injected client unexpectedly created a TLS context")

    monkeypatch.setattr(httpx, "create_ssl_context", fail_create_ssl_context)
    client = OpenAICompatibleClient(
        base_url="https://models.example/v1",
        client=injected,
    )
    await client.aclose()

    assert not injected.is_closed
    await injected.aclose()


@pytest.mark.asyncio
async def test_injected_http_client_is_not_sharded_or_closed():
    http_client = httpx.AsyncClient()
    client = OpenAICompatibleClient(
        base_url="https://models.example/v1", client=http_client
    )

    assert client._clients == (http_client,)
    assert client._next_http_slot() == (http_client, None)
    assert client._next_http_slot() == (http_client, None)
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
