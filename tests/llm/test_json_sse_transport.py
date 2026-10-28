from __future__ import annotations

import asyncio
import json
import socket

import httpx
import pytest

from pygent.llm._json_sse_transport import (
    _HTTPResponseError,
    _JsonSSETransport,
)
from pygent.llm.types import ModelErrorKind, ModelProviderError

_SENT_SECRETS = ("https://models.example/v1", "canary-key-1234567890")


def _secrets_transport(client: httpx.AsyncClient) -> _JsonSSETransport:
    return _JsonSSETransport(
        client=client,
        trust_env_url="https://models.example",
        secrets=_SENT_SECRETS,
    )


@pytest.mark.parametrize(
    ("error", "expected_kind"),
    [
        pytest.param(
            httpx.ProxyError(
                "Unable to connect to proxy https://models.example/v1"
                " (key canary-key-1234567890)"
            ),
            ModelErrorKind.UNAVAILABLE,
            id="proxy-error",
        ),
        pytest.param(
            httpx.ConnectError("[Errno -2] Name or service not known"),
            ModelErrorKind.UNAVAILABLE,
            id="dns-failure",
        ),
        pytest.param(
            httpx.ConnectError("[Errno 111] Connect call failed ('10.0.0.42', 443)"),
            ModelErrorKind.UNAVAILABLE,
            id="connection-refused",
        ),
        pytest.param(
            httpx.ReadTimeout("timed out while reading the response body"),
            ModelErrorKind.TIMEOUT,
            id="read-timeout",
        ),
    ],
)
@pytest.mark.asyncio
async def test_request_json_wraps_transport_errors_with_redacted_text(
    error: httpx.TransportError, expected_kind: ModelErrorKind
) -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        raise error

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = _secrets_transport(client)
    try:
        with pytest.raises(ModelProviderError) as raised:
            await transport.request_json("POST", "https://models.example/json", {})
    finally:
        await transport.aclose()
        await client.aclose()

    assert raised.value.kind is expected_kind
    message = raised.value.provider_message
    assert message
    assert str(error) in message or "[REDACTED]" in message
    assert "canary-key-1234567890" not in message
    assert "https://models.example/v1" not in message


@pytest.mark.asyncio
async def test_request_json_keeps_the_proxy_disconnect_root_cause() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ProxyError("Unable to connect to proxy https://models.example/v1")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = _secrets_transport(client)
    try:
        with pytest.raises(ModelProviderError) as raised:
            await transport.request_json("POST", "https://models.example/json", {})
    finally:
        await transport.aclose()
        await client.aclose()

    assert raised.value.kind is ModelErrorKind.UNAVAILABLE
    assert raised.value.provider_message == "Unable to connect to proxy [REDACTED]"


@pytest.mark.asyncio
async def test_stream_sse_wraps_mid_stream_read_errors_with_redacted_text() -> None:
    async def body() -> object:
        yield b'data: {"ok": true}\n\n'
        raise httpx.ReadError("stream interrupted: proxy closed the connection")

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=body(),
            headers={"content-type": "text/event-stream"},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = _secrets_transport(client)
    try:
        with pytest.raises(ModelProviderError) as raised:
            async for _frame in transport.stream_sse(
                "https://models.example/stream", {"stream": True}
            ):
                pass
    finally:
        await transport.aclose()
        await client.aclose()

    assert raised.value.kind is ModelErrorKind.UNAVAILABLE
    assert raised.value.provider_message == (
        "stream interrupted: proxy closed the connection"
    )


@pytest.mark.asyncio
async def test_request_json_does_not_wrap_non_transport_errors() -> None:
    """Only transport-phase errors are wrapped; other failures stay purified
    at the invoker boundary (UNKNOWN), so this edge must not widen."""

    async def handler(_request: httpx.Request) -> httpx.Response:
        raise RuntimeError("transport exploded: canary-key-1234567890")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = _secrets_transport(client)
    try:
        with pytest.raises(RuntimeError, match="transport exploded"):
            await transport.request_json("POST", "https://models.example/json", {})
    finally:
        await transport.aclose()
        await client.aclose()


@pytest.mark.asyncio
async def test_transport_without_secrets_still_wraps_transport_errors() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[Errno 111] Connect call failed")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = _JsonSSETransport(
        client=client,
        trust_env_url="https://models.example",
    )
    try:
        with pytest.raises(ModelProviderError) as raised:
            await transport.request_json("POST", "https://models.example/json", {})
    finally:
        await transport.aclose()
        await client.aclose()

    assert raised.value.kind is ModelErrorKind.UNAVAILABLE
    assert raised.value.provider_message == "[Errno 111] Connect call failed"


@pytest.mark.asyncio
async def test_native_client_transport_failure_is_wrapped_with_redacted_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default native client reports failures as RuntimeErrors; the wrap
    must keep the sanitized root cause (this is the proxy-disconnect case)."""

    import pygent.llm._json_sse_transport as transport_module

    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    await asyncio.sleep(0.05)

    monkeypatch.setattr(
        transport_module, "_trust_environment_for_url", lambda _url: False
    )
    base_url = f"http://127.0.0.1:{port}/v1"
    transport = _JsonSSETransport(
        trust_env_url=base_url,
        secrets=(base_url,),
    )
    try:
        with pytest.raises(ModelProviderError) as raised:
            await transport.request_json("POST", f"{base_url}/chat/completions", {})
    finally:
        await transport.aclose()

    assert raised.value.kind is ModelErrorKind.UNAVAILABLE
    assert raised.value.provider_message
    assert base_url not in raised.value.provider_message
    assert f"{base_url}/chat/completions" not in str(raised.value)


@pytest.mark.asyncio
async def test_json_transport_returns_objects_and_bounds_http_errors() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/error":
            return httpx.Response(429, content=b"x" * (70 * 1024))
        return httpx.Response(200, json={"ok": True})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = _JsonSSETransport(
        headers={"x-test": "value"},
        client=client,
        trust_env_url="https://models.example",
    )
    try:
        result = await transport.request_json(
            "POST", "https://models.example/json", {"input": 1}
        )
        assert result["ok"] is True
        assert json.loads(requests[0].content) == {"input": 1}
        assert requests[0].headers["x-test"] == "value"

        with pytest.raises(_HTTPResponseError) as raised:
            await transport.request_json("GET", "https://models.example/error", None)
        assert raised.value.status == 429
        assert len(raised.value.body) == 64 * 1024
    finally:
        await transport.aclose()
        await client.aclose()


@pytest.mark.asyncio
async def test_sse_transport_parses_comments_events_and_multiline_data() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {"stream": True}
        return httpx.Response(
            200,
            text=(
                ": keepalive\n"
                "event: message_start\n"
                'data: {"type":\n'
                'data: "message_start"}\n\n'
                "data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = _JsonSSETransport(
        client=client,
        trust_env_url="https://models.example",
    )
    try:
        frames = [
            frame
            async for frame in transport.stream_sse(
                "https://models.example/stream", {"stream": True}
            )
        ]
    finally:
        await transport.aclose()
        await client.aclose()

    assert [(frame.event, frame.data) for frame in frames] == [
        ("message_start", '{"type":\n"message_start"}'),
        (None, "[DONE]"),
    ]


@pytest.mark.asyncio
async def test_injected_client_is_borrowed_and_close_is_idempotent() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200))
    )
    transport = _JsonSSETransport(
        client=client,
        trust_env_url="https://models.example",
    )

    await transport.aclose()
    await transport.aclose()

    assert not client.is_closed
    await client.aclose()
