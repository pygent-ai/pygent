from __future__ import annotations

import asyncio
import json
import socket

import httpx
import pytest

import pygent.llm._json_sse_transport as transport_module
from pygent.llm._json_sse_transport import (
    _HTTPResponseError,
    _JsonSSETransport,
    _native_transport_error_parts,
)
from pygent.llm.types import ModelErrorKind, ModelProviderError

_SENT_SECRETS = ("https://models.example/v1", "canary-key-1234567890")


def _secrets_transport(client: httpx.AsyncClient) -> _JsonSSETransport:
    return _JsonSSETransport(
        client=client,
        trust_env_url="https://models.example",
        secrets=_SENT_SECRETS,
    )


class _FakeNativeTransportError(transport_module._native.NativeTransportError):
    """Scripted stand-in for the native structured transport error.

    Subclasses the real native exception so the type-based catch in the
    transport is exercised without touching the network.
    """

    def __init__(self, category: str, detail: str) -> None:
        super().__init__(detail)
        self.category = category
        self.detail = detail


class _FakeNativeClient:
    """Native client stand-in that always fails with one scripted error."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    async def request_json(self, *_args: object, **_kwargs: object) -> tuple[int, str]:
        raise self._error

    async def close(self) -> None:
        return None


def _patched_native_transport(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> _JsonSSETransport:
    monkeypatch.setattr(
        transport_module._native,
        "NativeHttpClient",
        lambda *_args, **_kwargs: _FakeNativeClient(error),
    )
    return _JsonSSETransport(
        trust_env_url="https://models.example",
        secrets=_SENT_SECRETS,
    )


class _FakeNativeStream:
    def __init__(self, items: list[tuple[str, object]]) -> None:
        self._items = list(items)

    def __aiter__(self) -> _FakeNativeStream:
        return self

    async def __anext__(self) -> tuple[str, object]:
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)

    def close(self) -> None:
        return None

    async def wait_closed(self) -> None:
        return None


class _FakeNativeStreamClient:
    def __init__(self, items: list[tuple[str, object]]) -> None:
        self._items = items

    def stream_sse(self, *_args: object) -> _FakeNativeStream:
        return _FakeNativeStream(self._items)

    async def close(self) -> None:
        return None


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


@pytest.mark.parametrize(
    ("category", "expected_kind"),
    [
        pytest.param("connect", ModelErrorKind.UNAVAILABLE, id="connect"),
        pytest.param("request", ModelErrorKind.UNAVAILABLE, id="request"),
        pytest.param("body", ModelErrorKind.UNAVAILABLE, id="body"),
        pytest.param("decode", ModelErrorKind.UNAVAILABLE, id="decode"),
        pytest.param("timeout", ModelErrorKind.TIMEOUT, id="timeout"),
        pytest.param("category-not-in-rust-yet", ModelErrorKind.UNAVAILABLE, id="unknown"),
    ],
)
def test_native_transport_error_parts_map_category_to_kind(
    category: str, expected_kind: ModelErrorKind
) -> None:
    error = _FakeNativeTransportError(category, "chain text: os error 10061")

    kind, detail = _native_transport_error_parts(error)

    assert kind is expected_kind
    assert detail == "chain text: os error 10061"


def test_native_transport_error_parts_require_structured_fields() -> None:
    """The native constructor guarantees the structured fields; a plain
    RuntimeError violates the contract and fails loudly instead of
    degrading into an unclassified wrap."""

    class _BareNativeTransportError(RuntimeError):
        pass

    with pytest.raises(AttributeError):
        _native_transport_error_parts(_BareNativeTransportError("plain text"))


def test_transport_failure_caps_the_redacted_detail() -> None:
    limit = transport_module._MAX_TRANSPORT_ERROR_MESSAGE_CHARS
    long_detail = "connect error: " + "x" * (limit * 2)

    failure = transport_module._transport_failure(
        ModelErrorKind.UNAVAILABLE, long_detail, ()
    )

    assert failure.provider_message is not None
    assert len(failure.provider_message) <= limit
    assert failure.provider_message.startswith("connect error: ")


def test_redact_secrets_scrubs_url_userinfo() -> None:
    """A system proxy echoing its own credentials inside a URL must lose
    them while the host stays readable for diagnosis."""

    message = (
        "error sending request for url"
        " (http://proxyuser:proxypass@proxy.corp:8080/chat/completions)"
    )

    redacted = transport_module._redact_secrets(message, ())

    assert "proxyuser" not in redacted
    assert "proxypass" not in redacted
    assert "[REDACTED]@proxy.corp:8080" in redacted


@pytest.mark.asyncio
async def test_hung_local_service_maps_to_timeout_on_both_transports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """httpx and the native client must agree that an elapsed request budget
    is TIMEOUT, not provider unavailability. A local listener that never
    speaks makes the hang deterministic without leaving the machine, so the
    parity holds in any CI environment."""

    monkeypatch.setattr(
        transport_module, "_trust_environment_for_url", lambda _url: False
    )
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    url = f"http://127.0.0.1:{port}/"
    native_transport = _JsonSSETransport(trust_env_url=url)
    injected_transport = _JsonSSETransport(
        client=httpx.AsyncClient(trust_env=False, timeout=1.0),
        trust_env_url=url,
    )
    try:
        for transport in (native_transport, injected_transport):
            with pytest.raises(ModelProviderError) as raised:
                await transport.request_json("GET", url, None, timeout=1.0)
            assert raised.value.kind is ModelErrorKind.TIMEOUT
    finally:
        await native_transport.aclose()
        await injected_transport.aclose()
        listener.close()


@pytest.mark.asyncio
async def test_httpx_connect_timeout_maps_to_timeout_kind() -> None:
    """httpx's connect-phase timeout must stay TIMEOUT through the wrap,
    pinned deterministically with a mock transport instead of a network
    topology assumption."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("simulated connect timeout", request=request)

    transport = _JsonSSETransport(
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        trust_env_url="https://models.example",
    )
    try:
        with pytest.raises(ModelProviderError) as raised:
            await transport.request_json("POST", "https://models.example/json", {})
    finally:
        await transport.aclose()

    assert raised.value.kind is ModelErrorKind.TIMEOUT


@pytest.mark.asyncio
async def test_request_json_maps_native_timeout_category_to_timeout_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native timeouts must classify as TIMEOUT; the old plain-text
    RuntimeError path misfiled them as UNAVAILABLE."""

    transport = _patched_native_transport(
        monkeypatch,
        _FakeNativeTransportError("timeout", "request timed out after 2.000s"),
    )
    try:
        with pytest.raises(ModelProviderError) as raised:
            await transport.request_json("POST", "https://models.example/json", {})
    finally:
        await transport.aclose()

    assert raised.value.kind is ModelErrorKind.TIMEOUT
    assert raised.value.provider_message == "request timed out after 2.000s"


@pytest.mark.asyncio
async def test_request_json_redacts_structured_native_transport_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _patched_native_transport(
        monkeypatch,
        _FakeNativeTransportError(
            "connect",
            "error sending request for url (https://models.example/v1): os error 10061",
        ),
    )
    try:
        with pytest.raises(ModelProviderError) as raised:
            await transport.request_json("POST", "https://models.example/json", {})
    finally:
        await transport.aclose()

    assert raised.value.kind is ModelErrorKind.UNAVAILABLE
    assert raised.value.provider_message == (
        "error sending request for url ([REDACTED]): os error 10061"
    )


@pytest.mark.asyncio
async def test_request_json_keeps_plain_runtime_error_fallback_wrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """State failures such as a closed native client are not transport
    errors; they keep the plain RuntimeError fallback wrap."""

    transport = _patched_native_transport(
        monkeypatch, RuntimeError("model provider client is closed")
    )
    try:
        with pytest.raises(ModelProviderError) as raised:
            await transport.request_json("POST", "https://models.example/json", {})
    finally:
        await transport.aclose()

    assert raised.value.kind is ModelErrorKind.UNAVAILABLE
    assert raised.value.provider_message == "model provider client is closed"


@pytest.mark.asyncio
async def test_stream_sse_maps_native_error_items_by_category(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        transport_module._native,
        "NativeHttpClient",
        lambda *_args, **_kwargs: _FakeNativeStreamClient(
            [("error", ("body", "stream read failed: os error 10054"))]
        ),
    )
    transport = _JsonSSETransport(
        trust_env_url="https://models.example",
        secrets=_SENT_SECRETS,
    )
    try:
        with pytest.raises(ModelProviderError) as raised:
            async for _frame in transport.stream_sse(
                "https://models.example/stream", {"stream": True}
            ):
                pass
    finally:
        await transport.aclose()

    assert raised.value.kind is ModelErrorKind.UNAVAILABLE
    assert raised.value.provider_message == "stream read failed: os error 10054"


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
