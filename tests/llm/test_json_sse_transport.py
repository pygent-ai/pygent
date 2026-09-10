from __future__ import annotations

import json

import httpx
import pytest

from pygent.llm._json_sse_transport import (
    _HTTPResponseError,
    _JsonSSETransport,
)


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
            await transport.request_json(
                "GET", "https://models.example/error", None
            )
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
                "data: {\"type\":\n"
                "data: \"message_start\"}\n\n"
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
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200)))
    transport = _JsonSSETransport(
        client=client,
        trust_env_url="https://models.example",
    )

    await transport.aclose()
    await transport.aclose()

    assert not client.is_closed
    await client.aclose()
