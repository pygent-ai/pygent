from __future__ import annotations

import json

import httpx
import pytest

from pygent.tool import (
    HttpToolExecutor,
    IdempotencyPolicy,
    ToolCall,
    ToolDefinition,
    ToolExecutionContext,
    ToolExecutionError,
    ToolSideEffect,
    ToolSpec,
)


def remote_spec() -> ToolSpec:
    return ToolSpec(
        tool_id="remote.write",
        version="7",
        definition=ToolDefinition("write", "", {"type": "object"}),
        side_effect=ToolSideEffect.EXTERNAL,
        idempotency=IdempotencyPolicy.REQUIRES_KEY,
        timeout=1,
    )


@pytest.mark.asyncio
async def test_http_executor_sends_stable_identity_arguments_and_idempotency_key() -> (
    None
):
    observed: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        observed.update(json.loads((await request.aread()).decode()))
        assert request.headers["x-api-key"] == "safe"
        return httpx.Response(200, json={"accepted": True})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    executor = HttpToolExecutor(
        "https://tools.example/invoke",
        client=client,
        headers={"x-api-key": "safe"},
    )
    result = await executor.execute(
        remote_spec(),
        ToolCall(
            "call-1",
            "write",
            {"value": 3},
            idempotency_key="run:module:call-1",
        ),
        ToolExecutionContext(),
    )

    assert result == {"accepted": True}
    assert observed == {
        "tool_id": "remote.write",
        "version": "7",
        "call_id": "call-1",
        "name": "write",
        "arguments": {"value": 3},
        "idempotency_key": "run:module:call-1",
    }
    assert not client.is_closed
    await client.aclose()


@pytest.mark.parametrize(
    ("status", "retryable"),
    [(400, False), (408, True), (429, True), (500, True), (503, True)],
)
@pytest.mark.asyncio
async def test_http_executor_classifies_status_without_leaking_response_body(
    status: int, retryable: bool
) -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text="secret remote diagnostic")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        executor = HttpToolExecutor("https://tools.example/invoke", client=client)
        with pytest.raises(ToolExecutionError) as raised:
            await executor.execute(
                remote_spec(),
                ToolCall("call", "write", {}, idempotency_key="key"),
                ToolExecutionContext(),
            )

    assert raised.value.kind == "remote_error"
    assert raised.value.code == f"http_{status}"
    assert raised.value.retryable is retryable
    assert raised.value.side_effect_committed is None
    assert "secret" not in str(raised.value)


@pytest.mark.asyncio
async def test_http_executor_classifies_timeout_and_invalid_json() -> None:
    async def timeout(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("late")

    async with httpx.AsyncClient(transport=httpx.MockTransport(timeout)) as client:
        with pytest.raises(ToolExecutionError) as raised:
            await HttpToolExecutor(
                "https://tools.example/invoke", client=client
            ).execute(
                remote_spec(),
                ToolCall("timeout", "write", {}, idempotency_key="key"),
                ToolExecutionContext(),
            )
    assert (raised.value.kind, raised.value.code, raised.value.retryable) == (
        "timeout",
        "http_timeout",
        True,
    )

    async def invalid_json(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json")

    async with httpx.AsyncClient(transport=httpx.MockTransport(invalid_json)) as client:
        with pytest.raises(ToolExecutionError) as raised:
            await HttpToolExecutor(
                "https://tools.example/invoke", client=client
            ).execute(
                remote_spec(),
                ToolCall("json", "write", {}, idempotency_key="key"),
                ToolExecutionContext(),
            )
    assert (raised.value.kind, raised.value.code) == (
        "transport_error",
        "http_transport",
    )
