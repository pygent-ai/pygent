from __future__ import annotations

import json

import httpx
import pytest

from examples.live_agent.fallback_probe import AuthenticationProxy


@pytest.mark.asyncio
async def test_authentication_proxy_rejects_invalid_and_forwards_valid_key():
    upstream_calls = 0

    async def upstream(request: httpx.Request) -> httpx.Response:
        nonlocal upstream_calls
        upstream_calls += 1
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer valid-key"
        assert json.loads(request.content) == {"model": "unit"}
        return httpx.Response(200, json={"ok": True})

    proxy = AuthenticationProxy(
        "https://upstream.example/v1",
        "valid-key",
        transport=httpx.MockTransport(upstream),
    )
    transport = httpx.ASGITransport(app=proxy.app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://proxy.example"
    ) as client:
        rejected = await client.post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer invalid-key"},
            json={"model": "unit"},
        )
        forwarded = await client.post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer valid-key"},
            json={"model": "unit"},
        )

    await proxy.aclose()
    assert rejected.status_code == 401
    assert forwarded.status_code == 200
    assert forwarded.json() == {"ok": True}
    assert proxy.rejected == 1
    assert proxy.forwarded == 1
    assert upstream_calls == 1
