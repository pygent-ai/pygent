from __future__ import annotations

import asyncio
from typing import Any

import pytest

from pygent.agent.pygent_v1.runtime.model_configuration import (
    CLIENT_FACTORIES,
    _create_client,
)
from pygent.llm import CredentialRef, ResolvedModelConnection


def _connection(**overrides: object) -> ResolvedModelConnection:
    values: dict[str, Any] = {
        "connection_key": "gateway",
        "provider": "custom_gateway",
        "protocol": "openai_chat_completions",
        "base_url": "https://gateway.example.com/v1",
        "credential": CredentialRef.environment("GATEWAY_API_KEY"),
    }
    values.update(overrides)
    return ResolvedModelConnection(**values)


class _FakeClient:
    async def aclose(self) -> None:
        return None


def test_create_client_passes_connection_headers_to_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def factory(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _FakeClient()

    monkeypatch.setitem(CLIENT_FACTORIES, "openai_chat_completions", factory)

    _create_client(
        connection=_connection(headers={"X-Title": "pygent"}),
        credential_environ={"GATEWAY_API_KEY": "gateway-key"},
    )
    assert captured["headers"] == {"X-Title": "pygent"}

    captured.clear()
    _create_client(
        connection=_connection(),
        credential_environ={"GATEWAY_API_KEY": "gateway-key"},
    )
    assert captured["headers"] is None


def test_create_client_passes_connection_headers_on_proxy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def factory(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _FakeClient()

    monkeypatch.setitem(CLIENT_FACTORIES, "openai_chat_completions", factory)

    client = _create_client(
        connection=_connection(
            proxy="https://proxy.example.com:8080",
            headers={"X-Title": "pygent"},
        ),
        credential_environ={"GATEWAY_API_KEY": "gateway-key"},
    )

    assert captured["headers"] == {"X-Title": "pygent"}
    assert captured["client"] is not None
    asyncio.run(client.aclose())
