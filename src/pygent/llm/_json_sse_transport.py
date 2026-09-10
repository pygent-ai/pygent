"""Private protocol-neutral JSON and SSE HTTP mechanics."""

from __future__ import annotations

import asyncio
import json
import urllib.parse
import urllib.request
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import cast

import httpx

from pygent import _native
from pygent.core import FrozenJsonObject, freeze_json_object

_DEFAULT_MAX_CONNECTIONS = 56
_MAX_HTTP_ERROR_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class _SSEFrame:
    event: str | None
    data: str


@dataclass(slots=True)
class _HTTPResponseError(Exception):
    status: int
    body: bytes = field(repr=False)


class _JsonSSETransport:
    def __init__(
        self,
        *,
        headers: Mapping[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
        verify_ssl: bool | None = None,
        trust_env_url: str,
    ) -> None:
        if verify_ssl is not None and not isinstance(verify_ssl, bool):
            raise TypeError("verify_ssl must be a bool or None")
        if client is not None and verify_ssl is not None:
            raise ValueError("verify_ssl cannot be set with an injected HTTP client")
        self._headers = dict(headers or {})
        self._client = client
        self._native_client = (
            None
            if client is not None
            else _native.NativeHttpClient(
                self._headers,
                _trust_environment_for_url(trust_env_url),
                _DEFAULT_MAX_CONNECTIONS,
                True if verify_ssl is None else verify_ssl,
            )
        )
        self._closed = False

    async def request_json(
        self,
        method: str,
        url: str,
        payload: Mapping[str, object] | None,
        *,
        timeout: float | None = None,
    ) -> FrozenJsonObject:
        self._ensure_open()
        native_client = self._native_client
        if native_client is not None:
            try:
                status, raw_body = await native_client.request_json(
                    method,
                    url,
                    None if payload is None else _wire_json(payload),
                    timeout,
                )
            except asyncio.CancelledError:
                raise
            except RuntimeError as exc:
                raise httpx.TransportError(str(exc)) from exc
            body_bytes = raw_body.encode("utf-8")
            if not 200 <= status < 300:
                raise _HTTPResponseError(
                    status, body_bytes[:_MAX_HTTP_ERROR_BYTES]
                )
            raw: str | bytes = raw_body
        else:
            assert self._client is not None
            response = await self._client.request(
                method,
                url,
                json=None if payload is None else dict(payload),
                headers=self._headers or None,
                timeout=timeout,
            )
            if not response.is_success:
                raise _HTTPResponseError(
                    response.status_code,
                    response.content[:_MAX_HTTP_ERROR_BYTES],
                )
            raw = response.content
        try:
            decoded = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("HTTP response is not valid JSON") from exc
        if not isinstance(decoded, Mapping):
            raise TypeError("HTTP JSON response must be an object")
        return freeze_json_object(cast(Mapping[str, object], decoded))

    async def stream_sse(
        self, url: str, payload: Mapping[str, object]
    ) -> AsyncIterator[_SSEFrame]:
        self._ensure_open()
        native_client = self._native_client
        if native_client is not None:
            native_stream = native_client.stream_sse(url, _wire_json(payload))
            completed = False
            try:
                async for kind, value in native_stream:
                    if kind == "status":
                        status = cast(int, value)
                        if not 200 <= status < 300:
                            raise _HTTPResponseError(status, b"")
                        continue
                    if kind == "error":
                        raise httpx.TransportError(cast(str, value))
                    if kind != "data":  # pragma: no cover - native invariant
                        raise RuntimeError("native SSE transport returned invalid item")
                    yield _SSEFrame(None, cast(str, value))
                completed = True
            finally:
                if not completed:
                    native_stream.close()
                    await asyncio.shield(native_stream.wait_closed())
            return

        assert self._client is not None
        async with self._client.stream(
            "POST",
            url,
            json=dict(payload),
            headers={**self._headers, "Accept": "text/event-stream"},
        ) as response:
            if not response.is_success:
                raise _HTTPResponseError(
                    response.status_code,
                    await _read_bounded_body(response),
                )
            event: str | None = None
            data_lines: list[str] = []
            async for line in response.aiter_lines():
                if line == "":
                    if data_lines:
                        yield _SSEFrame(event, "\n".join(data_lines))
                    event = None
                    data_lines.clear()
                    continue
                if line.startswith(":"):
                    continue
                field_name, separator, raw_value = line.partition(":")
                if not separator:
                    raw_value = ""
                value = raw_value.removeprefix(" ")
                if field_name == "event":
                    event = value or None
                elif field_name == "data":
                    data_lines.append(value)
            if data_lines:
                yield _SSEFrame(event, "\n".join(data_lines))

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._native_client is not None:
            await self._native_client.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("model provider client is closed")


async def _read_bounded_body(response: httpx.Response) -> bytes:
    body = bytearray()
    async for chunk in response.aiter_bytes():
        remaining = _MAX_HTTP_ERROR_BYTES - len(body)
        if remaining <= 0:
            break
        body.extend(chunk[:remaining])
    return bytes(body)


def _trust_environment_for_url(url: str) -> bool:
    hostname = urllib.parse.urlsplit(url).hostname
    if hostname is None:
        return True
    try:
        return not urllib.request.proxy_bypass(hostname)
    except OSError:
        return True


def _wire_json(value: Mapping[str, object]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


__all__ = ["_HTTPResponseError", "_JsonSSETransport", "_SSEFrame"]
