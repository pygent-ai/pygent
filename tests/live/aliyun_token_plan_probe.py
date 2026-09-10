"""Explicit, sanitized live verification for Alibaba Cloud Token Plan models.

This script intentionally uses raw provider protocols for catalog-only modalities. It is
not a Pygent Adapter and is not collected as a pytest test module.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import struct
import sys
import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, Self
from urllib.parse import urlsplit, urlunsplit

import httpx

_TOKEN_PLAN_ROOT = "https://token-plan.cn-beijing.maas.aliyuncs.com"
_OPENAI_BASE = f"{_TOKEN_PLAN_ROOT}/compatible-mode/v1"
_ANTHROPIC_BASE = f"{_TOKEN_PLAN_ROOT}/apps/anthropic"
_TEXT_MODELS = (
    "qwen3.8-max",
    "qwen3.8-flash",
    "qwen3.7-max",
    "qwen3.7-plus",
    "qwen3.6-flash",
    "deepseek-v4-pro",
    "deepseek-v4-pro-0813",
    "deepseek-v4-flash-0731",
    "glm-5.2",
)


@dataclass(frozen=True, slots=True)
class ProbeSpec:
    model_id: str
    protocols: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProbeResult:
    model_id: str
    protocol: str
    status: Literal["passed", "failed"]
    error_kind: str | None
    private_detail: object = field(default=None, repr=False, compare=False)


class _WebSocket(Protocol):
    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None: ...

    async def recv(self) -> str | bytes: ...


WebSocketConnect = Callable[[str, dict[str, str]], _WebSocket]


class _ProbeFailure(Exception):
    def __init__(self, error_kind: str) -> None:
        super().__init__(error_kind)
        self.error_kind = error_kind


def probe_inventory() -> tuple[ProbeSpec, ...]:
    return (
        *(ProbeSpec(model_id, ("openai_chat_completions", "anthropic_messages")) for model_id in _TEXT_MODELS),
        ProbeSpec("qwen-image-3.0-pro", ("dashscope_multimodal_generation",)),
        ProbeSpec("wan2.7-image", ("dashscope_multimodal_generation",)),
        ProbeSpec("wan2.7-image-pro", ("dashscope_multimodal_generation",)),
        ProbeSpec("happyhorse-1.1-i2v", ("dashscope_video_generation",)),
        ProbeSpec("happyhorse-1.1-t2v", ("dashscope_video_generation",)),
        ProbeSpec("happyhorse-1.1-r2v", ("dashscope_video_generation",)),
        ProbeSpec("qwen-audio-3.0-tts-plus", ("dashscope_speech_synthesis",)),
        ProbeSpec("qwen-audio-3.0-realtime-plus", ("dashscope_realtime",)),
        ProbeSpec("qwen-audio-3.0-asr-flash", ("dashscope_speech_recognition",)),
    )


def summarize_result(result: ProbeResult) -> dict[str, str | None]:
    return {
        "model_id": result.model_id,
        "protocol": result.protocol,
        "status": result.status,
        "error_kind": result.error_kind,
    }


def _passed(model_id: str, protocol: str, detail: object = None) -> ProbeResult:
    return ProbeResult(model_id, protocol, "passed", None, detail)


def _failed(model_id: str, protocol: str, exc: BaseException) -> ProbeResult:
    if isinstance(exc, _ProbeFailure):
        kind = exc.error_kind
    elif isinstance(exc, (httpx.TimeoutException, TimeoutError, asyncio.TimeoutError)):
        kind = "timeout"
    elif isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in (401, 403):
            kind = "authentication"
        elif status == 429:
            kind = "rate_limit"
        elif 400 <= status < 500:
            kind = "invalid_request"
        elif status >= 500:
            kind = "unavailable"
        else:
            kind = "http_error"
    elif isinstance(exc, httpx.TransportError):
        kind = "unavailable"
    else:
        kind = "unknown"
    return ProbeResult(model_id, protocol, "failed", kind, type(exc).__name__)


def _json(response: httpx.Response) -> Mapping[str, Any]:
    response.raise_for_status()
    try:
        value = response.json()
    except ValueError as exc:
        raise _ProbeFailure("invalid_response") from exc
    if not isinstance(value, dict):
        raise _ProbeFailure("invalid_response")
    return value


def _api_root(openai_base_url: str) -> str:
    suffix = "/compatible-mode/v1"
    base = openai_base_url.rstrip("/")
    if not base.endswith(suffix):
        raise ValueError("OpenAI base URL must end with /compatible-mode/v1")
    return base[: -len(suffix)]


def _solid_png_data_uri() -> str:
    width = height = 512
    rows = b"".join(b"\x00" + b"\xff\xff\xff" * width for _ in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        payload = kind + data
        return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload))

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows, level=9))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _default_websocket_connect(url: str, headers: dict[str, str]) -> _WebSocket:
    try:
        import websockets
    except ImportError as exc:  # pragma: no cover - explicit live dependency
        raise RuntimeError("install websockets to run the realtime probe") from exc
    return websockets.connect(  # type: ignore[no-any-return]
        url,
        additional_headers=headers,
        open_timeout=30,
        close_timeout=10,
    )


class TokenPlanProbeRunner:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        api_key: str,
        anthropic_api_key: str,
        openai_base_url: str = _OPENAI_BASE,
        anthropic_base_url: str = _ANTHROPIC_BASE,
        poll_interval: float = 15,
        poll_timeout: float = 600,
        websocket_connect: WebSocketConnect = _default_websocket_connect,
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._anthropic_api_key = anthropic_api_key
        self._openai_base_url = openai_base_url.rstrip("/")
        self._anthropic_base_url = anthropic_base_url.rstrip("/")
        self._root = _api_root(openai_base_url)
        self._poll_interval = poll_interval
        self._poll_timeout = poll_timeout
        self._websocket_connect = websocket_connect

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    async def probe_openai(self, model_id: str) -> ProbeResult:
        protocol = "openai_chat_completions"
        try:
            response = await self._client.post(
                f"{self._openai_base_url}/chat/completions",
                headers=self._headers,
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": "Reply OK."}],
                    "max_tokens": 8,
                    "stream": False,
                },
                timeout=120,
            )
            body = _json(response)
            if not isinstance(body.get("choices"), list) or not body["choices"]:
                raise _ProbeFailure("invalid_response")
            return _passed(model_id, protocol)
        except (_ProbeFailure, httpx.HTTPError, TimeoutError) as exc:
            return _failed(model_id, protocol, exc)

    async def probe_anthropic(self, model_id: str) -> ProbeResult:
        protocol = "anthropic_messages"
        try:
            response = await self._client.post(
                f"{self._anthropic_base_url}/v1/messages",
                headers={
                    "x-api-key": self._anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": "Reply OK."}],
                    "max_tokens": 8,
                },
                timeout=120,
            )
            body = _json(response)
            if not isinstance(body.get("content"), list):
                raise _ProbeFailure("invalid_response")
            return _passed(model_id, protocol)
        except (_ProbeFailure, httpx.HTTPError, TimeoutError) as exc:
            return _failed(model_id, protocol, exc)

    async def probe_image(self, model_id: str) -> tuple[ProbeResult, str | None]:
        protocol = "dashscope_multimodal_generation"
        try:
            response = await self._client.post(
                f"{self._root}/api/v1/services/aigc/multimodal-generation/generation",
                headers=self._headers,
                json={
                    "model": model_id,
                    "input": {
                        "messages": [
                            {
                                "role": "user",
                                "content": [{"text": "A plain blue square."}],
                            }
                        ]
                    },
                    "parameters": {"size": "1024*1024"},
                },
                timeout=180,
            )
            body = _json(response)
            url = body["output"]["choices"][0]["message"]["content"][0]["image"]
            if not isinstance(url, str) or not url:
                raise _ProbeFailure("invalid_response")
            return _passed(model_id, protocol, url), url
        except (KeyError, IndexError, TypeError) as exc:
            failure = _ProbeFailure("invalid_response")
            failure.__cause__ = exc
            return _failed(model_id, protocol, failure), None
        except (_ProbeFailure, httpx.HTTPError, TimeoutError) as exc:
            return _failed(model_id, protocol, exc), None

    async def probe_video(self, model_id: str) -> ProbeResult:
        protocol = "dashscope_video_generation"
        try:
            input_value: dict[str, object] = {"prompt": "A blue square moves slowly."}
            parameters: dict[str, object] = {
                "resolution": "480P",
                "duration": 3,
                "watermark": False,
            }
            if model_id == "happyhorse-1.1-t2v":
                parameters["ratio"] = "1:1"
            else:
                media_type = (
                    "first_frame"
                    if model_id == "happyhorse-1.1-i2v"
                    else "reference_image"
                )
                input_value["media"] = [
                    {"type": media_type, "url": _solid_png_data_uri()}
                ]
                if model_id == "happyhorse-1.1-r2v":
                    parameters["ratio"] = "1:1"
            response = await self._client.post(
                f"{self._root}/api/v1/services/aigc/video-generation/video-synthesis",
                headers={**self._headers, "X-DashScope-Async": "enable"},
                json={
                    "model": model_id,
                    "input": input_value,
                    "parameters": parameters,
                },
                timeout=60,
            )
            body = _json(response)
            submitted_output = body.get("output")
            if not isinstance(submitted_output, dict):
                raise _ProbeFailure("invalid_response")
            task_id = submitted_output.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                raise _ProbeFailure("invalid_response")
            deadline = asyncio.get_running_loop().time() + self._poll_timeout
            while True:
                if asyncio.get_running_loop().time() >= deadline:
                    raise _ProbeFailure("timeout")
                if self._poll_interval:
                    await asyncio.sleep(self._poll_interval)
                polled = _json(
                    await self._client.get(
                        f"{self._root}/api/v1/tasks/{task_id}",
                        headers=self._headers,
                        timeout=60,
                    )
                )
                output = polled.get("output")
                if not isinstance(output, dict):
                    raise _ProbeFailure("invalid_response")
                status = output.get("task_status")
                if status == "SUCCEEDED":
                    if not isinstance(output.get("video_url"), str):
                        raise _ProbeFailure("invalid_response")
                    return _passed(model_id, protocol)
                if status in {"FAILED", "CANCELED", "UNKNOWN"}:
                    raise _ProbeFailure("provider_rejected")
                if status not in {"PENDING", "RUNNING"}:
                    raise _ProbeFailure("invalid_response")
        except (_ProbeFailure, httpx.HTTPError, TimeoutError) as exc:
            return _failed(model_id, protocol, exc)

    async def probe_tts(self) -> tuple[ProbeResult, bytes]:
        model_id = "qwen-audio-3.0-tts-plus"
        protocol = "dashscope_speech_synthesis"
        try:
            response = await self._client.post(
                f"{self._root}/api/v1/services/audio/tts/SpeechSynthesizer",
                headers=self._headers,
                json={
                    "model": model_id,
                    "input": {
                        "text": "你好。",
                        "voice": "longanhuan_v3.6",
                        "format": "wav",
                        "sample_rate": 16000,
                    },
                },
                timeout=180,
            )
            body = _json(response)
            output = body.get("output")
            if not isinstance(output, dict):
                raise _ProbeFailure("invalid_response")
            audio_value = output.get("audio")
            if not isinstance(audio_value, dict):
                raise _ProbeFailure("invalid_response")
            audio_url = audio_value.get("url")
            parsed_audio_url = urlsplit(audio_url) if isinstance(audio_url, str) else None
            if (
                parsed_audio_url is None
                or parsed_audio_url.scheme not in {"http", "https"}
                or not parsed_audio_url.hostname
            ):
                raise _ProbeFailure("invalid_response")
            audio_response = await self._client.get(audio_url, timeout=60)
            audio_response.raise_for_status()
            audio = audio_response.content
            if not audio:
                raise _ProbeFailure("invalid_response")
            return _passed(model_id, protocol), audio
        except (_ProbeFailure, httpx.HTTPError, TimeoutError) as exc:
            return _failed(model_id, protocol, exc), b""

    async def probe_asr(self, audio: bytes) -> ProbeResult:
        model_id = "qwen-audio-3.0-asr-flash"
        protocol = "dashscope_speech_recognition"
        try:
            if not audio:
                raise _ProbeFailure("dependency_failed")
            audio_data = "data:audio/wav;base64," + base64.b64encode(audio).decode(
                "ascii"
            )
            response = await self._client.post(
                f"{self._root}/api/v1/services/aigc/multimodal-generation/generation",
                headers={**self._headers, "X-DashScope-SSE": "disable"},
                json={
                    "model": model_id,
                    "input": {
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_audio",
                                        "input_audio": {"data": audio_data},
                                    }
                                ],
                            }
                        ]
                    },
                    "parameters": {"format": "wav", "sample_rate": "16000"},
                },
                timeout=180,
            )
            body = _json(response)
            output = body.get("output")
            if not isinstance(output, dict) or not (
                isinstance(output.get("text"), str)
                or isinstance(output.get("output"), dict)
            ):
                raise _ProbeFailure("invalid_response")
            return _passed(model_id, protocol)
        except (_ProbeFailure, httpx.HTTPError, TimeoutError) as exc:
            return _failed(model_id, protocol, exc)

    async def probe_realtime(self) -> ProbeResult:
        model_id = "qwen-audio-3.0-realtime-plus"
        protocol = "dashscope_realtime"
        try:
            parts = urlsplit(self._root)
            url = urlunsplit(
                (
                    "wss",
                    parts.netloc,
                    "/api-ws/v1/realtime",
                    f"model={model_id}",
                    "",
                )
            )
            async with self._websocket_connect(url, self._headers) as websocket:
                event = await asyncio.wait_for(websocket.recv(), timeout=30)
            if isinstance(event, bytes):
                event = event.decode("utf-8")
            value = json.loads(event)
            if not isinstance(value, dict) or not value:
                raise _ProbeFailure("invalid_response")
            return _passed(model_id, protocol)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            failure = _ProbeFailure("invalid_response")
            failure.__cause__ = exc
            return _failed(model_id, protocol, failure)
        except Exception as exc:  # noqa: BLE001 - sanitize connector failures.
            return _failed(model_id, protocol, exc)

    async def run_all(self) -> tuple[ProbeResult, ...]:
        results: list[ProbeResult] = []
        for model_id in _TEXT_MODELS:
            results.append(await self.probe_openai(model_id))
            results.append(await self.probe_anthropic(model_id))
        for model_id in (
            "qwen-image-3.0-pro",
            "wan2.7-image",
            "wan2.7-image-pro",
        ):
            result, _url = await self.probe_image(model_id)
            results.append(result)
        for model_id in (
            "happyhorse-1.1-i2v",
            "happyhorse-1.1-t2v",
            "happyhorse-1.1-r2v",
        ):
            results.append(await self.probe_video(model_id))
        tts_result, audio = await self.probe_tts()
        results.append(tts_result)
        results.append(await self.probe_realtime())
        results.append(await self.probe_asr(audio))
        return tuple(results)


def _load_env_file(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(name, value)


async def _main(env_file: Path) -> int:
    _load_env_file(env_file)
    required = (
        "ALIYUN_TOKEN_PLAN_OPENAI_API_KEY",
        "ALIYUN_TOKEN_PLAN_ANTHROPIC_API_KEY",
    )
    if any(not os.environ.get(name) for name in required):
        result = ProbeResult("configuration", "environment", "failed", "missing_credential")
        print(json.dumps(summarize_result(result), separators=(",", ":")))
        return 1
    async with httpx.AsyncClient() as client:
        runner = TokenPlanProbeRunner(
            client=client,
            api_key=os.environ[required[0]],
            anthropic_api_key=os.environ[required[1]],
            openai_base_url=os.environ.get(
                "ALIYUN_TOKEN_PLAN_OPENAI_API_BASE", _OPENAI_BASE
            ),
            anthropic_base_url=os.environ.get(
                "ALIYUN_TOKEN_PLAN_ANTHROPIC_API_BASE", _ANTHROPIC_BASE
            ),
        )
        results = await runner.run_all()
    for result in results:
        print(
            json.dumps(
                summarize_result(result),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    expected_records = sum(len(item.protocols) for item in probe_inventory())
    return 0 if len(results) == expected_records and all(
        result.status == "passed" for result in results
    ) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, required=True)
    arguments = parser.parse_args()
    sys.exit(asyncio.run(_main(arguments.env_file)))
