from __future__ import annotations

import base64
import binascii
import io
import json
import math
import wave
from collections.abc import Mapping
from typing import Protocol, cast

import httpx

from tests.live.az_conformance.results import ErrorKind, ProbeResult
from tests.live.az_conformance.runner import ProbeContext
from tests.live.az_conformance.schemas import AzRoute

_MAX_MEDIA_BYTES = 20 * 1024 * 1024
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1Pe"
    "AAAADElEQVR4nGNgYPgPAAEDAQAIicLsAAAAAElFTkSuQmCC"
)
_VIDEO_FRAME = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2"
    "AAAAFUlEQVR4nGNgYPhPIhrVMKph2GoAAJLb/wFh5Z4R"
    "AAAAAElFTkSuQmCC"
)


class RawProbeClient(Protocol):
    audio_fixture: bytes | None

    async def request(
        self, method: str, path: str, **kwargs: object
    ) -> httpx.Response: ...

    async def websocket_exchange(
        self, path: str, event: dict[str, object]
    ) -> object: ...


def _client(context: ProbeContext) -> RawProbeClient:
    if context.client is None:
        raise ValueError("probe client is not configured")
    return cast(RawProbeClient, context.client)


def _result(
    context: ProbeContext,
    route: AzRoute,
    *,
    error_kind: ErrorKind | None = None,
) -> ProbeResult:
    return ProbeResult(
        snapshot_sha256=context.snapshot_sha256,
        source_revision=context.source_revision,
        route_id=route.route_id,
        canonical_provider=route.canonical_provider,
        canonical_model_id=route.canonical_model_id,
        protocol=context.protocol,
        scenario=context.scenario,
        status="passed" if error_kind is None else "failed",
        attempts=context.attempt,
        error_kind=error_kind,
    )


def _http_error(response: httpx.Response) -> ErrorKind | None:
    if response.status_code < 400:
        return None
    if response.status_code == 401:
        return ErrorKind.AUTHENTICATION
    if response.status_code == 403:
        return ErrorKind.PERMISSION
    if response.status_code == 429:
        return ErrorKind.RATE_LIMIT
    if response.status_code >= 500:
        return ErrorKind.GATEWAY_UNAVAILABLE
    return ErrorKind.INVALID_REQUEST


def _json(response: httpx.Response) -> Mapping[str, object]:
    value = response.json()
    if not isinstance(value, Mapping):
        raise TypeError("response body must be an object")
    return cast(Mapping[str, object], value)


def _decode_b64(value: object) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError("base64 media must be a non-empty string")
    try:
        result = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("invalid base64 media") from exc
    return _bounded(result)


def _bounded(value: bytes) -> bytes:
    if not value or len(value) > _MAX_MEDIA_BYTES:
        raise ValueError("media byte length is invalid")
    return value


async def _media_bytes(
    client: RawProbeClient, response: httpx.Response
) -> bytes:
    content_type = response.headers.get("content-type", "")
    looks_like_json = response.content.lstrip().startswith((b"{", b"["))
    if (content_type and "json" not in content_type) or (
        not content_type and not looks_like_json
    ):
        return _bounded(response.content)
    payload = _json(response)
    data = payload.get("data")
    item: Mapping[str, object] = payload
    if isinstance(data, list) and data and isinstance(data[0], Mapping):
        item = cast(Mapping[str, object], data[0])
    for key in ("b64_json", "b64", "data"):
        if key in item and isinstance(item[key], str):
            return _decode_b64(item[key])
    url = item.get("url")
    if isinstance(url, str) and url.startswith(("https://", "http://")):
        downloaded = await client.request("GET", url)
        error = _http_error(downloaded)
        if error is not None:
            raise ValueError("media download failed")
        return _bounded(downloaded.content)
    raise ValueError("response does not contain media")


def _is_png(value: bytes) -> bool:
    return value.startswith(b"\x89PNG\r\n\x1a\n") and value.endswith(
        b"IEND\xaeB\x60\x82"
    )


def _is_audio(value: bytes) -> bool:
    return (
        value.startswith((b"ID3", b"OggS", b"fLaC"))
        or (value.startswith(b"RIFF") and b"WAVE" in value[:64])
    )


def _audio_from_sse(value: bytes) -> bytes:
    chunks: list[bytes] = []
    for line in value.decode("utf-8").splitlines():
        if not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if not data or data == "[DONE]":
            continue
        payload = json.loads(data)
        if not isinstance(payload, Mapping):
            raise TypeError("audio SSE event must be an object")
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0]
        if not isinstance(choice, Mapping):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, Mapping):
            continue
        audio = delta.get("audio")
        if not isinstance(audio, Mapping) or "data" not in audio:
            continue
        chunks.append(_decode_b64(audio.get("data")))
    pcm = b"".join(chunks)
    if not pcm:
        raise ValueError("audio SSE response has no audio data")
    if _is_audio(pcm):
        return pcm
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(24_000)
        target.writeframes(pcm)
    return output.getvalue()


def _text_from_sse(value: bytes) -> str:
    chunks: list[str] = []
    for line in value.decode("utf-8").splitlines():
        if not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if not data or data == "[DONE]":
            continue
        payload = json.loads(data)
        if not isinstance(payload, Mapping):
            raise TypeError("text SSE event must be an object")
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0]
        if not isinstance(choice, Mapping):
            continue
        delta = choice.get("delta")
        if isinstance(delta, Mapping) and isinstance(delta.get("content"), str):
            chunks.append(delta["content"])
    text = "".join(chunks).strip()
    if not text:
        raise ValueError("text SSE response has no content")
    return text


def _is_video(value: bytes) -> bool:
    return len(value) >= 12 and value[4:8] == b"ftyp"


async def embedding_probe(context: ProbeContext, route: AzRoute) -> ProbeResult:
    client = _client(context)
    dimensions: list[int] = []
    try:
        for text in ("probe", "probe again"):
            response = await client.request(
                "POST",
                "/embeddings",
                json={"model": route.route_id, "input": text},
            )
            error = _http_error(response)
            if error is not None:
                return _result(context, route, error_kind=error)
            data = _json(response).get("data")
            if not isinstance(data, list) or not data or not isinstance(data[0], Mapping):
                raise ValueError("embedding response has no data")
            vector = data[0].get("embedding")
            if (
                not isinstance(vector, list)
                or not vector
                or any(
                    not isinstance(item, (int, float))
                    or isinstance(item, bool)
                    or not math.isfinite(item)
                    for item in vector
                )
            ):
                raise ValueError("embedding vector is invalid")
            dimensions.append(len(vector))
        if dimensions[0] != dimensions[1]:
            raise ValueError("embedding dimension is unstable")
    except (httpx.HTTPError, OSError, TypeError, ValueError, AttributeError):
        return _result(context, route, error_kind=ErrorKind.INVALID_RESPONSE)
    return _result(context, route)


async def image_output_probe(context: ProbeContext, route: AzRoute) -> ProbeResult:
    client = _client(context)
    try:
        response = await client.request(
            "POST",
            "/images/generations",
            json={"model": route.route_id, "prompt": "A single blue square."},
        )
        error = _http_error(response)
        if error is not None:
            return _result(context, route, error_kind=error)
        if not _is_png(await _media_bytes(client, response)):
            raise ValueError("image is not a valid PNG")
    except (httpx.HTTPError, OSError, TypeError, ValueError, AttributeError):
        return _result(context, route, error_kind=ErrorKind.INVALID_RESPONSE)
    return _result(context, route)


async def image_edit_probe(context: ProbeContext, route: AzRoute) -> ProbeResult:
    client = _client(context)
    try:
        response = await client.request(
            "POST",
            "/images/edits",
            data={"model": route.route_id, "prompt": "Keep the square blue."},
            files={"image": ("probe.png", _PNG, "image/png")},
        )
        error = _http_error(response)
        if error is not None:
            return _result(context, route, error_kind=error)
        if not _is_png(await _media_bytes(client, response)):
            raise ValueError("edited image is not a valid PNG")
    except (httpx.HTTPError, OSError, TypeError, ValueError, AttributeError):
        return _result(context, route, error_kind=ErrorKind.INVALID_RESPONSE)
    return _result(context, route)


async def audio_output_probe(context: ProbeContext, route: AzRoute) -> ProbeResult:
    client = _client(context)
    try:
        model_id = (route.canonical_model_id or "").lower()
        conversational = (
            any(marker in model_id for marker in ("audio", "omni", "realtime"))
            and "tts" not in model_id
        )
        if conversational:
            if route.canonical_provider == "alibaba_cloud":
                voice = "Chelsie" if "qwen-omni-turbo" in model_id else "Tina"
            else:
                voice = "alloy"
            streaming_audio = route.canonical_provider == "alibaba_cloud"
            response = await client.request(
                "POST",
                "/chat/completions",
                json={
                    "model": route.route_id,
                    "messages": [
                        {"role": "user", "content": "Say: Pygent audio probe."}
                    ],
                    "modalities": ["text", "audio"],
                    "audio": {"voice": voice, "format": "wav"},
                    "max_tokens": 1024,
                    **(
                        {"stream": True, "stream_options": {"include_usage": True}}
                        if streaming_audio
                        else {}
                    ),
                },
            )
            error = _http_error(response)
            if error is not None:
                return _result(context, route, error_kind=error)
            if streaming_audio:
                media = _audio_from_sse(response.content)
                client.audio_fixture = media
                return _result(context, route)
            choices = _json(response).get("choices")
            if (
                not isinstance(choices, list)
                or not choices
                or not isinstance(choices[0], Mapping)
                or not isinstance(choices[0].get("message"), Mapping)
            ):
                raise ValueError("audio chat response has no message")
            audio = choices[0]["message"].get("audio")
            if not isinstance(audio, Mapping):
                raise ValueError("audio chat response has no audio")
            media = _decode_b64(audio.get("data"))
            if not _is_audio(media):
                raise ValueError("audio chat response is not decodable")
            client.audio_fixture = media
            return _result(context, route)
        response = await client.request(
            "POST",
            "/audio/speech",
            json={
                "model": route.route_id,
                "input": "Pygent audio probe.",
                "voice": "alloy",
                "response_format": "wav",
            },
        )
        error = _http_error(response)
        if error is not None:
            return _result(context, route, error_kind=error)
        media = await _media_bytes(client, response)
        if not _is_audio(media):
            raise ValueError("audio response is not decodable")
        client.audio_fixture = media
    except (httpx.HTTPError, OSError, TypeError, ValueError, AttributeError):
        return _result(context, route, error_kind=ErrorKind.INVALID_RESPONSE)
    return _result(context, route)


async def audio_input_probe(context: ProbeContext, route: AzRoute) -> ProbeResult:
    client = _client(context)
    if not client.audio_fixture:
        return _result(context, route, error_kind=ErrorKind.DEPENDENCY_FAILED)
    try:
        encoded_audio = base64.b64encode(client.audio_fixture).decode("ascii")
        captioner = "captioner" in (route.canonical_model_id or "").lower()
        content: list[dict[str, object]] = [
            {
                "type": "input_audio",
                "input_audio": {
                    "data": (
                        f"data:;base64,{encoded_audio}" if captioner else encoded_audio
                    ),
                    "format": "wav",
                },
            }
        ]
        if not captioner:
            content.insert(
                0,
                {
                    "type": "text",
                    "text": "Transcribe the attached audio.",
                },
            )
        response = await client.request(
            "POST",
            "/chat/completions",
            json={
                "model": route.route_id,
                "messages": [
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                "max_tokens": 1024,
            },
        )
        error = _http_error(response)
        if error is not None:
            return _result(context, route, error_kind=error)
        choices = _json(response).get("choices")
        if (
            not isinstance(choices, list)
            or not choices
            or not isinstance(choices[0], Mapping)
            or not isinstance(choices[0].get("message"), Mapping)
            or not isinstance(choices[0]["message"].get("content"), str)
            or not choices[0]["message"]["content"].strip()
        ):
            raise ValueError("transcription text is empty")
    except (httpx.HTTPError, OSError, TypeError, ValueError, AttributeError):
        return _result(context, route, error_kind=ErrorKind.INVALID_RESPONSE)
    return _result(context, route)


async def realtime_probe(context: ProbeContext, route: AzRoute) -> ProbeResult:
    client = _client(context)
    try:
        event = await client.websocket_exchange(
            f"/realtime?model={route.route_id}",
            {
                "type": "response.create",
                "response": {"modalities": ["text"], "instructions": "Say probe."},
            },
        )
        if not isinstance(event, Mapping):
            raise TypeError("realtime event must be an object")
        event_type = event.get("type")
        if not isinstance(event_type, str) or not event_type:
            raise ValueError("realtime event type is missing")
    except (httpx.HTTPError, OSError, TypeError, ValueError, AttributeError):
        return _result(context, route, error_kind=ErrorKind.PROTOCOL_MISMATCH)
    return _result(context, route)


async def video_input_probe(context: ProbeContext, route: AzRoute) -> ProbeResult:
    client = _client(context)
    frame = (
        f"data:image/png;base64,{base64.b64encode(_VIDEO_FRAME).decode('ascii')}"
    )
    try:
        response = await client.request(
            "POST",
            "/chat/completions",
            json={
                "model": route.route_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": [frame] * 4},
                            {
                                "type": "text",
                                "text": "Describe the change between these frames.",
                            },
                        ],
                    }
                ],
                "modalities": ["text"],
                "stream": True,
                "stream_options": {"include_usage": True},
                "max_tokens": 1024,
            },
        )
        error = _http_error(response)
        if error is not None:
            return _result(context, route, error_kind=error)
        _text_from_sse(response.content)
    except (httpx.HTTPError, OSError, TypeError, ValueError, AttributeError):
        return _result(context, route, error_kind=ErrorKind.INVALID_RESPONSE)
    return _result(context, route)


async def video_output_probe(context: ProbeContext, route: AzRoute) -> ProbeResult:
    client = _client(context)
    try:
        response = await client.request(
            "POST",
            "/videos",
            json={"model": route.route_id, "prompt": "A blue square, static."},
        )
        error = _http_error(response)
        if error is not None:
            return _result(context, route, error_kind=error)
        payload = _json(response)
        task_id = payload.get("id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("video task id is missing")
        for _ in range(20):
            status = payload.get("status")
            if status in {"failed", "cancelled"}:
                return _result(
                    context, route, error_kind=ErrorKind.CAPABILITY_MISMATCH
                )
            if status in {"completed", "succeeded"}:
                media = await _media_bytes(client, response)
                if not _is_video(media):
                    raise ValueError("video response is not decodable")
                return _result(context, route)
            response = await client.request("GET", f"/videos/{task_id}")
            error = _http_error(response)
            if error is not None:
                return _result(context, route, error_kind=error)
            payload = _json(response)
        return _result(context, route, error_kind=ErrorKind.TIMEOUT)
    except (httpx.HTTPError, OSError, TypeError, ValueError, AttributeError):
        return _result(context, route, error_kind=ErrorKind.INVALID_RESPONSE)


__all__ = [
    "RawProbeClient",
    "audio_input_probe",
    "audio_output_probe",
    "embedding_probe",
    "image_edit_probe",
    "image_output_probe",
    "realtime_probe",
    "video_input_probe",
    "video_output_probe",
]
