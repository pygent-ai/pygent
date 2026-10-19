"""Volcengine Ark video-generation wire codec (task-based contents API).

The Ark video endpoint is a three-phase lifecycle — create a task, poll its
status, then download the resulting video — with no streaming and no token
deltas, so the client owns the whole lifecycle inside ``invoke`` and hands the
adapter a terminal task JSON augmented with ``video_b64`` (the downloaded
video, base64-encoded). The invoker only reaches this client through the
non-stream path because video models declare an empty streaming capability.

Retry discipline: once a task is created, every failure surfaced to the
invoker uses a non-retryable ``ModelErrorKind`` — the outer ``RetryPolicy``
must never re-run ``invoke`` for an existing task, or the retry would create a
second billed task. Transient poll/download transport errors are therefore
retried inside the client until the poll window closes.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import math
import time
import urllib.parse
import urllib.request
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Self, cast

import httpx

from pygent.core import AIMessage, FrozenJsonObject, UserMessage, freeze_json_object
from pygent.tool import MediaBlock, MediaSource

from ._adapter_contracts import (
    MediaTransportCapabilities,
    ModelProviderRequest,
    ModelProviderResponse,
    ModelProviderStreamPart,
    _canonical_usage,
)
from ._json_sse_transport import _trust_environment_for_url
from .configuration import ModelSpec
from .openai_compatible import (
    _normalize_openai_error,
    _provider_status_error,
    _validate_no_forbidden_option_keys,
)
from .types import ModelErrorKind, ModelFailureReason, ModelProviderError

_PROTOCOL = "ark_video_generation"
_RESERVED_FIELDS = frozenset({"model", "content"})
_BEHAVIOR_FIELDS = frozenset(
    {"poll_interval_seconds", "poll_timeout_seconds", "max_output_bytes"}
)
_DEFAULT_POLL_INTERVAL_SECONDS = 2.0
_DEFAULT_POLL_TIMEOUT_SECONDS = 900.0
_DEFAULT_MAX_OUTPUT_BYTES = 100 * 1024 * 1024
_TASK_REQUEST_TIMEOUT = 60.0
_VIDEO_DOWNLOAD_TIMEOUT = 300.0
_DOWNLOAD_ATTEMPTS = 3
# ModelProviderError kinds the outer RetryPolicy may retry on; after task
# creation these must never escape from this client.
_RETRYABLE_KINDS = frozenset(
    {
        ModelErrorKind.TIMEOUT,
        ModelErrorKind.RATE_LIMIT,
        ModelErrorKind.UNAVAILABLE,
        ModelErrorKind.INCOMPLETE_RESPONSE,
    }
)


@dataclass(frozen=True, slots=True)
class _VideoBehavior:
    interval_seconds: float
    timeout_seconds: float
    max_output_bytes: int


def _positive_number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"provider option {label!r} must be a positive finite number")
    return float(value)


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"provider option {label!r} must be a positive integer")
    return value


def _behavior_from_options(
    options: Mapping[str, object], base: _VideoBehavior
) -> _VideoBehavior:
    return _VideoBehavior(
        interval_seconds=(
            _positive_number(options["poll_interval_seconds"], "poll_interval_seconds")
            if "poll_interval_seconds" in options
            else base.interval_seconds
        ),
        timeout_seconds=(
            _positive_number(options["poll_timeout_seconds"], "poll_timeout_seconds")
            if "poll_timeout_seconds" in options
            else base.timeout_seconds
        ),
        max_output_bytes=(
            _positive_int(options["max_output_bytes"], "max_output_bytes")
            if "max_output_bytes" in options
            else base.max_output_bytes
        ),
    )


def _sniffed_video_mime(data: bytes) -> str | None:
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return "video/mp4"
    if data.startswith(b"\x1a\x45\xdf\xa3"):
        return "video/webm"
    return None


def _task_error_detail(payload: Mapping[str, object]) -> str:
    error = payload.get("error")
    if isinstance(error, Mapping):
        code = error.get("code")
        message = error.get("message")
        if isinstance(code, str) and code:
            return f"{code}: {message}" if isinstance(message, str) else code
    return "no error detail provided"


class ArkVideoClient:
    """Small HTTP client for the Ark video task lifecycle."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        headers: Mapping[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
        verify_ssl: bool | None = None,
        poll_interval_seconds: float | None = None,
        poll_timeout_seconds: float | None = None,
        max_output_bytes: int | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url must be non-empty")
        if verify_ssl is not None and not isinstance(verify_ssl, bool):
            raise TypeError("verify_ssl must be a bool or None")
        if client is not None and verify_ssl is not None:
            raise ValueError("verify_ssl cannot be set with an injected HTTP client")
        self._base_behavior = _VideoBehavior(
            interval_seconds=(
                _positive_number(poll_interval_seconds, "poll_interval_seconds")
                if poll_interval_seconds is not None
                else _DEFAULT_POLL_INTERVAL_SECONDS
            ),
            timeout_seconds=(
                _positive_number(poll_timeout_seconds, "poll_timeout_seconds")
                if poll_timeout_seconds is not None
                else _DEFAULT_POLL_TIMEOUT_SECONDS
            ),
            max_output_bytes=(
                _positive_int(max_output_bytes, "max_output_bytes")
                if max_output_bytes is not None
                else _DEFAULT_MAX_OUTPUT_BYTES
            ),
        )
        self._root = base_url.rstrip("/")
        self._tasks_endpoint = f"{self._root}/contents/generations/tasks"
        self._headers = dict(headers or {})
        if api_key is not None:
            self._headers.setdefault("Authorization", f"Bearer {api_key}")
        if client is None:
            self._client = httpx.AsyncClient(
                headers=self._headers,
                verify=True if verify_ssl is None else verify_ssl,
                trust_env=_trust_environment_for_url(self._root),
                timeout=httpx.Timeout(_TASK_REQUEST_TIMEOUT),
            )
            self._owns_client = True
        else:
            self._client = client
            self._owns_client = False
        self._closed = False

    async def invoke(
        self, model: ModelSpec, payload: FrozenJsonObject
    ) -> FrozenJsonObject:
        self._ensure_open()
        behavior = _behavior_from_options(
            cast(
                Mapping[str, object],
                cast(FrozenJsonObject, model.provider_options).to_dict(),
            ),
            self._base_behavior,
        )
        # Outside the task lifecycle, transport and HTTP failures keep their
        # natural retry semantics: no task exists yet, so an outer retry
        # cannot duplicate a billed task.
        created = await self._request_json(
            "POST", self._tasks_endpoint, payload.to_dict()
        )
        task_id = created.get("id")
        if not isinstance(task_id, str) or not task_id:
            raise ModelProviderError(
                ModelErrorKind.INVALID_RESPONSE,
                "video task response has no task id",
                reason_code=ModelFailureReason.PROVIDER_PAYLOAD_INVALID,
            )
        return await self._await_task(f"{self._tasks_endpoint}/{task_id}", behavior)

    async def stream(
        self, model: ModelSpec, payload: FrozenJsonObject
    ) -> AsyncIterator[FrozenJsonObject]:
        raise ModelProviderError(
            ModelErrorKind.INVALID_REQUEST,
            "ark video protocol does not support streaming",
            reason_code=ModelFailureReason.MEDIA_TRANSPORT_UNSUPPORTED,
        )
        yield freeze_json_object()  # pragma: no cover - generator return typing

    async def aclose(self) -> None:
        if self._closed or not self._owns_client:
            return
        self._closed = True
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        self._ensure_open()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("model provider client is closed")

    async def _await_task(
        self, task_url: str, behavior: _VideoBehavior
    ) -> FrozenJsonObject:
        deadline = time.monotonic() + behavior.timeout_seconds
        while True:
            if time.monotonic() >= deadline:
                raise ModelProviderError(
                    ModelErrorKind.INVALID_RESPONSE,
                    "video task did not reach a terminal status within the poll window",
                    reason_code=ModelFailureReason.PROVIDER_TIMEOUT,
                )
            await asyncio.sleep(behavior.interval_seconds)
            try:
                polled = await self._request_json(
                    "GET", task_url, None, timeout=_TASK_REQUEST_TIMEOUT
                )
            except ModelProviderError as exc:
                if exc.kind not in _RETRYABLE_KINDS:
                    raise
                # Transient poll failure: keep waiting inside the poll window
                # instead of surfacing a retryable failure to the invoker.
                if time.monotonic() >= deadline:
                    raise ModelProviderError(
                        ModelErrorKind.INVALID_RESPONSE,
                        "video task polling kept failing inside the poll window",
                        reason_code=ModelFailureReason.PROVIDER_TIMEOUT,
                    ) from None
                continue
            except httpx.TransportError as exc:
                if time.monotonic() >= deadline:
                    raise ModelProviderError(
                        ModelErrorKind.INVALID_RESPONSE,
                        "video task polling kept failing inside the poll window",
                        reason_code=ModelFailureReason.PROVIDER_TIMEOUT,
                    ) from exc
                continue
            status = polled.get("status")
            if status == "succeeded":
                return await self._attach_video(polled, behavior)
            if status in ("failed", "cancelled"):
                raise ModelProviderError(
                    ModelErrorKind.INVALID_RESPONSE,
                    f"video task {status}: {_task_error_detail(polled.to_dict())}",
                    reason_code=ModelFailureReason.INVALID_PROVIDER_RESPONSE,
                )
            if status not in ("queued", "running"):
                raise ModelProviderError(
                    ModelErrorKind.INVALID_RESPONSE,
                    f"video task returned unknown status {status!r}",
                    reason_code=ModelFailureReason.PROVIDER_PAYLOAD_INVALID,
                )

    async def _request_json(
        self,
        method: str,
        url: str,
        payload: Mapping[str, object] | None,
        *,
        timeout: float = _TASK_REQUEST_TIMEOUT,
    ) -> FrozenJsonObject:
        response = await self._client.request(
            method, url, json=payload, headers=self._headers or None, timeout=timeout
        )
        if not response.is_success:
            raise _provider_status_error(response.status_code, response.text)
        try:
            decoded = response.json()
        except ValueError as exc:
            raise ModelProviderError(
                ModelErrorKind.INVALID_RESPONSE,
                "provider returned invalid JSON",
                reason_code=ModelFailureReason.PROVIDER_PAYLOAD_INVALID,
            ) from exc
        if not isinstance(decoded, Mapping):
            raise ModelProviderError(
                ModelErrorKind.INVALID_RESPONSE,
                "provider response must be a JSON object",
                reason_code=ModelFailureReason.PROVIDER_PAYLOAD_INVALID,
            )
        return freeze_json_object(decoded)

    async def _attach_video(
        self, polled: FrozenJsonObject, behavior: _VideoBehavior
    ) -> FrozenJsonObject:
        body = polled.to_dict()
        content = body.get("content")
        video_url = content.get("video_url") if isinstance(content, Mapping) else None
        if not isinstance(video_url, str) or not video_url:
            raise ModelProviderError(
                ModelErrorKind.INVALID_RESPONSE,
                "succeeded video task has no video_url",
                reason_code=ModelFailureReason.PROVIDER_PAYLOAD_INVALID,
            )
        parts = urllib.parse.urlsplit(video_url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ModelProviderError(
                ModelErrorKind.INVALID_RESPONSE,
                "video_url is not an absolute HTTP(S) URL",
                reason_code=ModelFailureReason.PROVIDER_PAYLOAD_INVALID,
            )
        media = await self._download_video(video_url)
        if len(media) > behavior.max_output_bytes:
            raise ModelProviderError(
                ModelErrorKind.INVALID_RESPONSE,
                "video output exceeds the configured size limit",
                reason_code=ModelFailureReason.MEDIA_CONTENT_INVALID,
            )
        if _sniffed_video_mime(media) is None:
            raise ModelProviderError(
                ModelErrorKind.INVALID_RESPONSE,
                "video output is not a recognizable video container",
                reason_code=ModelFailureReason.MEDIA_CONTENT_INVALID,
            )
        result = dict(body)
        result["video_b64"] = base64.b64encode(media).decode("ascii")
        return freeze_json_object(result)

    async def _download_video(self, video_url: str) -> bytes:
        # Downloading the result of a terminal task is idempotent and free, so
        # transient transport failures may retry here without billing risk.
        for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
            try:
                response = await self._client.get(
                    video_url,
                    headers=self._headers or None,
                    timeout=_VIDEO_DOWNLOAD_TIMEOUT,
                )
            except httpx.TransportError as exc:
                if attempt == _DOWNLOAD_ATTEMPTS:
                    raise ModelProviderError(
                        ModelErrorKind.INVALID_RESPONSE,
                        "video result download kept failing",
                        reason_code=ModelFailureReason.PROVIDER_OUTCOME_UNKNOWN,
                    ) from exc
                await asyncio.sleep(0.1 * attempt)
                continue
            if response.is_success:
                return response.content
            error = _provider_status_error(response.status_code, response.text)
            if (
                error.kind not in _RETRYABLE_KINDS
                or attempt == _DOWNLOAD_ATTEMPTS
            ):
                raise error
            await asyncio.sleep(0.1 * attempt)
        raise ModelProviderError(
            ModelErrorKind.INVALID_RESPONSE,
            "video result download kept failing",
            reason_code=ModelFailureReason.PROVIDER_OUTCOME_UNKNOWN,
        )


class ArkVideoAdapter:
    """Strict codec for the Ark task-based video-generation wire."""

    protocol = _PROTOCOL

    # Image input (i2v/r2v routes) is not wired through the invoker yet;
    # preflight degrades user media to not_viewed notes instead of sending it.
    media_transport = MediaTransportCapabilities()

    def validate_model(self, model: ModelSpec) -> None:
        options = cast(
            Mapping[str, object],
            cast(FrozenJsonObject, model.provider_options).to_dict(),
        )
        conflicts = set(options) & _RESERVED_FIELDS
        if conflicts:
            raise ValueError(
                "provider options cannot override reserved fields: "
                + ", ".join(sorted(conflicts))
            )
        for key in _BEHAVIOR_FIELDS:
            if key not in options:
                continue
            if key == "max_output_bytes":
                _positive_int(options[key], key)
            else:
                _positive_number(options[key], key)
        _validate_no_forbidden_option_keys(options)

    def build_request(self, request: ModelProviderRequest) -> FrozenJsonObject:
        self.validate_model(request.model)
        if request.tools:
            raise ModelProviderError(
                ModelErrorKind.INVALID_REQUEST,
                "video protocol does not accept tools",
            )
        if request.generation.response_schema is not None:
            raise ModelProviderError(
                ModelErrorKind.INVALID_REQUEST,
                "video protocol does not accept response schemas",
            )
        message = request.message
        if not isinstance(message, UserMessage):
            raise ModelProviderError(
                ModelErrorKind.INVALID_REQUEST,
                "video protocol requires a user message prompt",
            )
        if message.media:
            raise ModelProviderError(
                ModelErrorKind.INVALID_REQUEST,
                "ark video adapter does not deliver image input (i2v/r2v)",
                reason_code=ModelFailureReason.MEDIA_TRANSPORT_UNSUPPORTED,
            )
        body: dict[str, object] = {
            "model": request.model.model_id,
            "content": [{"type": "text", "text": message.content}],
        }
        options = cast(
            Mapping[str, object],
            cast(FrozenJsonObject, request.model.provider_options).to_dict(),
        )
        body.update(
            {
                key: value
                for key, value in options.items()
                if key not in _BEHAVIOR_FIELDS
            }
        )
        return freeze_json_object(body)

    def parse_response(
        self, request: ModelProviderRequest, payload: FrozenJsonObject
    ) -> ModelProviderResponse:
        body = payload.to_dict()
        status = body.get("status")
        if status != "succeeded":
            raise ModelProviderError(
                ModelErrorKind.INVALID_RESPONSE,
                f"video task did not succeed: {_task_error_detail(body)}",
                reason_code=ModelFailureReason.INVALID_PROVIDER_RESPONSE,
            )
        encoded = body.get("video_b64")
        if not isinstance(encoded, str) or not encoded:
            raise ModelProviderError(
                ModelErrorKind.INVALID_RESPONSE,
                "video task response has no inline video payload",
                reason_code=ModelFailureReason.PROVIDER_PAYLOAD_INVALID,
            )
        try:
            media = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ModelProviderError(
                ModelErrorKind.INVALID_RESPONSE,
                "video task payload is not valid Base64",
                reason_code=ModelFailureReason.MEDIA_CONTENT_INVALID,
            ) from exc
        mime_type = _sniffed_video_mime(media)
        if mime_type is None:
            raise ModelProviderError(
                ModelErrorKind.INVALID_RESPONSE,
                "video task payload is not a recognizable video container",
                reason_code=ModelFailureReason.MEDIA_CONTENT_INVALID,
            )
        request_id = body.get("id")
        message = AIMessage(
            content="",
            media=(
                MediaBlock(
                    media_type="video",
                    mime_type=mime_type,
                    source=MediaSource.inline(media),
                ),
            ),
            metadata={"model_key": request.model_key},
        )
        return ModelProviderResponse(
            message=message,
            usage=_canonical_usage(body.get("usage")),
            provider_request_id=request_id if isinstance(request_id, str) else None,
            finish_reason="stop",
        )

    def create_stream_decoder(
        self, request: ModelProviderRequest
    ) -> _ArkVideoStreamDecoder:
        return _ArkVideoStreamDecoder()

    def normalize_error(self, error: BaseException) -> ModelErrorKind:
        return _normalize_openai_error(error)


class _ArkVideoStreamDecoder:
    def feed(self, payload: FrozenJsonObject) -> tuple[ModelProviderStreamPart, ...]:
        raise ModelProviderError(
            ModelErrorKind.INVALID_REQUEST,
            "ark video protocol does not support streaming",
            reason_code=ModelFailureReason.MEDIA_TRANSPORT_UNSUPPORTED,
        )

    def finish(self) -> tuple[ModelProviderStreamPart, ...]:
        return ()


def ark_video_adapters() -> dict[str, ArkVideoAdapter]:
    return {_PROTOCOL: ArkVideoAdapter()}


__all__ = ["ArkVideoAdapter", "ArkVideoClient", "ark_video_adapters"]
