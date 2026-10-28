"""OpenAI Images wire codec shared by image-generation providers."""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator, Mapping
from typing import Self, cast
from urllib.parse import urlsplit

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
from ._json_sse_transport import _HTTPResponseError, _JsonSSETransport
from .configuration import ModelSpec
from .openai_compatible import (
    _normalize_openai_error,
    _provider_status_error,
    _validate_no_forbidden_option_keys,
)
from .types import ModelErrorKind, ModelFailureReason, ModelProviderError

_PROTOCOL = "openai_images"
_IMAGES_RESERVED_FIELDS = frozenset({"model", "prompt", "response_format"})
_IMAGE_FORMATS = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "webp": "image/webp",
}
_DEFAULT_IMAGE_MIME = "image/png"


class OpenAIImagesClient:
    """Small HTTP client for the OpenAI Images wire (generations endpoint)."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        headers: Mapping[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
        verify_ssl: bool | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url must be non-empty")
        if verify_ssl is not None and not isinstance(verify_ssl, bool):
            raise TypeError("verify_ssl must be a bool or None")
        if client is not None and verify_ssl is not None:
            raise ValueError("verify_ssl cannot be set with an injected HTTP client")
        request_headers = dict(headers or {})
        if api_key is not None:
            request_headers.setdefault("Authorization", f"Bearer {api_key}")
        api_root = base_url.rstrip("/")
        self._endpoint = f"{api_root}/images/generations"
        # Values this client sent to the provider; provider error bodies and
        # transport failure text are untrusted and may echo them, so they are
        # redacted before any error text is kept.
        secrets = [
            base_url,
            api_root,
            self._endpoint,
            urlsplit(base_url).hostname or "",
            api_key or "",
        ]
        self._sent_secrets = tuple(secret for secret in secrets if secret)
        self._transport = _JsonSSETransport(
            headers=request_headers,
            client=client,
            verify_ssl=verify_ssl,
            trust_env_url=api_root,
            secrets=self._sent_secrets,
        )

    async def invoke(
        self, model: ModelSpec, payload: FrozenJsonObject
    ) -> FrozenJsonObject:
        try:
            return await self._transport.request_json(
                "POST", self._endpoint, payload.to_dict()
            )
        except _HTTPResponseError as exc:
            raise _provider_status_error(
                exc.status, exc.body, self._sent_secrets
            ) from None
        except (TypeError, ValueError) as exc:
            raise ModelProviderError(
                ModelErrorKind.INVALID_RESPONSE,
                "provider returned invalid JSON",
                reason_code=ModelFailureReason.PROVIDER_PAYLOAD_INVALID,
            ) from exc

    async def stream(
        self, model: ModelSpec, payload: FrozenJsonObject
    ) -> AsyncIterator[FrozenJsonObject]:
        raise ModelProviderError(
            ModelErrorKind.INVALID_REQUEST,
            "images protocol does not support streaming",
            reason_code=ModelFailureReason.MEDIA_TRANSPORT_UNSUPPORTED,
        )
        yield freeze_json_object()  # pragma: no cover - generator return typing

    async def aclose(self) -> None:
        await self._transport.aclose()

    async def __aenter__(self) -> Self:
        self._transport._ensure_open()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()


class OpenAIImagesAdapter:
    """Strict codec for the OpenAI Images wire (text prompt to inline image)."""

    protocol = _PROTOCOL

    # Image input (edits) is not wired through the invoker yet; preflight
    # degrades user media to not_viewed notes instead of sending it.
    media_transport = MediaTransportCapabilities()

    def validate_model(self, model: ModelSpec) -> None:
        options = cast(FrozenJsonObject, model.provider_options)
        conflicts = set(options) & _IMAGES_RESERVED_FIELDS
        if conflicts:
            raise ValueError(
                "provider options cannot override reserved fields: "
                + ", ".join(sorted(conflicts))
            )
        _validate_no_forbidden_option_keys(options)

    def build_request(self, request: ModelProviderRequest) -> FrozenJsonObject:
        self.validate_model(request.model)
        if request.tools:
            raise ModelProviderError(
                ModelErrorKind.INVALID_REQUEST,
                "images protocol does not accept tools",
            )
        if request.generation.response_schema is not None:
            raise ModelProviderError(
                ModelErrorKind.INVALID_REQUEST,
                "images protocol does not accept response schemas",
            )
        message = request.message
        if not isinstance(message, UserMessage):
            raise ModelProviderError(
                ModelErrorKind.INVALID_REQUEST,
                "images protocol requires a user message prompt",
            )
        if message.media:
            raise ModelProviderError(
                ModelErrorKind.INVALID_REQUEST,
                "images adapter does not deliver image input",
                reason_code=ModelFailureReason.MEDIA_TRANSPORT_UNSUPPORTED,
            )
        body: dict[str, object] = {
            "model": request.model.model_id,
            "prompt": message.content,
            "response_format": "b64_json",
        }
        options = cast(FrozenJsonObject, request.model.provider_options)
        body.update(cast(dict[str, object], options.to_dict()))
        return freeze_json_object(body)

    def parse_response(
        self, request: ModelProviderRequest, payload: FrozenJsonObject
    ) -> ModelProviderResponse:
        body = payload.to_dict()
        data = body.get("data")
        if not isinstance(data, list) or not data:
            raise ModelProviderError(
                ModelErrorKind.INVALID_RESPONSE,
                "images response has no data array",
                reason_code=ModelFailureReason.COMPLETION_SHAPE_INVALID,
            )
        blocks: list[MediaBlock] = []
        for item in data:
            if not isinstance(item, Mapping):
                raise ModelProviderError(
                    ModelErrorKind.INVALID_RESPONSE,
                    "images response data item must be an object",
                    reason_code=ModelFailureReason.COMPLETION_SHAPE_INVALID,
                )
            blocks.append(
                MediaBlock(
                    media_type="image",
                    mime_type=_image_mime_type(item),
                    source=MediaSource(
                        kind="inline",
                        base64_data=_inline_image_data(item),
                    ),
                )
            )
        message = AIMessage(
            content="",
            media=tuple(blocks),
            metadata={"model_key": request.model_key},
        )
        return ModelProviderResponse(
            message=message,
            usage=_canonical_usage(body.get("usage")),
            finish_reason="stop",
        )

    def create_stream_decoder(self, request: ModelProviderRequest) -> _ImagesStreamDecoder:
        return _ImagesStreamDecoder()

    def normalize_error(self, error: BaseException) -> ModelErrorKind:
        return _normalize_openai_error(error)


def _inline_image_data(item: Mapping[str, object]) -> str:
    value = item.get("b64_json")
    if isinstance(value, str) and value:
        return value
    if isinstance(item.get("url"), str):
        raise ModelProviderError(
            ModelErrorKind.INVALID_RESPONSE,
            "images response returned a URL instead of inline media data",
            reason_code=ModelFailureReason.MEDIA_CONTENT_INVALID,
        )
    raise ModelProviderError(
        ModelErrorKind.INVALID_RESPONSE,
        "images response item has no inline media data",
        reason_code=ModelFailureReason.MEDIA_CONTENT_INVALID,
    )


def _image_mime_type(item: Mapping[str, object]) -> str:
    data = base64.b64decode(_inline_image_data(item), validate=True)
    sniffed = _sniffed_image_mime(data)
    if sniffed is not None:
        return sniffed
    value = item.get("output_format")
    if value is None:
        return _DEFAULT_IMAGE_MIME
    if not isinstance(value, str) or value.lower() not in _IMAGE_FORMATS:
        raise ModelProviderError(
            ModelErrorKind.INVALID_RESPONSE,
            "images response item has an unsupported output format",
            reason_code=ModelFailureReason.MEDIA_MIME_TYPE_INVALID,
        )
    return _IMAGE_FORMATS[value.lower()]


def _sniffed_image_mime(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


class _ImagesStreamDecoder:
    def feed(self, payload: FrozenJsonObject) -> tuple[ModelProviderStreamPart, ...]:
        raise ModelProviderError(
            ModelErrorKind.INVALID_REQUEST,
            "images protocol does not support streaming",
            reason_code=ModelFailureReason.MEDIA_TRANSPORT_UNSUPPORTED,
        )

    def finish(self) -> tuple[ModelProviderStreamPart, ...]:
        return ()


def openai_images_adapters() -> dict[str, OpenAIImagesAdapter]:
    return {_PROTOCOL: OpenAIImagesAdapter()}


__all__ = [
    "OpenAIImagesAdapter",
    "OpenAIImagesClient",
    "openai_images_adapters",
]
