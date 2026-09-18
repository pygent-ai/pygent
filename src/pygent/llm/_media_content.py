"""Shared validation and byte resolution for provider media projections."""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Callable, Collection
from typing import NoReturn, cast

from pygent.tool import MediaSource, ToolResultMedia

from ._adapter_contracts import (
    MediaResolver,
    MediaTransportCapabilities,
)
from .configuration import ModelSpec
from .types import ModelErrorKind, ModelFailureReason, ModelProviderError

MediaResolverLike = MediaResolver | Callable[[MediaSource], bytes]


def validate_media_delivery(
    block: ToolResultMedia,
    *,
    model: ModelSpec,
    capabilities: MediaTransportCapabilities,
    allowed_mime_types: Collection[str] | None = None,
) -> None:
    """Validate one block against model, protocol, source, and size limits."""

    if block.media_type not in model.capabilities.modalities.input:
        _invalid(
            f"model does not support {block.media_type} input",
            ModelFailureReason.MODEL_INPUT_MODALITY_UNSUPPORTED,
        )
    if not capabilities.enabled or block.media_type not in capabilities.modalities:
        _invalid(
            f"endpoint does not support {block.media_type} in tool results",
            ModelFailureReason.MEDIA_TRANSPORT_UNSUPPORTED,
        )
    if block.source.kind not in capabilities.source_kinds:
        _invalid(
            f"endpoint does not support {block.source.kind} media sources",
            ModelFailureReason.MEDIA_SOURCE_UNSUPPORTED,
        )
    if allowed_mime_types is not None and block.mime_type not in allowed_mime_types:
        _invalid(
            f"endpoint does not support {block.mime_type} tool-result media",
            ModelFailureReason.MEDIA_TRANSPORT_UNSUPPORTED,
        )
    if (
        capabilities.max_media_bytes is not None
        and block.source.size_bytes is not None
        and block.source.size_bytes > capabilities.max_media_bytes
    ):
        _invalid(
            "media exceeds the configured endpoint limit",
            ModelFailureReason.MEDIA_TOO_LARGE,
        )


def media_base64(
    block: ToolResultMedia,
    *,
    capabilities: MediaTransportCapabilities,
    media_resolver: MediaResolverLike | None,
) -> str:
    """Return validated base64 bytes for an inline or resolvable resource."""

    source = block.source
    if source.kind == "url":
        _invalid(
            "media URL cannot be converted to inline content without fetching",
            ModelFailureReason.MEDIA_SOURCE_UNSUPPORTED,
        )
    if source.kind == "inline":
        encoded = cast(str, source.base64_data)
        data = base64.b64decode(encoded, validate=True)
    else:
        data = _resolve_resource(source, media_resolver)
        encoded = base64.b64encode(data).decode("ascii")
    validate_media_bytes(
        block,
        data,
        max_bytes=capabilities.max_media_bytes,
    )
    return encoded


def media_data_url(
    block: ToolResultMedia,
    *,
    capabilities: MediaTransportCapabilities,
    media_resolver: MediaResolverLike | None,
) -> str:
    """Return a remote URL unchanged or a validated inline data URL."""

    if block.source.kind == "url":
        return cast(str, block.source.url)
    encoded = media_base64(
        block,
        capabilities=capabilities,
        media_resolver=media_resolver,
    )
    return f"data:{block.mime_type};base64,{encoded}"


def _resolve_resource(
    source: MediaSource, media_resolver: MediaResolverLike | None
) -> bytes:
    if media_resolver is None:
        _invalid(
            "media resource cannot be resolved",
            ModelFailureReason.MEDIA_SOURCE_UNRESOLVABLE,
        )
    try:
        resolved = (
            media_resolver(source)
            if callable(media_resolver)
            else media_resolver.resolve(source)
        )
    except Exception:  # noqa: BLE001 - deployment resolver boundary
        _invalid(
            "media resource cannot be resolved",
            ModelFailureReason.MEDIA_SOURCE_UNRESOLVABLE,
        )
    if not isinstance(resolved, bytes) or not resolved:
        _invalid(
            "media resolver returned invalid content",
            ModelFailureReason.MEDIA_CONTENT_INVALID,
        )
    return resolved


def validate_media_bytes(
    block: ToolResultMedia,
    data: bytes,
    *,
    max_bytes: int | None = None,
) -> None:
    """Validate resolved canonical bytes without applying a transport encoding."""

    source = block.source
    if max_bytes is not None and len(data) > max_bytes:
        _invalid(
            "media exceeds the configured endpoint limit",
            ModelFailureReason.MEDIA_TOO_LARGE,
        )
    if source.size_bytes is not None and source.size_bytes != len(data):
        _invalid(
            "media size does not match its descriptor",
            ModelFailureReason.MEDIA_INTEGRITY_MISMATCH,
        )
    if source.sha256 is not None and hashlib.sha256(data).hexdigest() != source.sha256:
        _invalid(
            "media digest does not match its descriptor",
            ModelFailureReason.MEDIA_INTEGRITY_MISMATCH,
        )
    if not _media_signature_matches(data, block.mime_type):
        _invalid(
            "media content does not match its MIME type",
            ModelFailureReason.MEDIA_MIME_TYPE_INVALID,
        )


def _media_signature_matches(data: bytes, mime_type: str) -> bool:
    if mime_type == "image/png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    if mime_type == "image/jpeg":
        return data.startswith(b"\xff\xd8\xff")
    if mime_type == "image/gif":
        return data.startswith((b"GIF87a", b"GIF89a"))
    if mime_type == "image/webp":
        return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    if mime_type == "video/mp4":
        return len(data) >= 12 and data[4:8] == b"ftyp"
    return True


def _invalid(message: str, reason: ModelFailureReason) -> NoReturn:
    raise ModelProviderError(
        ModelErrorKind.INVALID_REQUEST,
        message,
        reason_code=reason,
    )


__all__ = [
    "MediaResolverLike",
    "media_base64",
    "media_data_url",
    "validate_media_bytes",
    "validate_media_delivery",
]
