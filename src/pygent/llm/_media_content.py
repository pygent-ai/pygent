"""Shared validation and byte resolution for provider media projections."""

from __future__ import annotations

import base64
import hashlib
import math
from collections.abc import Callable, Collection
from io import BytesIO
from typing import NoReturn, cast

from pygent.tool import MediaBlock, MediaSource

from ._adapter_contracts import (
    MediaResolver,
    MediaTransportCapabilities,
)
from .configuration import ModelSpec
from .types import ModelErrorKind, ModelFailureReason, ModelProviderError

MediaResolverLike = MediaResolver | Callable[[MediaSource], bytes]


def validate_media_delivery(
    block: MediaBlock,
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
            f"endpoint does not support {block.media_type} media",
            ModelFailureReason.MEDIA_TRANSPORT_UNSUPPORTED,
        )
    if block.source.kind not in capabilities.source_kinds:
        _invalid(
            f"endpoint does not support {block.source.kind} media sources",
            ModelFailureReason.MEDIA_SOURCE_UNSUPPORTED,
        )
    if allowed_mime_types is not None and block.mime_type not in allowed_mime_types:
        _invalid(
            f"endpoint does not support {block.mime_type} media",
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
    block: MediaBlock,
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
    block: MediaBlock,
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


def video_frame_data_urls(
    block: MediaBlock,
    *,
    capabilities: MediaTransportCapabilities,
    media_resolver: MediaResolverLike | None,
    max_frames: int = 32,
    max_edge: int = 1024,
) -> tuple[str, ...]:
    """Decode bounded, ordered JPEG frames for frame-based video APIs."""

    if block.media_type != "video":
        raise TypeError("frame extraction requires video media")
    if max_frames <= 0 or max_edge <= 0:
        raise ValueError("frame extraction limits must be positive")
    try:
        import av  # type: ignore[import-untyped]
    except (ImportError, OSError):
        _invalid(
            "frame-based video delivery requires the optional video backend",
            ModelFailureReason.MEDIA_TRANSPORT_UNSUPPORTED,
        )
    encoded = media_base64(
        block,
        capabilities=capabilities,
        media_resolver=media_resolver,
    )
    data = base64.b64decode(encoded, validate=True)
    try:
        container = av.open(BytesIO(data), mode="r")
        try:
            stream = next(iter(container.streams.video))
            duration = block.duration_seconds
            if duration is None and stream.duration is not None and stream.time_base:
                duration = float(stream.duration * stream.time_base)
            interval = (
                None
                if duration is None or duration <= 0 or max_frames == 1
                else duration / (max_frames - 1)
            )
            next_time = 0.0
            frames: list[str] = []
            decoded_index = 0
            fps = float(stream.average_rate or block.fps or 1.0)
            for frame in container.decode(stream):
                frame_time = float(
                    frame.time
                    if frame.time is not None
                    else decoded_index / max(fps, 1.0)
                )
                decoded_index += 1
                if interval is not None and frame_time + 1e-9 < next_time:
                    continue
                image = frame.to_image()
                try:
                    scale = min(1.0, max_edge / max(image.size))
                    if scale < 1.0:
                        resized = image.resize(
                            (
                                max(1, math.floor(image.width * scale)),
                                max(1, math.floor(image.height * scale)),
                            )
                        )
                        image.close()
                        image = resized
                    output = BytesIO()
                    image.save(output, format="JPEG", quality=85, optimize=True)
                    frames.append(
                        "data:image/jpeg;base64,"
                        + base64.b64encode(output.getvalue()).decode("ascii")
                    )
                finally:
                    image.close()
                if len(frames) >= max_frames:
                    break
                if interval is not None:
                    next_time = frame_time + interval
            if not frames:
                _invalid(
                    "video contains no decodable frames",
                    ModelFailureReason.MEDIA_CONTENT_INVALID,
                )
            return tuple(frames)
        finally:
            container.close()
    except ModelProviderError:
        raise
    except Exception as exc:
        raise ModelProviderError(
            ModelErrorKind.INVALID_REQUEST,
            "video frame extraction failed",
            reason_code=ModelFailureReason.MEDIA_CONTENT_INVALID,
        ) from exc


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
    block: MediaBlock,
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
    "video_frame_data_urls",
]
