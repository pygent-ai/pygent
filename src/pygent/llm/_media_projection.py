"""Provider-neutral request-local media planning and projection."""

from __future__ import annotations

import base64
import hashlib
import math
import os
import tempfile
from collections.abc import Callable
from contextlib import suppress
from dataclasses import replace
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, cast

from pygent.tool import MediaBlock, MediaSource

from ._adapter_contracts import (
    MediaProjectionPlan,
    MediaProjectionTrace,
    MediaResolver,
    MediaTransportCapabilities,
    ProjectedMedia,
)
from ._media_content import validate_media_bytes
from .configuration import ModelSpec
from .types import ModelErrorKind, ModelFailureReason, ModelProviderError

if TYPE_CHECKING:
    from PIL import Image

MediaResolverLike = MediaResolver | Callable[[MediaSource], bytes]

_IMAGE_FORMATS = {
    "image/png": "PNG",
    "image/jpeg": "JPEG",
    "image/webp": "WEBP",
}


class DefaultMediaProjector:
    """Built-in bounded image/video projector with no Provider knowledge."""

    version = "pygent.media_projection.v1"

    def __init__(self, *, media_resolver: MediaResolverLike | None = None) -> None:
        if media_resolver is not None and not (
            isinstance(media_resolver, MediaResolver) or callable(media_resolver)
        ):
            raise TypeError("media_resolver must provide resolve() or be callable")
        self._media_resolver = media_resolver

    def plan(
        self,
        block: MediaBlock,
        *,
        model: ModelSpec,
        endpoint: MediaTransportCapabilities,
    ) -> MediaProjectionPlan | None:
        if block.media_type not in model.capabilities.modalities.input:
            return None
        if not endpoint.enabled or block.media_type not in endpoint.modalities:
            return None
        image_details = (
            model.capabilities.media_input.image
            if block.media_type == "image"
            else None
        )
        video_details = (
            model.capabilities.media_input.video
            if block.media_type == "video"
            else None
        )
        if video_details is not None and video_details.native is False:
            return None

        details = image_details if image_details is not None else video_details
        model_mime_types = () if details is None else details.mime_types
        endpoint_mime_types = getattr(endpoint, f"{block.media_type}_mime_types")
        if model_mime_types and endpoint_mime_types:
            mime_types = tuple(
                value for value in model_mime_types if value in endpoint_mime_types
            )
            if not mime_types:
                return None
        else:
            mime_types = model_mime_types or endpoint_mime_types
        target_mime = block.mime_type if not mime_types else mime_types[0]
        if block.mime_type in mime_types or not mime_types:
            target_mime = block.mime_type
        max_bytes = _minimum_limit(
            None if details is None else details.max_bytes,
            endpoint.max_media_bytes,
        )
        transform_bytes = bool(
            mime_types and block.mime_type not in mime_types
        ) or _over(block.source.size_bytes, max_bytes)
        needs_projection = (
            transform_bytes or block.source.kind not in endpoint.source_kinds
        )
        if block.media_type == "image":
            max_width = None if image_details is None else image_details.max_width
            max_height = None if image_details is None else image_details.max_height
            max_pixels = None if image_details is None else image_details.max_pixels
            animated = None if image_details is None else image_details.animated
            resolution_modes = (
                () if image_details is None else image_details.resolution_modes
            )
            target_detail = block.detail
            if (
                resolution_modes
                and block.detail not in (None, "auto")
                and block.detail not in resolution_modes
            ):
                target_detail = None
                needs_projection = True
            transform_bytes = transform_bytes or _image_limits_exceeded(
                block, max_width=max_width, max_height=max_height, max_pixels=max_pixels
            )
            transform_bytes = transform_bytes or (
                (
                    max_width is not None
                    or max_height is not None
                    or max_pixels is not None
                )
                and (block.width is None or block.height is None)
            )
            transform_bytes = transform_bytes or animated is False
            needs_projection = needs_projection or transform_bytes
            max_duration = None
            max_fps = None
            audio = None
            if transform_bytes and target_mime not in _IMAGE_FORMATS:
                target_mime = next(
                    (value for value in mime_types if value in _IMAGE_FORMATS), ""
                )
                if not target_mime:
                    return None
        else:
            max_width = None if video_details is None else video_details.max_width
            max_height = None if video_details is None else video_details.max_height
            max_pixels = None
            max_duration = (
                None if video_details is None else video_details.max_duration_seconds
            )
            max_fps = None if video_details is None else video_details.max_fps
            animated = None
            audio = None if video_details is None else video_details.audio
            target_detail = block.detail
            if (
                max_duration is not None
                and block.duration_seconds is not None
                and block.duration_seconds > max_duration
            ):
                return None
            transform_bytes = transform_bytes or _video_limits_exceeded(
                block,
                max_width=max_width,
                max_height=max_height,
                max_fps=max_fps,
                audio=audio,
            )
            transform_bytes = transform_bytes or (
                (max_width is not None or max_height is not None)
                and (block.width is None or block.height is None)
            )
            transform_bytes = transform_bytes or (
                max_fps is not None and block.fps is None
            )
            transform_bytes = transform_bytes or (
                audio is not None and block.has_audio is None
            )
            transform_bytes = transform_bytes or (
                max_duration is not None and block.duration_seconds is None
            )
            needs_projection = needs_projection or transform_bytes
            if transform_bytes and target_mime != "video/mp4":
                return None

        if not needs_projection:
            return MediaProjectionPlan(
                target_source_kind=block.source.kind,
                target_mime_type=block.mime_type,
                target_detail=target_detail,
                passthrough=True,
                max_bytes=max_bytes,
                max_width=max_width,
                max_height=max_height,
                max_pixels=max_pixels,
                max_duration_seconds=max_duration,
                max_fps=max_fps,
                animated=animated,
                audio=audio,
            )
        if block.source.kind in endpoint.source_kinds and not transform_bytes:
            return MediaProjectionPlan(
                target_source_kind=block.source.kind,
                target_mime_type=target_mime,
                target_detail=target_detail,
                max_bytes=max_bytes,
                max_width=max_width,
                max_height=max_height,
                max_pixels=max_pixels,
                max_duration_seconds=max_duration,
                max_fps=max_fps,
                animated=animated,
                audio=audio,
            )
        if "inline" not in endpoint.source_kinds or not self._can_resolve(block.source):
            return None
        return MediaProjectionPlan(
            target_source_kind="inline",
            target_mime_type=target_mime,
            target_detail=target_detail,
            transform_bytes=transform_bytes,
            max_bytes=max_bytes,
            max_width=max_width,
            max_height=max_height,
            max_pixels=max_pixels,
            max_duration_seconds=max_duration,
            max_fps=max_fps,
            animated=animated,
            audio=audio,
        )

    def project(
        self,
        block: MediaBlock,
        *,
        call_id: str,
        plan: MediaProjectionPlan,
    ) -> ProjectedMedia:
        if plan.passthrough:
            return ProjectedMedia(
                media=block,
                trace=_trace(call_id, block, block, (), self.version),
            )
        if not plan.transform_bytes and plan.target_source_kind == block.source.kind:
            media = replace(block, detail=plan.target_detail)
            return ProjectedMedia(
                media=media,
                trace=_trace(call_id, block, media, ("detail",), self.version),
            )
        data = self._resolve(block.source)
        validate_media_bytes(block, data)
        if not plan.transform_bytes:
            media = replace(
                block,
                source=MediaSource.inline(data),
                detail=plan.target_detail,
            )
            transformations: tuple[str, ...] = (
                ("source_normalization", "detail")
                if block.detail != plan.target_detail
                else ("source_normalization",)
            )
            return ProjectedMedia(
                media=media,
                trace=_trace(
                    call_id,
                    block,
                    media,
                    transformations,
                    self.version,
                ),
            )
        if block.media_type == "image":
            media, transformations = _project_image(block, data, plan)
        else:
            media, transformations = _project_video(block, data, plan)
        return ProjectedMedia(
            media=media,
            trace=_trace(call_id, block, media, transformations, self.version),
        )

    def _can_resolve(self, source: MediaSource) -> bool:
        return source.kind == "inline" or self._media_resolver is not None

    def _resolve(self, source: MediaSource) -> bytes:
        if source.kind == "inline":
            return base64.b64decode(cast(str, source.base64_data), validate=True)
        resolver = self._media_resolver
        if resolver is None:
            _projection_error(
                "media source cannot be resolved for projection",
                ModelFailureReason.MEDIA_SOURCE_UNRESOLVABLE,
            )
        try:
            data = resolver(source) if callable(resolver) else resolver.resolve(source)
        except Exception:  # noqa: BLE001 - deployment resolver boundary
            _projection_error(
                "media source cannot be resolved for projection",
                ModelFailureReason.MEDIA_SOURCE_UNRESOLVABLE,
            )
        if not isinstance(data, bytes) or not data:
            _projection_error(
                "media resolver returned invalid content",
                ModelFailureReason.MEDIA_CONTENT_INVALID,
            )
        return data


def _project_image(
    block: MediaBlock, data: bytes, plan: MediaProjectionPlan
) -> tuple[MediaBlock, tuple[str, ...]]:
    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        with Image.open(BytesIO(data)) as opened:
            is_animated = bool(getattr(opened, "is_animated", False))
            opened.seek(0)
            image = ImageOps.exif_transpose(opened).copy()
    except ModelProviderError:
        raise
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ModelProviderError(
            ModelErrorKind.INVALID_REQUEST,
            "image decoding failed",
            reason_code=ModelFailureReason.MEDIA_CONTENT_INVALID,
        ) from exc
    transformations: list[str] = []
    if block.detail != plan.target_detail:
        transformations.append("detail")
    if is_animated and plan.animated is False:
        transformations.append("flatten_animation")
    try:
        width, height = image.size
        scale = _image_scale(
            width,
            height,
            max_width=plan.max_width,
            max_height=plan.max_height,
            max_pixels=plan.max_pixels,
        )
        if scale < 1.0:
            resized = image.resize(
                (max(1, int(width * scale)), max(1, int(height * scale))),
                Image.Resampling.LANCZOS,
            )
            image.close()
            image = resized
            transformations.append("resize")
        target_mime = plan.target_mime_type
        if target_mime != block.mime_type:
            transformations.append("transcode")
        encoded, image = _encode_image_with_budget(
            image,
            target_mime,
            plan.max_bytes,
            transformations,
        )
        source = MediaSource.inline(encoded)
        return (
            MediaBlock(
                media_type="image",
                mime_type=target_mime,
                source=source,
                detail=plan.target_detail,
                width=image.width,
                height=image.height,
            ),
            tuple(dict.fromkeys(transformations)),
        )
    finally:
        image.close()


def _encode_image_with_budget(
    image: Image.Image,
    mime_type: str,
    max_bytes: int | None,
    transformations: list[str],
) -> tuple[bytes, Image.Image]:
    from PIL import Image

    current = image
    quality = 90
    while True:
        buffer = BytesIO()
        prepared = current
        if mime_type == "image/jpeg" and current.mode not in ("RGB", "L"):
            prepared = current.convert("RGB")
        options: dict[str, object] = {"optimize": True}
        if mime_type in ("image/jpeg", "image/webp"):
            options["quality"] = quality
        prepared.save(buffer, format=_IMAGE_FORMATS[mime_type], **options)
        if prepared is not current:
            prepared.close()
        encoded = buffer.getvalue()
        if max_bytes is None or len(encoded) <= max_bytes:
            if current is not image:
                image.close()
            return encoded, current
        if quality > 45 and mime_type in ("image/jpeg", "image/webp"):
            quality -= 10
            if "reencode" not in transformations:
                transformations.append("reencode")
            continue
        if current.width <= 1 or current.height <= 1:
            _projection_error(
                "image cannot fit the target media byte limit",
                ModelFailureReason.MEDIA_TOO_LARGE,
            )
        resized = current.resize(
            (max(1, int(current.width * 0.85)), max(1, int(current.height * 0.85))),
            Image.Resampling.LANCZOS,
        )
        if current is not image:
            current.close()
        current = resized
        quality = 90
        if "resize" not in transformations:
            transformations.append("resize")


def _project_video(
    block: MediaBlock, data: bytes, plan: MediaProjectionPlan
) -> tuple[MediaBlock, tuple[str, ...]]:
    try:
        import av  # type: ignore[import-untyped]
    except (ImportError, OSError):
        _projection_error(
            "video projection requires the optional video backend",
            ModelFailureReason.MEDIA_TRANSPORT_UNSUPPORTED,
        )
    try:
        input_container = av.open(BytesIO(data), mode="r")
        video_stream = next(iter(input_container.streams.video))
        duration = _video_duration(input_container, video_stream, av)
        width = int(video_stream.codec_context.width)
        height = int(video_stream.codec_context.height)
        fps = float(video_stream.average_rate or 0)
        has_audio = bool(tuple(input_container.streams.audio))
        if (
            plan.max_duration_seconds is not None
            and duration > plan.max_duration_seconds
        ):
            _projection_error(
                "video exceeds the target duration limit",
                ModelFailureReason.MEDIA_TOO_LARGE,
            )
        target_width, target_height = _video_dimensions(
            width,
            height,
            max_width=plan.max_width,
            max_height=plan.max_height,
        )
        target_fps = fps if plan.max_fps is None else min(fps, plan.max_fps)
        keep_audio = has_audio and plan.audio is not False
        transformations: list[str] = []
        if (target_width, target_height) != (width, height):
            transformations.append("resize")
        if target_fps < fps:
            transformations.append("frame_rate")
        if plan.audio is False and has_audio:
            transformations.append("remove_audio")
        transformations.append("transcode")
        output_descriptor, output_name = tempfile.mkstemp(
            prefix="pygent-media-projection-", suffix=".mp4"
        )
        os.close(output_descriptor)
        output_path = Path(output_name)
        output_container = av.open(
            str(output_path),
            mode="w",
            format="mp4",
            options={"movflags": "+faststart"},
        )
        try:
            rate = Fraction(max(target_fps, 1.0)).limit_denominator(1001)
            output_video = output_container.add_stream("libx264", rate=rate)
            output_video.width = target_width
            output_video.height = target_height
            output_video.pix_fmt = "yuv420p"
            if plan.max_bytes is not None:
                output_video.bit_rate = max(
                    100_000,
                    int(plan.max_bytes * 8 * 0.85 / max(duration, 0.001)),
                )
            input_audio = next(iter(input_container.streams.audio), None)
            output_audio = (
                output_container.add_stream("aac", rate=48_000)
                if keep_audio and input_audio is not None
                else None
            )
            if output_audio is not None:
                output_audio.layout = "mono"
            frame_index = 0
            next_time = 0.0
            selected: list[Any] = [video_stream]
            if input_audio is not None and output_audio is not None:
                selected.append(input_audio)
            for packet in input_container.demux(selected):
                for frame in packet.decode():
                    if isinstance(frame, av.VideoFrame):
                        frame_time = float(frame.time or frame_index / max(fps, 1.0))
                        if frame_time + 1e-9 < next_time:
                            continue
                        next_time = frame_time + 1.0 / max(target_fps, 1.0)
                        normalized = frame.reformat(
                            width=target_width,
                            height=target_height,
                            format="yuv420p",
                        )
                        normalized.pts = frame_index
                        normalized.time_base = Fraction(
                            rate.denominator, rate.numerator
                        )
                        frame_index += 1
                        for encoded in output_video.encode(normalized):
                            output_container.mux(encoded)
                    elif output_audio is not None and isinstance(frame, av.AudioFrame):
                        for encoded in output_audio.encode(frame):
                            output_container.mux(encoded)
            for encoded in output_video.encode():
                output_container.mux(encoded)
            if output_audio is not None:
                for encoded in output_audio.encode():
                    output_container.mux(encoded)
            output_container.close()
            output_data = output_path.read_bytes()
        finally:
            input_container.close()
            with suppress(Exception):
                output_container.close()
            output_path.unlink(missing_ok=True)
    except ModelProviderError:
        raise
    except Exception as exc:
        raise ModelProviderError(
            ModelErrorKind.INVALID_REQUEST,
            "video projection failed",
            reason_code=ModelFailureReason.MEDIA_CONTENT_INVALID,
        ) from exc
    if plan.max_bytes is not None and len(output_data) > plan.max_bytes:
        _projection_error(
            "video cannot fit the target media byte limit",
            ModelFailureReason.MEDIA_TOO_LARGE,
        )
    source = MediaSource.inline(output_data)
    return (
        MediaBlock(
            media_type="video",
            mime_type="video/mp4",
            source=source,
            width=target_width,
            height=target_height,
            duration_seconds=duration,
            fps=target_fps,
            has_audio=keep_audio,
        ),
        tuple(transformations),
    )


def _video_duration(container: Any, stream: Any, av: Any) -> float:
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return float(container.duration / av.time_base)
    _projection_error(
        "video duration is unavailable",
        ModelFailureReason.MEDIA_CONTENT_INVALID,
    )


def _video_dimensions(
    width: int,
    height: int,
    *,
    max_width: int | None,
    max_height: int | None,
) -> tuple[int, int]:
    scale = min(
        1.0,
        math.inf if max_width is None else max_width / width,
        math.inf if max_height is None else max_height / height,
    )
    return (
        max(2, int(width * scale) // 2 * 2),
        max(2, int(height * scale) // 2 * 2),
    )


def _trace(
    call_id: str,
    canonical: MediaBlock,
    projected: MediaBlock,
    transformations: tuple[str, ...],
    version: str,
) -> MediaProjectionTrace:
    return MediaProjectionTrace(
        call_id=call_id,
        media_type=projected.media_type,
        canonical_reference=_source_reference(canonical.source),
        canonical_sha256=canonical.source.sha256,
        projected_reference=_source_reference(projected.source),
        projected_sha256=projected.source.sha256,
        projected_size_bytes=projected.source.size_bytes,
        mime_type=projected.mime_type,
        width=projected.width,
        height=projected.height,
        duration_seconds=projected.duration_seconds,
        fps=projected.fps,
        has_audio=projected.has_audio,
        transformations=transformations,
        projector_version=version,
    )


def _source_reference(source: MediaSource) -> str:
    if source.kind == "resource":
        return cast(str, source.uri)
    if source.kind == "url":
        return cast(str, source.url)
    digest = source.sha256
    if digest is None:  # pragma: no cover - inline sources always derive a digest
        decoded = base64.b64decode(cast(str, source.base64_data), validate=True)
        digest = hashlib.sha256(decoded).hexdigest()
    return f"inline:sha256:{digest}"


def _image_limits_exceeded(
    block: MediaBlock,
    *,
    max_width: int | None,
    max_height: int | None,
    max_pixels: int | None,
) -> bool:
    if block.width is None or block.height is None:
        return False
    return (
        _over(block.width, max_width)
        or _over(block.height, max_height)
        or _over(block.width * block.height, max_pixels)
    )


def _video_limits_exceeded(
    block: MediaBlock,
    *,
    max_width: int | None,
    max_height: int | None,
    max_fps: float | None,
    audio: bool | None,
) -> bool:
    return (
        _over(block.width, max_width)
        or _over(block.height, max_height)
        or _over(block.fps, max_fps)
        or (audio is False and block.has_audio is True)
    )


def _image_scale(
    width: int,
    height: int,
    *,
    max_width: int | None,
    max_height: int | None,
    max_pixels: int | None,
) -> float:
    scale = min(
        1.0,
        math.inf if max_width is None else max_width / width,
        math.inf if max_height is None else max_height / height,
    )
    if max_pixels is not None and width * height * scale * scale > max_pixels:
        scale = min(scale, math.sqrt(max_pixels / (width * height)))
    return scale


def _minimum_limit(first: int | None, second: int | None) -> int | None:
    values = tuple(value for value in (first, second) if value is not None)
    return min(values) if values else None


def _over(value: float | None, limit: float | None) -> bool:
    return value is not None and limit is not None and value > limit


def _projection_error(message: str, reason: ModelFailureReason) -> NoReturn:
    raise ModelProviderError(
        ModelErrorKind.INVALID_REQUEST,
        message,
        reason_code=reason,
    )


__all__ = ["DefaultMediaProjector"]
