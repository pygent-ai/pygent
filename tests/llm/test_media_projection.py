from __future__ import annotations

import hashlib
from dataclasses import replace
from fractions import Fraction
from io import BytesIO

import pytest
from PIL import Image

from pygent import MediaBlock, MediaSource
from pygent.llm import (
    DefaultMediaProjector,
    MediaTransportCapabilities,
    ModelErrorKind,
    ModelFailureReason,
    ModelImageInputCapabilities,
    ModelMediaInputCapabilities,
    ModelProviderError,
    ModelVideoInputCapabilities,
)
from tests.support.model_specs import model_entry


def _png(width: int = 320, height: int = 160) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), (32, 96, 192)).save(buffer, format="PNG")
    return buffer.getvalue()


def _image_model(*, max_width: int = 64):
    entry = model_entry("vision", "openai", "vision", streaming=False)
    capabilities = replace(
        entry.spec.capabilities,
        modalities=replace(
            entry.spec.capabilities.modalities,
            input=("text", "image"),
        ),
        media_input=ModelMediaInputCapabilities(
            image=ModelImageInputCapabilities(
                mime_types=("image/jpeg",),
                max_width=max_width,
                max_height=max_width,
                animated=False,
            )
        ),
    )
    return replace(entry.spec, capabilities=capabilities)


def _mp4() -> bytes:
    av = pytest.importorskip("av")
    buffer = BytesIO()
    container = av.open(buffer, mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=10)
    stream.width = 64
    stream.height = 32
    stream.pix_fmt = "yuv420p"
    for index in range(10):
        frame = av.VideoFrame(64, 32, "rgb24")
        frame.planes[0].update(bytes((index * 10, 40, 80)) * (64 * 32))
        frame.pts = index
        frame.time_base = Fraction(1, 10)
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return buffer.getvalue()


def test_image_projection_is_request_local_and_has_distinct_integrity_facts() -> None:
    original = MediaBlock(
        media_type="image",
        mime_type="image/png",
        source=MediaSource.inline(_png()),
        width=320,
        height=160,
    )
    projector = DefaultMediaProjector()
    plan = projector.plan(
        original,
        model=_image_model(),
        endpoint=MediaTransportCapabilities(
            enabled=True,
            modalities=("image",),
            source_kinds=("inline",),
            image_mime_types=("image/jpeg",),
        ),
    )

    assert plan is not None
    projected = projector.project(original, call_id="image-1", plan=plan)

    assert original.mime_type == "image/png"
    assert (original.width, original.height) == (320, 160)
    assert projected.media.mime_type == "image/jpeg"
    assert (projected.media.width, projected.media.height) == (64, 32)
    assert projected.media.source.sha256 != original.source.sha256
    assert projected.trace.canonical_sha256 == original.source.sha256
    assert projected.trace.projected_sha256 == projected.media.source.sha256
    assert projected.trace.transformations == ("resize", "transcode")


def test_resource_projection_checks_declared_integrity_before_decoding() -> None:
    expected = _png()
    original = MediaBlock(
        media_type="image",
        mime_type="image/png",
        source=MediaSource.resource(
            "media://image/1",
            sha256=hashlib.sha256(expected).hexdigest(),
            size_bytes=len(expected),
        ),
        width=320,
        height=160,
    )
    projector = DefaultMediaProjector(media_resolver=lambda _source: b"wrong")
    plan = projector.plan(
        original,
        model=_image_model(),
        endpoint=MediaTransportCapabilities(
            enabled=True,
            modalities=("image",),
            source_kinds=("inline",),
            image_mime_types=("image/jpeg",),
        ),
    )

    assert plan is not None
    with pytest.raises(ModelProviderError) as raised:
        projector.project(original, call_id="image-1", plan=plan)
    assert raised.value.kind is ModelErrorKind.INVALID_REQUEST
    assert raised.value.reason_code is ModelFailureReason.MEDIA_INTEGRITY_MISMATCH


def test_projection_plan_requires_a_model_endpoint_mime_intersection() -> None:
    original = MediaBlock(
        media_type="image",
        mime_type="image/png",
        source=MediaSource.inline(_png()),
        width=320,
        height=160,
    )

    plan = DefaultMediaProjector().plan(
        original,
        model=_image_model(),
        endpoint=MediaTransportCapabilities(
            enabled=True,
            modalities=("image",),
            source_kinds=("inline",),
            image_mime_types=("image/webp",),
        ),
    )

    assert plan is None


def test_unsupported_image_detail_is_removed_only_from_the_request_projection() -> None:
    original = MediaBlock(
        media_type="image",
        mime_type="image/png",
        source=MediaSource.inline(_png()),
        detail="low",
        width=320,
        height=160,
    )
    entry = model_entry("vision", "openai", "vision", streaming=False)
    model = replace(
        entry.spec,
        capabilities=replace(
            entry.spec.capabilities,
            modalities=replace(
                entry.spec.capabilities.modalities,
                input=("text", "image"),
            ),
            media_input=ModelMediaInputCapabilities(
                image=ModelImageInputCapabilities(
                    mime_types=("image/png",),
                    resolution_modes=("high", "original"),
                )
            ),
        ),
    )
    projector = DefaultMediaProjector()
    plan = projector.plan(
        original,
        model=model,
        endpoint=MediaTransportCapabilities(
            enabled=True,
            modalities=("image",),
            source_kinds=("inline",),
            image_mime_types=("image/png",),
        ),
    )

    assert plan is not None
    projected = projector.project(original, call_id="image-1", plan=plan)

    assert original.detail == "low"
    assert projected.media.detail is None
    assert projected.media.source is original.source
    assert projected.trace.transformations == ("detail",)


def test_video_projection_applies_model_dimensions_and_frame_rate() -> None:
    original = MediaBlock(
        media_type="video",
        mime_type="video/mp4",
        source=MediaSource.inline(_mp4()),
        width=64,
        height=32,
        duration_seconds=1.0,
        fps=10.0,
        has_audio=False,
    )
    entry = model_entry("video", "google", "video", streaming=False)
    model = replace(
        entry.spec,
        capabilities=replace(
            entry.spec.capabilities,
            modalities=replace(
                entry.spec.capabilities.modalities,
                input=("text", "video"),
            ),
            media_input=ModelMediaInputCapabilities(
                video=ModelVideoInputCapabilities(
                    native=True,
                    mime_types=("video/mp4",),
                    max_width=32,
                    max_height=16,
                    max_fps=5.0,
                    audio=False,
                )
            ),
        ),
    )
    projector = DefaultMediaProjector()
    plan = projector.plan(
        original,
        model=model,
        endpoint=MediaTransportCapabilities(
            enabled=True,
            modalities=("video",),
            source_kinds=("inline",),
            video_mime_types=("video/mp4",),
        ),
    )

    assert plan is not None
    projected = projector.project(original, call_id="video-1", plan=plan)

    assert projected.media.mime_type == "video/mp4"
    assert (projected.media.width, projected.media.height) == (32, 16)
    assert projected.media.fps == 5.0
    assert projected.media.duration_seconds == 1.0
    assert projected.media.has_audio is False
    assert projected.trace.transformations == ("resize", "frame_rate", "transcode")
    assert projected.media.source.sha256 != original.source.sha256
