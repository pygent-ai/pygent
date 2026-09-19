"""Workspace file adapters expressed through the Pygent 0.2 Tool contract."""

from __future__ import annotations

import asyncio
import base64
import fnmatch
import json
import locale
import os
import re
import shutil
import subprocess
import tempfile
import threading
import warnings
from contextlib import suppress
from dataclasses import dataclass
from fractions import Fraction
from io import BytesIO, TextIOWrapper
from itertools import islice
from math import ceil, sqrt
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, Never, TextIO

import pypdfium2 as pdfium  # type: ignore[import-untyped]
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import Field
from pypdf import PdfReader

from pygent.core import freeze_json_object
from pygent.core._tool_values import MediaSource, ToolResultMedia, ToolResultText
from pygent.tool.executors import ToolExecutionError
from pygent.tool.functional import tool
from pygent.tool.types import (
    IdempotencyPolicy,
    ToolOutput,
    ToolSideEffect,
)

from ._file_services import (
    FileDiagnosticsService,
    FileIOService,
    FileSearchService,
    NotebookService,
)
from ._paths import (
    ToolPathContext,
    is_absolute_tool_path,
    resolve_dir_path,
    resolve_file_path,
    resolve_tool_path,
)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico"}
_IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
_VIDEO_MIME_TYPES = {".mp4": "video/mp4"}
_DEFAULT_MAX_MEDIA_BYTES = 20 * 1024 * 1024
_DEFAULT_MAX_IMAGE_OUTPUT_BYTES = 3_500_000
_DEFAULT_MAX_IMAGE_EDGE = 2048
_DEFAULT_MAX_IMAGE_PIXELS = 40_000_000
_DEFAULT_MAX_VIDEO_OUTPUT_BYTES = 12_000_000
_DEFAULT_MAX_VIDEO_DURATION_SECONDS = 120.0
_DEFAULT_MAX_VIDEO_EDGE = 1280
_DEFAULT_MAX_VIDEO_FPS = 15.0
_VIDEO_AUDIO_BIT_RATE = 64_000
_VIDEO_MAX_BIT_RATE = 1_000_000
_VIDEO_BACKEND_HINT = (
    "Install pygent-ai[video] or provide FFmpeg with both ffmpeg and ffprobe on "
    "PATH to enable video metadata validation and automatic normalization."
)
_PDF_RENDER_SCALE = 2.0
_MAX_PDF_RENDER_PIXELS = 20_000_000
_IMAGE_FORMAT_MIME_TYPES = {
    "GIF": "image/gif",
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}
_EXIF_ORIENTATION_TAG = 274
_TEXT_BYTES = frozenset({*range(0x20, 0x100), 7, 8, 9, 10, 12, 13, 27})
_TEXT_SNIFF_BYTES = 64 * 1024
_UTF16_BOMS = (b"\xff\xfe", b"\xfe\xff")
_TRUNCATED_TAIL_REASONS = frozenset(
    {
        "unexpected end of data",
        "incomplete multibyte sequence",
        "truncated data",
    }
)
_SEARCH_MAX_BYTES = 50 * 1024
_GREP_MAX_LINE_LENGTH = 500
_WRITE_TOOL_DESCRIPTION = (
    "Create a new UTF-8 file or completely replace an existing file.\n\n"
    "Usage:\n"
    "- Use this tool only when creating a new file or intentionally replacing the "
    "whole file.\n"
    "- Prefer edit for focused changes to an existing file. For complex or long "
    "changes to an existing file, you MUST use multiple smaller atomic edit calls "
    "instead of rewriting the whole file.\n"
    "- Provide the complete literal file content. Do not use omission placeholders "
    'such as "...", "rest of file", or "unchanged code".'
)
_EDIT_TOOL_DESCRIPTION = (
    "Replace exact literal text in an existing UTF-8 file.\n\n"
    "Usage:\n"
    "- Read the current file before editing and preserve exact whitespace, "
    "indentation, and newlines.\n"
    "- Keep every edit focused. For complex or long changes, you MUST split the "
    "work into multiple smaller atomic edit calls.\n"
    "- Include only enough unchanged surrounding context to target the intended "
    "occurrence; do not include long runs of unchanged text.\n"
    "- Use write instead when intentionally replacing the whole file."
)
_DEFAULT_SEARCH_IGNORES = frozenset(
    {
        ".git",
        ".hg",
        ".lora",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".venv",
        "__pycache__",
        "node_modules",
        "venv",
    }
)
_READ_TOOL_DESCRIPTION = (
    "Read text, PDF, image, or video content from a workspace file. "
    "PDF files return extracted text by default; when pages is provided, the "
    "selected PDF pages are rendered as images for direct model inspection. "
    "Images and videos are returned as bounded inline multimodal content. "
    "Images are safely resized and re-encoded when needed. Videos are inspected "
    "and normalized when PyAV or FFmpeg is available; otherwise eligible files are "
    "returned unchanged."
)


def _fail(
    message: str,
    code: str,
    *,
    committed: bool | None = False,
    retryable: bool = False,
) -> Never:
    raise ToolExecutionError(
        message,
        kind="filesystem_error",
        code=code,
        retryable=retryable,
        side_effect_committed=committed,
    )


def _text_output(value: str) -> ToolOutput:
    return ToolOutput(output=value, content=(ToolResultText(value),))


@dataclass(frozen=True, slots=True)
class _PreparedImage:
    data: bytes
    mime_type: str
    original_dimensions: tuple[int, int]
    delivered_dimensions: tuple[int, int]
    transformations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _VideoMetadata:
    duration_seconds: float
    dimensions: tuple[int, int]
    fps: float
    video_codec: str
    pixel_format: str | None
    audio_codec: str | None
    audio_channels: int | None


@dataclass(frozen=True, slots=True)
class _PreparedVideo:
    data: bytes
    original_metadata: _VideoMetadata | None
    delivered_metadata: _VideoMetadata | None
    transformations: tuple[str, ...]
    processing: Literal["normalized", "passthrough", "passthrough_unverified"]


def _video_metadata_value(metadata: _VideoMetadata | None) -> dict[str, object] | None:
    if metadata is None:
        return None
    return {
        "duration_seconds": round(metadata.duration_seconds, 3),
        "dimensions": metadata.dimensions,
        "fps": round(metadata.fps, 3),
        "video_codec": metadata.video_codec,
        "pixel_format": metadata.pixel_format,
        "audio_codec": metadata.audio_codec,
        "audio_channels": metadata.audio_channels,
    }


def _encode_image(image: Image.Image, mime_type: str, quality: int | None) -> bytes:
    buffer = BytesIO()
    encode_image = image
    converted: Image.Image | None = None
    try:
        if mime_type == "image/jpeg":
            if image.mode not in ("L", "RGB"):
                converted = image.convert("RGB")
                encode_image = converted
            encode_image.save(
                buffer,
                format="JPEG",
                quality=quality,
                optimize=True,
            )
        elif mime_type == "image/webp":
            encode_image.save(
                buffer,
                format="WEBP",
                quality=quality,
                method=6,
            )
        else:
            encode_image.save(
                buffer,
                format="PNG",
                optimize=True,
                compress_level=9,
            )
        return buffer.getvalue()
    finally:
        if converted is not None:
            converted.close()


def _encode_image_under_limit(
    image: Image.Image,
    mime_type: str,
    max_output_bytes: int,
) -> tuple[bytes, str, tuple[int, int]]:
    output_mime_type = "image/png" if mime_type == "image/gif" else mime_type
    qualities: tuple[int | None, ...]
    if output_mime_type in ("image/jpeg", "image/webp"):
        qualities = (85, 75, 65, 55)
    else:
        qualities = (None,)

    working = image.copy()
    try:
        for _ in range(10):
            smallest: bytes | None = None
            for quality in qualities:
                candidate = _encode_image(working, output_mime_type, quality)
                if len(candidate) <= max_output_bytes:
                    return candidate, output_mime_type, working.size
                if smallest is None or len(candidate) < len(smallest):
                    smallest = candidate
            if smallest is None:  # pragma: no cover - every format has an encoder
                raise AssertionError("image encoder produced no candidate")
            width, height = working.size
            if width == 1 and height == 1:
                break
            scale = min(0.9, max(0.5, sqrt(max_output_bytes / len(smallest)) * 0.95))
            next_size = (
                max(1, int(width * scale)),
                max(1, int(height * scale)),
            )
            if next_size == working.size:
                break
            resized = working.resize(next_size, Image.Resampling.LANCZOS)
            working.close()
            working = resized
    finally:
        working.close()
    _fail(
        f"image cannot be encoded within the {max_output_bytes}-byte output limit",
        "image_output_too_large",
    )


def _prepare_image(
    data: bytes,
    mime_type: str,
    *,
    max_output_bytes: int,
    max_edge: int,
    max_pixels: int,
) -> _PreparedImage:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as source:
                detected_mime_type = _IMAGE_FORMAT_MIME_TYPES.get(source.format or "")
                if detected_mime_type != mime_type:
                    _fail(
                        f"image content does not match {mime_type}",
                        "media_mime_mismatch",
                    )
                original_dimensions = source.size
                width, height = original_dimensions
                if width <= 0 or height <= 0:
                    _fail("image dimensions must be positive", "invalid_image_dimensions")
                if width * height > max_pixels:
                    _fail(
                        f"image exceeds the {max_pixels}-pixel decode limit",
                        "image_too_many_pixels",
                    )
                if getattr(source, "is_animated", False) and getattr(
                    source, "n_frames", 1
                ) > 1:
                    _fail(
                        "animated images are not supported",
                        "animated_image_unsupported",
                    )
                orientation = source.getexif().get(_EXIF_ORIENTATION_TAG, 1)
                source.load()
                needs_orientation = orientation not in (None, 1)
                needs_resize = max(width, height) > max_edge
                needs_reencode = len(data) > max_output_bytes
                if not needs_orientation and not needs_resize and not needs_reencode:
                    return _PreparedImage(
                        data=data,
                        mime_type=mime_type,
                        original_dimensions=original_dimensions,
                        delivered_dimensions=original_dimensions,
                        transformations=(),
                    )

                oriented = ImageOps.exif_transpose(source)
                try:
                    working = oriented.copy()
                finally:
                    if oriented is not source:
                        oriented.close()
                try:
                    transformations: list[str] = []
                    if needs_orientation:
                        transformations.append("exif_orientation")
                    if max(working.size) > max_edge:
                        scale = max_edge / max(working.size)
                        target_size = (
                            max(1, int(working.width * scale)),
                            max(1, int(working.height * scale)),
                        )
                        resized = working.resize(target_size, Image.Resampling.LANCZOS)
                        working.close()
                        working = resized
                        transformations.append("resize")
                    prepared_data, prepared_mime_type, delivered_dimensions = (
                        _encode_image_under_limit(
                            working,
                            mime_type,
                            max_output_bytes,
                        )
                    )
                    if delivered_dimensions != working.size and "resize" not in transformations:
                        transformations.append("resize")
                    transformations.append("reencode")
                    return _PreparedImage(
                        data=prepared_data,
                        mime_type=prepared_mime_type,
                        original_dimensions=original_dimensions,
                        delivered_dimensions=delivered_dimensions,
                        transformations=tuple(transformations),
                    )
                finally:
                    working.close()
    except ToolExecutionError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ToolExecutionError(
            "image exceeds the safe decode limit",
            kind="filesystem_error",
            code="image_too_many_pixels",
            side_effect_committed=False,
        ) from exc
    except (OSError, UnidentifiedImageError) as exc:
        raise ToolExecutionError(
            "image decoding failed",
            kind="filesystem_error",
            code="image_decode_failed",
            side_effect_committed=False,
        ) from exc


@dataclass(frozen=True, slots=True)
class _FfmpegVideoBackend:
    ffmpeg: str
    ffprobe: str


@dataclass(frozen=True, slots=True)
class _PyAvVideoBackend:
    module: Any


def _load_video_backend() -> _PyAvVideoBackend | _FfmpegVideoBackend | None:
    try:
        import av  # type: ignore[import-untyped]
    except (ImportError, OSError):
        pass
    else:
        return _PyAvVideoBackend(module=av)
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        return None
    return _FfmpegVideoBackend(ffmpeg=ffmpeg, ffprobe=ffprobe)


def _run_video_process(
    command: list[str], *, timeout: float, failure_code: str
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ToolExecutionError(
            "video processing failed",
            kind="filesystem_error",
            code=failure_code,
            side_effect_committed=False,
        ) from exc
    if result.returncode != 0:
        raise ToolExecutionError(
            "video processing failed",
            kind="filesystem_error",
            code=failure_code,
            side_effect_committed=False,
        )
    return result


def _parse_video_rate(value: object) -> float:
    if not isinstance(value, str) or not value or value == "0/0":
        return 0.0
    numerator_text, separator, denominator_text = value.partition("/")
    try:
        numerator = float(numerator_text)
        denominator = float(denominator_text) if separator else 1.0
    except ValueError:
        return 0.0
    return numerator / denominator if denominator else 0.0


def _inspect_video_file(backend: _FfmpegVideoBackend, path: Path) -> _VideoMetadata:
    result = _run_video_process(
        [
            backend.ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,codec_name,width,height,pix_fmt,avg_frame_rate,channels",
            "-of",
            "json",
            str(path),
        ],
        timeout=30,
        failure_code="video_decode_failed",
    )
    try:
        probe = json.loads(result.stdout)
        streams = probe["streams"]
        video_streams = [item for item in streams if item.get("codec_type") == "video"]
        audio_streams = [item for item in streams if item.get("codec_type") == "audio"]
        if len(video_streams) != 1 or len(audio_streams) > 1:
            _fail(
                "video must contain exactly one video stream and at most one audio stream",
                "invalid_video_streams",
            )
        video = video_streams[0]
        duration_seconds = float(probe["format"]["duration"])
        width, height = int(video["width"]), int(video["height"])
        fps = _parse_video_rate(video.get("avg_frame_rate"))
        if duration_seconds <= 0:
            _fail("video duration is unavailable", "invalid_video_duration")
        if width <= 0 or height <= 0:
            _fail("video dimensions must be positive", "invalid_video_dimensions")
        if fps <= 0:
            _fail("video frame rate is unavailable", "invalid_video_frame_rate")
        audio = audio_streams[0] if audio_streams else None
        return _VideoMetadata(
            duration_seconds=duration_seconds,
            dimensions=(width, height),
            fps=fps,
            video_codec=str(video["codec_name"]),
            pixel_format=(str(video["pix_fmt"]) if video.get("pix_fmt") else None),
            audio_codec=(str(audio["codec_name"]) if audio is not None else None),
            audio_channels=(int(audio["channels"]) if audio is not None else None),
        )
    except ToolExecutionError:
        raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ToolExecutionError(
            "video metadata is invalid",
            kind="filesystem_error",
            code="video_decode_failed",
            side_effect_committed=False,
        ) from exc


def _inspect_video_with_pyav(backend: _PyAvVideoBackend, data: bytes) -> _VideoMetadata:
    av = backend.module
    try:
        container = av.open(BytesIO(data), mode="r", format="mp4")
        try:
            video_streams = tuple(container.streams.video)
            audio_streams = tuple(container.streams.audio)
            if len(video_streams) != 1 or len(audio_streams) > 1:
                _fail(
                    "video must contain exactly one video stream and at most one audio stream",
                    "invalid_video_streams",
                )
            video = video_streams[0]
            duration_seconds: float | None = None
            if video.duration is not None and video.time_base is not None:
                duration_seconds = float(video.duration * video.time_base)
            elif container.duration is not None:
                duration_seconds = float(container.duration / av.time_base)
            fps = float(video.average_rate) if video.average_rate is not None else 0.0
            width = int(video.codec_context.width)
            height = int(video.codec_context.height)
            if duration_seconds is None or duration_seconds <= 0:
                _fail("video duration is unavailable", "invalid_video_duration")
            if width <= 0 or height <= 0:
                _fail("video dimensions must be positive", "invalid_video_dimensions")
            if fps <= 0:
                _fail("video frame rate is unavailable", "invalid_video_frame_rate")
            audio = audio_streams[0].codec_context if audio_streams else None
            layout = audio.layout if audio is not None else None
            pixel_format = video.codec_context.format
            return _VideoMetadata(
                duration_seconds=duration_seconds,
                dimensions=(width, height),
                fps=fps,
                video_codec=video.codec_context.name,
                pixel_format=(pixel_format.name if pixel_format is not None else None),
                audio_codec=(audio.name if audio is not None else None),
                audio_channels=(layout.nb_channels if layout is not None else None),
            )
        finally:
            container.close()
    except ToolExecutionError:
        raise
    except Exception as exc:
        raise ToolExecutionError(
            "video decoding failed",
            kind="filesystem_error",
            code="video_decode_failed",
            side_effect_committed=False,
        ) from exc


def _inspect_video(
    backend: _PyAvVideoBackend | _FfmpegVideoBackend, data: bytes
) -> _VideoMetadata:
    if isinstance(backend, _PyAvVideoBackend):
        return _inspect_video_with_pyav(backend, data)
    with tempfile.TemporaryDirectory(prefix="pygent-video-probe-") as directory:
        input_path = Path(directory) / "input.mp4"
        input_path.write_bytes(data)
        return _inspect_video_file(backend, input_path)


def _transcode_video_with_pyav(
    backend: _PyAvVideoBackend,
    data: bytes,
    metadata: _VideoMetadata,
    *,
    max_output_bytes: int,
    max_edge: int,
    max_fps: float,
    crf: int,
) -> bytes:
    av = backend.module
    output_descriptor, output_name = tempfile.mkstemp(
        prefix="pygent-video-pyav-", suffix=".mp4"
    )
    os.close(output_descriptor)
    output_path = Path(output_name)
    input_container = av.open(BytesIO(data), mode="r", format="mp4")
    output_container = av.open(
        str(output_path),
        mode="w",
        format="mp4",
        options={"movflags": "+faststart"},
    )
    output_closed = False
    try:
        input_video = next(iter(input_container.streams.video))
        input_audio_streams = tuple(input_container.streams.audio)
        input_audio = input_audio_streams[0] if input_audio_streams else None
        scale = min(1.0, max_edge / max(metadata.dimensions))
        target_dimensions = (
            max(2, int(metadata.dimensions[0] * scale) // 2 * 2),
            max(2, int(metadata.dimensions[1] * scale) // 2 * 2),
        )
        target_fps = min(metadata.fps, max_fps)
        target_rate = Fraction(target_fps).limit_denominator(1001)
        duration = max(metadata.duration_seconds, 0.001)
        total_bit_rate = int(max_output_bytes * 8 * 0.88 / duration)
        audio_bit_rate = _VIDEO_AUDIO_BIT_RATE if input_audio is not None else 0
        video_bit_rate = max(
            100_000,
            min(_VIDEO_MAX_BIT_RATE, total_bit_rate - audio_bit_rate),
        )
        output_video = output_container.add_stream("libx264", rate=target_rate)
        output_video.width, output_video.height = target_dimensions
        output_video.pix_fmt = "yuv420p"
        output_video.bit_rate = video_bit_rate
        output_video.options = {
            "preset": "veryfast",
            "crf": str(crf),
            "maxrate": str(video_bit_rate),
            "bufsize": str(video_bit_rate * 2),
        }
        output_audio = None
        audio_resampler = None
        if input_audio is not None:
            output_audio = output_container.add_stream("aac", rate=48_000)
            output_audio.bit_rate = _VIDEO_AUDIO_BIT_RATE
            output_audio.layout = "mono"
            audio_resampler = av.AudioResampler(
                format="fltp",
                layout="mono",
                rate=48_000,
            )
        selected_streams = [input_video]
        if input_audio is not None:
            selected_streams.append(input_audio)
        emitted_video_frames = 0
        next_video_time = 0.0
        for packet in input_container.demux(selected_streams):
            for frame in packet.decode():
                if isinstance(frame, av.VideoFrame):
                    frame_time = (
                        float(frame.time)
                        if frame.time is not None
                        else emitted_video_frames / metadata.fps
                    )
                    if frame_time + 1e-9 < next_video_time:
                        continue
                    next_video_time = frame_time + 1.0 / target_fps
                    normalized = frame.reformat(
                        width=target_dimensions[0],
                        height=target_dimensions[1],
                        format="yuv420p",
                    )
                    normalized.pts = emitted_video_frames
                    normalized.time_base = Fraction(
                        target_rate.denominator,
                        target_rate.numerator,
                    )
                    emitted_video_frames += 1
                    for output_packet in output_video.encode(normalized):
                        output_container.mux(output_packet)
                elif output_audio is not None and audio_resampler is not None:
                    for normalized_audio in audio_resampler.resample(frame):
                        for output_packet in output_audio.encode(normalized_audio):
                            output_container.mux(output_packet)
        if output_audio is not None and audio_resampler is not None:
            for normalized_audio in audio_resampler.resample(None):
                for output_packet in output_audio.encode(normalized_audio):
                    output_container.mux(output_packet)
        for output_packet in output_video.encode():
            output_container.mux(output_packet)
        if output_audio is not None:
            for output_packet in output_audio.encode():
                output_container.mux(output_packet)
        output_container.close()
        output_closed = True
        return output_path.read_bytes()
    except ToolExecutionError:
        raise
    except Exception as exc:
        raise ToolExecutionError(
            "video processing failed",
            kind="filesystem_error",
            code="video_processing_failed",
            side_effect_committed=False,
        ) from exc
    finally:
        input_container.close()
        if not output_closed:
            with suppress(Exception):
                output_container.close()
        output_path.unlink(missing_ok=True)


def _transcode_video_once(
    backend: _PyAvVideoBackend | _FfmpegVideoBackend,
    data: bytes,
    metadata: _VideoMetadata,
    *,
    max_output_bytes: int,
    max_edge: int,
    max_fps: float,
    crf: int,
) -> bytes:
    if isinstance(backend, _PyAvVideoBackend):
        return _transcode_video_with_pyav(
            backend,
            data,
            metadata,
            max_output_bytes=max_output_bytes,
            max_edge=max_edge,
            max_fps=max_fps,
            crf=crf,
        )
    duration = max(metadata.duration_seconds, 0.001)
    total_bit_rate = int(max_output_bytes * 8 * 0.88 / duration)
    audio_bit_rate = _VIDEO_AUDIO_BIT_RATE if metadata.audio_codec is not None else 0
    video_bit_rate = max(
        100_000,
        min(_VIDEO_MAX_BIT_RATE, total_bit_rate - audio_bit_rate),
    )
    filter_graph = (
        f"scale={max_edge}:{max_edge}:force_original_aspect_ratio=decrease:"
        f"force_divisible_by=2,fps={max_fps:g}"
    )
    with tempfile.TemporaryDirectory(prefix="pygent-video-transcode-") as directory:
        input_path = Path(directory) / "input.mp4"
        output_path = Path(directory) / "output.mp4"
        input_path.write_bytes(data)
        command = [
            backend.ffmpeg,
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-i",
            str(input_path),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-vf",
            filter_graph,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-crf",
            str(crf),
            "-maxrate",
            str(video_bit_rate),
            "-bufsize",
            str(video_bit_rate * 2),
            "-c:a",
            "aac",
            "-ac",
            "1",
            "-b:a",
            str(_VIDEO_AUDIO_BIT_RATE),
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        _run_video_process(
            command,
            timeout=max(30, min(300, duration * 3)),
            failure_code="video_processing_failed",
        )
        try:
            return output_path.read_bytes()
        except OSError as exc:
            raise ToolExecutionError(
                "video processing produced no readable output",
                kind="filesystem_error",
                code="video_processing_failed",
                side_effect_committed=False,
            ) from exc


def _prepare_video(
    data: bytes,
    *,
    max_output_bytes: int,
    max_duration_seconds: float,
    max_edge: int,
    max_fps: float,
) -> _PreparedVideo:
    backend = _load_video_backend()
    if backend is None:
        if len(data) > max_output_bytes:
            _fail(
                "video exceeds the inline output limit and no video processing "
                f"backend is available. {_VIDEO_BACKEND_HINT}",
                "video_processing_unavailable",
            )
        return _PreparedVideo(
            data=data,
            original_metadata=None,
            delivered_metadata=None,
            transformations=(),
            processing="passthrough_unverified",
        )

    metadata = _inspect_video(backend, data)
    if metadata.duration_seconds > max_duration_seconds:
        _fail(
            f"video exceeds the {max_duration_seconds:g}-second duration limit",
            "video_too_long",
        )
    transformations: list[str] = []
    if max(metadata.dimensions) > max_edge:
        transformations.append("resize")
    if metadata.fps > max_fps:
        transformations.append("frame_rate")
    if metadata.video_codec != "h264" or metadata.pixel_format != "yuv420p":
        transformations.append("video_codec")
    if metadata.audio_codec not in (None, "aac") or (
        metadata.audio_channels is not None and metadata.audio_channels > 1
    ):
        transformations.append("audio_codec")
    if len(data) > max_output_bytes:
        transformations.append("byte_limit")
    if not transformations:
        return _PreparedVideo(
            data=data,
            original_metadata=metadata,
            delivered_metadata=metadata,
            transformations=(),
            processing="passthrough",
        )

    for crf in (28, 32, 36):
        transcoded = _transcode_video_once(
            backend,
            data,
            metadata,
            max_output_bytes=max_output_bytes,
            max_edge=max_edge,
            max_fps=max_fps,
            crf=crf,
        )
        if len(transcoded) <= max_output_bytes:
            normalized_metadata = _inspect_video(backend, transcoded)
            return _PreparedVideo(
                data=transcoded,
                original_metadata=metadata,
                delivered_metadata=normalized_metadata,
                transformations=(*transformations, "reencode"),
                processing="normalized",
            )
    _fail(
        f"video cannot be encoded within the {max_output_bytes}-byte output limit",
        "video_output_too_large",
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
    return False


def _decodes_as_text(data: bytes, encoding: str) -> bool:
    """Report whether a sample decodes strictly, tolerating a split last character."""

    try:
        data.decode(encoding, errors="strict")
    except UnicodeDecodeError as exc:
        return exc.reason in _TRUNCATED_TAIL_REASONS and exc.end >= len(data)
    except LookupError:
        return False
    return True


def _text_code_page(sample: bytes) -> str:
    """Choose the code page for a sample that is already known to be text.

    Windows editors commonly save Chinese text as cp936/GBK rather than UTF-8.
    Single-byte codecs such as cp1252 accept every byte, so the multibyte
    candidates must come before the locale and single-byte fallbacks; otherwise
    such a file silently decodes into mojibake instead of failing cleanly.
    """

    for encoding in (
        "utf-8-sig",
        "utf-8",
        "gb18030",
        "cp936",
        locale.getpreferredencoding(False),
        "cp1252",
        "latin-1",
    ):
        if encoding and _decodes_as_text(sample, encoding):
            return encoding
    return "utf-8"


def _detect_text_encoding(sample: bytes) -> str | None:
    """Select the text code page for a file sample, or None when it is binary."""

    if sample.startswith(_UTF16_BOMS):
        return "utf-16"
    if sample and not all(byte in _TEXT_BYTES for byte in sample):
        return None
    return _text_code_page(sample)


def _file_text_encoding(path: Path, cache: dict[Path, str | None]) -> str | None:
    """Detect the code page of one file, cached per search operation."""

    if path not in cache:
        try:
            with path.open("rb") as stream:
                sample = stream.read(_TEXT_SNIFF_BYTES)
        except OSError:
            cache[path] = "utf-8"
        else:
            cache[path] = _detect_text_encoding(sample)
    return cache[path]


def _read_text_document(path: Path, errors: str) -> str | None:
    """Decode a whole text file with its detected code page, or None when binary."""

    data = path.read_bytes()
    encoding = _detect_text_encoding(data[:_TEXT_SNIFF_BYTES])
    if encoding is None:
        return None
    return data.decode(encoding, errors=errors)


def _ripgrep_encodings(pattern: str) -> tuple[tuple[str, ...], ...]:
    """Return the ripgrep encodings needed to match one pattern faithfully.

    ripgrep only sniffs a byte-order mark, so a Chinese pattern never matches
    cp936/GBK text unless the file is decoded with that code page first.  A
    second pass cannot invent matches: it can only reach text that really is
    encoded that way.  Patterns that match ASCII bytes only, and therefore
    cannot depend on the code page, skip the extra pass.
    """

    if pattern.isascii() and "\\" not in pattern:
        return ((),)
    return ((), ("--encoding", "gb18030"))


def _read_text_range(
    stream: TextIO, offset: int | None, limit: int | None, max_bytes: int
) -> str:
    start_line = offset or 1
    # Bound each read even when a skipped line is much larger than the page.
    for _ in range(start_line - 1):
        while True:
            chunk = stream.readline(64 * 1024)
            if not chunk:
                return ""
            if chunk.endswith("\n"):
                break

    output: list[str] = []
    used = 0
    line_number = start_line
    while limit is None or line_number - start_line < limit:
        line = stream.readline(max_bytes + 1)
        if not line:
            break
        encoded = line.encode("utf-8")
        if len(encoded) > max_bytes - used:
            if output:
                output.append(
                    f"\n[read truncated to {max_bytes} bytes; "
                    f"continue with offset={line_number}]"
                )
            else:
                prefix = encoded[:max_bytes].decode("utf-8", errors="ignore")
                output.append(f"{line_number}|{prefix}")
                output.append(
                    f"\n[read truncated to {max_bytes} bytes within line {line_number}; "
                    "line-based offset cannot retrieve the remainder of this line. "
                    "Use a tool supporting byte ranges to read the full line.]"
                )
            break
        output.append(f"{line_number}|{line}")
        used += len(encoded)
        line_number += 1
    return "".join(output)


def _parse_pdf_page_range(pages: str | None, total_pages: int) -> list[int]:
    if total_pages <= 0:
        return []
    if pages is None:
        return list(range(min(total_pages, 20)))
    page_spec = pages.strip()
    if not page_spec:
        _fail("pages must not be empty", "invalid_page_range")
    try:
        if "-" in page_spec:
            start_text, end_text = page_spec.split("-", 1)
            start_page, end_page = int(start_text.strip()), int(end_text.strip())
        else:
            start_page = end_page = int(page_spec)
    except ValueError:
        _fail("pages must be a page number or range such as 1-5", "invalid_page_range")
    if start_page < 1 or end_page < start_page:
        _fail("PDF page ranges are one-based and increasing", "invalid_page_range")
    if end_page - start_page + 1 > 20:
        _fail("PDF reads are limited to 20 pages", "page_limit_exceeded")
    if start_page > total_pages:
        _fail(f"start page exceeds PDF page count {total_pages}", "page_out_of_range")
    return list(range(start_page - 1, min(end_page, total_pages)))


def _read_pdf_text(path: Path, pages: str | None) -> str:
    try:
        reader = PdfReader(str(path))
        output = []
        for page_index in _parse_pdf_page_range(pages, len(reader.pages)):
            text = reader.pages[page_index].extract_text() or ""
            output.append(f"--- page {page_index + 1} ---\n{text}".rstrip())
        return "\n\n".join(output)
    except ToolExecutionError:
        raise
    except Exception as exc:
        raise ToolExecutionError(
            "PDF extraction failed",
            kind="filesystem_error",
            code="pdf_read_failed",
            side_effect_committed=False,
        ) from exc


def _render_pdf_pages(
    path: Path,
    file_path: str,
    pages: str,
    max_media_bytes: int,
    max_image_output_bytes: int,
    max_image_edge: int,
    max_image_pixels: int,
) -> ToolOutput:
    try:
        document = pdfium.PdfDocument(path)
    except Exception as exc:
        raise ToolExecutionError(
            "PDF rendering failed",
            kind="filesystem_error",
            code="pdf_render_failed",
            side_effect_committed=False,
        ) from exc

    try:
        page_indexes = _parse_pdf_page_range(pages, len(document))
        if not page_indexes:
            _fail("PDF has no pages to render", "empty_pdf")

        content: list[ToolResultText | ToolResultMedia] = []
        rendered_pages: list[dict[str, object]] = []
        total_size_bytes = 0
        for page_index in page_indexes:
            page = document[page_index]
            try:
                width, height = page.get_size()
                pixel_count = ceil(width * _PDF_RENDER_SCALE) * ceil(
                    height * _PDF_RENDER_SCALE
                )
                render_pixel_limit = min(_MAX_PDF_RENDER_PIXELS, max_image_pixels)
                if pixel_count > render_pixel_limit:
                    _fail(
                        f"PDF page {page_index + 1} exceeds the render pixel limit",
                        "pdf_page_too_large",
                    )
                bitmap = page.render(scale=_PDF_RENDER_SCALE)
                try:
                    image = bitmap.to_pil()
                    try:
                        buffer = BytesIO()
                        image.save(buffer, format="PNG")
                        data = buffer.getvalue()
                    finally:
                        image.close()
                finally:
                    bitmap.close()
            finally:
                page.close()

            prepared = _prepare_image(
                data,
                "image/png",
                max_output_bytes=max_image_output_bytes,
                max_edge=max_image_edge,
                max_pixels=max_image_pixels,
            )
            total_size_bytes += len(prepared.data)
            if total_size_bytes > max_media_bytes:
                _fail(
                    "rendered PDF pages exceed the "
                    f"{max_media_bytes}-byte media limit: {path}",
                    "media_too_large",
                )
            source = MediaSource.inline(prepared.data)
            if source.sha256 is None:  # pragma: no cover - inline sources always hash bytes
                raise AssertionError("inline media source is missing its digest")
            page_number = page_index + 1
            rendered_pages.append(
                {
                    "page": page_number,
                    "mime_type": prepared.mime_type,
                    "size_bytes": len(prepared.data),
                    "sha256": source.sha256,
                    "original_dimensions": prepared.original_dimensions,
                    "delivered_dimensions": prepared.delivered_dimensions,
                    "transformed": bool(prepared.transformations),
                    "transformations": prepared.transformations,
                }
            )
            content.extend(
                (
                    ToolResultText(f"Rendered page {page_number} from {path.name}."),
                    ToolResultMedia(
                        media_type="image",
                        mime_type=prepared.mime_type,
                        source=source,
                        detail="high",
                        width=prepared.delivered_dimensions[0],
                        height=prepared.delivered_dimensions[1],
                    ),
                )
            )
        return ToolOutput(
            output=freeze_json_object(
                {
                    "file_path": file_path,
                    "mime_type": "application/pdf",
                    "rendered_pages": rendered_pages,
                    "total_size_bytes": total_size_bytes,
                }
            ),
            content=tuple(content),
        )
    except ToolExecutionError:
        raise
    except Exception as exc:
        raise ToolExecutionError(
            "PDF rendering failed",
            kind="filesystem_error",
            code="pdf_render_failed",
            side_effect_committed=False,
        ) from exc
    finally:
        document.close()


def _truncate_search_line(line: str) -> tuple[str, bool]:
    if len(line) <= _GREP_MAX_LINE_LENGTH:
        return line, False
    return f"{line[:_GREP_MAX_LINE_LENGTH]}... [truncated]", True


def _expand_search_glob(pattern: str) -> list[str]:
    match = re.search(r"\{([^{}]+)\}", pattern)
    if match is None:
        return [pattern]
    prefix, suffix = pattern[: match.start()], pattern[match.end() :]
    expanded: list[str] = []
    for option in match.group(1).split(","):
        expanded.extend(_expand_search_glob(prefix + option + suffix))
    return expanded


def _matches_search_glob(relative_path: str, pattern: str) -> bool:
    candidates: set[str] = set()
    pending = _expand_search_glob(pattern.replace("\\", "/").lstrip("/"))
    while pending:
        candidate = pending.pop()
        if candidate in candidates:
            continue
        candidates.add(candidate)
        marker = candidate.find("**/")
        if marker >= 0:
            pending.append(candidate[:marker] + candidate[marker + 3 :])
    path = PurePosixPath(relative_path)
    return any(path.match(candidate) for candidate in candidates)


def _truncate_search_output(lines: list[str]) -> tuple[str, bool]:
    selected: list[str] = []
    size = 0
    for line in lines:
        encoded_size = len(line.encode("utf-8")) + (1 if selected else 0)
        if size + encoded_size > _SEARCH_MAX_BYTES:
            return "\n".join(selected), True
        selected.append(line)
        size += encoded_size
    return "\n".join(selected), False


def _search_executable(*names: str) -> str | None:
    for name in names:
        executable = shutil.which(name)
        if executable:
            return executable
    return None


def _inside_git_tree(path: Path) -> bool:
    return any((parent / ".git").exists() for parent in (path, *path.parents))


def _gitignore_rules(path: Path) -> tuple[tuple[str, bool, bool], ...]:
    ignore_file = path / ".gitignore"
    try:
        lines = ignore_file.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ()
    rules: list[tuple[str, bool, bool]] = []
    for raw in lines:
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        negated = value.startswith("!")
        if negated:
            value = value[1:]
        value = value.replace("\\", "/").lstrip("/")
        directory_only = value.endswith("/")
        value = value.rstrip("/")
        if value:
            rules.append((value, negated, directory_only))
    return tuple(rules)


def _gitignore_matches(
    relative: str,
    *,
    is_dir: bool,
    rules: tuple[tuple[PurePosixPath, tuple[tuple[str, bool, bool], ...]], ...],
) -> bool:
    ignored = False
    candidate = PurePosixPath(relative)
    for base, entries in rules:
        try:
            scoped = candidate.relative_to(base) if base.parts else candidate
        except ValueError:
            continue
        for pattern, negated, directory_only in entries:
            if directory_only and not is_dir:
                continue
            if "/" in pattern:
                matched = scoped.match(pattern) or scoped.match(f"**/{pattern}")
            else:
                matched = any(fnmatch.fnmatchcase(part, pattern) for part in scoped.parts)
            if matched:
                ignored = not negated
    return ignored


async def _terminate_search_process(
    process: asyncio.subprocess.Process,
    stderr_task: asyncio.Task[bytes],
) -> None:
    """Kill a search backend, drain its pipes, and reap it in that order.

    On Windows, waiting for a subprocess that still has buffered PIPE output can
    block even after ``kill()``.  The caller owns stdout while ``stderr_task``
    owns stderr, so both readers must reach EOF before the process is reaped.
    """

    if process.returncode is None:
        with suppress(ProcessLookupError):
            process.kill()

    assert process.stdout is not None
    with suppress(OSError, RuntimeError):
        await process.stdout.read()

    try:
        await stderr_task
    except asyncio.CancelledError:
        # asyncio.run() cancels every task at shutdown, so the independently
        # owned stderr reader may already have been cancelled.  Resume draining
        # only after that reader has released StreamReader's single-reader lock.
        assert process.stderr is not None
        with suppress(OSError, RuntimeError):
            await process.stderr.read()

    with suppress(ProcessLookupError):
        await process.wait()


def _relative_search_path(candidate: Path, root: Path) -> str:
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        relative = Path(os.path.relpath(candidate, root))
    value = relative.as_posix()
    if candidate.is_dir() and not value.endswith("/"):
        value += "/"
    return value


def _walk_search_files(
    root: Path,
    workspace_root: Path,
    limit: int,
    cancelled: threading.Event | None = None,
) -> list[Path]:
    """Bounded fallback traversal with default and hierarchical git ignores."""

    try:
        root_relative = root.relative_to(workspace_root)
        rule_root = workspace_root
    except ValueError:
        root_relative = Path()
        rule_root = root

    inherited: list[
        tuple[PurePosixPath, tuple[tuple[str, bool, bool], ...]]
    ] = []
    current = rule_root
    relative = PurePosixPath()
    for part in root_relative.parts:
        rules = _gitignore_rules(current)
        if rules:
            inherited.append((relative, rules))
        current /= part
        relative /= part

    files: list[Path] = []

    def visit(
        directory: Path,
        relative_directory: PurePosixPath,
        parent_rules: tuple[
            tuple[PurePosixPath, tuple[tuple[str, bool, bool], ...]], ...
        ],
    ) -> None:
        if len(files) >= limit or (cancelled is not None and cancelled.is_set()):
            return
        local_rules = _gitignore_rules(directory)
        rules = parent_rules + (
            ((relative_directory, local_rules),) if local_rules else ()
        )
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError:
            return
        for entry in entries:
            if len(files) >= limit or (cancelled is not None and cancelled.is_set()):
                return
            entry_relative = relative_directory / entry.name
            entry_text = entry_relative.as_posix()
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
                is_file = entry.is_file(follow_symlinks=False)
            except OSError:
                continue
            if is_dir:
                if entry.name in _DEFAULT_SEARCH_IGNORES or _gitignore_matches(
                    entry_text, is_dir=True, rules=rules
                ):
                    continue
                visit(Path(entry.path), entry_relative, rules)
            elif is_file and not _gitignore_matches(
                entry_text, is_dir=False, rules=rules
            ):
                files.append(Path(entry.path))

    visit(root, PurePosixPath(root_relative.as_posix()), tuple(inherited))
    return files


async def _run_owned_thread(function, *args):
    """Do not return cancellation while a blocking adapter still owns resources."""

    operation = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(operation)
    except asyncio.CancelledError:
        with suppress(Exception):
            await operation
        raise


async def _run_cancellable_search_thread(function, *args):
    """Request cooperative stop and join a fallback search before cancellation."""

    cancelled = threading.Event()
    operation = asyncio.create_task(asyncio.to_thread(function, *args, cancelled))
    try:
        return await asyncio.shield(operation)
    except asyncio.CancelledError:
        cancelled.set()
        with suppress(Exception):
            await operation
        raise


def _atomic_write_text(path: Path, content: str) -> None:
    """Commit complete UTF-8 content with one same-directory replace."""

    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".pygent-tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


class FileTools:
    """Deployment-local workspace file handlers.

    The instance and its workspace configuration stay in the executor registry;
    only projections created by :class:`ToolKit` are portable.
    """

    def __init__(
        self,
        *,
        workspace_root: str | Path,
        restrict_to_workspace: bool = True,
        max_read_bytes: int = 1024 * 1024,
        max_media_bytes: int = _DEFAULT_MAX_MEDIA_BYTES,
        max_image_output_bytes: int = _DEFAULT_MAX_IMAGE_OUTPUT_BYTES,
        max_image_edge: int = _DEFAULT_MAX_IMAGE_EDGE,
        max_image_pixels: int = _DEFAULT_MAX_IMAGE_PIXELS,
        max_video_output_bytes: int = _DEFAULT_MAX_VIDEO_OUTPUT_BYTES,
        max_video_duration_seconds: float = _DEFAULT_MAX_VIDEO_DURATION_SECONDS,
        max_video_edge: int = _DEFAULT_MAX_VIDEO_EDGE,
        max_video_fps: float = _DEFAULT_MAX_VIDEO_FPS,
        max_search_files: int = 10_000,
    ) -> None:
        if any(
            limit <= 0
            for limit in (
                max_read_bytes,
                max_media_bytes,
                max_image_output_bytes,
                max_image_edge,
                max_image_pixels,
                max_video_output_bytes,
                max_video_duration_seconds,
                max_video_edge,
                max_video_fps,
                max_search_files,
            )
        ):
            raise ValueError("file tool limits must be positive")
        self.path_context = ToolPathContext.from_workspace_root(
            workspace_root, restrict_to_workspace=restrict_to_workspace
        )
        self.workspace_root = self.path_context.workspace_root
        self.max_read_bytes = max_read_bytes
        self.max_media_bytes = max_media_bytes
        self.max_image_output_bytes = max_image_output_bytes
        self.max_image_edge = max_image_edge
        self.max_image_pixels = max_image_pixels
        self.max_video_output_bytes = max_video_output_bytes
        self.max_video_duration_seconds = max_video_duration_seconds
        self.max_video_edge = max_video_edge
        self.max_video_fps = max_video_fps
        self.max_search_files = max_search_files
        self._mutation_locks = tuple(threading.Lock() for _ in range(64))
        self._io_service = FileIOService(
            lambda *args: self._read(*args),
            lambda *args: self._write(*args),
            lambda *args: self._edit(*args),
        )
        self._notebook_service = NotebookService(
            lambda *args: self._edit_notebook(*args)
        )
        self._diagnostics_service = FileDiagnosticsService(
            lambda *args: self._read_lints(*args)
        )
        self._search_service = FileSearchService(
            lambda *args: self._glob(*args), lambda *args: self._grep(*args)
        )

    def _mutation_lock(self, path: Path) -> threading.Lock:
        return self._mutation_locks[hash(path) % len(self._mutation_locks)]

    @property
    def handlers(self) -> tuple[Any, ...]:
        """Return handlers in their stable model-visible order."""

        return (
            self.edit,
            self.edit_notebook,
            self.glob,
            self.grep,
            self.read,
            self.read_lints,
            self.write,
        )

    @tool(
        tool_id="standard.files.read",
        version="3.3.0",
        description=_READ_TOOL_DESCRIPTION,
        side_effect=ToolSideEffect.READ,
        timeout=30,
        resource_key="filesystem",
        sandbox_profile="workspace",
        required_permissions=("filesystem:read",),
    )
    async def read(
        self,
        file_path: Annotated[
            str,
            Field(
                description="File path resolved from workspace_root and restricted to it by default."
            ),
        ],
        limit: Annotated[
            int | None, Field(gt=0, description="Maximum number of text lines to return.")
        ] = None,
        offset: Annotated[
            int | None, Field(gt=0, description="Starting text line, numbered from 1.")
        ] = None,
        pages: Annotated[
            str | None,
            Field(
                description="PDF page number or inclusive range (for example, 2 or 2-5). When provided, pages are rendered as images."
            ),
        ] = None,
    ) -> ToolOutput:
        """Read text, PDF, image or video content from a workspace file.

        PDF files return extracted text by default. When pages is provided, the
        selected PDF pages are rendered as images for direct model inspection.
        """

        return await _run_owned_thread(
            self._io_service.read, file_path, limit, offset, pages
        )

    def _read(
        self, file_path: str, limit: int | None, offset: int | None, pages: str | None
    ) -> ToolOutput:
        path = resolve_file_path(file_path, self.path_context)
        if not path.exists():
            _fail(f"file does not exist: {path}", "file_not_found")
        if not path.is_file():
            _fail(f"path is not a file: {path}", "not_a_file")
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            if pages is not None:
                if limit is not None or offset is not None:
                    _fail(
                        "limit and offset cannot be combined with rendered PDF pages",
                        "invalid_read_options",
                    )
                return _render_pdf_pages(
                    path,
                    file_path,
                    pages,
                    self.max_media_bytes,
                    self.max_image_output_bytes,
                    self.max_image_edge,
                    self.max_image_pixels,
                )
            return _text_output(_read_pdf_text(path, None))
        if pages:
            _fail("pages applies only to PDF files", "invalid_page_range")
        media_type: Literal["image", "video"] | None = None
        mime_type = _IMAGE_MIME_TYPES.get(suffix)
        if mime_type is not None:
            media_type = "image"
        else:
            mime_type = _VIDEO_MIME_TYPES.get(suffix)
            if mime_type is not None:
                media_type = "video"
        if media_type is not None:
            if mime_type is None:  # pragma: no cover - paired by dispatch above
                raise AssertionError("media dispatch is missing a MIME type")
            if limit is not None or offset is not None:
                _fail(
                    "limit and offset apply only to text files",
                    "invalid_read_options",
                )
            return self._read_media(path, file_path, media_type, mime_type)
        try:
            with path.open("rb") as stream:
                sample = stream.read(self.max_read_bytes)
                encoding = _detect_text_encoding(sample)
                if encoding is not None:
                    stream.seek(0)
                    with TextIOWrapper(
                        stream, encoding=encoding, errors="replace"
                    ) as text:
                        return _text_output(
                            _read_text_range(text, offset, limit, self.max_read_bytes)
                        )
        except OSError as exc:
            raise ToolExecutionError(
                f"could not read file: {path}",
                kind="filesystem_error",
                code="read_failed",
                retryable=True,
                side_effect_committed=False,
            ) from exc
        label = "image" if suffix in _IMAGE_SUFFIXES else "binary"
        return _text_output(
            f"[{label} file {path.name}, size {path.stat().st_size} bytes]"
        )

    def _read_media(
        self,
        path: Path,
        file_path: str,
        media_type: Literal["image", "video"],
        mime_type: str,
    ) -> ToolOutput:
        try:
            with path.open("rb") as stream:
                data = stream.read(self.max_media_bytes + 1)
        except OSError as exc:
            raise ToolExecutionError(
                f"could not read media file: {path}",
                kind="filesystem_error",
                code="read_failed",
                retryable=True,
                side_effect_committed=False,
            ) from exc
        if not data:
            _fail(f"media file is empty: {path}", "empty_media")
        if len(data) > self.max_media_bytes:
            _fail(
                f"media file exceeds the {self.max_media_bytes}-byte limit: {path}",
                "media_too_large",
            )
        if not _media_signature_matches(data, mime_type):
            _fail(
                f"media content does not match {mime_type}: {path}",
                "media_mime_mismatch",
            )
        metadata: dict[str, object] = {
            "file_path": file_path,
            "mime_type": mime_type,
            "size_bytes": len(data),
        }
        delivered_image_dimensions: tuple[int, int] | None = None
        delivered_video_metadata: _VideoMetadata | None = None
        if media_type == "image":
            original_mime_type = mime_type
            original_size_bytes = len(data)
            prepared = _prepare_image(
                data,
                mime_type,
                max_output_bytes=self.max_image_output_bytes,
                max_edge=self.max_image_edge,
                max_pixels=self.max_image_pixels,
            )
            data = prepared.data
            mime_type = prepared.mime_type
            delivered_image_dimensions = prepared.delivered_dimensions
            metadata.update(
                {
                    "mime_type": mime_type,
                    "size_bytes": len(data),
                    "original_mime_type": original_mime_type,
                    "original_size_bytes": original_size_bytes,
                    "original_dimensions": prepared.original_dimensions,
                    "delivered_dimensions": prepared.delivered_dimensions,
                    "transformed": bool(prepared.transformations),
                    "transformations": prepared.transformations,
                }
            )
        elif media_type == "video":
            original_size_bytes = len(data)
            prepared_video = _prepare_video(
                data,
                max_output_bytes=self.max_video_output_bytes,
                max_duration_seconds=self.max_video_duration_seconds,
                max_edge=self.max_video_edge,
                max_fps=self.max_video_fps,
            )
            data = prepared_video.data
            delivered_video_metadata = prepared_video.delivered_metadata
            metadata.update(
                {
                    "size_bytes": len(data),
                    "original_size_bytes": original_size_bytes,
                    "processing": prepared_video.processing,
                    "transformed": bool(prepared_video.transformations),
                    "transformations": prepared_video.transformations,
                    "original_video": _video_metadata_value(
                        prepared_video.original_metadata
                    ),
                    "delivered_video": _video_metadata_value(
                        prepared_video.delivered_metadata
                    ),
                }
            )
            if prepared_video.processing == "passthrough_unverified":
                metadata["processing_hint"] = _VIDEO_BACKEND_HINT
        source = MediaSource.inline(data)
        if source.sha256 is None:  # pragma: no cover - inline sources always hash bytes
            raise AssertionError("inline media source is missing its digest")
        metadata["sha256"] = source.sha256
        result_text = f"Read {media_type} file {path.name}."
        if (
            media_type == "video"
            and metadata.get("processing") == "passthrough_unverified"
        ):
            result_text = f"{result_text} {_VIDEO_BACKEND_HINT}"
        return ToolOutput(
            output=freeze_json_object(metadata),
            content=(
                ToolResultText(result_text),
                ToolResultMedia(
                    media_type=media_type,
                    mime_type=mime_type,
                    source=source,
                    width=(
                        delivered_image_dimensions[0]
                        if delivered_image_dimensions is not None
                        else (
                            None
                            if delivered_video_metadata is None
                            else delivered_video_metadata.dimensions[0]
                        )
                    ),
                    height=(
                        delivered_image_dimensions[1]
                        if delivered_image_dimensions is not None
                        else (
                            None
                            if delivered_video_metadata is None
                            else delivered_video_metadata.dimensions[1]
                        )
                    ),
                    duration_seconds=(
                        None
                        if delivered_video_metadata is None
                        else delivered_video_metadata.duration_seconds
                    ),
                    fps=(
                        None
                        if delivered_video_metadata is None
                        else delivered_video_metadata.fps
                    ),
                    has_audio=(
                        None
                        if delivered_video_metadata is None
                        else delivered_video_metadata.audio_codec is not None
                    ),
                ),
            ),
        )

    @tool(
        tool_id="standard.files.write",
        version="2.1.0",
        side_effect=ToolSideEffect.WRITE,
        description=_WRITE_TOOL_DESCRIPTION,
        idempotency=IdempotencyPolicy.INHERENT,
        timeout=30,
        resource_key="filesystem",
        sandbox_profile="workspace",
        required_permissions=("filesystem:write",),
    )
    async def write(
        self,
        file_path: Annotated[
            str,
            Field(
                description="Destination resolved from workspace_root and restricted to it by default."
            ),
        ],
        content: Annotated[
            str,
            Field(
                description=(
                    "Complete literal UTF-8 content for the entire file. Do not use "
                    "omission placeholders such as '...', 'rest of file', or "
                    "'unchanged code'."
                )
            ),
        ],
    ) -> str:
        """Create or completely replace a UTF-8 file."""

        return await _run_owned_thread(self._io_service.write, file_path, content)

    def _write(self, file_path: str, content: str) -> str:
        path = resolve_file_path(file_path, self.path_context)
        if path.exists() and path.is_dir():
            _fail(f"file_path points to a directory: {path}", "not_a_file")
        try:
            with self._mutation_lock(path):
                path.parent.mkdir(parents=True, exist_ok=True)
                _atomic_write_text(path, content)
        except OSError as exc:
            raise ToolExecutionError(
                f"could not write file: {path}",
                kind="filesystem_error",
                code="write_failed",
                retryable=True,
                side_effect_committed=None,
            ) from exc
        return "写入完成"

    @tool(
        tool_id="standard.files.edit",
        version="2.1.0",
        side_effect=ToolSideEffect.WRITE,
        description=_EDIT_TOOL_DESCRIPTION,
        idempotency=IdempotencyPolicy.NOT_IDEMPOTENT,
        timeout=30,
        resource_key="filesystem",
        sandbox_profile="workspace",
        required_permissions=("filesystem:write",),
    )
    async def edit(
        self,
        file_path: Annotated[
            str,
            Field(
                description="File resolved from workspace_root and restricted to it by default."
            ),
        ],
        old_string: Annotated[
            str,
            Field(
                description=(
                    "Exact literal text to replace, including whitespace and "
                    "indentation. Include only enough unchanged surrounding context "
                    "to target the intended occurrence; do not include long runs of "
                    "unchanged text. For complex or long changes, you MUST split the "
                    "work into multiple smaller atomic edit calls."
                )
            ),
        ],
        new_string: Annotated[
            str,
            Field(
                description=(
                    "Exact literal replacement text, including whitespace and "
                    "indentation."
                )
            ),
        ],
        replace_all: Annotated[
            bool,
            Field(
                description=(
                    "Replace every exact occurrence when true; otherwise replace only "
                    "the first occurrence."
                )
            ),
        ] = False,
    ) -> str:
        """Replace exact literal text in an existing file."""

        return await _run_owned_thread(
            self._io_service.edit, file_path, old_string, new_string, replace_all
        )

    def _edit(
        self, file_path: str, old_string: str, new_string: str, replace_all: bool
    ) -> str:
        if old_string == new_string:
            _fail("new_string must differ from old_string", "identical_replacement")
        path = resolve_file_path(file_path, self.path_context)
        if not path.exists() or not path.is_file():
            _fail(f"file does not exist: {path}", "file_not_found")
        try:
            with self._mutation_lock(path):
                text = _read_text_document(path, errors="strict")
                if text is None:
                    _fail(f"file is not text: {path}", "unsupported_text_encoding")
                if old_string not in text:
                    _fail("exact old_string was not found", "match_not_found")
                updated = text.replace(
                    old_string, new_string, -1 if replace_all else 1
                )
                _atomic_write_text(path, updated)
        except ToolExecutionError:
            raise
        except UnicodeDecodeError as exc:
            raise ToolExecutionError(
                f"file is not decodable text: {path}",
                kind="filesystem_error",
                code="unsupported_text_encoding",
                side_effect_committed=False,
            ) from exc
        except OSError as exc:
            raise ToolExecutionError(
                f"could not edit file: {path}",
                kind="filesystem_error",
                code="edit_failed",
                retryable=True,
                side_effect_committed=None,
            ) from exc
        return "替换完成"

    @tool(
        tool_id="standard.files.edit_notebook",
        version="2.0.0",
        side_effect=ToolSideEffect.WRITE,
        idempotency=IdempotencyPolicy.NOT_IDEMPOTENT,
        timeout=30,
        resource_key="filesystem",
        sandbox_profile="workspace",
        required_permissions=("filesystem:write",),
    )
    async def edit_notebook(
        self,
        target_notebook: str,
        cell_idx: Annotated[int, Field(ge=0)],
        is_new_cell: bool,
        cell_language: str,
        old_string: str,
        new_string: str,
    ) -> str:
        """Insert or edit one Jupyter notebook cell."""

        return await _run_owned_thread(
            self._notebook_service.edit,
            target_notebook,
            cell_idx,
            is_new_cell,
            cell_language,
            old_string,
            new_string,
        )

    def _edit_notebook(
        self,
        target_notebook: str,
        cell_idx: int,
        is_new_cell: bool,
        cell_language: str,
        old_string: str,
        new_string: str,
    ) -> str:
        path = resolve_file_path(target_notebook, self.path_context)
        if not path.exists() or not path.is_file():
            _fail(f"notebook does not exist: {path}", "file_not_found")
        try:
            with self._mutation_lock(path):
                document = _read_text_document(path, errors="strict")
                if document is None:
                    _fail(f"notebook is not text: {path}", "unsupported_text_encoding")
                notebook = json.loads(document)
                cells = notebook.get("cells")
                if not isinstance(cells, list):
                    _fail("notebook cells must be a list", "invalid_notebook")
                if is_new_cell:
                    if cell_idx > len(cells):
                        _fail(
                            f"cell index out of range 0..{len(cells)}",
                            "cell_index_out_of_range",
                        )
                    code_languages = {
                        "python",
                        "javascript",
                        "typescript",
                        "r",
                        "sql",
                        "shell",
                    }
                    cell_type = (
                        "code"
                        if cell_language in code_languages
                        else "markdown"
                        if cell_language == "markdown"
                        else "raw"
                    )
                    cells.insert(
                        cell_idx,
                        {
                            "cell_type": cell_type,
                            "metadata": {},
                            "source": new_string.splitlines(keepends=True),
                        },
                    )
                else:
                    if cell_idx >= len(cells):
                        _fail(
                            f"cell index out of range 0..{len(cells) - 1}",
                            "cell_index_out_of_range",
                        )
                    cell = cells[cell_idx]
                    source = cell.get("source", [])
                    content = (
                        "".join(source) if isinstance(source, list) else str(source)
                    )
                    if old_string not in content:
                        _fail("old_string was not found in the cell", "match_not_found")
                    cell["source"] = content.replace(
                        old_string, new_string, 1
                    ).splitlines(keepends=True)
                notebook["cells"] = cells
                _atomic_write_text(
                    path, json.dumps(notebook, ensure_ascii=False, indent=1)
                )
        except ToolExecutionError:
            raise
        except UnicodeDecodeError as exc:
            raise ToolExecutionError(
                f"notebook is not decodable text: {path}",
                kind="filesystem_error",
                code="unsupported_text_encoding",
                side_effect_committed=False,
            ) from exc
        except json.JSONDecodeError as exc:
            raise ToolExecutionError(
                "notebook is not valid JSON",
                kind="filesystem_error",
                code="invalid_notebook",
                side_effect_committed=False,
            ) from exc
        except OSError as exc:
            raise ToolExecutionError(
                f"could not update notebook: {path}",
                kind="filesystem_error",
                code="notebook_write_failed",
                side_effect_committed=None,
            ) from exc
        return "笔记本已更新"

    @tool(
        tool_id="standard.files.read_lints",
        version="2.0.0",
        side_effect=ToolSideEffect.READ,
        timeout=30,
        resource_key="filesystem",
        sandbox_profile="workspace",
        required_permissions=("filesystem:read",),
    )
    async def read_lints(self, paths: list[str] | None = None) -> str:
        """Return bounded Python syntax diagnostics for files or directories."""

        return await _run_owned_thread(self._diagnostics_service.read_lints, paths)

    def _read_lints(self, paths: list[str] | None) -> str:
        requested = paths or ["."]
        files: list[Path] = []
        for value in requested:
            path = resolve_tool_path(value, self.path_context)
            if not path.exists():
                _fail(f"lint path does not exist: {path}", "file_not_found")
            if path.is_file() and path.suffix.lower() == ".py":
                files.append(path)
            elif path.is_dir():
                remaining = max(0, self.max_search_files - len(files))
                files.extend(islice(path.rglob("*.py"), remaining))
        diagnostics = []
        for path in sorted(set(files))[: self.max_search_files]:
            try:
                source = _read_text_document(path, errors="replace")
                if source is None:
                    continue
                compile(source, str(path), "exec")
            except SyntaxError as exc:
                diagnostics.append(
                    {
                        "path": str(path),
                        "line": exc.lineno,
                        "column": exc.offset,
                        "message": exc.msg,
                    }
                )
            except OSError:
                continue
        return json.dumps(
            {"tool": "python.compile", "diagnostics": diagnostics},
            ensure_ascii=False,
            sort_keys=True,
        )

    @tool(
        tool_id="standard.files.glob",
        version="3.0.0",
        side_effect=ToolSideEffect.READ,
        timeout=30,
        resource_key="filesystem",
        sandbox_profile="workspace",
        required_permissions=("filesystem:read",),
    )
    async def glob(
        self,
        pattern: Annotated[
            str,
            Field(
                description=(
                    "Glob pattern to match files, e.g. '*.ts', '**/*.json', "
                    "or 'src/**/*.spec.ts'"
                )
            ),
        ],
        path: Annotated[
            str,
            Field(description="Directory to search in (default: current directory)"),
        ] = "",
        limit: Annotated[
            int,
            Field(gt=0, description="Maximum number of results (default: 1000)"),
        ] = 1000,
    ) -> str:
        """Search for files by glob pattern while respecting ignore files."""

        return await self._search_service.glob(pattern, path, limit)

    async def _glob(self, pattern: str, path: str | None, limit: int) -> str:
        root = resolve_dir_path(path, self.path_context)
        if not root.exists():
            _fail(f"path does not exist: {root}", "file_not_found")
        if not root.is_dir():
            _fail(f"path must be a directory: {root}", "not_a_directory")
        pattern_parts = Path(pattern.replace("\\", "/")).parts
        if is_absolute_tool_path(pattern) or ".." in pattern_parts:
            _fail(
                "glob pattern must stay within workspace_root",
                "path_outside_workspace",
            )

        effective_limit = min(limit, self.max_search_files)
        executable = _search_executable("fd", "fdfind")
        process_cwd: Path | None = None
        if executable:
            arguments = ["--glob", "--color=never", "--hidden"]
            for ignored in sorted(_DEFAULT_SEARCH_IGNORES):
                arguments.extend(("--exclude", ignored))
            if not _inside_git_tree(root):
                arguments.append("--no-require-git")
            arguments.extend(("--max-results", str(effective_limit)))
            effective_pattern = pattern
            if "/" in pattern:
                arguments.append("--full-path")
                if (
                    not pattern.startswith("/")
                    and not pattern.startswith("**/")
                    and pattern != "**"
                ):
                    effective_pattern = f"**/{pattern}"
                if os.name == "nt":
                    effective_pattern = effective_pattern.replace("/", "[/\\\\]")
            arguments.extend(("--", effective_pattern, str(root)))
        else:
            executable = _search_executable("rg")
            if executable is None:
                return await _run_cancellable_search_thread(
                    self._glob_fallback, root, pattern, effective_limit
                )
            arguments = [
                "--files",
                "--hidden",
            ]
            for ignored in sorted(_DEFAULT_SEARCH_IGNORES):
                arguments.extend(("--glob", f"!**/{ignored}/**"))
            if not _inside_git_tree(root):
                arguments.append("--no-require-git")
            arguments.extend(("--", "."))
            process_cwd = root

        process = await self._start_search_process(
            executable, arguments, cwd=process_cwd
        )
        assert process.stderr is not None
        stderr_task = asyncio.create_task(process.stderr.read())
        lines: list[str] = []
        reached_limit = False
        try:
            assert process.stdout is not None
            while raw_line := await process.stdout.readline():
                value = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                if not value:
                    continue
                candidate = Path(value)
                if not candidate.is_absolute():
                    candidate = root / candidate
                candidate = resolve_tool_path(str(candidate), self.path_context)
                relative = _relative_search_path(candidate, root)
                if not _matches_search_glob(relative.rstrip("/"), pattern):
                    continue
                lines.append(relative)
                if len(lines) >= effective_limit:
                    reached_limit = True
                    await _terminate_search_process(process, stderr_task)
                    break
            stderr = await stderr_task
            code = await process.wait()
        except BaseException:
            await _terminate_search_process(process, stderr_task)
            with suppress(asyncio.CancelledError):
                await stderr_task
            raise

        if not reached_limit and code not in (0, 1):
            self._raise_search_failure(stderr, code, "file search backend")
        if not lines:
            return "No files found matching pattern"
        output, bytes_truncated = _truncate_search_output(lines)
        notices = []
        if reached_limit:
            notices.append(
                f"{effective_limit} results limit reached. Use limit="
                f"{effective_limit * 2} for more, or refine pattern"
            )
        if bytes_truncated:
            notices.append("50KB limit reached")
        return output + (f"\n\n[{'. '.join(notices)}]" if notices else "")

    def _glob_fallback(
        self,
        root: Path,
        pattern: str,
        limit: int,
        cancelled: threading.Event | None = None,
    ) -> str:
        candidates = _walk_search_files(
            root, self.workspace_root, self.max_search_files, cancelled
        )
        lines = []
        for candidate in candidates:
            if cancelled is not None and cancelled.is_set():
                break
            relative = _relative_search_path(candidate, root)
            if _matches_search_glob(relative.rstrip("/"), pattern):
                lines.append(relative)
        reached_limit = len(lines) > limit
        lines = lines[:limit]
        if not lines:
            return "No files found matching pattern"
        output, bytes_truncated = _truncate_search_output(lines)
        notices = []
        if reached_limit:
            notices.append(
                f"{limit} results limit reached. Use limit={limit * 2} for more, "
                "or refine pattern"
            )
        if bytes_truncated:
            notices.append("50KB limit reached")
        return output + (f"\n\n[{'. '.join(notices)}]" if notices else "")

    @tool(
        tool_id="standard.files.grep",
        version="3.0.0",
        side_effect=ToolSideEffect.READ,
        timeout=30,
        resource_key="filesystem",
        sandbox_profile="workspace",
        required_permissions=("filesystem:read",),
    )
    async def grep(
        self,
        pattern: Annotated[
            str, Field(description="Search pattern (regex or literal string)")
        ],
        path: Annotated[
            str,
            Field(description="Directory or file to search (default: current directory)"),
        ] = "",
        glob: Annotated[
            str,
            Field(
                description="Filter files by glob pattern, e.g. '*.ts' or '**/*.spec.ts'"
            ),
        ] = "",
        ignoreCase: Annotated[
            bool, Field(description="Case-insensitive search (default: false)")
        ] = False,
        literal: Annotated[
            bool,
            Field(
                description="Treat pattern as literal string instead of regex (default: false)"
            ),
        ] = False,
        context: Annotated[
            int,
            Field(
                ge=0,
                description="Number of lines to show before and after each match (default: 0)",
            ),
        ] = 0,
        limit: Annotated[
            int,
            Field(gt=0, description="Maximum number of matches to return (default: 100)"),
        ] = 100,
    ) -> str:
        """Search file contents with ripgrep while respecting ignore files."""

        return await self._search_service.grep(
            pattern, path, glob, ignoreCase, literal, context, limit
        )

    async def _grep(
        self,
        pattern: str,
        path: str | None,
        glob: str | None,
        ignore_case: bool,
        literal: bool,
        context: int,
        limit: int,
    ) -> str:
        root = resolve_tool_path(path, self.path_context, default=".")
        if not root.exists():
            _fail(f"path does not exist: {root}", "file_not_found")
        executable = _search_executable("rg")
        if executable is None:
            return await _run_cancellable_search_thread(
                self._grep_fallback,
                root,
                pattern,
                glob,
                ignore_case,
                literal,
                context,
                min(limit, self.max_search_files),
            )
        effective_limit = min(limit, self.max_search_files)
        arguments = ["--json", "--line-number", "--color=never", "--hidden"]
        for ignored in sorted(_DEFAULT_SEARCH_IGNORES):
            arguments.extend(("--glob", f"!**/{ignored}/**"))
        if ignore_case:
            arguments.append("--ignore-case")
        if literal:
            arguments.append("--fixed-strings")
        if not _inside_git_tree(root if root.is_dir() else root.parent):
            arguments.append("--no-require-git")
        process_cwd = root if root.is_dir() else root.parent
        search_target = "." if root.is_dir() else root.name
        search_root = root if root.is_dir() else root.parent

        encodings: dict[Path, str | None] = {}
        matches: list[tuple[Path, int, str]] = []
        seen: set[tuple[Path, int]] = set()
        reached_limit = False
        for encoding_arguments in _ripgrep_encodings(pattern):
            found, reached_limit = await self._run_ripgrep_pass(
                executable,
                [*arguments, *encoding_arguments, "--", pattern, search_target],
                cwd=process_cwd,
                root=root,
                search_root=search_root,
                glob=glob,
                limit=effective_limit - len(matches),
                encodings=encodings,
                seen=seen,
            )
            for found_path, found_line, found_text in found:
                seen.add((found_path, found_line))
                matches.append((found_path, found_line, found_text))
            if reached_limit:
                break

        if not matches:
            return "No matches found"

        file_cache: dict[Path, list[str]] = {}
        output_lines: list[str] = []
        lines_truncated = False
        for file_path, line_number, matched_text in matches:
            relative = _relative_search_path(file_path, search_root)
            if context == 0:
                value = matched_text.replace("\r\n", "\n").replace("\r", "")
                value = value.removesuffix("\n")
                value, truncated = _truncate_search_line(value)
                lines_truncated |= truncated
                output_lines.append(f"{relative}:{line_number}: {value}")
                continue
            lines = file_cache.get(file_path)
            if lines is None:
                try:
                    text = _read_text_document(file_path, errors="replace")
                except OSError:
                    text = None
                lines = (
                    []
                    if text is None
                    else text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
                )
                file_cache[file_path] = lines
            if not lines:
                output_lines.append(f"{relative}:{line_number}: (unable to read file)")
                continue
            start = max(1, line_number - context)
            end = min(len(lines), line_number + context)
            for current in range(start, end + 1):
                value, truncated = _truncate_search_line(lines[current - 1])
                lines_truncated |= truncated
                separator = ":" if current == line_number else "-"
                output_lines.append(f"{relative}{separator}{current}{separator} {value}")

        output, bytes_truncated = _truncate_search_output(output_lines)
        notices = []
        if reached_limit:
            notices.append(
                f"{effective_limit} matches limit reached. Use limit="
                f"{effective_limit * 2} for more, or refine pattern"
            )
        if bytes_truncated:
            notices.append("50KB limit reached")
        if lines_truncated:
            notices.append(
                "Some lines truncated to 500 chars. Use read tool to see full lines"
            )
        return output + (f"\n\n[{'. '.join(notices)}]" if notices else "")

    async def _run_ripgrep_pass(
        self,
        executable: str,
        arguments: list[str],
        *,
        cwd: Path,
        root: Path,
        search_root: Path,
        glob: str | None,
        limit: int,
        encodings: dict[Path, str | None],
        seen: set[tuple[Path, int]],
    ) -> tuple[list[tuple[Path, int, str]], bool]:
        """Collect matches from one ripgrep invocation."""

        matches: list[tuple[Path, int, str]] = []
        reached_limit = False
        process = await self._start_search_process(executable, arguments, cwd=cwd)
        assert process.stderr is not None
        stderr_task = asyncio.create_task(process.stderr.read())
        try:
            assert process.stdout is not None
            while raw_line := await process.stdout.readline():
                try:
                    event = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if event.get("type") != "match":
                    continue
                data = event.get("data", {})
                raw_path = data.get("path", {}).get("text")
                line_number = data.get("line_number")
                fields = data.get("lines", {})
                line_text = fields.get("text")
                if not isinstance(raw_path, str) or not isinstance(line_number, int):
                    continue
                candidate = Path(raw_path)
                if not candidate.is_absolute():
                    candidate = root.parent / candidate if root.is_file() else root / candidate
                candidate = resolve_file_path(str(candidate), self.path_context)
                relative = _relative_search_path(candidate, search_root)
                if glob and not _matches_search_glob(relative, glob):
                    continue
                if (candidate, line_number) in seen:
                    continue
                if not isinstance(line_text, str):
                    # ripgrep reports non-UTF-8 lines as raw base64 bytes.
                    raw_bytes = fields.get("bytes")
                    line_text = (
                        base64.b64decode(raw_bytes).decode(
                            _file_text_encoding(candidate, encodings) or "utf-8",
                            errors="replace",
                        )
                        if isinstance(raw_bytes, str)
                        else ""
                    )
                matches.append((candidate, line_number, line_text))
                if len(matches) >= limit:
                    reached_limit = True
                    await _terminate_search_process(process, stderr_task)
                    break
            stderr = await stderr_task
            code = await process.wait()
        except BaseException:
            await _terminate_search_process(process, stderr_task)
            with suppress(asyncio.CancelledError):
                await stderr_task
            raise

        if not reached_limit and code not in (0, 1):
            self._raise_search_failure(stderr, code, "ripgrep")
        return matches, reached_limit

    def _grep_fallback(
        self,
        root: Path,
        pattern: str,
        glob: str | None,
        ignore_case: bool,
        literal: bool,
        context: int,
        limit: int,
        cancelled: threading.Event | None = None,
    ) -> str:
        flags = re.IGNORECASE if ignore_case else 0
        try:
            matcher = re.compile(re.escape(pattern) if literal else pattern, flags)
        except re.error as exc:
            raise ToolExecutionError(
                f"invalid search pattern: {exc}",
                kind="filesystem_error",
                code="search_backend_failed",
                side_effect_committed=False,
            ) from exc

        search_root = root if root.is_dir() else root.parent
        candidates = (
            _walk_search_files(
                root, self.workspace_root, self.max_search_files, cancelled
            )
            if root.is_dir()
            else [root]
        )
        matches: list[tuple[Path, int, str, list[str]]] = []
        reached_limit = False
        for candidate in candidates:
            if cancelled is not None and cancelled.is_set():
                break
            relative = _relative_search_path(candidate, search_root)
            if glob and not _matches_search_glob(relative, glob):
                continue
            try:
                data = candidate.read_bytes()
            except OSError:
                continue
            if b"\x00" in data[:8192]:
                continue
            encoding = _text_code_page(data[:_TEXT_SNIFF_BYTES])
            lines = data.decode(encoding, errors="replace").replace(
                "\r\n", "\n"
            ).replace("\r", "\n").split("\n")
            for line_number, line in enumerate(lines, start=1):
                if cancelled is not None and cancelled.is_set():
                    break
                if matcher.search(line) is None:
                    continue
                matches.append((candidate, line_number, line, lines))
                if len(matches) >= limit:
                    reached_limit = True
                    break
            if reached_limit:
                break
        if not matches:
            return "No matches found"

        output_lines: list[str] = []
        lines_truncated = False
        for file_path, line_number, matched_text, lines in matches:
            relative = _relative_search_path(file_path, search_root)
            if context == 0:
                value, truncated = _truncate_search_line(matched_text)
                lines_truncated |= truncated
                output_lines.append(f"{relative}:{line_number}: {value}")
                continue
            start = max(1, line_number - context)
            end = min(len(lines), line_number + context)
            for current in range(start, end + 1):
                value, truncated = _truncate_search_line(lines[current - 1])
                lines_truncated |= truncated
                separator = ":" if current == line_number else "-"
                output_lines.append(f"{relative}{separator}{current}{separator} {value}")

        output, bytes_truncated = _truncate_search_output(output_lines)
        notices = []
        if reached_limit:
            notices.append(
                f"{limit} matches limit reached. Use limit={limit * 2} for more, "
                "or refine pattern"
            )
        if lines_truncated:
            notices.append("long lines truncated to 500 characters")
        if bytes_truncated:
            notices.append("50KB limit reached")
        return output + (f"\n\n[{'. '.join(notices)}]" if notices else "")

    @staticmethod
    async def _start_search_process(
        executable: str,
        arguments: list[str],
        *,
        cwd: Path | None = None,
    ) -> asyncio.subprocess.Process:
        try:
            return await asyncio.create_subprocess_exec(
                executable,
                *arguments,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
        except OSError as exc:
            raise ToolExecutionError(
                "could not start search backend",
                kind="filesystem_error",
                code="search_backend_failed",
                retryable=True,
                side_effect_committed=False,
            ) from exc

    @staticmethod
    def _raise_search_failure(stderr: bytes, code: int, backend: str) -> None:
        message = stderr.decode("utf-8", errors="replace").strip()
        raise ToolExecutionError(
            message or f"{backend} exited with code {code}",
            kind="filesystem_error",
            code="search_backend_failed",
            retryable=True,
            side_effect_committed=False,
        )


__all__ = ["FileTools"]
