from __future__ import annotations

import asyncio
import base64
import json
import locale
import os
import random
import shutil
import struct
import subprocess
import sys
import threading
import time
import zlib
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image
from pypdf import PdfWriter

from pygent import (
    IdempotencyPolicy,
    MediaBlock,
    ToolKit,
    ToolResultText,
    ToolSideEffect,
)
from pygent.tool.standard import _files as file_module
from pygent.tool.standard._encoding import detect_text_encoding
from pygent.tool.standard._files import FileTools
from pygent.tool.standard._paths import normalize_tool_path

from ._helpers import invoke_tool, succeeded


def _to_msys_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    rest = resolved.as_posix()[3:]
    return f"/{drive}/{rest}"


def _parameters(handler) -> dict:
    return ToolKit(handler).definitions[0].parameters.to_dict()


def _encode_test_image(
    image: Image.Image,
    image_format: str,
    **options: object,
) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format=image_format, **options)
    return buffer.getvalue()


def _oversized_png_header(width: int, height: int) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IEND", b"")


def _write_blank_pdf(
    path: Path,
    page_count: int = 1,
    *,
    width: float = 72,
    height: float = 72,
) -> None:
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=width, height=height)
    with path.open("wb") as stream:
        writer.write(stream)


def _write_test_video(
    path: Path,
    *,
    size: str = "320x180",
    fps: int = 10,
    duration: float = 1,
    with_audio: bool = False,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is not installed")
    command = [
        ffmpeg,
        "-nostdin",
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={size}:rate={fps}",
    ]
    if with_audio:
        command.extend(("-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"))
    command.extend(
        (
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
        )
    )
    if with_audio:
        command.extend(("-c:a", "aac", "-ac", "2"))
    else:
        command.append("-an")
    command.append(str(path))
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        timeout=30,
    )
    if result.returncode != 0:
        pytest.skip("ffmpeg does not provide the libx264 test encoder")


def test_pdf_reader_uses_current_pypdf_backend(tmp_path) -> None:
    path = tmp_path / "blank.pdf"
    _write_blank_pdf(path)
    assert file_module._read_pdf_text(path, "1") == "--- page 1 ---"


def test_resolve_path_handles_relative_paths(tmp_path):
    assert (
        normalize_tool_path("notes.txt", str(tmp_path))
        == (tmp_path / "notes.txt").resolve()
    )
    # Desktop-like paths are not rewritten by a framework alias; they are
    # resolved as ordinary paths from the workspace base when relative.
    assert (
        normalize_tool_path("C:/Users/Desktop/report.txt", str(tmp_path))
        == (tmp_path / "C:/Users/Desktop/report.txt").resolve()
    )


def test_file_toolkit_exposes_only_lowercase_tool_names(tmp_path):
    tools = FileTools(workspace_root=tmp_path)
    definitions = ToolKit(*tools.handlers).definitions

    assert [item.name for item in definitions] == [
        "edit",
        "edit_notebook",
        "glob",
        "grep",
        "read",
        "read_lints",
        "write",
    ]
    assert not {
        "Edit",
        "Glob",
        "Read",
        "Write",
        "delete_file",
        "read_file",
        "search_replace",
    }.intersection(item.name for item in definitions)


@pytest.mark.parametrize(
    "option",
    [
        {"max_read_bytes": 0},
        {"max_media_bytes": 0},
        {"max_image_output_bytes": 0},
        {"max_image_edge": 0},
        {"max_image_pixels": 0},
        {"max_video_output_bytes": 0},
        {"max_video_duration_seconds": 0},
        {"max_video_edge": 0},
        {"max_video_fps": 0},
        {"max_search_files": 0},
    ],
)
def test_file_tools_require_positive_resource_limits(tmp_path, option) -> None:
    with pytest.raises(ValueError, match="limits must be positive"):
        FileTools(workspace_root=tmp_path, **option)


def test_read_schema_covers_text_pdf_and_media_options(tmp_path) -> None:
    tools = FileTools(workspace_root=tmp_path)

    definition = ToolKit(tools.read).definitions[0]
    schema = _parameters(tools.read)

    assert "rendered as images" in definition.description
    assert schema["required"] == ["file_path"]
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {"file_path", "limit", "offset", "pages"}
    assert schema["properties"]["file_path"]["type"] == "string"
    assert "rendered as images" in schema["properties"]["pages"]["description"]


@pytest.mark.asyncio
async def test_read_renders_requested_pdf_pages_as_inline_images(tmp_path) -> None:
    _write_blank_pdf(tmp_path / "document.pdf", page_count=3)
    tools = FileTools(workspace_root=tmp_path)

    result = await invoke_tool(
        tools.read,
        {"file_path": "document.pdf", "pages": "2-3"},
    )

    assert result.status == "succeeded", (result.error_code, result.error)
    assert result.output["file_path"] == "document.pdf"
    assert result.output["mime_type"] == "application/pdf"
    assert [page["page"] for page in result.output["rendered_pages"]] == [2, 3]
    assert result.output["total_size_bytes"] == sum(
        page["size_bytes"] for page in result.output["rendered_pages"]
    )
    assert len(result.content) == 4
    for content_index, page_number in ((0, 2), (2, 3)):
        label = result.content[content_index]
        media = result.content[content_index + 1]
        assert isinstance(label, ToolResultText)
        assert f"page {page_number}" in label.text
        assert isinstance(media, MediaBlock)
        assert media.media_type == "image"
        assert media.mime_type == "image/png"
        assert media.detail == "high"
        assert base64.b64decode(media.source.base64_data or "").startswith(
            b"\x89PNG\r\n\x1a\n"
        )


@pytest.mark.asyncio
async def test_rendered_pdf_pages_use_image_normalization_limits(tmp_path) -> None:
    _write_blank_pdf(tmp_path / "document.pdf", width=300, height=100)
    tools = FileTools(workspace_root=tmp_path, max_image_edge=128)

    result = await invoke_tool(
        tools.read,
        {"file_path": "document.pdf", "pages": "1"},
    )

    assert result.status == "succeeded"
    page = result.output["rendered_pages"][0]
    assert page["original_dimensions"] == (600, 200)
    assert page["delivered_dimensions"] == (128, 42)
    assert page["transformed"] is True
    assert page["transformations"] == ("resize", "reencode")


@pytest.mark.asyncio
async def test_read_pdf_without_pages_still_extracts_text(tmp_path) -> None:
    _write_blank_pdf(tmp_path / "document.pdf")
    tools = FileTools(workspace_root=tmp_path)

    result = await invoke_tool(tools.read, {"file_path": "document.pdf"})

    assert result.status == "succeeded"
    assert result.output == "--- page 1 ---"
    assert result.content == (ToolResultText("--- page 1 ---"),)


@pytest.mark.asyncio
async def test_rendered_pdf_pages_share_media_byte_limit(tmp_path) -> None:
    _write_blank_pdf(tmp_path / "document.pdf", page_count=2)
    tools = FileTools(workspace_root=tmp_path, max_media_bytes=8)

    result = await invoke_tool(
        tools.read,
        {"file_path": "document.pdf", "pages": "1-2"},
    )

    assert result.status == "failed"
    assert result.error_code == "media_too_large"


@pytest.mark.asyncio
async def test_rendered_pdf_page_rejects_excessive_pixel_dimensions(tmp_path) -> None:
    _write_blank_pdf(
        tmp_path / "document.pdf",
        width=100_000,
        height=100_000,
    )
    tools = FileTools(workspace_root=tmp_path)

    result = await invoke_tool(
        tools.read,
        {"file_path": "document.pdf", "pages": "1"},
    )

    assert result.status == "failed"
    assert result.error_code == "pdf_page_too_large"


@pytest.mark.asyncio
async def test_rendered_pdf_pages_reject_empty_page_selection(tmp_path) -> None:
    _write_blank_pdf(tmp_path / "document.pdf")
    tools = FileTools(workspace_root=tmp_path)

    result = await invoke_tool(
        tools.read,
        {"file_path": "document.pdf", "pages": ""},
    )

    assert result.status == "failed"
    assert result.error_code == "invalid_page_range"


@pytest.mark.asyncio
@pytest.mark.parametrize("option", [{"limit": 1}, {"offset": 1}])
async def test_rendered_pdf_pages_reject_text_range_options(tmp_path, option) -> None:
    _write_blank_pdf(tmp_path / "document.pdf")
    tools = FileTools(workspace_root=tmp_path)

    result = await invoke_tool(
        tools.read,
        {"file_path": "document.pdf", "pages": "1", **option},
    )

    assert result.status == "failed"
    assert result.error_code == "invalid_read_options"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "media_type", "mime_type"),
    [
        ("shape.png", "image", "image/png"),
        ("clip.mp4", "video", "video/mp4"),
    ],
)
async def test_read_returns_inline_structured_media_content(
    tmp_path, monkeypatch, name, media_type, mime_type
) -> None:
    if media_type == "image":
        image = Image.new("RGB", (32, 24), "blue")
        data = _encode_test_image(image, "PNG")
        image.close()
    else:
        monkeypatch.setattr(file_module, "_load_video_backend", lambda: None)
        data = b"\x00\x00\x00\x18ftypmp42fixture"
    (tmp_path / name).write_bytes(data)
    tools = FileTools(workspace_root=tmp_path)

    result = await invoke_tool(tools.read, {"file_path": name})

    assert result.status == "succeeded"
    assert result.output["file_path"] == name
    assert result.output["mime_type"] == mime_type
    assert result.output["size_bytes"] == len(data)
    assert isinstance(result.content[0], ToolResultText)
    media = result.content[1]
    assert isinstance(media, MediaBlock)
    assert media.media_type == media_type
    assert media.mime_type == mime_type
    assert media.source.kind == "inline"
    assert base64.b64decode(media.source.base64_data or "") == data
    if media_type == "image":
        assert result.output["original_dimensions"] == (32, 24)
        assert result.output["delivered_dimensions"] == (32, 24)
        assert result.output["transformed"] is False
        assert result.output["transformations"] == ()
    else:
        assert result.output["processing"] == "passthrough_unverified"
        assert result.output["original_video"] is None
        assert result.output["delivered_video"] is None
        assert "pygent-ai[video]" in result.output["processing_hint"]
        assert "ffprobe" in result.content[0].text


@pytest.mark.asyncio
async def test_read_video_without_ffmpeg_fails_only_when_normalization_is_required(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(file_module, "_load_video_backend", lambda: None)
    data = b"\x00\x00\x00\x18ftypmp42" + b"x" * 128
    (tmp_path / "large.mp4").write_bytes(data)
    tools = FileTools(
        workspace_root=tmp_path,
        max_media_bytes=256,
        max_video_output_bytes=64,
    )

    result = await invoke_tool(tools.read, {"file_path": "large.mp4"})

    assert result.status == "failed"
    assert result.error_code == "video_processing_unavailable"


@pytest.mark.asyncio
async def test_read_inspects_safe_video_without_changing_bytes(tmp_path) -> None:
    path = tmp_path / "safe.mp4"
    _write_test_video(path)
    original = path.read_bytes()
    tools = FileTools(workspace_root=tmp_path)

    result = await invoke_tool(tools.read, {"file_path": "safe.mp4"})

    assert result.status == "succeeded"
    assert result.output["processing"] == "passthrough"
    assert result.output["transformed"] is False
    assert result.output["original_video"]["dimensions"] == (320, 180)
    assert result.output["original_video"]["fps"] == 10
    assert result.output["delivered_video"] == result.output["original_video"]
    assert base64.b64decode(result.content[1].source.base64_data or "") == original
    assert (result.content[1].width, result.content[1].height) == (320, 180)
    assert result.content[1].duration_seconds == pytest.approx(1)
    assert result.content[1].fps == pytest.approx(10)
    assert result.content[1].has_audio is False


@pytest.mark.asyncio
async def test_read_normalizes_video_for_model_delivery(tmp_path) -> None:
    path = tmp_path / "source.mp4"
    _write_test_video(path, size="640x360", fps=30)
    tools = FileTools(
        workspace_root=tmp_path,
        max_video_edge=160,
        max_video_fps=8,
    )

    result = await invoke_tool(tools.read, {"file_path": "source.mp4"})

    assert result.status == "succeeded", (result.error_code, result.error)
    assert result.output["processing"] == "normalized"
    assert result.output["transformed"] is True
    assert result.output["transformations"] == ("resize", "frame_rate", "reencode")
    assert max(result.output["delivered_video"]["dimensions"]) <= 160
    assert result.output["delivered_video"]["fps"] <= 8
    assert result.output["delivered_video"]["video_codec"] == "h264"
    assert result.output["delivered_video"]["pixel_format"] == "yuv420p"
    assert result.output["size_bytes"] <= 12_000_000
    assert result.content[1].duration_seconds is not None
    assert result.content[1].fps is not None
    assert result.content[1].fps <= 8


@pytest.mark.asyncio
async def test_read_rejects_video_over_duration_limit(tmp_path) -> None:
    path = tmp_path / "long.mp4"
    _write_test_video(path, duration=1)
    tools = FileTools(workspace_root=tmp_path, max_video_duration_seconds=0.2)

    result = await invoke_tool(tools.read, {"file_path": "long.mp4"})

    assert result.status == "failed"
    assert result.error_code == "video_too_long"


@pytest.mark.asyncio
async def test_read_normalizes_video_audio_to_aac_mono(tmp_path) -> None:
    path = tmp_path / "stereo.mp4"
    _write_test_video(path, fps=20, with_audio=True)
    tools = FileTools(workspace_root=tmp_path, max_video_fps=10)

    result = await invoke_tool(tools.read, {"file_path": "stereo.mp4"})

    assert result.status == "succeeded", (result.error_code, result.error)
    assert result.output["processing"] == "normalized"
    assert "audio_codec" in result.output["transformations"]
    assert result.output["delivered_video"]["audio_codec"] == "aac"
    assert result.output["delivered_video"]["audio_channels"] == 1


@pytest.mark.asyncio
async def test_read_rejects_invalid_mp4_when_ffprobe_is_available(tmp_path) -> None:
    if file_module._load_video_backend() is None:
        pytest.skip("ffmpeg and ffprobe are not installed")
    (tmp_path / "broken.mp4").write_bytes(
        b"\x00\x00\x00\x18ftypmp42not-a-real-video"
    )
    tools = FileTools(workspace_root=tmp_path)

    result = await invoke_tool(tools.read, {"file_path": "broken.mp4"})

    assert result.status == "failed"
    assert result.error_code == "video_decode_failed"


@pytest.mark.asyncio
async def test_read_normalizes_large_image_for_model_delivery(tmp_path) -> None:
    image = Image.new("RGB", (3000, 1200), "navy")
    data = _encode_test_image(image, "JPEG", quality=95)
    image.close()
    (tmp_path / "large.jpg").write_bytes(data)
    tools = FileTools(workspace_root=tmp_path)

    result = await invoke_tool(tools.read, {"file_path": "large.jpg"})

    assert result.status == "succeeded"
    assert result.output["original_dimensions"] == (3000, 1200)
    assert result.output["delivered_dimensions"] == (2048, 819)
    assert result.output["transformed"] is True
    assert result.output["transformations"] == ("resize", "reencode")
    assert result.output["size_bytes"] <= 3_500_000
    delivered = base64.b64decode(result.content[1].source.base64_data or "")
    with Image.open(BytesIO(delivered)) as normalized:
        assert normalized.size == (2048, 819)


@pytest.mark.asyncio
async def test_read_reencodes_image_until_output_fits_byte_limit(tmp_path) -> None:
    random_bytes = random.Random(0).randbytes(512 * 512 * 3)
    image = Image.frombytes("RGB", (512, 512), random_bytes)
    data = _encode_test_image(image, "PNG")
    image.close()
    (tmp_path / "noise.png").write_bytes(data)
    tools = FileTools(
        workspace_root=tmp_path,
        max_image_output_bytes=20_000,
    )

    result = await invoke_tool(tools.read, {"file_path": "noise.png"})

    assert result.status == "succeeded"
    assert result.output["size_bytes"] <= 20_000
    assert result.output["transformed"] is True
    assert "resize" in result.output["transformations"]
    assert "reencode" in result.output["transformations"]


@pytest.mark.asyncio
async def test_read_fails_when_image_cannot_fit_output_limit(tmp_path) -> None:
    image = Image.new("RGB", (8, 8), "black")
    data = _encode_test_image(image, "PNG")
    image.close()
    (tmp_path / "tiny.png").write_bytes(data)
    tools = FileTools(workspace_root=tmp_path, max_image_output_bytes=1)

    result = await invoke_tool(tools.read, {"file_path": "tiny.png"})

    assert result.status == "failed"
    assert result.error_code == "image_output_too_large"


@pytest.mark.asyncio
async def test_read_applies_exif_orientation(tmp_path) -> None:
    image = Image.new("RGB", (10, 20), "red")
    exif = Image.Exif()
    exif[274] = 6
    data = _encode_test_image(image, "JPEG", exif=exif)
    image.close()
    (tmp_path / "rotated.jpg").write_bytes(data)
    tools = FileTools(workspace_root=tmp_path)

    result = await invoke_tool(tools.read, {"file_path": "rotated.jpg"})

    assert result.status == "succeeded"
    assert result.output["original_dimensions"] == (10, 20)
    assert result.output["delivered_dimensions"] == (20, 10)
    assert result.output["transformations"] == ("exif_orientation", "reencode")
    delivered = base64.b64decode(result.content[1].source.base64_data or "")
    with Image.open(BytesIO(delivered)) as normalized:
        assert normalized.getexif().get(274) is None


@pytest.mark.asyncio
async def test_read_preserves_transparency_when_resizing_png(tmp_path) -> None:
    image = Image.new("RGBA", (600, 300), (0, 128, 255, 64))
    data = _encode_test_image(image, "PNG")
    image.close()
    (tmp_path / "transparent.png").write_bytes(data)
    tools = FileTools(workspace_root=tmp_path, max_image_edge=256)

    result = await invoke_tool(tools.read, {"file_path": "transparent.png"})

    assert result.status == "succeeded"
    assert result.output["delivered_dimensions"] == (256, 128)
    delivered = base64.b64decode(result.content[1].source.base64_data or "")
    with Image.open(BytesIO(delivered)) as normalized:
        assert normalized.mode == "RGBA"
        assert normalized.getpixel((0, 0))[3] == 64


@pytest.mark.asyncio
async def test_read_rejects_image_pixel_bomb_before_decode(tmp_path) -> None:
    (tmp_path / "bomb.png").write_bytes(_oversized_png_header(10_000, 5_000))
    tools = FileTools(workspace_root=tmp_path)

    result = await invoke_tool(tools.read, {"file_path": "bomb.png"})

    assert result.status == "failed"
    assert result.error_code == "image_too_many_pixels"


@pytest.mark.asyncio
async def test_read_rejects_animated_gif(tmp_path) -> None:
    first = Image.new("RGB", (16, 16), "red")
    second = Image.new("RGB", (16, 16), "blue")
    data = _encode_test_image(
        first,
        "GIF",
        save_all=True,
        append_images=[second],
        duration=100,
        loop=0,
    )
    first.close()
    second.close()
    (tmp_path / "animated.gif").write_bytes(data)
    tools = FileTools(workspace_root=tmp_path)

    result = await invoke_tool(tools.read, {"file_path": "animated.gif"})

    assert result.status == "failed"
    assert result.error_code == "animated_image_unsupported"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "data", "error_code"),
    [
        ("empty.png", b"", "empty_media"),
        ("wrong.png", b"not a png", "media_mime_mismatch"),
        ("truncated.png", b"\x89PNG\r\n\x1a\ntruncated", "image_decode_failed"),
        ("wrong.mp4", b"not an mp4", "media_mime_mismatch"),
    ],
)
async def test_read_fails_clearly_for_invalid_media(
    tmp_path, name, data, error_code
) -> None:
    (tmp_path / name).write_bytes(data)
    tools = FileTools(workspace_root=tmp_path)

    result = await invoke_tool(tools.read, {"file_path": name})

    assert result.status == "failed"
    assert result.error_code == error_code
    assert result.side_effect_committed is False


@pytest.mark.asyncio
async def test_media_file_tool_enforces_configured_byte_limit(tmp_path) -> None:
    (tmp_path / "shape.png").write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    tools = FileTools(workspace_root=tmp_path, max_media_bytes=8)

    result = await invoke_tool(tools.read, {"file_path": "shape.png"})

    assert result.status == "failed"
    assert result.error_code == "media_too_large"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "data"),
    [
        ("outside-media.png", b"\x89PNG\r\n\x1a\nfixture"),
        ("outside-media.mp4", b"\x00\x00\x00\x18ftypmp42fixture"),
    ],
)
async def test_read_rejects_media_paths_outside_workspace(tmp_path, name, data) -> None:
    outside = tmp_path.parent / name
    outside.write_bytes(data)
    tools = FileTools(workspace_root=tmp_path)

    result = await invoke_tool(
        tools.read, {"file_path": str(outside)}
    )

    assert result.status == "failed"
    assert result.error_code == "path_outside_workspace"


@pytest.mark.asyncio
@pytest.mark.parametrize("option", [{"limit": 1}, {"offset": 1}])
async def test_read_rejects_text_ranges_for_media(tmp_path, option) -> None:
    (tmp_path / "shape.png").write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    tools = FileTools(workspace_root=tmp_path)

    result = await invoke_tool(
        tools.read,
        {"file_path": "shape.png", **option},
    )

    assert result.status == "failed"
    assert result.error_code == "invalid_read_options"


@pytest.mark.asyncio
async def test_write_read_edit_grep_flow(tmp_path):
    tools = FileTools(workspace_root=tmp_path)
    target = tmp_path / "docs" / "example.txt"

    assert (
        await succeeded(
            tools.write, file_path="docs/example.txt", content="alpha\nbeta\nalpha\n"
        )
        == "写入完成"
    )
    assert target.read_text(encoding="utf-8") == "alpha\nbeta\nalpha\n"

    assert (
        await succeeded(tools.read, file_path="docs/example.txt", offset=2, limit=1)
        == "2|beta\n"
    )

    await succeeded(
        tools.edit,
        file_path="docs/example.txt",
        old_string="alpha",
        new_string="gamma",
    )
    await succeeded(
        tools.edit,
        file_path="docs/example.txt",
        old_string="alpha",
        new_string="delta",
        replace_all=True,
    )
    assert target.read_text(encoding="utf-8") == "gamma\nbeta\ndelta\n"

    grep_output = await succeeded(
        tools.grep,
        pattern="DELTA",
        path="docs",
        ignoreCase=True,
    )
    assert grep_output == "example.txt:3: delta"


@pytest.mark.asyncio
async def test_file_tools_resolve_relative_paths_from_workspace_root(tmp_path):
    tools = FileTools(workspace_root=tmp_path)
    target = tmp_path / "docs" / "relative.txt"

    await succeeded(tools.write, file_path="docs/relative.txt", content="alpha\nbeta\n")
    assert (
        await succeeded(tools.read, file_path="docs/relative.txt", offset=2, limit=1)
        == "2|beta\n"
    )
    await succeeded(
        tools.edit,
        file_path="docs/relative.txt",
        old_string="beta",
        new_string="gamma",
    )
    assert target.read_text(encoding="utf-8") == "alpha\ngamma\n"


@pytest.mark.asyncio
async def test_file_tools_restrict_paths_to_workspace_by_default(tmp_path):
    outside = tmp_path.parent / "outside-pygent-path.txt"
    outside.write_text("secret\n", encoding="utf-8")
    tools = FileTools(workspace_root=tmp_path)

    result = await invoke_tool(tools.read, {"file_path": str(outside)})

    assert result.status == "failed"
    assert result.error_code == "path_outside_workspace"
    assert str(outside.resolve()) in (result.error or "")
    assert result.side_effect_committed is False


@pytest.mark.asyncio
async def test_glob_rejects_parent_traversal_outside_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("secret\n", encoding="utf-8")
    tools = FileTools(workspace_root=workspace)

    result = await invoke_tool(tools.glob, {"pattern": "../outside/*.txt"})

    assert result.status == "failed"
    assert result.error_code == "path_outside_workspace"
    assert result.side_effect_committed is False


@pytest.mark.asyncio
async def test_cancelled_file_write_finishes_before_cancellation_returns(tmp_path):
    tools = FileTools(workspace_root=tmp_path)
    target = tmp_path / "late.txt"
    started = threading.Event()

    def slow_write(_file_path: str, content: str) -> str:
        started.set()
        time.sleep(0.2)
        target.write_text(content, encoding="utf-8")
        return "done"

    tools._write = slow_write  # type: ignore[method-assign]
    invocation = asyncio.create_task(
        invoke_tool(tools.write, {"file_path": "ignored.txt", "content": "late"})
    )
    assert await asyncio.to_thread(started.wait, 1)

    invocation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await invocation

    assert target.read_text(encoding="utf-8") == "late"


@pytest.mark.asyncio
async def test_concurrent_edits_to_one_file_do_not_lose_updates(tmp_path, monkeypatch):
    target = tmp_path / "shared.txt"
    target.write_text("alpha beta\n", encoding="utf-8")
    tools = FileTools(workspace_root=tmp_path)
    original_atomic_write = file_module._atomic_write_text

    def slow_atomic_write(path: Path, content: str) -> None:
        time.sleep(0.05)
        original_atomic_write(path, content)

    monkeypatch.setattr(file_module, "_atomic_write_text", slow_atomic_write)

    first, second = await asyncio.gather(
        invoke_tool(
            tools.edit,
            {
                "file_path": "shared.txt",
                "old_string": "alpha",
                "new_string": "one",
            },
            call_id="edit-alpha",
        ),
        invoke_tool(
            tools.edit,
            {
                "file_path": "shared.txt",
                "old_string": "beta",
                "new_string": "two",
            },
            call_id="edit-beta",
        ),
    )

    assert first.status == second.status == "succeeded"
    assert target.read_text(encoding="utf-8") == "one two\n"


@pytest.mark.asyncio
async def test_file_tools_can_disable_workspace_restriction(tmp_path):
    outside = tmp_path.parent / "outside-pygent-unrestricted.txt"
    outside.write_text("alpha\n", encoding="utf-8")
    tools = FileTools(workspace_root=tmp_path, restrict_to_workspace=False)

    assert await succeeded(tools.read, file_path=str(outside)) == "1|alpha\n"


@pytest.mark.asyncio
async def test_file_tools_accept_git_bash_msys_paths_on_windows(tmp_path):
    if os.name != "nt":
        pytest.skip("MSYS drive path compatibility is Windows-specific")

    tools = FileTools(workspace_root=tmp_path)
    target = tmp_path / "docs" / "example.txt"
    msys_target = _to_msys_path(target)
    msys_root = _to_msys_path(tmp_path)

    await succeeded(tools.write, file_path=msys_target, content="alpha\nbeta\n")
    assert (
        await succeeded(tools.read, file_path=msys_target, offset=1, limit=1)
        == "1|alpha\n"
    )
    assert (
        await succeeded(tools.glob, pattern="**/*.txt", path=msys_root)
    ).splitlines() == ["docs/example.txt"]
    grep_output = await succeeded(tools.grep, pattern="beta", path=msys_root)
    assert grep_output == "docs/example.txt:2: beta"
    await succeeded(
        tools.edit,
        file_path=msys_target,
        old_string="beta",
        new_string="gamma",
    )
    assert target.read_text(encoding="utf-8") == "alpha\ngamma\n"


@pytest.mark.asyncio
async def test_file_tool_path_errors_are_structured_tool_results(tmp_path):
    tools = FileTools(workspace_root=tmp_path)
    missing = tmp_path / "missing.txt"

    result = await invoke_tool(tools.read, {"file_path": str(missing)})

    assert result.status == "failed"
    assert result.error_kind == "filesystem_error"
    assert result.error_code == "file_not_found"
    assert "missing.txt" in (result.error or "")
    assert result.output is None


@pytest.mark.asyncio
async def test_glob_grep_and_edit_path_errors_include_standard_codes(tmp_path):
    tools = FileTools(workspace_root=tmp_path)
    missing_dir = tmp_path / "missing-dir"
    missing_file = tmp_path / "missing.txt"

    glob_result = await invoke_tool(
        tools.glob, {"pattern": "*.md", "path": str(missing_dir)}
    )
    grep_result = await invoke_tool(
        tools.grep, {"pattern": "x", "path": str(missing_dir)}
    )
    edit_result = await invoke_tool(
        tools.edit,
        {
            "file_path": str(missing_file),
            "old_string": "x",
            "new_string": "y",
        },
    )

    assert glob_result.error_code == "file_not_found"
    assert grep_result.error_code == "file_not_found"
    assert edit_result.error_code == "file_not_found"
    assert edit_result.side_effect_committed is False
    assert not missing_file.exists()


@pytest.mark.asyncio
async def test_write_uses_workspace_path_schema_and_resolves_relative_paths(tmp_path):
    tools = FileTools(workspace_root=tmp_path)
    target = tmp_path / "strict" / "out.txt"

    await succeeded(tools.write, file_path="strict/out.txt", content="hello\n")
    assert target.read_text(encoding="utf-8") == "hello\n"

    toolkit = ToolKit(tools.write)
    definition = toolkit.definitions[0]
    parameters = definition.parameters.to_dict()
    assert parameters["additionalProperties"] is False
    assert parameters["required"] == ["file_path", "content"]
    assert set(parameters["properties"]) == {"file_path", "content"}
    assert "workspace_root" in parameters["properties"]["file_path"]["description"]
    assert toolkit.specs[0].version == "2.1.0"
    assert "\n\nUsage:\n" in definition.description
    assert "MUST use multiple smaller atomic edit calls" in definition.description
    assert "Do not use omission placeholders" in parameters["properties"]["content"][
        "description"
    ]


@pytest.mark.asyncio
async def test_read_uses_requested_schema_and_reads_text_ranges(tmp_path):
    tools = FileTools(workspace_root=tmp_path)
    target = tmp_path / "strict" / "input.txt"
    target.parent.mkdir()
    target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    assert (
        await succeeded(tools.read, file_path=str(target), offset=2, limit=1)
        == "2|beta\n"
    )
    assert (
        await succeeded(tools.read, file_path="strict/input.txt", offset=1, limit=1)
        == "1|alpha\n"
    )

    parameters = _parameters(tools.read)
    assert parameters["additionalProperties"] is False
    assert parameters["required"] == ["file_path"]
    assert set(parameters["properties"]) == {"file_path", "limit", "offset", "pages"}

    invalid = await invoke_tool(tools.read, {"file_path": str(target), "limit": 0})
    assert invalid.status == "rejected"
    assert invalid.error_kind == "validation_error"


@pytest.mark.asyncio
async def test_edit_uses_workspace_path_schema_and_exact_replacement(tmp_path):
    tools = FileTools(workspace_root=tmp_path)
    target = tmp_path / "strict" / "edit.txt"
    target.parent.mkdir()
    target.write_text("alpha\nbeta\nalpha\n", encoding="utf-8")

    await succeeded(
        tools.edit, file_path=str(target), old_string="alpha", new_string="gamma"
    )
    await succeeded(
        tools.edit,
        file_path=str(target),
        old_string="alpha",
        new_string="delta",
        replace_all=True,
    )
    await succeeded(
        tools.edit,
        file_path="strict/edit.txt",
        old_string="delta",
        new_string="epsilon",
    )
    assert target.read_text(encoding="utf-8") == "gamma\nbeta\nepsilon\n"

    same = await invoke_tool(
        tools.edit,
        {"file_path": str(target), "old_string": "same", "new_string": "same"},
    )
    assert same.error_code == "identical_replacement"

    toolkit = ToolKit(tools.edit)
    definition = toolkit.definitions[0]
    parameters = definition.parameters.to_dict()
    assert parameters["additionalProperties"] is False
    assert parameters["required"] == ["file_path", "old_string", "new_string"]
    assert set(parameters["properties"]) == {
        "file_path",
        "old_string",
        "new_string",
        "replace_all",
    }
    assert parameters["properties"]["replace_all"]["default"] is False
    assert toolkit.specs[0].version == "2.3.0"
    assert "\n\nUsage:\n" in definition.description
    assert "MUST split" in definition.description
    assert "smaller atomic edit calls" in definition.description
    assert "do not include long runs" in parameters["properties"]["old_string"][
        "description"
    ]


@pytest.mark.asyncio
async def test_edit_matches_anchors_across_line_ending_styles(tmp_path):
    tools = FileTools(workspace_root=tmp_path)
    target = tmp_path / "crlf.txt"
    target.write_bytes(b"alpha\r\nbeta\r\nalpha\r\n")

    # LF anchors match a CRLF file and the file keeps its CRLF style.
    await succeeded(
        tools.edit,
        file_path=str(target),
        old_string="alpha\nbeta",
        new_string="gamma\ndelta",
    )
    assert target.read_bytes() == b"gamma\r\ndelta\r\nalpha\r\n"

    repeated = tmp_path / "repeated.txt"
    repeated.write_bytes(b"a\r\nb\r\na\r\nb\r\n")
    await succeeded(
        tools.edit,
        file_path=str(repeated),
        old_string="a\nb",
        new_string="x\ny",
        replace_all=True,
    )
    assert repeated.read_bytes() == b"x\r\ny\r\nx\r\ny\r\n"

    # A match opening on a folded CRLF consumes that CR instead of orphaning it.
    boundary = tmp_path / "boundary.txt"
    boundary.write_bytes(b"head\r\ntail\r\nmore\r\n")
    await succeeded(
        tools.edit,
        file_path=str(boundary),
        old_string="\ntail\nmore",
        new_string="\nHEAD\nMORE",
    )
    assert boundary.read_bytes() == b"head\r\nHEAD\r\nMORE\r\n"

    # CRLF anchors also match LF files; the changed endings land verbatim.
    source = tmp_path / "lf.txt"
    source.write_bytes(b"one\ntwo\n")
    await succeeded(
        tools.edit,
        file_path=str(source),
        old_string="one\r\ntwo",
        new_string="three\nfour",
    )
    assert source.read_bytes() == b"three\nfour\n"

    # Byte-exact anchors with a consistent ending sequence adopt the region style.
    exact = tmp_path / "exact.txt"
    exact.write_bytes(b"keep\r\nme\r\n")
    await succeeded(
        tools.edit,
        file_path=str(exact),
        old_string="keep\r\nme",
        new_string="kept\r\nyou",
    )
    assert exact.read_bytes() == b"kept\r\nyou\r\n"

    missing = await invoke_tool(
        tools.edit,
        {
            "file_path": str(target),
            "old_string": "absent\nanchor",
            "new_string": "x",
        },
    )
    assert missing.error_code == "match_not_found"


@pytest.mark.asyncio
async def test_edit_replacement_line_endings_are_unified_and_reported(tmp_path):
    tools = FileTools(workspace_root=tmp_path)

    # Exact single-line anchors follow the same EOL adoption as tolerant ones.
    target = tmp_path / "crlf.txt"
    target.write_bytes(b"A\r\nB\r\nC")
    report = await succeeded(
        tools.edit, file_path=str(target), old_string="B", new_string="X\nY"
    )
    assert target.read_bytes() == b"A\r\nX\r\nY\r\nC"
    assert report == "替换完成(1 处,新文本行尾已适配为 CRLF)"

    # An anchor carrying CRLF with a changed ending sequence lands verbatim.
    explicit = tmp_path / "explicit.txt"
    explicit.write_bytes(b"keep\r\nme\r\n")
    report = await succeeded(
        tools.edit,
        file_path=str(explicit),
        old_string="keep\r\nme",
        new_string="keep\nme",
    )
    assert explicit.read_bytes() == b"keep\nme\r\n"
    assert report == "替换完成(1 处,行尾按 new_string 原样写入)"

    # A byte-exact anchor with CRLF and an all-LF replacement is also explicit.
    rewritten = tmp_path / "rewritten.txt"
    rewritten.write_bytes(b"A\nB\r\nC\r\nD")
    await succeeded(
        tools.edit,
        file_path=str(rewritten),
        old_string="A\nB\r\nC",
        new_string="X\nY\nZ",
    )
    assert rewritten.read_bytes() == b"X\nY\nZ\r\nD"

    # CRLF introduced by new_string is deliberate and lands verbatim.
    inserted = tmp_path / "insert.txt"
    inserted.write_bytes(b"A\nB\n")
    await succeeded(
        tools.edit,
        file_path=str(inserted),
        old_string="B",
        new_string="X\r\nY",
    )
    assert inserted.read_bytes() == b"A\nX\r\nY\n"

    # Mixed regions follow the majority style with a document-level fallback.
    mixed = tmp_path / "mixed.txt"
    mixed.write_bytes(b"A\r\nB\nC\nD\r\nE")
    await succeeded(
        tools.edit,
        file_path=str(mixed),
        old_string="B\nC\nD\n",
        new_string="X\nY\nZ\n",
    )
    assert mixed.read_bytes() == b"A\r\nX\nY\nZ\nE"

    # Region style ties fall back to the document's dominant style.
    tie = tmp_path / "tie.txt"
    tie.write_bytes(b"A\r\nq\r\nr\nB")
    await succeeded(
        tools.edit,
        file_path=str(tie),
        old_string="q\nr\n",
        new_string="s\nt\n",
    )
    assert tie.read_bytes() == b"A\r\ns\r\nt\r\nB"

    # replace_all reports every style applied across differently-styled regions.
    repeated = tmp_path / "repeated.txt"
    repeated.write_bytes(b"p\r\nq\np\nq\n")
    report = await succeeded(
        tools.edit,
        file_path=str(repeated),
        old_string="p\nq",
        new_string="x\ny",
        replace_all=True,
    )
    assert repeated.read_bytes() == b"x\r\ny\nx\ny\n"
    assert report == "替换完成(2 处,新文本行尾已按命中区域适配为 CRLF/LF)"


@pytest.mark.asyncio
async def test_glob_finds_relative_files_and_exposes_pi_schema(tmp_path):
    tools = FileTools(workspace_root=tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    old_match = tmp_path / "src" / "old.py"
    new_match = tmp_path / "tests" / "new.py"
    non_match = tmp_path / "src" / "note.md"
    old_match.write_text("old", encoding="utf-8")
    new_match.write_text("new", encoding="utf-8")
    non_match.write_text("note", encoding="utf-8")
    os.utime(old_match, (100, 100))
    os.utime(new_match, (200, 200))

    assert set((await succeeded(tools.glob, pattern="**/*.py")).splitlines()) == {
        "src/old.py",
        "tests/new.py",
    }
    assert (await succeeded(tools.glob, pattern="*.py", path="src")).splitlines() == [
        "old.py"
    ]
    not_directory = await invoke_tool(
        tools.glob, {"pattern": "*.py", "path": "src/old.py"}
    )
    assert not_directory.error_code == "not_a_directory"

    parameters = _parameters(tools.glob)
    assert parameters["required"] == ["pattern"]
    assert parameters["additionalProperties"] is False
    assert set(parameters["properties"]) == {"pattern", "path", "limit"}
    assert parameters["properties"]["path"]["type"] == "string"
    assert parameters["properties"]["limit"]["default"] == 1000


def test_grep_tool_exposes_pi_input_schema(tmp_path):
    tools = FileTools(workspace_root=tmp_path)
    parameters = _parameters(tools.grep)
    properties = parameters["properties"]

    assert parameters["additionalProperties"] is False
    assert set(properties) == {
        "pattern",
        "path",
        "glob",
        "ignoreCase",
        "literal",
        "context",
        "limit",
    }
    assert parameters["required"] == ["pattern"]
    assert properties["path"]["type"] == "string"
    assert properties["glob"]["type"] == "string"
    assert properties["ignoreCase"]["default"] is False
    assert properties["literal"]["default"] is False
    assert properties["context"]["default"] == 0
    assert properties["limit"]["default"] == 100


@pytest.mark.asyncio
async def test_grep_pi_arguments_work_through_tool_call_layer(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.py").write_text("before\nAlpha\nAfter\n", encoding="utf-8")
    (docs / "b.rs").write_text("Alpha rust\n", encoding="utf-8")
    (docs / "c.txt").write_text("Alpha text\n", encoding="utf-8")
    tools = FileTools(workspace_root=tmp_path)

    output = await succeeded(
        tools.grep,
        pattern="alpha",
        path="docs",
        ignoreCase=True,
        context=1,
        glob="*.py",
    )
    assert "a.py-1- before" in output
    assert "a.py:2: Alpha" in output
    assert "a.py-3- After" in output
    assert "b.rs" not in output

    default_output = await succeeded(tools.grep, pattern="Alpha", path="docs")
    assert "a.py:2: Alpha" in default_output
    assert "b.rs:1: Alpha rust" in default_output

    brace_glob = await succeeded(
        tools.grep, pattern="Alpha", path="docs", glob="*.{py,rs}"
    )
    assert "a.py" in brace_glob
    assert "b.rs" in brace_glob
    assert "c.txt" not in brace_glob


@pytest.mark.asyncio
async def test_grep_literal_limit_and_no_match_messages(tmp_path):
    (tmp_path / "notes.txt").write_text("a.b\naxb\na.b\n", encoding="utf-8")
    tools = FileTools(workspace_root=tmp_path)

    output = await succeeded(
        tools.grep,
        pattern="a.b",
        path="notes.txt",
        literal=True,
        limit=1,
    )
    assert output.startswith("notes.txt:1: a.b")
    assert "1 matches limit reached" in output
    assert await succeeded(tools.grep, pattern="missing") == "No matches found"


@pytest.mark.asyncio
async def test_glob_and_grep_respect_gitignore_but_include_unignored_hidden_files(
    tmp_path,
):
    (tmp_path / ".gitignore").write_text(".venv/\n", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "ignored.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "visible.py").write_text("needle\n", encoding="utf-8")
    tools = FileTools(workspace_root=tmp_path)

    glob_output = await succeeded(tools.glob, pattern="**/*.py")
    grep_output = await succeeded(tools.grep, pattern="needle", glob="*.py")

    assert ".hidden/visible.py" in glob_output
    assert ".venv/ignored.py" not in glob_output
    assert ".hidden/visible.py:1: needle" in grep_output
    assert ".venv/ignored.py" not in grep_output


@pytest.mark.asyncio
async def test_glob_supports_directory_prefixed_patterns_and_limits_results(tmp_path):
    target = tmp_path / "src" / "nested" / "example.spec.py"
    target.parent.mkdir(parents=True)
    target.write_text("pass\n", encoding="utf-8")
    (tmp_path / "src" / "direct.spec.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("pass\n", encoding="utf-8")
    tools = FileTools(workspace_root=tmp_path)

    assert set(
        (await succeeded(tools.glob, pattern="src/**/*.spec.py")).splitlines()
    ) == {"src/direct.spec.py", "src/nested/example.spec.py"}
    assert await succeeded(tools.glob, pattern="src/*.spec.py") == (
        "src/direct.spec.py"
    )
    assert "other.py" in await succeeded(tools.glob, pattern="**/*.py")
    limited = await succeeded(tools.glob, pattern="**/*.py", limit=1)
    assert "1 results limit reached" in limited


@pytest.mark.asyncio
async def test_nested_gitignore_rules_do_not_leak_into_siblings(tmp_path):
    for directory in (tmp_path / "a", tmp_path / "b"):
        directory.mkdir()
        (directory / "ignored.txt").write_text("needle\n", encoding="utf-8")
        (directory / "kept.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / "a" / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    tools = FileTools(workspace_root=tmp_path)

    paths = set((await succeeded(tools.glob, pattern="**/*.txt")).splitlines())
    grep_output = await succeeded(tools.grep, pattern="needle", glob="*.txt")

    assert paths == {"a/kept.txt", "b/ignored.txt", "b/kept.txt"}
    assert "a/ignored.txt" not in grep_output
    assert "b/ignored.txt:1: needle" in grep_output


@pytest.mark.asyncio
async def test_glob_and_grep_fall_back_without_external_search_binaries(
    tmp_path, monkeypatch
):
    (tmp_path / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    (tmp_path / "ignored").mkdir()
    (tmp_path / "ignored" / "hidden.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "visible.py").write_text("Needle\n", encoding="utf-8")
    monkeypatch.setattr(file_module, "_search_executable", lambda *_names: None)
    tools = FileTools(workspace_root=tmp_path)

    assert await succeeded(tools.glob, pattern="**/*.py") == "src/visible.py"
    assert (
        await succeeded(tools.grep, pattern="needle", ignoreCase=True, glob="*.py")
        == "src/visible.py:1: Needle"
    )


@pytest.mark.asyncio
async def test_grep_invalid_regex_is_a_failed_tool_result(tmp_path):
    (tmp_path / "example.txt").write_text("text\n", encoding="utf-8")
    tools = FileTools(workspace_root=tmp_path)

    result = await invoke_tool(tools.grep, {"pattern": "["})

    assert result.status == "failed"
    assert result.error_code == "search_backend_failed"


@pytest.mark.asyncio
async def test_cancelled_grep_terminates_its_owned_process(tmp_path, monkeypatch):
    if file_module._search_executable("rg") is None:
        pytest.skip("ripgrep is required to exercise subprocess cancellation")
    (tmp_path / "example.txt").write_text("text\n", encoding="utf-8")
    started: list[asyncio.subprocess.Process] = []

    async def start_slow_process(*_args, **_kwargs):
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        started.append(process)
        return process

    monkeypatch.setattr(
        FileTools, "_start_search_process", staticmethod(start_slow_process)
    )
    tools = FileTools(workspace_root=tmp_path)
    invocation = asyncio.create_task(tools.grep("text"))
    for _ in range(100):
        if started:
            break
        await asyncio.sleep(0.01)
    assert started

    invocation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await invocation

    assert started[0].returncode is not None


@pytest.mark.asyncio
async def test_search_process_cleanup_drains_full_stdout_pipe():
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        (
            "import os, time\n"
            "chunk = b'x' * 65536\n"
            "for _ in range(64):\n"
            "    os.write(1, chunk)\n"
            "time.sleep(30)\n"
        ),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stderr is not None
    stderr_task = asyncio.create_task(process.stderr.read())
    await asyncio.sleep(0.1)

    await asyncio.wait_for(
        file_module._terminate_search_process(process, stderr_task), timeout=2
    )

    assert process.returncode is not None
    assert stderr_task.done()


@pytest.mark.asyncio
async def test_grep_result_limit_drains_buffered_backend_output(tmp_path, monkeypatch):
    (tmp_path / "example.txt").write_text("text\n", encoding="utf-8")
    monkeypatch.setattr(file_module, "_search_executable", lambda *_names: "rg")
    event = json.dumps(
        {
            "type": "match",
            "data": {
                "path": {"text": "example.txt"},
                "line_number": 1,
                "lines": {"text": "text\n"},
            },
        }
    )
    started: list[asyncio.subprocess.Process] = []

    async def start_noisy_process(*_args, **_kwargs):
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            (
                "import sys\n"
                f"line = {event + chr(10)!r}\n"
                "while True:\n"
                "    sys.stdout.write(line)\n"
                "    sys.stdout.flush()\n"
            ),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        started.append(process)
        await asyncio.sleep(0.1)
        return process

    monkeypatch.setattr(
        FileTools, "_start_search_process", staticmethod(start_noisy_process)
    )
    tools = FileTools(workspace_root=tmp_path)

    output = await asyncio.wait_for(tools.grep("text", limit=1), timeout=2)

    assert output.startswith("example.txt:1: text")
    assert "1 matches limit reached" in output
    assert started[0].returncode is not None


@pytest.mark.asyncio
async def test_cancelled_grep_fallback_joins_its_worker_thread(tmp_path, monkeypatch):
    (tmp_path / "example.txt").write_text("text\n", encoding="utf-8")
    monkeypatch.setattr(file_module, "_search_executable", lambda *_names: None)
    started = threading.Event()
    stopped = threading.Event()
    tools = FileTools(workspace_root=tmp_path)

    def slow_fallback(*args):
        cancelled = args[-1]
        started.set()
        cancelled.wait(30)
        stopped.set()
        return "No matches found"

    monkeypatch.setattr(tools, "_grep_fallback", slow_fallback)
    invocation = asyncio.create_task(tools.grep("text"))
    assert await asyncio.to_thread(started.wait, 1)

    invocation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(invocation, timeout=1)

    assert stopped.is_set()


@pytest.mark.asyncio
async def test_file_tool_errors_are_non_throwing_tool_results(tmp_path):
    tools = FileTools(workspace_root=tmp_path)

    read = await invoke_tool(tools.read, {"file_path": "missing.txt"})
    edit = await invoke_tool(
        tools.edit,
        {"file_path": "missing.txt", "old_string": "a", "new_string": "b"},
    )
    grep = await invoke_tool(tools.grep, {"pattern": "x", "path": "missing"})

    assert read.error_code == "file_not_found"
    assert edit.error_code == "file_not_found"
    assert grep.error_code == "file_not_found"


@pytest.mark.asyncio
async def test_read_describes_unknown_binary_files(tmp_path):
    tools = FileTools(workspace_root=tmp_path)
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01\x02")

    assert "blob.bin" in await succeeded(tools.read, file_path="blob.bin")


@pytest.mark.asyncio
async def test_edit_notebook_insert_replace_and_bounds(tmp_path):
    notebook = tmp_path / "nb.ipynb"
    notebook.write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": "markdown", "metadata": {}, "source": ["old text\n"]}
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )
    tools = FileTools(workspace_root=tmp_path)

    await succeeded(
        tools.edit_notebook,
        target_notebook="nb.ipynb",
        cell_idx=0,
        is_new_cell=False,
        cell_language="markdown",
        old_string="old",
        new_string="new",
    )
    data = json.loads(notebook.read_text(encoding="utf-8"))
    assert data["cells"][0]["source"] == ["new text\n"]

    await succeeded(
        tools.edit_notebook,
        target_notebook="nb.ipynb",
        cell_idx=1,
        is_new_cell=True,
        cell_language="python",
        old_string="",
        new_string="print('ok')\n",
    )
    data = json.loads(notebook.read_text(encoding="utf-8"))
    assert data["cells"][1]["cell_type"] == "code"
    assert data["cells"][1]["source"] == ["print('ok')\n"]

    bounds = await invoke_tool(
        tools.edit_notebook,
        {
            "target_notebook": "nb.ipynb",
            "cell_idx": 99,
            "is_new_cell": False,
            "cell_language": "markdown",
            "old_string": "x",
            "new_string": "y",
        },
    )
    assert bounds.error_code == "cell_index_out_of_range"
    assert "0..1" in (bounds.error or "")


@pytest.mark.asyncio
async def test_read_lints_reports_python_syntax_diagnostics(tmp_path):
    (tmp_path / "good.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "bad.py").write_text("if True print('bad')\n", encoding="utf-8")
    tools = FileTools(workspace_root=tmp_path)

    output = await succeeded(tools.read_lints, paths=["good.py", "bad.py"])

    diagnostics = json.loads(output)
    assert diagnostics["tool"] == "python.compile"
    assert len(diagnostics["diagnostics"]) == 1
    assert diagnostics["diagnostics"][0]["path"].endswith("bad.py")
    assert diagnostics["diagnostics"][0]["line"] == 1


def test_file_tools_publish_explicit_02_side_effect_and_permission_policies(tmp_path):
    tools = FileTools(workspace_root=tmp_path)
    specs = {item.definition.name: item for item in ToolKit(*tools.handlers).specs}

    assert specs["read"].side_effect is ToolSideEffect.READ
    assert specs["glob"].side_effect is ToolSideEffect.READ
    assert specs["grep"].side_effect is ToolSideEffect.READ
    assert specs["glob"].version == "3.0.0"
    assert specs["grep"].version == "3.0.0"
    assert specs["read_lints"].side_effect is ToolSideEffect.READ
    assert specs["write"].side_effect is ToolSideEffect.WRITE
    assert specs["edit"].side_effect is ToolSideEffect.WRITE
    assert specs["edit_notebook"].side_effect is ToolSideEffect.WRITE
    assert specs["write"].idempotency is IdempotencyPolicy.INHERENT
    assert specs["edit"].idempotency is IdempotencyPolicy.NOT_IDEMPOTENT
    assert specs["edit_notebook"].idempotency is IdempotencyPolicy.NOT_IDEMPOTENT
    assert specs["read"].required_permissions == ("filesystem:read",)
    assert specs["write"].required_permissions == ("filesystem:write",)
    assert all(item.sandbox_profile == "workspace" for item in specs.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("newline", [b"\n", b"\r\n", b"\r"])
async def test_read_offset_reaches_beyond_default_byte_cap(tmp_path, newline):
    tools = FileTools(workspace_root=tmp_path)
    # The skipped line itself exceeds the byte cap and must be scanned in chunks.
    (tmp_path / "large.txt").write_bytes(
        b"x" * (tools.max_read_bytes + 100) + newline + b"target" + newline + b"last"
    )
    assert await succeeded(
        tools.read, file_path="large.txt", offset=2, limit=1
    ) == "2|target\n"
    assert await succeeded(
        tools.read, file_path="large.txt", offset=3, limit=1
    ) == "3|last"
    assert await succeeded(tools.read, file_path="large.txt", offset=4) == ""


@pytest.mark.asyncio
async def test_read_byte_cap_returns_complete_lines_and_next_offset(tmp_path):
    tools = FileTools(workspace_root=tmp_path, max_read_bytes=10)
    (tmp_path / "pages.txt").write_bytes("一二\n三四\n五六\n".encode())
    first = await succeeded(tools.read, file_path="pages.txt")
    assert first.startswith("1|一二\n")
    assert "2|" not in first
    assert "continue with offset=2" in first
    second = await succeeded(tools.read, file_path="pages.txt", offset=2)
    assert second.startswith("2|三四\n")
    assert "continue with offset=3" in second
    assert await succeeded(tools.read, file_path="pages.txt", offset=3) == "3|五六\n"
    assert await succeeded(tools.read, file_path="pages.txt", limit=1) == "1|一二\n"


@pytest.mark.asyncio
async def test_read_oversized_line_explains_partial_line_without_broken_utf8(tmp_path):
    tools = FileTools(workspace_root=tmp_path, max_read_bytes=5)
    (tmp_path / "long.txt").write_bytes("中文中文\nend\n".encode())
    output = await succeeded(tools.read, file_path="long.txt")
    assert output.startswith("1|中\n[read truncated")
    assert "within line 1" in output
    assert "offset cannot retrieve the remainder" in output
    assert "\ufffd" not in output
    assert await succeeded(tools.read, file_path="long.txt", offset=2) == "2|end\n"


@pytest.mark.asyncio
async def test_read_exact_byte_cap_does_not_report_false_truncation(tmp_path):
    tools = FileTools(workspace_root=tmp_path, max_read_bytes=4)
    (tmp_path / "exact.txt").write_bytes(b"abc\n")
    assert await succeeded(tools.read, file_path="exact.txt") == "1|abc\n"


def test_detect_text_encoding_prefers_multibyte_codecs_and_rejects_binary(
    monkeypatch,
):
    monkeypatch.setattr(locale, "getpreferredencoding", lambda _do_setlocale: "cp1252")

    assert detect_text_encoding(b"") == "utf-8-sig"
    assert detect_text_encoding("中文\n".encode()) == "utf-8-sig"
    assert detect_text_encoding("中文测试".encode("gbk")[:5]) == "gb18030"
    assert detect_text_encoding("café\n".encode("cp1252")) == "cp1252"
    assert detect_text_encoding("中文\n".encode("utf-16")) == "utf-16"
    assert detect_text_encoding(b"blob\x00\x01") is None


@pytest.mark.asyncio
async def test_read_decodes_cp936_chinese_text(tmp_path, monkeypatch):
    monkeypatch.setattr(locale, "getpreferredencoding", lambda _do_setlocale: "cp1252")
    (tmp_path / "gbk.txt").write_bytes("中文测试\n第二行\n".encode("gbk"))
    tools = FileTools(workspace_root=tmp_path)

    output = await succeeded(tools.read, file_path="gbk.txt")

    assert output == "1|中文测试\n2|第二行\n"
    assert "\ufffd" not in output


@pytest.mark.asyncio
async def test_read_decodes_cp936_when_the_sample_splits_a_character(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(locale, "getpreferredencoding", lambda _do_setlocale: "cp1252")
    (tmp_path / "gbk.txt").write_bytes("中\n".encode("gbk") + "文".encode("gbk") * 20)
    tools = FileTools(workspace_root=tmp_path, max_read_bytes=5)

    output = await succeeded(tools.read, file_path="gbk.txt")

    assert output.startswith("1|中\n")
    assert "continue with offset=2" in output
    assert "\ufffd" not in output


@pytest.mark.asyncio
async def test_read_decodes_utf16_text_with_bom(tmp_path):
    (tmp_path / "unicode.txt").write_bytes("中文测试\n".encode("utf-16"))
    tools = FileTools(workspace_root=tmp_path)

    assert await succeeded(tools.read, file_path="unicode.txt") == "1|中文测试\n"


@pytest.mark.asyncio
async def test_read_does_not_leak_utf8_bom_into_content(tmp_path):
    (tmp_path / "bom.txt").write_bytes(b"\xef\xbb\xbf" + "中文\n".encode())
    tools = FileTools(workspace_root=tmp_path)

    assert await succeeded(tools.read, file_path="bom.txt") == "1|中文\n"


@pytest.mark.asyncio
async def test_grep_matches_non_ascii_pattern_in_cp936_text(tmp_path, monkeypatch):
    monkeypatch.setattr(locale, "getpreferredencoding", lambda _do_setlocale: "cp1252")
    (tmp_path / "gbk.txt").write_bytes("第一行\n第二行中文说明\n".encode("gbk"))
    tools = FileTools(workspace_root=tmp_path)

    assert await succeeded(tools.grep, pattern="中文说明", path="gbk.txt") == (
        "gbk.txt:2: 第二行中文说明"
    )


@pytest.mark.asyncio
async def test_grep_fallback_matches_non_ascii_pattern_in_cp936_text(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(locale, "getpreferredencoding", lambda _do_setlocale: "cp1252")
    monkeypatch.setattr(file_module, "_search_executable", lambda *_names: None)
    (tmp_path / "gbk.txt").write_bytes("第一行\n第二行中文说明\n".encode("gbk"))
    tools = FileTools(workspace_root=tmp_path)

    assert await succeeded(tools.grep, pattern="中文说明", path="gbk.txt") == (
        "gbk.txt:2: 第二行中文说明"
    )


@pytest.mark.asyncio
async def test_grep_covers_utf8_and_cp936_files_in_one_search(tmp_path, monkeypatch):
    monkeypatch.setattr(locale, "getpreferredencoding", lambda _do_setlocale: "cp1252")
    (tmp_path / "utf8.txt").write_text("中文说明\n", encoding="utf-8")
    (tmp_path / "gbk.txt").write_bytes("中文说明\n".encode("gbk"))
    tools = FileTools(workspace_root=tmp_path)

    output = await succeeded(tools.grep, pattern="中文说明", path=".")

    assert set(output.splitlines()) == {
        "utf8.txt:1: 中文说明",
        "gbk.txt:1: 中文说明",
    }


@pytest.mark.asyncio
async def test_grep_decodes_cp936_line_text_for_ascii_pattern(tmp_path, monkeypatch):
    monkeypatch.setattr(locale, "getpreferredencoding", lambda _do_setlocale: "cp1252")
    # "一" is cp936 D2BB, which is also valid UTF-8, so the matched line alone
    # does not reveal the code page of the file it belongs to.
    (tmp_path / "gbk.txt").write_bytes("value = 一\n第二行中文\n".encode("gbk"))
    tools = FileTools(workspace_root=tmp_path)

    output = await succeeded(tools.grep, pattern="value", path="gbk.txt")

    assert output == "gbk.txt:1: value = 一"
    assert "\ufffd" not in output


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback", [False, True])
async def test_grep_context_lines_decode_cp936_text(tmp_path, monkeypatch, fallback):
    monkeypatch.setattr(locale, "getpreferredencoding", lambda _do_setlocale: "cp1252")
    if fallback:
        monkeypatch.setattr(file_module, "_search_executable", lambda *_names: None)
    (tmp_path / "gbk.txt").write_bytes("第一行\n第二行中文说明\n".encode("gbk"))
    tools = FileTools(workspace_root=tmp_path)

    output = await succeeded(
        tools.grep, pattern="中文说明", path="gbk.txt", context=1
    )

    assert output == "gbk.txt-1- 第一行\ngbk.txt:2: 第二行中文说明\ngbk.txt-3- "
    assert "\ufffd" not in output


@pytest.mark.asyncio
async def test_edit_reads_cp936_text_and_rewrites_utf8(tmp_path, monkeypatch):
    monkeypatch.setattr(locale, "getpreferredencoding", lambda _do_setlocale: "cp1252")
    target = tmp_path / "notes.txt"
    target.write_bytes("第一行\n第二行中文\n".encode("gbk"))
    tools = FileTools(workspace_root=tmp_path)

    assert (
        await succeeded(
            tools.edit,
            file_path="notes.txt",
            old_string="第二行中文",
            new_string="第二行中文改好",
        )
        == "替换完成(1 处,新文本行尾已适配为 LF)"
    )
    assert target.read_text(encoding="utf-8") == "第一行\n第二行中文改好\n"


@pytest.mark.asyncio
async def test_edit_rejects_undecodable_text_file(tmp_path):
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01\x02blob")
    tools = FileTools(workspace_root=tmp_path)

    result = await invoke_tool(
        tools.edit,
        {"file_path": "blob.bin", "old_string": "a", "new_string": "b"},
    )

    assert result.status == "failed"
    assert result.error_code == "unsupported_text_encoding"


@pytest.mark.asyncio
async def test_edit_notebook_reads_cp936_notebook(tmp_path, monkeypatch):
    monkeypatch.setattr(locale, "getpreferredencoding", lambda _do_setlocale: "cp1252")
    notebook = tmp_path / "nb.ipynb"
    notebook.write_bytes(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": ["中文说明\n"],
                    }
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            },
            ensure_ascii=False,
        ).encode("gbk")
    )
    tools = FileTools(workspace_root=tmp_path)

    await succeeded(
        tools.edit_notebook,
        target_notebook="nb.ipynb",
        cell_idx=0,
        is_new_cell=False,
        cell_language="markdown",
        old_string="中文说明",
        new_string="中文已改",
    )

    data = json.loads(notebook.read_text(encoding="utf-8"))
    assert data["cells"][0]["source"] == ["中文已改\n"]


@pytest.mark.asyncio
async def test_read_lints_decodes_cp936_source(tmp_path, monkeypatch):
    monkeypatch.setattr(locale, "getpreferredencoding", lambda _do_setlocale: "cp1252")
    (tmp_path / "chinese.py").write_bytes("中文变量 = 1\n".encode("gbk"))
    tools = FileTools(workspace_root=tmp_path)

    diagnostics = json.loads(await succeeded(tools.read_lints, paths=["chinese.py"]))

    assert diagnostics == {"tool": "python.compile", "diagnostics": []}
