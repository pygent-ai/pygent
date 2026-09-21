from __future__ import annotations

import pytest

from pygent import ToolResultMedia, ToolResultText
from tests.live.multimodal_tool_result_probe import (
    _tool_message,
    az_catalog_media_cases,
)


def test_az_catalog_media_cases_cover_tool_capable_declared_modalities() -> None:
    cases = az_catalog_media_cases()

    assert len(cases) == 85
    assert len(cases) == len(set(cases))
    assert ("gpt-4.1", "image") in cases
    assert ("qwen3.8-max", "video") in cases
    assert ("gpt-4.1", "video") not in cases
    assert ("grok-imagine-image-2.0", "image") not in cases
    assert ("qvq-max", "video") not in cases


@pytest.mark.asyncio
@pytest.mark.parametrize("modality", ["image", "video"])
async def test_probe_read_tools_return_correlated_media_content(modality: str) -> None:
    call, message = await _tool_message(modality)

    result = message.results[0]
    expected_path = "blue-square.png" if modality == "image" else "red-then-blue.mp4"
    assert call.arguments.to_dict() == {"file_path": expected_path}
    assert result.call_id == call.call_id
    assert result.name == "read"
    assert result.status == "succeeded"
    assert isinstance(result.content[0], ToolResultText)
    media = result.content[1]
    assert isinstance(media, ToolResultMedia)
    assert media.media_type == modality
    assert media.source.kind == "inline"
    assert media.source.size_bytes is not None and media.source.size_bytes > 0
    assert media.source.sha256 is not None
