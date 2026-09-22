"""Deterministic media-input token rules for context-window admission.

Provider adapters select a rule from this module. Keeping formulas here makes
provider documentation changes local and leaves routing/admission code stable.
"""

from __future__ import annotations

import math
import re

from pygent.core._tool_values import _media_block_dimensions
from pygent.tool import MediaBlock

_UNKNOWN_IMAGE_TOKEN_BUDGET = 36_000
_UNKNOWN_VIDEO_TOKEN_BUDGET = 2_000_000
_GENERIC_VIDEO_TOKENS_PER_SECOND = 1_000


def generic_media_input_tokens(block: MediaBlock) -> int:
    """Conservative fallback for media without a provider-owned rule."""

    if block.media_type == "video":
        if block.duration_seconds is None:
            return _UNKNOWN_VIDEO_TOKEN_BUDGET
        return max(
            1,
            math.ceil(block.duration_seconds * _GENERIC_VIDEO_TOKENS_PER_SECOND),
        )
    dimensions = _media_block_dimensions(block)
    if dimensions is None:
        return _UNKNOWN_IMAGE_TOKEN_BUDGET
    return 2 * _patch_count(*dimensions)


def openai_media_input_tokens(block: MediaBlock, model_id: str) -> int | None:
    """Return OpenAI's documented rule for recognized native input media."""

    if block.media_type != "image":
        return None
    dimensions = _media_block_dimensions(block)
    if dimensions is None:
        return None
    return _openai_image_tokens(
        model_id, dimensions[0], dimensions[1], block.detail or "auto"
    )


def gemini_media_input_tokens(block: MediaBlock) -> int | None:
    """Return Gemini static-processing media tokens using documented rates."""

    if block.media_type == "video":
        if block.duration_seconds is None:
            return None
        # High/static video is approximately 300 tokens per second. Using the
        # high rate is conservative when a request does not expose resolution.
        return max(1, math.ceil(block.duration_seconds * 300))
    dimensions = _media_block_dimensions(block)
    if dimensions is None:
        return None
    width, height = dimensions
    if width <= 384 and height <= 384:
        return 258
    return max(1, math.ceil(width / 768) * math.ceil(height / 768)) * 258


def anthropic_media_input_tokens(block: MediaBlock, model_id: str) -> int | None:
    """Return Claude's documented 28px-patch rule after native resizing."""

    if block.media_type != "image":
        return None
    dimensions = _media_block_dimensions(block)
    if dimensions is None:
        return None
    high_resolution = _anthropic_high_resolution_model(model_id)
    long_edge = 2576 if high_resolution else 1568
    token_budget = 4784 if high_resolution else 1568
    width, height = _fit_max_dimension(*dimensions, long_edge)
    if _anthropic_patch_count(width, height) > token_budget:
        width, height = _fit_anthropic_patch_budget(width, height, token_budget)
    return _anthropic_patch_count(width, height)


def _openai_image_tokens(
    model_id: str, width: int, height: int, detail: str
) -> int | None:
    name = model_id.lower()
    max_dimension: int
    budget: int | None
    if name.startswith("gpt-4.1-mini"):
        return _patch_tokens(
            width, height, max_dimension=2048, budget=6144, multiplier=1.62
        )
    if name.startswith("gpt-5.2"):
        return _patch_tokens(
            width, height, max_dimension=2048, budget=6144, multiplier=1.2
        )
    if name.startswith("gpt-5.4"):
        if detail == "low":
            max_dimension, budget = 2048, 6144
        else:
            max_dimension, budget = 2048, 2500
        return _patch_tokens(
            width,
            height,
            max_dimension=max_dimension,
            budget=budget,
            multiplier=1.2,
        )
    if name.startswith("gpt-5.5"):
        if detail == "low":
            max_dimension, budget = 512, None
        elif detail == "high":
            max_dimension, budget = 2048, 2500
        else:
            max_dimension, budget = 6000, 10_000
        return _patch_tokens(
            width,
            height,
            max_dimension=max_dimension,
            budget=budget,
            multiplier=1.2,
        )
    if name.startswith(("gpt-5.6-", "gpt-6-astra")):
        if detail == "low":
            max_dimension, budget = 512, None
        elif detail == "high":
            max_dimension = 65_535 if name.startswith("gpt-6-astra") else 2048
            budget = 2500
        else:
            max_dimension, budget = 65_535, None
        return _patch_tokens(
            width,
            height,
            max_dimension=max_dimension,
            budget=budget,
            multiplier=1.2,
        )
    if name.startswith("gpt-5.1"):
        return _tile_tokens(width, height, detail, base=70, per_tile=140)
    if name.startswith("gpt-4o-mini"):
        return _tile_tokens(width, height, detail, base=2833, per_tile=5667)
    if name.startswith(("gpt-4o", "chatgpt-4o", "gpt-4.1")):
        return _tile_tokens(width, height, detail, base=85, per_tile=170)
    if name == "o1" or name.startswith(("o1-", "o1-pro", "o3-")):
        return _tile_tokens(width, height, detail, base=75, per_tile=150)
    return None


def _patch_tokens(
    width: int,
    height: int,
    *,
    max_dimension: int,
    budget: int | None,
    multiplier: float,
) -> int:
    width, height = _fit_max_dimension(width, height, max_dimension)
    if budget is not None and _patch_count(width, height) > budget:
        width, height = _fit_patch_budget(width, height, budget)
    return math.ceil(_patch_count(width, height) * multiplier)


def _tile_tokens(
    width: int, height: int, detail: str, *, base: int, per_tile: int
) -> int:
    if detail == "low":
        return base
    width, height = _fit_max_dimension(width, height, 2048)
    shortest = min(width, height)
    if shortest > 768:
        scale = 768 / shortest
        width = max(1, math.floor(width * scale))
        height = max(1, math.floor(height * scale))
    tiles = math.ceil(width / 512) * math.ceil(height / 512)
    return base + tiles * per_tile


def _fit_max_dimension(width: int, height: int, limit: int) -> tuple[int, int]:
    largest = max(width, height)
    if largest <= limit:
        return width, height
    scale = limit / largest
    return max(1, math.floor(width * scale)), max(1, math.floor(height * scale))


def _fit_patch_budget(width: int, height: int, budget: int) -> tuple[int, int]:
    low, high = 0.0, 1.0
    for _ in range(64):
        scale = (low + high) / 2
        candidate_width = max(1, math.floor(width * scale))
        candidate_height = max(1, math.floor(height * scale))
        if _patch_count(candidate_width, candidate_height) <= budget:
            low = scale
        else:
            high = scale
    return max(1, math.floor(width * low)), max(1, math.floor(height * low))


def _patch_count(width: int, height: int) -> int:
    return math.ceil(width / 32) * math.ceil(height / 32)


def _anthropic_high_resolution_model(model_id: str) -> bool:
    name = model_id.lower()
    match = re.search(r"claude-(?:opus|sonnet|haiku)-(\d+)(?:[.-](\d+))?", name)
    return bool(match and (int(match.group(1)), int(match.group(2) or 0)) >= (4, 7))


def _anthropic_patch_count(width: int, height: int) -> int:
    return math.ceil(width / 28) * math.ceil(height / 28)


def _fit_anthropic_patch_budget(
    width: int, height: int, budget: int
) -> tuple[int, int]:
    low, high = 0.0, 1.0
    for _ in range(64):
        scale = (low + high) / 2
        candidate_width = max(1, math.floor(width * scale))
        candidate_height = max(1, math.floor(height * scale))
        if _anthropic_patch_count(candidate_width, candidate_height) <= budget:
            low = scale
        else:
            high = scale
    return max(1, math.floor(width * low)), max(1, math.floor(height * low))


__all__ = [
    "anthropic_media_input_tokens",
    "gemini_media_input_tokens",
    "generic_media_input_tokens",
    "openai_media_input_tokens",
]
