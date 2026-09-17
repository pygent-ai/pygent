"""Target-aware routing and request projection for media Tool results."""

from __future__ import annotations

import json
from dataclasses import replace

from pygent.core import (
    Context,
    Message,
    ToolMessage,
    freeze_json,
    thaw_json,
)
from pygent.tool import (
    ToolResult,
    ToolResultJson,
    ToolResultMedia,
    ToolResultText,
)

from ._adapter_contracts import ModelProviderAdapter
from .configuration import ModelSpec
from .types import ModelFailureReason


def pending_tool_result_media(
    message: Message, context: Context
) -> tuple[ToolResultMedia, ...]:
    return tuple(
        block
        for candidate in (*context.messages, message)
        if isinstance(candidate, ToolMessage)
        for result in candidate.results
        for block in result.content
        if type(block) is ToolResultMedia
    )


def media_delivery_gaps(
    blocks: tuple[ToolResultMedia, ...],
    *,
    model: ModelSpec,
    adapter: ModelProviderAdapter,
) -> tuple[str, ...]:
    gaps: set[str] = set()
    capabilities = adapter.tool_result_content
    for block in blocks:
        if block.media_type not in model.capabilities.modalities.input:
            gaps.add(f"modalities.input.{block.media_type}")
        if not capabilities.enabled or block.media_type not in capabilities.modalities:
            gaps.add(f"tool_result_content.{block.media_type}")
        if block.source.kind not in capabilities.source_kinds:
            gaps.add(f"tool_result_content.source.{block.source.kind}")
        if (
            capabilities.max_media_bytes is not None
            and block.source.size_bytes is not None
            and block.source.size_bytes > capabilities.max_media_bytes
        ):
            gaps.add("tool_result_content.max_media_bytes")
    return tuple(sorted(gaps))


def project_request_for_model(
    message: Message,
    context: Context,
    *,
    model: ModelSpec,
    adapter: ModelProviderAdapter,
) -> tuple[Message, Context]:
    projected_message = _project_message(message, model=model, adapter=adapter)
    projected_messages = tuple(
        _project_message(item, model=model, adapter=adapter)
        for item in context.messages
    )
    projected_context = (
        context
        if projected_messages == context.messages
        else replace(context, messages=projected_messages)
    )
    return projected_message, projected_context


def _delivery_reason(
    block: ToolResultMedia,
    *,
    model: ModelSpec,
    adapter: ModelProviderAdapter,
) -> ModelFailureReason | None:
    capabilities = adapter.tool_result_content
    if block.media_type not in model.capabilities.modalities.input:
        return ModelFailureReason.MODEL_INPUT_MODALITY_UNSUPPORTED
    if not capabilities.enabled or block.media_type not in capabilities.modalities:
        return ModelFailureReason.TOOL_RESULT_CONTENT_UNSUPPORTED
    if block.source.kind not in capabilities.source_kinds:
        return ModelFailureReason.MEDIA_SOURCE_UNSUPPORTED
    if (
        capabilities.max_media_bytes is not None
        and block.source.size_bytes is not None
        and block.source.size_bytes > capabilities.max_media_bytes
    ):
        return ModelFailureReason.MEDIA_TOO_LARGE
    return None


def _unavailable_media_value(
    block: ToolResultMedia, reason: ModelFailureReason
) -> dict[str, object]:
    return {
        "type": block.media_type,
        "mime_type": block.mime_type,
        "status": "unavailable",
        "reason_code": reason.value,
        "message": f"The current model cannot read this {block.media_type}.",
        "sha256": block.source.sha256,
        "size_bytes": block.source.size_bytes,
    }


def _project_result(
    result: ToolResult,
    *,
    model: ModelSpec,
    adapter: ModelProviderAdapter,
) -> ToolResult:
    if not any(type(block) is ToolResultMedia for block in result.content):
        return result
    if adapter.tool_result_content.enabled:
        projected_content = tuple(
            ToolResultText(
                json.dumps(
                    _unavailable_media_value(block, reason),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            if type(block) is ToolResultMedia
            and (reason := _delivery_reason(block, model=model, adapter=adapter))
            is not None
            else block
            for block in result.content
        )
        return replace(result, content=projected_content)

    projected_blocks: list[dict[str, object]] = []
    for block in result.content:
        if type(block) is ToolResultText:
            projected_blocks.append({"type": "text", "text": block.text})
        elif type(block) is ToolResultJson:
            projected_blocks.append(
                {"type": "json", "value": thaw_json(block.value)}
            )
        elif type(block) is ToolResultMedia:
            reason = _delivery_reason(block, model=model, adapter=adapter)
            if reason is None:  # pragma: no cover - disabled endpoint rejects all content
                reason = ModelFailureReason.TOOL_RESULT_CONTENT_UNSUPPORTED
            projected_blocks.append(_unavailable_media_value(block, reason))
    return replace(
        result,
        output=freeze_json(
            {
                "tool_output": thaw_json(result.output),
                "model_visible_content": projected_blocks,
            }
        ),
        content=(),
    )


def _project_message(
    message: Message,
    *,
    model: ModelSpec,
    adapter: ModelProviderAdapter,
) -> Message:
    if not isinstance(message, ToolMessage):
        return message
    results = tuple(
        _project_result(result, model=model, adapter=adapter)
        for result in message.results
    )
    return message if results == message.results else replace(message, results=results)


__all__ = [
    "media_delivery_gaps",
    "pending_tool_result_media",
    "project_request_for_model",
]
