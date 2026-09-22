"""Target-aware routing and request projection for media Tool results."""

from __future__ import annotations

import json
from dataclasses import replace

from pygent.core import (
    Context,
    Message,
    ToolMessage,
    UserMessage,
    freeze_json,
    thaw_json,
)
from pygent.tool import (
    ToolResult,
    ToolResultJson,
    MediaBlock,
    ToolResultText,
)

from ._adapter_contracts import (
    MediaProjectionTrace,
    MediaProjector,
    ModelProviderAdapter,
    ModelProviderMediaDeliveryValidator,
)
from .configuration import ModelSpec
from .types import ModelFailureReason


def pending_media_blocks(
    message: Message, context: Context
) -> tuple[MediaBlock, ...]:
    return tuple(
        block
        for candidate in (*context.messages, message)
        for block in _message_media_blocks(candidate)
    )


def _message_media_blocks(message: Message) -> tuple[MediaBlock, ...]:
    if isinstance(message, UserMessage):
        return message.media
    if isinstance(message, ToolMessage):
        return tuple(
            block
            for result in message.results
            for block in result.content
            if type(block) is MediaBlock
        )
    return ()


def media_delivery_gaps(
    blocks: tuple[MediaBlock, ...],
    *,
    model: ModelSpec,
    adapter: ModelProviderAdapter,
    projector: MediaProjector | None = None,
) -> tuple[str, ...]:
    gaps: set[str] = set()
    capabilities = adapter.media_transport
    for block in blocks:
        plan = (
            None
            if projector is None
            else projector.plan(block, model=model, endpoint=capabilities)
        )
        if block.media_type not in model.capabilities.modalities.input:
            gaps.add(f"modalities.input.{block.media_type}")
        if not capabilities.enabled or block.media_type not in capabilities.modalities:
            gaps.add(f"media_transport.{block.media_type}")
        if block.source.kind not in capabilities.source_kinds and plan is None:
            gaps.add(f"media_transport.source.{block.source.kind}")
        if (
            capabilities.max_media_bytes is not None
            and block.source.size_bytes is not None
            and block.source.size_bytes > capabilities.max_media_bytes
            and plan is None
        ):
            gaps.add("media_transport.max_media_bytes")
        if (
            block.media_type in model.capabilities.modalities.input
            and capabilities.enabled
            and block.media_type in capabilities.modalities
            and plan is None
        ):
            gaps.add(f"media_projection.{block.media_type}")
        if isinstance(adapter, ModelProviderMediaDeliveryValidator):
            delivery_block = (
                block
                if plan is None or plan.target_mime_type == block.mime_type
                else replace(block, mime_type=plan.target_mime_type)
            )
            gaps.update(adapter.media_delivery_gaps(delivery_block, model))
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


def project_media_for_model(
    message: Message,
    context: Context,
    *,
    model: ModelSpec,
    adapter: ModelProviderAdapter,
    projector: MediaProjector,
) -> tuple[Message, Context, tuple[MediaProjectionTrace, ...]]:
    """Create one model-specific request projection without mutating Context."""

    traces: list[MediaProjectionTrace] = []
    projected_message = _project_media_message(
        message,
        model=model,
        adapter=adapter,
        projector=projector,
        traces=traces,
    )
    projected_messages = tuple(
        _project_media_message(
            item,
            model=model,
            adapter=adapter,
            projector=projector,
            traces=traces,
        )
        for item in context.messages
    )
    projected_context = (
        context
        if projected_messages == context.messages
        else replace(context, messages=projected_messages)
    )
    return projected_message, projected_context, tuple(traces)


def _delivery_reason(
    block: MediaBlock,
    *,
    model: ModelSpec,
    adapter: ModelProviderAdapter,
) -> ModelFailureReason | None:
    capabilities = adapter.media_transport
    if block.media_type not in model.capabilities.modalities.input:
        return ModelFailureReason.MODEL_INPUT_MODALITY_UNSUPPORTED
    if not capabilities.enabled or block.media_type not in capabilities.modalities:
        return ModelFailureReason.MEDIA_TRANSPORT_UNSUPPORTED
    if block.source.kind not in capabilities.source_kinds:
        return ModelFailureReason.MEDIA_SOURCE_UNSUPPORTED
    if (
        capabilities.max_media_bytes is not None
        and block.source.size_bytes is not None
        and block.source.size_bytes > capabilities.max_media_bytes
    ):
        return ModelFailureReason.MEDIA_TOO_LARGE
    if isinstance(adapter, ModelProviderMediaDeliveryValidator) and (
        adapter.media_delivery_gaps(block, model)
    ):
        return ModelFailureReason.MEDIA_TRANSPORT_UNSUPPORTED
    return None


def _not_viewed_media_value(
    block: MediaBlock, reason: ModelFailureReason
) -> dict[str, object]:
    media_ref = _media_reference(block)
    return {
        "type": block.media_type,
        "mime_type": block.mime_type,
        "status": "not_viewed",
        "reason_code": reason.value,
        "media_ref": media_ref,
        "message": (
            f"The current model cannot view this {block.media_type}. "
            f"The historical {block.media_type} remains available at {media_ref}."
        ),
        "sha256": block.source.sha256,
        "size_bytes": block.source.size_bytes,
        "width": block.width,
        "height": block.height,
        "duration_seconds": block.duration_seconds,
        "fps": block.fps,
        "has_audio": block.has_audio,
    }


def _project_result(
    result: ToolResult,
    *,
    model: ModelSpec,
    adapter: ModelProviderAdapter,
) -> ToolResult:
    if not any(type(block) is MediaBlock for block in result.content):
        return result
    if adapter.media_transport.enabled:
        projected_content = tuple(
            ToolResultText(
                json.dumps(
                    _not_viewed_media_value(block, reason),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            if type(block) is MediaBlock
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
            projected_blocks.append({"type": "json", "value": thaw_json(block.value)})
        elif type(block) is MediaBlock:
            reason = _delivery_reason(block, model=model, adapter=adapter)
            if (
                reason is None
            ):  # pragma: no cover - disabled endpoint rejects all content
                reason = ModelFailureReason.MEDIA_TRANSPORT_UNSUPPORTED
            projected_blocks.append(_not_viewed_media_value(block, reason))
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
    if isinstance(message, UserMessage) and message.media:
        notes: list[str] = []
        for block in message.media:
            reason = _delivery_reason(block, model=model, adapter=adapter)
            if reason is None:  # pragma: no cover - fallback implies delivery gaps
                reason = ModelFailureReason.MEDIA_TRANSPORT_UNSUPPORTED
            notes.append(
                json.dumps(
                    _not_viewed_media_value(block, reason),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        content = message.content
        for note in notes:
            content = f"{content}\n{note}" if content else note
        return replace(message, content=content, media=())
    if not isinstance(message, ToolMessage):
        return message
    results = tuple(
        _project_result(result, model=model, adapter=adapter)
        for result in message.results
    )
    return message if results == message.results else replace(message, results=results)


def _project_media_message(
    message: Message,
    *,
    model: ModelSpec,
    adapter: ModelProviderAdapter,
    projector: MediaProjector,
    traces: list[MediaProjectionTrace],
) -> Message:
    if isinstance(message, UserMessage):
        if not message.media:
            return message
        projected_media: list[MediaBlock] = []
        for block in message.media:
            plan = projector.plan(
                block,
                model=model,
                endpoint=adapter.media_transport,
            )
            if plan is None:
                raise ValueError("media projector received an incompatible route")
            projected = projector.project(block, call_id="user", plan=plan)
            projected_media.append(projected.media)
            traces.append(projected.trace)
        return (
            message
            if tuple(projected_media) == message.media
            else replace(message, media=tuple(projected_media))
        )
    if not isinstance(message, ToolMessage):
        return message
    results: list[ToolResult] = []
    for result in message.results:
        content: list[ToolResultText | ToolResultJson | MediaBlock] = []
        for block in result.content:
            if type(block) is not MediaBlock:
                content.append(block)
                continue
            plan = projector.plan(
                block,
                model=model,
                endpoint=adapter.media_transport,
            )
            if plan is None:
                raise ValueError("media projector received an incompatible route")
            projected = projector.project(block, call_id=result.call_id, plan=plan)
            content.append(projected.media)
            traces.append(projected.trace)
        results.append(
            result
            if tuple(content) == result.content
            else replace(result, content=tuple(content))
        )
    projected_results = tuple(results)
    return (
        message
        if projected_results == message.results
        else replace(message, results=projected_results)
    )


def _media_reference(block: MediaBlock) -> str:
    source = block.source
    if source.kind == "resource":
        return str(source.uri)
    if source.kind == "url":
        return str(source.url)
    return f"inline:sha256:{source.sha256}"


__all__ = [
    "media_delivery_gaps",
    "pending_media_blocks",
    "project_media_for_model",
    "project_request_for_model",
]
