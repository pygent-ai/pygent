"""Stateful reduction of provider stream parts into one model response."""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

import jsonschema  # type: ignore[import-untyped]

from pygent.core import (
    AIMessage,
    FrozenJsonObject,
    ModelContinuation,
    freeze_json_object,
)
from pygent.core._tool_values import MediaBlockType
from pygent.tool import MediaBlock, MediaSource, ToolCall, ToolDefinition
from pygent.tool._schema import validate_instance

from ._adapter_contracts import (
    EventSink,
    ModelEventKind,
    ModelProviderResponse,
    ModelProviderStreamKind,
    ModelProviderStreamPart,
    _emit,
    _normalized_finish_reason,
    _raise_invalid_model_response,
    _tool_delta_payload,
    _tool_item_payload,
    _usage_event_payload,
    _validated_canonical_usage,
)
from .openai_compatible import (
    _decode_json,
    _original_tool_name,
    _synthetic_tool_call_id,
    _wire_json,
)
from .types import (
    GenerationConfig,
    ModelCallError,
    ModelErrorKind,
    ModelFailureReason,
)


@dataclass(slots=True)
class _MediaShardBuffer:
    """Per-item buffer for one media item assembled from stream fragments."""

    media_type: str
    mime_type: str
    mime_parameters: str
    fragments: list[str] = field(default_factory=list)
    expected_fragment: int = 0


@dataclass(slots=True)
class ModelStreamAccumulator:
    """Own mutable stream reduction state while the invoker owns iteration."""

    generation: GenerationConfig
    tools: tuple[ToolDefinition, ...]
    text_parts: list[str] = field(default_factory=list)
    media: list[MediaBlock] = field(default_factory=list)
    media_shards: dict[int, _MediaShardBuffer] = field(default_factory=dict)
    usage: FrozenJsonObject = field(default_factory=freeze_json_object)
    calls: dict[int, dict[str, str]] = field(default_factory=dict)
    selected_model_key: str | None = None
    selected_attempt: int | None = None
    finish_reason: str = "other"
    provider_request_id: str | None = None
    continuation: ModelContinuation | None = None
    media_provider: str | None = None
    media_model_id: str | None = None

    async def consume(
        self, part: ModelProviderStreamPart, event_sink: EventSink | None
    ) -> None:
        data = freeze_json_object(part.data)
        if part.kind == ModelProviderStreamKind.RESET:
            public_output = data.get("public_output")
            if not isinstance(public_output, bool):
                raise TypeError("model stream reset public_output must be a bool")
            self.text_parts.clear()
            self.media.clear()
            self.usage = freeze_json_object()
            self.calls.clear()
            self.selected_model_key = None
            self.selected_attempt = None
            self.finish_reason = "other"
            self.provider_request_id = None
            self.continuation = None
            self.media_provider = None
            self.media_model_id = None
            self.media_shards.clear()
            if public_output:
                await _emit(
                    event_sink,
                    ModelEventKind.OUTPUT_RESET,
                    {
                        "model_key": data["model_key"],
                        "attempt": data["attempt"],
                    },
                )
            return
        model_key_value = data.get("model_key")
        if isinstance(model_key_value, str):
            self.selected_model_key = model_key_value
        attempt_value = data.get("attempt")
        if isinstance(attempt_value, int) and not isinstance(attempt_value, bool):
            self.selected_attempt = attempt_value

        if part.kind == ModelProviderStreamKind.CONTINUATION:
            continuation_data = data.get("data")
            if not isinstance(continuation_data, FrozenJsonObject):
                raise TypeError("model continuation data must be a JSON object")
            self.continuation = ModelContinuation(
                model_key=cast(str, data.get("model_key")),
                provider=cast(str, data.get("provider")),
                model_id=cast(str, data.get("model_id")),
                protocol=cast(str, data.get("protocol")),
                data=continuation_data,
            )
        elif part.kind == ModelProviderStreamKind.REASONING:
            await _emit(event_sink, ModelEventKind.REASONING_DELTA, data)
        elif part.kind == ModelProviderStreamKind.TEXT:
            value = data.get("text", "")
            if isinstance(value, str):
                self.text_parts.append(value)
            await _emit(event_sink, ModelEventKind.TEXT_DELTA, data)
        elif part.kind == ModelProviderStreamKind.MEDIA:
            block = _media_block_from_part(data)
            provider = data.get("provider")
            model_id = data.get("model_id")
            if self.media_provider is None and isinstance(provider, str):
                self.media_provider = provider
            if self.media_model_id is None and isinstance(model_id, str):
                self.media_model_id = model_id
            index = len(self.media)
            self.media.append(block)
            await _emit(
                event_sink,
                ModelEventKind.MEDIA,
                {
                    "media_index": index,
                    "media_type": block.media_type,
                    "mime_type": block.mime_type,
                    "mime_parameters": block.mime_parameters or "",
                    "sha256": block.source.sha256,
                    "size_bytes": block.source.size_bytes,
                    "data": data["data"],
                },
            )
        elif part.kind == ModelProviderStreamKind.MEDIA_DELTA:
            await self._consume_media_delta(data, event_sink)
        elif part.kind == ModelProviderStreamKind.TOOL_CALL:
            await self._consume_tool_call(data, event_sink)
        elif part.kind == ModelProviderStreamKind.USAGE:
            usage_data = data.to_dict()
            usage_data.pop("model_key", None)
            usage_data.pop("attempt", None)
            self.usage = _validated_canonical_usage(usage_data)
        elif part.kind == ModelProviderStreamKind.FINISH:
            raw_reason = data.get("finish_reason")
            if isinstance(raw_reason, str):
                normalized_reason = _normalized_finish_reason(raw_reason)
                if normalized_reason != "other" or self.finish_reason == "other":
                    self.finish_reason = normalized_reason
            raw_request_id = data.get("provider_request_id")
            if isinstance(raw_request_id, str) and raw_request_id:
                self.provider_request_id = raw_request_id

    async def _consume_media_delta(
        self, data: FrozenJsonObject, event_sink: EventSink | None
    ) -> None:
        """Merge one transport media fragment into its media item buffer."""

        media_index = data.get("media_index")
        fragment_index = data.get("fragment_index")
        final = data.get("final")
        media_type = data.get("media_type")
        mime_type = data.get("mime_type")
        mime_parameters = data.get("mime_parameters", "")
        base64_data = data.get("data")
        if (
            not isinstance(media_index, int)
            or isinstance(media_index, bool)
            or media_index < 0
        ):
            raise TypeError("model media delta requires a non-negative media index")
        if (
            not isinstance(fragment_index, int)
            or isinstance(fragment_index, bool)
            or fragment_index < 0
        ):
            raise TypeError(
                "model media delta requires a non-negative fragment index"
            )
        if not isinstance(final, bool):
            raise TypeError("model media delta requires a final flag")
        if media_type not in ("image", "video", "audio"):
            raise TypeError("model media delta has an unsupported media type")
        if not isinstance(mime_type, str) or not mime_type:
            raise TypeError("model media delta requires a mime type")
        if not isinstance(base64_data, str):
            raise TypeError("model media delta requires base64 data")
        # An empty data payload is the final=True completion marker; the
        # fragments joined so far form the media content.
        if base64_data and not mime_type.startswith(f"{media_type}/"):
            raise TypeError("model media delta mime type does not match its type")
        if not isinstance(mime_parameters, str):
            raise TypeError("model media delta mime_parameters must be a string")

        provider = data.get("provider")
        model_id = data.get("model_id")
        if self.media_provider is None and isinstance(provider, str):
            self.media_provider = provider
        if self.media_model_id is None and isinstance(model_id, str):
            self.media_model_id = model_id

        buffer = self.media_shards.get(media_index)
        if buffer is None:
            buffer = _MediaShardBuffer(
                media_type=cast(str, media_type),
                mime_type=mime_type,
                mime_parameters=mime_parameters,
            )
            self.media_shards[media_index] = buffer
        if (
            buffer.media_type != media_type
            or buffer.mime_type != mime_type
            or buffer.mime_parameters != mime_parameters
        ):
            raise TypeError("model media delta does not match its media item")
        if fragment_index != buffer.expected_fragment:
            raise TypeError(
                "model media delta arrived out of order for its media item"
            )
        buffer.fragments.append(base64_data)
        buffer.expected_fragment += 1

        await _emit(
            event_sink,
            ModelEventKind.MEDIA_DELTA,
            {
                "media_index": media_index,
                "fragment_index": fragment_index,
                "media_type": media_type,
                "mime_type": mime_type,
                "data": base64_data,
                "final": final,
            },
        )
        if not final:
            return

        try:
            joined = b"".join(
                base64.b64decode(fragment, validate=True)
                for fragment in buffer.fragments
            )
        except (binascii.Error, ValueError) as exc:
            raise TypeError("model media delta fragments are not valid base64") from exc
        block = MediaBlock(
            media_type=cast(MediaBlockType, media_type),
            mime_type=mime_type,
            source=MediaSource.inline(joined),
            mime_parameters=buffer.mime_parameters or None,
        )
        del self.media_shards[media_index]
        index = len(self.media)
        self.media.append(block)
        await _emit(
            event_sink,
            ModelEventKind.MEDIA,
            {
                "media_index": index,
                "media_type": block.media_type,
                "mime_type": block.mime_type,
                "mime_parameters": block.mime_parameters or "",
                "sha256": block.source.sha256,
                "size_bytes": block.source.size_bytes,
                "data": base64.b64encode(joined).decode("ascii"),
            },
        )

    async def _consume_tool_call(
        self, data: FrozenJsonObject, event_sink: EventSink | None
    ) -> None:
        index = data.get("index", 0)
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            await _raise_invalid_model_response(
                event_sink,
                "model stream returned an invalid tool-call index",
                model_key=self.selected_model_key,
                attempt=self.selected_attempt,
                usage=self.usage,
            )
        if index not in self.calls:
            self.calls[index] = {"call_id": "", "name": "", "arguments": ""}
            await _emit(
                event_sink,
                ModelEventKind.TOOL_CALL_STARTED,
                _tool_item_payload(data, index=index),
            )
        current = self.calls[index]
        for source, target in (
            ("call_id_delta", "call_id"),
            ("name_delta", "name"),
            ("arguments_delta", "arguments"),
        ):
            value = data.get(source, "")
            if isinstance(value, str):
                current[target] += value
        await _emit(
            event_sink,
            ModelEventKind.TOOL_CALL_DELTA,
            _tool_delta_payload(data, index=index),
        )

    async def finish(self, event_sink: EventSink | None) -> ModelProviderResponse:
        if self.media_shards:
            # The provider stream ended without finalizing media it started
            # delivering; completing the call here would silently drop it.
            raise ModelCallError(
                "model stream ended with incomplete media",
                kind=ModelErrorKind.INVALID_RESPONSE,
            )
        content = "".join(self.text_parts)
        await self._validate_content(content, event_sink)
        content = "".join(self.text_parts)
        tool_calls = await self._finish_tool_calls(event_sink)
        if self.selected_model_key is None or self.selected_attempt is None:
            raise ModelCallError(
                "model stream completed without attempt identity",
                kind=ModelErrorKind.INVALID_RESPONSE,
            )
        if self.finish_reason == "other":
            self.finish_reason = "tool_calls" if tool_calls else "stop"
        if (
            self.media
            and self.generation.response_schema is not None
            and self.media_provider is not None
            and self.media_model_id is not None
        ):
            # Media arrived although the request declared a JSON text schema.
            await _emit(
                event_sink,
                ModelEventKind.CAPABILITY_WARNING,
                {
                    "model_key": self.selected_model_key,
                    "provider": self.media_provider,
                    "model_id": self.media_model_id,
                    "missing_capabilities": ("structured_output.media_present",),
                },
            )
        await _emit(
            event_sink,
            ModelEventKind.USAGE,
            _usage_event_payload(
                self.usage,
                model_key=self.selected_model_key,
                attempt=self.selected_attempt,
                final=True,
            ),
        )
        await _emit(
            event_sink,
            ModelEventKind.ATTEMPT_SUCCEEDED,
            {"model_key": self.selected_model_key, "attempt": self.selected_attempt},
        )
        await _emit(
            event_sink,
            ModelEventKind.COMPLETED,
            {
                "model_key": self.selected_model_key,
                "attempt": self.selected_attempt,
                "finish_reason": self.finish_reason,
                "provider_request_id": self.provider_request_id,
            },
        )
        return ModelProviderResponse(
            message=AIMessage(
                content=content,
                media=tuple(self.media),
                tool_calls=tuple(tool_calls),
                metadata={"model_key": self.selected_model_key},
                continuation=self.continuation,
            ),
            usage=self.usage,
            provider_request_id=self.provider_request_id,
            finish_reason=self.finish_reason,
        )

    async def _validate_content(
        self, content: str, event_sink: EventSink | None
    ) -> None:
        if self.generation.response_schema is None:
            return
        try:
            value = _decode_json(content)
            validate_instance(
                freeze_json_object(self.generation.response_schema), value
            )
            self.text_parts[:] = [_wire_json(value)]
        except (json.JSONDecodeError, jsonschema.ValidationError):
            await _raise_invalid_model_response(
                event_sink,
                "model output does not match the declared JSON schema",
                model_key=self.selected_model_key,
                attempt=self.selected_attempt,
                usage=self.usage,
                reason_code=ModelFailureReason.GENERATION_SCHEMA_INVALID,
            )

    async def _finish_tool_calls(self, event_sink: EventSink | None) -> list[ToolCall]:
        tool_calls: list[ToolCall] = []
        for index in sorted(self.calls):
            call = self.calls[index]
            try:
                arguments = _decode_json(call["arguments"] or "{}")
                if not isinstance(arguments, Mapping):
                    raise TypeError
                name = _original_tool_name(call["name"], self.tools)
                call_id = call["call_id"] or _synthetic_tool_call_id(
                    model_key=self.selected_model_key or "provider",
                    provider_request_id=self.provider_request_id,
                    index=index,
                    name=name,
                    arguments=cast(Mapping[str, object], arguments),
                )
                tool_calls.append(
                    ToolCall(
                        call_id=call_id,
                        name=name,
                        arguments=cast(Mapping[str, object], arguments),
                    )
                )
                await _emit(
                    event_sink,
                    ModelEventKind.TOOL_CALL_COMPLETED,
                    {
                        "item_id": f"tool-{index}",
                        "index": index,
                        "call_id": call_id,
                        "name": name,
                        "arguments": cast(Mapping[str, object], arguments),
                        "model_key": self.selected_model_key,
                        "attempt": self.selected_attempt,
                    },
                )
            except (json.JSONDecodeError, TypeError, ValueError):
                await _raise_invalid_model_response(
                    event_sink,
                    "model stream returned an invalid tool call",
                    model_key=self.selected_model_key,
                    attempt=self.selected_attempt,
                    usage=self.usage,
                    reason_code=ModelFailureReason.TOOL_CALL_INVALID,
                )
        return tool_calls


def _media_block_from_part(data: FrozenJsonObject) -> MediaBlock:
    """Build one validated complete media block from a provider media part."""

    media_type = data.get("media_type")
    mime_type = data.get("mime_type")
    mime_parameters = data.get("mime_parameters", "")
    base64_data = data.get("data")
    if media_type not in ("image", "video", "audio"):
        raise TypeError("model media stream part has an unsupported media type")
    if not isinstance(mime_type, str) or not mime_type:
        raise TypeError("model media stream part requires a mime type")
    if not isinstance(base64_data, str) or not base64_data:
        raise TypeError("model media stream part requires base64 data")
    if not mime_type.startswith(f"{media_type}/"):
        raise TypeError("model media stream part mime type does not match its type")
    if not isinstance(mime_parameters, str):
        raise TypeError("model media stream part mime_parameters must be a string")
    return MediaBlock(
        media_type=cast(MediaBlockType, media_type),
        mime_type=mime_type,
        source=MediaSource(kind="inline", base64_data=base64_data),
        mime_parameters=mime_parameters or None,
    )


__all__ = ["ModelStreamAccumulator"]
