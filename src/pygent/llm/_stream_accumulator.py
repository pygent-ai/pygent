"""Stateful reduction of provider stream parts into one model response."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

import jsonschema  # type: ignore[import-untyped]

from pygent.core import AIMessage, FrozenJsonObject, freeze_json_object
from pygent.tool import ToolCall, ToolDefinition

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
class ModelStreamAccumulator:
    """Own mutable stream reduction state while the invoker owns iteration."""

    generation: GenerationConfig
    tools: tuple[ToolDefinition, ...]
    text_parts: list[str] = field(default_factory=list)
    usage: FrozenJsonObject = field(default_factory=freeze_json_object)
    calls: dict[int, dict[str, str]] = field(default_factory=dict)
    selected_route: str | None = None
    selected_attempt: int | None = None
    finish_reason: str = "other"
    provider_request_id: str | None = None

    async def consume(
        self, part: ModelProviderStreamPart, event_sink: EventSink | None
    ) -> None:
        data = freeze_json_object(part.data)
        route_value = data.get("route_id")
        if isinstance(route_value, str):
            self.selected_route = route_value
        attempt_value = data.get("attempt")
        if isinstance(attempt_value, int) and not isinstance(attempt_value, bool):
            self.selected_attempt = attempt_value

        if part.kind == ModelProviderStreamKind.REASONING:
            await _emit(event_sink, ModelEventKind.REASONING_DELTA, data)
        elif part.kind == ModelProviderStreamKind.TEXT:
            value = data.get("text", "")
            if isinstance(value, str):
                self.text_parts.append(value)
            await _emit(event_sink, ModelEventKind.TEXT_DELTA, data)
        elif part.kind == ModelProviderStreamKind.TOOL_CALL:
            await self._consume_tool_call(data, event_sink)
        elif part.kind == ModelProviderStreamKind.USAGE:
            usage_data = data.to_dict()
            usage_data.pop("route_id", None)
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

    async def _consume_tool_call(
        self, data: FrozenJsonObject, event_sink: EventSink | None
    ) -> None:
        index = data.get("index", 0)
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            await _raise_invalid_model_response(
                event_sink,
                "model stream returned an invalid tool-call index",
                route_id=self.selected_route,
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
        content = "".join(self.text_parts)
        await self._validate_content(content, event_sink)
        content = "".join(self.text_parts)
        tool_calls = await self._finish_tool_calls(event_sink)
        if self.selected_route is None or self.selected_attempt is None:
            raise ModelCallError(
                "model stream completed without attempt identity",
                kind=ModelErrorKind.INVALID_RESPONSE,
            )
        if self.finish_reason == "other":
            self.finish_reason = "tool_calls" if tool_calls else "stop"
        await _emit(
            event_sink,
            ModelEventKind.USAGE,
            _usage_event_payload(
                self.usage,
                route_id=self.selected_route,
                attempt=self.selected_attempt,
                final=True,
            ),
        )
        await _emit(
            event_sink,
            ModelEventKind.ATTEMPT_SUCCEEDED,
            {"route_id": self.selected_route, "attempt": self.selected_attempt},
        )
        await _emit(
            event_sink,
            ModelEventKind.COMPLETED,
            {
                "route_id": self.selected_route,
                "attempt": self.selected_attempt,
                "finish_reason": self.finish_reason,
                "provider_request_id": self.provider_request_id,
            },
        )
        return ModelProviderResponse(
            message=AIMessage(
                content=content,
                tool_calls=tuple(tool_calls),
                metadata={"route_id": self.selected_route},
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
            jsonschema.validate(
                value,
                freeze_json_object(self.generation.response_schema).to_dict(),
            )
            self.text_parts[:] = [_wire_json(value)]
        except (json.JSONDecodeError, jsonschema.ValidationError):
            await _raise_invalid_model_response(
                event_sink,
                "model output does not match the declared JSON schema",
                route_id=self.selected_route,
                attempt=self.selected_attempt,
                usage=self.usage,
                reason_code=ModelFailureReason.GENERATION_SCHEMA_INVALID,
            )

    async def _finish_tool_calls(
        self, event_sink: EventSink | None
    ) -> list[ToolCall]:
        tool_calls: list[ToolCall] = []
        for index in sorted(self.calls):
            call = self.calls[index]
            try:
                arguments = _decode_json(call["arguments"] or "{}")
                if not isinstance(arguments, Mapping):
                    raise TypeError
                name = _original_tool_name(call["name"], self.tools)
                call_id = call["call_id"] or _synthetic_tool_call_id(
                    route_id=self.selected_route or "provider",
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
                        "route_id": self.selected_route,
                        "attempt": self.selected_attempt,
                    },
                )
            except (json.JSONDecodeError, TypeError, ValueError):
                await _raise_invalid_model_response(
                    event_sink,
                    "model stream returned an invalid tool call",
                    route_id=self.selected_route,
                    attempt=self.selected_attempt,
                    usage=self.usage,
                    reason_code=ModelFailureReason.TOOL_CALL_INVALID,
                )
        return tool_calls


__all__ = ["ModelStreamAccumulator"]
