"""Bounded provider-neutral snapshots for actual model attempts."""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import cast

from pygent.core import AIMessage, JsonValue, Message, ToolMessage, thaw_json

from ._adapter_contracts import ModelProviderRequest
from .types import ModelErrorKind, ModelProviderError

MAX_MODEL_REQUEST_SNAPSHOT_BYTES = 1024 * 1024


def prepared_request_event(
    request: ModelProviderRequest, *, attempt: int
) -> dict[str, object]:
    projection = _request_projection(request)
    encoded = json.dumps(
        projection,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    if len(encoded) > MAX_MODEL_REQUEST_SNAPSHOT_BYTES:
        raise ModelProviderError(
            ModelErrorKind.INVALID_REQUEST,
            "model request snapshot exceeds the 1 MiB public event limit",
        )
    return {
        "route_id": request.route.route_id,
        "attempt": attempt,
        "request_id": f"model-request-{uuid.uuid4().hex}",
        "request_digest": f"sha256:{hashlib.sha256(encoded).hexdigest()}",
        "request": projection,
    }


def _request_projection(request: ModelProviderRequest) -> dict[str, object]:
    generation = request.generation
    return {
        "provider": request.route.provider,
        "model": request.route.model,
        "system_prompt": request.context.system_prompt,
        "messages": [
            _message_projection(message) for message in request.context.messages
        ],
        "current_message": _message_projection(request.message),
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": thaw_json(cast(JsonValue, tool.parameters)),
                "output_schema": (
                    None
                    if tool.output_schema is None
                    else thaw_json(cast(JsonValue, tool.output_schema))
                ),
            }
            for tool in request.tools
        ],
        "generation": {
            "temperature": generation.temperature,
            "max_output_tokens": generation.max_output_tokens,
            "response_schema": (
                None
                if generation.response_schema is None
                else thaw_json(cast(JsonValue, generation.response_schema))
            ),
            "response_schema_name": generation.response_schema_name,
            "tool_choice": generation.tool_choice,
        },
        "projection_revision": request.context.projection_revision,
    }


def _message_projection(message: Message) -> dict[str, object]:
    value: dict[str, object] = {
        "role": message.role,
        "content": message.content,
        "kind": message.kind,
        "slot": message.slot,
    }
    if isinstance(message, AIMessage):
        value["tool_calls"] = [
            {
                "call_id": call.call_id,
                "name": call.name,
                "arguments": thaw_json(cast(JsonValue, call.arguments)),
                "tool_id": call.tool_id,
                "tool_version": call.tool_version,
                "idempotency_key": call.idempotency_key,
            }
            for call in message.tool_calls
        ]
    elif isinstance(message, ToolMessage):
        value["results"] = [
            {
                "call_id": result.call_id,
                "name": result.name,
                "status": result.status,
                "output": thaw_json(result.output),
                "error": result.error,
                "error_kind": result.error_kind,
                "error_code": result.error_code,
                "retryable": result.retryable,
                "side_effect_committed": result.side_effect_committed,
                "tool_id": result.tool_id,
                "tool_version": result.tool_version,
            }
            for result in message.results
        ]
    return value


__all__ = ["MAX_MODEL_REQUEST_SNAPSHOT_BYTES", "prepared_request_event"]
