"""Provider-neutral snapshots for actual model attempts."""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import cast

from pygent.core import (
    JsonValue,
    Message,
    MessageProfile,
    ModelContinuation,
    project_message_fields,
    thaw_json,
)
from pygent.core._message_fields import (
    _continuation_digest as _core_continuation_digest,
)

from ._adapter_contracts import ModelProviderRequest


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
    return {
        "model_key": request.model_key,
        "attempt": attempt,
        "request_id": f"model-request-{uuid.uuid4().hex}",
        "request_digest": f"sha256:{hashlib.sha256(encoded).hexdigest()}",
        "request": projection,
    }


def _request_projection(request: ModelProviderRequest) -> dict[str, object]:
    generation = request.generation
    return {
        "provider": request.model.provider,
        "model": request.model.model_id,
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
            "output_modalities": list(generation.output_modalities),
        },
        "projection_revision": request.context.projection_revision,
        "media_projections": [
            {
                "call_id": item.call_id,
                "media_type": item.media_type,
                "canonical_reference": item.canonical_reference,
                "canonical_sha256": item.canonical_sha256,
                "projected_reference": item.projected_reference,
                "projected_sha256": item.projected_sha256,
                "projected_size_bytes": item.projected_size_bytes,
                "mime_type": item.mime_type,
                "width": item.width,
                "height": item.height,
                "duration_seconds": item.duration_seconds,
                "fps": item.fps,
                "has_audio": item.has_audio,
                "transformations": list(item.transformations),
                "projector_version": item.projector_version,
            }
            for item in request.media_projections
        ],
    }


def _message_projection(message: Message) -> dict[str, object]:
    return project_message_fields(message, MessageProfile.SNAPSHOT)


def _continuation_digest(continuation: ModelContinuation | None) -> str | None:
    return _core_continuation_digest(continuation)


__all__ = ["prepared_request_event"]
