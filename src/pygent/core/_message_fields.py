"""Unified Message projection shared by the four framework consumers.

Single projection point for Message trees (WP-core of the message projection
unification refactor, ``docs/llm/MESSAGE_PROJECTION_REFACTOR.md``).  The
authoritative behavior spec is the golden corpus in
``tests/support/message_projection_golden.json``; the four profiles reproduce
the pre-refactor consumers byte-for-byte.

Profile semantics
-----------------

``MessageProfile.WIRE``
    Strict codec base ``{role, content, slot, kind, data, metadata}`` with
    ``AIMessage.usage``, full tool call identities, full ToolResult metadata
    including ``task`` (plus ``job_id`` inside the task object and
    ``missing_capabilities``), full MediaBlock payloads, and the full
    continuation object.
``MessageProfile.EFFECT``
    Model-call effect base ``{role, content, slot, metadata}`` with full
    tool call identities, full ToolResult metadata whose task object omits
    ``job_id`` and whose result omits ``missing_capabilities``, full
    MediaBlock payloads, and the full continuation object.  Never projects
    UserMessage media (preexisting gap pinned by the golden corpus).
``MessageProfile.SNAPSHOT``
    Internal base ``{role, content, slot, kind}`` with full tool call
    identities, minimal ToolResult metadata (drops
    ``task``/``missing_capabilities``), MediaBlock payloads without the
    ``base64_data`` key, and ``continuation_digest`` (sha256 over the
    canonical continuation JSON) instead of the continuation object.
``MessageProfile.ESTIMATE``
    Internal base ``{role, content, slot, kind}`` with minimal tool call
    identities and minimal ToolResult metadata, no usage, no continuation,
    and no UserMessage/AIMessage media; ToolMessage content keeps the
    ``identity_null`` media shape.

ToolAuthorizationRequest/ToolAuthorizationDecision are fully encoded only in
the WIRE profile; every other profile projects their base fields only,
matching the current consumers.  Strict decode field sets derive from the
profile via ``message_fields_expected_keys``; unknown fields are rejected
explicitly.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from enum import Enum
from typing import Any, Literal, cast

from ._tool_values import (
    IdempotencyPolicy,
    MediaBlock,
    ToolAuthorizationDecision,
    ToolAuthorizationRequest,
    ToolCall,
    ToolDefinition,
    ToolResult,
    ToolResultContent,
    ToolSideEffect,
    ToolSpec,
    ToolTask,
    ToolTaskState,
    _tool_result_content_from_value,
    _tool_result_content_to_value,
)
from .json_values import JsonValue, thaw_json
from .values import AIMessage, Message, ModelContinuation, ToolMessage, UserMessage

__all__ = [
    "MessageProfile",
    "message_fields_expected_keys",
    "message_from_fields",
    "project_message_fields",
]


class MessageProfile(Enum):
    """Explicit projection scenario for the unified Message field projector."""

    WIRE = "wire"
    EFFECT = "effect"
    SNAPSHOT = "snapshot"
    ESTIMATE = "estimate"


def _tool_call_identity(profile: MessageProfile) -> Literal["full", "minimal"]:
    """ESTIMATE is the only profile that trims tool call identities."""

    if profile is MessageProfile.ESTIMATE:
        return "minimal"
    return "full"


def _media_payload(
    profile: MessageProfile,
) -> Literal["full", "identity_drop_key", "identity_null"]:
    """Media payload shape for projected content blocks.

    ESTIMATE never projects UserMessage/AIMessage media, so its ToolMessage
    content keeps the ``identity_null`` shape.
    """

    if profile is MessageProfile.SNAPSHOT:
        return "identity_drop_key"
    if profile is MessageProfile.ESTIMATE:
        return "identity_null"
    return "full"


def _user_media_projected(profile: MessageProfile) -> bool:
    """EFFECT never projects UserMessage media; ESTIMATE omits all media."""

    return profile in (MessageProfile.WIRE, MessageProfile.SNAPSHOT)


def _thaw(value: object) -> object:
    return thaw_json(cast(JsonValue, value))


def _object_value(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ValueError(f"{name} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _only(value: Mapping[str, Any], allowed: frozenset[str], name: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{name} contains unknown fields: {sorted(unknown)}")


def project_message_fields(
    message: Message, profile: MessageProfile
) -> dict[str, object]:
    """Project one Message tree for ``profile``."""

    if not isinstance(message, Message):
        raise TypeError(f"unsupported Message value: {type(message).__name__}")
    value: dict[str, object] = {
        "role": message.role,
        "content": message.content,
        "slot": message.slot,
    }
    if profile is MessageProfile.WIRE:
        value["kind"] = message.kind
        value["data"] = thaw_json(cast(JsonValue, message.data))
        value["metadata"] = thaw_json(cast(JsonValue, message.metadata))
    elif profile is MessageProfile.EFFECT:
        value["metadata"] = thaw_json(cast(JsonValue, message.metadata))
    else:
        value["kind"] = message.kind
    if isinstance(message, UserMessage):
        if _user_media_projected(profile):
            value["media"] = [
                _content_value(block, profile) for block in message.media
            ]
    elif isinstance(message, AIMessage):
        if profile is not MessageProfile.ESTIMATE:
            value["media"] = [
                _content_value(block, profile) for block in message.media
            ]
        value["tool_calls"] = [
            _tool_call_to_dict(call, _tool_call_identity(profile))
            for call in message.tool_calls
        ]
        if profile is MessageProfile.WIRE:
            value["usage"] = thaw_json(cast(JsonValue, message.usage))
        if profile in (MessageProfile.WIRE, MessageProfile.EFFECT):
            value["continuation"] = _continuation_value(message.continuation)
        elif profile is MessageProfile.SNAPSHOT:
            value["continuation_digest"] = _continuation_digest(message.continuation)
    elif isinstance(message, ToolMessage):
        value["results"] = [
            _tool_result_to_dict(result, profile) for result in message.results
        ]
    elif profile is MessageProfile.WIRE and type(message) is not Message:
        if isinstance(message, ToolAuthorizationRequest):
            value["message_type"] = "tool.authorization.request"
            value["call"] = _tool_call_to_dict(message.call, "full")
            value["spec"] = _tool_spec_to_dict(message.spec)
            value["permissions"] = list(message.permissions)
        elif isinstance(message, ToolAuthorizationDecision):
            value["message_type"] = "tool.authorization.decision"
            value["call_id"] = message.call_id
            value["allowed"] = message.allowed
            value["reason_code"] = message.reason_code
            value["lifecycle"] = message.lifecycle
    return value


def message_fields_expected_keys(
    role: str | None, profile: MessageProfile, message_type: str | None
) -> frozenset[str]:
    """Strict decode field set for one projected message.

    The result equals the keys ``project_message_fields`` emits for the same
    role, subtype discriminator, and profile; unrecognized roles yield the
    bare base profile.
    """

    return frozenset(_message_field_keys(role, profile, message_type))


def _message_field_keys(
    role: str | None, profile: MessageProfile, message_type: str | None
) -> list[str]:
    keys = ["role", "content", "slot"]
    if profile is MessageProfile.WIRE:
        keys.extend(["kind", "data", "metadata"])
    elif profile is MessageProfile.EFFECT:
        keys.append("metadata")
    else:
        keys.append("kind")
    if profile is MessageProfile.WIRE and message_type == "tool.authorization.request":
        keys.extend(["message_type", "call", "spec", "permissions"])
        return keys
    if profile is MessageProfile.WIRE and message_type == "tool.authorization.decision":
        keys.extend(["message_type", "call_id", "allowed", "reason_code", "lifecycle"])
        return keys
    if role == "assistant" and profile is MessageProfile.WIRE:
        keys.append("usage")
    if role == "user":
        if _user_media_projected(profile):
            keys.append("media")
    elif role == "assistant":
        if profile is not MessageProfile.ESTIMATE:
            keys.append("media")
        keys.append("tool_calls")
        if profile in (MessageProfile.WIRE, MessageProfile.EFFECT):
            keys.append("continuation")
        elif profile is MessageProfile.SNAPSHOT:
            keys.append("continuation_digest")
    elif role == "tool":
        keys.append("results")
    return keys


def message_from_fields(value: object, profile: MessageProfile) -> Message:
    """Decode one projected message; unknown fields are rejected explicitly."""

    data = _object_value(value, "Message")
    role = cast("str | None", data.get("role"))
    message_type = cast("str | None", data.get("message_type"))
    unknown = set(data) - message_fields_expected_keys(role, profile, message_type)
    if unknown:
        raise ValueError(f"Message contains unknown fields: {sorted(unknown)}")
    kwargs: dict[str, Any] = {
        "content": data.get("content", ""),
        "slot": data.get("slot"),
        "kind": data.get("kind"),
        "data": _object_value(data.get("data", {}), "Message.data"),
        "metadata": _object_value(data.get("metadata", {}), "Message.metadata"),
    }
    try:
        if message_type == "tool.authorization.request":
            if role != "message":
                raise TypeError
            permissions = data.get("permissions", [])
            if not isinstance(permissions, (list, tuple)):
                raise TypeError
            return ToolAuthorizationRequest(
                **kwargs,
                call=_tool_call_from_dict(data["call"]),
                spec=_tool_spec_from_dict(data["spec"]),
                permissions=tuple(permissions),
            )
        if message_type == "tool.authorization.decision":
            if role != "message":
                raise TypeError
            return ToolAuthorizationDecision(
                **kwargs,
                call_id=data["call_id"],
                allowed=data["allowed"],
                reason_code=data["reason_code"],
                lifecycle=data.get("lifecycle", "sync"),
            )
        if role == "user":
            media = data.get("media", [])
            if not isinstance(media, (list, tuple)):
                raise TypeError
            blocks: list[MediaBlock] = []
            for item in media:
                block = _tool_result_content_from_value(item)
                if type(block) is not MediaBlock:
                    raise TypeError
                blocks.append(block)
            return UserMessage(**kwargs, media=tuple(blocks))
        if role == "assistant":
            return _assistant_from_fields(data, kwargs, profile)
        if role == "tool":
            results = data.get("results", [])
            if not isinstance(results, (list, tuple)):
                raise TypeError
            return ToolMessage(
                **kwargs,
                results=tuple(
                    _tool_result_from_dict(item, profile) for item in results
                ),
            )
        if role == "message":
            return Message(**kwargs)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid Message") from exc
    raise ValueError(f"unsupported Message role: {role!r}")


def _assistant_from_fields(
    data: Mapping[str, Any],
    kwargs: dict[str, Any],
    profile: MessageProfile,
) -> AIMessage:
    calls = data.get("tool_calls", [])
    if not isinstance(calls, (list, tuple)):
        raise TypeError
    media = data.get("media", [])
    if not isinstance(media, (list, tuple)):
        raise TypeError
    blocks: list[MediaBlock] = []
    for item in media:
        block = _tool_result_content_from_value(item)
        if type(block) is not MediaBlock:
            raise TypeError
        blocks.append(block)
    continuation = _continuation_from_value(data, profile)
    usage = (
        _object_value(data["usage"], "Message.usage")
        if profile is MessageProfile.WIRE
        else _object_value(data.get("usage", {}), "Message.usage")
    )
    return AIMessage(
        **kwargs,
        media=tuple(blocks),
        tool_calls=tuple(_tool_call_from_dict(item) for item in calls),
        usage=usage,
        continuation=continuation,
    )


def _continuation_from_value(
    data: Mapping[str, Any], profile: MessageProfile
) -> ModelContinuation | None:
    if profile is MessageProfile.ESTIMATE:
        return None
    if profile is MessageProfile.SNAPSHOT:
        # A digest is one-way; only the explicit None projection round-trips.
        if data["continuation_digest"] is not None:
            raise ValueError("continuation_digest cannot be decoded")
        return None
    raw_continuation = data["continuation"]
    if raw_continuation is None:
        return None
    continuation_data = _object_value(raw_continuation, "Message.continuation")
    _only(
        continuation_data,
        frozenset({"model_key", "provider", "model_id", "protocol", "data"}),
        "Message.continuation",
    )
    return ModelContinuation(
        model_key=continuation_data["model_key"],
        provider=continuation_data["provider"],
        model_id=continuation_data["model_id"],
        protocol=continuation_data["protocol"],
        data=_object_value(continuation_data["data"], "Message.continuation.data"),
    )


def _content_value(
    value: ToolResultContent,
    profile: MessageProfile,
) -> dict[str, object]:
    projected = _tool_result_content_to_value(value)
    if type(value) is MediaBlock:
        source = cast(dict[str, object], projected["source"])
        payload = _media_payload(profile)
        if payload == "identity_drop_key":
            del source["base64_data"]
        elif payload == "identity_null":
            source["base64_data"] = None
    return projected


def _tool_call_to_dict(
    value: ToolCall, identity: Literal["full", "minimal"] = "full"
) -> dict[str, object]:
    if identity == "minimal":
        return {
            "call_id": value.call_id,
            "name": value.name,
            "arguments": thaw_json(cast(JsonValue, value.arguments)),
        }
    return {
        "call_id": value.call_id,
        "name": value.name,
        "arguments": thaw_json(cast(JsonValue, value.arguments)),
        "tool_id": value.tool_id,
        "tool_version": value.tool_version,
        "idempotency_key": value.idempotency_key,
    }


def _continuation_value(value: ModelContinuation | None) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "model_key": value.model_key,
        "provider": value.provider,
        "model_id": value.model_id,
        "protocol": value.protocol,
        "data": thaw_json(cast(JsonValue, value.data)),
    }


def _continuation_digest(continuation: ModelContinuation | None) -> str | None:
    if continuation is None:
        return None
    encoded = json.dumps(
        {
            "model_key": continuation.model_key,
            "provider": continuation.provider,
            "model_id": continuation.model_id,
            "protocol": continuation.protocol,
            "data": thaw_json(cast(JsonValue, continuation.data)),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _tool_spec_to_dict(value: ToolSpec) -> dict[str, object]:
    return {
        "tool_id": value.tool_id,
        "version": value.version,
        "definition": _tool_definition_to_dict(value.definition),
        "side_effect": value.side_effect.value,
        "idempotency": value.idempotency.value,
        "timeout": value.timeout,
        "wait_timeout": value.wait_timeout,
        "wait_timeout_parameter": value.wait_timeout_parameter,
        "resource_key": value.resource_key,
        "sandbox_profile": value.sandbox_profile,
        "required_permissions": list(value.required_permissions),
    }


def _tool_definition_to_dict(
    value: ToolDefinition, project: Callable[[object], object] = _thaw
) -> dict[str, object]:
    return {
        "name": value.name,
        "description": value.description,
        "parameters": project(value.parameters),
        "output_schema": (
            None
            if value.output_schema is None
            else project(value.output_schema)
        ),
    }


def _tool_task_to_dict(
    value: ToolTask | None, *, include_job_id: bool = True
) -> dict[str, object] | None:
    if value is None:
        return None
    task: dict[str, object] = {
        "task_id": value.task_id,
        "call_id": value.call_id,
        "tool_id": value.tool_id,
        "version": value.version,
        "state": value.state.value,
        "metadata": thaw_json(cast(JsonValue, value.metadata)),
    }
    if include_job_id:
        task["job_id"] = value.job_id
    return task


def _tool_result_to_dict(
    value: ToolResult, profile: MessageProfile = MessageProfile.WIRE
) -> dict[str, object]:
    content = [
        _content_value(item, profile) for item in value.content
    ]
    if profile in (MessageProfile.WIRE, MessageProfile.EFFECT):
        result: dict[str, object] = {
            "call_id": value.call_id,
            "name": value.name,
            "status": value.status,
            "task": _tool_task_to_dict(
                value.task, include_job_id=profile is MessageProfile.WIRE
            ),
            "output": thaw_json(value.output),
            "content": content,
            "error": value.error,
            "error_kind": value.error_kind,
            "error_code": value.error_code,
            "retryable": value.retryable,
            "side_effect_committed": value.side_effect_committed,
            "tool_id": value.tool_id,
            "tool_version": value.tool_version,
        }
        if profile is not MessageProfile.EFFECT:
            result["missing_capabilities"] = list(value.missing_capabilities)
        return result
    if profile is MessageProfile.ESTIMATE:
        return {
            "call_id": value.call_id,
            "name": value.name,
            "status": value.status,
            "output": thaw_json(value.output),
            "content": content,
            "error": value.error,
        }
    return {
        "call_id": value.call_id,
        "name": value.name,
        "status": value.status,
        "output": thaw_json(value.output),
        "content": content,
        "error": value.error,
        "error_kind": value.error_kind,
        "error_code": value.error_code,
        "retryable": value.retryable,
        "side_effect_committed": value.side_effect_committed,
        "tool_id": value.tool_id,
        "tool_version": value.tool_version,
    }


def _tool_result_field_keys(profile: MessageProfile) -> frozenset[str]:
    if profile in (MessageProfile.WIRE, MessageProfile.EFFECT):
        keys = {
            "call_id",
            "name",
            "status",
            "task",
            "output",
            "content",
            "error",
            "error_kind",
            "error_code",
            "retryable",
            "side_effect_committed",
            "tool_id",
            "tool_version",
        }
        if profile is not MessageProfile.EFFECT:
            keys.add("missing_capabilities")
        return frozenset(keys)
    if profile is MessageProfile.ESTIMATE:
        return frozenset({"call_id", "name", "status", "output", "content", "error"})
    return frozenset(
        {
            "call_id",
            "name",
            "status",
            "output",
            "content",
            "error",
            "error_kind",
            "error_code",
            "retryable",
            "side_effect_committed",
            "tool_id",
            "tool_version",
        }
    )


def _tool_call_from_dict(value: object) -> ToolCall:
    data = _object_value(value, "ToolCall")
    _only(
        data,
        frozenset(
            {"call_id", "name", "arguments", "tool_id", "tool_version", "idempotency_key"}
        ),
        "ToolCall",
    )
    try:
        return ToolCall(
            call_id=data["call_id"],
            name=data["name"],
            arguments=_object_value(data["arguments"], "ToolCall.arguments"),
            tool_id=data.get("tool_id"),
            tool_version=data.get("tool_version"),
            idempotency_key=data.get("idempotency_key"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid ToolCall") from exc


def _tool_task_from_dict(value: object) -> ToolTask | None:
    if value is None:
        return None
    data = _object_value(value, "ToolTask")
    _only(
        data,
        frozenset(
            {"task_id", "call_id", "tool_id", "version", "state", "job_id", "metadata"}
        ),
        "ToolTask",
    )
    try:
        return ToolTask(
            task_id=data["task_id"],
            call_id=data["call_id"],
            tool_id=data["tool_id"],
            version=data["version"],
            state=ToolTaskState(data["state"]),
            job_id=data.get("job_id"),
            metadata=_object_value(data.get("metadata", {}), "ToolTask.metadata"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid ToolTask") from exc


def _tool_result_from_dict(
    value: object, profile: MessageProfile = MessageProfile.WIRE
) -> ToolResult:
    data = _object_value(value, "ToolResult")
    _only(data, _tool_result_field_keys(profile), "ToolResult")
    try:
        content = tuple(
            _tool_result_content_from_value(item) for item in data.get("content", ())
        )
        if profile in (MessageProfile.WIRE, MessageProfile.EFFECT):
            if profile is MessageProfile.EFFECT:
                return ToolResult(
                    call_id=data["call_id"],
                    name=data["name"],
                    status=data["status"],
                    task=_tool_task_from_dict(data.get("task")),
                    output=data.get("output"),
                    content=content,
                    error=data.get("error"),
                    error_kind=data.get("error_kind"),
                    error_code=data.get("error_code"),
                    retryable=data.get("retryable", False),
                    side_effect_committed=data.get("side_effect_committed"),
                    tool_id=data.get("tool_id"),
                    tool_version=data.get("tool_version"),
                )
            return ToolResult(
                call_id=data["call_id"],
                name=data["name"],
                status=data["status"],
                task=_tool_task_from_dict(data.get("task")),
                output=data.get("output"),
                content=content,
                error=data.get("error"),
                error_kind=data.get("error_kind"),
                error_code=data.get("error_code"),
                retryable=data.get("retryable", False),
                side_effect_committed=data.get("side_effect_committed"),
                tool_id=data.get("tool_id"),
                tool_version=data.get("tool_version"),
                missing_capabilities=tuple(
                    data.get("missing_capabilities", ())
                ),
            )
        if profile is MessageProfile.ESTIMATE:
            return ToolResult(
                call_id=data["call_id"],
                name=data["name"],
                status=data["status"],
                output=data.get("output"),
                content=content,
                error=data.get("error"),
            )
        return ToolResult(
            call_id=data["call_id"],
            name=data["name"],
            status=data["status"],
            output=data.get("output"),
            content=content,
            error=data.get("error"),
            error_kind=data.get("error_kind"),
            error_code=data.get("error_code"),
            retryable=data.get("retryable", False),
            side_effect_committed=data.get("side_effect_committed"),
            tool_id=data.get("tool_id"),
            tool_version=data.get("tool_version"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid ToolResult") from exc


def _tool_spec_from_dict(value: object) -> ToolSpec:
    data = _object_value(value, "ToolSpec")
    _only(
        data,
        frozenset(
            {
                "tool_id",
                "version",
                "definition",
                "side_effect",
                "idempotency",
                "timeout",
                "wait_timeout",
                "wait_timeout_parameter",
                "resource_key",
                "sandbox_profile",
                "required_permissions",
            }
        ),
        "ToolSpec",
    )
    permissions = data.get("required_permissions", [])
    if not isinstance(permissions, (list, tuple)):
        raise TypeError("ToolSpec.required_permissions must be an array")
    try:
        return ToolSpec(
            tool_id=data["tool_id"],
            version=data["version"],
            definition=_tool_definition_from_dict(data["definition"]),
            side_effect=ToolSideEffect(data.get("side_effect", "pure")),
            idempotency=IdempotencyPolicy(data.get("idempotency", "inherent")),
            timeout=data.get("timeout"),
            wait_timeout=data.get("wait_timeout"),
            wait_timeout_parameter=data.get("wait_timeout_parameter"),
            resource_key=data.get("resource_key"),
            sandbox_profile=data.get("sandbox_profile"),
            required_permissions=tuple(permissions),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid ToolSpec") from exc


def _tool_definition_from_dict(value: object) -> ToolDefinition:
    data = _object_value(value, "ToolDefinition")
    _only(
        data,
        frozenset({"name", "description", "parameters", "output_schema"}),
        "ToolDefinition",
    )
    try:
        return ToolDefinition(
            name=data["name"],
            description=data["description"],
            parameters=_object_value(
                data["parameters"], "ToolDefinition.parameters"
            ),
            output_schema=(
                None
                if data.get("output_schema") is None
                else _object_value(
                    data["output_schema"], "ToolDefinition.output_schema"
                )
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid ToolDefinition") from exc
