"""Strict provider-neutral wire codec for Message and Context values."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from pygent.core import (
    AIMessage,
    Context,
    FrozenJsonObject,
    JsonValue,
    Message,
    ToolMessage,
    UserMessage,
    freeze_json_object,
    thaw_json,
)
from pygent.tool import (
    IdempotencyPolicy,
    ToolAuthorizationDecision,
    ToolAuthorizationRequest,
    ToolCall,
    ToolDefinition,
    ToolResult,
    ToolSideEffect,
    ToolSpec,
    ToolTask,
    ToolTaskState,
)

from .context_codec import (
    DEFAULT_CONTEXT_CODECS,
    ContextCodecError,
    ContextCodecRegistry,
)


class WireCodecError(ValueError):
    """Raised when an untrusted Worker payload violates the public schema."""


def _thaw(value: object) -> object:
    return thaw_json(cast(JsonValue, value))


def _object(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise WireCodecError(f"{name} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _only(value: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise WireCodecError(f"{name} contains unknown fields: {sorted(unknown)}")


def tool_definition_to_dict(value: ToolDefinition) -> dict[str, object]:
    if type(value) is not ToolDefinition:
        raise WireCodecError("unsupported ToolDefinition subtype")
    return {
        "name": value.name,
        "description": value.description,
        "parameters": _thaw(value.parameters),
        "output_schema": (
            None if value.output_schema is None else _thaw(value.output_schema)
        ),
    }


def tool_definition_from_dict(value: object) -> ToolDefinition:
    data = _object(value, "ToolDefinition")
    _only(data, {"name", "description", "parameters", "output_schema"}, "ToolDefinition")
    try:
        return ToolDefinition(
            name=data["name"],
            description=data["description"],
            parameters=_object(data["parameters"], "ToolDefinition.parameters"),
            output_schema=(
                None
                if data.get("output_schema") is None
                else _object(data["output_schema"], "ToolDefinition.output_schema")
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise WireCodecError("invalid ToolDefinition") from exc


def _tool_spec_to_dict(value: ToolSpec) -> dict[str, object]:
    if type(value) is not ToolSpec:
        raise WireCodecError("unsupported ToolSpec subtype")
    return {
        "tool_id": value.tool_id,
        "version": value.version,
        "definition": tool_definition_to_dict(value.definition),
        "side_effect": value.side_effect.value,
        "idempotency": value.idempotency.value,
        "timeout": value.timeout,
        "resource_key": value.resource_key,
        "sandbox_profile": value.sandbox_profile,
        "required_permissions": list(value.required_permissions),
    }


def _tool_spec_from_dict(value: object) -> ToolSpec:
    data = _object(value, "ToolSpec")
    _only(
        data,
        {
            "tool_id",
            "version",
            "definition",
            "side_effect",
            "idempotency",
            "timeout",
            "resource_key",
            "sandbox_profile",
            "required_permissions",
        },
        "ToolSpec",
    )
    permissions = data.get("required_permissions", [])
    if not isinstance(permissions, (list, tuple)):
        raise WireCodecError("ToolSpec.required_permissions must be an array")
    try:
        return ToolSpec(
            tool_id=data["tool_id"],
            version=data["version"],
            definition=tool_definition_from_dict(data["definition"]),
            side_effect=ToolSideEffect(data.get("side_effect", "pure")),
            idempotency=IdempotencyPolicy(
                data.get("idempotency", "inherent")
            ),
            timeout=data.get("timeout"),
            resource_key=data.get("resource_key"),
            sandbox_profile=data.get("sandbox_profile"),
            required_permissions=tuple(permissions),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise WireCodecError("invalid ToolSpec") from exc


def _tool_call_to_dict(value: ToolCall) -> dict[str, object]:
    if type(value) is not ToolCall:
        raise WireCodecError("unsupported ToolCall subtype")
    return {
        "call_id": value.call_id,
        "name": value.name,
        "arguments": _thaw(value.arguments),
        "tool_id": value.tool_id,
        "tool_version": value.tool_version,
        "idempotency_key": value.idempotency_key,
    }


def _tool_call_from_dict(value: object) -> ToolCall:
    data = _object(value, "ToolCall")
    _only(
        data,
        {
            "call_id",
            "name",
            "arguments",
            "tool_id",
            "tool_version",
            "idempotency_key",
        },
        "ToolCall",
    )
    try:
        return ToolCall(
            call_id=data["call_id"],
            name=data["name"],
            arguments=_object(data["arguments"], "ToolCall.arguments"),
            tool_id=data.get("tool_id"),
            tool_version=data.get("tool_version"),
            idempotency_key=data.get("idempotency_key"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise WireCodecError("invalid ToolCall") from exc


def _tool_task_to_dict(value: ToolTask | None) -> object:
    if value is None:
        return None
    if type(value) is not ToolTask:
        raise WireCodecError("unsupported ToolTask subtype")
    return {
        "task_id": value.task_id,
        "call_id": value.call_id,
        "tool_id": value.tool_id,
        "version": value.version,
        "state": value.state.value,
        "job_id": value.job_id,
        "metadata": _thaw(value.metadata),
    }


def _tool_task_from_dict(value: object) -> ToolTask | None:
    if value is None:
        return None
    data = _object(value, "ToolTask")
    _only(
        data,
        {"task_id", "call_id", "tool_id", "version", "state", "job_id", "metadata"},
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
            metadata=_object(data.get("metadata", {}), "ToolTask.metadata"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise WireCodecError("invalid ToolTask") from exc


def tool_task_to_dict(value: ToolTask | None) -> object:
    return _tool_task_to_dict(value)


def tool_task_from_dict(value: object) -> ToolTask | None:
    return _tool_task_from_dict(value)


def _tool_result_to_dict(value: ToolResult) -> dict[str, object]:
    if type(value) is not ToolResult:
        raise WireCodecError("unsupported ToolResult subtype")
    return {
        "call_id": value.call_id,
        "name": value.name,
        "status": value.status,
        "task": _tool_task_to_dict(value.task),
        "output": _thaw(value.output),
        "error": value.error,
        "error_kind": value.error_kind,
        "error_code": value.error_code,
        "retryable": value.retryable,
        "side_effect_committed": value.side_effect_committed,
        "tool_id": value.tool_id,
        "tool_version": value.tool_version,
        "missing_capabilities": list(value.missing_capabilities),
    }


def _tool_result_from_dict(value: object) -> ToolResult:
    data = _object(value, "ToolResult")
    fields = {
        "call_id", "name", "status", "task", "output", "error", "error_kind",
        "error_code", "retryable", "side_effect_committed", "tool_id", "tool_version",
        "missing_capabilities",
    }
    _only(data, fields, "ToolResult")
    try:
        return ToolResult(
            call_id=data["call_id"],
            name=data["name"],
            status=data["status"],
            task=_tool_task_from_dict(data.get("task")),
            output=data.get("output"),
            error=data.get("error"),
            error_kind=data.get("error_kind"),
            error_code=data.get("error_code"),
            retryable=data.get("retryable", False),
            side_effect_committed=data.get("side_effect_committed"),
            tool_id=data.get("tool_id"),
            tool_version=data.get("tool_version"),
            missing_capabilities=tuple(data.get("missing_capabilities", ())),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise WireCodecError("invalid ToolResult") from exc


def tool_result_to_dict(value: ToolResult) -> dict[str, object]:
    return _tool_result_to_dict(value)


def tool_result_from_dict(value: object) -> ToolResult:
    return _tool_result_from_dict(value)


def message_to_dict(value: Message) -> dict[str, object]:
    if type(value) not in (
        Message,
        UserMessage,
        AIMessage,
        ToolMessage,
        ToolAuthorizationRequest,
        ToolAuthorizationDecision,
    ):
        raise WireCodecError(
            "unsupported Message subtype; use Message(kind=..., data=...) "
            "for portable domain messages"
        )
    data: dict[str, object] = {
        "role": value.role,
        "content": value.content,
        "slot": value.slot,
        "kind": value.kind,
        "data": _thaw(value.data),
        "metadata": _thaw(value.metadata),
    }
    if isinstance(value, AIMessage):
        data["tool_calls"] = [_tool_call_to_dict(call) for call in value.tool_calls]
    elif isinstance(value, ToolMessage):
        data["results"] = [_tool_result_to_dict(result) for result in value.results]
    elif isinstance(value, ToolAuthorizationRequest):
        data.update(
            {
                "message_type": "tool.authorization.request",
                "call": _tool_call_to_dict(value.call),
                "spec": _tool_spec_to_dict(value.spec),
                "permissions": list(value.permissions),
            }
        )
    elif isinstance(value, ToolAuthorizationDecision):
        data.update(
            {
                "message_type": "tool.authorization.decision",
                "call_id": value.call_id,
                "allowed": value.allowed,
                "reason_code": value.reason_code,
                "lifecycle": value.lifecycle,
            }
        )
    return data


def message_from_dict(value: object) -> Message:
    data = _object(value, "Message")
    common = {"role", "content", "slot", "kind", "data", "metadata"}
    role = data.get("role")
    message_type = data.get("message_type")
    if message_type == "tool.authorization.request":
        expected = common | {"message_type", "call", "spec", "permissions"}
    elif message_type == "tool.authorization.decision":
        expected = common | {
            "message_type",
            "call_id",
            "allowed",
            "reason_code",
            "lifecycle",
        }
    else:
        expected = common | ({"tool_calls"} if role == "assistant" else set())
        expected |= {"results"} if role == "tool" else set()
    _only(data, expected, "Message")
    kwargs = {
        "content": data.get("content", ""),
        "slot": data.get("slot"),
        "kind": data.get("kind"),
        "data": _object(data.get("data", {}), "Message.data"),
        "metadata": _object(data.get("metadata", {}), "Message.metadata"),
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
            return UserMessage(**kwargs)
        if role == "assistant":
            calls = data.get("tool_calls", [])
            if not isinstance(calls, (list, tuple)):
                raise TypeError
            return AIMessage(
                **kwargs, tool_calls=tuple(_tool_call_from_dict(item) for item in calls)
            )
        if role == "tool":
            results = data.get("results", [])
            if not isinstance(results, (list, tuple)):
                raise TypeError
            return ToolMessage(
                **kwargs,
                results=tuple(_tool_result_from_dict(item) for item in results),
            )
        if role == "message":
            return Message(**kwargs)
    except (KeyError, TypeError, ValueError) as exc:
        raise WireCodecError("invalid Message") from exc
    raise WireCodecError(f"unsupported Message role: {role!r}")


def context_to_dict(
    value: Context, *, registry: ContextCodecRegistry = DEFAULT_CONTEXT_CODECS
) -> dict[str, object]:
    try:
        codec = registry.for_value(value)
        return {
            "schema": codec.schema,
            "version": codec.version,
            "codec": codec.codec,
            "codec_digest": codec.codec_digest,
            "data": codec.encode(value),
        }
    except ContextCodecError as exc:
        raise WireCodecError("invalid or unregistered Context") from exc


def context_from_dict(
    value: object, *, registry: ContextCodecRegistry = DEFAULT_CONTEXT_CODECS
) -> Context:
    data = _object(value, "Context")
    _only(data, {"schema", "version", "codec", "codec_digest", "data"}, "Context")
    try:
        identity = (
            cast(str, data["schema"]),
            cast(int, data["version"]),
            cast(str, data["codec"]),
            cast(str, data["codec_digest"]),
        )
        if (
            not isinstance(identity[0], str)
            or not isinstance(identity[1], int)
            or isinstance(identity[1], bool)
            or not isinstance(identity[2], str)
            or not isinstance(identity[3], str)
        ):
            raise ContextCodecError("invalid Context codec identity")
        return registry.for_identity(identity).decode(data["data"])
    except (KeyError, TypeError, ValueError, ContextCodecError) as exc:
        raise WireCodecError("invalid Context") from exc


def invocation_to_dict(
    message: Message,
    context: Context,
    *,
    registry: ContextCodecRegistry = DEFAULT_CONTEXT_CODECS,
) -> FrozenJsonObject:
    return freeze_json_object(
        {
            "message": message_to_dict(message),
            "context": context_to_dict(context, registry=registry),
        }
    )


def invocation_from_dict(
    value: object, *, registry: ContextCodecRegistry = DEFAULT_CONTEXT_CODECS
) -> tuple[Message, Context]:
    data = _object(value, "invocation")
    _only(data, {"message", "context"}, "invocation")
    try:
        return message_from_dict(data["message"]), context_from_dict(
            data["context"], registry=registry
        )
    except KeyError as exc:
        raise WireCodecError("invocation requires message and context") from exc


__all__ = [
    "WireCodecError",
    "context_from_dict",
    "context_to_dict",
    "invocation_from_dict",
    "invocation_to_dict",
    "message_from_dict",
    "message_to_dict",
    "tool_definition_from_dict",
    "tool_definition_to_dict",
    "tool_result_from_dict",
    "tool_result_to_dict",
    "tool_task_from_dict",
    "tool_task_to_dict",
]
