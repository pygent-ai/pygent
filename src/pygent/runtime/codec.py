"""Strict provider-neutral wire codec for Message and Context values."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

from pygent.core import (
    AIMessage,
    Context,
    ExecutionInput,
    ExecutionInputDelivery,
    FrozenJsonObject,
    JsonValue,
    Message,
    MessageProfile,
    ToolMessage,
    UserMessage,
    freeze_json_object,
    message_fields_expected_keys,
    message_from_fields,
    project_message_fields,
    thaw_json,
)
from pygent.core._message_fields import (
    _tool_definition_from_dict,
    _tool_definition_to_dict,
    _tool_result_from_dict,
    _tool_result_to_dict,
    _tool_task_from_dict,
    _tool_task_to_dict,
)
from pygent.tool import (
    ToolAuthorizationDecision,
    ToolAuthorizationRequest,
    ToolDefinition,
    ToolResult,
    ToolTask,
)

from .context_codec import (
    DEFAULT_CONTEXT_CODECS,
    ContextCodec,
    ContextCodecError,
    ContextCodecRegistry,
)


class WireCodecError(ValueError):
    """Raised when an untrusted Worker payload violates the public schema."""


def execution_input_to_dict(value: ExecutionInput) -> dict[str, object]:
    if type(value) is not ExecutionInput:
        raise WireCodecError("unsupported ExecutionInput subtype")
    return value.to_dict()


def execution_input_from_dict(value: object) -> ExecutionInput:
    try:
        return ExecutionInput.from_dict(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise WireCodecError("invalid ExecutionInput") from exc


def execution_input_delivery_to_dict(
    value: ExecutionInputDelivery,
) -> dict[str, object]:
    if type(value) is not ExecutionInputDelivery:
        raise WireCodecError("unsupported ExecutionInputDelivery subtype")
    return value.to_dict()


def execution_input_delivery_from_dict(value: object) -> ExecutionInputDelivery:
    try:
        return ExecutionInputDelivery.from_dict(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise WireCodecError("invalid ExecutionInputDelivery") from exc


def _thaw(value: object) -> object:
    return thaw_json(cast(JsonValue, value))


def _retain_json(value: object) -> object:
    if isinstance(value, FrozenJsonObject) and type(value) is not FrozenJsonObject:
        return _thaw(value)
    return value


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
    return _tool_definition_value(value, _thaw)


def _tool_definition_value(
    value: ToolDefinition, project: Callable[[object], object]
) -> dict[str, object]:
    if type(value) is not ToolDefinition:
        raise WireCodecError("unsupported ToolDefinition subtype")
    return _tool_definition_to_dict(value, project)


def tool_definition_from_dict(value: object) -> ToolDefinition:
    try:
        return _tool_definition_from_dict(value)
    except ValueError as exc:
        # The core decode helpers raise the wire error texts verbatim as
        # ValueError; only the exception type is narrowed to the public one.
        raise WireCodecError(str(exc)) from exc


def tool_task_to_dict(value: ToolTask | None) -> object:
    if value is not None and type(value) is not ToolTask:
        raise WireCodecError("unsupported ToolTask subtype")
    return _tool_task_to_dict(value)


def tool_task_from_dict(value: object) -> ToolTask | None:
    try:
        return _tool_task_from_dict(value)
    except ValueError as exc:
        raise WireCodecError(str(exc)) from exc


def tool_result_to_dict(value: ToolResult) -> dict[str, object]:
    if type(value) is not ToolResult:
        raise WireCodecError("unsupported ToolResult subtype")
    return _tool_result_to_dict(value)


def tool_result_from_dict(value: object) -> ToolResult:
    try:
        return _tool_result_from_dict(value)
    except ValueError as exc:
        raise WireCodecError(str(exc)) from exc


def message_to_dict(value: Message) -> dict[str, object]:
    return _message_value(value, _thaw)


def _message_value(
    value: Message, project: Callable[[object], object]
) -> dict[str, object]:
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
    projected = project_message_fields(value, MessageProfile.WIRE)
    if project is _retain_json:
        # ``_retain_json`` call sites (invocation payloads) keep the identity
        # of values the Message tree already stores as FrozenJsonObject; the
        # unified core projection always thaws, so re-attach the originals.
        _retain_stored_frozen_values(value, projected)
    return projected


def _retain_stored_frozen_values(
    message: Message, projected: dict[str, object]
) -> None:
    """Swap thawed JSON values back to their stored frozen instances."""

    projected["data"] = message.data
    projected["metadata"] = message.metadata
    if isinstance(message, UserMessage):
        return
    if isinstance(message, AIMessage):
        projected["usage"] = message.usage
        for call, item in zip(
            message.tool_calls, cast("list[dict[str, object]]", projected["tool_calls"])
        ):
            item["arguments"] = call.arguments
        if message.continuation is not None:
            cast("dict[str, object]", projected["continuation"])[
                "data"
            ] = message.continuation.data
        return
    if isinstance(message, ToolMessage):
        for result, item in zip(
            message.results, cast("list[dict[str, object]]", projected["results"])
        ):
            item["output"] = result.output
            if result.task is not None:
                cast("dict[str, object]", item["task"])[
                    "metadata"
                ] = result.task.metadata
        return
    if isinstance(message, ToolAuthorizationRequest):
        cast("dict[str, object]", projected["call"])[
            "arguments"
        ] = message.call.arguments
        definition = cast(
            "dict[str, object]", cast("dict[str, object]", projected["spec"])["definition"]
        )
        definition["parameters"] = message.spec.definition.parameters
        if message.spec.definition.output_schema is not None:
            definition["output_schema"] = message.spec.definition.output_schema
    # ToolAuthorizationDecision projects scalar fields only.


def message_from_dict(value: object) -> Message:
    data = _object(value, "Message")
    role = data.get("role")
    message_type = data.get("message_type")
    _only(
        data,
        set(message_fields_expected_keys(role, MessageProfile.WIRE, message_type)),
        "Message",
    )
    try:
        return message_from_fields(data, MessageProfile.WIRE)
    except ValueError as exc:
        # ``message_from_fields`` mirrors this function's wire error texts
        # verbatim; only the exception type is narrowed back to the public one.
        raise WireCodecError(str(exc)) from exc


def context_to_dict(
    value: Context, *, registry: ContextCodecRegistry = DEFAULT_CONTEXT_CODECS
) -> dict[str, object]:
    return _context_value(value, registry, _thaw)


def _context_value(
    value: Context, registry: ContextCodecRegistry, project: Callable[[object], object]
) -> dict[str, object]:
    try:
        codec = registry.for_value(value)
        return {
            "schema": codec.schema,
            "version": codec.version,
            "codec": codec.codec,
            "codec_digest": codec.codec_digest,
            "data": (
                codec._encode(value, project)
                if project is _retain_json and type(codec) is ContextCodec
                else codec.encode(value)
            ),
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
            "message": _message_value(message, _retain_json),
            "context": _context_value(context, registry, _retain_json),
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
    "execution_input_delivery_from_dict",
    "execution_input_delivery_to_dict",
    "execution_input_from_dict",
    "execution_input_to_dict",
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
