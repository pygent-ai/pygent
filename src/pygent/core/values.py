"""Immutable values passed through a Pygent Module graph."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from types import NotImplementedType
from typing import ClassVar

from ._tool_values import ToolCall, ToolDefinition, ToolResult
from .json_values import (
    FrozenJsonObject,
    JsonObjectInput,
    freeze_json,
    freeze_json_object,
)

_MESSAGE_SUBCLASS_TOKEN = object()
_PENDING_MESSAGE_TYPES: set[tuple[str, str]] = set()
_PORTABLE_MESSAGE_TYPES: set[type[Message]] = set()


@dataclass(frozen=True, slots=True, kw_only=True)
class Message:
    """One typed increment passed between Modules in a dataflow."""

    content: str = ""
    slot: str | None = None
    kind: str | None = None
    data: JsonObjectInput = ()
    metadata: JsonObjectInput = ()
    role: ClassVar[str] = "message"

    def __init_subclass__(
        cls, *, _framework_token: object | None = None, **kwargs: object
    ) -> None:
        if kwargs:
            raise TypeError(f"unsupported Message subclass options: {sorted(kwargs)}")
        identity = (cls.__module__, cls.__qualname__)
        if _framework_token is _MESSAGE_SUBCLASS_TOKEN:
            # dataclass(slots=True) creates a replacement class.  Remember the
            # exact declaration identity for that immediate second creation.
            _PENDING_MESSAGE_TYPES.add(identity)
            return
        if identity in _PENDING_MESSAGE_TYPES:
            _PENDING_MESSAGE_TYPES.remove(identity)
            _PORTABLE_MESSAGE_TYPES.add(cls)
            return
        # The portable user extension point is Message(kind=..., data=...).
        # Framework-owned variants must opt in with the private identity token;
        # a forgeable module-name prefix is not an authorization boundary.
        if _framework_token is not _MESSAGE_SUBCLASS_TOKEN:
            raise TypeError(
                "Message cannot be subclassed outside pygent; use "
                "Message(kind=..., data=...) for portable domain messages"
            )

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError("message content must be a string")
        if self.slot is not None:
            if not isinstance(self.slot, str):
                raise TypeError("message slot must be a string or None")
            if not self.slot:
                raise ValueError("message slot must be non-empty when provided")
        if self.kind is not None:
            if not isinstance(self.kind, str):
                raise TypeError("message kind must be a string or None")
            if not self.kind:
                raise ValueError("message kind must be non-empty when provided")
        elif self.data:
            raise ValueError("message data requires a non-empty kind")
        object.__setattr__(self, "data", freeze_json_object(self.data))
        object.__setattr__(self, "metadata", freeze_json_object(self.metadata))


@dataclass(frozen=True, slots=True, kw_only=True)
class UserMessage(Message, _framework_token=_MESSAGE_SUBCLASS_TOKEN):
    role: ClassVar[str] = "user"


@dataclass(frozen=True, slots=True, kw_only=True)
class AIMessage(Message, _framework_token=_MESSAGE_SUBCLASS_TOKEN):
    tool_calls: tuple[ToolCall, ...] = ()
    usage: JsonObjectInput = ()
    role: ClassVar[str] = "assistant"

    def __post_init__(self) -> None:
        Message.__post_init__(self)
        tool_calls = tuple(self.tool_calls)
        if any(type(call) is not ToolCall for call in tool_calls):
            raise TypeError("AIMessage.tool_calls must contain only ToolCall values")
        object.__setattr__(self, "tool_calls", tool_calls)
        usage = freeze_json_object(self.usage)
        allowed = {
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cached_input_tokens",
            "reasoning_tokens",
        }
        unknown = set(usage) - allowed
        if unknown:
            raise ValueError(f"unsupported AIMessage usage counters: {sorted(unknown)}")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in usage.values()
        ):
            raise ValueError("AIMessage usage counters must be non-negative integers")
        object.__setattr__(self, "usage", usage)


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolMessage(Message, _framework_token=_MESSAGE_SUBCLASS_TOKEN):
    """One message containing the ordered results of one model turn."""

    results: tuple[ToolResult, ...] = ()
    role: ClassVar[str] = "tool"

    def __post_init__(self) -> None:
        Message.__post_init__(self)
        results = tuple(self.results)
        if any(type(result) is not ToolResult for result in results):
            raise TypeError("ToolMessage.results must contain only ToolResult values")
        object.__setattr__(self, "results", results)


@dataclass(frozen=True, slots=True)
class Context:
    """Current effective prompt, history, tools, and request facts."""

    system_prompt: str = ""
    messages: tuple[Message, ...] = ()
    tools: tuple[ToolDefinition, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=FrozenJsonObject)
    projection_revision: int = 0

    context_schema: ClassVar[str] = "pygent.context"
    context_schema_version: ClassVar[int] = 2

    def __init_subclass__(cls, **kwargs: object) -> None:
        if kwargs:
            raise TypeError(f"unsupported Context subclass options: {sorted(kwargs)}")
        schema = cls.__dict__.get("context_schema")
        version = cls.__dict__.get("context_schema_version")
        if not isinstance(schema, str) or not schema:
            raise TypeError("Context subclasses must declare a non-empty context_schema")
        if (
            not isinstance(version, int)
            or isinstance(version, bool)
            or version <= 0
        ):
            raise TypeError(
                "Context subclasses must declare a positive context_schema_version"
            )

    def __post_init__(self) -> None:
        if not isinstance(self.system_prompt, str):
            raise TypeError("context system_prompt must be a string")
        messages = tuple(self.messages)
        if any(
            type(message) is not Message
            and type(message) not in _PORTABLE_MESSAGE_TYPES
            for message in messages
        ):
            raise TypeError("Context.messages must contain only Message values")
        tools = tuple(self.tools)
        if any(type(tool) is not ToolDefinition for tool in tools):
            raise TypeError("Context.tools must contain only ToolDefinition values")
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "tools", tools)
        object.__setattr__(self, "metadata", freeze_json_object(self.metadata))
        if (
            isinstance(self.projection_revision, bool)
            or not isinstance(self.projection_revision, int)
            or self.projection_revision < 0
        ):
            raise ValueError("context projection_revision must be non-negative")
        if type(self) is not Context:
            _validate_context_extension(self)

    def __add__(self, message: object) -> Context | NotImplementedType:
        if not isinstance(message, Message):
            return NotImplemented
        messages = self.messages
        if message.slot is not None:
            messages = tuple(item for item in messages if item.slot != message.slot)
        return replace(self, messages=messages + (message,))


def _validate_context_extension(value: Context) -> None:
    """Reject mutable or process-local values in user Context fields."""

    context_type = type(value)
    parameters = getattr(context_type, "__dataclass_params__", None)
    if not is_dataclass(value) or parameters is None or not parameters.frozen:
        raise TypeError("Context subclasses must be frozen dataclasses")
    if "__slots__" not in context_type.__dict__:
        raise TypeError("Context subclasses must use dataclass(slots=True)")
    base_names = {item.name for item in fields(Context)}
    for item in fields(value):
        if item.name not in base_names:
            _validate_portable_context_value(getattr(value, item.name), set())


def _validate_portable_context_value(value: object, active: set[int]) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        freeze_json(value)
        return
    if isinstance(value, (Message, ToolDefinition, FrozenJsonObject)):
        return
    if isinstance(value, tuple):
        identity = id(value)
        if identity in active:
            raise TypeError("Context fields must not contain cyclic values")
        active.add(identity)
        try:
            for item in value:
                _validate_portable_context_value(item, active)
        finally:
            active.remove(identity)
        return
    if is_dataclass(value) and not isinstance(value, type):
        parameters = getattr(type(value), "__dataclass_params__", None)
        if parameters is None or not parameters.frozen or "__slots__" not in type(value).__dict__:
            raise TypeError("nested Context dataclasses must be frozen and use slots")
        identity = id(value)
        if identity in active:
            raise TypeError("Context fields must not contain cyclic values")
        active.add(identity)
        try:
            for item in fields(value):
                _validate_portable_context_value(getattr(value, item.name), active)
        finally:
            active.remove(identity)
        return
    raise TypeError(
        f"unsupported portable Context field value: {type(value).__name__}"
    )


def validate_context(value: Context) -> None:
    """Validate one Context at an execution boundary."""

    if not isinstance(value, Context):
        raise TypeError("value must be a Context")
    Context.__post_init__(value)


__all__ = [
    "AIMessage",
    "Context",
    "FrozenJsonObject",
    "Message",
    "ToolMessage",
    "UserMessage",
    "validate_context",
]
