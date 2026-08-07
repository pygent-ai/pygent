"""Immutable values passed through a Pygent Module graph."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import NotImplementedType
from typing import TYPE_CHECKING, ClassVar

from .json_values import (
    FrozenJsonObject,
    JsonObjectInput,
    freeze_json_object,
)

if TYPE_CHECKING:
    from pygent.tool.types import ToolCall, ToolDefinition, ToolResult


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
    role: ClassVar[str] = "assistant"

    def __post_init__(self) -> None:
        Message.__post_init__(self)
        from pygent.tool.types import ToolCall

        tool_calls = tuple(self.tool_calls)
        if any(type(call) is not ToolCall for call in tool_calls):
            raise TypeError("AIMessage.tool_calls must contain only ToolCall values")
        object.__setattr__(self, "tool_calls", tool_calls)


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolMessage(Message, _framework_token=_MESSAGE_SUBCLASS_TOKEN):
    """One message containing the ordered results of one model turn."""

    results: tuple[ToolResult, ...] = ()
    role: ClassVar[str] = "tool"

    def __post_init__(self) -> None:
        Message.__post_init__(self)
        from pygent.tool.types import ToolResult

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
    metadata: JsonObjectInput = ()

    def __init_subclass__(cls, **kwargs: object) -> None:
        if kwargs:
            raise TypeError(f"unsupported Context subclass options: {sorted(kwargs)}")
        raise TypeError(
            "Context cannot be subclassed; put portable request facts in metadata"
        )

    def __post_init__(self) -> None:
        from pygent.tool.types import ToolDefinition

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

    def __add__(self, message: object) -> Context | NotImplementedType:
        if not isinstance(message, Message):
            return NotImplemented
        messages = self.messages
        if message.slot is not None:
            messages = tuple(item for item in messages if item.slot != message.slot)
        return replace(self, messages=messages + (message,))


__all__ = [
    "AIMessage",
    "Context",
    "FrozenJsonObject",
    "Message",
    "ToolMessage",
    "UserMessage",
]
