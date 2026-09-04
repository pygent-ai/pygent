"""Portable ReAct projection operations interpreted only by ReActLayer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, cast

from pygent.core import JsonValue, Message, UserMessage, freeze_json

from .reminder import InjectionKind

REACT_PROJECTION_OPERATION_KIND = "react.projection.operation.v2"


@dataclass(frozen=True, slots=True)
class AppendToolResultContent:
    content: str
    kind: InjectionKind = InjectionKind.RUNTIME_CONTEXT

    def __post_init__(self) -> None:
        if not isinstance(self.content, str) or not self.content:
            raise ValueError("AppendToolResultContent.content must be non-empty")
        object.__setattr__(self, "kind", InjectionKind(self.kind))


@dataclass(frozen=True, slots=True)
class StandaloneUserMessage:
    message: UserMessage

    def __post_init__(self) -> None:
        if type(self.message) is not UserMessage:
            raise TypeError("StandaloneUserMessage.message must be a UserMessage")


@dataclass(frozen=True, slots=True)
class ReplaceMessageProjection:
    messages: tuple[Message, ...]
    expected_revision: int
    rebase_appended: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", tuple(self.messages))
        if any(not isinstance(message, Message) for message in self.messages):
            raise TypeError(
                "ReplaceMessageProjection.messages must contain Message values"
            )
        if (
            isinstance(self.expected_revision, bool)
            or not isinstance(self.expected_revision, int)
            or self.expected_revision < 0
        ):
            raise ValueError("expected_revision must be non-negative")
        if not isinstance(self.rebase_appended, bool):
            raise TypeError("rebase_appended must be a bool")


ReActProjectionOperation: TypeAlias = (
    AppendToolResultContent | StandaloneUserMessage | ReplaceMessageProjection
)


def encode_react_projection_operation(operation: ReActProjectionOperation) -> JsonValue:
    from pygent.runtime.codec import message_to_dict

    if type(operation) is AppendToolResultContent:
        raw: object = {
            "type": "append_tool_result_content",
            "content": operation.content,
            "kind": operation.kind.value,
        }
    elif type(operation) is StandaloneUserMessage:
        raw = {
            "type": "standalone_user_message",
            "message": message_to_dict(operation.message),
        }
    elif type(operation) is ReplaceMessageProjection:
        raw = {
            "type": "replace_message_projection",
            "messages": [message_to_dict(item) for item in operation.messages],
            "expected_revision": operation.expected_revision,
            "rebase_appended": operation.rebase_appended,
        }
    else:
        raise TypeError("unsupported ReAct projection operation")
    return freeze_json(raw)


def decode_react_projection_operation(value: JsonValue) -> ReActProjectionOperation:
    from pygent.runtime.codec import message_from_dict

    raw = cast(object, value)
    if not isinstance(raw, dict) and not hasattr(raw, "to_dict"):
        raise ValueError("ReAct projection operation must be an object")
    data = raw.to_dict() if hasattr(raw, "to_dict") else raw
    assert isinstance(data, dict)
    operation_type = data.get("type")
    if operation_type == "append_tool_result_content" and set(data) == {
        "type",
        "content",
        "kind",
    }:
        return AppendToolResultContent(
            cast(str, data["content"]), InjectionKind(data["kind"])
        )
    if operation_type == "standalone_user_message" and set(data) == {"type", "message"}:
        message = message_from_dict(data["message"])
        if type(message) is not UserMessage:
            raise ValueError("standalone operation requires a UserMessage")
        return StandaloneUserMessage(message)
    if operation_type == "replace_message_projection" and set(data) == {
        "type",
        "messages",
        "expected_revision",
        "rebase_appended",
    }:
        messages = data["messages"]
        if not isinstance(messages, list):
            raise ValueError("replacement messages must be an array")
        return ReplaceMessageProjection(
            tuple(message_from_dict(item) for item in messages),
            cast(int, data["expected_revision"]),
            cast(bool, data["rebase_appended"]),
        )
    raise ValueError("unknown or malformed ReAct projection operation")


__all__ = [
    "REACT_PROJECTION_OPERATION_KIND",
    "AppendToolResultContent",
    "ReActProjectionOperation",
    "ReplaceMessageProjection",
    "StandaloneUserMessage",
    "decode_react_projection_operation",
    "encode_react_projection_operation",
]
