"""Portable tool declarations, admission values, and task snapshots."""

from __future__ import annotations

from dataclasses import dataclass

from pygent.core import JsonValue
from pygent.core._tool_values import (
    IdempotencyPolicy,
    MediaBlock,
    MediaSource,
    ToolAuthorizationDecision,
    ToolAuthorizationRequest,
    ToolCall,
    ToolDefinition,
    ToolLifecycle,
    ToolResult,
    ToolResultContent,
    ToolResultJson,
    ToolResultStatus,
    ToolResultText,
    ToolSideEffect,
    ToolSpec,
    ToolTask,
    ToolTaskState,
)


@dataclass(frozen=True, slots=True)
class ToolOutput:
    """Explicit business output and model-visible structured Tool content."""

    output: JsonValue = None
    content: tuple[ToolResultContent, ...] = ()

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ToolOutput cannot be subclassed")

    def __post_init__(self) -> None:
        from pygent.core import freeze_json

        content = tuple(self.content)
        if not content:
            raise ValueError("ToolOutput content must be non-empty")
        if any(
            type(value) not in (ToolResultText, ToolResultJson, MediaBlock)
            for value in content
        ):
            raise TypeError("ToolOutput content contains an unsupported value")
        object.__setattr__(self, "output", freeze_json(self.output))
        object.__setattr__(self, "content", content)


__all__ = [
    "IdempotencyPolicy",
    "MediaBlock",
    "MediaSource",
    "ToolAuthorizationDecision",
    "ToolAuthorizationRequest",
    "ToolCall",
    "ToolDefinition",
    "ToolLifecycle",
    "ToolOutput",
    "ToolResult",
    "ToolResultContent",
    "ToolResultJson",
    "ToolResultStatus",
    "ToolResultText",
    "ToolSideEffect",
    "ToolSpec",
    "ToolTask",
    "ToolTaskState",
]
