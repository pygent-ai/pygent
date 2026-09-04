"""Pygent's application-facing API.

Runtime, deployment, persistence, and provider extension contracts live in
their owning subpackages and are intentionally unavailable from this module.
"""

from __future__ import annotations

from .agent import (
    InjectionKind,
    PygentAgent,
    PygentAgentContext,
    ReActBudgetExceeded,
    ReActLayer,
    Reminder,
)
from .core import (
    Agent,
    AIMessage,
    Context,
    FrozenJsonObject,
    JsonValueError,
    Message,
    Module,
    RecurrentModule,
    ToolMessage,
    UserMessage,
    freeze_json,
    freeze_json_object,
    thaw_json,
)
from .llm import (
    ExponentialBackoff,
    FallbackPolicy,
    GenerationConfig,
    ModelCallError,
    ModelCallLayer,
    ModelCallOptions,
    ModelCallPolicy,
    ModelErrorKind,
    ModelGroupConfig,
    ModelRoute,
    RetryPolicy,
)
from .runtime.context_codec import ContextCodec
from .tool import (
    IdempotencyPolicy,
    ToolAuthorizationDecision,
    ToolAuthorizationRequest,
    ToolCall,
    ToolCallLayer,
    ToolDefinition,
    ToolKit,
    ToolResult,
    ToolSideEffect,
    ToolSpec,
    tool,
)

__all__ = [
    "AIMessage",
    "Agent",
    "Context",
    "ContextCodec",
    "ExponentialBackoff",
    "FallbackPolicy",
    "FrozenJsonObject",
    "GenerationConfig",
    "IdempotencyPolicy",
    "InjectionKind",
    "JsonValueError",
    "Message",
    "ModelCallError",
    "ModelCallLayer",
    "ModelCallOptions",
    "ModelCallPolicy",
    "ModelErrorKind",
    "ModelGroupConfig",
    "ModelRoute",
    "Module",
    "PygentAgent",
    "PygentAgentContext",
    "ReActBudgetExceeded",
    "ReActLayer",
    "RecurrentModule",
    "Reminder",
    "RetryPolicy",
    "ToolAuthorizationDecision",
    "ToolAuthorizationRequest",
    "ToolCall",
    "ToolCallLayer",
    "ToolDefinition",
    "ToolKit",
    "ToolMessage",
    "ToolResult",
    "ToolSideEffect",
    "ToolSpec",
    "UserMessage",
    "freeze_json",
    "freeze_json_object",
    "thaw_json",
    "tool",
]


def __dir__() -> list[str]:
    return sorted(__all__)
