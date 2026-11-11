from typing import Any

from .react import TOOL_BATCH_INTERRUPT_NOTICE, ReActBudgetExceeded, ReActLayer
from .react_agent import (
    ContextCompressionLayer,
    ContextCompressionLimitExceeded,
    ContextCompressionUnavailable,
    ReActAgent,
    ReActAgentContext,
)
from .react_projection_operations import (
    REACT_PROJECTION_OPERATION_KIND,
    AppendToolResultContent,
    ReActProjectionOperation,
    ReplaceMessageProjection,
    StandaloneUserMessage,
    SteeringMode,
    decode_react_projection_operation,
    encode_react_projection_operation,
)
from .reminder import InjectionKind, Reminder, format_context

_LARA_EXPORTS = ("PygentAgent",)


def __getattr__(name: str) -> Any:
    # The lara individual's class resolves lazily through the ``pygent.agent.lara``
    # package boundary: an eager import here would make `import pygent.agent` pull
    # the whole lara stack and cycle against pygent's own __init__, which imports
    # ``.agent`` before ``.core`` and ``.llm`` are bound.
    if name in _LARA_EXPORTS:
        from .lara import PygentAgent

        globals()["PygentAgent"] = PygentAgent
        return PygentAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "REACT_PROJECTION_OPERATION_KIND",
    "TOOL_BATCH_INTERRUPT_NOTICE",
    "AppendToolResultContent",
    "ContextCompressionLayer",
    "ContextCompressionLimitExceeded",
    "ContextCompressionUnavailable",
    "InjectionKind",
    "PygentAgent",
    "ReActAgent",
    "ReActAgentContext",
    "ReActBudgetExceeded",
    "ReActLayer",
    "ReActProjectionOperation",
    "Reminder",
    "ReplaceMessageProjection",
    "StandaloneUserMessage",
    "SteeringMode",
    "decode_react_projection_operation",
    "encode_react_projection_operation",
    "format_context",
]
