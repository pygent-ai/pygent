"""The built-in ReAct agent implementation entry.

``pygent.agent`` hosts both reusable base capabilities and independent agent
implementations. This package owns the built-in ReAct individual's
implementation (``agent.py``): the foreground ``ReActAgent`` with
token-window compression, its portable context, and the compression layer.
The pygent_v1 individual lives at ``pygent.agent.pygent_v1``; future implementations
land as sibling subpackages without touching the base capabilities.
"""

from .agent import (
    ContextCompressionLayer,
    ContextCompressionLimitExceeded,
    ContextCompressionUnavailable,
    ReActAgent,
    ReActAgentContext,
)

__all__ = [
    "ContextCompressionLayer",
    "ContextCompressionLimitExceeded",
    "ContextCompressionUnavailable",
    "ReActAgent",
    "ReActAgentContext",
]
