"""LARA agent scaffold: main agent, fork agents, and independent children.

This package implements the basic form of the Layered Autonomous Reactive
Agent architecture (docs/design/lara-architecture/README.md): one agent
appearance, any number of background agents derived by fork or spawned
independently, all invoked through a single runner seam. Tools reach agents
only as :class:`~lara.agents.runner.ToolProfile` references into a
composition-root :class:`~lara.agents.runner.ToolRegistry` — no agent can
submit executable tool modules.

Eternal conversation builds on this scaffold; nothing in this package
depends on the durable runtime.
"""

from .child import ChildAgent, ChildSpec
from .context import (
    ContextSource,
    ContextTransform,
    ContextWindow,
    MessageSlice,
    ReminderTransform,
    SystemPromptTransform,
    align_slice_bounds,
    messages_from_history,
)
from .fork import ForkAgent, ForkSnapshot, ForkSpec
from .main import BackgroundJobHandle, BackgroundOutcome, MainAgent
from .runner import (
    AgentLimits,
    AgentRunner,
    AgentRunResult,
    ModelFactory,
    ToolkitToolSurface,
    ToolProfile,
    ToolRegistry,
    ToolSurface,
    build_model_layer_factory,
)

__all__ = [
    "AgentLimits",
    "AgentRunResult",
    "AgentRunner",
    "BackgroundJobHandle",
    "BackgroundOutcome",
    "ChildAgent",
    "ChildSpec",
    "ContextSource",
    "ContextTransform",
    "ContextWindow",
    "ForkAgent",
    "ForkSnapshot",
    "ForkSpec",
    "MainAgent",
    "MessageSlice",
    "ModelFactory",
    "ReminderTransform",
    "SystemPromptTransform",
    "ToolProfile",
    "ToolRegistry",
    "ToolSurface",
    "ToolkitToolSurface",
    "align_slice_bounds",
    "build_model_layer_factory",
    "messages_from_history",
]
