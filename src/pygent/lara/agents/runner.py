"""Single invocation seam for every LARA agent (main, fork, child).

Design invariants (docs/design/lara-architecture/README.md):

- G4: tools reach an agent only through a :class:`ToolProfile` reference into
  a :class:`ToolRegistry` the composition root populated. Agents can never
  submit executable modules, so no side-channel tool path can exist.
- Single source of truth for the tool surface: the runner resolves the
  profile once per run and feeds the resolved definitions into the model
  factory, so model-visible tools and executable tools cannot diverge.
- G5: budget limits live on the runner side, never with the agent.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Protocol, cast, runtime_checkable

from pygent import (
    AIMessage,
    Context,
    ExponentialBackoff,
    GenerationConfig,
    Message,
    ModelCallLayer,
    ModelErrorKind,
    Module,
    ModelGroup,
    PygentAgent,
    RetryPolicy,
    ToolAuthorizationDecision,
    ToolAuthorizationRequest,
    ToolDefinition,
    ToolKit,
    ToolMessage,
    UserMessage,
)
from pygent.core import JsonValue
from pygent.llm import ModelCallPolicy, ModelEntry, ModelInvoker

from .context import ContextWindow

# Aligned with lara.runtime.agent.core.MODEL_RETRYABLE_ERROR_KINDS; redeclared
# here so the agent scaffold does not depend on the durable runtime package.
MODEL_RETRYABLE_ERROR_KINDS = (
    ModelErrorKind.TIMEOUT,
    ModelErrorKind.RATE_LIMIT,
    ModelErrorKind.UNAVAILABLE,
    ModelErrorKind.INVALID_RESPONSE,
)

DISABLED_COMPRESSION_WINDOW_TOKENS = 1 << 60

ModelFactory = Callable[[tuple[ToolDefinition, ...]], Module[Message, AIMessage]]


@dataclass(frozen=True)
class AgentLimits:
    """Per-run budget limits; defaults follow background-job practice."""

    max_steps: int = 16
    max_model_calls: int = 16
    max_tool_calls: int = 64
    context_window_tokens: int = DISABLED_COMPRESSION_WINDOW_TOKENS
    compression_trigger_ratio: float = 0.9
    max_compressions: int = 1


@dataclass(frozen=True)
class ToolProfile:
    """A reference to a tool surface registered by the composition root.

    Agents hold profiles, never surfaces: the registry is the architectural
    gate that keeps definitions, executors, authorization, and audit on one
    pipeline (G4). An unknown profile id fails the run before any model call.
    """

    surface_id: str

    def __post_init__(self) -> None:
        if not self.surface_id.strip():
            raise ValueError("ToolProfile.surface_id must be non-empty")


@runtime_checkable
class ToolSurface(Protocol):
    """One tool pipeline: definitions plus executor module.

    Surfaces are built by the composition root only. The in-process
    implementation is :class:`ToolkitToolSurface`. Known limitation before
    production use (lara-architecture §6.8): the production foreground
    pipeline (audit, EventStore, file effects) runs *after* tool execution
    and needs a managed ``LaraContext``, so adapting it will require
    extending this protocol with a context factory and post-tool hooks —
    the current two-member protocol alone cannot carry it.
    """

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]: ...

    def tool_module(self) -> Module[AIMessage, ToolMessage]: ...


class ToolkitToolSurface:
    """In-process tool surface backed by a pygent ``ToolKit``.

    Intended for composition roots (and tests). It executes tools locally
    with no audit/event/file-effect plumbing; the production runtime must
    register surfaces that route through the durable tool pipeline, which
    additionally requires extending ``ToolSurface`` with context and
    post-tool hooks (lara-architecture §6.8).
    """

    # Subclasses may override to keep their own authorization reason code.
    _AUTHORIZATION_REASON_CODE = "lara-agents-uniform-tool-surface"

    def __init__(self, toolkit: ToolKit) -> None:
        self._toolkit = toolkit

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        return self._toolkit.definitions

    def tool_module(self) -> Module[AIMessage, ToolMessage]:
        def authorize(
            request: ToolAuthorizationRequest,
            context: Context,
        ) -> ToolAuthorizationDecision:
            del context
            return ToolAuthorizationDecision(
                call_id=request.call.call_id,
                allowed=True,
                reason_code=self._AUTHORIZATION_REASON_CODE,
            )

        return self._toolkit.local_layer(authorization_adapter=authorize)


class ToolRegistry:
    """Composition-root registry of tool surfaces, addressable by id."""

    def __init__(self) -> None:
        self._surfaces: dict[str, ToolSurface] = {}

    def register(self, surface_id: str, surface: ToolSurface) -> None:
        if not surface_id.strip():
            raise ValueError("surface_id must be non-empty")
        if surface_id in self._surfaces:
            raise ValueError(f"tool surface {surface_id!r} is already registered")
        self._surfaces[surface_id] = surface

    def resolve(self, profile: ToolProfile) -> ToolSurface:
        try:
            return self._surfaces[profile.surface_id]
        except KeyError:
            raise ValueError(
                f"tool surface {profile.surface_id!r} is not registered; "
                "tool profiles must reference surfaces registered by the "
                "composition root"
            ) from None


class _NoToolLayer(Module[AIMessage, ToolMessage]):
    """Tool layer for runs without a tool surface; calls are a bug."""

    async def forward(
        self, message: AIMessage, context: Context
    ) -> tuple[ToolMessage, Context]:
        raise RuntimeError(
            "agent run has no tool surface configured but the model emitted "
            f"a tool call ({message.tool_calls!r})"
        )


@dataclass(frozen=True)
class AgentRunResult:
    """Outcome of one agent invocation."""

    label: str
    answer: AIMessage
    context: Context

    @property
    def text(self) -> str:
        return self.answer.content

    def window(self) -> ContextWindow:
        """The run's resulting context as fork material (G3 chaining)."""

        return ContextWindow(
            system_prompt=self.context.system_prompt,
            messages=tuple(self.context.messages),
            # pygent types Context.metadata as Mapping[str, object] but only
            # ever stores FrozenJsonObject (JSON) values in it.
            metadata=cast("Mapping[str, JsonValue]", self.context.metadata),
        )


class AgentRunner:
    """Assemble and invoke a pygent ReAct agent for one run.

    Each run is isolated: the runner resolves the tool profile, builds the
    model layer from the resolved definitions, and assembles a fresh
    ``PygentAgent`` so concurrent runs never share mutable state.
    """

    def __init__(
        self,
        *,
        model_factory: ModelFactory,
        registry: ToolRegistry | None = None,
        default_tools: ToolProfile | None = None,
        limits: AgentLimits | None = None,
        label: str = "lara-agent",
    ) -> None:
        self._model_factory = model_factory
        self._registry = registry
        self._default_tools = default_tools
        self._limits = limits or AgentLimits()
        self._label = label

    async def run(
        self,
        *,
        message: str,
        system_prompt: str | None = None,
        prefix: ContextWindow | None = None,
        tools: ToolProfile | None = None,
        limits: AgentLimits | None = None,
        label: str | None = None,
    ) -> AgentRunResult:
        """Run one ReAct turn.

        ``prefix`` is fork block 1: an existing context window whose system
        prompt and messages are continued verbatim. When ``prefix`` is given,
        its system prompt wins and ``system_prompt`` must not also be set.
        ``message`` is the steering user message (fork block 2, the turn's
        user input, or a child's directive); the multi-turn ReAct loop that
        follows is block 3.
        """

        if (prefix is None) == (system_prompt is None):
            raise ValueError("provide exactly one of prefix or system_prompt")
        surface: ToolSurface | None = None
        profile = tools if tools is not None else self._default_tools
        if profile is not None:
            if self._registry is None:
                raise ValueError(
                    "runner has no ToolRegistry; cannot resolve the ToolProfile"
                )
            surface = self._registry.resolve(profile)
        resolved_limits = limits if limits is not None else self._limits
        resolved_label = label or self._label
        definitions = surface.definitions if surface is not None else ()
        effective_system_prompt = (
            prefix.system_prompt if prefix is not None else system_prompt
        )
        assert effective_system_prompt is not None
        model = self._model_factory(definitions)
        agent = PygentAgent(
            system_prompt=effective_system_prompt,
            compression_prompt=(
                "Compress this agent run's transcript without losing tool errors "
                "or tool results."
            ),
            model=model,
            compressor=model,
            tools=surface.tool_module() if surface is not None else _NoToolLayer(),
            context_window_tokens=resolved_limits.context_window_tokens,
            compression_trigger_ratio=resolved_limits.compression_trigger_ratio,
            max_compressions=resolved_limits.max_compressions,
            max_steps=resolved_limits.max_steps,
            max_model_calls=resolved_limits.max_model_calls,
            max_tool_calls=resolved_limits.max_tool_calls,
        )
        metadata: dict[str, JsonValue] = {"lara.agent.label": resolved_label}
        if prefix is not None:
            metadata = {**dict(prefix.metadata), **metadata}
        context = agent.new_context(tools=definitions, metadata=metadata)
        if prefix is not None:
            context = replace(
                context,
                system_prompt=prefix.system_prompt,
                messages=prefix.messages,
            )
        answer, final_context = await agent.invoke(
            UserMessage(content=message), context
        )
        return AgentRunResult(
            label=resolved_label, answer=answer, context=final_context
        )


def build_model_layer_factory(
    invoker: ModelInvoker,
    *,
    model_group_name: str,
    models: Sequence[ModelEntry],
    max_attempts_per_model: int = 3,
    attempt_idle_timeout_seconds: float = 120.0,
) -> ModelFactory:
    """Build a :class:`ModelFactory` producing concrete model call layers.

    Mirrors ``LaraAgent.new_model_layer`` with background-job retry defaults.
    The factory receives the resolved tool definitions of each run, keeping
    model-visible tools and the executable tool surface on one source of
    truth. ``models`` are the group's ``ModelEntry`` values from the run
    config; the invoker executes requests locally.
    """

    def factory(definitions: tuple[ToolDefinition, ...]) -> ModelCallLayer:
        return ModelCallLayer(
            model_group=ModelGroup(name=model_group_name, models=tuple(models)),
            policy=ModelCallPolicy(),
            retry_policy=RetryPolicy(
                max_attempts_per_model=max_attempts_per_model,
                retry_on=MODEL_RETRYABLE_ERROR_KINDS,
                attempt_idle_timeout_seconds=attempt_idle_timeout_seconds,
                backoff=ExponentialBackoff(initial=0.5, maximum=4.0, multiplier=2.0),
            ),
            generation=GenerationConfig(),
            tools=tuple(definitions),
            invoker=invoker,
        )

    return factory
