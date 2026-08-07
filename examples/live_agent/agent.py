"""A stateless ReAct Agent backed by an opt-in real model service."""

from __future__ import annotations

import os
import secrets
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import cast

import httpx

from pygent import (
    AIMessage,
    Context,
    ExponentialBackoff,
    FallbackPolicy,
    GenerationConfig,
    IdempotencyPolicy,
    ModelCallLayer,
    ModelErrorKind,
    ModelGroupConfig,
    ModelRoute,
    Module,
    ReActLayer,
    RetryPolicy,
    ToolAuthorizationDecision,
    ToolAuthorizationRequest,
    ToolCallLayer,
    ToolDefinition,
    ToolSideEffect,
    ToolSpec,
    UserMessage,
)
from pygent.core import FrozenJsonObject, JsonValue
from pygent.llm import (
    DefaultModelInvoker,
    ModelInvoker,
    ModelProviderCapabilities,
    ModelProviderClient,
    OpenAICompatibleAdapter,
    OpenAICompatibleClient,
)
from pygent.tool import ExecutorRegistry, LocalToolExecutor

INVALID_ROUTE_ID = "invalid-primary"
VALID_ROUTE_ID = "configured-fallback"
MODEL_GROUP = "live-agent"
TOOL_NAME = "benchmark_add"
TOOL_PERMISSION = "benchmark:add"


@dataclass(frozen=True, slots=True, repr=False)
class LiveAgentConfig:
    """Required external configuration without a printable secret representation."""

    api_base: str
    api_key: str
    model_name: str

    def __post_init__(self) -> None:
        if not self.api_base or not self.api_key or not self.model_name:
            raise ValueError(
                "GLM_API_BASE, GLM_API_KEY, and GLM_MODEL_NAME must be non-empty"
            )
        try:
            parsed = httpx.URL(self.api_base)
        except ValueError as exc:
            raise ValueError("GLM_API_BASE must be an absolute HTTP(S) URL") from exc
        if parsed.scheme not in ("http", "https") or not parsed.host:
            raise ValueError("GLM_API_BASE must be an absolute HTTP(S) URL")
        if parsed.path.rstrip("/").endswith("/chat/completions"):
            raise ValueError(
                "GLM_API_BASE must be an API root, not a completion endpoint"
            )

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str] | None = None
    ) -> LiveAgentConfig:
        values = os.environ if environment is None else environment

        def required(name: str) -> str:
            value = values.get(name, "").strip()
            if not value:
                raise RuntimeError(f"missing required environment variable {name}")
            return value

        return cls(
            api_base=required("GLM_API_BASE"),
            api_key=required("GLM_API_KEY"),
            model_name=required("GLM_MODEL_NAME"),
        )

    def safe_summary(self) -> dict[str, object]:
        """Return only non-sensitive readiness facts."""

        return {
            "api_configured": True,
            "model_configured": True,
            "fallback_configured": True,
        }

    def __repr__(self) -> str:
        return "LiveAgentConfig(api_configured=True, model_configured=True)"


class ProviderConcurrencyTracker:
    """Observe aggregate provider calls without retaining request content."""

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0
        self.calls = 0

    def entered(self) -> None:
        self.active += 1
        self.calls += 1
        self.peak = max(self.peak, self.active)

    def exited(self) -> None:
        self.active -= 1


class _TrackedClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        tracker: ProviderConcurrencyTracker,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._tracker = tracker
        self._http_client = (
            None
            if transport is None
            else httpx.AsyncClient(transport=transport, timeout=None)
        )
        self._client = OpenAICompatibleClient(
            base_url=base_url,
            api_key=api_key,
            client=self._http_client,
        )
        self._closed = False

    async def invoke(
        self, route: ModelRoute, payload: FrozenJsonObject
    ) -> FrozenJsonObject:
        self._tracker.entered()
        try:
            return await self._client.invoke(route, payload)
        finally:
            self._tracker.exited()

    async def stream(
        self, route: ModelRoute, payload: FrozenJsonObject
    ) -> AsyncIterator[FrozenJsonObject]:
        self._tracker.entered()
        try:
            async for item in self._client.stream(route, payload):
                yield item
        finally:
            self._tracker.exited()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._client.aclose()
        if self._http_client is not None:
            await self._http_client.aclose()


class BenchmarkAuthorization(
    Module[ToolAuthorizationRequest, ToolAuthorizationDecision]
):
    """Application-owned authorization for the pure benchmark tool."""

    async def forward(
        self, request: ToolAuthorizationRequest, context: Context
    ) -> tuple[ToolAuthorizationDecision, Context]:
        allowed = TOOL_PERMISSION in request.permissions
        return (
            ToolAuthorizationDecision(
                call_id=request.call.call_id,
                allowed=allowed,
                reason_code="allowed" if allowed else "missing_permission",
            ),
            context,
        )


class LiveBenchmarkAgent(Module[UserMessage, AIMessage]):
    """An ordinary stateless Module that composes a bounded ReAct loop."""

    def __init__(self, react: ReActLayer) -> None:
        super().__init__()
        self.react = react

    async def forward(
        self, message: UserMessage, context: Context
    ) -> tuple[AIMessage, Context]:
        return await self.react(message, context)


@dataclass(slots=True, repr=False)
class LiveAgentResources:
    """Deployment-owned live resources; none are stored on the Agent."""

    agent: LiveBenchmarkAgent
    invoker: ModelInvoker
    registry: ExecutorRegistry
    tracker: ProviderConcurrencyTracker
    tool_definition: ToolDefinition
    _closed: bool = False

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self.invoker, "aclose", None)
        if callable(close):
            await close()


def build_live_agent(
    model_name: str = "configured-at-runtime",
    *,
    model_invoker: ModelInvoker | None = None,
    executor_registry: ExecutorRegistry | None = None,
) -> tuple[LiveBenchmarkAgent, ToolDefinition]:
    definition = ToolDefinition(
        name=TOOL_NAME,
        description="Add two integers. Use this tool exactly once for benchmark prompts.",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"sum": {"type": "integer"}},
            "required": ["sum"],
            "additionalProperties": False,
        },
    )
    spec = ToolSpec(
        tool_id="benchmark.add",
        version="1.0.0",
        definition=definition,
        side_effect=ToolSideEffect.PURE,
        idempotency=IdempotencyPolicy.INHERENT,
        timeout=5.0,
        resource_key="benchmark-cpu",
        required_permissions=(TOOL_PERMISSION,),
    )
    tools = ToolCallLayer(
        tools=(spec,),
        authorization=BenchmarkAuthorization(),
        executor_registry=executor_registry,
        max_concurrency=4,
    )
    model = ModelCallLayer(
        model_group=ModelGroupConfig(
            name=MODEL_GROUP,
            routes=(
                ModelRoute(INVALID_ROUTE_ID, "openai", model_name),
                ModelRoute(VALID_ROUTE_ID, "openai", model_name),
            ),
            fallback=FallbackPolicy((INVALID_ROUTE_ID, VALID_ROUTE_ID)),
            capacity_key="configured-endpoint-model",
        ),
        retry_policy=RetryPolicy(
            max_attempts_per_route=1,
            retry_on=(
                ModelErrorKind.TIMEOUT,
                ModelErrorKind.RATE_LIMIT,
                ModelErrorKind.UNAVAILABLE,
            ),
            backoff=ExponentialBackoff(initial=0.0, maximum=0.0),
        ),
        generation=GenerationConfig(
            temperature=0.0,
            max_output_tokens=256,
            tool_choice="auto",
        ),
        tools=tools.definitions,
        invoker=model_invoker,
    )
    react = ReActLayer(
        model=model,
        tools=tools,
        max_steps=3,
        max_model_calls=3,
        max_tool_calls=2,
    )
    return LiveBenchmarkAgent(react), definition


def build_live_resources(
    config: LiveAgentConfig,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> LiveAgentResources:
    """Build route-specific clients; the primary always gets a fresh invalid key."""

    tracker = ProviderConcurrencyTracker()
    primary = _TrackedClient(
        base_url=config.api_base,
        api_key=f"invalid-{secrets.token_urlsafe(32)}",
        tracker=tracker,
        transport=transport,
    )
    fallback = _TrackedClient(
        base_url=config.api_base,
        api_key=config.api_key,
        tracker=tracker,
        transport=transport,
    )
    invoker = DefaultModelInvoker(
        adapters={"openai": OpenAICompatibleAdapter()},
        clients=cast(
            Mapping[str, ModelProviderClient],
            {INVALID_ROUTE_ID: primary, VALID_ROUTE_ID: fallback},
        ),
        capabilities=(
            {
                INVALID_ROUTE_ID: ModelProviderCapabilities(streaming=False),
                VALID_ROUTE_ID: ModelProviderCapabilities(streaming=False),
            }
            if isinstance(transport, httpx.MockTransport)
            else None
        ),
    )
    registry = ExecutorRegistry()

    def add(arguments: Mapping[str, JsonValue]) -> object:
        a = arguments.get("a")
        b = arguments.get("b")
        if isinstance(a, bool) or not isinstance(a, int):
            raise TypeError("a must be an integer")
        if isinstance(b, bool) or not isinstance(b, int):
            raise TypeError("b must be an integer")
        return {"sum": a + b}

    registry.register(
        "benchmark.add", "1.0.0", LocalToolExecutor(add)
    )
    agent, definition = build_live_agent(config.model_name)
    return LiveAgentResources(agent, invoker, registry, tracker, definition)


def benchmark_context(request_id: str, definition: ToolDefinition) -> Context:
    return Context(
        system_prompt=(
            "You are a benchmark agent. For arithmetic requests, call "
            "benchmark_add exactly once, then answer briefly using its result."
        ),
        tools=(definition,),
        metadata={
            "benchmark_request_id": request_id,
            "permissions": [TOOL_PERMISSION],
        },
    )


def benchmark_message(index: int) -> UserMessage:
    return UserMessage(
        content=f"Call benchmark_add with a={index} and b=7, then give the sum."
    )


__all__ = [
    "INVALID_ROUTE_ID",
    "MODEL_GROUP",
    "TOOL_NAME",
    "TOOL_PERMISSION",
    "VALID_ROUTE_ID",
    "BenchmarkAuthorization",
    "LiveAgentConfig",
    "LiveAgentResources",
    "LiveBenchmarkAgent",
    "ProviderConcurrencyTracker",
    "benchmark_context",
    "benchmark_message",
    "build_live_agent",
    "build_live_resources",
]
