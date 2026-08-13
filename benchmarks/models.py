"""Deterministic and real success-only model resources for load scenarios."""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import cast

from pygent import (
    AIMessage,
    Context,
    ContextCodec,
    ExponentialBackoff,
    FallbackPolicy,
    GenerationConfig,
    IdempotencyPolicy,
    ModelCallLayer,
    ModelCallPolicy,
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
from pygent.core import FrozenJsonObject, JsonValue, freeze_json_object
from pygent.llm import (
    DefaultModelInvoker,
    ModelProviderCapabilities,
    ModelProviderClient,
    OpenAICompatibleAdapter,
    OpenAICompatibleClient,
)
from pygent.tool import ExecutorRegistry, LocalToolExecutor

from .config import ModelSettings

MODEL_GROUP = "benchmark-success-model"
ROUTE_ID = "success-only"
TOOL_NAME = "benchmark_add"
TOOL_ID = "benchmark.add"
TOOL_PERMISSION = "benchmark:add"


@dataclass(frozen=True, slots=True)
class BenchmarkState:
    request_id: str
    mode: str


@dataclass(frozen=True, slots=True)
class BenchmarkContext(Context):
    context_schema = "pygent.benchmark-context"
    context_schema_version = 1

    benchmark_state: BenchmarkState = BenchmarkState("", "model")


BENCHMARK_CONTEXT_CODEC = ContextCodec.dataclass(BenchmarkContext)


async def _sleep_at_least_ms(duration_ms: float) -> None:
    """Honor synthetic latency even when the event-loop timer wakes early."""

    deadline = perf_counter() + duration_ms / 1000
    while True:
        remaining = deadline - perf_counter()
        if remaining <= 0:
            return
        await asyncio.sleep(remaining)


@dataclass(frozen=True, slots=True, repr=False)
class LiveModelConfig:
    api_base: str
    api_key: str
    model_name: str

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str] | None = None
    ) -> LiveModelConfig:
        values = os.environ if environment is None else environment

        def required(name: str) -> str:
            value = values.get(name, "").strip()
            if not value:
                raise RuntimeError(f"missing required environment variable {name}")
            return value

        return cls(
            required("GLM_API_BASE"),
            required("GLM_API_KEY"),
            required("GLM_MODEL_NAME"),
        )

    def safe_summary(self) -> dict[str, bool]:
        return {"api_configured": True, "model_configured": True}

    def __repr__(self) -> str:
        return "LiveModelConfig(api_configured=True, model_configured=True)"


class ConcurrencyTracker:
    def __init__(self) -> None:
        self.active = 0
        self.peak = 0
        self.calls = 0
        self.durations_ms: dict[int, list[float]] = {}
        self.calls_by_index: dict[int, int] = {}
        self.peak_by_index: dict[int, int] = {}
        self.usage_by_index: dict[int, list[int]] = {}

    def enter(self, index: int) -> None:
        self.active += 1
        self.calls += 1
        self.peak = max(self.peak, self.active)
        self.calls_by_index[index] = self.calls_by_index.get(index, 0) + 1
        self.peak_by_index[index] = max(
            self.peak_by_index.get(index, 0), self.active
        )

    def exit(self) -> None:
        self.active -= 1

    def record(self, index: int, started: float) -> None:
        self.durations_ms.setdefault(index, []).append(
            (perf_counter() - started) * 1000
        )

    def record_usage(self, index: int, payload: FrozenJsonObject) -> None:
        usage = payload.to_dict().get("usage")
        if not isinstance(usage, dict):
            return
        counters = self.usage_by_index.setdefault(index, [0, 0])
        prompt = usage.get("prompt_tokens", 0)
        completion = usage.get("completion_tokens", 0)
        if isinstance(prompt, int) and not isinstance(prompt, bool) and prompt >= 0:
            counters[0] += prompt
        if (
            isinstance(completion, int)
            and not isinstance(completion, bool)
            and completion >= 0
        ):
            counters[1] += completion

    def take(self, index: int) -> tuple[float, int, int, int, int]:
        duration = sum(self.durations_ms.pop(index, ()))
        calls = self.calls_by_index.pop(index, 0)
        peak = self.peak_by_index.pop(index, 0)
        usage = self.usage_by_index.pop(index, [0, 0])
        return duration, calls, peak, usage[0], usage[1]


def request_index(payload: FrozenJsonObject) -> int:
    messages = payload.to_dict().get("messages", [])
    if not isinstance(messages, list):
        return 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        match = re.search(r"(?:a=|request )(\d+)", content)
        if match:
            return int(match.group(1))
    return 0


class SyntheticSuccessClient:
    """OpenAI-compatible client with deterministic successful responses."""

    def __init__(self, settings: ModelSettings, *, seed: int) -> None:
        self.settings = settings
        self.random = random.Random(seed)
        self.tracker = ConcurrencyTracker()
        self.closed = False
        self._random_lock = asyncio.Lock()

    async def _delay(self, *, streaming: bool = False) -> None:
        async with self._random_lock:
            jitter = self.random.uniform(-self.settings.jitter_ms, self.settings.jitter_ms)
        base = self.settings.ttft_ms if streaming else self.settings.latency_ms
        await _sleep_at_least_ms(max(0.0, base + jitter))

    @staticmethod
    def _tool_request(payload: FrozenJsonObject) -> tuple[bool, int]:
        body = payload.to_dict()
        messages = cast(list[dict[str, object]], body.get("messages", []))
        last = messages[-1] if messages else {}
        wants_tool = bool(body.get("tools")) and last.get("role") != "tool"
        return wants_tool, request_index(payload)

    async def invoke(
        self, route: ModelRoute, payload: FrozenJsonObject
    ) -> FrozenJsonObject:
        del route
        index = request_index(payload)
        started = perf_counter()
        self.tracker.enter(index)
        try:
            await self._delay()
            wants_tool, index = self._tool_request(payload)
            message: dict[str, object]
            if wants_tool:
                message = {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call-{index}",
                            "type": "function",
                            "function": {
                                "name": TOOL_NAME,
                                "arguments": json.dumps({"a": index, "b": 7}),
                            },
                        }
                    ],
                }
            else:
                message = {"content": "done"}
            response = freeze_json_object(
                {
                    "id": f"synthetic-{index}",
                    "choices": [{"message": message}],
                    "usage": {
                        "prompt_tokens": 8,
                        "completion_tokens": 2,
                        "total_tokens": 10,
                    },
                }
            )
            self.tracker.record_usage(index, response)
            return response
        finally:
            self.tracker.exit()
            self.tracker.record(index, started)

    async def stream(
        self, route: ModelRoute, payload: FrozenJsonObject
    ) -> AsyncIterator[FrozenJsonObject]:
        del route
        index = request_index(payload)
        started = perf_counter()
        self.tracker.enter(index)
        try:
            await self._delay(streaming=True)
            wants_tool, index = self._tool_request(payload)
            if wants_tool:
                yield freeze_json_object(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": f"call-{index}",
                                            "function": {
                                                "name": TOOL_NAME,
                                                "arguments": json.dumps(
                                                    {"a": index, "b": 7}
                                                ),
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                )
            else:
                chunks = max(1, self.settings.chunks)
                parts = ["done"] + [""] * (chunks - 1)
                remaining = max(0.0, self.settings.latency_ms - self.settings.ttft_ms)
                for part in parts:
                    if remaining:
                        await _sleep_at_least_ms(remaining / chunks)
                    if part:
                        yield freeze_json_object(
                            {"choices": [{"delta": {"content": part}}]}
                        )
            usage = freeze_json_object(
                {
                    "usage": {
                        "prompt_tokens": 8,
                        "completion_tokens": 2,
                        "total_tokens": 10,
                    }
                }
            )
            self.tracker.record_usage(index, usage)
            yield usage
            yield freeze_json_object({"done": True})
        finally:
            self.tracker.exit()
            self.tracker.record(index, started)

    async def aclose(self) -> None:
        self.closed = True


class TrackedClient:
    def __init__(self, client: ModelProviderClient) -> None:
        self.client = client
        self.tracker = ConcurrencyTracker()

    async def invoke(
        self, route: ModelRoute, payload: FrozenJsonObject
    ) -> FrozenJsonObject:
        index = request_index(payload)
        started = perf_counter()
        self.tracker.enter(index)
        try:
            response = await self.client.invoke(route, payload)
            self.tracker.record_usage(index, response)
            return response
        finally:
            self.tracker.exit()
            self.tracker.record(index, started)

    async def stream(
        self, route: ModelRoute, payload: FrozenJsonObject
    ) -> AsyncIterator[FrozenJsonObject]:
        index = request_index(payload)
        started = perf_counter()
        self.tracker.enter(index)
        try:
            async for event in self.client.stream(route, payload):
                self.tracker.record_usage(index, event)
                yield event
        finally:
            self.tracker.exit()
            self.tracker.record(index, started)

    async def aclose(self) -> None:
        await self.client.aclose()


class AllowBenchmarkTool(Module[ToolAuthorizationRequest, ToolAuthorizationDecision]):
    async def forward(
        self, message: ToolAuthorizationRequest, context: Context
    ) -> tuple[ToolAuthorizationDecision, Context]:
        allowed = TOOL_PERMISSION in message.permissions
        return (
            ToolAuthorizationDecision(
                call_id=message.call.call_id,
                allowed=allowed,
                reason_code="allowed" if allowed else "permission_missing",
            ),
            context,
        )


class BenchmarkAgent(Module[UserMessage, AIMessage]):
    def __init__(self, react: ReActLayer) -> None:
        super().__init__()
        self.react = react

    async def forward(
        self, message: UserMessage, context: Context
    ) -> tuple[AIMessage, Context]:
        return await self.react(message, context)


@dataclass(slots=True)
class ModelResources:
    model: ModelCallLayer
    agent: BenchmarkAgent
    definition: ToolDefinition
    registry: ExecutorRegistry
    invoker: DefaultModelInvoker
    tracker: ConcurrencyTracker
    tool_durations_ms: dict[int, list[float]]
    configured_group: ModelGroupConfig

    async def aclose(self) -> None:
        await self.invoker.aclose()


def _tool_resources(
    settings: ModelSettings,
) -> tuple[
    ToolDefinition,
    ToolCallLayer,
    ExecutorRegistry,
    dict[int, list[float]],
]:
    definition = ToolDefinition(
        name=TOOL_NAME,
        description="Add two integers exactly once.",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
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
        tool_id=TOOL_ID,
        version="1.0.0",
        definition=definition,
        side_effect=ToolSideEffect.PURE,
        idempotency=IdempotencyPolicy.INHERENT,
        timeout=10.0,
        resource_key="benchmark-tool",
        required_permissions=(TOOL_PERMISSION,),
    )
    registry = ExecutorRegistry()
    durations: dict[int, list[float]] = {}

    async def add(arguments: Mapping[str, JsonValue]) -> object:
        index = int(cast(int, arguments["a"]))
        started = perf_counter()
        if settings.tool_latency_ms:
            await _sleep_at_least_ms(settings.tool_latency_ms)
        durations.setdefault(index, []).append((perf_counter() - started) * 1000)
        return {"sum": index + int(cast(int, arguments["b"]))}

    registry.register(TOOL_ID, "1.0.0", LocalToolExecutor(add))
    layer = ToolCallLayer(
        tools=(spec,),
        authorization=AllowBenchmarkTool(),
        executor_registry=registry,
        max_concurrency=32,
    )
    return definition, layer, registry, durations


def build_resources(
    *,
    backend: str,
    settings: ModelSettings,
    seed: int,
    streaming: bool,
    dynamic_model: bool = False,
    live: LiveModelConfig | None = None,
) -> ModelResources:
    if backend == "synthetic":
        client: ModelProviderClient = SyntheticSuccessClient(settings, seed=seed)
        tracker = cast(SyntheticSuccessClient, client).tracker
        model_name = "synthetic-model"
    elif backend == "live":
        if live is None:
            raise ValueError("live backend requires LiveModelConfig")
        tracked = TrackedClient(
            OpenAICompatibleClient(
                base_url=live.api_base,
                api_key=live.api_key,
            )
        )
        client = tracked
        tracker = tracked.tracker
        model_name = live.model_name
    else:
        raise ValueError(f"unsupported backend {backend!r}")

    invoker = DefaultModelInvoker(
        adapters={"openai": OpenAICompatibleAdapter()},
        clients=cast(Mapping[str, ModelProviderClient], {ROUTE_ID: client}),
        capabilities={"openai": ModelProviderCapabilities(streaming=streaming)},
    )
    definition, tools, registry, tool_durations = _tool_resources(settings)
    configured_group = ModelGroupConfig(
        name=MODEL_GROUP,
        routes=(ModelRoute(ROUTE_ID, "openai", model_name),),
        fallback=FallbackPolicy((ROUTE_ID,)),
        capacity_key="benchmark-model-resource",
    )
    model_group = (
        ModelGroupConfig.deferred(
            name=MODEL_GROUP,
            capacity_key="benchmark-model-resource",
        )
        if dynamic_model
        else configured_group
    )
    model_policy = ModelCallPolicy(allow_profile_override=dynamic_model)
    retry = RetryPolicy(
        max_attempts_per_route=settings.retry_max_attempts,
        retry_on=tuple(ModelErrorKind(kind) for kind in settings.retry_on),
        backoff=ExponentialBackoff(
            settings.retry_backoff_seconds,
            settings.retry_backoff_seconds,
        ),
        attempt_timeout_seconds=settings.attempt_timeout_seconds,
    )
    model = ModelCallLayer(
        model_group=model_group,
        retry_policy=retry,
        generation=GenerationConfig(
            temperature=0.0,
            max_output_tokens=64,
        ),
        policy=model_policy,
        invoker=None if dynamic_model else invoker,
    )
    tool_model = ModelCallLayer(
        model_group=model_group,
        retry_policy=retry,
        generation=GenerationConfig(
            temperature=0.0,
            max_output_tokens=64,
            tool_choice="auto",
        ),
        policy=model_policy,
        tools=tools.definitions,
        invoker=None if dynamic_model else invoker,
    )
    agent = BenchmarkAgent(
        ReActLayer(
            model=tool_model,
            tools=tools,
            max_steps=3,
            max_model_calls=3,
            max_tool_calls=2,
        )
    )
    return ModelResources(
        model,
        agent,
        definition,
        registry,
        invoker,
        tracker,
        tool_durations,
        configured_group,
    )


def model_context(request_id: str) -> Context:
    return BenchmarkContext(
        metadata={"benchmark_request_id": request_id},
        benchmark_state=BenchmarkState(request_id, "model"),
    )


def agent_context(request_id: str, definition: ToolDefinition) -> Context:
    return BenchmarkContext(
        system_prompt="Call benchmark_add exactly once for arithmetic, then answer briefly.",
        tools=(definition,),
        metadata={
            "benchmark_request_id": request_id,
            "permissions": [TOOL_PERMISSION],
        },
        benchmark_state=BenchmarkState(request_id, "agent"),
    )


def model_message(index: int) -> UserMessage:
    return UserMessage(content=f"Return the word done for request {index}.")


def agent_message(index: int) -> UserMessage:
    return UserMessage(content=f"Call benchmark_add with a={index} and b=7 then answer.")


__all__ = [
    "BENCHMARK_CONTEXT_CODEC",
    "MODEL_GROUP",
    "BenchmarkAgent",
    "ConcurrencyTracker",
    "LiveModelConfig",
    "ModelResources",
    "agent_context",
    "agent_message",
    "build_resources",
    "model_context",
    "model_message",
]
