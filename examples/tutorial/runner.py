"""Runnable direct, streaming, live, and managed tutorial paths."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

from pygent import (
    Context,
    FallbackPolicy,
    ModelCallOptions,
    ModelRoute,
    UserMessage,
)
from pygent.llm import ModelResourceOwnership
from pygent.llm.spi import ModelInvoker
from pygent.runtime import (
    CapacityPolicy,
    CapacityScope,
    ExecutionCapacityPolicy,
    ExecutionOptions,
    LocalRuntime,
)

from .agent import (
    TUTORIAL_MODEL_GROUP,
    build_agent,
    build_context,
    deferred_model_group,
    fixed_model_group,
)
from .providers import OfflineModelInvoker


@dataclass(frozen=True, slots=True)
class DemoResult:
    """Small printable snapshot returned by the runnable tutorial paths."""

    answer: str
    context: Context
    event_kinds: tuple[str, ...] = ()


async def run_direct_demo(
    invoker: ModelInvoker,
    *,
    model_name: str = "offline-tutorial",
    stream: bool = False,
) -> DemoResult:
    """Run the tutorial graph directly; the caller owns invoker closure."""

    agent = build_agent(fixed_model_group(model_name), invoker=invoker)
    message = UserMessage(content="Please add 2 and 3.")
    context = build_context()
    if not stream:
        answer, next_context = await agent.invoke(message, context)
        return DemoResult(answer.content, next_context)

    event_kinds: list[str] = []
    async with agent.stream(message, context) as execution_stream:
        async for event in execution_stream:
            event_kinds.append(event.kind)
        answer, next_context = await execution_stream.final_result()
    return DemoResult(answer.content, next_context, tuple(event_kinds))


async def run_managed_demo(
    *,
    selected_profile: str | None = "quality",
) -> tuple[DemoResult, OfflineModelInvoker, OfflineModelInvoker]:
    """Configure immutable profiles and select one for a managed invocation."""

    requirement = deferred_model_group()
    agent = build_agent(requirement)
    runtime = LocalRuntime()
    quick = OfflineModelInvoker("quick")
    quality = OfflineModelInvoker("quality")
    try:
        binding = runtime.create_binding(
            name="tutorial-service",
            execution_capacity=ExecutionCapacityPolicy(
                scope=CapacityScope.RUNTIME_INSTANCE,
                max_live_executions=16,
                max_runnable_executions=4,
                max_queue_size=16,
                max_waiters=16,
                max_child_depth=4,
                max_children_per_execution=8,
            ),
            model_capacity=CapacityPolicy.passthrough(),
            tool_capacity=CapacityPolicy.limited(
                max_concurrency=8,
                max_queue_size=16,
            ),
        )
        bound = binding.bind(agent)
        group = bound.model_groups.get(requirement)
        await group.ensure_profile(
            profile="quick",
            routes=(ModelRoute("primary", "offline", "quick"),),
            fallback=FallbackPolicy(("primary",)),
            invoker=quick,
            ownership=ModelResourceOwnership.OWNED,
            deadline=monotonic() + 5,
        )
        await group.ensure_profile(
            profile="quality",
            routes=(ModelRoute("primary", "offline", "quality"),),
            fallback=FallbackPolicy(("primary",)),
            invoker=quality,
            ownership=ModelResourceOwnership.OWNED,
            deadline=monotonic() + 5,
        )
        await group.set_default("quick", deadline=monotonic() + 5)

        model_calls = (
            {}
            if selected_profile is None
            else {
                TUTORIAL_MODEL_GROUP: ModelCallOptions(
                    profile=selected_profile,
                    temperature=0.1,
                )
            }
        )
        execution = ExecutionOptions(
            deadline=monotonic() + 30.0,
            model_calls=model_calls,
        )
        answer, context = await bound.invoke(
            UserMessage(content="Please add 2 and 3."),
            build_context(),
            execution=execution,
        )
        result = DemoResult(answer.content, context)
    finally:
        await runtime.close()
    return result, quick, quality


async def close_invoker(invoker: ModelInvoker) -> None:
    """Close an optional provider resource after all streams have joined."""

    close = getattr(invoker, "aclose", None)
    if callable(close):
        await close()


__all__ = [
    "DemoResult",
    "close_invoker",
    "run_direct_demo",
    "run_managed_demo",
]
