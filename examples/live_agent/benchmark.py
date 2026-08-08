"""Managed Runtime concurrency benchmark for the opt-in live Agent."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from time import monotonic, perf_counter
from typing import Any, cast
from uuid import uuid4

from pygent import (
    Context,
    ToolMessage,
    UserMessage,
)
from pygent.core import FrozenJsonObject, JsonValue, thaw_json
from pygent.runtime import (
    CapacityPolicy,
    CapacityScope,
    ExecutionCapacityPolicy,
    ExecutionEvent,
    ExecutionOptions,
    LocalRuntime,
)

from .agent import (
    INVALID_ROUTE_ID,
    MODEL_GROUP,
    VALID_ROUTE_ID,
    LiveAgentConfig,
    LiveAgentResources,
    benchmark_context,
    benchmark_message,
    build_live_resources,
)


@dataclass(frozen=True, slots=True)
class ExecutionSample:
    request_id: str
    latency_seconds: float
    succeeded: bool
    events: tuple[ExecutionEvent, ...]
    context_isolated: bool
    failure_type: str | None = None
    tool_statuses: tuple[str, ...] = ()


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _tool_sum(value: JsonValue) -> object:
    decoded = thaw_json(value)
    return decoded.get("sum") if isinstance(decoded, dict) else None


def _failure_context_isolated(
    context: Context,
    request_id: str,
    events: tuple[ExecutionEvent, ...],
    execution_id: str | None,
) -> bool:
    metadata = cast(FrozenJsonObject, context.metadata)
    if metadata.get("benchmark_request_id") != request_id:
        return False
    if execution_id is not None and any(
        event.execution_id != execution_id for event in events
    ):
        return False
    started = tuple(
        event
        for event in events
        if event.kind == "execution.started" and (execution_id is None or event.execution_id == execution_id)
    )
    return not started or all(
        cast(FrozenJsonObject, event.data).get("request_id") == request_id
        for event in started
    )


def aggregate_samples(
    samples: Sequence[ExecutionSample],
    *,
    wall_seconds: float,
    peak_model_inflight: int,
    provider_calls: int,
) -> dict[str, object]:
    """Aggregate portable metrics without retaining prompts, outputs, or secrets."""

    latencies_ms = [sample.latency_seconds * 1000 for sample in samples]
    attempt_kinds: Counter[str] = Counter()
    usage: Counter[str] = Counter()
    fallback_count = 0
    tool_statuses: Counter[str] = Counter()
    failure_types: Counter[str] = Counter()
    for sample in samples:
        kinds = [event.kind for event in sample.events]
        primary_failed = False
        fallback_succeeded = False
        for event in sample.events:
            event_data = cast(FrozenJsonObject, event.data)
            if event.kind == "model.attempt.failed":
                route = event_data.get("route_id")
                kind = event_data.get("error_kind")
                if isinstance(kind, str):
                    attempt_kinds[kind] += 1
                primary_failed = primary_failed or route == INVALID_ROUTE_ID
            elif event.kind == "model.attempt.succeeded":
                fallback_succeeded = (
                    fallback_succeeded
                    or event_data.get("route_id") == VALID_ROUTE_ID
                )
            elif event.kind == "model.usage":
                for event_key, report_key in (
                    ("input_tokens", "prompt_tokens"),
                    ("output_tokens", "completion_tokens"),
                    ("total_tokens", "total_tokens"),
                ):
                    value = event_data.get(event_key)
                    if isinstance(value, int) and not isinstance(value, bool):
                        usage[report_key] += value
        if primary_failed and fallback_succeeded:
            fallback_count += 1
        tool_statuses.update(sample.tool_statuses)
        if sample.failure_type is not None:
            failure_types[sample.failure_type] += 1
        if kinds and any(
            event.sequence != index for index, event in enumerate(sample.events)
        ):
            failure_types["non_contiguous_events"] += 1

    completed = len(samples)
    succeeded = sum(sample.succeeded for sample in samples)
    mean = sum(latencies_ms) / completed if completed else 0.0
    return {
        "requested": completed,
        "completed": completed,
        "succeeded": succeeded,
        "failed": completed - succeeded,
        "wall_seconds": round(wall_seconds, 6),
        "throughput_rps": round(completed / wall_seconds, 6)
        if wall_seconds > 0
        else 0.0,
        "latency_ms": {
            "min": round(min(latencies_ms), 3) if latencies_ms else 0.0,
            "mean": round(mean, 3),
            "p50": round(_percentile(latencies_ms, 0.50), 3),
            "p95": round(_percentile(latencies_ms, 0.95), 3),
            "p99": round(_percentile(latencies_ms, 0.99), 3),
            "max": round(max(latencies_ms), 3) if latencies_ms else 0.0,
        },
        "peak_model_inflight": peak_model_inflight,
        "provider_calls": provider_calls,
        "fallback_count": fallback_count,
        "fallback_rate": round(fallback_count / completed, 6) if completed else 0.0,
        "attempt_failures_by_kind": dict(sorted(attempt_kinds.items())),
        "usage": dict(sorted(usage.items())),
        "tool_statuses": dict(sorted(tool_statuses.items())),
        "context_isolation_failures": sum(
            not sample.context_isolated for sample in samples
        ),
        "failure_types": dict(sorted(failure_types.items())),
    }


async def execute_benchmark(
    resources: LiveAgentResources,
    *,
    requests: int,
    concurrency: int,
    model_concurrency: int,
    deadline_seconds: float,
) -> dict[str, object]:
    for name, value in (
        ("requests", requests),
        ("concurrency", concurrency),
        ("model_concurrency", model_concurrency),
    ):
        if isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if deadline_seconds <= 0 or not math.isfinite(deadline_seconds):
        raise ValueError("deadline_seconds must be a positive finite number")

    runtime = LocalRuntime()
    runtime.register_model_invoker(MODEL_GROUP, resources.invoker)
    runtime.attach_executor_registry(resources.registry)
    binding = runtime.create_binding(
        name=f"live-agent-benchmark-{uuid4().hex}",
        execution_capacity=ExecutionCapacityPolicy(
            scope=CapacityScope.RUNTIME_INSTANCE,
            max_live_executions=concurrency,
            max_runnable_executions=concurrency,
            max_queue_size=requests,
            max_waiters=requests,
            max_child_depth=8,
            max_children_per_execution=16,
        ),
        model_capacity=CapacityPolicy.limited(
            max_concurrency=model_concurrency,
            max_queue_size=requests,
        ),
        tool_capacity=CapacityPolicy.limited(
            max_concurrency=concurrency,
            max_queue_size=requests,
        ),
    )
    bound = binding.bind(resources.agent)

    async def one(index: int) -> ExecutionSample:
        request_id = f"benchmark-{index}-{uuid4().hex}"
        context = benchmark_context(request_id, resources.tool_definition)
        started = perf_counter()
        events: tuple[ExecutionEvent, ...] = ()
        handle = None
        try:
            handle = await bound.start(
                benchmark_message(index),
                context,
                execution=ExecutionOptions(
                    request_id=request_id,
                    identity="live-benchmark",
                    context_ref=f"benchmark:{request_id}",
                    deadline=monotonic() + deadline_seconds,
                ),
            )
            _, next_context = await handle.result()
            async with handle.subscribe() as subscription:
                events = tuple([event async for event in subscription])
            statuses = tuple(
                result.status
                for message in next_context.messages
                if isinstance(message, ToolMessage)
                for result in message.results
            )
            user_contents = tuple(
                item.content
                for item in next_context.messages
                if isinstance(item, UserMessage)
            )
            expected_content = benchmark_message(index).content
            tool_sums = tuple(
                _tool_sum(result.output)
                for item in next_context.messages
                if isinstance(item, ToolMessage)
                for result in item.results
            )
            isolated = (
                cast(FrozenJsonObject, next_context.metadata).get(
                    "benchmark_request_id"
                )
                == request_id
                and user_contents == (expected_content,)
                and all(value == index + 7 for value in tool_sums)
            )
            return ExecutionSample(
                request_id,
                perf_counter() - started,
                True,
                events,
                isolated,
                tool_statuses=statuses,
            )
        except Exception as exc:  # noqa: BLE001 - benchmark result boundary
            if handle is not None:
                async with handle.subscribe() as subscription:
                    events = tuple([event async for event in subscription])
            return ExecutionSample(
                request_id,
                perf_counter() - started,
                False,
                events,
                _failure_context_isolated(
                    context,
                    request_id,
                    events,
                    None if handle is None else handle.execution_id,
                ),
                failure_type=type(exc).__name__,
            )

    wall_started = perf_counter()
    try:
        samples = await asyncio.gather(*(one(index) for index in range(requests)))
        wall_seconds = perf_counter() - wall_started
        result = aggregate_samples(
            samples,
            wall_seconds=wall_seconds,
            peak_model_inflight=resources.tracker.peak,
            provider_calls=resources.tracker.calls,
        )
        result.update(
            {
                "concurrency": concurrency,
                "model_concurrency": model_concurrency,
            }
        )
        return result
    finally:
        await runtime.close(cancel=True)


async def run_live_benchmark(
    config: LiveAgentConfig,
    *,
    requests: int,
    concurrency: int,
    model_concurrency: int,
    deadline_seconds: float,
) -> dict[str, object]:
    resources = build_live_resources(config)
    try:
        metrics = await execute_benchmark(
            resources,
            requests=requests,
            concurrency=concurrency,
            model_concurrency=model_concurrency,
            deadline_seconds=deadline_seconds,
        )
        return {"configuration": config.safe_summary(), "metrics": metrics}
    finally:
        await resources.aclose()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the opt-in Pygent live Agent concurrency benchmark."
    )
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--model-concurrency", type=int, default=2)
    parser.add_argument("--deadline-seconds", type=float, default=60.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = LiveAgentConfig.from_environment()
    report = asyncio.run(
        run_live_benchmark(
            config,
            requests=args.requests,
            concurrency=args.concurrency,
            model_concurrency=args.model_concurrency,
            deadline_seconds=args.deadline_seconds,
        )
    )
    print(json.dumps(cast(dict[str, Any], report), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ExecutionSample",
    "aggregate_samples",
    "execute_benchmark",
    "main",
    "run_live_benchmark",
]
