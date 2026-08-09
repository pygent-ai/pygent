"""End-to-end direct, managed, durable, Tool, and HTTP Worker scenarios."""

from __future__ import annotations

import asyncio
import hashlib
import tempfile
from collections.abc import Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, perf_counter
from typing import Any, Self, cast
from uuid import uuid4

import httpx

from pygent import (
    AIMessage,
    Context,
    ModelCallError,
    Module,
    ToolMessage,
    UserMessage,
)
from pygent.core import (
    ExecutionFailureError,
    FrozenJsonObject,
    JsonValue,
    freeze_json,
    thaw_json,
)
from pygent.runtime import (
    CapacityPolicy,
    CapacityScope,
    ExecutionCapacityPolicy,
    ExecutionDeadlineExceeded,
    ExecutionOptions,
    HTTPWorkerApp,
    HTTPWorkerClient,
    LocalRuntime,
    SQLiteHistoryStore,
    WorkerOutcomeUnknownError,
    WorkerProtocolError,
    WorkerRegistry,
    WorkerRemoteError,
    WorkerTarget,
    WorkerUnavailableError,
)

from .config import LoadProfile
from .metrics import Sample
from .models import (
    MODEL_GROUP,
    LiveModelConfig,
    ModelResources,
    agent_context,
    agent_message,
    build_resources,
    model_context,
    model_message,
)

SUPPORTED_SCENARIOS = (
    "direct-invoke",
    "direct-stream",
    "local-invoke",
    "local-stream",
    "dynamic-model-invoke",
    "dynamic-model-stream",
    "react-tool-invoke",
    "react-tool-stream",
    "sqlite-durable-invoke",
    "sqlite-durable-stream",
    "http-worker-invoke",
    "http-worker-stream",
    "lifecycle-cancel-deadline",
)


class ExternalEnvironmentInvalid(RuntimeError):
    """The live provider violated the benchmark's success-only precondition."""

    def __init__(
        self,
        message: str,
        *,
        error_domain: str | None = None,
        error_kind: str | None = None,
        sample: Sample | None = None,
    ) -> None:
        super().__init__(message)
        self.error_domain = error_domain
        self.error_kind = error_kind
        self.sample = sample


def _failure_attempt_count(details: object) -> int:
    if not isinstance(details, Mapping):
        return 0
    attempts = details.get("attempts", ())
    return len(attempts) if isinstance(attempts, (list, tuple)) else 0


def _classify_error(exc: Exception) -> tuple[str, str | None, str, int]:
    """Return sanitized type/domain/kind and observed model attempts."""
    if isinstance(exc, ModelCallError):
        return type(exc).__name__, "model", exc.kind.value, len(exc.attempts)
    if isinstance(exc, ExecutionFailureError):
        return (
            type(exc).__name__,
            exc.failure.domain,
            exc.failure.kind,
            _failure_attempt_count(exc.failure.details),
        )
    if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
        return type(exc).__name__, None, "timeout", 0
    if isinstance(exc, WorkerOutcomeUnknownError):
        return type(exc).__name__, "worker", "outcome_unknown", 0
    if isinstance(exc, WorkerUnavailableError):
        return type(exc).__name__, "worker", "unavailable", 0
    if isinstance(exc, WorkerRemoteError):
        return (
            type(exc).__name__,
            exc.failure.domain,
            exc.failure.kind,
            _failure_attempt_count(exc.failure.details),
        )
    if isinstance(exc, WorkerProtocolError):
        return type(exc).__name__, "worker", "worker_protocol", 0
    if isinstance(exc, httpx.TransportError):
        return type(exc).__name__, None, "unavailable", 0
    if isinstance(exc, ExecutionDeadlineExceeded):
        return type(exc).__name__, "execution", "timeout", 0
    return type(exc).__name__, None, "unknown", 0


class LifecycleModule(Module[UserMessage, AIMessage]):
    async def forward(
        self, message: UserMessage, context: Context
    ) -> tuple[AIMessage, Context]:
        await asyncio.sleep(0.05)
        return AIMessage(content="completed"), context


@dataclass(frozen=True, slots=True)
class _ModelTrace:
    trace_id: str | None
    consistent: bool
    prompt_tokens: int
    completion_tokens: int
    model_spans: int
    model_attempts: int
    model_ms: float | None
    ttft_ms: float | None


def _model_trace(events: list[Any]) -> _ModelTrace:
    """Reduce public trace events without double-counting cumulative usage."""

    if not events:
        return _ModelTrace(None, False, 0, 0, 0, 0, None, None)
    trace_ids = {getattr(event, "trace_id", None) for event in events}
    event_ids = [getattr(event, "event_id", None) for event in events]
    span_ids = {getattr(event, "span_id", None) for event in events}
    parents = {
        getattr(event, "parent_span_id", None)
        for event in events
        if getattr(event, "parent_span_id", None) is not None
    }
    consistent = (
        len(trace_ids) == 1
        and None not in trace_ids
        and all(isinstance(value, str) and value for value in event_ids)
        and len(set(event_ids)) == len(event_ids)
        and None not in span_ids
        and parents <= span_ids
        and _ordered(events)
    )

    usage_by_attempt: dict[tuple[str, str, int], tuple[int, int]] = {}
    model_started: dict[str, int] = {}
    model_durations_ns: list[int] = []
    first_output_ns: int | None = None
    execution_started_ns: int | None = None
    model_attempts = 0
    for event in events:
        kind = getattr(event, "kind", "")
        timestamp = getattr(event, "timestamp_unix_ns", None)
        span_id = getattr(event, "span_id", None)
        data = getattr(event, "data", {})
        if kind == "execution.started" and isinstance(timestamp, int):
            execution_started_ns = timestamp
        elif (
            kind == "model.started"
            and isinstance(span_id, str)
            and isinstance(timestamp, int)
        ):
            model_started[span_id] = timestamp
        elif kind == "model.attempt.started":
            model_attempts += 1
        elif kind in ("model.completed", "model.failed", "model.cancelled"):
            started = model_started.get(span_id) if isinstance(span_id, str) else None
            if (
                started is not None
                and isinstance(timestamp, int)
                and timestamp >= started
            ):
                model_durations_ns.append(timestamp - started)
        elif kind in (
            "model.reasoning.delta",
            "model.text.delta",
            "model.tool_call.started",
        ) and isinstance(timestamp, int):
            first_output_ns = (
                timestamp
                if first_output_ns is None
                else min(first_output_ns, timestamp)
            )
        if kind != "model.usage" or not isinstance(span_id, str):
            continue
        route_id = data.get("route_id")
        attempt = data.get("attempt")
        if (
            not isinstance(route_id, str)
            or not isinstance(attempt, int)
            or isinstance(attempt, bool)
        ):
            consistent = False
            continue
        prompt = data.get("input_tokens")
        completion = data.get("output_tokens")
        usage_by_attempt[(span_id, route_id, attempt)] = (
            prompt if isinstance(prompt, int) and not isinstance(prompt, bool) else 0,
            completion
            if isinstance(completion, int) and not isinstance(completion, bool)
            else 0,
        )
    prompt_tokens = sum(value[0] for value in usage_by_attempt.values())
    completion_tokens = sum(value[1] for value in usage_by_attempt.values())
    ttft_ms = None
    if (
        execution_started_ns is not None
        and first_output_ns is not None
        and first_output_ns >= execution_started_ns
    ):
        ttft_ms = (first_output_ns - execution_started_ns) / 1_000_000
    return _ModelTrace(
        next(iter(trace_ids)) if len(trace_ids) == 1 else None,
        consistent,
        prompt_tokens,
        completion_tokens,
        len(model_started),
        model_attempts,
        sum(model_durations_ns) / 1_000_000 if model_durations_ns else None,
        ttft_ms,
    )


def _ordered(events: list[Any]) -> bool:
    indexes = [
        int(getattr(event, "sequence", index)) for index, event in enumerate(events)
    ]
    return indexes == list(range(len(indexes)))


def _context_isolated(context: Context, request_id: str, *, tool: bool) -> bool:
    metadata = cast(FrozenJsonObject, context.metadata)
    if metadata.get("benchmark_request_id") != request_id:
        return False
    if not tool:
        return True
    results = [
        result
        for message in context.messages
        if isinstance(message, ToolMessage)
        for result in message.results
    ]
    return len(results) == 1 and results[0].status == "succeeded"


class ScenarioSession:
    def __init__(
        self,
        profile: LoadProfile,
        scenario: str,
        *,
        capacity: int,
        worker_count: int = 1,
    ) -> None:
        if scenario not in SUPPORTED_SCENARIOS:
            raise ValueError(f"unsupported scenario {scenario!r}")
        self.profile = profile
        self.scenario = scenario
        self.capacity = capacity
        self.worker_count = worker_count
        self.live = (
            LiveModelConfig.from_environment() if profile.backend == "live" else None
        )
        self.resources: ModelResources | None = None
        self.runtime: LocalRuntime | None = None
        self.bound: Any = None
        self.stack = AsyncExitStack()
        self.temporary: tempfile.TemporaryDirectory[str] | None = None
        self.sqlite_paths: list[Path] = []
        self.worker_apps: list[HTTPWorkerApp] = []
        self.worker_clients: list[HTTPWorkerClient] = []
        self.worker_runtimes: list[LocalRuntime] = []
        self.worker_bounds: list[Any] = []
        self._closed = False

    async def __aenter__(self) -> Self:
        streaming = self.scenario.endswith("-stream")
        dynamic_model = self.scenario.startswith("dynamic-model-")
        self.resources = build_resources(
            backend=self.profile.backend,
            settings=self.profile.model,
            seed=self.profile.seed,
            streaming=streaming,
            dynamic_model=dynamic_model,
            live=self.live,
        )
        if self.scenario == "lifecycle-cancel-deadline":
            if self.profile.backend != "synthetic":
                raise ValueError("lifecycle pressure is synthetic-only")
            self.runtime = LocalRuntime()
            self.bound = self._binding(self.runtime, 1).bind(LifecycleModule())
            return self
        if self.scenario.startswith("direct-"):
            return self
        if self.scenario.startswith("http-worker-"):
            await self._open_workers()
            return self
        history = None
        if self.scenario.startswith("sqlite-durable-"):
            self.temporary = tempfile.TemporaryDirectory(prefix="pygent-benchmark-")
            path = Path(self.temporary.name) / "runtime.sqlite3"
            self.sqlite_paths.append(path)
            history = await self.stack.enter_async_context(SQLiteHistoryStore(path))
        self.runtime = LocalRuntime(history=history)
        self._configure_runtime(self.runtime, self.resources)
        module = (
            self.resources.agent
            if self.scenario.startswith("react-tool-")
            or self.scenario.startswith("sqlite-durable-")
            else self.resources.model
        )
        self.bound = self._binding(self.runtime, self.capacity).bind(module)
        if dynamic_model:
            await self._configure_dynamic_profiles()
        return self

    async def _configure_dynamic_profiles(self) -> None:
        assert self.resources is not None
        requirement = self.resources.model.model_group
        group = self.bound.model_groups.get(requirement)
        configured = self.resources.configured_group
        for profile in ("default", "alternate"):
            await group.ensure_profile(
                profile=profile,
                routes=configured.routes,
                fallback=configured.fallback,
                invoker=self.resources.invoker,
                deadline=monotonic() + 5,
            )
        await group.set_default("default", deadline=monotonic() + 5)

    def _configure_runtime(
        self, runtime: LocalRuntime, resources: ModelResources
    ) -> None:
        if not resources.model.model_group.is_deferred:
            runtime.register_model_invoker(MODEL_GROUP, resources.invoker)
        runtime.attach_executor_registry(resources.registry)

    def _binding(self, runtime: LocalRuntime, capacity: int):
        queue = max(capacity * 4, self.profile.max_inflight)
        return runtime.create_binding(
            name=f"benchmark-{uuid4().hex}",
            execution_capacity=ExecutionCapacityPolicy(
                scope=CapacityScope.RUNTIME_INSTANCE,
                max_live_executions=capacity,
                max_runnable_executions=capacity,
                max_queue_size=queue,
                max_waiters=queue,
                max_child_depth=8,
                max_children_per_execution=16,
            ),
            model_capacity=CapacityPolicy.limited(
                max_concurrency=capacity, max_queue_size=queue
            ),
            tool_capacity=CapacityPolicy.limited(
                max_concurrency=capacity, max_queue_size=queue
            ),
        )

    async def _open_workers(self) -> None:
        assert self.resources is not None
        resources = self.resources
        self.temporary = tempfile.TemporaryDirectory(prefix="pygent-workers-")
        plan_hash = hashlib.sha256(b"pygent-benchmark-plan-v1").hexdigest()
        self._plan_id = f"sha256:{plan_hash}"
        self._graph_hash = plan_hash
        for index in range(self.worker_count):
            path = Path(self.temporary.name) / f"worker-{index}.sqlite3"
            runtime_path = Path(self.temporary.name) / f"worker-runtime-{index}.sqlite3"
            self.sqlite_paths.extend((path, runtime_path))
            history = await self.stack.enter_async_context(SQLiteHistoryStore(path))
            runtime_history = await self.stack.enter_async_context(
                SQLiteHistoryStore(runtime_path)
            )
            runtime = LocalRuntime(history=runtime_history)
            self._configure_runtime(runtime, self.resources)
            bound = self._binding(runtime, self.capacity).bind(self.resources.agent)

            async def handler(
                invocation: Any,
                event_sink: Any,
                *,
                worker_index: int = index,
                worker_bound: Any = bound,
            ) -> JsonValue:
                payload = cast(FrozenJsonObject, invocation.input)
                request_id = cast(str, payload["request_id"])
                item_index = cast(int, payload["index"])
                context = agent_context(request_id, resources.definition)
                events: list[Any] = []
                if self.scenario.endswith("-stream"):
                    async with worker_bound.stream(
                        agent_message(item_index),
                        context,
                        execution=ExecutionOptions(
                            request_id=request_id,
                            idempotency_key=request_id,
                            deadline=monotonic()
                            + self.profile.request_deadline_seconds,
                        ),
                    ) as stream:
                        async for event in stream:
                            events.append(event)
                            await event_sink(event)
                        _, next_context = await stream.final_result()
                else:
                    handle = await worker_bound.start(
                        agent_message(item_index),
                        context,
                        execution=ExecutionOptions(
                            request_id=request_id,
                            idempotency_key=request_id,
                            deadline=monotonic()
                            + self.profile.request_deadline_seconds,
                        ),
                    )

                    async def relay() -> None:
                        async with handle.subscribe() as subscription:
                            async for event in subscription:
                                events.append(event)
                                await event_sink(event)

                    relay_task = asyncio.create_task(
                        relay(), name="benchmark-worker-event-relay"
                    )
                    try:
                        _, next_context = await handle.result()
                    except BaseException:
                        relay_task.cancel()
                        await asyncio.gather(relay_task, return_exceptions=True)
                        raise
                    else:
                        await relay_task
                trace = _model_trace(events)
                model_ms, provider_calls, model_peak, _, _ = resources.tracker.take(
                    item_index
                )
                prompt, completion = trace.prompt_tokens, trace.completion_tokens
                tool_ms = sum(resources.tool_durations_ms.pop(item_index, ()))
                return freeze_json(
                    {
                        "worker_id": f"worker-{worker_index}",
                        "context_isolated": _context_isolated(
                            next_context, request_id, tool=True
                        ),
                        "event_count": len(events),
                        "events_ordered": _ordered(events),
                        "prompt_tokens": prompt,
                        "completion_tokens": completion,
                        "model_ms": model_ms,
                        "provider_calls": provider_calls,
                        "model_peak_inflight": model_peak,
                        "tool_ms": tool_ms,
                        "trace_id": trace.trace_id,
                        "trace_consistent": trace.consistent,
                        "model_spans": trace.model_spans,
                        "model_attempts": trace.model_attempts,
                        "model_trace_ms": trace.model_ms,
                        "ttft_ms": trace.ttft_ms,
                    }
                )

            app = HTTPWorkerApp(
                handler,
                capabilities=("local", "sse", "durability.sqlite"),
                max_retained_executions=max(1024, self.profile.max_inflight * 2),
                history=history,
            )
            endpoint = f"http://benchmark-worker-{index}"
            registry = WorkerRegistry()
            registry.publish(
                "benchmark-agent",
                (
                    WorkerTarget(
                        f"worker-{index}",
                        endpoint,
                        ("local", "sse", "durability.sqlite"),
                    ),
                ),
            )
            client = HTTPWorkerClient(
                registry,
                transport=httpx.ASGITransport(app=app.app),
                timeout=self.profile.request_deadline_seconds,
            )
            self.worker_apps.append(app)
            self.worker_clients.append(client)
            self.worker_runtimes.append(runtime)
            self.worker_bounds.append(bound)

    async def execute(self, index: int, scheduled_at: float, load_shape: str) -> Sample:
        request_id = f"bench-{index}-{uuid4().hex}"
        started = perf_counter()
        scheduling_delay = max(0.0, (started - scheduled_at) * 1000)
        try:
            if self.scenario == "lifecycle-cancel-deadline":
                return await self._execute_lifecycle(
                    index, request_id, started, scheduling_delay, load_shape
                )
            if self.scenario.startswith("http-worker-"):
                outcome = await self._execute_worker(index, request_id)
                latency = (perf_counter() - started) * 1000
                return Sample(
                    request_id,
                    self.scenario,
                    self.profile.backend,
                    load_shape,
                    latency,
                    scheduling_delay,
                    True,
                    bool(outcome["context_isolated"]),
                    int(outcome["event_count"]),
                    events_ordered=bool(outcome["events_ordered"]),
                    prompt_tokens=int(outcome["prompt_tokens"]),
                    completion_tokens=int(outcome["completion_tokens"]),
                    model_ms=float(outcome["model_ms"]),
                    tool_ms=float(outcome["tool_ms"]),
                    worker_id=cast(str, outcome["worker_id"]),
                    provider_calls=int(outcome["provider_calls"]),
                    model_peak_inflight=int(outcome["model_peak_inflight"]),
                    trace_id=cast(str | None, outcome["trace_id"]),
                    trace_consistent=bool(outcome["trace_consistent"]),
                    model_spans=int(outcome["model_spans"]),
                    model_attempts=int(outcome["model_attempts"]),
                    model_trace_ms=float(outcome["model_trace_ms"]),
                    ttft_ms=(
                        None
                        if outcome["ttft_ms"] is None
                        else float(outcome["ttft_ms"])
                    ),
                )
            async with asyncio.timeout(self.profile.request_deadline_seconds):
                result, context, events, ttft = await self._execute_local(
                    index, request_id
                )
            del result
            trace = _model_trace(events)
            tool = self.scenario.startswith(("react-tool-", "sqlite-durable-"))
            assert self.resources is not None
            model_ms, provider_calls, model_peak, _, _ = self.resources.tracker.take(
                index
            )
            prompt, completion = trace.prompt_tokens, trace.completion_tokens
            tool_ms = sum(self.resources.tool_durations_ms.pop(index, ()))
            return Sample(
                request_id,
                self.scenario,
                self.profile.backend,
                load_shape,
                (perf_counter() - started) * 1000,
                scheduling_delay,
                True,
                _context_isolated(context, request_id, tool=tool),
                len(events),
                events_ordered=_ordered(events),
                ttft_ms=ttft if ttft is not None else trace.ttft_ms,
                model_ms=model_ms,
                tool_ms=tool_ms if tool else None,
                prompt_tokens=prompt,
                completion_tokens=completion,
                provider_calls=provider_calls,
                model_peak_inflight=model_peak,
                trace_id=trace.trace_id,
                trace_consistent=trace.consistent,
                model_spans=trace.model_spans,
                model_attempts=trace.model_attempts,
                model_trace_ms=trace.model_ms,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error_type, error_domain, error_kind, error_attempts = _classify_error(exc)
            model_ms = None
            provider_calls = 0
            model_peak = 0
            prompt = 0
            completion = 0
            if self.resources is not None:
                (
                    recorded_ms,
                    provider_calls,
                    model_peak,
                    prompt,
                    completion,
                ) = self.resources.tracker.take(index)
                model_ms = recorded_ms or None
                self.resources.tool_durations_ms.pop(index, None)
            failed = Sample(
                request_id,
                self.scenario,
                self.profile.backend,
                load_shape,
                (perf_counter() - started) * 1000,
                scheduling_delay,
                False,
                None,
                events_observed=False,
                events_ordered=None,
                model_ms=model_ms,
                prompt_tokens=prompt,
                completion_tokens=completion,
                error_type=error_type,
                error_domain=error_domain,
                error_kind=error_kind,
                provider_calls=provider_calls,
                model_peak_inflight=model_peak,
                trace_consistent=None,
                model_attempts=error_attempts,
            )
            if self.profile.backend == "live":
                raise ExternalEnvironmentInvalid(
                    error_type,
                    error_domain=error_domain,
                    error_kind=error_kind,
                    sample=failed,
                ) from exc
            return failed

    async def _execute_lifecycle(
        self,
        index: int,
        request_id: str,
        started: float,
        scheduling_delay: float,
        load_shape: str,
    ) -> Sample:
        context = model_context(request_id)
        handle = await self.bound.start(
            UserMessage(content=str(index)),
            context,
            execution=ExecutionOptions(
                request_id=request_id,
                deadline=monotonic() + (0.005 if index % 2 == 0 else 1.0),
            ),
        )
        expected = "deadline" if index % 2 == 0 else "cancelled"
        if expected == "cancelled":
            await handle.cancel()
        try:
            await handle.result()
            observed = "completed"
        except asyncio.CancelledError:
            observed = "cancelled"
        except ExecutionDeadlineExceeded:
            observed = "deadline"
        async with handle.subscribe() as subscription:
            events = [event async for event in subscription]
        valid = observed == expected and _ordered(events)
        return Sample(
            request_id,
            self.scenario,
            self.profile.backend,
            load_shape,
            (perf_counter() - started) * 1000,
            scheduling_delay,
            valid,
            True,
            len(events),
            events_ordered=_ordered(events),
            error_type=None if valid else f"expected_{expected}_observed_{observed}",
        )

    async def _execute_local(
        self, index: int, request_id: str
    ) -> tuple[Any, Context, list[Any], float | None]:
        assert self.resources is not None
        tool = self.scenario.startswith(("react-tool-", "sqlite-durable-"))
        module = self.resources.agent if tool else self.resources.model
        message = agent_message(index) if tool else model_message(index)
        context = (
            agent_context(request_id, self.resources.definition)
            if tool
            else model_context(request_id)
        )
        stream_mode = self.scenario.endswith("-stream")
        events: list[Any] = []
        ttft: float | None = None
        observed = perf_counter()
        target: Any = module if self.scenario.startswith("direct-") else self.bound
        if stream_mode:
            kwargs: dict[str, Any] = {}
            if target is not module:
                kwargs["execution"] = ExecutionOptions(
                    request_id=request_id,
                    deadline=monotonic() + self.profile.request_deadline_seconds,
                    model_calls=(
                        {MODEL_GROUP: {"profile": "alternate"}}
                        if self.scenario.startswith("dynamic-model-") and index % 2
                        else {}
                    ),
                )
            async with target.stream(message, context, **kwargs) as stream:
                async for event in stream:
                    events.append(event)
                    if ttft is None and getattr(event, "kind", "") in (
                        "model.text.delta",
                        "model.tool_call.delta",
                    ):
                        ttft = (perf_counter() - observed) * 1000
                result, next_context = await stream.final_result()
        elif target is module:
            handle = await module.start(
                message,
                context,
                execution=ExecutionOptions(request_id=request_id),
            )
            result, next_context = await handle.result()
            async with handle.subscribe() as subscription:
                events = [event async for event in subscription]
        else:
            handle = await target.start(
                message,
                context,
                execution=ExecutionOptions(
                    request_id=request_id,
                    deadline=monotonic() + self.profile.request_deadline_seconds,
                    model_calls=(
                        {MODEL_GROUP: {"profile": "alternate"}}
                        if self.scenario.startswith("dynamic-model-") and index % 2
                        else {}
                    ),
                ),
            )
            result, next_context = await handle.result()
            async with handle.subscribe() as subscription:
                events = [event async for event in subscription]
        return result, next_context, events, ttft

    async def _execute_worker(self, index: int, request_id: str) -> Mapping[str, Any]:
        selected = index % len(self.worker_clients)
        client = self.worker_clients[selected]
        _, result = await client.invoke(
            "benchmark-agent",
            {"index": index, "request_id": request_id},
            request_id=request_id,
            plan_id=self._plan_id,
            graph_hash=self._graph_hash,
            deadline=monotonic() + self.profile.request_deadline_seconds,
            required_capabilities=("local", "durability.sqlite"),
            idempotency_key=request_id,
        )
        decoded = thaw_json(result)
        if not isinstance(decoded, dict):
            raise TypeError("Worker result must be an object")
        return decoded

    def sqlite_bytes(self) -> int:
        total = 0
        for path in self.sqlite_paths:
            for suffix in ("", "-wal", "-shm"):
                candidate = Path(f"{path}{suffix}")
                if candidate.exists():
                    total += candidate.stat().st_size
        return total

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for client in self.worker_clients:
            await client.close()
        for app in self.worker_apps:
            await app.close()
        for runtime in self.worker_runtimes:
            await runtime.close(cancel=True)
        if self.runtime is not None:
            await self.runtime.close(cancel=True)
        await self.stack.aclose()
        if self.resources is not None:
            await self.resources.aclose()
        if self.temporary is not None:
            self.temporary.cleanup()

    async def __aexit__(self, *_: object) -> None:
        await self.close()


def worker_count_for(scenario: str, capacity: int) -> int:
    if not scenario.startswith("http-worker-"):
        return 0
    if capacity >= 32:
        return 4
    if capacity >= 16:
        return 2
    return 1


__all__ = [
    "SUPPORTED_SCENARIOS",
    "ExternalEnvironmentInvalid",
    "ScenarioSession",
    "worker_count_for",
]
