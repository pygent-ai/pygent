"""Portable samples, summaries, resource monitoring, and regression gates."""

from __future__ import annotations

import asyncio
import math
import os
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from time import perf_counter
from typing import Any


class ExitCode(IntEnum):
    OK = 0
    REGRESSION = 1
    CONFIGURATION = 2
    ENVIRONMENT_INVALID = 3


@dataclass(frozen=True, slots=True)
class Sample:
    request_id: str
    scenario: str
    backend: str
    load_shape: str
    latency_ms: float
    scheduling_delay_ms: float
    succeeded: bool
    context_isolated: bool | None
    event_count: int = 0
    events_observed: bool = True
    events_ordered: bool | None = True
    ttft_ms: float | None = None
    model_ms: float | None = None
    tool_ms: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    worker_id: str | None = None
    error_type: str | None = None
    error_domain: str | None = None
    error_kind: str | None = None
    phase: str = "measurement"
    provider_calls: int = 0
    model_peak_inflight: int = 0
    trace_id: str | None = None
    trace_consistent: bool | None = True
    model_spans: int = 0
    model_attempts: int = 0
    model_trace_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    cpu_percent: float = 0.0
    rss_bytes: int = 0
    event_loop_lag_ms: float = 0.0


class ResourceMonitor:
    """Best-effort process and event-loop monitor; psutil is optional."""

    def __init__(self, interval: float = 0.1) -> None:
        self.interval = interval
        self.samples: list[ResourceSnapshot] = []
        self._stopped = asyncio.Event()

    async def run(self) -> None:
        process = None
        try:
            import psutil  # type: ignore[import-untyped]

            process = psutil.Process(os.getpid())
            process.cpu_percent(None)
        except ImportError:
            pass
        expected = perf_counter() + self.interval
        while not self._stopped.is_set():
            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=self.interval)
                break
            except TimeoutError:
                pass
            now = perf_counter()
            lag = max(0.0, (now - expected) * 1000)
            expected = now + self.interval
            cpu = 0.0
            rss = 0
            if process is not None:
                cpu = float(process.cpu_percent(None))
                rss = int(process.memory_info().rss)
            self.samples.append(ResourceSnapshot(cpu, rss, lag))

    def stop(self) -> None:
        self._stopped.set()

    def summary(self) -> dict[str, float | int]:
        return {
            "cpu_peak_percent": max((item.cpu_percent for item in self.samples), default=0.0),
            "rss_peak_bytes": max((item.rss_bytes for item in self.samples), default=0),
            "event_loop_lag_p99_ms": percentile(
                [item.event_loop_lag_ms for item in self.samples], 0.99
            ),
        }


@dataclass(frozen=True, slots=True)
class StageResult:
    scenario: str
    backend: str
    load_shape: str
    concurrency: int | None
    offered_rps: float | None
    duration_seconds: float
    samples: tuple[Sample, ...]
    dropped: int = 0
    resources: dict[str, float | int] = field(default_factory=dict)
    sqlite_bytes: int = 0
    workers: int = 0
    load_factor: float | None = None
    peak_model_inflight: int = 0
    provider_calls: int = 0

    def summary(self) -> dict[str, Any]:
        latencies = [item.latency_ms for item in self.samples if item.succeeded]
        completed = len(self.samples)
        succeeded = sum(item.succeeded for item in self.samples)
        prompt = sum(item.prompt_tokens for item in self.samples)
        completion = sum(item.completion_tokens for item in self.samples)
        failures_by_domain: dict[str, int] = {}
        failures_by_kind: dict[str, int] = {}
        for item in self.samples:
            if item.succeeded:
                continue
            domain = item.error_domain or "unknown"
            kind = item.error_kind or "unknown"
            failures_by_domain[domain] = failures_by_domain.get(domain, 0) + 1
            failures_by_kind[kind] = failures_by_kind.get(kind, 0) + 1
        return {
            "scenario": self.scenario,
            "backend": self.backend,
            "load_shape": self.load_shape,
            "concurrency": self.concurrency,
            "offered_rps": self.offered_rps,
            "achieved_rps": completed / self.duration_seconds if self.duration_seconds else 0.0,
            "duration_seconds": self.duration_seconds,
            "completed": completed,
            "succeeded": succeeded,
            "failed": completed - succeeded,
            "failures_by_domain": dict(sorted(failures_by_domain.items())),
            "failures_by_kind": dict(sorted(failures_by_kind.items())),
            "dropped": self.dropped,
            "success_rate": succeeded / completed if completed else 0.0,
            "latency_ms": {
                "p50": percentile(latencies, 0.50),
                "p95": percentile(latencies, 0.95),
                "p99": percentile(latencies, 0.99),
                "max": max(latencies, default=0.0),
            },
            "scheduling_delay_p99_ms": percentile(
                [item.scheduling_delay_ms for item in self.samples], 0.99
            ),
            "ttft_p95_ms": percentile(
                [item.ttft_ms for item in self.samples if item.ttft_ms is not None],
                0.95,
            ),
            "model_p95_ms": percentile(
                [item.model_ms for item in self.samples if item.model_ms is not None],
                0.95,
            ),
            "model_trace_p95_ms": percentile(
                [
                    item.model_trace_ms
                    for item in self.samples
                    if item.model_trace_ms is not None
                ],
                0.95,
            ),
            "tool_p95_ms": percentile(
                [item.tool_ms for item in self.samples if item.tool_ms is not None],
                0.95,
            ),
            "tokens_per_second": (prompt + completion) / self.duration_seconds
            if self.duration_seconds
            else 0.0,
            "context_isolation_failures": sum(
                item.context_isolated is False for item in self.samples
            ),
            "context_observation_unavailable": sum(
                item.context_isolated is None for item in self.samples
            ),
            "event_order_failures": sum(
                item.events_ordered is False for item in self.samples
            ),
            "event_order_observation_unavailable": sum(
                item.events_ordered is None for item in self.samples
            ),
            "trace_integrity_failures": sum(
                item.trace_consistent is False for item in self.samples
            ),
            "trace_observation_unavailable": sum(
                item.trace_consistent is None for item in self.samples
            ),
            "model_spans": sum(item.model_spans for item in self.samples),
            "model_attempts": sum(item.model_attempts for item in self.samples),
            "event_observation_unavailable": sum(
                not item.events_observed for item in self.samples
            ),
            "sqlite_bytes": self.sqlite_bytes,
            "workers": self.workers,
            "load_factor": self.load_factor,
            "peak_model_inflight": self.peak_model_inflight,
            "provider_calls": self.provider_calls,
            **self.resources,
        }


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    if not 0 <= quantile <= 1 or not math.isfinite(quantile):
        raise ValueError("quantile must be between zero and one")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return round(ordered[low], 3)
    weight = position - low
    return round(ordered[low] + (ordered[high] - ordered[low]) * weight, 3)


def median_summary(stage_summaries: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not stage_summaries:
        raise ValueError("at least one stage summary is required")
    return {
        "achieved_rps": statistics.median(
            float(item["achieved_rps"]) for item in stage_summaries
        ),
        "p95_ms": statistics.median(
            float(item["latency_ms"]["p95"]) for item in stage_summaries
        ),
    }


def compare_summaries(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    throughput_tolerance: float = 0.15,
    p95_tolerance: float = 0.25,
) -> dict[str, Any]:
    base_rps = float(baseline["achieved_rps"])
    next_rps = float(candidate["achieved_rps"])
    base_p95 = float(baseline["p95_ms"])
    next_p95 = float(candidate["p95_ms"])
    throughput_change = 0.0 if base_rps == 0 else (next_rps - base_rps) / base_rps
    p95_change = 0.0 if base_p95 == 0 else (next_p95 - base_p95) / base_p95
    reasons = []
    if throughput_change < -throughput_tolerance:
        reasons.append("throughput")
    if p95_change > p95_tolerance:
        reasons.append("p95")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "throughput_change": throughput_change,
        "p95_change": p95_change,
    }


__all__ = [
    "ExitCode",
    "ResourceMonitor",
    "Sample",
    "StageResult",
    "compare_summaries",
    "median_summary",
    "percentile",
]
