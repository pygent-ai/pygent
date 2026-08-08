from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import pytest

from benchmarks.config import load_profile
from benchmarks.scenarios import (
    SUPPORTED_SCENARIOS,
    ScenarioSession,
    _classify_error,
    _model_trace,
)
from pygent import (
    ModelCallError,
    ModelErrorKind,
)
from pygent.core import (
    ExecutionFailure,
    ExecutionFailureError,
)
from pygent.llm import ModelAttempt
from pygent.runtime import WorkerRemoteError


def _profile():
    return replace(
        load_profile("benchmarks/profiles/synthetic-smoke.toml"),
        warmup_seconds=0.0,
        cooldown_seconds=0.0,
        request_deadline_seconds=5.0,
    )


@pytest.mark.asyncio
@pytest.mark.performance
@pytest.mark.parametrize("scenario", SUPPORTED_SCENARIOS)
async def test_every_synthetic_execution_path_succeeds_and_isolates_context(scenario):
    profile = replace(_profile(), scenarios=(scenario,))
    async with ScenarioSession(
        profile,
        scenario,
        capacity=2,
        worker_count=1,
    ) as session:
        sample = await session.execute(3, perf_counter(), "closed")

    assert sample.succeeded
    assert sample.context_isolated
    assert sample.events_ordered
    assert sample.error_type is None
    if scenario != "lifecycle-cancel-deadline":
        assert sample.model_ms is not None and sample.model_ms > 0
        assert sample.prompt_tokens > 0
        assert sample.completion_tokens > 0
        assert sample.provider_calls >= 1
        assert sample.model_peak_inflight >= 1
        assert sample.trace_id
        assert sample.trace_consistent
        assert sample.model_spans >= 1
        assert sample.model_attempts >= sample.model_spans
        assert sample.model_trace_ms is not None and sample.model_trace_ms > 0
    if scenario.startswith(("react-tool-", "sqlite-durable-", "http-worker-")):
        assert sample.tool_ms is not None and sample.tool_ms > 0
    if scenario != "lifecycle-cancel-deadline":
        assert sample.event_count > 0
        assert sample.events_observed
    if scenario == "direct-stream":
        assert sample.model_ms is not None
        assert sample.model_ms >= profile.model.latency_ms * 0.9


@pytest.mark.asyncio
@pytest.mark.performance
async def test_direct_invoke_observes_public_execution_and_model_events():
    profile = replace(_profile(), scenarios=("direct-invoke",))
    async with ScenarioSession(
        profile,
        "direct-invoke",
        capacity=1,
    ) as session:
        _, _, events, _ = await session._execute_local(3, "direct-observation")

    kinds = [event.kind for event in events]
    assert kinds[0] == "execution.started"
    assert "model.attempt.started" in kinds
    assert "model.usage" in kinds
    assert "model.attempt.succeeded" in kinds
    assert kinds[-1] == "execution.completed"
    assert [event.sequence for event in events] == list(range(len(events)))


@pytest.mark.asyncio
@pytest.mark.performance
async def test_durable_http_worker_handles_concurrent_successful_executions():
    profile = replace(
        _profile(),
        scenarios=("http-worker-invoke",),
        # This is a correctness test, not an SLA assertion. Windows 3.11
        # release runners can take more than 20 seconds under shared-host load.
        request_deadline_seconds=60.0,
    )
    async with ScenarioSession(
        profile,
        "http-worker-invoke",
        capacity=8,
        worker_count=2,
    ) as session:
        samples = await asyncio.gather(
            *(session.execute(index, perf_counter(), "closed") for index in range(8))
        )
        sqlite_bytes = session.sqlite_bytes()

    assert all(sample.succeeded and sample.context_isolated for sample in samples)
    assert {sample.worker_id for sample in samples} == {"worker-0", "worker-1"}
    assert sqlite_bytes > 0


def test_model_trace_uses_last_cumulative_usage_snapshot_per_attempt():
    def event(sequence, kind, *, span="model", parent="root", data=None, timestamp=1):
        return SimpleNamespace(
            sequence=sequence,
            kind=kind,
            trace_id="trace-1",
            event_id=f"event-{sequence}",
            span_id=span,
            parent_span_id=parent,
            timestamp_unix_ns=timestamp,
            data=data or {},
        )

    events = [
        event(0, "execution.started", span="root", parent=None, timestamp=1_000_000),
        event(1, "model.started", timestamp=2_000_000),
        event(2, "model.attempt.started", data={"route_id": "r", "attempt": 1}),
        event(
            3,
            "model.usage",
            data={"route_id": "r", "attempt": 1, "input_tokens": 2, "output_tokens": 1},
        ),
        event(
            4,
            "model.usage",
            data={"route_id": "r", "attempt": 1, "input_tokens": 3, "output_tokens": 2},
        ),
        event(5, "model.text.delta", timestamp=4_000_000),
        event(6, "model.completed", timestamp=7_000_000),
    ]

    trace = _model_trace(events)

    assert trace.trace_id == "trace-1"
    assert trace.consistent
    assert (trace.prompt_tokens, trace.completion_tokens) == (3, 2)
    assert trace.model_spans == 1
    assert trace.model_attempts == 1
    assert trace.model_ms == 5.0
    assert trace.ttft_ms == 3.0


def test_live_model_error_classification_preserves_kind_and_attempt_count():
    error = ModelCallError(
        "sanitized",
        kind=ModelErrorKind.TIMEOUT,
        attempts=(
            ModelAttempt("glm", "failed", ModelErrorKind.TIMEOUT, 1),
            ModelAttempt("glm", "failed", ModelErrorKind.TIMEOUT, 2),
        ),
    )

    assert _classify_error(error) == ("ModelCallError", "model", "timeout", 2)
    assert _classify_error(TimeoutError()) == ("TimeoutError", None, "timeout", 0)

    remote = WorkerRemoteError(
        ExecutionFailure(
            domain="model",
            kind="rate_limit",
            message="remote model failure",
            retryable=True,
            details={"attempts": [{"attempt": 1}, {"attempt": 2}]},
        )
    )
    assert _classify_error(remote) == (
        "WorkerRemoteError",
        "model",
        "rate_limit",
        2,
    )

    portable = ExecutionFailureError(
        ExecutionFailure(
            domain="tool",
            kind="outcome_unknown",
            message="sanitized tool failure",
            outcome_unknown=True,
        )
    )
    assert _classify_error(portable) == (
        "ExecutionFailureError",
        "tool",
        "outcome_unknown",
        0,
    )


def test_canonical_live_suite_waits_before_reading_child_exit_code():
    script = Path("benchmarks/run-live-suite.ps1").read_text(encoding="utf-8")
    wait = script.index("$process.WaitForExit()")
    read = script.index("$process.ExitCode", wait)
    assert wait < read
    assert "$null -eq $process.ExitCode" in script
