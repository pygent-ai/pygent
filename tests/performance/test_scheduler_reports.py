from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import pytest

from benchmarks.cli import _has_invariant_failure
from benchmarks.config import load_profile
from benchmarks.metrics import Sample, StageResult
from benchmarks.reports import CampaignRecorder, write_campaign
from benchmarks.runner import StageProgress
from benchmarks.scenarios import ExternalEnvironmentInvalid
from benchmarks.scheduler import run_closed_loop, run_open_loop


def _sample(index: int, scheduled: float, shape: str) -> Sample:
    return Sample(
        request_id=str(index),
        scenario="unit",
        backend="synthetic",
        load_shape=shape,
        latency_ms=1.0,
        scheduling_delay_ms=max(0.0, (perf_counter() - scheduled) * 1000),
        succeeded=True,
        context_isolated=True,
    )


@pytest.mark.asyncio
@pytest.mark.performance
async def test_closed_and_open_schedulers_produce_bounded_samples():
    async def request(index: int, scheduled: float, shape: str) -> Sample:
        await asyncio.sleep(0.002)
        return _sample(index, scheduled, shape)

    closed = await run_closed_loop(
        request,
        scenario="unit",
        backend="synthetic",
        concurrency=2,
        duration_seconds=0.03,
    )
    opened = await run_open_loop(
        request,
        scenario="unit",
        backend="synthetic",
        offered_rps=100.0,
        duration_seconds=0.05,
        max_inflight=16,
    )
    assert len(closed.samples) >= 2
    assert 3 <= len(opened.samples) <= 6
    assert opened.dropped == 0
    assert closed.summary()["context_isolation_failures"] == 0

    tiny = await run_closed_loop(
        request,
        scenario="unit",
        backend="synthetic",
        concurrency=1,
        duration_seconds=0.000001,
    )
    assert len(tiny.samples) == 1


@pytest.mark.asyncio
@pytest.mark.performance
async def test_open_scheduler_cancels_inflight_work_when_parent_is_cancelled():
    entered = asyncio.Event()
    cancelled = 0

    async def request(index: int, scheduled: float, shape: str) -> Sample:
        nonlocal cancelled
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled += 1
            raise
        return _sample(index, scheduled, shape)

    task = asyncio.create_task(
        run_open_loop(
            request,
            scenario="cancel",
            backend="synthetic",
            offered_rps=100.0,
            duration_seconds=10.0,
            max_inflight=4,
        )
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled >= 1


@pytest.mark.asyncio
@pytest.mark.performance
async def test_closed_scheduler_observes_successes_before_a_later_failure():
    observed: list[Sample] = []

    async def request(index: int, scheduled: float, shape: str) -> Sample:
        if index == 0:
            return _sample(index, scheduled, shape)
        failed = replace(
            _sample(index, scheduled, shape),
            succeeded=False,
            context_isolated=None,
            events_ordered=None,
            events_observed=False,
            trace_consistent=None,
            error_type="TimeoutError",
            error_domain="model",
            error_kind="timeout",
            provider_calls=2,
            model_peak_inflight=1,
        )
        raise ExternalEnvironmentInvalid(
            "TimeoutError",
            error_domain="model",
            error_kind="timeout",
            sample=failed,
        )

    with pytest.raises(ExternalEnvironmentInvalid, match="TimeoutError"):
        await run_closed_loop(
            request,
            scenario="unit",
            backend="live",
            concurrency=1,
            duration_seconds=10.0,
            on_sample=observed.append,
        )

    assert [sample.request_id for sample in observed] == ["0", "1"]
    assert not observed[-1].succeeded
    assert observed[-1].provider_calls == 2


@pytest.mark.performance
def test_report_artifacts_are_complete_and_secret_safe(tmp_path: Path):
    profile = replace(
        load_profile("benchmarks/profiles/synthetic-smoke.toml"),
        scenarios=("direct-invoke",),
    )
    sample = _sample(1, perf_counter(), "closed")
    stage = StageResult(
        "direct-invoke",
        "synthetic",
        "closed",
        1,
        None,
        1.0,
        (sample,),
    )
    destination = write_campaign(tmp_path, profile, (stage,))
    assert {path.name for path in destination.iterdir()} == {
        "samples.jsonl",
        "stages.csv",
        "summary.json",
        "summary.md",
    }
    serialized = json.dumps(json.loads((destination / "summary.json").read_text()))
    assert "GLM_API_KEY" not in serialized
    assert json.loads(serialized)["outcome"] == "completed"
    assert json.loads((destination / "samples.jsonl").read_text())["request_id"] == "1"
    markdown = (destination / "summary.md").read_text(encoding="utf-8")
    assert "Model trace P95 ms" in markdown
    assert "Trace failures" in markdown


@pytest.mark.performance
def test_live_failure_persists_partial_samples_and_stage_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    async def fail_campaign(profile, *, on_sample, on_stage, on_stage_start):
        del profile
        on_stage_start(StageProgress("direct-invoke", "closed", "measurement", 1))
        completed_sample = replace(
            _sample(6, perf_counter(), "closed"), scenario="direct-invoke"
        )
        on_sample(completed_sample)
        on_stage(
            StageResult(
                "direct-invoke",
                "live",
                "closed",
                1,
                None,
                1.0,
                (completed_sample,),
            )
        )
        on_stage_start(StageProgress("direct-invoke", "closed", "measurement", 2))
        on_sample(
            replace(
                _sample(7, perf_counter(), "closed"),
                scenario="direct-invoke",
                provider_calls=2,
                model_peak_inflight=1,
            )
        )
        raise ExternalEnvironmentInvalid(
            "TimeoutError", error_domain="model", error_kind="timeout"
        )

    monkeypatch.setattr("benchmarks.cli.run_campaign", fail_campaign)
    from benchmarks.cli import main

    assert main(
        (
            "run",
            "live-long-direct-invoke",
            "--confirm-live",
            "--output",
            str(tmp_path),
        )
    ) == 3
    error = json.loads(capsys.readouterr().err)
    destination = Path(error["result"])
    report = json.loads((destination / "summary.json").read_text(encoding="utf-8"))

    assert report["outcome"] == "incomplete"
    assert report["completed"] is False
    assert report["termination"] == {
        "kind": "external_environment_invalid",
        "error_type": "TimeoutError",
        "error_domain": "model",
        "error_kind": "timeout",
    }
    assert len(report["stages"]) == 1
    assert report["stages"][0]["concurrency"] == 1
    assert report["partial_stage"]["scenario"] == "direct-invoke"
    assert report["partial_stage"]["concurrency"] == 2
    assert report["partial_stage"]["phase"] == "measurement"
    assert report["partial_stage"]["completed"] == 1
    assert report["partial_stage"]["failures_by_domain"] == {}
    assert report["partial_stage"]["failures_by_kind"] == {}
    assert report["partial_stage"]["provider_calls"] == 2
    assert report["partial_stage"]["peak_model_inflight"] == 1
    sample_lines = (destination / "samples.jsonl").read_text().splitlines()
    assert [json.loads(line)["request_id"] for line in sample_lines] == ["6", "7"]
    assert "TimeoutError" in (destination / "summary.md").read_text(encoding="utf-8")


@pytest.mark.performance
def test_warmup_failure_preserves_samples_without_creating_a_measured_stage(
    tmp_path: Path,
):
    profile = replace(
        load_profile("benchmarks/profiles/synthetic-smoke.toml"),
        scenarios=("direct-invoke",),
    )
    recorder = CampaignRecorder(tmp_path, profile)
    recorder.begin_stage(StageProgress("direct-invoke", "closed", "warmup", 4))
    recorder.record_sample(replace(
        _sample(2, perf_counter(), "closed"),
        phase="warmup",
        succeeded=False,
        context_isolated=None,
        events_ordered=None,
        events_observed=False,
        trace_consistent=None,
        error_type="ModelCallError",
        error_domain="model",
        error_kind="timeout",
        provider_calls=2,
        prompt_tokens=3,
    ))
    destination = recorder.fail(
        kind="external_environment_invalid",
        error_type="ModelCallError",
        error_domain="model",
        error_kind="timeout",
    )
    report = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
    samples = (destination / "samples.jsonl").read_text(encoding="utf-8")

    assert report["stages"] == []
    assert report["partial_stage"]["phase"] == "warmup"
    assert report["partial_stage"]["completed"] == 1
    assert report["partial_stage"]["failed"] == 1
    assert report["partial_stage"]["failures_by_domain"] == {"model": 1}
    assert report["partial_stage"]["failures_by_kind"] == {"timeout": 1}
    assert report["partial_stage"]["provider_calls"] == 2
    assert report["partial_stage"]["context_isolation_failures"] == 0
    assert report["partial_stage"]["context_observation_unavailable"] == 1
    assert report["partial_stage"]["event_order_failures"] == 0
    assert report["partial_stage"]["event_order_observation_unavailable"] == 1
    assert report["partial_stage"]["trace_integrity_failures"] == 0
    assert report["partial_stage"]["trace_observation_unavailable"] == 1
    assert report["termination"]["error_kind"] == "timeout"
    assert report["termination"]["error_domain"] == "model"
    assert json.loads(samples)["phase"] == "warmup"


@pytest.mark.performance
def test_stage_summary_separates_invariant_failures_from_unavailable_observations():
    unavailable = replace(
        _sample(1, perf_counter(), "closed"),
        succeeded=False,
        context_isolated=None,
        events_ordered=None,
        events_observed=False,
        trace_consistent=None,
        error_domain="model",
        error_kind="timeout",
    )
    violated = replace(
        _sample(2, perf_counter(), "closed"),
        context_isolated=False,
        events_ordered=False,
        trace_consistent=False,
    )
    summary = StageResult(
        "unit", "synthetic", "closed", 2, None, 1.0, (unavailable, violated)
    ).summary()

    assert summary["context_isolation_failures"] == 1
    assert summary["context_observation_unavailable"] == 1
    assert summary["event_order_failures"] == 1
    assert summary["event_order_observation_unavailable"] == 1
    assert summary["trace_integrity_failures"] == 1
    assert summary["trace_observation_unavailable"] == 1
    assert summary["failures_by_domain"] == {"model": 1}
    assert summary["failures_by_kind"] == {"timeout": 1}


@pytest.mark.performance
def test_completed_campaign_gates_unavailable_or_failed_observations():
    unavailable = replace(
        _sample(1, perf_counter(), "closed"),
        context_isolated=None,
        events_ordered=None,
        events_observed=False,
        trace_consistent=None,
    )
    stage = StageResult(
        "unit", "synthetic", "closed", 1, None, 1.0, (unavailable,)
    )

    assert _has_invariant_failure((stage,))
