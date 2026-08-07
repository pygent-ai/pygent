"""Campaign orchestration across profiles, scenarios, and load shapes."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from typing import Any

from .config import LoadProfile
from .metrics import Sample, StageResult
from .scenarios import ScenarioSession, worker_count_for
from .scheduler import run_closed_loop, run_open_loop

SampleObserver = Callable[[Sample], None]
StageObserver = Callable[[StageResult], None]


@dataclass(frozen=True, slots=True)
class StageProgress:
    scenario: str
    load_shape: str
    phase: str
    concurrency: int | None = None
    offered_rps: float | None = None
    load_factor: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


StageStartObserver = Callable[[StageProgress], None]


async def _phase_warmup(
    session: ScenarioSession,
    *,
    scenario: str,
    backend: str,
    capacity: int,
    seconds: float,
    on_sample: SampleObserver | None = None,
) -> None:
    if seconds <= 0:
        return
    await run_closed_loop(
        session.execute,
        scenario=scenario,
        backend=backend,
        concurrency=min(capacity, 4),
        duration_seconds=seconds,
        on_sample=(
            (lambda sample: on_sample(replace(sample, phase="warmup")))
            if on_sample is not None
            else None
        ),
    )


async def closed_stage(
    profile: LoadProfile,
    scenario: str,
    concurrency: int,
    *,
    on_sample: SampleObserver | None = None,
    on_stage_start: StageStartObserver | None = None,
) -> StageResult:
    workers = worker_count_for(scenario, concurrency)
    async with ScenarioSession(
        profile,
        scenario,
        capacity=concurrency,
        worker_count=max(1, workers),
    ) as session:
        if on_stage_start is not None and profile.warmup_seconds > 0:
            on_stage_start(StageProgress(scenario, "closed", "warmup", concurrency))
        await _phase_warmup(
            session,
            scenario=scenario,
            backend=profile.backend,
            capacity=concurrency,
            seconds=profile.warmup_seconds,
            on_sample=on_sample,
        )
        if on_stage_start is not None:
            on_stage_start(StageProgress(scenario, "closed", "measurement", concurrency))
        result = await run_closed_loop(
            session.execute,
            scenario=scenario,
            backend=profile.backend,
            concurrency=concurrency,
            duration_seconds=profile.closed_duration_seconds,
            on_sample=on_sample,
        )
        assert session.resources is not None
        result = replace(
            result,
            sqlite_bytes=session.sqlite_bytes(),
            workers=workers,
            peak_model_inflight=max(
                (sample.model_peak_inflight for sample in result.samples), default=0
            ),
            provider_calls=sum(sample.provider_calls for sample in result.samples),
        )
    if profile.cooldown_seconds:
        await asyncio.sleep(profile.cooldown_seconds)
    return result


async def open_stage(
    profile: LoadProfile,
    scenario: str,
    offered_rps: float,
    *,
    capacity: int,
    load_factor: float | None = None,
    on_sample: SampleObserver | None = None,
    on_stage_start: StageStartObserver | None = None,
) -> StageResult:
    workers = worker_count_for(scenario, capacity)
    async with ScenarioSession(
        profile,
        scenario,
        capacity=capacity,
        worker_count=max(1, workers),
    ) as session:
        if on_stage_start is not None and profile.warmup_seconds > 0:
            on_stage_start(
                StageProgress(
                    scenario, "open", "warmup", offered_rps=offered_rps,
                    load_factor=load_factor,
                )
            )
        await _phase_warmup(
            session,
            scenario=scenario,
            backend=profile.backend,
            capacity=capacity,
            seconds=profile.warmup_seconds,
            on_sample=on_sample,
        )
        if on_stage_start is not None:
            on_stage_start(
                StageProgress(
                    scenario, "open", "measurement", offered_rps=offered_rps,
                    load_factor=load_factor,
                )
            )
        result = await run_open_loop(
            session.execute,
            scenario=scenario,
            backend=profile.backend,
            offered_rps=offered_rps,
            duration_seconds=profile.open_duration_seconds,
            max_inflight=profile.max_inflight,
            on_sample=on_sample,
        )
        assert session.resources is not None
        result = replace(
            result,
            sqlite_bytes=session.sqlite_bytes(),
            workers=workers,
            load_factor=load_factor,
            peak_model_inflight=max(
                (sample.model_peak_inflight for sample in result.samples), default=0
            ),
            provider_calls=sum(sample.provider_calls for sample in result.samples),
        )
    if profile.cooldown_seconds:
        await asyncio.sleep(profile.cooldown_seconds)
    return result


async def calibrate(
    profile: LoadProfile,
    *,
    on_sample: SampleObserver | None = None,
    on_stage: StageObserver | None = None,
    on_stage_start: StageStartObserver | None = None,
) -> tuple[list[StageResult], dict[str, float]]:
    results: list[StageResult] = []
    peaks: dict[str, float] = {}
    for scenario in profile.scenarios:
        scenario_results = []
        for concurrency in profile.concurrency:
            result = await closed_stage(
                profile,
                scenario,
                concurrency,
                on_sample=on_sample,
                on_stage_start=on_stage_start,
            )
            results.append(result)
            scenario_results.append(result)
            if on_stage is not None:
                on_stage(result)
        healthy = [
            result
            for result in scenario_results
            if result.summary()["success_rate"] == 1.0
            and result.summary()["context_isolation_failures"] == 0
        ]
        peaks[scenario] = max(
            (float(result.summary()["achieved_rps"]) for result in healthy),
            default=0.0,
        )
    return results, peaks


async def run_campaign(
    profile: LoadProfile,
    *,
    on_sample: SampleObserver | None = None,
    on_stage: StageObserver | None = None,
    on_stage_start: StageStartObserver | None = None,
) -> list[StageResult]:
    results: list[StageResult] = []
    for _ in range(profile.repetitions):
        closed, peaks = await calibrate(
            replace(profile, repetitions=1),
            on_sample=on_sample,
            on_stage=on_stage,
            on_stage_start=on_stage_start,
        )
        results.extend(closed)
        if profile.open_duration_seconds <= 0:
            continue
        for scenario in profile.scenarios:
            peak = peaks[scenario]
            if peak <= 0:
                continue
            capacity = max(profile.concurrency)
            for multiplier in profile.open_multipliers:
                results.append(
                    await open_stage(
                        profile,
                        scenario,
                        peak * multiplier,
                        capacity=capacity,
                        load_factor=multiplier,
                        on_sample=on_sample,
                        on_stage_start=on_stage_start,
                    )
                )
                if on_stage is not None:
                    on_stage(results[-1])
    return results


__all__ = [
    "StageProgress",
    "calibrate",
    "closed_stage",
    "open_stage",
    "run_campaign",
]
