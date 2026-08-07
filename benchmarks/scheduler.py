"""Closed-loop and fixed-arrival-rate asyncio load schedulers."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from time import perf_counter
from typing import Any

from .metrics import ResourceMonitor, Sample, StageResult

Request = Callable[[int, float, str], Coroutine[Any, Any, Sample]]
SampleObserver = Callable[[Sample], None]


def _observe_failed_sample(
    error: BaseException, observer: SampleObserver | None
) -> None:
    sample = getattr(error, "sample", None)
    if observer is not None and isinstance(sample, Sample):
        observer(sample)


async def _monitored(coroutine: Awaitable[tuple[list[Sample], int]]) -> tuple[list[Sample], int, dict[str, float | int]]:
    monitor = ResourceMonitor()
    task = asyncio.create_task(monitor.run(), name="benchmark-resource-monitor")
    try:
        samples, dropped = await coroutine
    finally:
        monitor.stop()
        await task
    return samples, dropped, monitor.summary()


async def run_closed_loop(
    request: Request,
    *,
    scenario: str,
    backend: str,
    concurrency: int,
    duration_seconds: float,
    on_sample: SampleObserver | None = None,
) -> StageResult:
    if concurrency <= 0 or duration_seconds <= 0:
        raise ValueError("closed-loop concurrency and duration must be positive")
    started = perf_counter()
    stop_at = started + duration_seconds
    sequence = 0
    sequence_lock = asyncio.Lock()

    async def worker() -> list[Sample]:
        nonlocal sequence
        results: list[Sample] = []
        first = True
        while first or perf_counter() < stop_at:
            first = False
            async with sequence_lock:
                index = sequence
                sequence += 1
            try:
                sample = await request(index, perf_counter(), "closed")
            except BaseException as exc:
                _observe_failed_sample(exc, on_sample)
                raise
            if on_sample is not None:
                on_sample(sample)
            results.append(sample)
        return results

    async def execute() -> tuple[list[Sample], int]:
        tasks = [
            asyncio.create_task(worker(), name=f"benchmark-closed-{index}")
            for index in range(concurrency)
        ]
        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_EXCEPTION
            )
            failure = next(
                (
                    task.exception()
                    for task in done
                    if not task.cancelled() and task.exception() is not None
                ),
                None,
            )
            if failure is not None:
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                raise failure
            groups = await asyncio.gather(*tasks)
            return [item for group in groups for item in group], 0
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    samples, dropped, resources = await _monitored(execute())
    elapsed = perf_counter() - started
    return StageResult(
        scenario,
        backend,
        "closed",
        concurrency,
        None,
        elapsed,
        tuple(samples),
        dropped,
        resources,
    )


async def run_open_loop(
    request: Request,
    *,
    scenario: str,
    backend: str,
    offered_rps: float,
    duration_seconds: float,
    max_inflight: int,
    on_sample: SampleObserver | None = None,
) -> StageResult:
    if offered_rps <= 0 or duration_seconds <= 0 or max_inflight <= 0:
        raise ValueError("open-loop rate, duration, and max_inflight must be positive")
    started = perf_counter()
    interval = 1.0 / offered_rps
    tasks: set[asyncio.Task[Sample]] = set()
    finished_samples: list[Sample] = []
    dropped = 0
    sequence = 0

    async def observed_request(index: int, due: float) -> Sample:
        try:
            sample = await request(index, due, "open")
        except BaseException as exc:
            _observe_failed_sample(exc, on_sample)
            raise
        if on_sample is not None:
            on_sample(sample)
        return sample

    async def execute() -> tuple[list[Sample], int]:
        nonlocal dropped, sequence
        try:
            due = started
            stop_at = started + duration_seconds
            while due < stop_at:
                delay = due - perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
                completed = tuple(task for task in tasks if task.done())
                tasks.difference_update(completed)
                finished_samples.extend(task.result() for task in completed)
                if len(tasks) >= max_inflight:
                    dropped += 1
                else:
                    task: asyncio.Task[Sample] = asyncio.create_task(
                        observed_request(sequence, due),
                        name=f"benchmark-open-{sequence}",
                    )
                    tasks.add(task)
                sequence += 1
                due = started + sequence * interval
            finished_samples.extend(await asyncio.gather(*tasks))
            return finished_samples, dropped
        finally:
            pending = tuple(task for task in tasks if not task.done())
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    samples, dropped, resources = await _monitored(execute())
    elapsed = perf_counter() - started
    return StageResult(
        scenario,
        backend,
        "open",
        None,
        offered_rps,
        elapsed,
        tuple(samples),
        dropped,
        resources,
    )


__all__ = ["Request", "SampleObserver", "run_closed_loop", "run_open_loop"]
