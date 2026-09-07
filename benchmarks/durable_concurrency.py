"""Durable Agent burst benchmark; run each revision in an isolated process.

Example: python benchmarks/durable_concurrency.py --source src --concurrency 200
Uses the existing synthetic Agent scenario (two model calls and one tool call).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter, process_time


async def measure(count: int, scenario: str = "sqlite-durable-invoke") -> dict:
    from benchmarks.config import load_profile
    from benchmarks.scenarios import ScenarioSession

    profile = replace(
        load_profile(Path(__file__).parent / "profiles/synthetic-smoke.toml"),
        max_inflight=count, request_deadline_seconds=120,
    )
    async with ScenarioSession(profile, scenario, capacity=count) as session:
        queries = 0

        def trace(sql: str) -> None:
            nonlocal queries
            if sql.startswith("SELECT") and "execution_cancellations" in sql:
                queries += 1

        if session.runtime is not None and session.runtime.history is not None:
            await session.runtime.history._db().set_trace_callback(trace)
        cpu_start, start = process_time(), perf_counter()
        samples = await asyncio.gather(*(
            session.execute(index, start, "closed") for index in range(count)
        ))
        elapsed, cpu = perf_counter() - start, process_time() - cpu_start
        latencies = sorted(sample.latency_ms for sample in samples)
        result = {
            "scenario": scenario,
            "concurrency": count, "seconds": elapsed, "cpu_seconds": cpu,
            "agents_per_second": count / elapsed,
            "p95_ms": latencies[min(count - 1, int(count * .95))],
            "cancellation_queries": queries,
            "cancellation_queries_per_second": queries / elapsed,
            "failures": sum(not sample.succeeded for sample in samples),
            "integrity_failures": sum(
                sample.context_isolated is False or sample.events_ordered is False
                or sample.trace_consistent is False for sample in samples
            ),
        }
        if result["failures"] or result["integrity_failures"]:
            raise RuntimeError(result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="src")
    parser.add_argument("--concurrency", type=int, default=200)
    parser.add_argument(
        "--scenario", default="sqlite-durable-invoke",
        choices=("sqlite-durable-invoke", "direct-invoke", "local-invoke", "react-tool-invoke"),
    )
    parser.add_argument("--cpus", help="Optional CPU affinity, e.g. 0,2 (requires psutil)")
    args = parser.parse_args()
    if args.cpus:
        import psutil

        psutil.Process().cpu_affinity([int(cpu) for cpu in args.cpus.split(",")])
    if args.concurrency < 1:
        parser.error("concurrency must be positive")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(args.source).resolve()))
    print(json.dumps(asyncio.run(measure(args.concurrency, args.scenario))))
