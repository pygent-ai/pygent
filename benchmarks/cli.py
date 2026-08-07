"""Command line interface for Pygent benchmark campaigns."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import LoadProfile, load_profile
from .metrics import ExitCode, compare_summaries, median_summary
from .reports import CampaignRecorder, load_samples, render_markdown
from .runner import calibrate, run_campaign
from .scenarios import ExternalEnvironmentInvalid

PROFILE_ROOT = Path(__file__).with_name("profiles")
LIVE_LONG_STEP_PREFIX = "live-long-"


def _profile(value: str) -> LoadProfile:
    candidate = Path(value)
    if not candidate.exists():
        candidate = PROFILE_ROOT / f"{value}.toml"
    if not candidate.exists() and value.startswith(LIVE_LONG_STEP_PREFIX):
        scenario = value.removeprefix(LIVE_LONG_STEP_PREFIX)
        aggregate = load_profile(PROFILE_ROOT / "live-long.toml")
        if scenario in aggregate.scenarios:
            return replace(aggregate, name=value, scenarios=(scenario,))
    return load_profile(candidate)


def dry_run(profile: LoadProfile, *, assumed_live_latency: float = 10.0) -> dict[str, object]:
    seconds = profile.estimated_seconds()
    estimated_calls = None
    if profile.backend == "live":
        closed_calls = sum(
            concurrency * profile.closed_duration_seconds / assumed_live_latency
            for concurrency in profile.concurrency
        )
        closed_calls *= len(profile.scenarios) * profile.repetitions
        # ReAct paths use two successful model calls; other paths use one.
        tool_weight = sum(
            2 if any(name in scenario for name in ("react", "sqlite", "worker")) else 1
            for scenario in profile.scenarios
        ) / len(profile.scenarios)
        estimated_calls = int(closed_calls * tool_weight)
    return {
        "profile": profile.name,
        "backend": profile.backend,
        "scenario_count": len(profile.scenarios),
        "estimated_duration_seconds": seconds,
        "estimated_duration_hours": round(seconds / 3600, 3),
        "estimated_live_model_calls_closed_loop": estimated_calls,
        "estimated_live_model_attempts_closed_loop_max": (
            None
            if estimated_calls is None
            else estimated_calls * profile.model.retry_max_attempts
        ),
        "estimate_assumed_latency_seconds": assumed_live_latency
        if profile.backend == "live"
        else None,
        "open_loop_calls": "derived after calibration",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pygent native asyncio load system")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "calibrate", "dry-run"):
        command = commands.add_parser(name)
        command.add_argument("profile")
        command.add_argument("--scenario", action="append", dest="scenarios")
        command.add_argument("--confirm-live", action="store_true")
        command.add_argument("--output", type=Path, default=Path(".benchmarks/results"))
    dry = commands.choices["dry-run"]
    dry.add_argument("--assumed-live-latency", type=float, default=10.0)
    compare = commands.add_parser("compare")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    summarize = commands.add_parser("summarize")
    summarize.add_argument("result", type=Path)
    return parser


def _validate_live(profile: LoadProfile, confirm: bool) -> None:
    if profile.backend == "live" and not confirm:
        raise ValueError("live campaigns require explicit --confirm-live")


def _has_invariant_failure(stages: Sequence[Any]) -> bool:
    for stage in stages:
        summary = stage.summary()  # type: ignore[attr-defined]
        if (
            summary["failed"]
            or summary["context_isolation_failures"]
            or summary["context_observation_unavailable"]
            or summary["event_order_failures"]
            or summary["event_order_observation_unavailable"]
            or summary["event_observation_unavailable"]
            or summary["trace_integrity_failures"]
            or summary["trace_observation_unavailable"]
        ):
            return True
    return False


def _summary_path(path: Path) -> Path:
    return path / "summary.json" if path.is_dir() else path


def _reduce_report(report: dict[str, Any]) -> dict[str, dict[str, float]]:
    stages = report.get("stages")
    if not isinstance(stages, list):
        return {"summary": {"achieved_rps": report["achieved_rps"], "p95_ms": report["p95_ms"]}}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for stage in stages:
        load = (
            f"c={stage['concurrency']}"
            if stage["load_shape"] == "closed"
            else f"factor={stage.get('load_factor')}"
        )
        key = f"{stage['scenario']}|{stage['load_shape']}|{load}"
        grouped.setdefault(key, []).append(stage)
    return {key: median_summary(values) for key, values in grouped.items()}


async def _execute(args: argparse.Namespace) -> int:
    if args.command == "compare":
        baseline = _reduce_report(
            json.loads(_summary_path(args.baseline).read_text(encoding="utf-8"))
        )
        candidate = _reduce_report(
            json.loads(_summary_path(args.candidate).read_text(encoding="utf-8"))
        )
        common = sorted(set(baseline) & set(candidate))
        if not common:
            raise ValueError("benchmark reports have no comparable stages")
        comparisons = {
            key: compare_summaries(baseline[key], candidate[key]) for key in common
        }
        comparison = {
            "passed": all(item["passed"] for item in comparisons.values()),
            "stages": comparisons,
        }
        print(json.dumps(comparison, sort_keys=True))
        return ExitCode.OK if comparison["passed"] else ExitCode.REGRESSION
    if args.command == "summarize":
        summary_path = args.result / "summary.json" if args.result.is_dir() else args.result
        if summary_path.name == "samples.jsonl":
            samples = load_samples(summary_path)
            print(json.dumps({"samples": len(samples)}, sort_keys=True))
        else:
            report = json.loads(summary_path.read_text(encoding="utf-8"))
            print(render_markdown(report))
        return ExitCode.OK

    profile = _profile(args.profile)
    if args.scenarios:
        profile = replace(profile, scenarios=tuple(args.scenarios))
    if args.command == "dry-run":
        print(
            json.dumps(
                dry_run(profile, assumed_live_latency=args.assumed_live_latency),
                indent=2,
                sort_keys=True,
            )
        )
        return ExitCode.OK
    _validate_live(profile, args.confirm_live)
    recorder = CampaignRecorder(args.output, profile)
    callbacks = {
        "on_sample": recorder.record_sample,
        "on_stage": lambda stage: recorder.record_stage(
            stage, samples_already_recorded=True
        ),
        "on_stage_start": recorder.begin_stage,
    }
    try:
        if args.command == "calibrate":
            stages, peaks = await calibrate(
                replace(profile, repetitions=1), **callbacks
            )
        else:
            stages = await run_campaign(profile, **callbacks)
            peaks = None
    except ExternalEnvironmentInvalid as exc:
        destination = recorder.fail(
            kind="external_environment_invalid",
            error_type=str(exc),
            error_domain=exc.error_domain,
            error_kind=exc.error_kind,
        )
        error_payload = {
            "environment_invalid": str(exc),
            "result": str(destination),
        }
        if exc.error_kind is not None:
            error_payload["error_kind"] = exc.error_kind
        if exc.error_domain is not None:
            error_payload["error_domain"] = exc.error_domain
        print(json.dumps(error_payload, sort_keys=True), file=sys.stderr)
        return ExitCode.ENVIRONMENT_INVALID
    except BaseException as exc:
        recorder.fail(kind="campaign_error", error_type=type(exc).__name__)
        raise
    destination = recorder.complete()
    payload: dict[str, object] = {"result": str(destination)}
    if peaks is not None:
        payload["peak_rps"] = peaks
    print(json.dumps(payload, sort_keys=True))
    return ExitCode.REGRESSION if _has_invariant_failure(stages) else ExitCode.OK


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return int(asyncio.run(_execute(_parser().parse_args(argv))))
    except ExternalEnvironmentInvalid as exc:
        print(json.dumps({"environment_invalid": str(exc)}), file=sys.stderr)
        return int(ExitCode.ENVIRONMENT_INVALID)
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"configuration_error": str(exc)}), file=sys.stderr)
        return int(ExitCode.CONFIGURATION)
    except KeyboardInterrupt:
        print(json.dumps({"cancelled": True}), file=sys.stderr)
        return 130


__all__ = ["dry_run", "main"]
