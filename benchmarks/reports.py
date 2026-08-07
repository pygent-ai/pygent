"""Secret-safe benchmark artifacts and human-readable summaries."""

from __future__ import annotations

import csv
import io
import json
import platform
import subprocess
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, TextIO

from .config import LoadProfile
from .metrics import Sample, StageResult
from .runner import StageProgress


def machine_metadata(profile: LoadProfile) -> dict[str, Any]:
    def git(*arguments: str) -> str:
        try:
            return subprocess.run(
                ("git", *arguments),
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return "unavailable"

    status = git("status", "--porcelain")
    return {
        "schema": "pygent.benchmark-report.v2",
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(status and status != "unavailable"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "processor": platform.processor() or "unknown",
        "logical_cpu_count": __import__("os").cpu_count(),
        "profile_name": profile.name,
        "profile_digest": profile.digest,
        "backend": profile.backend,
        "configuration": {
            "scenarios": list(profile.scenarios),
            "concurrency": list(profile.concurrency),
            "closed_duration_seconds": profile.closed_duration_seconds,
            "open_multipliers": list(profile.open_multipliers),
            "open_duration_seconds": profile.open_duration_seconds,
            "repetitions": profile.repetitions,
            "live_credentials_configured": profile.backend == "live",
        },
    }


class CampaignRecorder:
    """Incrementally preserve sanitized samples and campaign checkpoints."""

    def __init__(self, output_root: Path, profile: LoadProfile) -> None:
        execution_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        self.destination = output_root / execution_id
        self.destination.mkdir(parents=True, exist_ok=False)
        self.profile = profile
        self.metadata = machine_metadata(profile)
        self.stages: list[dict[str, Any]] = []
        self._current: StageProgress | None = None
        self._current_started = perf_counter()
        self._partial_samples: list[Sample] = []
        self._sample_file: TextIO = (self.destination / "samples.jsonl").open(
            "w", encoding="utf-8"
        )
        self._closed = False
        self._persist("running")

    def begin_stage(self, progress: StageProgress) -> None:
        self._current = progress
        self._current_started = perf_counter()
        self._partial_samples = []
        self._persist("running")

    def record_sample(self, sample: Sample) -> None:
        if self._closed:
            raise RuntimeError("campaign recorder is closed")
        self._partial_samples.append(sample)
        self._sample_file.write(json.dumps(sample.to_dict(), sort_keys=True) + "\n")

    def record_stage(
        self, stage: StageResult, *, samples_already_recorded: bool = False
    ) -> None:
        if not samples_already_recorded:
            for sample in stage.samples:
                self.record_sample(sample)
        self._sample_file.flush()
        self.stages.append(stage.summary())
        self._current = None
        self._partial_samples = []
        self._persist("running")

    def complete(self) -> Path:
        self._persist("completed")
        self.close()
        return self.destination

    def fail(
        self,
        *,
        kind: str,
        error_type: str,
        error_domain: str | None = None,
        error_kind: str | None = None,
    ) -> Path:
        termination = {"kind": kind, "error_type": error_type}
        if error_domain is not None:
            termination["error_domain"] = error_domain
        if error_kind is not None:
            termination["error_kind"] = error_kind
        self._persist("incomplete", termination=termination)
        self.close()
        return self.destination

    def close(self) -> None:
        if self._closed:
            return
        self._sample_file.flush()
        self._sample_file.close()
        self._closed = True

    def _partial_stage(self) -> dict[str, Any] | None:
        if self._current is None:
            return None
        elapsed = max(0.0, perf_counter() - self._current_started)
        stage = StageResult(
            self._current.scenario,
            self.profile.backend,
            self._current.load_shape,
            self._current.concurrency,
            self._current.offered_rps,
            elapsed,
            tuple(self._partial_samples),
            load_factor=self._current.load_factor,
            peak_model_inflight=max(
                (sample.model_peak_inflight for sample in self._partial_samples),
                default=0,
            ),
            provider_calls=sum(
                sample.provider_calls for sample in self._partial_samples
            ),
        )
        return {**stage.summary(), "phase": self._current.phase, "incomplete": True}

    def _persist(
        self,
        outcome: str,
        *,
        termination: dict[str, str] | None = None,
    ) -> None:
        report = {
            **self.metadata,
            "outcome": outcome,
            "completed": outcome == "completed",
            "termination": termination,
            "partial_stage": self._partial_stage(),
            "stages": self.stages,
        }
        _atomic_write(
            self.destination / "summary.json",
            json.dumps(report, indent=2, sort_keys=True),
        )
        rows = [_flatten_summary(item) for item in self.stages]
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=sorted(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
        _atomic_write(self.destination / "stages.csv", stream.getvalue())
        _atomic_write(self.destination / "summary.md", render_markdown(report))


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_campaign(
    output_root: Path,
    profile: LoadProfile,
    stages: Iterable[StageResult],
) -> Path:
    recorder = CampaignRecorder(output_root, profile)
    try:
        for stage in stages:
            recorder.record_stage(stage)
        return recorder.complete()
    except BaseException:
        recorder.close()
        raise


def _flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in summary.items() if not isinstance(value, dict)}
    latency = summary.get("latency_ms", {})
    for key, value in latency.items():
        result[f"latency_{key}_ms"] = value
    return result


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Pygent benchmark: {report['profile_name']}",
        "",
        f"- Backend: `{report['backend']}`",
        f"- Outcome: `{report.get('outcome', 'completed')}`",
        f"- Git: `{report['git_sha']}`",
        f"- Platform: `{report['platform']}` / Python `{report['python']}`",
        "",
        "| Scenario | Shape | Load | RPS | Success | P50 ms | P95 ms | P99 ms | Model trace P95 ms | Attempts | Trace failures | Dropped |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in report.get("stages", []):
        latency = stage["latency_ms"]
        load = stage["concurrency"] or stage["offered_rps"]
        lines.append(
            "| {scenario} | {shape} | {load} | {rps:.3f} | {success:.3f} | "
            "{p50:.3f} | {p95:.3f} | {p99:.3f} | {model_p95:.3f} | "
            "{attempts} | {trace_failures} | {dropped} |".format(
                scenario=stage["scenario"],
                shape=stage["load_shape"],
                load=load,
                rps=stage["achieved_rps"],
                success=stage["success_rate"],
                p50=latency["p50"],
                p95=latency["p95"],
                p99=latency["p99"],
                model_p95=stage.get("model_trace_p95_ms", 0.0),
                attempts=stage.get("model_attempts", 0),
                trace_failures=stage.get("trace_integrity_failures", 0),
                dropped=stage["dropped"],
            )
        )
    partial = report.get("partial_stage")
    if isinstance(partial, dict):
        lines.extend(
            [
                "",
                "## Incomplete stage",
                "",
                f"- Scenario: `{partial['scenario']}`",
                f"- Shape: `{partial['load_shape']}`",
                f"- Phase: `{partial['phase']}`",
                f"- Preserved samples: `{partial.get('completed', 0)}`",
            ]
        )
    termination = report.get("termination")
    if isinstance(termination, dict):
        lines.extend(
            [
                f"- Termination: `{termination['kind']}`",
                f"- Error type: `{termination['error_type']}`",
            ]
        )
        if termination.get("error_kind"):
            lines.append(f"- Error kind: `{termination['error_kind']}`")
        if termination.get("error_domain"):
            lines.append(f"- Error domain: `{termination['error_domain']}`")
    lines.extend(
        [
            "",
            "The report intentionally excludes endpoints, model names, credentials, prompts, and raw provider payloads.",
            "",
        ]
    )
    return "\n".join(lines)


def load_samples(path: Path) -> list[Sample]:
    source = path / "samples.jsonl" if path.is_dir() else path
    samples: list[Sample] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            samples.append(Sample(**json.loads(line)))
    return samples


__all__ = [
    "CampaignRecorder",
    "load_samples",
    "machine_metadata",
    "render_markdown",
    "write_campaign",
]
