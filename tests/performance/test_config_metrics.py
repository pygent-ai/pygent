from __future__ import annotations

import json
from dataclasses import replace

import pytest

from benchmarks.cli import _profile, dry_run, main
from benchmarks.config import LoadProfile, ModelSettings, load_profile
from benchmarks.metrics import compare_summaries, percentile
from benchmarks.models import LiveModelConfig


@pytest.mark.performance
def test_profiles_are_validated_and_live_has_a_hard_concurrency_limit():
    smoke = load_profile("benchmarks/profiles/synthetic-smoke.toml")
    live = load_profile("benchmarks/profiles/live-long.toml")
    assert smoke.backend == "synthetic"
    assert smoke.digest == smoke.digest
    assert smoke.estimated_seconds() > 0
    assert live.model.retry_max_attempts == 2
    assert live.model.retry_on == ("timeout", "unavailable")
    assert live.model.attempt_timeout_seconds == 90.0
    assert live.request_deadline_seconds > (
        live.model.attempt_timeout_seconds * live.model.retry_max_attempts
        + live.model.retry_backoff_seconds
    )

    with pytest.raises(ValueError, match="hard-limited"):
        replace(
            load_profile("benchmarks/profiles/live-long.toml"),
            concurrency=(64,),
        )


@pytest.mark.performance
def test_dry_run_and_live_config_never_disclose_secrets(capsys):
    secret = "do-not-print-this"
    config = LiveModelConfig.from_environment(
        {
            "GLM_API_BASE": "https://models.example/v1",
            "GLM_API_KEY": secret,
            "GLM_MODEL_NAME": "private-model",
        }
    )
    assert secret not in repr(config)
    assert "models.example" not in repr(config)
    report = dry_run(load_profile("benchmarks/profiles/live-long.toml"))
    assert report["estimated_live_model_calls_closed_loop"] > 0
    assert report["estimated_live_model_attempts_closed_loop_max"] == (
        report["estimated_live_model_calls_closed_loop"] * 2
    )
    assert main(("dry-run", "synthetic-smoke")) == 0
    output = capsys.readouterr().out
    assert secret not in output
    assert json.loads(output)["profile"] == "synthetic-smoke"


@pytest.mark.performance
def test_every_live_long_step_is_independent_and_under_ten_hours():
    aggregate = load_profile("benchmarks/profiles/live-long.toml")

    for scenario in aggregate.scenarios:
        step = _profile(f"live-long-{scenario}")
        report = dry_run(step)

        assert step.scenarios == (scenario,)
        assert step.backend == "live"
        assert report["estimated_duration_hours"] < 10


@pytest.mark.performance
def test_unknown_live_long_step_fails_as_a_configuration_error(capsys):
    assert main(("dry-run", "live-long-not-a-scenario")) == 2
    assert "configuration_error" in capsys.readouterr().err


@pytest.mark.performance
def test_percentiles_and_default_regression_thresholds():
    assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert compare_summaries(
        {"achieved_rps": 100.0, "p95_ms": 20.0},
        {"achieved_rps": 84.0, "p95_ms": 20.0},
    )["reasons"] == ["throughput"]
    assert compare_summaries(
        {"achieved_rps": 100.0, "p95_ms": 20.0},
        {"achieved_rps": 100.0, "p95_ms": 26.0},
    )["reasons"] == ["p95"]


@pytest.mark.performance
def test_invalid_profile_shapes_fail_closed():
    with pytest.raises(ValueError, match="at least one scenario"):
        LoadProfile(
            name="invalid",
            backend="synthetic",
            scenarios=(),
            concurrency=(1,),
            closed_duration_seconds=1.0,
            open_multipliers=(),
            open_duration_seconds=0.0,
            model=ModelSettings(),
        )

    with pytest.raises(ValueError, match="retry_on is required"):
        ModelSettings(retry_max_attempts=2)

    with pytest.raises(ValueError, match="attempt_timeout_seconds"):
        ModelSettings(attempt_timeout_seconds=0)


@pytest.mark.performance
def test_compare_command_reduces_repetitions_and_enforces_gate(tmp_path, capsys):
    def report(rps: float, p95: float) -> dict[str, object]:
        return {
            "stages": [
                {
                    "scenario": "direct-invoke",
                    "load_shape": "closed",
                    "concurrency": 4,
                    "load_factor": None,
                    "achieved_rps": rps + offset,
                    "latency_ms": {"p95": p95 + offset},
                }
                for offset in (-1.0, 0.0, 1.0)
            ]
        }

    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps(report(100.0, 20.0)), encoding="utf-8")
    candidate.write_text(json.dumps(report(80.0, 20.0)), encoding="utf-8")
    assert main(("compare", str(baseline), str(candidate))) == 1
    result = json.loads(capsys.readouterr().out)
    assert not result["passed"]
    assert next(iter(result["stages"].values()))["reasons"] == ["throughput"]
