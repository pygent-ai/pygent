from __future__ import annotations

import json
import os
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from tests.live.az_conformance import results as results_module
from tests.live.az_conformance.results import (
    ErrorKind,
    ProbeResult,
    ResultLedger,
    result_from_mapping,
)
from tests.live.az_conformance.schemas import Scenario

_SNAPSHOT_SHA = "7" * 64


def _passed(**changes: object) -> ProbeResult:
    values: dict[str, object] = {
        "snapshot_sha256": _SNAPSHOT_SHA,
        "source_revision": "abc1234",
        "route_id": "gpt-5.4",
        "canonical_provider": "openai",
        "canonical_model_id": "gpt-5.4",
        "protocol": "openai_chat_completions",
        "scenario": Scenario.TEXT,
        "status": "passed",
        "attempts": 1,
        "error_kind": None,
        "private_detail": None,
    }
    values.update(changes)
    return ProbeResult(**values)  # type: ignore[arg-type]


def test_result_key_includes_snapshot_revision_route_protocol_and_scenario() -> None:
    result = _passed()
    assert result.key == (
        _SNAPSHOT_SHA,
        "abc1234",
        "gpt-5.4",
        "openai_chat_completions",
        Scenario.TEXT,
    )


def test_public_json_excludes_private_detail_and_secrets() -> None:
    result = _passed(
        status="failed",
        error_kind=ErrorKind.INVALID_RESPONSE,
        private_detail="Bearer secret response body",
    )
    serialized = json.dumps(result.to_public_mapping())
    assert "secret" not in serialized
    assert "response body" not in serialized
    assert "private_detail" not in serialized
    assert "Bearer" not in repr(result)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"attempts": 0}, "attempts"),
        ({"status": "skipped"}, "status"),
        ({"status": "failed", "error_kind": None}, "error_kind"),
        ({"status": "passed", "error_kind": "timeout"}, "error_kind"),
        ({"scenario": "not-real"}, "scenario"),
        ({"snapshot_sha256": "short"}, "snapshot_sha256"),
        ({"canonical_provider": None}, "canonical identity"),
    ],
)
def test_result_mapping_is_strict(changes: dict[str, object], message: str) -> None:
    mapping = _passed().to_public_mapping()
    mapping.update(changes)
    with pytest.raises((TypeError, ValueError), match=message):
        result_from_mapping(mapping)

    with pytest.raises((TypeError, ValueError), match="unknown fields"):
        result_from_mapping({**_passed().to_public_mapping(), "extra": True})


def test_result_is_immutable() -> None:
    result = _passed()
    with pytest.raises(FrozenInstanceError):
        result.attempts = 2  # type: ignore[misc]


def test_ledger_writes_atomically_and_round_trips(tmp_path: Path) -> None:
    checkpoint = tmp_path / "results.jsonl"
    ledger = ResultLedger(checkpoint)
    ledger.record(_passed())

    assert checkpoint.is_file()
    assert not checkpoint.with_suffix(".jsonl.tmp").exists()
    lines = checkpoint.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert ResultLedger.load(checkpoint).results == (_passed(),)


def test_ledger_retries_when_atomic_replace_is_temporarily_denied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "results.jsonl"
    real_replace = os.replace
    attempts = 0

    def replace_once_unavailable(source: str | Path, target: str | Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("target is temporarily open by a reader")
        real_replace(source, target)

    monkeypatch.setattr(results_module.os, "replace", replace_once_unavailable)

    ledger = ResultLedger(checkpoint)
    ledger.record(_passed())

    assert attempts == 2
    assert ResultLedger.load(checkpoint).results == (_passed(),)
    assert not tuple(tmp_path.glob("*.tmp"))


def test_ledger_rejects_malformed_or_duplicate_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "results.jsonl"
    checkpoint.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint"):
        ResultLedger.load(checkpoint)

    line = json.dumps(_passed().to_public_mapping(), sort_keys=True)
    checkpoint.write_text(f"{line}\n{line}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        ResultLedger.load(checkpoint)


def test_ledger_reuses_only_exact_passed_key(tmp_path: Path) -> None:
    ledger = ResultLedger(tmp_path / "results.jsonl")
    passed = _passed()
    failed = _passed(
        scenario=Scenario.TEXT_STREAM,
        status="failed",
        error_kind=ErrorKind.TIMEOUT,
    )
    ledger.record(passed)
    ledger.record(failed)

    assert ledger.reusable_pass(passed.key) == passed
    assert ledger.reusable_pass(failed.key) is None
    assert ledger.reusable_pass(
        (
            _SNAPSHOT_SHA,
            "different-revision",
            "gpt-5.4",
            "openai_chat_completions",
            Scenario.TEXT,
        )
    ) is None


def test_new_pass_supersedes_failure_for_same_key(tmp_path: Path) -> None:
    ledger = ResultLedger(tmp_path / "results.jsonl")
    failed = _passed(
        status="failed", error_kind=ErrorKind.TIMEOUT, private_detail="transient"
    )
    ledger.record(failed)
    passed = _passed(attempts=2)
    ledger.record(passed)

    assert ledger.results == (passed,)
    assert ResultLedger.load(ledger.path).results == (passed,)
