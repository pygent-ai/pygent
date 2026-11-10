"""Contract tests for scheduler signing keys."""

from __future__ import annotations

from pathlib import Path

import pytest

from pygent.agent.lara.scheduler.keys import (
    SchedulerKeysMissing,
    init_keys,
    load_secret,
    sign,
    verify,
)


def test_sign_verify_and_source_separation(tmp_path: Path) -> None:
    init_keys(tmp_path)
    payload = {"name": "task", "schedule": "every 1h"}
    user_signature = sign(payload, load_secret(tmp_path, "user"))
    agent_signature = sign(payload, load_secret(tmp_path, "agent"))
    assert verify(payload, "user", user_signature, tmp_path)
    assert verify(payload, "agent", agent_signature, tmp_path)
    assert not verify(payload, "agent", user_signature, tmp_path)


def test_tampering_and_unsigned_are_rejected(tmp_path: Path) -> None:
    init_keys(tmp_path)
    payload = {"name": "task", "prompt": "original"}
    signature = sign(payload, load_secret(tmp_path, "user"))
    assert not verify({**payload, "prompt": "tampered"}, "user", signature, tmp_path)
    assert not verify(payload, "user", "", tmp_path)


def test_missing_keys_raise(tmp_path: Path) -> None:
    with pytest.raises(SchedulerKeysMissing):
        load_secret(tmp_path, "user")
