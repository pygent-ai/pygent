"""Rendered-prompt file naming stays safe for realtime turn ids."""

from __future__ import annotations

from pygent.lara.tracing.events import _turn_file_name


def test_turn_file_name_preserves_normal_turn_ids() -> None:
    assert _turn_file_name("turn-0001") == "turn-0001"


def test_turn_file_name_sanitizes_realtime_supersede_turn_ids() -> None:
    # Supersede forks are named fork:<input>:g<generation>; raw colons are
    # invalid in Windows file names.
    assert _turn_file_name("fork:user-2:g2") == "fork-user-2-g2"


def test_turn_file_name_falls_back_for_empty_ids() -> None:
    assert _turn_file_name(None) == "turn-unknown"
    assert _turn_file_name("") == "turn-unknown"
