"""Contract tests for scheduler schedule parsing and advancement."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from pygent.lara.scheduler.schedule import advance, next_run, parse_schedule


def test_parse_all_schedule_kinds() -> None:
    interval = parse_schedule("every 30m")
    assert interval.kind == "interval"
    assert interval.interval_seconds == 1800
    once = parse_schedule("at 2026-05-01T12:30:00+00:00")
    assert once.kind == "once"
    assert once.at == datetime(2026, 5, 1, 12, 30, tzinfo=UTC)
    daily = parse_schedule("daily 09:05")
    assert daily.kind == "daily"
    assert daily.time_of_day is not None
    event = parse_schedule("on build.finished")
    assert event.kind == "event"
    assert event.event == "build.finished"


@pytest.mark.parametrize(
    "text",
    ["", "every 0m", "every soon", "at nope", "daily 24:00", "on", "cron * *"],
)
def test_invalid_schedules(text: str) -> None:
    with pytest.raises(ValueError):
        parse_schedule(text)


def test_next_run_and_advance_are_deterministic() -> None:
    now = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    every = parse_schedule("every 30m")
    assert next_run(every, after=now, now=now) == datetime(
        2026, 1, 1, 10, 30, tzinfo=UTC
    )
    daily = parse_schedule("daily 09:00")
    assert next_run(daily, after=now, now=now) == datetime(
        2026, 1, 2, 9, 0, tzinfo=UTC
    )
    once = parse_schedule("at 2026-01-01T09:00:00+00:00")
    assert next_run(once, after=now, now=now) is None
    assert advance(once, now) is None
    assert next_run(parse_schedule("on build"), after=now, now=now) is None
