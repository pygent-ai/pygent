"""Schedule parsing and deterministic advancement.

Supported syntaxes: ``every <n>[smhd]``, ``at <ISO timestamp>``,
``daily HH:MM``, and ``on <event>``. All returned datetimes are timezone-aware
UTC values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from typing import Literal

ScheduleKind = Literal["interval", "once", "daily", "event"]
_EVERY_RE = re.compile(r"^every\s+(\d+)\s*([smhd])$", re.IGNORECASE)
_DAILY_RE = re.compile(r"^daily\s+(\d{1,2}):(\d{2})$")
_ON_RE = re.compile(r"^on\s+(\S+)$")
_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


@dataclass(frozen=True, slots=True)
class Schedule:
    kind: ScheduleKind
    interval_seconds: int | None = None
    at: datetime | None = None
    time_of_day: time | None = None
    event: str | None = None


def _utc(value: datetime) -> datetime:
    return (
        value.replace(tzinfo=UTC)
        if value.tzinfo is None
        else value.astimezone(UTC)
    )


def parse_schedule(spec: str) -> Schedule:
    text = " ".join(str(spec).split())
    match = _EVERY_RE.fullmatch(text)
    if match:
        amount = int(match.group(1))
        if amount <= 0:
            raise ValueError("every interval must be positive")
        return Schedule(
            kind="interval",
            interval_seconds=amount * _UNITS[match.group(2).lower()],
        )
    if text.lower().startswith("at "):
        value = text[3:].strip()
        if not value:
            raise ValueError("at schedule requires an ISO timestamp")
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"invalid ISO timestamp: {value!r}") from exc
        return Schedule(kind="once", at=_utc(parsed))
    match = _DAILY_RE.fullmatch(text)
    if match:
        hour, minute = int(match.group(1)), int(match.group(2))
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError("daily time must be HH:MM")
        return Schedule(kind="daily", time_of_day=time(hour, minute))
    match = _ON_RE.fullmatch(text)
    if match:
        return Schedule(kind="event", event=match.group(1))
    raise ValueError(
        "schedule must be 'every <n>[smhd]', 'at <ISO>', 'daily HH:MM', or 'on <event>'"
    )


def next_run(
    schedule: Schedule, *, after: datetime, now: datetime | None = None
) -> datetime | None:
    current = _utc(now or after)
    point = _utc(after)
    if schedule.kind == "interval":
        assert schedule.interval_seconds is not None
        return current + timedelta(seconds=schedule.interval_seconds)
    if schedule.kind == "once":
        assert schedule.at is not None
        return schedule.at if schedule.at > point else None
    if schedule.kind == "daily":
        assert schedule.time_of_day is not None
        candidate = datetime.combine(
            current.date(), schedule.time_of_day, tzinfo=UTC
        )
        if candidate <= current:
            candidate += timedelta(days=1)
        return candidate
    return None


def advance(schedule: Schedule, fired_at: datetime) -> datetime | None:
    return next_run(schedule, after=fired_at, now=fired_at)


__all__ = ["Schedule", "advance", "next_run", "parse_schedule"]
