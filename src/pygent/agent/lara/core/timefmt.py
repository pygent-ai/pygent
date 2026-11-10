"""Shared formatting for the ``<time>`` context block.

Both the reminder renderer and the scheduler task message emit the same
"Current system time" line; the format lives here so the two cannot drift.
"""

from __future__ import annotations

from datetime import datetime

TIME_LINE_FORMAT = "%Y-%m-%d %H:%M:%S %Z"


def format_current_time_line(moment: datetime) -> str:
    """Render the canonical ``Current system time: ...`` line."""

    return f"Current system time: {moment.strftime(TIME_LINE_FORMAT)}"


__all__ = ["TIME_LINE_FORMAT", "format_current_time_line"]
