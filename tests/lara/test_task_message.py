"""Tests for the scheduled task execution message."""

from __future__ import annotations

from datetime import UTC, datetime

from pygent.lara.scheduler.message import SCHEDULER_GUIDANCE, build_task_message


def test_task_message_golden_and_escaping() -> None:
    message = build_task_message(
        task_id="task-1",
        name="Nightly <build>",
        source="user",
        instructions="Check <repo> & report",
        user_message="Run <all> & summarize",
        user_identity="alice & bob",
        now=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
    )
    assert message == (
        "<task-context>\n"
        '  <task id="task-1" name="Nightly &lt;build&gt;" source="user" />\n'
        "  <user-identity>alice &amp; bob</user-identity>\n"
        "  <time>\n"
        "    Current system time: 2026-01-02 03:04:05 UTC\n"
        "  </time>\n"
        "  <instructions>\n"
        f"    {SCHEDULER_GUIDANCE}\n"
        "    \n"
        "    Check <repo> & report\n"
        "  </instructions>\n"
        "  <user-message>Run &lt;all&gt; &amp; summarize</user-message>\n"
        "</task-context>"
    )
    assert "alice & bob" not in message
    assert "<all>" not in message
