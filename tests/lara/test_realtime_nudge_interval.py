"""Configured nudge interval reaches the runtime module."""

from __future__ import annotations

from collections.abc import Sequence

from pygent import ToolMessage, ToolResult
from pygent.lara.runtime.context import LaraContext
from pygent.lara.runtime.realtime.envelope import render_exchange_envelope
from pygent.lara.runtime.realtime.injection import RealtimeNudgeModule
from pygent.lara.runtime.realtime.nudge import nudge_reminder
from pygent.lara.runtime.realtime.speak import SPEAK_TOOL_SPEC

OTHER_INJECTION = render_exchange_envelope(
    user_asked="what changed?", response="the exchange envelope"
)


def _context() -> LaraContext:
    return LaraContext(tools=(SPEAK_TOOL_SPEC.definition,))


def _tool_message(
    *,
    call_id: str = "call-1",
    names: Sequence[str] = ("bash",),
    content: str = "",
) -> ToolMessage:
    return ToolMessage(
        content=content,
        results=tuple(
            ToolResult(
                call_id=f"{call_id}-{index}",
                name=name,
                status="succeeded",
                output="ok",
            )
            for index, name in enumerate(names)
        ),
    )


async def test_configured_interval_changes_nudge_cadence() -> None:
    module = RealtimeNudgeModule(interval=3)
    context = _context()
    message, _ = await module.forward(
        _tool_message(content=OTHER_INJECTION), context
    )
    assert message.content == OTHER_INJECTION

    for round_number in range(2, 4):
        message, _ = await module.forward(
            _tool_message(call_id=f"call-{round_number}"), context
        )
        expected = (
            nudge_reminder(rounds_since_speak=3)
            if round_number == 3
            else ""
        )
        assert message.content == expected
