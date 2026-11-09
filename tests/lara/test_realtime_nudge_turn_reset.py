"""Turn switch resets silent-round counting."""

from __future__ import annotations

from collections.abc import Sequence

from pygent import ToolMessage, ToolResult
from pygent.lara.runtime.context import LaraContext
from pygent.lara.runtime.realtime.envelope import render_exchange_envelope
from pygent.lara.runtime.realtime.injection import RealtimeNudgeModule
from pygent.lara.runtime.realtime.speak import SPEAK_TOOL_SPEC

OTHER_INJECTION = render_exchange_envelope(
    user_asked="what changed?", response="the exchange envelope"
)


def _context(turn_id: str) -> LaraContext:
    return LaraContext(turn_id=turn_id, tools=(SPEAK_TOOL_SPEC.definition,))


def _tool_message(*, call_id: str) -> ToolMessage:
    names: Sequence[str] = ("bash",)
    return ToolMessage(
        content=OTHER_INJECTION,
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


async def test_new_turn_resets_silent_rounds() -> None:
    module = RealtimeNudgeModule()
    for round_number in range(1, 5):
        message, _ = await module.forward(
            _tool_message(call_id=f"turn-1-{round_number}"),
            _context("turn-1"),
        )
        assert message.content == OTHER_INJECTION

    new_turn, _ = await module.forward(
        _tool_message(call_id="turn-2-1"), _context("turn-2")
    )
    assert new_turn.content == OTHER_INJECTION
