"""First-tool-result speak reminder at the realtime injection boundary."""

from __future__ import annotations

from collections.abc import Sequence

from pygent import ToolMessage, ToolResult

from pygent.lara.runtime.context import LaraContext
from pygent.lara.runtime.realtime.envelope import render_exchange_envelope
from pygent.lara.runtime.realtime.injection import (
    FIRST_RESULT_SPEAK_REMINDER,
    RealtimeNudgeModule,
)
from pygent.lara.runtime.realtime.nudge import nudge_reminder
from pygent.lara.runtime.realtime.speak import SPEAK_TOOL_NAME, SPEAK_TOOL_SPEC

OTHER_INJECTION = render_exchange_envelope(
    user_asked="what changed?",
    response="the exchange envelope",
)


def _context(
    *,
    turn_id: str = "turn-1",
    include_speak: bool = True,
) -> LaraContext:
    return LaraContext(
        turn_id=turn_id,
        tools=(SPEAK_TOOL_SPEC.definition,) if include_speak else (),
    )


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


async def test_first_tool_result_without_speak_appends_reminder() -> None:
    module = RealtimeNudgeModule()

    message, _ = await module.forward(_tool_message(), _context())

    assert message.content == FIRST_RESULT_SPEAK_REMINDER


async def test_first_action_speak_skips_reminder() -> None:
    module = RealtimeNudgeModule()

    message, _ = await module.forward(
        _tool_message(names=(SPEAK_TOOL_NAME, "bash")),
        _context(),
    )

    assert message.content == ""


async def test_exchange_envelope_on_same_boundary_skips_reminder() -> None:
    module = RealtimeNudgeModule()

    message, _ = await module.forward(
        _tool_message(content=OTHER_INJECTION),
        _context(),
    )

    assert message.content == OTHER_INJECTION


async def test_later_tool_results_do_not_repeat_the_first_reminder() -> None:
    module = RealtimeNudgeModule()

    first, _ = await module.forward(_tool_message(), _context())
    second, _ = await module.forward(
        _tool_message(call_id="call-2"), _context()
    )
    third, _ = await module.forward(
        _tool_message(call_id="call-3"), _context()
    )

    assert first.content == FIRST_RESULT_SPEAK_REMINDER
    assert second.content == ""
    assert third.content == ""


async def test_new_turn_gets_its_own_first_result_reminder() -> None:
    module = RealtimeNudgeModule()

    first_turn, _ = await module.forward(_tool_message(), _context())
    second_turn, _ = await module.forward(
        _tool_message(call_id="call-2"),
        _context(turn_id="turn-2"),
    )

    assert first_turn.content == FIRST_RESULT_SPEAK_REMINDER
    assert second_turn.content == FIRST_RESULT_SPEAK_REMINDER


async def test_missing_speak_tool_skips_reminder() -> None:
    module = RealtimeNudgeModule()

    message, _ = await module.forward(
        _tool_message(), _context(include_speak=False)
    )

    assert message.content == ""


async def test_periodic_nudge_still_fires_every_five_rounds() -> None:
    module = RealtimeNudgeModule()
    context = _context()
    # The first boundary already carries an injection, so the first-result
    # reminder stays out of the way and the periodic rule is isolated.
    message, _ = await module.forward(
        _tool_message(content=OTHER_INJECTION), context
    )
    assert message.content == OTHER_INJECTION

    for round_number in range(2, 10):
        message, _ = await module.forward(
            _tool_message(call_id=f"call-{round_number}"), context
        )
        expected = (
            nudge_reminder(rounds_since_speak=round_number)
            if round_number == 5
            else ""
        )
        assert message.content == expected

    message, _ = await module.forward(
        _tool_message(call_id="call-10"), context
    )
    assert message.content == nudge_reminder(rounds_since_speak=10)
