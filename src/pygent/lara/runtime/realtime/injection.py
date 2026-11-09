"""Tool-result boundary nudge module (Phase 2 wiring).

The nudge rides the same injection seam as the reminders: after each tool
result, if the agent has been silent (no speak call) for ``interval``
rounds, the reminder text is appended to the tool result so the model
decides whether to speak. A plain mutable counter is the whole state, and
the first-result flags are keyed by ``context.turn_id`` because one module
instance may serve several turns. A crash restarts counting, which is
acceptable for a soft UX reminder.

The same seam also carries the first-result reminder: the first tool result
of a turn is the last moment the agent can tell the user what it is about to
do before working silently. If the turn has not spoken yet and no other
injection claimed the boundary, the reminder is appended there once.
"""

from __future__ import annotations

from dataclasses import replace

from pygent import Module, ToolMessage
from pygent.lara.runtime.context import LaraContext

from .nudge import DEFAULT_NUDGE_INTERVAL, nudge_reminder
from .speak import SPEAK_TOOL_NAME

FIRST_RESULT_SPEAK_REMINDER = (
    "The user cannot see your plan — you haven't told them what you are "
    "about to do. Call speak(text=...) with a brief plan now."
)


class _NudgeCounter:
    """Mutable per-turn round and first-result state.

    Pygent freezes Module attributes after the first invoke, so the counter
    must live on a referenced object — rebinding ``self._counter`` would be
    rejected, mutating its fields is not.
    """

    __slots__ = ("first_result_seen", "rounds", "spoke_yet", "turn_id")

    def __init__(self) -> None:
        self.rounds = 0
        self.turn_id: str | None = None
        self.first_result_seen = False
        self.spoke_yet = False


class RealtimeNudgeModule(Module[ToolMessage, ToolMessage]):
    """Append the silent-rounds and first-result reminders at tool boundaries.

    The round counter fires on exact multiples of ``interval`` (5, 10, ...
    rounds) so the reminder stays periodic instead of spamming every
    boundary. Any speak call in the round's tool results resets the counter.

    The first-result reminder fires at most once per turn: after the first
    tool result, when the turn has not called speak, and when no earlier
    module already injected content into this boundary. A boundary that
    carries an exchange envelope or runtime reminder keeps that injection
    only.

    The counter is a trusted live resource: deliberately ephemeral (lost on
    crash, which restarts counting) and never checkpointed.
    """

    trusted_live_resource_attributes = ("counter",)

    def __init__(self, interval: int = DEFAULT_NUDGE_INTERVAL) -> None:
        if interval <= 0:
            raise ValueError("interval must be positive")
        super().__init__()
        self.interval = interval
        self.counter = _NudgeCounter()

    async def forward(
        self, message: ToolMessage, context: LaraContext
    ) -> tuple[ToolMessage, LaraContext]:
        counter = self.counter
        if counter.turn_id != context.turn_id:
            counter.turn_id = context.turn_id
            counter.rounds = 0
            counter.first_result_seen = False
            counter.spoke_yet = False

        spoke = any(
            result.name == SPEAK_TOOL_NAME for result in message.results
        )
        if spoke:
            counter.spoke_yet = True
        counter.rounds = 0 if spoke else counter.rounds + 1

        first_result = not counter.first_result_seen
        counter.first_result_seen = True
        if (
            first_result
            and not counter.spoke_yet
            and not message.content
            and any(
                definition.name == SPEAK_TOOL_NAME
                for definition in context.tools
            )
        ):
            return (
                replace(message, content=FIRST_RESULT_SPEAK_REMINDER),
                context,
            )

        if counter.rounds == 0 or counter.rounds % self.interval != 0:
            return message, context
        reminder = nudge_reminder(rounds_since_speak=counter.rounds)
        content = (
            f"{message.content}\n{reminder}" if message.content else reminder
        )
        return replace(message, content=content), context


__all__ = ["FIRST_RESULT_SPEAK_REMINDER", "RealtimeNudgeModule"]
