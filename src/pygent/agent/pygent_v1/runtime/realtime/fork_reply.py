"""Fork-snapshot request construction and speak extraction (D-RT2/D-RT3).

The interrupt path copies the main agent's request state verbatim (cache
prefix), appends the user's message, and runs the model call; an undeliverable
answer (no speak call, empty content) retries once with the interrupt wrapped
and reply instructions in the user-message tail (``fork_agent``). The call
itself
is a pygent managed execution (see ``fork_agent.ForkReplyModule``) so it
shares admission, capacity coordination, journaling and recovery with every
other runtime call. Delivery is capability-gated (D-RT3): the main path
forces ``tool_choice="speak"`` and extracts the speak arguments; a stray text
response is deliverable through the harness's fallback rendering
(``via="text"``).

The fork reply is transient by design: it never enters the session transcript
(fork agents carry no conversation-checkpoint modules), and a lost reply is
simply re-asked.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pygent.agent.pygent_v1.runtime.realtime.speak import (
    SPEAK_TOOL_NAME,
    SpeakArgs,
    SpeakArgsError,
    parse_speak_args,
)
from pygent.core import AIMessage, Context, Message
from pygent.tool import ToolDefinition


@dataclass(frozen=True)
class ForkReplyOutcome:
    """One fork reply; ``via`` is ``"speak"`` (main path) or ``"text"`` (fallback)."""

    via: str
    speak_args: SpeakArgs | None
    text: str | None
    message: AIMessage


def build_fork_context(
    *,
    system_prompt: str,
    messages: Sequence[Message],
    tools: Sequence[ToolDefinition],
) -> Context:
    """Copy the request state verbatim — this prefix IS the cache key."""

    return Context(
        system_prompt=system_prompt,
        messages=tuple(messages),
        tools=tuple(tools),
    )


def extract_speak_args(message: AIMessage) -> SpeakArgs | None:
    """Find the speak call in a model response and validate its arguments.

    ``ToolCall.arguments`` is frozen at construction (``FrozenJsonObject`` —
    a Mapping, not a dict), so the raw value is normalized via ``dict``.
    """

    for call in message.tool_calls:
        if call.name != SPEAK_TOOL_NAME:
            continue
        try:
            return parse_speak_args(dict(call.arguments))
        except SpeakArgsError:
            # Malformed speak arguments (e.g. empty args) count as "no speak
            # call": the answer falls through to the text path, or — when the
            # content is empty too — to the wrapped-message retry
            # (``fork_agent``) instead of aborting the fork.
            return None
    return None


__all__ = [
    "SPEAK_TOOL_NAME",
    "ForkReplyOutcome",
    "build_fork_context",
    "extract_speak_args",
]
