"""Fork-snapshot request construction and speak extraction (D-RT2/D-RT3).

The interrupt path copies the main agent's request state verbatim (cache
prefix), appends the user's message, and runs ONE model call. The call itself
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

from pygent import AIMessage, Context, Message, ToolDefinition
from pygent.agent.lara.runtime.realtime.speak import (
    SPEAK_TOOL_NAME,
    SpeakArgs,
    parse_speak_args,
)


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
        return parse_speak_args(dict(call.arguments))
    return None


__all__ = [
    "SPEAK_TOOL_NAME",
    "ForkReplyOutcome",
    "build_fork_context",
    "extract_speak_args",
]
