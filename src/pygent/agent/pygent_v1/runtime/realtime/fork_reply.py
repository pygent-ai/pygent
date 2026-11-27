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

The host records the user message and fork reply together for the foreground to consume.
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
from pygent.core import AIMessage, Context, Message, ToolMessage
from pygent.tool import ToolDefinition, ToolResult


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
    """Copy the request state verbatim — this prefix IS the cache key.

    A snapshot taken while the foreground turn is mid-tool-batch ends with
    assistant ``tool_calls`` whose paired ``ToolMessage`` does not exist yet
    (the tools are still running). Strict providers reject the dangling
    ``tool_use`` with HTTP 400, so the tail is paired with per-call results
    that say exactly that: the calls are in flight, their output is not part
    of this snapshot. Shaping only: the fork never writes the session
    transcript, and the added tail sits outside the cached block-1 prefix.
    """

    window = tuple(messages)
    tail = window[-1] if window else None
    if isinstance(tail, AIMessage) and tail.tool_calls:
        window = (*window, _in_flight_tool_message(tail))
    return Context(
        system_prompt=system_prompt,
        messages=window,
        tools=tuple(tools),
    )


def _in_flight_tool_message(answer: AIMessage) -> ToolMessage:
    """Pair every dangling call with its in-flight status.

    The foreground batch is still running — not cancelled — so ``cancelled``
    would lie to the model. ``unknown`` keeps the side-effect verdict open
    (the fork cannot know what the running call already committed) and the
    rendered note tells the model where the work actually continues;
    ``detached`` cannot be used because it requires a ToolTask snapshot the
    fork does not own.
    """

    return ToolMessage(
        results=tuple(
            ToolResult(
                call_id=call.call_id,
                name=call.name,
                status="unknown",
                output=(
                    "the tool call is still executing in the foreground "
                    "session; its result is not part of this snapshot — "
                    "answer the user's interrupt without waiting for it"
                ),
            )
            for call in answer.tool_calls
        ),
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
