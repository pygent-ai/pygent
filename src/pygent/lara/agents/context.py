"""Forkable context windows for the LARA agent architecture.

Background agents are derived from reusable context windows rather than
compressed summaries. This module defines the window type, the sources a
window can come from, and the transforms applied while assembling a fork's
block-1 prefix. See docs/design/lara-architecture/README.md for the
architecture goals (G3 fork primitives, G4 uniform tool surface).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable

from pygent import AIMessage, Message, ToolMessage, UserMessage
from pygent.core import JsonValue
from pygent.lara.core.prompt import Prompt
from pygent.runtime.codec import message_from_dict


def _hash_text(text: str) -> str:
    """Content-address a transform payload without storing its text.

    Snapshot metadata flows into durable contexts, event stores, and
    diagnostic exports; only hashes go there. The effective content itself
    lives in the run result's context, which is the artifact that persists.
    """

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def system_reminder(text: str) -> str:
    """Wrap directive text in the ``<system-reminder>`` steering envelope."""

    stripped = text.strip()
    if not stripped:
        raise ValueError("system reminder text must be non-empty")
    return Prompt(stripped).wrap("system-reminder")


def messages_from_history(
    history: Sequence[Mapping[str, object]],
) -> tuple[Message, ...]:
    """Convert persisted conversation entries into pygent messages.

    Applies the same conversion the foreground agent applies to session
    history — pygent's wire codec plus the frame rule that system entries
    are not conversation frames — so a fork prefix built from a session
    range matches what the foreground actually saw, including assistant
    tool calls and tool results.
    """

    converted: list[Message] = []
    for item in history:
        entry = dict(item)
        if entry.get("role") == "system":
            continue
        converted.append(message_from_dict(entry))
    return tuple(converted)


@dataclass(frozen=True)
class ContextWindow:
    """A first-class, forkable agent context.

    ``ContextWindow`` is the currency of the fork primitive: it can come from
    a static message list, a session range, or the product of an earlier run,
    and any of those can be forked again (lara-architecture G3).
    """

    system_prompt: str
    messages: tuple[Message, ...] = ()
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.system_prompt, str) or not self.system_prompt.strip():
            raise ValueError("ContextWindow.system_prompt must be a non-empty string")
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class MessageSlice:
    """Message-range selector applied when materializing a fork prefix.

    Bounds use Python slice semantics (negatives and open ends work). Cuts
    are always completed to interaction-frame boundaries: a slice never
    starts on a tool result separated from its assistant tool call, and never
    ends between an assistant tool call and its results. This keeps every
    fork prefix a replayable conversation frame instead of a torn history.
    """

    start: int | None = None
    end: int | None = None

    def apply(self, messages: tuple[Message, ...]) -> tuple[Message, ...]:
        start, end = align_slice_bounds(messages, self.start, self.end)
        return messages[start:end]


def align_slice_bounds(
    messages: tuple[Message, ...], start: int | None, end: int | None
) -> tuple[int, int]:
    """Normalize a slice against frame boundaries; returns ``(start, end)``.

    A frame begins at a user message or at an assistant message (with or
    without tool calls); tool results always belong to the assistant call
    frame that precedes them. The returned bounds may be wider than
    requested when a requested cut would tear a frame.
    """

    total = len(messages)
    resolved_start = 0 if start is None else (start + total if start < 0 else start)
    resolved_end = total if end is None else (end + total if end < 0 else end)
    resolved_start = max(0, min(resolved_start, total))
    resolved_end = max(resolved_start, min(resolved_end, total))
    # A tool result never begins a frame: walk back to its assistant call.
    # The start < total guard keeps empty slices (start == total) empty.
    while 0 < resolved_start < total and isinstance(
        messages[resolved_start], ToolMessage
    ):
        resolved_start -= 1
    # Never end between an assistant tool call and its results, nor between
    # the results of one call: trailing tool results of an included call are
    # always pulled into the slice.
    while resolved_end < total and isinstance(messages[resolved_end], ToolMessage):
        resolved_end += 1
    return resolved_start, resolved_end


@runtime_checkable
class ContextSource(Protocol):
    """Anything a fork can derive its block-1 prefix from.

    ``describe()`` identifies the freeze point a fork is based on; sources
    backed by durable state (session ranges) must describe their session and
    cursor so snapshots are auditable and replayable.
    """

    def resolve(self) -> ContextWindow: ...

    def describe(self) -> str: ...


@runtime_checkable
class ContextTransform(Protocol):
    """A window-to-window rewrite applied while planning a fork.

    Implementations must keep interaction frames intact: a transform may not
    insert messages between an assistant tool call and its results, or
    between the results of one call. ``describe()`` returns a structured,
    JSON-serializable, **content-addressed** identity of the transform:
    payloads appear as hashes, never as text, so snapshot metadata cannot
    spread directive or prompt content into durable stores. The effective
    content lives in the run result's context.
    """

    def apply(self, window: ContextWindow) -> ContextWindow: ...

    def describe(self) -> dict[str, JsonValue]: ...


@dataclass(frozen=True)
class ReminderTransform:
    """Inject a ``<system-reminder>`` user message into the prefix.

    ``position=None`` appends after the last message; integer positions follow
    ``list.insert`` semantics (negative values count from the end). The
    insertion point is snapped to the nearest legal frame boundary: a
    reminder can never land between an assistant tool call and its results,
    or between the results of one call.
    """

    text: str
    position: int | None = None

    def apply(self, window: ContextWindow) -> ContextWindow:
        reminder = UserMessage(content=system_reminder(self.text))
        messages = list(window.messages)
        if self.position is None:
            position = len(messages)
        else:
            position = self.position
            if position < 0:
                position += len(messages) + 1
        position = max(0, min(position, len(messages)))
        # Snap to a legal frame boundary: an insertion at index i is legal
        # when it does not split an assistant tool call from its results.
        while position > 0:
            previous = messages[position - 1]
            if isinstance(previous, ToolMessage) or (
                isinstance(previous, AIMessage) and previous.tool_calls
            ):
                position -= 1
            else:
                break
        messages.insert(position, reminder)
        return replace(window, messages=tuple(messages))

    def describe(self) -> dict[str, JsonValue]:
        # Content-addressed: the wrapped text is hashed, not stored, so
        # sensitive directive content cannot spread through snapshot
        # metadata. The effective text remains in the run result's context.
        return {
            "kind": "reminder",
            "position": self.position,
            "text_hash": _hash_text(system_reminder(self.text)),
        }


@dataclass(frozen=True)
class SystemPromptTransform:
    """Override the window's system prompt (G3: prompt rewrite freedom)."""

    system_prompt: str

    def __post_init__(self) -> None:
        if not self.system_prompt.strip():
            raise ValueError("SystemPromptTransform.system_prompt must be non-empty")

    def apply(self, window: ContextWindow) -> ContextWindow:
        return replace(window, system_prompt=self.system_prompt)

    def describe(self) -> dict[str, JsonValue]:
        # Content-addressed for the same reason as ReminderTransform.
        return {
            "kind": "system_prompt",
            "system_prompt_hash": _hash_text(self.system_prompt),
        }
