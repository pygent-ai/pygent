"""Conversation-exchange envelope injected into the running turn (D-RT1).

When the fork child answers an interrupt, the main agent must learn what
happened without being interrupted itself. The exchange travels as a ReAct
projection operation — a wait-mode ``StandaloneUserMessage`` — so the react
layer applies it at its next drain point: after the pending tool results and
before the next model call. The delivery journal retains the exchange until the foreground takes it.
"""

from __future__ import annotations

import textwrap
from collections.abc import Sequence
from dataclasses import dataclass, field

from pygent.agent.pygent_v1.core.prompt import Prompt, system_reminder
from pygent.agent.pygent_v1.runtime.attachments import (
    ResolvedAttachment,
    render_attachments_xml,
)
from pygent.agent.react_projection_operations import (
    StandaloneUserMessage,
    SteeringMode,
    encode_react_projection_operation,
)
from pygent.core import JsonValue, UserMessage

EXCHANGE_MESSAGE_KIND = "pygent.v1.realtime.exchange"

_ALREADY_ANSWERED_DIRECTIVE = """\
The exchange above was already answered in another session — the
your-response-in-another-session block quotes that reply verbatim. Treat both
blocks as transcript context only.
Do not re-answer the question or repeat the reply."""


def format_user_asked(user_asked: str | Sequence[str]) -> str:
    """Format one message or a numbered supersede-and-merge batch."""

    entries = [user_asked] if isinstance(user_asked, str) else list(user_asked)
    if not entries:
        return ""
    if len(entries) == 1:
        return entries[0].strip()
    return "\n".join(
        f"{index}. {entry.strip()}" for index, entry in enumerate(entries, start=1)
    )


def render_interrupt_attachments(attachments: Sequence[ResolvedAttachment]) -> str:
    """The ``<attachments>`` block as a sibling of unindented interrupt frames.

    ``render_attachments_xml`` indents for its place inside ``<user-context>``;
    the fork request and the exchange envelope place it at the top level.
    """

    return textwrap.dedent(render_attachments_xml(attachments))


def render_exchange_envelope(
    *,
    user_asked: str | Sequence[str],
    response: str | None = None,
    attachments: Sequence[ResolvedAttachment] = (),
) -> str:
    """Build the ``<system-reminder>`` envelope for one fork-reply exchange.

    A delivered reply is followed by the ``already-answered`` frame: the
    foreground agent must treat the exchange as transcript context and not
    re-answer the question the fork just handled.

    ``response=None`` renders the unanswered-interrupt form: only the question
    (and attachments) reach the main agent — the total-fork-failure fallback,
    so an accepted interrupt is never silently dropped from the context. No
    directive rides along: the question is open and the foreground agent is
    expected to answer it.
    """

    asked_block = Prompt(format_user_asked(user_asked)).wrap("user-asked")
    parts = [str(asked_block)]
    if attachments:
        parts.append(render_interrupt_attachments(attachments))
    if response is not None:
        response_block = Prompt(response.strip()).wrap(
            "your-response-in-another-session"
        )
        parts.append(str(response_block))
        parts.append(
            str(Prompt(_ALREADY_ANSWERED_DIRECTIVE).wrap("already-answered"))
        )
    return system_reminder("\n".join(parts))


@dataclass(frozen=True, slots=True)
class ExchangeEnvelope:
    """One rendered fork-reply exchange addressed to its session's main agent.

    ``exchange_id`` is the delivery identity: it keys the execution input when
    the exchange is injected into a running turn, so a retried delivery is a
    duplicate rather than a second envelope.
    """

    exchange_id: str
    session_id: str
    content: str
    # Workspace-relative paths of the user's attachments; the transcript
    # checkpoint records them on the steering message the main agent reads.
    attachments: tuple[str, ...] = ()

    source_input_ids: tuple[str, ...] = field(default=(), compare=False)
    reply: JsonValue = field(default=None, compare=False)

    def __post_init__(self) -> None:
        for name in ("exchange_id", "session_id", "content"):
            if not getattr(self, name).strip():
                raise ValueError(f"ExchangeEnvelope.{name} must be non-empty")
        object.__setattr__(self, "attachments", tuple(self.attachments))

    def as_user_message(self, *, case_run_id: str) -> UserMessage:
        """The message the main agent reads; ``data`` binds it to one run so
        the transcript checkpoint can tell it apart from carried history."""

        data: dict[str, JsonValue] = {
            "case_run_id": case_run_id,
            "input_id": self.exchange_id,
        }
        if self.attachments:
            data["attachments"] = self.attachments
        return UserMessage(
            content=self.content,
            kind=EXCHANGE_MESSAGE_KIND,
            data=data,
        )

    def projection_operation(self, *, case_run_id: str) -> JsonValue:
        """Encode the wait-mode steering operation for ``send_input``."""

        return encode_react_projection_operation(
            StandaloneUserMessage(
                self.as_user_message(case_run_id=case_run_id),
                SteeringMode.WAIT,
            )
        )


def build_exchange_envelope(
    *,
    session_id: str,
    input_id: str,
    user_asked: str | Sequence[str],
    response: str | None,
    attachments: Sequence[ResolvedAttachment] = (),
) -> ExchangeEnvelope:
    """The session-addressed envelope for one fork exchange (D-RT3).

    ``response=None`` is the unanswered-interrupt form used after a total fork
    failure: the question reaches the main agent without a fabricated reply.
    """

    return ExchangeEnvelope(
        exchange_id=f"realtime-exchange:{input_id}",
        session_id=session_id,
        content=render_exchange_envelope(
            user_asked=user_asked,
            response=response,
            attachments=attachments,
        ),
        attachments=tuple(item.relative for item in attachments),
    )


__all__ = [
    "EXCHANGE_MESSAGE_KIND",
    "ExchangeEnvelope",
    "build_exchange_envelope",
    "format_user_asked",
    "render_exchange_envelope",
    "render_interrupt_attachments",
]
