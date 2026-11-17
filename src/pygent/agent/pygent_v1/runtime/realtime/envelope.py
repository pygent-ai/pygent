"""Conversation-exchange envelope injected into the running turn (D-RT1).

When the fork child answers an interrupt, the main agent must learn what
happened without being interrupted itself. The exchange travels as a ReAct
projection operation — a wait-mode ``StandaloneUserMessage`` — so the react
layer applies it at its next drain point: after the pending tool results and
before the next model call. The same operation kind already carries the
memory projection replacement, so no pipeline module and no persistence of
its own are needed. See docs/design/realtime-interaction 3.3.2/3.3.3 for the
timing diagrams.
"""

from __future__ import annotations

import textwrap
from collections.abc import Sequence
from dataclasses import dataclass

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
    response: str,
    attachments: Sequence[ResolvedAttachment] = (),
) -> str:
    """Build the ``<system-reminder>`` envelope for one fork-reply exchange."""

    asked_block = Prompt(format_user_asked(user_asked)).wrap("user-asked")
    response_block = Prompt(response.strip()).wrap(
        "your-response-in-another-session"
    )
    parts = [str(asked_block)]
    if attachments:
        parts.append(render_interrupt_attachments(attachments))
    parts.append(str(response_block))
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


__all__ = [
    "EXCHANGE_MESSAGE_KIND",
    "ExchangeEnvelope",
    "format_user_asked",
    "render_exchange_envelope",
    "render_interrupt_attachments",
]
