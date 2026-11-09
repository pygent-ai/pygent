"""Conversation-exchange envelope injected after the next tool result (D-RT1).

When the fork child answers an interrupt, the main agent must learn what
happened without being interrupted itself: the exchange rides along as a
system reminder attached to the next tool result. See
docs/design/realtime-interaction 3.3.2/3.3.3 for the timing diagrams.
"""

from __future__ import annotations

from collections.abc import Sequence

from pygent.lara.agents.context import system_reminder
from pygent.lara.core.prompt import Prompt


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


def render_exchange_envelope(
    *, user_asked: str | Sequence[str], response: str
) -> str:
    """Build the ``<system-reminder>`` envelope for one fork-reply exchange."""

    asked_block = Prompt(format_user_asked(user_asked)).wrap("user-asked")
    response_block = Prompt(response.strip()).wrap(
        "your-response-in-another-session"
    )
    return system_reminder("\n".join((str(asked_block), str(response_block))))


__all__ = ["format_user_asked", "render_exchange_envelope"]
