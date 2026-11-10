"""The continuation turn: post-chain delivery for realtime sessions (D-RT1/RT11).

When a turn chain ends with user input still pending — a lane-queued turn, or
an exchange envelope the fork answered after every claim point passed — the
plain finalizer round is the wrong vehicle: it would decide against a stale
draft pointer, and its ``:finalize`` key pins the inbox shut afterwards. The
coordinator submits a continuation turn instead: a normal turn whose opening
context carries the queued envelopes (via the turn-start claim) and whose
message quotes any undelivered draft. Its own ending receives the standard
bottom-line check, so nothing in the chain bypasses the speak contract.
"""

from __future__ import annotations

from pygent.lara.core.prompt import Prompt, system_reminder

CONTINUATION_MESSAGE_KIND = "lara.realtime.continuation"

_DRAFT_DELIVERY_BODY = (
    "Your previous turn ended without delivering its final text — speak is the "
    "only channel the user sees, and the draft below has never reached them. "
    "Deliver it now with speak, rephrasing freely and reconciling it with "
    "anything that happened since; if it has become obsolete, call "
    "speak(say_nothing=true) instead."
)

_PENDING_INPUT_BODY = (
    "New user exchanges arrived after your turn chain ended; a fast reply in "
    "another session already acknowledged them. The full exchanges are listed "
    "in the opening context below — address them now through speak and "
    "continue any work they imply."
)


def render_draft_delivery_message(*, draft: str) -> str:
    """The continuation message for an ending whose draft was never delivered."""

    body = "\n".join(
        (_DRAFT_DELIVERY_BODY, str(Prompt(draft).wrap("undelivered-draft")))
    )
    return system_reminder(body)


def render_pending_input_message() -> str:
    """The continuation message for envelopes queued after the chain ended."""

    return system_reminder(_PENDING_INPUT_BODY)


__all__ = [
    "CONTINUATION_MESSAGE_KIND",
    "render_draft_delivery_message",
    "render_pending_input_message",
]
