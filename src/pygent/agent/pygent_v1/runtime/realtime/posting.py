"""Queue the fork-reply exchange envelope for the main agent (D-RT1).

Delivery reuses the existing agent-message channel: the exchange is posted
as a queued ``AgentMessage`` on the session's collaboration store, and the
runtime's ``RuntimeReminderModule`` already claims and renders queued agent
messages at the next tool-result boundary. No new pipeline module and no
context-schema change are needed for the envelope itself.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from pygent.agent.pygent_v1.sessions import AgentMessage, SessionCollaborationStore

from .envelope import render_exchange_envelope

__all__ = ["queue_exchange_envelope"]


def queue_exchange_envelope(
    store: SessionCollaborationStore,
    *,
    session_id: str,
    submission_id: str,
    user_asked: str | Sequence[str],
    response: str,
    source_agent_alias: str | None = None,
) -> AgentMessage:
    """Post one fork-reply exchange as a queued agent message.

    ``submission_id`` keys idempotency: retrying the same interrupt delivery
    re-posts the same message instead of duplicating it; a different content
    under the same id is rejected by the store.
    """

    content = render_exchange_envelope(user_asked=user_asked, response=response)
    return store.enqueue_message(
        submission_id=submission_id,
        fingerprint=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        source_session_id=None,
        source_agent_alias=source_agent_alias,
        target_session_id=session_id,
        content=content,
    )
