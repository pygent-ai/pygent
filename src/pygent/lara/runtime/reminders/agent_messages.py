from __future__ import annotations

from html import escape

from pygent.lara.core.prompt import Prompt
from pygent.lara.sessions import AgentMessage


def render_agent_message(message: AgentMessage) -> str:
    source_session_id = message.source_session_id or "external"
    attributes = [
        f'message-id="{escape(message.message_id, quote=True)}"',
        f'source-session-id="{escape(source_session_id, quote=True)}"',
    ]
    if message.source_agent_alias:
        attributes.append(
            f'source-agent-alias="{escape(message.source_agent_alias, quote=True)}"'
        )
    content = escape(message.content, quote=False)
    content_lines = content.splitlines() or [""]
    body = "\n".join(
        [
            f"  <agent-message {' '.join(attributes)}>",
            *[f"    {line}" for line in content_lines],
            "  </agent-message>",
        ]
    )
    return Prompt(body).wrap("runtime-context")


__all__ = ["render_agent_message"]
