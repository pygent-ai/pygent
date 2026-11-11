"""The ``speak`` tool: the only user-visible output channel (D-RT5/D-RT10).

The agent's content and thinking stay out of the user-facing stream; anything
the user should see must go through ``speak``. The tool itself is channel
agnostic — a :class:`SpeakRenderer` implementation owns the form (chat box,
bubble) and the source (typing, IM, voice). Schema decisions live in
docs/design/realtime-interaction (D-RT10).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pygent.agent.pygent_v1.core.prompt import Prompt
from pygent.agent.pygent_v1.runtime.attachments import resolve_attachments
from pygent.tool import IdempotencyPolicy, ToolDefinition, ToolSideEffect, ToolSpec

SPEAK_TOOL_ID = "pygent.v1.realtime.speak"
SPEAK_RESOURCE_KEY = "pygent.v1-realtime-speak"
SPEAK_TOOL_NAME = "speak"

# Turn-start directive injected into <user-context> for realtime sessions
# (D-RT5): the user sees nothing until the first speak call, so the turn must
# open with a short acknowledgment before any tool work. Plain-text sessions
# never receive it; FIRST_RESULT_SPEAK_REMINDER then only covers the case
# where the agent ignored this directive and went silent into tool calls.
REPLY_INSTRUCTION_BODY = """Open this turn by calling speak with one short reassurance line to the
user before any tool call or long work. The acknowledgment is in addition to
the work, never a replacement: after speaking, carry the task through. Even
if you end up needing to do nothing, still speak before finishing."""


def render_reply_instruction() -> str:
    """Render the ``<reply-instruction>`` envelope for one user turn."""

    return "\n".join(
        f"  {line}"
        for line in str(
            Prompt(REPLY_INSTRUCTION_BODY).wrap("reply-instruction", indent="  ")
        ).splitlines()
    )

# The tool result for a deliberate ``say_nothing`` decision. It must steer the
# model back to a plain-text turn ending: after every tool result pygent runs
# another round, and a bare acknowledgement left the finalizer round looping
# on say_nothing calls until the step cap (observed 70 in a row). An exchange
# envelope claimed at this same tool boundary is the exception — without it,
# a say_nothing ending would swallow the queued user input (D-RT1).
SILENT_ACKNOWLEDGEMENT = (
    "acknowledged (silent). Your decision is recorded - end this turn with a "
    "normal plain-text reply and do not speak it. Exception: when a new user "
    "exchange is appended below this result, answer it with speak."
)

SPEAK_TOOL_SPEC = ToolSpec(
    tool_id=SPEAK_TOOL_ID,
    version="1",
    definition=ToolDefinition(
        name="speak",
        description=(
            "Say something to the user. This is the only channel the user can see "
            "— your content, thinking and tool activity are invisible to them. Use "
            "it for answers, progress updates, and questions alike. To hand files "
            "to the user, list them in attachments. To deliberately decline "
            "speaking (e.g. when the end-of-turn reminder asks you to decide), "
            "pass say_nothing=true."
        ),
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": (
                        "What to say, in the user's language. One message — say it "
                        "in one go."
                    ),
                },
                "attachments": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": (
                            "Workspace-relative path of a file to hand to the user "
                            "alongside the text."
                        ),
                    },
                },
                "say_nothing": {
                    "type": "boolean",
                    "description": (
                        "True to deliberately decline speaking this turn. Set it "
                        "only when you have decided to say nothing; omit text and "
                        "attachments when set."
                    ),
                },
            },
        },
    ),
    side_effect=ToolSideEffect.EXTERNAL,
    idempotency=IdempotencyPolicy.INHERENT,
    resource_key=SPEAK_RESOURCE_KEY,
)


@dataclass(frozen=True)
class SpeakArgs:
    """Validated speak arguments (D-RT10 contract)."""

    text: str | None
    attachments: tuple[str, ...]
    say_nothing: bool


class SpeakArgsError(ValueError):
    """Raised when speak arguments violate the D-RT10 contract."""


def parse_speak_args(args: Mapping[str, object]) -> SpeakArgs:
    """Validate raw tool arguments into :class:`SpeakArgs`.

    Contract: ``say_nothing=True`` excludes text and attachments; otherwise at
    least one of text (non-blank) or attachments (non-empty) must be present.
    """

    raw_text = args.get("text")
    raw_attachments = args.get("attachments")
    say_nothing = bool(args.get("say_nothing"))

    if say_nothing:
        if raw_text or raw_attachments:
            raise SpeakArgsError(
                "say_nothing=true excludes text and attachments"
            )
        return SpeakArgs(text=None, attachments=(), say_nothing=True)

    text: str | None = None
    if raw_text is not None:
        if not isinstance(raw_text, str):
            raise SpeakArgsError("text must be a string")
        text = raw_text.strip() or None

    attachments: list[str] = []
    if raw_attachments is not None:
        if not isinstance(raw_attachments, Sequence) or isinstance(
            raw_attachments, (str, bytes)
        ):
            raise SpeakArgsError("attachments must be an array of paths")
        for item in raw_attachments:
            if not isinstance(item, str) or not item.strip():
                raise SpeakArgsError("attachment paths must be non-empty strings")
            attachments.append(item)

    if text is None and not attachments:
        raise SpeakArgsError(
            "speak requires text, attachments, or say_nothing=true"
        )
    return SpeakArgs(text=text, attachments=tuple(attachments), say_nothing=False)


class SpeakRenderer(Protocol):
    """A user-facing delivery form (chat box, bubble, IM, voice, …)."""

    def render(self, text: str, attachments: Sequence[Path]) -> None:
        """Deliver the message; must not raise for content reasons."""
        ...


def resolve_speak_attachments(
    workspace_root: Path, raw: Sequence[str]
) -> tuple[Path, ...]:
    """Resolve attachment paths through the canonical workspace resolver."""

    resolved = resolve_attachments(raw, workspace_root)
    return tuple(item.path for item in resolved)


def execute_speak(
    args: SpeakArgs,
    *,
    renderer: SpeakRenderer,
    workspace_root: Path | None = None,
) -> str:
    """Deliver a validated speak call and return the tool_result text."""

    if args.say_nothing:
        return SILENT_ACKNOWLEDGEMENT
    attachments: tuple[Path, ...] = ()
    if args.attachments:
        if workspace_root is None:
            raise SpeakArgsError(
                "workspace_root is required to resolve attachments"
            )
        attachments = resolve_speak_attachments(workspace_root, args.attachments)
    renderer.render(args.text or "", attachments)
    return "delivered"


__all__ = [
    "SPEAK_TOOL_SPEC",
    "SpeakArgs",
    "SpeakArgsError",
    "SpeakRenderer",
    "execute_speak",
    "parse_speak_args",
    "render_reply_instruction",
    "resolve_speak_attachments",
]
