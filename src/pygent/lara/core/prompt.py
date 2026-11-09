"""Streaming XML envelope construction with a registered envelope catalog.

``Prompt`` is a ``str`` subclass so existing f-strings, joins, and message
content keep working unchanged while envelope sites compose tags explicitly.
The catalog records which envelopes have a parser contract and therefore must
never acquire incidental indentation.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Any

_TAG_RE = re.compile(r"^[a-z][a-z-]*$")
_CR_REF = "&#13;"


@dataclass(frozen=True, slots=True)
class EnvelopeInfo:
    """Registry metadata for one XML envelope class."""

    parser_contract: bool
    note: str


KNOWN_ENVELOPES: dict[str, EnvelopeInfo] = {
    "system-reminder": EnvelopeInfo(
        False, "Agent steering and realtime exchange envelope."
    ),
    "runtime-context": EnvelopeInfo(
        False, "Reminder wrapper for runtime context and agent messages."
    ),
    "agent-message": EnvelopeInfo(False, "Collaborator message from another session."),
    "git-context": EnvelopeInfo(False, "Git status snapshot or delta."),
    "skills-context": EnvelopeInfo(False, "Available or newly discovered skills."),
    "cli-context": EnvelopeInfo(False, "Available or newly installed CLI tools."),
    "time": EnvelopeInfo(False, "Current system time block."),
    "user-context": EnvelopeInfo(False, "User turn wrapper with optional attachments."),
    "session-context": EnvelopeInfo(
        True, "Compression continuation consumed as parser-contract text."
    ),
    "file-read": EnvelopeInfo(
        True, "Compression file evidence block; entries must stay unindented."
    ),
    "file": EnvelopeInfo(
        True, "One compression file evidence entry; body is parser-contract text."
    ),
    "memory-access-instruction": EnvelopeInfo(
        False, "Eternal conversation memory access instructions."
    ),
    "memory-extract-instruction": EnvelopeInfo(
        False, "Extractor memory job directive inside the fork's user message."
    ),
    "memory-build-instruction": EnvelopeInfo(
        False, "Builder memory job directive inside the child's user message."
    ),
    "attachments": EnvelopeInfo(False, "Attachment metadata inside user-context."),
    "reply-instruction": EnvelopeInfo(
        False, "Realtime turn-start speak-first directive inside user-context."
    ),
    "task-context": EnvelopeInfo(False, "Scheduled task execution message."),
    "summary": EnvelopeInfo(
        True, "model-generated compression summary; no renderer is owned by Lara."
    ),
}


def _validate_tag(tag: str) -> None:
    if not _TAG_RE.fullmatch(tag):
        raise ValueError(f"invalid XML tag name: {tag!r}")


def _escape_text(text: str) -> str:
    return html.escape(text, quote=False).replace("\r", _CR_REF)


def _attr_text(value: Any) -> str:
    if isinstance(value, bool):
        text = "true" if value else "false"
    else:
        text = str(value)
    return html.escape(text, quote=True).replace("\r", _CR_REF)


def _render_attrs(attrs: dict[str, Any]) -> str:
    rendered = [
        f'{name}="{_attr_text(value)}"'
        for name, value in attrs.items()
        if value is not None
    ]
    return f" {' '.join(rendered)}" if rendered else ""


def _indent_body(body: str, indent: str | None) -> str:
    if indent is None:
        return body
    return "\n".join(f"{indent}{line}" for line in body.splitlines())


class Prompt(str):
    """A prompt fragment that can wrap itself in validated XML envelopes."""

    def wrap(
        self, tag: str, *, indent: str | None = None, **attrs: Any
    ) -> Prompt:
        """Wrap this body in ``tag`` without escaping the body.

        ``indent=None`` keeps the body byte-for-byte intact. A string
        ``indent`` prefixes each body line. Empty bodies render as
        ``<tag></tag>`` so caller-visible empty content is explicit.
        """

        _validate_tag(tag)
        rendered_attrs = _render_attrs(attrs)
        body = str(self)
        if not body:
            return Prompt(f"<{tag}{rendered_attrs}></{tag}>")
        body = _indent_body(body, indent)
        return Prompt(f"<{tag}{rendered_attrs}>\n{body}\n</{tag}>")

    def wrap_escaped(
        self, tag: str, *, indent: str | None = None, **attrs: Any
    ) -> Prompt:
        """Escape this body as text, then wrap it in ``tag``."""

        return self.escape().wrap(tag, indent=indent, **attrs)

    def escape(self) -> Prompt:
        """Return this prompt with text escaping applied."""

        return Prompt(_escape_text(str(self)))

    @staticmethod
    def self_closing(tag: str, **attrs: Any) -> Prompt:
        """Render a self-closing tag with the repo's slash spacing."""

        _validate_tag(tag)
        rendered_attrs = _render_attrs(attrs)
        return Prompt(f"<{tag}{rendered_attrs} />")

    def __add__(self, other: object) -> Prompt:
        if isinstance(other, str):
            return Prompt(str(self) + other)
        return NotImplemented

    def __radd__(self, other: object) -> Prompt:
        if isinstance(other, str):
            return Prompt(other + str(self))
        return NotImplemented


__all__ = ["KNOWN_ENVELOPES", "EnvelopeInfo", "Prompt"]
