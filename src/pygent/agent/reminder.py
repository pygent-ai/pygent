"""Destination-independent construction of supplementary context."""

from __future__ import annotations

import re
from enum import Enum
from typing import TypeVar
from xml.etree import ElementTree
from xml.sax.saxutils import escape

from pygent.core import (
    Context,
    EffectSafety,
    ExecutionRequirements,
    Message,
    Module,
    RecoverySafety,
)


class InjectionKind(str, Enum):
    RUNTIME_CONTEXT = "pygent.runtime_context"
    USER_CONTEXT = "pygent.user_context"


_TAGS = {
    InjectionKind.RUNTIME_CONTEXT: "runtime-context",
    InjectionKind.USER_CONTEXT: "user-context",
}
_ContextT = TypeVar("_ContextT", bound=Context)
_INVALID_XML_TEXT = re.compile(
    "[^\x09\x0a\x0d\x20-\ud7ff\ue000-\ufffd\U00010000-\U0010ffff]"
)


def _escape_text(content: str) -> str:
    # XML parsers normalize literal CR; a reference preserves the original text.
    return escape(content).replace("\r", "&#13;")


def format_context(
    content: str, *, kind: InjectionKind = InjectionKind.RUNTIME_CONTEXT
) -> str:
    """Render a fixed context kind, preserving its canonical text-only wrapper."""
    tag = _TAGS[InjectionKind(kind)]
    if not isinstance(content, str):
        raise TypeError("context content must be a string")
    if not content.strip():
        raise ValueError("context content must be non-empty")
    if _INVALID_XML_TEXT.search(content):
        raise ValueError("context content contains invalid XML characters")
    if content.startswith(f"<{tag}>") and content.endswith(f"</{tag}>"):
        try:
            root = ElementTree.fromstring(content)
        except ElementTree.ParseError:
            pass
        else:
            text = root.text or ""
            if (
                root.tag == tag
                and not root.attrib
                and not len(root)
                and content == f"<{tag}>{_escape_text(text)}</{tag}>"
            ):
                if not text.strip():
                    raise ValueError("context content must be non-empty")
                return content
    return f"<{tag}>{_escape_text(content)}</{tag}>"


class Reminder(Module[Message, Message]):
    """Render a neutral Message for explicit placement by the caller."""

    execution_requirements = ExecutionRequirements(
        recovery_safety=RecoverySafety.MODULE_BOUNDARY_RETRY,
        effect_safety=EffectSafety.EFFECT_FREE,
    )

    async def forward(
        self, message: Message, context: _ContextT
    ) -> tuple[Message, _ContextT]:
        if type(message) is not Message:
            raise TypeError("Reminder expects a neutral Message")
        if not isinstance(context, Context):
            raise TypeError("Reminder expects a Context")
        kind = (
            InjectionKind(message.kind)
            if message.kind is not None
            else InjectionKind.RUNTIME_CONTEXT
        )
        return Message(
            content=format_context(message.content, kind=kind),
            kind=kind.value,
            slot=message.slot,
            data=message.data,
            metadata=message.metadata,
        ), context


__all__ = ["InjectionKind", "Reminder", "format_context"]
