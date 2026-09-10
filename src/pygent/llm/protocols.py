"""Canonical identifiers for model wire protocols built into Pygent."""

from enum import StrEnum


class BuiltinModelProtocol(StrEnum):
    """Open protocol identifiers with built-in adapter implementations."""

    OPENAI_CHAT_COMPLETIONS = "openai_chat_completions"
    ANTHROPIC_MESSAGES = "anthropic_messages"


__all__ = ["BuiltinModelProtocol"]
