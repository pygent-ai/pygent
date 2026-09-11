"""Canonical identifiers for model wire protocols built into Pygent."""

from enum import StrEnum


class BuiltinModelProtocol(StrEnum):
    """Open protocol identifiers with built-in adapter implementations."""

    OPENAI_CHAT_COMPLETIONS = "openai_chat_completions"
    OPENAI_RESPONSES = "openai_responses"
    ANTHROPIC_MESSAGES = "anthropic_messages"
    GEMINI_GENERATE_CONTENT = "gemini_generate_content"


__all__ = ["BuiltinModelProtocol"]
