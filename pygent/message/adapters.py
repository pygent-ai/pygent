"""Provider-specific message format adapters.

The core message classes stay provider-neutral. These adapters translate them
to the wire shapes expected by individual model providers while the legacy
``BaseMessage.to_*_format`` methods delegate here for compatibility.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List


def _message_iterable_from_context(context: Any) -> Iterable[Any]:
    history = getattr(context, "history", None)
    if history is None:
        return []
    return history.data if hasattr(history, "data") else history


class OpenAIMessageAdapter:
    """Translate Pygent messages to OpenAI-compatible chat messages."""

    MESSAGE_KEYS = {"role", "content", "name", "tool_calls", "tool_call_id"}
    TOOL_CALL_KEYS = {"id", "type", "function"}

    @staticmethod
    def to_message_dict(message: Any) -> Dict[str, Any]:
        message_dict = message.to_dict()
        result = {
            key: value
            for key, value in message_dict.items()
            if key in OpenAIMessageAdapter.MESSAGE_KEYS
        }
        role = getattr(getattr(message, "role", None), "data", None)
        if role in {"function", "tool"}:
            result["role"] = role
        if result.get("role") == "assistant" and "tool_calls" in result:
            result["tool_calls"] = [
                {
                    key: value
                    for key, value in tool_call.items()
                    if key in OpenAIMessageAdapter.TOOL_CALL_KEYS
                }
                for tool_call in result["tool_calls"]
                if isinstance(tool_call, dict)
            ]
            if result["tool_calls"] and "reasoning_content" in message_dict:
                result["reasoning_content"] = message_dict["reasoning_content"]
        return result

    @classmethod
    def messages_from_context(cls, context: Any) -> List[Dict[str, Any]]:
        return [cls.to_message_dict(message) for message in _message_iterable_from_context(context)]


class OllamaMessageAdapter:
    """Translate Pygent messages to Ollama chat messages."""

    @staticmethod
    def to_message_dict(message: Any) -> Dict[str, Any]:
        result = OpenAIMessageAdapter.to_message_dict(message)
        result.pop("reasoning_content", None)

        # Ollama expects tool_calls.function.arguments to be an object, while
        # OpenAI-compatible payloads use a JSON string.
        if result.get("role") == "assistant" and "tool_calls" in result:
            tool_calls = []
            for tool_call in result.get("tool_calls", []):
                function_obj = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
                arguments = function_obj.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except (json.JSONDecodeError, TypeError):
                        arguments = {"raw_arguments": arguments}
                tool_calls.append(
                    {
                        "id": tool_call.get("id") if isinstance(tool_call, dict) else None,
                        "type": tool_call.get("type", "function") if isinstance(tool_call, dict) else "function",
                        "function": {
                            "name": function_obj.get("name", ""),
                            "arguments": arguments if isinstance(arguments, dict) else {},
                        },
                    }
                )
            result["tool_calls"] = tool_calls

        return result

    @classmethod
    def messages_from_context(cls, context: Any) -> List[Dict[str, Any]]:
        return [cls.to_message_dict(message) for message in _message_iterable_from_context(context)]


__all__ = [
    "OpenAIMessageAdapter",
    "OllamaMessageAdapter",
]
