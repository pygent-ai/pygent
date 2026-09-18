"""Private continuation identity and request-projection helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace

from pygent.core import AIMessage, Context, Message, ModelContinuation, ToolMessage

from .configuration import ModelSpec


@dataclass(frozen=True, slots=True)
class PendingToolContinuation:
    assistant_index: int
    continuation: ModelContinuation


def pending_tool_continuation(
    message: Message, context: Context
) -> PendingToolContinuation | None:
    if (
        not isinstance(message, ToolMessage)
        or not message.results
        or not context.messages
    ):
        return None
    assistant_index = len(context.messages) - 1
    assistant = context.messages[assistant_index]
    if (
        not isinstance(assistant, AIMessage)
        or not assistant.tool_calls
        or assistant.continuation is None
    ):
        return None
    call_ids = tuple(call.call_id for call in assistant.tool_calls)
    result_ids = tuple(result.call_id for result in message.results)
    if len(call_ids) != len(result_ids) or set(call_ids) != set(result_ids):
        return None
    return PendingToolContinuation(assistant_index, assistant.continuation)


def neutral_tool_context(context: Context, pending: PendingToolContinuation) -> Context:
    messages = list(context.messages)
    assistant = messages[pending.assistant_index]
    if not isinstance(assistant, AIMessage):  # pragma: no cover - helper invariant
        raise TypeError("pending continuation must reference an AIMessage")
    messages[pending.assistant_index] = replace(assistant, continuation=None)
    return replace(context, messages=tuple(messages))


def continuation_matches(
    continuation: ModelContinuation,
    *,
    model_key: str,
    model: ModelSpec,
) -> bool:
    return (
        continuation.model_key == model_key
        and continuation.provider == model.provider
        and continuation.model_id == model.model_id
        and continuation.protocol == model.protocol
    )


__all__ = [
    "PendingToolContinuation",
    "continuation_matches",
    "neutral_tool_context",
    "pending_tool_continuation",
]
