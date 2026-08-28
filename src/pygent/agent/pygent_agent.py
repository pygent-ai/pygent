"""Ready-to-use foreground ReAct Agent with bounded context compression."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import ClassVar, cast

from pygent.core import (
    Agent,
    AIMessage,
    Context,
    JsonValue,
    Message,
    Module,
    ToolMessage,
    UserMessage,
    thaw_json,
)
from pygent.tool import ToolDefinition

from .react import ReActLayer

_CONTEXT_SNAPSHOT_KIND = "pygent.context.snapshot"
_CONTEXT_SNAPSHOT_SLOT = "pygent.context.snapshot"


class ContextCompressionLimitExceeded(RuntimeError):
    """Raised when a PygentAgent exhausts its compression budget."""


class ContextCompressionUnavailable(RuntimeError):
    """Raised when an oversized projection cannot be compressed safely."""


@dataclass(frozen=True, slots=True)
class PygentAgentContext(Context):
    """Portable foreground-Agent state with an uncompressed committed history."""

    context_schema: ClassVar[str] = "pygent.agent-context"
    context_schema_version: ClassVar[int] = 1

    full_history: tuple[Message, ...] = ()
    compression_count: int = 0

    def __post_init__(self) -> None:
        super(PygentAgentContext, self).__post_init__()
        history = tuple(self.full_history)
        if any(not isinstance(message, Message) for message in history):
            raise TypeError("full_history must contain only Message values")
        if (
            isinstance(self.compression_count, bool)
            or not isinstance(self.compression_count, int)
            or self.compression_count < 0
        ):
            raise ValueError("compression_count must be non-negative")
        object.__setattr__(self, "full_history", history)

    def __add__(self, value: object):
        updated = Context.__add__(self, value)
        if updated is NotImplemented:
            return NotImplemented
        assert isinstance(value, Message)
        return replace(updated, full_history=self.full_history + (value,))


class _ContextCompressionLayer(Module[Message, AIMessage]):
    def __init__(
        self,
        *,
        model: Module[Message, AIMessage],
        compression_model: Module[Message, AIMessage],
        compression_prompt: str,
        compression_threshold_bytes: int,
        keep_recent_units: int,
        max_compressions: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.compression_model = compression_model
        self.compression_prompt = compression_prompt
        self.compression_threshold_bytes = compression_threshold_bytes
        self.keep_recent_units = keep_recent_units
        self.max_compressions = max_compressions

    async def forward(
        self, message: Message, context: Context
    ) -> tuple[AIMessage, Context]:
        current, prepared = await self._compress_if_needed(message, context)
        return await self.model(current, prepared)

    async def _compress_if_needed(
        self, current: Message, context: Context
    ) -> tuple[Message, Context]:
        resolve_tools = getattr(self.model, "effective_tools", None)
        effective_tools = (
            tuple(resolve_tools(context))
            if callable(resolve_tools)
            else context.tools
        )
        if (
            _projection_size_bytes(current, context, effective_tools)
            < self.compression_threshold_bytes
        ):
            return current, context
        if not isinstance(context, PygentAgentContext):
            raise TypeError("PygentAgent requires PygentAgentContext")
        if context.compression_count >= self.max_compressions:
            raise ContextCompressionLimitExceeded(
                "foreground context compression budget exhausted"
            )

        projected = context.messages + (current,)
        units = _conversation_units(projected)
        if len(units) <= self.keep_recent_units:
            raise ContextCompressionUnavailable(
                "oversized context has no complete compressible prefix"
            )
        split = len(units) - self.keep_recent_units
        compressible = tuple(message for unit in units[:split] for message in unit)
        preserved = tuple(message for unit in units[split:] for message in unit)
        if not compressible or not preserved or preserved[-1] != current:
            raise ContextCompressionUnavailable(
                "oversized context cannot preserve the current message"
            )

        request = UserMessage(
            content=self.compression_prompt,
            kind="pygent.context.compression_request",
        )
        compression_context = replace(
            context,
            system_prompt="",
            messages=compressible,
            tools=(),
        )
        summary, _ = await self.compression_model(request, compression_context)
        if not summary.content.strip() or summary.tool_calls:
            raise ContextCompressionUnavailable(
                "compression model must return non-empty text without tool calls"
            )

        snapshot = UserMessage(
            content=summary.content,
            kind=_CONTEXT_SNAPSHOT_KIND,
            slot=_CONTEXT_SNAPSHOT_SLOT,
            metadata={
                "compression_number": context.compression_count + 1,
                "source_projection_revision": context.projection_revision,
            },
        )
        return preserved[-1], replace(
            context,
            messages=(snapshot,) + preserved[:-1],
            compression_count=context.compression_count + 1,
            projection_revision=context.projection_revision + 1,
        )


class PygentAgent(Agent[UserMessage, AIMessage]):
    """Standard foreground ReAct Agent with configurable context compression."""

    context_type = PygentAgentContext

    def __init__(
        self,
        *,
        system_prompt: str,
        compression_prompt: str,
        model: Module[Message, AIMessage],
        compression_model: Module[Message, AIMessage],
        tools: Module[AIMessage, ToolMessage],
        compression_threshold_bytes: int,
        keep_recent_units: int = 8,
        max_compressions: int = 4,
        max_steps: int = 16,
        max_model_calls: int = 16,
        max_tool_calls: int = 64,
    ) -> None:
        super().__init__()
        for name, value in (
            ("system_prompt", system_prompt),
            ("compression_prompt", compression_prompt),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        for integer_name, integer_value in (
            ("compression_threshold_bytes", compression_threshold_bytes),
            ("keep_recent_units", keep_recent_units),
            ("max_compressions", max_compressions),
        ):
            if (
                isinstance(integer_value, bool)
                or not isinstance(integer_value, int)
                or integer_value <= 0
            ):
                raise ValueError(f"{integer_name} must be a positive integer")
        self.system_prompt = system_prompt
        self.compression_prompt = compression_prompt
        compressed_model = _ContextCompressionLayer(
            model=model,
            compression_model=compression_model,
            compression_prompt=compression_prompt,
            compression_threshold_bytes=compression_threshold_bytes,
            keep_recent_units=keep_recent_units,
            max_compressions=max_compressions,
        )
        self.react = ReActLayer(
            model=compressed_model,
            tools=tools,
            max_steps=max_steps,
            max_model_calls=max_model_calls,
            max_tool_calls=max_tool_calls,
        )

    def new_context(
        self,
        *,
        tools: tuple[ToolDefinition, ...] = (),
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> PygentAgentContext:
        return PygentAgentContext(
            system_prompt=self.system_prompt,
            tools=tools,
            metadata={} if metadata is None else metadata,
        )

    async def forward(
        self, message: UserMessage, context: PygentAgentContext
    ) -> tuple[AIMessage, PygentAgentContext]:
        answer, next_context = await self.react(message, context)
        return answer, cast(PygentAgentContext, next_context)


def _conversation_units(messages: tuple[Message, ...]) -> tuple[tuple[Message, ...], ...]:
    units: list[tuple[Message, ...]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if (
            isinstance(message, AIMessage)
            and message.tool_calls
            and index + 1 < len(messages)
            and isinstance(messages[index + 1], ToolMessage)
        ):
            units.append((message, messages[index + 1]))
            index += 2
            continue
        units.append((message,))
        index += 1
    return tuple(units)


def _projection_size_bytes(
    current: Message,
    context: Context,
    tools: tuple[ToolDefinition, ...],
) -> int:
    value = {
        "system_prompt": context.system_prompt,
        "messages": [_message_projection(message) for message in context.messages],
        "current_message": _message_projection(current),
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": thaw_json(cast(JsonValue, tool.parameters)),
                "output_schema": (
                    None
                    if tool.output_schema is None
                    else thaw_json(cast(JsonValue, tool.output_schema))
                ),
            }
            for tool in tools
        ],
    }
    return len(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    )


def _message_projection(message: Message) -> dict[str, object]:
    value: dict[str, object] = {
        "role": message.role,
        "content": message.content,
        "kind": message.kind,
        "slot": message.slot,
    }
    if isinstance(message, AIMessage):
        value["tool_calls"] = [
            {
                "call_id": call.call_id,
                "name": call.name,
                "arguments": thaw_json(cast(JsonValue, call.arguments)),
            }
            for call in message.tool_calls
        ]
    elif isinstance(message, ToolMessage):
        value["results"] = [
            {
                "call_id": result.call_id,
                "name": result.name,
                "status": result.status,
                "output": thaw_json(result.output),
                "error": result.error,
            }
            for result in message.results
        ]
    return value


__all__ = [
    "ContextCompressionLimitExceeded",
    "ContextCompressionUnavailable",
    "PygentAgent",
    "PygentAgentContext",
]
