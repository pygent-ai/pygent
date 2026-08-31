"""Ready-to-use foreground ReAct Agent with token-window compression."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import ClassVar, TypeVar, cast

from pygent.core import (
    Agent,
    AIMessage,
    Context,
    EffectSafety,
    ExecutionRequirements,
    JsonValue,
    Message,
    Module,
    RecoverySafety,
    ToolMessage,
    UserMessage,
    thaw_json,
)
from pygent.tool import ToolDefinition

from .react import ReActLayer

_CONTEXT_SNAPSHOT_KIND = "pygent.context.snapshot"
_CONTEXT_SNAPSHOT_SLOT = "pygent.context.snapshot"
_TOKEN_SCALE_BASE = 1_000_000
_INITIAL_TOKEN_SCALE_PPM = 1_100_000
_PygentAgentContextT = TypeVar(
    "_PygentAgentContextT", bound="PygentAgentContext"
)
_COORDINATOR_EXECUTION_REQUIREMENTS = ExecutionRequirements(
    requires_finite_deadline=True,
    recovery_safety=RecoverySafety.MODULE_BOUNDARY_RETRY,
    effect_safety=EffectSafety.MANAGED_EFFECTS,
)


class ContextCompressionLimitExceeded(RuntimeError):
    """Raised when a PygentAgent exhausts its compression budget."""


class ContextCompressionUnavailable(RuntimeError):
    """Raised when an oversized projection cannot be compressed safely."""


@dataclass(frozen=True, slots=True)
class PygentAgentContext(Context):
    """Portable foreground-Agent state with bounded invocation commits."""

    context_schema: ClassVar[str] = "pygent.agent-context"
    context_schema_version: ClassVar[int] = 3

    committed_messages: tuple[Message, ...] = ()
    compression_count: int = 0
    input_token_scale_ppm: int = _INITIAL_TOKEN_SCALE_PPM
    last_input_tokens: int | None = None

    def __post_init__(self) -> None:
        super(PygentAgentContext, self).__post_init__()
        committed_messages = tuple(self.committed_messages)
        if any(not isinstance(message, Message) for message in committed_messages):
            raise TypeError("committed_messages must contain only Message values")
        if (
            isinstance(self.compression_count, bool)
            or not isinstance(self.compression_count, int)
            or self.compression_count < 0
        ):
            raise ValueError("compression_count must be non-negative")
        if (
            isinstance(self.input_token_scale_ppm, bool)
            or not isinstance(self.input_token_scale_ppm, int)
            or self.input_token_scale_ppm <= 0
        ):
            raise ValueError("input_token_scale_ppm must be positive")
        if self.last_input_tokens is not None and (
            isinstance(self.last_input_tokens, bool)
            or not isinstance(self.last_input_tokens, int)
            or self.last_input_tokens < 0
        ):
            raise ValueError("last_input_tokens must be non-negative when provided")
        object.__setattr__(self, "committed_messages", committed_messages)

    def __add__(self, value: object):
        updated = Context.__add__(self, value)
        if updated is NotImplemented:
            return NotImplemented
        assert isinstance(value, Message)
        return replace(
            updated,
            committed_messages=self.committed_messages + (value,),
        )


class _ContextCompressionLayer(Module[Message, AIMessage]):
    execution_requirements = _COORDINATOR_EXECUTION_REQUIREMENTS

    def __init__(
        self,
        *,
        model: Module[Message, AIMessage],
        compressor: Module[Message, AIMessage],
        compression_prompt: str,
        context_window_tokens: int,
        compression_trigger_ratio: float,
        compression_context_window_tokens: int,
        max_compressions: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.compressor = compressor
        self.compression_prompt = compression_prompt
        self.context_window_tokens = context_window_tokens
        self.compression_trigger_ratio = compression_trigger_ratio
        self.compression_context_window_tokens = compression_context_window_tokens
        self.max_compressions = max_compressions

    async def forward(
        self, message: Message, context: Context
    ) -> tuple[AIMessage, Context]:
        current, prepared, raw_units = await self._compress_if_needed(message, context)
        answer, returned = await self.model(current, prepared)
        if not isinstance(returned, PygentAgentContext):
            raise TypeError("PygentAgent model must preserve PygentAgentContext")
        actual = answer.usage.get("input_tokens")
        if isinstance(actual, int) and not isinstance(actual, bool):
            observed_ppm = _ceil_div(actual * _TOKEN_SCALE_BASE, raw_units)
            calibrated_ppm = _ceil_div(observed_ppm * 11, 10)
            returned = replace(
                returned,
                input_token_scale_ppm=max(
                    returned.input_token_scale_ppm,
                    calibrated_ppm,
                ),
                last_input_tokens=actual,
            )
        return answer, returned

    async def _compress_if_needed(
        self, current: Message, context: Context
    ) -> tuple[Message, PygentAgentContext, int]:
        if not isinstance(context, PygentAgentContext):
            raise TypeError("PygentAgent requires PygentAgentContext")
        resolve_tools = getattr(self.model, "effective_tools", None)
        effective_tools = (
            tuple(resolve_tools(context))
            if callable(resolve_tools)
            else context.tools
        )
        foreground_units = _request_token_units(current, context, effective_tools)
        foreground_estimate = _scaled_token_estimate(
            foreground_units,
            context.input_token_scale_ppm,
        )
        foreground_trigger = int(
            self.context_window_tokens * self.compression_trigger_ratio
        )
        compression_request = UserMessage(
            content=self.compression_prompt,
            kind="pygent.context.compression_request",
        )
        compression_units = _request_token_units(compression_request, context, ())
        compression_estimate = _scaled_token_estimate(
            compression_units,
            _INITIAL_TOKEN_SCALE_PPM,
        )
        compression_trigger = int(
            self.compression_context_window_tokens
            * self.compression_trigger_ratio
        )
        if (
            foreground_estimate < foreground_trigger
            and compression_estimate < compression_trigger
        ):
            return current, context, foreground_units
        if context.compression_count >= self.max_compressions:
            raise ContextCompressionLimitExceeded(
                "foreground context compression budget exhausted"
            )
        if not context.messages:
            raise ContextCompressionUnavailable(
                "oversized request has no projected history to compress"
            )
        if compression_estimate >= self.compression_context_window_tokens:
            raise ContextCompressionUnavailable(
                "compression request exceeds the compression context window"
            )

        compression_context = replace(context, tools=())
        summary, returned_context = await self.compressor(
            compression_request,
            compression_context,
        )
        if returned_context != compression_context:
            raise ContextCompressionUnavailable(
                "compressor must preserve its fork context"
            )
        if not summary.content.strip() or summary.tool_calls:
            raise ContextCompressionUnavailable(
                "compressor must return non-empty text without tool calls"
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
        prepared = replace(
            context,
            messages=(snapshot,),
            compression_count=context.compression_count + 1,
            projection_revision=context.projection_revision + 1,
        )
        compressed_units = _request_token_units(current, prepared, effective_tools)
        compressed_estimate = _scaled_token_estimate(
            compressed_units,
            prepared.input_token_scale_ppm,
        )
        if compressed_estimate >= foreground_trigger:
            raise ContextCompressionUnavailable(
                "compressed foreground request remains oversized"
            )
        return current, prepared, compressed_units


class PygentAgent(Agent[UserMessage, AIMessage]):
    """Standard foreground ReAct Agent with configurable context compression."""

    context_type = PygentAgentContext
    execution_requirements = _COORDINATOR_EXECUTION_REQUIREMENTS

    def __init__(
        self,
        *,
        system_prompt: str,
        compression_prompt: str,
        model: Module[Message, AIMessage],
        compressor: Module[Message, AIMessage],
        tools: Module[AIMessage, ToolMessage],
        context_window_tokens: int,
        compression_trigger_ratio: float = 0.9,
        compression_context_window_tokens: int | None = None,
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
        resolved_compression_window = (
            context_window_tokens
            if compression_context_window_tokens is None
            else compression_context_window_tokens
        )
        for integer_name, integer_value in (
            ("context_window_tokens", context_window_tokens),
            ("compression_context_window_tokens", resolved_compression_window),
            ("max_compressions", max_compressions),
        ):
            if (
                isinstance(integer_value, bool)
                or not isinstance(integer_value, int)
                or integer_value <= 0
            ):
                raise ValueError(f"{integer_name} must be a positive integer")
        if (
            isinstance(compression_trigger_ratio, bool)
            or not isinstance(compression_trigger_ratio, (int, float))
            or not 0 < compression_trigger_ratio < 1
        ):
            raise ValueError("compression_trigger_ratio must be between zero and one")
        self.system_prompt = system_prompt
        self.compression_prompt = compression_prompt
        compressed_model = _ContextCompressionLayer(
            model=model,
            compressor=compressor,
            compression_prompt=compression_prompt,
            context_window_tokens=context_window_tokens,
            compression_trigger_ratio=float(compression_trigger_ratio),
            compression_context_window_tokens=resolved_compression_window,
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
        self, message: UserMessage, context: _PygentAgentContextT
    ) -> tuple[AIMessage, _PygentAgentContextT]:
        invocation_context = replace(
            context,
            committed_messages=(),
            compression_count=0,
        )
        answer, next_context = await self.react(message, invocation_context)
        return answer, cast(_PygentAgentContextT, next_context)


def _request_token_units(
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
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    ascii_bytes = sum(ord(character) < 128 for character in canonical)
    non_ascii_codepoints = len(canonical) - ascii_bytes
    lexical_units = _ceil_div(ascii_bytes, 3) + _ceil_div(
        non_ascii_codepoints * 3,
        2,
    )
    structural_units = 8 * (len(context.messages) + 1) + 16 * len(tools) + 4
    return max(1, lexical_units + structural_units)


def _scaled_token_estimate(raw_units: int, scale_ppm: int) -> int:
    return _ceil_div(raw_units * scale_ppm, _TOKEN_SCALE_BASE)


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


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
