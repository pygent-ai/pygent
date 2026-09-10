"""Anthropic Messages transport client and provider wire codec."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import AsyncIterator, Mapping
from itertools import pairwise
from typing import Self, cast

import httpx

from pygent.core import (
    AIMessage,
    FrozenJsonObject,
    JsonValue,
    Message,
    ModelContinuation,
    ToolMessage,
    freeze_json_object,
    thaw_json,
)
from pygent.tool import ToolCall, ToolDefinition, ToolResult

from ._adapter_contracts import (
    ModelProviderRequest,
    ModelProviderResponse,
    ModelProviderStreamPart,
)
from ._json_sse_transport import _HTTPResponseError, _JsonSSETransport
from .catalog import ModelCatalog, ModelInfo
from .configuration import ModelSpec
from .types import (
    ModelErrorKind,
    ModelFailureReason,
    ModelProviderError,
)

_PROTOCOL = "anthropic_messages"
_API_VERSION = "2023-06-01"
_DISPLAY_VALUES = {"summarized", "omitted"}
_EFFORT_VALUES = {"low", "medium", "high", "xhigh", "max"}
_SERVICE_TIERS = {"auto", "standard_only"}
_STOP_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "refusal": "content_filter",
}


class AnthropicMessagesClient:
    """HTTP client for Anthropic's Messages wire protocol."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        headers: Mapping[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
        verify_ssl: bool | None = None,
    ) -> None:
        if not isinstance(base_url, str) or not base_url:
            raise ValueError("base_url must be non-empty")
        request_headers = dict(headers or {})
        if api_key is not None:
            request_headers.setdefault("x-api-key", api_key)
        request_headers.setdefault("anthropic-version", _API_VERSION)
        api_root = base_url.rstrip("/")
        self._messages_endpoint = f"{api_root}/v1/messages"
        self._models_endpoint = f"{api_root}/v1/models"
        self._transport = _JsonSSETransport(
            headers=request_headers,
            client=client,
            verify_ssl=verify_ssl,
            trust_env_url=api_root,
        )
        self._models: ModelCatalog = _AnthropicModelCatalog(self)

    @property
    def models(self) -> ModelCatalog:
        return self._models

    async def invoke(
        self, model: ModelSpec, payload: FrozenJsonObject
    ) -> FrozenJsonObject:
        try:
            return await self._transport.request_json(
                "POST", self._messages_endpoint, payload.to_dict()
            )
        except _HTTPResponseError as exc:
            raise _anthropic_status_error(exc.status, exc.body) from None
        except (TypeError, ValueError) as exc:
            raise ModelProviderError(
                ModelErrorKind.INVALID_RESPONSE,
                "provider returned invalid JSON",
                reason_code=ModelFailureReason.PROVIDER_PAYLOAD_INVALID,
            ) from exc

    async def stream(
        self, model: ModelSpec, payload: FrozenJsonObject
    ) -> AsyncIterator[FrozenJsonObject]:
        body = payload.to_dict()
        body["stream"] = True
        try:
            async for frame in self._transport.stream_sse(
                self._messages_endpoint, body
            ):
                try:
                    item = json.loads(frame.data)
                except json.JSONDecodeError as exc:
                    raise ModelProviderError(
                        ModelErrorKind.INVALID_RESPONSE,
                        "provider returned an invalid SSE event",
                        reason_code=ModelFailureReason.STREAM_EVENT_INVALID,
                    ) from exc
                if not isinstance(item, Mapping):
                    raise ModelProviderError(
                        ModelErrorKind.INVALID_RESPONSE,
                        "provider SSE event must be an object",
                        reason_code=ModelFailureReason.STREAM_EVENT_INVALID,
                    )
                yield freeze_json_object(cast(Mapping[str, object], item))
        except _HTTPResponseError as exc:
            raise _anthropic_status_error(exc.status, exc.body) from None

    async def aclose(self) -> None:
        await self._transport.aclose()

    async def __aenter__(self) -> Self:
        self._transport._ensure_open()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def _list_models_payload(self, *, timeout: float | None) -> FrozenJsonObject:
        self._transport._ensure_open()
        try:
            return await self._transport.request_json(
                "GET", self._models_endpoint, None, timeout=timeout
            )
        except asyncio.CancelledError:
            raise
        except _HTTPResponseError as exc:
            raise _anthropic_status_error(exc.status, exc.body) from None
        except (TypeError, ValueError) as exc:
            raise ModelProviderError(
                ModelErrorKind.INVALID_RESPONSE,
                "model catalog returned invalid JSON",
            ) from exc
        except ModelProviderError:
            raise
        except Exception as exc:  # noqa: BLE001 - provider transport boundary
            raise ModelProviderError(
                _normalize_anthropic_error(exc), "model catalog request failed"
            ) from None


class _AnthropicModelCatalog:
    def __init__(self, client: AnthropicMessagesClient) -> None:
        self._client = client

    async def list(self, *, timeout: float | None = 10.0) -> tuple[ModelInfo, ...]:
        _validate_timeout(timeout)
        payload = (await self._client._list_models_payload(timeout=timeout)).to_dict()
        data = payload.get("data")
        if not isinstance(data, list):
            raise ModelProviderError(
                ModelErrorKind.INVALID_RESPONSE,
                "model catalog data must be an array",
            )
        result: list[ModelInfo] = []
        seen: set[str] = set()
        for item in data:
            if not isinstance(item, Mapping):
                raise ModelProviderError(
                    ModelErrorKind.INVALID_RESPONSE,
                    "model catalog entry must be an object",
                )
            model_id = item.get("id")
            try:
                model = ModelInfo(id=cast(str, model_id), owned_by="anthropic")
            except (TypeError, ValueError) as exc:
                raise ModelProviderError(
                    ModelErrorKind.INVALID_RESPONSE,
                    "model catalog entry is invalid",
                ) from exc
            if model.id in seen:
                raise ModelProviderError(
                    ModelErrorKind.INVALID_RESPONSE,
                    "model catalog contains duplicate model IDs",
                )
            seen.add(model.id)
            result.append(model)
        return tuple(result)


class AnthropicMessagesAdapter:
    """Strict codec for Anthropic Messages-compatible providers."""

    protocol = _PROTOCOL

    def validate_model(self, model: ModelSpec) -> None:
        _validate_provider_options(model.provider_options)

    def build_request(self, request: ModelProviderRequest) -> FrozenJsonObject:
        try:
            self.validate_model(request.model)
            generation = request.generation
            max_tokens = generation.max_output_tokens
            if max_tokens is None:
                raise ValueError("Anthropic Messages requires max_output_tokens")
            options = cast(FrozenJsonObject, request.model.provider_options)
            _validate_option_combinations(options, generation.temperature, max_tokens)
            messages: list[dict[str, object]] = []
            for message in (*request.context.messages, request.message):
                messages.append(_encode_message(message, request.model))
            body: dict[str, object] = {
                "model": request.model.model_id,
                "max_tokens": max_tokens,
                "messages": messages,
            }
            if request.context.system_prompt:
                body["system"] = request.context.system_prompt
            if generation.temperature is not None:
                body["temperature"] = generation.temperature
            if request.tools:
                body["tools"] = [_tool_value(tool) for tool in request.tools]
                body["tool_choice"] = _tool_choice(
                    generation.tool_choice, request.tools
                )
            elif generation.tool_choice not in (None, "none"):
                raise ValueError("tool_choice requires at least one visible tool")
            output_config: dict[str, object] = {}
            raw_output = options.get("output_config")
            if isinstance(raw_output, FrozenJsonObject):
                output_config.update(raw_output.to_dict())
            if generation.response_schema is not None:
                output_config["format"] = {
                    "type": "json_schema",
                    "schema": _thaw(generation.response_schema),
                }
            if output_config:
                body["output_config"] = output_config
            for name in ("thinking", "service_tier", "stop_sequences"):
                if name in options:
                    body[name] = _thaw(options[name])
            return freeze_json_object(body)
        except ModelProviderError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelProviderError(ModelErrorKind.INVALID_REQUEST, str(exc)) from None

    def parse_response(
        self, request: ModelProviderRequest, payload: FrozenJsonObject
    ) -> ModelProviderResponse:
        try:
            body = payload.to_dict()
            if body.get("type") != "message" or body.get("role") != "assistant":
                raise TypeError
            request_id = body.get("id")
            if request_id is not None and not isinstance(request_id, str):
                raise TypeError
            blocks = body.get("content")
            if not isinstance(blocks, list):
                raise TypeError
            content, calls, layout = _decode_blocks(blocks)
            stop_reason = body.get("stop_reason")
            finish_reason = _finish_reason(stop_reason)
            usage = _anthropic_usage(body.get("usage"))
        except ModelProviderError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelProviderError(
                ModelErrorKind.INVALID_RESPONSE,
                "provider response has an invalid completion shape",
                reason_code=ModelFailureReason.COMPLETION_SHAPE_INVALID,
            ) from exc
        continuation = None
        if layout and any(item["type"] != "text" for item in layout):
            continuation = ModelContinuation(
                provider=request.model.provider,
                protocol=self.protocol,
                data={"version": 1, "blocks": layout},
            )
        return ModelProviderResponse(
            message=AIMessage(
                content=content,
                tool_calls=calls,
                metadata={"model_key": request.model_key},
                continuation=continuation,
            ),
            usage=usage,
            provider_request_id=cast(str | None, request_id),
            finish_reason=finish_reason,
        )

    def create_stream_decoder(self, request: ModelProviderRequest) -> _AnthropicStreamDecoder:
        return _AnthropicStreamDecoder(request)

    def normalize_error(self, error: BaseException) -> ModelErrorKind:
        return _normalize_anthropic_error(error)


class _AnthropicStreamDecoder:
    def __init__(self, request: ModelProviderRequest) -> None:
        self._request = request
        self._started = False
        self._completed = False
        self._blocks: dict[int, dict[str, object]] = {}
        self._layout: list[dict[str, object]] = []
        self._text_offset = 0
        self._tool_count = 0
        self._request_id: str | None = None
        self._input_tokens = 0
        self._stop_reason: str | None = None

    def feed(self, payload: FrozenJsonObject) -> tuple[ModelProviderStreamPart, ...]:
        try:
            event = payload.to_dict()
            kind = event.get("type")
            if kind == "ping":
                return ()
            if kind == "error":
                raise _stream_provider_error(event.get("error"))
            if kind == "message_start":
                return self._message_start(event)
            if kind == "content_block_start":
                return self._block_start(event)
            if kind == "content_block_delta":
                return self._block_delta(event)
            if kind == "content_block_stop":
                return self._block_stop(event)
            if kind == "message_delta":
                return self._message_delta(event)
            if kind == "message_stop":
                return self._message_stop()
            if isinstance(kind, str):
                return ()
            raise TypeError
        except ModelProviderError:
            raise
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ModelProviderError(
                ModelErrorKind.INVALID_RESPONSE,
                "provider SSE event has an invalid shape",
                reason_code=ModelFailureReason.STREAM_EVENT_INVALID,
            ) from exc

    def finish(self) -> tuple[ModelProviderStreamPart, ...]:
        if not self._completed:
            raise ModelProviderError(
                ModelErrorKind.INVALID_RESPONSE,
                "model stream ended before a completion marker",
                reason_code=ModelFailureReason.STREAM_INCOMPLETE,
            )
        return ()

    def _message_start(
        self, event: Mapping[str, object]
    ) -> tuple[ModelProviderStreamPart, ...]:
        if self._started or self._completed:
            raise TypeError
        message = event.get("message")
        if not isinstance(message, Mapping):
            raise TypeError
        request_id = message.get("id")
        if not isinstance(request_id, str) or not request_id:
            raise TypeError
        usage = _anthropic_usage(message.get("usage"))
        raw_input = usage.get("input_tokens", 0)
        self._input_tokens = cast(int, raw_input)
        self._request_id = request_id
        self._started = True
        return ()

    def _block_start(
        self, event: Mapping[str, object]
    ) -> tuple[ModelProviderStreamPart, ...]:
        if not self._started or self._completed:
            raise TypeError
        index = _event_index(event)
        if index in self._blocks:
            raise TypeError
        raw = event.get("content_block")
        if not isinstance(raw, Mapping):
            raise TypeError
        kind = raw.get("type")
        state: dict[str, object] = {"type": kind, "open": True}
        parts: list[ModelProviderStreamPart] = []
        if kind == "text":
            text = raw.get("text", "")
            if not isinstance(text, str):
                raise TypeError
            state["text"] = text
            if text:
                parts.append(ModelProviderStreamPart("text", {"text": text}))
        elif kind == "thinking":
            thinking = raw.get("thinking", "")
            if not isinstance(thinking, str):
                raise TypeError
            state.update({"thinking": thinking, "signature": ""})
            if thinking:
                parts.append(
                    ModelProviderStreamPart("reasoning", {"text": thinking})
                )
        elif kind == "redacted_thinking":
            data = raw.get("data")
            if not isinstance(data, str) or not data:
                raise TypeError
            state["data"] = data
        elif kind == "tool_use":
            call_id, name, tool_input = raw.get("id"), raw.get("name"), raw.get("input", {})
            if (
                not isinstance(call_id, str)
                or not call_id
                or not isinstance(name, str)
                or not name
                or not isinstance(tool_input, Mapping)
            ):
                raise TypeError
            initial = "" if not tool_input else json.dumps(tool_input, ensure_ascii=False, separators=(",", ":"))
            state.update({"id": call_id, "name": name, "json": initial})
            parts.append(
                ModelProviderStreamPart(
                    "tool_call",
                    {
                        "index": self._tool_count,
                        "call_id_delta": call_id,
                        "name_delta": name,
                        "arguments_delta": initial,
                    },
                )
            )
            state["tool_index"] = self._tool_count
            self._tool_count += 1
        else:
            raise TypeError
        self._blocks[index] = state
        return tuple(parts)

    def _block_delta(
        self, event: Mapping[str, object]
    ) -> tuple[ModelProviderStreamPart, ...]:
        index = _event_index(event)
        state = self._blocks.get(index)
        if state is None or state.get("open") is not True:
            raise TypeError
        delta = event.get("delta")
        if not isinstance(delta, Mapping):
            raise TypeError
        delta_kind = delta.get("type")
        block_kind = state["type"]
        if block_kind == "text" and delta_kind == "text_delta":
            text = delta.get("text")
            if not isinstance(text, str):
                raise TypeError
            state["text"] = cast(str, state["text"]) + text
            return (ModelProviderStreamPart("text", {"text": text}),) if text else ()
        if block_kind == "thinking" and delta_kind == "thinking_delta":
            thinking = delta.get("thinking")
            if not isinstance(thinking, str):
                raise TypeError
            state["thinking"] = cast(str, state["thinking"]) + thinking
            return (
                ModelProviderStreamPart("reasoning", {"text": thinking}),
            ) if thinking else ()
        if block_kind == "thinking" and delta_kind == "signature_delta":
            signature = delta.get("signature")
            if not isinstance(signature, str):
                raise TypeError
            state["signature"] = cast(str, state["signature"]) + signature
            return ()
        if block_kind == "tool_use" and delta_kind == "input_json_delta":
            partial = delta.get("partial_json")
            if not isinstance(partial, str):
                raise TypeError
            state["json"] = cast(str, state["json"]) + partial
            return (
                ModelProviderStreamPart(
                    "tool_call",
                    {
                        "index": state["tool_index"],
                        "call_id_delta": "",
                        "name_delta": "",
                        "arguments_delta": partial,
                    },
                ),
            ) if partial else ()
        raise TypeError

    def _block_stop(
        self, event: Mapping[str, object]
    ) -> tuple[ModelProviderStreamPart, ...]:
        index = _event_index(event)
        state = self._blocks.get(index)
        if state is None or state.get("open") is not True:
            raise TypeError
        kind = state["type"]
        if kind == "text":
            text = cast(str, state["text"])
            if not text:
                raise TypeError
            self._layout.append(
                {"type": "text", "start": self._text_offset, "end": self._text_offset + len(text)}
            )
            self._text_offset += len(text)
        elif kind == "thinking":
            signature = state["signature"]
            if not isinstance(signature, str) or not signature:
                raise TypeError
            self._layout.append(
                {"type": "thinking", "thinking": state["thinking"], "signature": signature}
            )
        elif kind == "redacted_thinking":
            self._layout.append({"type": "redacted_thinking", "data": state["data"]})
        elif kind == "tool_use":
            raw_json = cast(str, state["json"]) or "{}"
            tool_input = json.loads(raw_json)
            if not isinstance(tool_input, Mapping):
                raise TypeError
            self._layout.append({"type": "tool_use", "index": state["tool_index"]})
        else:  # pragma: no cover - block-start invariant
            raise TypeError
        state["open"] = False
        return ()

    def _message_delta(
        self, event: Mapping[str, object]
    ) -> tuple[ModelProviderStreamPart, ...]:
        if not self._started or self._completed or any(
            state.get("open") is True for state in self._blocks.values()
        ):
            raise TypeError
        delta = event.get("delta")
        if not isinstance(delta, Mapping):
            raise TypeError
        stop_reason = delta.get("stop_reason")
        try:
            _finish_reason(stop_reason)
        except ModelProviderError as exc:
            if exc.reason_code is ModelFailureReason.CONTEXT_LENGTH_EXCEEDED:
                raise
            raise ModelProviderError(
                ModelErrorKind.INVALID_RESPONSE,
                "provider SSE event has an invalid stop reason",
                reason_code=ModelFailureReason.STREAM_EVENT_INVALID,
            ) from None
        self._stop_reason = cast(str, stop_reason)
        usage = event.get("usage")
        if not isinstance(usage, Mapping):
            raise TypeError
        combined = dict(usage)
        combined.setdefault("input_tokens", self._input_tokens)
        return (ModelProviderStreamPart("usage", _anthropic_usage(combined)),)

    def _message_stop(self) -> tuple[ModelProviderStreamPart, ...]:
        if (
            not self._started
            or self._completed
            or self._stop_reason is None
            or any(state.get("open") is True for state in self._blocks.values())
        ):
            raise TypeError
        finish = ModelProviderStreamPart(
            "finish",
            {
                "finish_reason": _finish_reason(self._stop_reason),
                "provider_request_id": self._request_id,
            },
        )
        self._completed = True
        if not any(block["type"] != "text" for block in self._layout):
            return (finish,)
        continuation = ModelProviderStreamPart(
            "continuation",
            {
                "provider": self._request.model.provider,
                "protocol": _PROTOCOL,
                "data": {"version": 1, "blocks": self._layout},
            },
        )
        return continuation, finish


def _validate_provider_options(value: object) -> None:
    if not isinstance(value, FrozenJsonObject):
        raise TypeError("provider_options must be an object")
    unknown = set(value) - {"thinking", "output_config", "service_tier", "stop_sequences"}
    if unknown:
        raise ValueError("unknown Anthropic provider option fields")
    thinking = value.get("thinking")
    if thinking is not None:
        if not isinstance(thinking, FrozenJsonObject):
            raise TypeError("thinking must be an object")
        kind = thinking.get("type")
        allowed = (
            {"type"}
            if kind == "disabled"
            else {"type", "display"}
            if kind == "adaptive"
            else {"type", "budget_tokens", "display"}
            if kind == "enabled"
            else set()
        )
        if not allowed or set(thinking) - allowed:
            raise ValueError("thinking has an invalid shape")
        display = thinking.get("display")
        if display is not None and display not in _DISPLAY_VALUES:
            raise ValueError("thinking display is invalid")
        if kind == "enabled":
            budget = thinking.get("budget_tokens")
            if type(budget) is not int or budget < 1024:
                raise ValueError("thinking budget_tokens must be at least 1024")
    output = value.get("output_config")
    if output is not None:
        if not isinstance(output, FrozenJsonObject) or set(output) != {"effort"}:
            raise ValueError("output_config accepts only effort")
        if output["effort"] not in _EFFORT_VALUES:
            raise ValueError("output_config effort is invalid")
    tier = value.get("service_tier")
    if tier is not None and tier not in _SERVICE_TIERS:
        raise ValueError("service_tier is invalid")
    sequences = value.get("stop_sequences")
    if sequences is not None and (
        not isinstance(sequences, tuple)
        or not sequences
        or any(not isinstance(item, str) or not item for item in sequences)
    ):
        raise ValueError("stop_sequences must contain non-empty strings")


def _validate_option_combinations(
    options: FrozenJsonObject, temperature: float | None, max_tokens: int
) -> None:
    thinking = options.get("thinking")
    if not isinstance(thinking, FrozenJsonObject) or thinking["type"] == "disabled":
        return
    if temperature is not None and temperature != 1:
        raise ValueError("temperature must be 1 when thinking is enabled")
    budget = thinking.get("budget_tokens")
    if (
        thinking["type"] == "enabled"
        and isinstance(budget, int)
        and budget >= max_tokens
    ):
        raise ValueError("thinking budget_tokens must be below max_output_tokens")


def _encode_message(message: Message, model: ModelSpec) -> dict[str, object]:
    if isinstance(message, ToolMessage):
        return {
            "role": "user",
            "content": [_tool_result_value(result) for result in message.results],
        }
    blocks: list[dict[str, object]] = []
    if message.content:
        blocks.append({"type": "text", "text": message.content})
    if isinstance(message, AIMessage):
        continuation_matches = (
            message.continuation is not None
            and message.continuation.provider == model.provider
            and message.continuation.protocol == model.protocol
        )
        blocks = _assistant_blocks(message, model, blocks)
        if not continuation_matches:
            for call in message.tool_calls:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.call_id,
                        "name": call.name,
                        "input": _thaw(call.arguments),
                    }
                )
    return {"role": message.role, "content": blocks or [{"type": "text", "text": ""}]}


def _assistant_blocks(
    message: AIMessage, model: ModelSpec, default: list[dict[str, object]]
) -> list[dict[str, object]]:
    continuation = message.continuation
    if continuation is None or (
        continuation.provider != model.provider or continuation.protocol != model.protocol
    ):
        return default
    try:
        data = cast(FrozenJsonObject, continuation.data)
        if set(data) != {"version", "blocks"} or type(data["version"]) is not int or data["version"] != 1:
            raise ValueError
        layout = data["blocks"]
        if not isinstance(layout, tuple):
            raise TypeError
        rebuilt: list[dict[str, object]] = []
        used_text: list[tuple[int, int]] = []
        used_tools: set[int] = set()
        for raw in layout:
            if not isinstance(raw, FrozenJsonObject):
                raise TypeError
            kind = raw.get("type")
            if kind == "text" and set(raw) == {"type", "start", "end"}:
                raw_start, raw_end = raw["start"], raw["end"]
                if type(raw_start) is not int or type(raw_end) is not int:
                    raise ValueError
                start, end = cast(int, raw_start), cast(int, raw_end)
                if not 0 <= start < end <= len(message.content):
                    raise ValueError
                used_text.append((start, end))
                rebuilt.append({"type": "text", "text": message.content[start:end]})
            elif kind == "tool_use" and set(raw) == {"type", "index"}:
                index = raw["index"]
                if type(index) is not int or index < 0 or index >= len(message.tool_calls) or index in used_tools:
                    raise ValueError
                used_tools.add(index)
                call = message.tool_calls[index]
                rebuilt.append({"type": "tool_use", "id": call.call_id, "name": call.name, "input": _thaw(call.arguments)})
            elif (
                kind == "thinking"
                and set(raw) == {"type", "thinking", "signature"}
                and isinstance(raw["thinking"], str)
                and isinstance(raw["signature"], str)
            ) or (
                kind == "redacted_thinking"
                and set(raw) == {"type", "data"}
                and isinstance(raw["data"], str)
            ):
                rebuilt.append(raw.to_dict())
            else:
                raise ValueError
        if sorted(used_text) != used_text or any(
            left[1] > right[0] for left, right in pairwise(used_text)
        ):
            raise ValueError
        if "".join(message.content[start:end] for start, end in used_text) != message.content or used_tools != set(range(len(message.tool_calls))):
            raise ValueError
        return rebuilt
    except (KeyError, TypeError, ValueError):
        raise ModelProviderError(
            ModelErrorKind.INVALID_REQUEST,
            "Anthropic continuation has an invalid shape",
        ) from None


def _tool_result_value(result: ToolResult) -> dict[str, object]:
    content = result.output if result.status == "succeeded" else result.error
    if not isinstance(content, str):
        content = json.dumps(thaw_json(content), ensure_ascii=False, separators=(",", ":"))
    return {
        "type": "tool_result",
        "tool_use_id": result.call_id,
        "content": content or "",
        "is_error": result.status != "succeeded",
    }


def _event_index(event: Mapping[str, object]) -> int:
    index = event.get("index")
    if type(index) is not int or cast(int, index) < 0:
        raise TypeError
    return cast(int, index)


def _stream_provider_error(value: object) -> ModelProviderError:
    if not isinstance(value, Mapping):
        raise TypeError
    error_type = value.get("type")
    if not isinstance(error_type, str):
        raise TypeError
    if error_type in {"overloaded_error", "api_error"}:
        kind = ModelErrorKind.UNAVAILABLE
        reason = ModelFailureReason.PROVIDER_UNAVAILABLE
    elif error_type == "rate_limit_error":
        kind = ModelErrorKind.RATE_LIMIT
        reason = ModelFailureReason.RATE_LIMITED
    elif error_type in {"authentication_error", "permission_error"}:
        kind = ModelErrorKind.AUTHENTICATION
        reason = (
            ModelFailureReason.AUTHENTICATION_FAILED
            if error_type == "authentication_error"
            else ModelFailureReason.PERMISSION_DENIED
        )
    elif error_type == "invalid_request_error":
        kind = ModelErrorKind.INVALID_REQUEST
        reason = ModelFailureReason.INVALID_PARAMETER
    else:
        kind = ModelErrorKind.UNKNOWN
        reason = ModelFailureReason.UNKNOWN_PROVIDER_FAILURE
    return ModelProviderError(kind, "provider stream failed", reason_code=reason)


def _tool_value(tool: ToolDefinition) -> dict[str, object]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": _thaw(tool.parameters),
    }


def _tool_choice(
    choice: str | None, tools: tuple[ToolDefinition, ...]
) -> dict[str, object]:
    if choice is None or choice == "auto":
        return {"type": "auto"}
    if choice == "required":
        return {"type": "any"}
    if choice == "none":
        return {"type": "none"}
    if choice not in {tool.name for tool in tools}:
        raise ValueError("tool_choice must reference a visible tool")
    return {"type": "tool", "name": choice}


def _decode_blocks(
    blocks: list[object],
) -> tuple[str, tuple[ToolCall, ...], list[dict[str, object]]]:
    text_parts: list[str] = []
    calls: list[ToolCall] = []
    layout: list[dict[str, object]] = []
    offset = 0
    for block in blocks:
        if not isinstance(block, Mapping):
            raise TypeError
        kind = block.get("type")
        if kind == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise TypeError
            text_parts.append(text)
            layout.append({"type": "text", "start": offset, "end": offset + len(text)})
            offset += len(text)
        elif kind == "tool_use":
            call_id, name, arguments = block.get("id"), block.get("name"), block.get("input")
            if not isinstance(call_id, str) or not call_id or not isinstance(name, str) or not name or not isinstance(arguments, Mapping):
                raise TypeError
            layout.append({"type": "tool_use", "index": len(calls)})
            calls.append(ToolCall(call_id=call_id, name=name, arguments=cast(Mapping[str, object], arguments)))
        elif kind == "thinking":
            thinking, signature = block.get("thinking"), block.get("signature")
            if not isinstance(thinking, str) or not isinstance(signature, str) or not signature:
                raise TypeError
            layout.append({"type": "thinking", "thinking": thinking, "signature": signature})
        elif kind == "redacted_thinking":
            data = block.get("data")
            if not isinstance(data, str) or not data:
                raise TypeError
            layout.append({"type": "redacted_thinking", "data": data})
        else:
            raise TypeError
    return "".join(text_parts), tuple(calls), layout


def _finish_reason(value: object) -> str:
    if value == "model_context_window_exceeded":
        raise ModelProviderError(
            ModelErrorKind.INVALID_REQUEST,
            "model context window exceeded",
            reason_code=ModelFailureReason.CONTEXT_LENGTH_EXCEEDED,
        )
    if value == "pause_turn" or value not in _STOP_REASONS:
        raise ModelProviderError(
            ModelErrorKind.INVALID_RESPONSE,
            "provider returned an unsupported stop reason",
            reason_code=ModelFailureReason.COMPLETION_SHAPE_INVALID,
        )
    return _STOP_REASONS[cast(str, value)]


def _anthropic_usage(value: object) -> FrozenJsonObject:
    if value is None:
        return freeze_json_object()
    if not isinstance(value, Mapping):
        raise TypeError
    result: dict[str, int] = {}
    for source, target in (("input_tokens", "input_tokens"), ("output_tokens", "output_tokens")):
        count = value.get(source)
        if count is not None:
            if type(count) is not int or count < 0:
                raise TypeError
            result[target] = count
    cached = 0
    for name in ("cache_creation_input_tokens", "cache_read_input_tokens"):
        if name not in value:
            continue
        count = value[name]
        if type(count) is not int or count < 0:
            raise TypeError
        cached += count
    if cached:
        result["cached_input_tokens"] = cached
    if "input_tokens" in result and "output_tokens" in result:
        result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
    return freeze_json_object(result)


def _anthropic_status_error(status: int, body: bytes) -> ModelProviderError:
    reason = _anthropic_failure_reason(status, body)
    return ModelProviderError(
        _kind_for_status(status),
        f"model provider request failed: {reason.value}",
        reason_code=reason,
        http_status=status,
    )


def _anthropic_failure_reason(status: int, body: bytes) -> ModelFailureReason:
    error_type = ""
    try:
        decoded = json.loads(body)
        if isinstance(decoded, Mapping) and isinstance(decoded.get("error"), Mapping):
            raw_type = cast(Mapping[str, object], decoded["error"]).get("type")
            if isinstance(raw_type, str):
                error_type = raw_type
    except (TypeError, ValueError):
        pass
    if status == 402 or error_type in {"billing_error", "credit_balance_too_low"}:
        return ModelFailureReason.QUOTA_EXHAUSTED
    return {
        401: ModelFailureReason.AUTHENTICATION_FAILED,
        403: ModelFailureReason.PERMISSION_DENIED,
        404: ModelFailureReason.MODEL_NOT_FOUND,
        429: ModelFailureReason.RATE_LIMITED,
        504: ModelFailureReason.PROVIDER_TIMEOUT,
    }.get(status, ModelFailureReason.PROVIDER_UNAVAILABLE if status >= 500 else ModelFailureReason.INVALID_PARAMETER)


def _kind_for_status(status: int) -> ModelErrorKind:
    if status in (401, 403):
        return ModelErrorKind.AUTHENTICATION
    if status in (402, 429):
        return ModelErrorKind.RATE_LIMIT
    if status == 504:
        return ModelErrorKind.TIMEOUT
    if 400 <= status < 500:
        return ModelErrorKind.INVALID_REQUEST
    if status >= 500:
        return ModelErrorKind.UNAVAILABLE
    return ModelErrorKind.UNKNOWN


def _normalize_anthropic_error(error: BaseException) -> ModelErrorKind:
    if isinstance(error, ModelProviderError):
        return error.kind
    if isinstance(error, (TimeoutError, httpx.TimeoutException)):
        return ModelErrorKind.TIMEOUT
    if isinstance(error, httpx.TransportError):
        return ModelErrorKind.UNAVAILABLE
    return ModelErrorKind.UNKNOWN


def _validate_timeout(timeout: float | None) -> None:
    if timeout is not None and (
        not isinstance(timeout, (int, float))
        or isinstance(timeout, bool)
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("timeout must be finite and positive, or None")


def _thaw(value: object) -> object:
    return thaw_json(cast(JsonValue, value))


def anthropic_messages_adapters() -> dict[str, AnthropicMessagesAdapter]:
    return {_PROTOCOL: AnthropicMessagesAdapter()}


__all__ = [
    "AnthropicMessagesAdapter",
    "AnthropicMessagesClient",
    "anthropic_messages_adapters",
]
