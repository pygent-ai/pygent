"""OpenAI Responses HTTP transport and wire codec."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from typing import Self, cast

import httpx
import jsonschema  # type: ignore[import-untyped]

from pygent.core import (
    AIMessage,
    FrozenJsonObject,
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
    _canonical_usage,
)
from ._json_sse_transport import _HTTPResponseError, _JsonSSETransport
from .configuration import ModelSpec
from .types import (
    ModelErrorKind,
    ModelFailureReason,
    ModelProviderError,
)

_PROTOCOL = "openai_responses"
_RESERVED_OPTIONS = frozenset(
    {
        "model",
        "input",
        "instructions",
        "max_output_tokens",
        "temperature",
        "tools",
        "tool_choice",
        "text",
        "stream",
    }
)
_ALLOWED_OPTIONS = frozenset({"reasoning", "store", "include", "parallel_tool_calls"})
_REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh"})
_REASONING_SUMMARIES = frozenset({"auto", "concise", "detailed"})


class OpenAIResponsesClient:
    """Small HTTP/SSE client for the OpenAI Responses protocol."""

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
            request_headers.setdefault("Authorization", f"Bearer {api_key}")
        api_root = base_url.rstrip("/")
        self._endpoint = f"{api_root}/responses"
        self._transport = _JsonSSETransport(
            headers=request_headers,
            client=client,
            verify_ssl=verify_ssl,
            trust_env_url=api_root,
        )

    async def invoke(
        self, model: ModelSpec, payload: FrozenJsonObject
    ) -> FrozenJsonObject:
        try:
            return await self._transport.request_json(
                "POST", self._endpoint, payload.to_dict()
            )
        except _HTTPResponseError as exc:
            raise _status_error(exc.status) from None
        except asyncio.CancelledError:
            raise
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
            async for frame in self._transport.stream_sse(self._endpoint, body):
                try:
                    value = json.loads(frame.data)
                except json.JSONDecodeError as exc:
                    raise ModelProviderError(
                        ModelErrorKind.INVALID_RESPONSE,
                        "provider returned an invalid Responses SSE event",
                        reason_code=ModelFailureReason.STREAM_EVENT_INVALID,
                    ) from exc
                if not isinstance(value, Mapping):
                    raise ModelProviderError(
                        ModelErrorKind.INVALID_RESPONSE,
                        "provider Responses SSE event must be an object",
                        reason_code=ModelFailureReason.STREAM_EVENT_INVALID,
                    )
                yield freeze_json_object(cast(Mapping[str, object], value))
        except _HTTPResponseError as exc:
            raise _status_error(exc.status) from None

    async def aclose(self) -> None:
        await self._transport.aclose()

    async def __aenter__(self) -> Self:
        self._transport._ensure_open()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()


class OpenAIResponsesAdapter:
    """Strict codec for the OpenAI Responses wire protocol."""

    protocol = _PROTOCOL

    def validate_model(self, model: ModelSpec) -> None:
        options = cast(FrozenJsonObject, model.provider_options)
        conflicts = set(options) & _RESERVED_OPTIONS
        if conflicts:
            raise ValueError(
                "provider options cannot override reserved Responses fields: "
                + ", ".join(sorted(conflicts))
            )
        unknown = set(options) - _ALLOWED_OPTIONS
        if unknown:
            raise ValueError(
                "unknown Responses provider options: " + ", ".join(sorted(unknown))
            )
        reasoning = options.get("reasoning")
        if reasoning is not None:
            if not isinstance(reasoning, FrozenJsonObject):
                raise TypeError("Responses reasoning option must be an object")
            if set(reasoning) - {"effort", "summary"}:
                raise ValueError("Responses reasoning option has unknown fields")
            if reasoning.get("effort") not in _REASONING_EFFORTS:
                raise ValueError("Responses reasoning effort is invalid")
            summary = reasoning.get("summary")
            if summary is not None and summary not in _REASONING_SUMMARIES:
                raise ValueError("Responses reasoning summary is invalid")
        for name in ("store", "parallel_tool_calls"):
            if name in options and not isinstance(options[name], bool):
                raise TypeError(f"Responses {name} option must be a bool")
        include = options.get("include")
        if include is not None and (
            not isinstance(include, tuple)
            or any(not isinstance(item, str) or not item for item in include)
        ):
            raise ValueError("Responses include option must contain strings")

    def build_request(self, request: ModelProviderRequest) -> FrozenJsonObject:
        try:
            self.validate_model(request.model)
            body: dict[str, object] = {
                "model": request.model.model_id,
                "input": [
                    item
                    for message in (*request.context.messages, request.message)
                    for item in _input_items(message, request.model)
                ],
            }
            if request.context.system_prompt:
                body["instructions"] = request.context.system_prompt
            generation = request.generation
            if generation.max_output_tokens is not None:
                body["max_output_tokens"] = generation.max_output_tokens
            if generation.temperature is not None:
                body["temperature"] = generation.temperature
            if request.tools:
                body["tools"] = [_tool_value(tool) for tool in request.tools]
                body["tool_choice"] = _tool_choice(
                    generation.tool_choice, request.tools
                )
            elif generation.tool_choice not in (None, "none"):
                raise ValueError("tool_choice requires at least one visible tool")
            if generation.response_schema is not None:
                body["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": generation.response_schema_name,
                        "schema": thaw_json(
                            cast(FrozenJsonObject, generation.response_schema)
                        ),
                        "strict": True,
                    }
                }
            body.update(cast(FrozenJsonObject, request.model.provider_options).to_dict())
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
            status = body.get("status")
            if status not in {"completed", "incomplete"}:
                raise TypeError
            output = body.get("output")
            if not isinstance(output, list):
                raise TypeError
            content, calls, continuation_items = _decode_output(output, request)
            content = _validate_structured_output(request, content)
            response_id = body.get("id")
            if response_id is not None and not isinstance(response_id, str):
                raise TypeError
            usage = _canonical_usage(body.get("usage"))
        except ModelProviderError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelProviderError(
                ModelErrorKind.INVALID_RESPONSE,
                "provider response has an invalid Responses shape",
                reason_code=ModelFailureReason.COMPLETION_SHAPE_INVALID,
            ) from exc
        continuation = (
            ModelContinuation(
                provider=request.model.provider,
                protocol=self.protocol,
                data={"version": 1, "items": continuation_items},
            )
            if continuation_items
            else None
        )
        finish_reason = (
            "tool_calls"
            if calls
            else "length"
            if status == "incomplete"
            else "stop"
        )
        return ModelProviderResponse(
            message=AIMessage(
                content=content,
                tool_calls=calls,
                metadata={"model_key": request.model_key},
                continuation=continuation,
            ),
            usage=usage,
            provider_request_id=cast(str | None, response_id),
            finish_reason=finish_reason,
        )

    def create_stream_decoder(
        self, request: ModelProviderRequest
    ) -> _OpenAIResponsesStreamDecoder:
        return _OpenAIResponsesStreamDecoder(request)

    def normalize_error(self, error: BaseException) -> ModelErrorKind:
        if isinstance(error, ModelProviderError):
            return error.kind
        if isinstance(error, (TimeoutError, httpx.TimeoutException)):
            return ModelErrorKind.TIMEOUT
        if isinstance(error, httpx.TransportError):
            return ModelErrorKind.UNAVAILABLE
        return ModelErrorKind.UNKNOWN


class _OpenAIResponsesStreamDecoder:
    def __init__(self, request: ModelProviderRequest) -> None:
        self._request = request
        self._completed = False

    def feed(self, payload: FrozenJsonObject) -> tuple[ModelProviderStreamPart, ...]:
        try:
            event = payload.to_dict()
            kind = event.get("type")
            if kind in {
                "response.reasoning_text.delta",
                "response.reasoning_summary_text.delta",
            }:
                return (ModelProviderStreamPart("reasoning", {"text": _delta(event)}),)
            if kind == "response.output_text.delta":
                return (ModelProviderStreamPart("text", {"text": _delta(event)}),)
            if kind == "response.output_item.added":
                item = event.get("item")
                if isinstance(item, Mapping) and item.get("type") == "function_call":
                    return (_stream_call(item, event),)
                return ()
            if kind == "response.function_call_arguments.delta":
                index = _index(event)
                return (
                    ModelProviderStreamPart(
                        "tool_call",
                        {
                            "index": index,
                            "call_id_delta": "",
                            "name_delta": "",
                            "arguments_delta": _delta(event),
                        },
                    ),
                )
            if kind == "response.completed":
                response = event.get("response")
                if not isinstance(response, Mapping) or response.get("status") != "completed":
                    raise TypeError
                parts: list[ModelProviderStreamPart] = []
                usage = _canonical_usage(response.get("usage"))
                if usage:
                    parts.append(ModelProviderStreamPart("usage", usage))
                output = response.get("output", [])
                if not isinstance(output, list):
                    raise TypeError
                _, _, continuation_items = _decode_output(output, self._request)
                if continuation_items:
                    parts.append(
                        ModelProviderStreamPart(
                            "continuation",
                            {
                                "provider": self._request.model.provider,
                                "protocol": _PROTOCOL,
                                "data": {"version": 1, "items": continuation_items},
                            },
                        )
                    )
                response_id = response.get("id")
                finish_data: dict[str, object] = {"finish_reason": "stop"}
                if isinstance(response_id, str):
                    finish_data["provider_request_id"] = response_id
                parts.append(ModelProviderStreamPart("finish", finish_data))
                self._completed = True
                return tuple(parts)
            if kind in {"response.failed", "error"}:
                raise ModelProviderError(
                    ModelErrorKind.UNAVAILABLE,
                    "Responses stream reported failure",
                    reason_code=ModelFailureReason.PROVIDER_UNAVAILABLE,
                )
            if isinstance(kind, str):
                return ()
            raise TypeError
        except ModelProviderError:
            raise
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ModelProviderError(
                ModelErrorKind.INVALID_RESPONSE,
                "provider Responses SSE event has an invalid shape",
                reason_code=ModelFailureReason.STREAM_EVENT_INVALID,
            ) from exc

    def finish(self) -> tuple[ModelProviderStreamPart, ...]:
        if not self._completed:
            raise ModelProviderError(
                ModelErrorKind.INVALID_RESPONSE,
                "Responses stream ended before response.completed",
                reason_code=ModelFailureReason.STREAM_INCOMPLETE,
            )
        return ()


def _input_items(message: Message, model: ModelSpec) -> list[dict[str, object]]:
    if isinstance(message, ToolMessage):
        return [_tool_result_value(result) for result in message.results]
    items: list[dict[str, object]] = []
    if isinstance(message, AIMessage):
        continuation = message.continuation
        if continuation is not None and (
            continuation.provider == model.provider
            and continuation.protocol == model.protocol
        ):
            items.extend(_continuation_items(continuation))
    if message.content:
        items.append({"role": message.role, "content": message.content})
    if isinstance(message, AIMessage):
        items.extend(
            {
                "type": "function_call",
                "call_id": call.call_id,
                "name": call.name,
                "arguments": json.dumps(
                    cast(FrozenJsonObject, call.arguments).to_dict(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
            for call in message.tool_calls
        )
    return items or [{"role": message.role, "content": ""}]


def _continuation_items(continuation: ModelContinuation) -> list[dict[str, object]]:
    data = cast(FrozenJsonObject, continuation.data)
    if set(data) != {"version", "items"} or data["version"] != 1:
        raise ValueError("Responses continuation has an invalid shape")
    raw_items = data["items"]
    if not isinstance(raw_items, tuple):
        raise TypeError("Responses continuation items must be an array")
    items = [item.to_dict() for item in raw_items if isinstance(item, FrozenJsonObject)]
    if len(items) != len(raw_items) or any(item.get("type") != "reasoning" for item in items):
        raise ValueError("Responses continuation contains an invalid item")
    return items


def _tool_result_value(result: ToolResult) -> dict[str, object]:
    output = result.output if result.status == "succeeded" else result.error
    if not isinstance(output, str):
        output = json.dumps(thaw_json(output), ensure_ascii=False, separators=(",", ":"))
    return {
        "type": "function_call_output",
        "call_id": result.call_id,
        "output": output or "",
    }


def _tool_value(tool: ToolDefinition) -> dict[str, object]:
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": thaw_json(cast(FrozenJsonObject, tool.parameters)),
    }


def _tool_choice(
    choice: str | None, tools: tuple[ToolDefinition, ...]
) -> object:
    if choice is None or choice == "auto":
        return "auto"
    if choice in {"required", "none"}:
        return choice
    if choice not in {tool.name for tool in tools}:
        raise ValueError("tool_choice must reference a visible tool")
    return {"type": "function", "name": choice}


def _decode_output(
    output: list[object], request: ModelProviderRequest
) -> tuple[str, tuple[ToolCall, ...], list[dict[str, object]]]:
    text: list[str] = []
    calls: list[ToolCall] = []
    reasoning: list[dict[str, object]] = []
    visible_tools = {tool.name for tool in request.tools}
    for raw in output:
        if not isinstance(raw, Mapping):
            raise TypeError
        kind = raw.get("type")
        if kind == "reasoning":
            reasoning.append(dict(raw))
        elif kind == "message":
            content = raw.get("content")
            if not isinstance(content, list):
                raise TypeError
            for part in content:
                if not isinstance(part, Mapping):
                    raise TypeError
                if part.get("type") == "output_text":
                    value = part.get("text")
                elif part.get("type") == "refusal":
                    value = part.get("refusal")
                else:
                    continue
                if not isinstance(value, str):
                    raise TypeError
                text.append(value)
        elif kind == "function_call":
            call_id, name, arguments = (
                raw.get("call_id"),
                raw.get("name"),
                raw.get("arguments"),
            )
            if (
                not isinstance(call_id, str)
                or not call_id
                or not isinstance(name, str)
                or not name
                or not isinstance(arguments, str)
                or (visible_tools and name not in visible_tools)
            ):
                raise TypeError
            decoded = json.loads(arguments)
            if not isinstance(decoded, Mapping):
                raise TypeError
            calls.append(ToolCall(call_id=call_id, name=name, arguments=decoded))
    return "".join(text), tuple(calls), reasoning


def _validate_structured_output(
    request: ModelProviderRequest, content: str
) -> str:
    schema = request.generation.response_schema
    if schema is None:
        return content
    try:
        value = json.loads(content)
        jsonschema.validate(
            value,
            cast(FrozenJsonObject, schema).to_dict(),
        )
    except (json.JSONDecodeError, jsonschema.ValidationError) as exc:
        raise ModelProviderError(
            ModelErrorKind.INVALID_RESPONSE,
            "Responses output does not match the declared JSON schema",
            reason_code=ModelFailureReason.GENERATION_SCHEMA_INVALID,
        ) from exc
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _delta(event: Mapping[str, object]) -> str:
    value = event.get("delta")
    if not isinstance(value, str):
        raise TypeError
    return value


def _index(event: Mapping[str, object]) -> int:
    value = event.get("output_index")
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TypeError
    return value


def _stream_call(
    item: Mapping[str, object], event: Mapping[str, object]
) -> ModelProviderStreamPart:
    call_id, name = item.get("call_id"), item.get("name")
    if not isinstance(call_id, str) or not call_id or not isinstance(name, str) or not name:
        raise TypeError
    return ModelProviderStreamPart(
        "tool_call",
        {
            "index": _index(event),
            "call_id_delta": call_id,
            "name_delta": name,
            "arguments_delta": "",
        },
    )


def _status_error(status: int) -> ModelProviderError:
    kind = (
        ModelErrorKind.AUTHENTICATION
        if status in {401, 403}
        else ModelErrorKind.RATE_LIMIT
        if status == 429
        else ModelErrorKind.TIMEOUT
        if status in {408, 504}
        else ModelErrorKind.INVALID_REQUEST
        if 400 <= status < 500
        else ModelErrorKind.UNAVAILABLE
        if status >= 500
        else ModelErrorKind.UNKNOWN
    )
    return ModelProviderError(kind, "Responses provider request failed")


def openai_responses_adapters() -> dict[str, OpenAIResponsesAdapter]:
    return {_PROTOCOL: OpenAIResponsesAdapter()}


__all__ = [
    "OpenAIResponsesAdapter",
    "OpenAIResponsesClient",
    "openai_responses_adapters",
]
