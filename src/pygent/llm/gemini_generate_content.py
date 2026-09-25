"""Google Gemini generateContent HTTP transport and wire codec."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import Self, cast
from urllib.parse import quote

import httpx
import jsonschema  # type: ignore[import-untyped]

from pygent.core import (
    AIMessage,
    FrozenJsonObject,
    JsonValue,
    Message,
    ModelContinuation,
    ToolMessage,
    UserMessage,
    freeze_json_object,
    thaw_json,
)
from pygent.tool import (
    MediaBlock,
    MediaSource,
    ToolCall,
    ToolDefinition,
    ToolResult,
    ToolResultJson,
    ToolResultText,
)
from pygent.tool._schema import validate_instance

from ._adapter_contracts import (
    MediaResolver,
    MediaTransportCapabilities,
    ModelProviderRequest,
    ModelProviderResponse,
    ModelProviderStreamPart,
)
from ._continuation import continuation_matches
from ._json_sse_transport import _HTTPResponseError, _JsonSSETransport
from ._media_content import media_base64, validate_media_delivery
from ._media_tokens import gemini_media_input_tokens
from ._tool_context import tool_context_payload
from .configuration import ModelSpec
from .types import (
    ModelErrorKind,
    ModelFailureReason,
    ModelProviderError,
)

_PROTOCOL = "gemini_generate_content"
_SAFETY_REASONS = frozenset(
    {"SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "IMAGE_SAFETY"}
)
_GEMINI_IMAGE_MIME_TYPES = frozenset({"image/png", "image/jpeg", "image/webp"})


class GeminiGenerateContentClient:
    """HTTP/SSE client for Gemini generateContent and streamGenerateContent."""

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
            request_headers.setdefault("x-goog-api-key", api_key)
        self._api_root = base_url.rstrip("/")
        self._transport = _JsonSSETransport(
            headers=request_headers,
            client=client,
            verify_ssl=verify_ssl,
            trust_env_url=self._api_root,
        )

    def _endpoint(self, model: ModelSpec, *, stream: bool) -> str:
        model_id = quote(model.model_id, safe="-._~")
        method = "streamGenerateContent?alt=sse" if stream else "generateContent"
        return f"{self._api_root}/models/{model_id}:{method}"

    async def invoke(
        self, model: ModelSpec, payload: FrozenJsonObject
    ) -> FrozenJsonObject:
        try:
            return await self._transport.request_json(
                "POST", self._endpoint(model, stream=False), payload.to_dict()
            )
        except _HTTPResponseError as exc:
            raise _status_error(exc.status) from None
        except asyncio.CancelledError:
            raise
        except (TypeError, ValueError) as exc:
            raise ModelProviderError(
                ModelErrorKind.INVALID_RESPONSE,
                "Gemini returned invalid JSON",
                reason_code=ModelFailureReason.PROVIDER_PAYLOAD_INVALID,
            ) from exc

    async def stream(
        self, model: ModelSpec, payload: FrozenJsonObject
    ) -> AsyncIterator[FrozenJsonObject]:
        try:
            async for frame in self._transport.stream_sse(
                self._endpoint(model, stream=True), payload.to_dict()
            ):
                try:
                    value = json.loads(frame.data)
                except json.JSONDecodeError as exc:
                    raise ModelProviderError(
                        ModelErrorKind.INVALID_RESPONSE,
                        "Gemini returned an invalid SSE event",
                        reason_code=ModelFailureReason.STREAM_EVENT_INVALID,
                    ) from exc
                if not isinstance(value, Mapping):
                    raise ModelProviderError(
                        ModelErrorKind.INVALID_RESPONSE,
                        "Gemini SSE event must be an object",
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


class GeminiGenerateContentAdapter:
    """Strict codec for Google's Gemini generateContent contract."""

    protocol = _PROTOCOL

    def __init__(
        self,
        *,
        media_resolver: MediaResolver | Callable[[MediaSource], bytes] | None = None,
    ) -> None:
        if media_resolver is not None and not (
            isinstance(media_resolver, MediaResolver) or callable(media_resolver)
        ):
            raise TypeError("media_resolver must provide resolve() or be callable")
        source_kinds = ("inline", "resource") if media_resolver else ("inline",)
        self.media_transport = MediaTransportCapabilities(
            enabled=True,
            modalities=("image",),
            source_kinds=source_kinds,
            image_mime_types=tuple(sorted(_GEMINI_IMAGE_MIME_TYPES)),
        )
        self.media_resolver = media_resolver

    def media_delivery_gaps(
        self, block: MediaBlock, model: ModelSpec
    ) -> tuple[str, ...]:
        return _gemini_media_delivery_gaps(block, model)

    def estimate_media_input_tokens(
        self, block: MediaBlock, model: ModelSpec
    ) -> int | None:
        if model.provider != "google":
            return None
        return gemini_media_input_tokens(block)

    def validate_model(self, model: ModelSpec) -> None:
        options = cast(FrozenJsonObject, model.provider_options)
        unknown = set(options) - {"thinking_config"}
        if unknown:
            raise ValueError(
                "unknown Gemini provider options: " + ", ".join(sorted(unknown))
            )
        thinking = options.get("thinking_config")
        if thinking is None:
            return
        if not isinstance(thinking, FrozenJsonObject):
            raise TypeError("thinking_config must be an object")
        if set(thinking) - {"include_thoughts", "thinking_budget"}:
            raise ValueError("thinking_config has unknown fields")
        include = thinking.get("include_thoughts")
        if include is not None and not isinstance(include, bool):
            raise TypeError("thinking_config.include_thoughts must be a bool")
        budget = thinking.get("thinking_budget")
        if budget is not None and (
            not isinstance(budget, int) or isinstance(budget, bool) or budget < -1
        ):
            raise ValueError("thinking_config.thinking_budget must be at least -1")

    def build_request(self, request: ModelProviderRequest) -> FrozenJsonObject:
        try:
            self.validate_model(request.model)
            body: dict[str, object] = {
                "contents": [
                    _content(
                        message,
                        request.model,
                        request.model_key,
                        capabilities=self.media_transport,
                        media_resolver=self.media_resolver,
                    )
                    for message in (*request.context.messages, request.message)
                ]
            }
            if request.context.system_prompt:
                body["systemInstruction"] = {
                    "parts": [{"text": request.context.system_prompt}]
                }
            generation: dict[str, object] = {}
            if request.generation.max_output_tokens is not None:
                generation["maxOutputTokens"] = request.generation.max_output_tokens
            if request.generation.temperature is not None:
                generation["temperature"] = request.generation.temperature
            if request.generation.response_schema is not None:
                generation["responseMimeType"] = "application/json"
                generation["responseJsonSchema"] = thaw_json(
                    cast(FrozenJsonObject, request.generation.response_schema)
                )
            options = cast(FrozenJsonObject, request.model.provider_options)
            thinking = options.get("thinking_config")
            if isinstance(thinking, FrozenJsonObject):
                projected: dict[str, object] = {}
                if "include_thoughts" in thinking:
                    projected["includeThoughts"] = thinking["include_thoughts"]
                if "thinking_budget" in thinking:
                    projected["thinkingBudget"] = thinking["thinking_budget"]
                generation["thinkingConfig"] = projected
            if generation:
                body["generationConfig"] = generation
            if request.tools:
                body["tools"] = [
                    {
                        "functionDeclarations": [
                            _tool_value(tool) for tool in request.tools
                        ]
                    }
                ]
                body["toolConfig"] = {
                    "functionCallingConfig": _tool_choice(
                        request.generation.tool_choice, request.tools
                    )
                }
            elif request.generation.tool_choice not in (None, "none"):
                raise ValueError("tool_choice requires at least one visible tool")
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
            candidate = _candidate(body)
            finish = candidate.get("finishReason")
            _raise_for_safety(finish)
            content = candidate.get("content")
            if not isinstance(content, Mapping) or content.get("role") not in {
                None,
                "model",
            }:
                raise TypeError
            parts = content.get("parts")
            if not isinstance(parts, list):
                raise TypeError
            text, calls, continuation_parts = _decode_parts(parts)
            text = _validate_structured_output(request, text)
            usage = _usage(body.get("usageMetadata"))
        except ModelProviderError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelProviderError(
                ModelErrorKind.INVALID_RESPONSE,
                "Gemini response has an invalid generateContent shape",
                reason_code=ModelFailureReason.COMPLETION_SHAPE_INVALID,
            ) from exc
        continuation = (
            ModelContinuation(
                model_key=request.model_key,
                provider=request.model.provider,
                model_id=request.model.model_id,
                protocol=self.protocol,
                data={"version": 1, "parts": continuation_parts},
            )
            if continuation_parts
            else None
        )
        return ModelProviderResponse(
            message=AIMessage(
                content=text,
                tool_calls=calls,
                metadata={"model_key": request.model_key},
                continuation=continuation,
            ),
            usage=usage,
            finish_reason="tool_calls" if calls else _finish_reason(finish),
        )

    def create_stream_decoder(
        self, request: ModelProviderRequest
    ) -> _GeminiStreamDecoder:
        return _GeminiStreamDecoder(request)

    def normalize_error(self, error: BaseException) -> ModelErrorKind:
        if isinstance(error, ModelProviderError):
            return error.kind
        if isinstance(error, (TimeoutError, httpx.TimeoutException)):
            return ModelErrorKind.TIMEOUT
        if isinstance(error, httpx.TransportError):
            return ModelErrorKind.UNAVAILABLE
        return ModelErrorKind.UNKNOWN


class _GeminiStreamDecoder:
    def __init__(self, request: ModelProviderRequest) -> None:
        self._request = request
        self._completed = False
        self._continuation_parts: list[dict[str, object]] = []
        self._has_thought_signature = False
        self._tool_index = 0

    def feed(self, payload: FrozenJsonObject) -> tuple[ModelProviderStreamPart, ...]:
        try:
            body = payload.to_dict()
            candidate = _candidate(body)
            finish = candidate.get("finishReason")
            _raise_for_safety(finish)
            content = candidate.get("content")
            if not isinstance(content, Mapping):
                raise TypeError
            raw_parts = content.get("parts")
            if not isinstance(raw_parts, list):
                raise TypeError
            parts: list[ModelProviderStreamPart] = []
            for raw in raw_parts:
                if not isinstance(raw, Mapping):
                    raise TypeError
                self._continuation_parts.append(dict(raw))
                self._has_thought_signature = self._has_thought_signature or isinstance(
                    raw.get("thoughtSignature"), str
                )
                if raw.get("thought") is True:
                    text = raw.get("text")
                    if not isinstance(text, str):
                        raise TypeError
                    if text:
                        parts.append(
                            ModelProviderStreamPart("reasoning", {"text": text})
                        )
                elif "text" in raw:
                    text = raw["text"]
                    if not isinstance(text, str):
                        raise TypeError
                    if text:
                        parts.append(ModelProviderStreamPart("text", {"text": text}))
                elif "functionCall" in raw:
                    call = raw["functionCall"]
                    if not isinstance(call, Mapping):
                        raise TypeError
                    call_id = call.get("id")
                    name, arguments = call.get("name"), call.get("args", {})
                    if (
                        (
                            call_id is not None
                            and (not isinstance(call_id, str) or not call_id)
                        )
                        or not isinstance(name, str)
                        or not name
                        or not isinstance(arguments, Mapping)
                    ):
                        raise TypeError
                    parts.append(
                        ModelProviderStreamPart(
                            "tool_call",
                            {
                                "index": self._tool_index,
                                "call_id_delta": call_id
                                or f"gemini-call-{self._tool_index}",
                                "name_delta": name,
                                "arguments_delta": json.dumps(
                                    dict(arguments),
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                ),
                            },
                        )
                    )
                    self._tool_index += 1
            usage = _usage(body.get("usageMetadata"))
            if usage:
                parts.append(ModelProviderStreamPart("usage", usage))
            if finish is not None:
                if self._has_thought_signature:
                    parts.append(
                        ModelProviderStreamPart(
                            "continuation",
                            {
                                "model_key": self._request.model_key,
                                "provider": self._request.model.provider,
                                "model_id": self._request.model.model_id,
                                "protocol": _PROTOCOL,
                                "data": {
                                    "version": 1,
                                    "parts": self._continuation_parts,
                                },
                            },
                        )
                    )
                parts.append(
                    ModelProviderStreamPart(
                        "finish", {"finish_reason": _finish_reason(finish)}
                    )
                )
                self._completed = True
            return tuple(parts)
        except ModelProviderError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelProviderError(
                ModelErrorKind.INVALID_RESPONSE,
                "Gemini SSE event has an invalid shape",
                reason_code=ModelFailureReason.STREAM_EVENT_INVALID,
            ) from exc

    def finish(self) -> tuple[ModelProviderStreamPart, ...]:
        if not self._completed:
            raise ModelProviderError(
                ModelErrorKind.INVALID_RESPONSE,
                "Gemini stream ended before a finish reason",
                reason_code=ModelFailureReason.STREAM_INCOMPLETE,
            )
        return ()


def _content(
    message: Message,
    model: ModelSpec,
    model_key: str,
    *,
    capabilities: MediaTransportCapabilities,
    media_resolver: MediaResolver | Callable[[MediaSource], bytes] | None,
) -> dict[str, object]:
    role = "model" if isinstance(message, AIMessage) else "user"
    if isinstance(message, ToolMessage):
        return {
            "role": "user",
            "parts": [
                _function_response(
                    result,
                    model=model,
                    capabilities=capabilities,
                    media_resolver=media_resolver,
                )
                for result in message.results
            ],
        }
    parts: list[dict[str, object]] = []
    if isinstance(message, AIMessage):
        continuation = message.continuation
        if continuation is not None and continuation_matches(
            continuation,
            model_key=model_key,
            model=model,
        ):
            parts = _continuation_parts(continuation)
            replay_text, replay_calls, _ = _decode_parts(parts)
            if replay_text != message.content or replay_calls != message.tool_calls:
                raise ValueError(
                    "Gemini continuation does not match the assistant message"
                )
            return {"role": role, "parts": parts}
    if message.content:
        parts.append({"text": message.content})
    if isinstance(message, UserMessage):
        parts.extend(
            _inline_data_part(
                block,
                model=model,
                capabilities=capabilities,
                media_resolver=media_resolver,
            )
            for block in message.media
        )
    if isinstance(message, AIMessage):
        parts.extend(
            {
                "functionCall": {
                    "id": call.call_id,
                    "name": call.name,
                    "args": cast(FrozenJsonObject, call.arguments).to_dict(),
                }
            }
            for call in message.tool_calls
        )
    return {"role": role, "parts": parts or [{"text": ""}]}


def _continuation_parts(continuation: ModelContinuation) -> list[dict[str, object]]:
    data = cast(FrozenJsonObject, continuation.data)
    if set(data) != {"version", "parts"} or data["version"] != 1:
        raise ValueError("Gemini continuation has an invalid shape")
    raw_parts = data["parts"]
    if not isinstance(raw_parts, tuple):
        raise TypeError("Gemini continuation parts must be an array")
    parts = [part.to_dict() for part in raw_parts if isinstance(part, FrozenJsonObject)]
    if len(parts) != len(raw_parts) or not any(
        isinstance(part.get("thoughtSignature"), str) and bool(part["thoughtSignature"])
        for part in parts
    ):
        raise ValueError("Gemini continuation must preserve signed response parts")
    return parts


def _inline_data_part(
    block: MediaBlock,
    *,
    model: ModelSpec,
    capabilities: MediaTransportCapabilities,
    media_resolver: MediaResolver | Callable[[MediaSource], bytes] | None,
) -> dict[str, object]:
    validate_media_delivery(
        block,
        model=model,
        capabilities=capabilities,
        allowed_mime_types=_GEMINI_IMAGE_MIME_TYPES,
    )
    return {
        "inlineData": {
            "mimeType": block.mime_type,
            "data": media_base64(
                block,
                capabilities=capabilities,
                media_resolver=media_resolver,
            ),
        }
    }


def _function_response(
    result: ToolResult,
    *,
    model: ModelSpec,
    capabilities: MediaTransportCapabilities,
    media_resolver: MediaResolver | Callable[[MediaSource], bytes] | None,
) -> dict[str, object]:
    parts: list[dict[str, object]] = []
    if result.content:
        values: list[object] = []
        for index, block in enumerate(result.content):
            if type(block) is ToolResultText:
                values.append(block.text)
            elif type(block) is ToolResultJson:
                values.append(thaw_json(block.value))
            elif type(block) is MediaBlock:
                if _gemini_media_delivery_gaps(block, model):
                    raise ModelProviderError(
                        ModelErrorKind.INVALID_REQUEST,
                        "model does not support multimodal function responses",
                        reason_code=ModelFailureReason.MEDIA_TRANSPORT_UNSUPPORTED,
                    )
                validate_media_delivery(
                    block,
                    model=model,
                    capabilities=capabilities,
                    allowed_mime_types=_GEMINI_IMAGE_MIME_TYPES,
                )
                display_name = f"{result.call_id}-{index}"
                values.append({"$ref": display_name})
                parts.append(
                    {
                        "inlineData": {
                            "mimeType": block.mime_type,
                            "displayName": display_name,
                            "data": media_base64(
                                block,
                                capabilities=capabilities,
                                media_resolver=media_resolver,
                            ),
                        }
                    }
                )
            else:
                raise TypeError("tool-result content block is invalid")
        response: object = {"content": values}
    else:
        if result.status == "succeeded":
            value: object = result.output
        else:
            value = {"tool_context": tool_context_payload(result)}
        if isinstance(value, FrozenJsonObject):
            response = value.to_dict()
        elif isinstance(value, Mapping):
            response = dict(value)
        else:
            response = {"result": thaw_json(cast(JsonValue, value))}
    function_response: dict[str, object] = {
        "id": result.call_id,
        "name": result.name,
        "response": response,
    }
    if parts:
        function_response["parts"] = parts
    return {"functionResponse": function_response}


def _gemini_media_delivery_gaps(
    block: MediaBlock, model: ModelSpec
) -> tuple[str, ...]:
    gaps: list[str] = []
    if not model.model_id.lower().startswith("gemini-3"):
        gaps.append("media_transport.model")
    if block.mime_type not in _GEMINI_IMAGE_MIME_TYPES:
        gaps.append("media_transport.mime_type")
    return tuple(gaps)


def _tool_value(tool: ToolDefinition) -> dict[str, object]:
    return {
        "name": tool.name,
        "description": tool.description,
        "parametersJsonSchema": thaw_json(cast(FrozenJsonObject, tool.parameters)),
    }


def _tool_choice(
    choice: str | None, tools: tuple[ToolDefinition, ...]
) -> dict[str, object]:
    if choice is None or choice == "auto":
        return {"mode": "AUTO"}
    if choice == "none":
        return {"mode": "NONE"}
    if choice == "required":
        return {"mode": "ANY"}
    if choice not in {tool.name for tool in tools}:
        raise ValueError("tool_choice must reference a visible tool")
    return {"mode": "ANY", "allowedFunctionNames": [choice]}


def _candidate(body: Mapping[str, object]) -> Mapping[str, object]:
    candidates = body.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise TypeError
    candidate = candidates[0]
    if not isinstance(candidate, Mapping):
        raise TypeError
    return candidate


def _decode_parts(
    parts: Sequence[object],
) -> tuple[str, tuple[ToolCall, ...], list[dict[str, object]]]:
    text: list[str] = []
    calls: list[ToolCall] = []
    originals: list[dict[str, object]] = []
    has_thought_signature = False
    for raw in parts:
        if not isinstance(raw, Mapping):
            raise TypeError
        originals.append(dict(raw))
        has_thought_signature = has_thought_signature or isinstance(
            raw.get("thoughtSignature"), str
        )
        if raw.get("thought") is True:
            thought = raw.get("text")
            if not isinstance(thought, str):
                raise TypeError
        elif "text" in raw:
            value = raw["text"]
            if not isinstance(value, str):
                raise TypeError
            text.append(value)
        elif "functionCall" in raw:
            call = raw["functionCall"]
            if not isinstance(call, Mapping):
                raise TypeError
            call_id = call.get("id")
            name, arguments = call.get("name"), call.get("args", {})
            if (
                (call_id is not None and (not isinstance(call_id, str) or not call_id))
                or not isinstance(name, str)
                or not name
                or not isinstance(arguments, Mapping)
            ):
                raise TypeError
            calls.append(
                ToolCall(
                    call_id=call_id or f"gemini-call-{len(calls)}",
                    name=name,
                    arguments=arguments,
                )
            )
        else:
            raise TypeError
    continuation = originals if has_thought_signature else []
    return "".join(text), tuple(calls), continuation


def _validate_structured_output(request: ModelProviderRequest, content: str) -> str:
    schema = request.generation.response_schema
    if schema is None:
        return content
    try:
        value = json.loads(content)
        validate_instance(cast(FrozenJsonObject, schema), value)
    except (json.JSONDecodeError, jsonschema.ValidationError) as exc:
        raise ModelProviderError(
            ModelErrorKind.INVALID_RESPONSE,
            "Gemini output does not match the declared JSON schema",
            reason_code=ModelFailureReason.GENERATION_SCHEMA_INVALID,
        ) from exc
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _usage(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    fields = {
        "input_tokens": value.get("promptTokenCount"),
        "output_tokens": value.get("candidatesTokenCount"),
        "total_tokens": value.get("totalTokenCount"),
        "cached_input_tokens": value.get("cachedContentTokenCount"),
        "reasoning_tokens": value.get("thoughtsTokenCount"),
    }
    return {
        name: count
        for name, count in fields.items()
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0
    }


def _raise_for_safety(value: object) -> None:
    if value in _SAFETY_REASONS:
        raise ModelProviderError(
            ModelErrorKind.INVALID_RESPONSE,
            "Gemini generation was blocked by content policy",
            reason_code=ModelFailureReason.CONTENT_POLICY_REJECTED,
        )


def _finish_reason(value: object) -> str:
    if value in {None, "STOP"}:
        return "stop"
    if value == "MAX_TOKENS":
        return "length"
    if value in _SAFETY_REASONS:
        _raise_for_safety(value)
    if not isinstance(value, str):
        raise TypeError
    return "other"


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
    return ModelProviderError(kind, "Gemini provider request failed")


def gemini_generate_content_adapters() -> dict[str, GeminiGenerateContentAdapter]:
    return {_PROTOCOL: GeminiGenerateContentAdapter()}


__all__ = [
    "GeminiGenerateContentAdapter",
    "GeminiGenerateContentClient",
    "gemini_generate_content_adapters",
]
