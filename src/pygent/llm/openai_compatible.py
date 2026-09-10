"""OpenAI-compatible HTTP transport and wire codec."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from collections.abc import AsyncIterator, Mapping
from functools import lru_cache
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
    ModelProviderStreamKind,
    ModelProviderStreamPart,
    _canonical_usage,
    _normalized_finish_reason,
)
from ._json_sse_transport import _HTTPResponseError, _JsonSSETransport
from .catalog import ModelCatalog, ModelInfo
from .configuration import ModelSpec
from .types import (
    ModelErrorKind,
    ModelFailureReason,
    ModelProviderError,
)

_OPENAI_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MAX_PROVIDER_ERROR_BYTES = 64 * 1024
_PROVIDER_ERROR_REASONS = {
    "authentication_error": ModelFailureReason.AUTHENTICATION_FAILED,
    "invalid_api_key": ModelFailureReason.AUTHENTICATION_FAILED,
    "permission_denied": ModelFailureReason.PERMISSION_DENIED,
    "rate_limit_exceeded": ModelFailureReason.RATE_LIMITED,
    "insufficient_quota": ModelFailureReason.QUOTA_EXHAUSTED,
    "quota_exceeded": ModelFailureReason.QUOTA_EXHAUSTED,
    "model_not_found": ModelFailureReason.MODEL_NOT_FOUND,
    "context_length_exceeded": ModelFailureReason.CONTEXT_LENGTH_EXCEEDED,
    "content_policy_violation": ModelFailureReason.CONTENT_POLICY_REJECTED,
    "content_filter": ModelFailureReason.CONTENT_POLICY_REJECTED,
}
_OPENAI_RESERVED_PROVIDER_FIELDS = frozenset(
    {
        "model",
        "messages",
        "temperature",
        "response_format",
        "tools",
        "tool_choice",
        "stream",
    }
)
_TOKEN_LIMIT_PROVIDER_FIELDS = frozenset({"max_tokens", "max_completion_tokens"})
_JSON_FENCE = re.compile(
    r"\A\s*```(?:json)?\s*(.*?)\s*```\s*\Z",
    flags=re.IGNORECASE | re.DOTALL,
)
_FORBIDDEN_PROVIDER_OPTION_KEYS = frozenset(
    {
        "apikey",
        "apitoken",
        "token",
        "bearertoken",
        "accesstoken",
        "refreshtoken",
        "secret",
        "secrets",
        "password",
        "credential",
        "credentials",
        "cookie",
        "auth",
        "authentication",
        "authorization",
        "headers",
        "httpheaders",
        "endpoint",
        "baseurl",
        "proxy",
        "proxyurl",
        "proxyauthentication",
        "proxycredentials",
        "tlsprivatekey",
        "verifyssl",
        "privatekey",
        "client",
        "session",
        "connection",
        "connectionstring",
        "connectionpool",
        "dsn",
        "lock",
        "task",
        "coroutine",
        "callback",
        "retry",
        "retries",
        "backoff",
        "attempttimeout",
        "timeout",
        "deadline",
        "stream",
        "runtime",
        "binding",
        "execution",
        "resourceresolver",
        "rawresponse",
        "internalexception",
    }
)


def _normalized_option_key(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _validate_no_forbidden_option_keys(value: object) -> None:
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, FrozenJsonObject):
            for key, item in current.items():
                if _normalized_option_key(key) in _FORBIDDEN_PROVIDER_OPTION_KEYS:
                    raise ValueError(f"provider option field {key!r} is not portable")
                pending.append(item)
        elif isinstance(current, tuple):
            pending.extend(current)


def _validate_openai_provider_options(model: ModelSpec) -> None:
    options = cast(FrozenJsonObject, model.provider_options)
    conflicts = set(options) & _OPENAI_RESERVED_PROVIDER_FIELDS
    if conflicts:
        raise ValueError(
            "provider options cannot override reserved fields: "
            + ", ".join(sorted(conflicts))
        )
    _validate_no_forbidden_option_keys(options)
    token_fields = set(options) & _TOKEN_LIMIT_PROVIDER_FIELDS
    if len(token_fields) > 1:
        raise ValueError(
            "provider options accept only one of max_tokens and "
            "max_completion_tokens"
        )
    for key in token_fields:
        value = options[key]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"provider option {key!r} must be a positive integer")
    if model.provider != "deepseek" or "thinking" not in options:
        return
    thinking = options["thinking"]
    if not isinstance(thinking, FrozenJsonObject):
        raise TypeError("provider option 'thinking' must be an object")
    if set(thinking) != {"type"}:
        raise ValueError("provider option 'thinking' accepts only the 'type' field")
    if thinking["type"] not in ("enabled", "disabled"):
        raise ValueError(
            "provider option 'thinking.type' must be 'enabled' or 'disabled'"
        )


class OpenAICompatibleClient:
    """Small HTTP/SSE client for OpenAI, GLM, Qwen, and compatible endpoints."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        headers: Mapping[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
        verify_ssl: bool | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url must be non-empty")
        if verify_ssl is not None and not isinstance(verify_ssl, bool):
            raise TypeError("verify_ssl must be a bool or None")
        if client is not None and verify_ssl is not None:
            raise ValueError("verify_ssl cannot be set with an injected HTTP client")
        request_headers = dict(headers or {})
        if api_key is not None:
            request_headers.setdefault("Authorization", f"Bearer {api_key}")
        api_root = base_url.rstrip("/")
        self._endpoint = f"{api_root}/chat/completions"
        self._models_endpoint = f"{api_root}/models"
        self._transport = _JsonSSETransport(
            headers=request_headers,
            client=client,
            verify_ssl=verify_ssl,
            trust_env_url=api_root,
        )
        self._models: ModelCatalog = _OpenAICompatibleModelCatalog(self)

    @property
    def models(self) -> ModelCatalog:
        """Models visible to this endpoint and credential at query time."""

        return self._models

    async def invoke(
        self, model: ModelSpec, payload: FrozenJsonObject
    ) -> FrozenJsonObject:
        try:
            return await self._transport.request_json(
                "POST", self._endpoint, payload.to_dict()
            )
        except _HTTPResponseError as exc:
            raise _provider_status_error(exc.status, exc.body) from None
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
        options = body.get("stream_options", {})
        if options is None:
            body.pop("stream_options", None)
        elif isinstance(options, dict):
            options.setdefault("include_usage", True)
            body["stream_options"] = options
        try:
            async for frame in self._transport.stream_sse(self._endpoint, body):
                data = frame.data.strip()
                if data == "[DONE]":
                    yield freeze_json_object({"done": True})
                    return
                try:
                    item = json.loads(data)
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
            raise _provider_status_error(exc.status, exc.body) from None

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
            raise _provider_status_error(exc.status, exc.body) from None
        except (TypeError, ValueError) as exc:
            raise ModelProviderError(
                ModelErrorKind.INVALID_RESPONSE,
                "model catalog returned invalid JSON",
            ) from exc
        except ModelProviderError:
            raise
        except Exception as exc:  # noqa: BLE001 - provider transport boundary
            kind = _normalize_openai_error(exc)
            raise ModelProviderError(kind, "model catalog request failed") from None


class _OpenAICompatibleModelCatalog:
    def __init__(self, client: OpenAICompatibleClient) -> None:
        self._client = client

    async def list(self, *, timeout: float | None = 10.0) -> tuple[ModelInfo, ...]:
        _validate_catalog_timeout(timeout)
        payload = (await self._client._list_models_payload(timeout=timeout)).to_dict()
        object_type = payload.get("object")
        if object_type not in (None, "list"):
            raise ModelProviderError(
                ModelErrorKind.INVALID_RESPONSE,
                "model catalog response has an invalid object type",
            )
        data = payload.get("data")
        if not isinstance(data, list):
            raise ModelProviderError(
                ModelErrorKind.INVALID_RESPONSE,
                "model catalog data must be an array",
            )
        models: list[ModelInfo] = []
        seen: set[str] = set()
        for item in data:
            if not isinstance(item, Mapping):
                raise ModelProviderError(
                    ModelErrorKind.INVALID_RESPONSE,
                    "model catalog entry must be an object",
                )
            item_type = item.get("object")
            model_id = item.get("id")
            created = item.get("created")
            owned_by = item.get("owned_by")
            if item_type not in (None, "model"):
                raise ModelProviderError(
                    ModelErrorKind.INVALID_RESPONSE,
                    "model catalog entry has an invalid object type",
                )
            try:
                model = ModelInfo(
                    id=cast(str, model_id),
                    created=cast(int | None, created),
                    owned_by=cast(str | None, owned_by),
                )
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
            models.append(model)
        return tuple(models)


class OpenAICompatibleAdapter:
    """OpenAI chat-completions codec shared by compatible providers."""

    protocol = "openai_chat_completions"

    def validate_model(self, model: ModelSpec) -> None:
        """Validate portable provider options without performing provider I/O."""

        _validate_openai_provider_options(model)

    def build_request(self, request: ModelProviderRequest) -> FrozenJsonObject:
        try:
            self.validate_model(request.model)
        except (TypeError, ValueError) as exc:
            raise ModelProviderError(ModelErrorKind.INVALID_REQUEST, str(exc)) from None
        messages: list[dict[str, object]] = []
        wire_names = {tool.name: _openai_tool_name(tool.name) for tool in request.tools}
        if request.context.system_prompt:
            messages.append(
                {"role": "system", "content": request.context.system_prompt}
            )
        for item in request.context.messages:
            messages.extend(_encode_messages(item, wire_names, model=request.model))
        messages.extend(
            _encode_messages(request.message, wire_names, model=request.model)
        )
        body: dict[str, object] = {
            "model": request.model.model_id,
            "messages": messages,
        }
        generation = request.generation
        options = cast(FrozenJsonObject, request.model.provider_options)
        token_fields = set(options) & _TOKEN_LIMIT_PROVIDER_FIELDS
        if generation.max_output_tokens is not None and token_fields:
            raise ModelProviderError(
                ModelErrorKind.INVALID_REQUEST,
                "provider token limit options conflict with max_output_tokens",
            )
        if generation.temperature is not None:
            body["temperature"] = generation.temperature
        if generation.max_output_tokens is not None:
            body["max_tokens"] = generation.max_output_tokens
        if generation.response_schema is not None:
            body["response_format"] = _response_format_projection(
                cast(FrozenJsonObject, generation.response_schema),
                generation.response_schema_name,
            )
        if request.tools:
            body["tools"] = [
                _tool_projection(tool, wire_names[tool.name]) for tool in request.tools
            ]
            choice = generation.tool_choice
            if choice in ("auto", "required", "none"):
                body["tool_choice"] = choice
            elif choice is not None:
                wire_name = wire_names.get(choice)
                if wire_name is None:
                    raise ModelProviderError(
                        ModelErrorKind.INVALID_REQUEST,
                        "tool_choice must reference a visible tool",
                    )
                body["tool_choice"] = {
                    "type": "function",
                    "function": {"name": wire_name},
                }
        elif generation.tool_choice not in (None, "none"):
            raise ModelProviderError(
                ModelErrorKind.INVALID_REQUEST,
                "tool_choice requires at least one visible tool",
            )
        body.update(options.to_dict())
        return freeze_json_object(body)

    def parse_response(
        self, request: ModelProviderRequest, payload: FrozenJsonObject
    ) -> ModelProviderResponse:
        body = payload.to_dict()
        request_id = body.get("id")
        if request_id is not None and not isinstance(request_id, str):
            request_id = None
        try:
            choices = body["choices"]
            if not isinstance(choices, list) or not choices:
                raise TypeError
            choice = choices[0]
            if not isinstance(choice, dict):
                raise TypeError
            raw_finish_reason = choice.get("finish_reason")
            if raw_finish_reason is not None and not isinstance(raw_finish_reason, str):
                raise TypeError
            finish_reason = (
                "other"
                if raw_finish_reason is None
                else _normalized_finish_reason(raw_finish_reason)
            )
            raw_message = choice.get("message")
            reasoning_content: object = None
            if isinstance(raw_message, str):
                content = raw_message
                raw_tool_calls: object = []
            elif isinstance(raw_message, dict):
                reasoning_content = raw_message.get("reasoning_content")
                if (
                    request.model.provider == "deepseek"
                    and "reasoning_content" in raw_message
                    and not isinstance(reasoning_content, str)
                ):
                    raise TypeError
                content_value = raw_message.get("content")
                if content_value is None and isinstance(
                    raw_message.get("refusal"), str
                ):
                    content_value = raw_message["refusal"]
                if content_value is None and "text" in choice:
                    content_value = choice["text"]
                content = _decode_text_content(content_value)
                raw_tool_calls = raw_message.get("tool_calls")
                if raw_tool_calls is None and raw_message.get("function_call") is not None:
                    raw_tool_calls = [
                        {"type": "function", "function": raw_message["function_call"]}
                    ]
            elif isinstance(choice.get("text"), str):
                content = cast(str, choice["text"])
                raw_tool_calls = []
            else:
                raise TypeError
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelProviderError(
                ModelErrorKind.INVALID_RESPONSE,
                "provider response has an invalid completion shape",
                reason_code=ModelFailureReason.COMPLETION_SHAPE_INVALID,
            ) from exc
        try:
            tool_calls = _decode_tool_calls(
                raw_tool_calls,
                {_openai_tool_name(tool.name): tool.name for tool in request.tools},
                model_key=request.model_key,
                provider_request_id=cast(str | None, request_id),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelProviderError(
                ModelErrorKind.INVALID_RESPONSE,
                "provider response contains an invalid tool call",
                reason_code=ModelFailureReason.TOOL_CALL_INVALID,
            ) from exc

        if request.generation.response_schema is not None:
            try:
                value = _decode_json(content)
                jsonschema.validate(
                    value,
                    _schema_projection(
                        cast(FrozenJsonObject, request.generation.response_schema)
                    ),
                )
                content = _wire_json(value)
            except (json.JSONDecodeError, jsonschema.ValidationError) as exc:
                raise ModelProviderError(
                    ModelErrorKind.INVALID_RESPONSE,
                    "model output does not match the declared JSON schema",
                    reason_code=ModelFailureReason.GENERATION_SCHEMA_INVALID,
                ) from exc

        usage = _canonical_usage(body.get("usage"))
        continuation = None
        if request.model.provider == "deepseek" and isinstance(
            reasoning_content, str
        ):
            continuation = ModelContinuation(
                provider="deepseek",
                protocol=self.protocol,
                data={"version": 1, "reasoning_content": reasoning_content},
            )
        message = AIMessage(
            content=content,
            tool_calls=tool_calls,
            metadata={"model_key": request.model_key},
            continuation=continuation,
        )
        return ModelProviderResponse(
            message=message,
            usage=usage,
            provider_request_id=request_id,
            finish_reason=finish_reason,
        )

    def create_stream_decoder(
        self, request: ModelProviderRequest
    ) -> _OpenAIStreamDecoder:
        return _OpenAIStreamDecoder(self, request)

    def _decode_stream_payload(
        self, request: ModelProviderRequest, payload: FrozenJsonObject
    ) -> tuple[ModelProviderStreamPart, ...]:
        body = payload.to_dict()
        if body.get("done") is True:
            return (ModelProviderStreamPart("finish", {"finish_reason": "other"}),)
        parts: list[ModelProviderStreamPart] = []
        usage = body.get("usage")
        if isinstance(usage, Mapping):
            parts.append(ModelProviderStreamPart("usage", _canonical_usage(usage)))
        if "choices" not in body:
            return tuple(parts)
        try:
            choices = body["choices"]
            if not isinstance(choices, list):
                raise TypeError
            if not choices:
                return tuple(parts)
            choice = choices[0]
            if not isinstance(choice, dict):
                raise TypeError
            delta = choice.get("delta")
            if delta is None:
                delta = {}
            if not isinstance(delta, dict):
                raise TypeError
            reasoning = next(
                (
                    delta.get(key)
                    for key in ("reasoning_content", "reasoning", "thinking")
                    if isinstance(delta.get(key), str) and delta.get(key)
                ),
                None,
            )
            if isinstance(reasoning, str):
                parts.append(ModelProviderStreamPart("reasoning", {"text": reasoning}))
            content_value = delta.get("content")
            if content_value is None and isinstance(delta.get("refusal"), str):
                content_value = delta["refusal"]
            content = _decode_text_content(content_value)
            if content:
                parts.append(ModelProviderStreamPart("text", {"text": content}))
            tool_calls = delta.get("tool_calls")
            if tool_calls is not None and not isinstance(tool_calls, list):
                raise TypeError
            for position, call in enumerate(tool_calls or ()):
                if not isinstance(call, dict):
                    raise TypeError
                function = call.get("function")
                if function is None:
                    function = {}
                if not isinstance(function, dict):
                    raise TypeError
                index = call.get("index", position)
                if isinstance(index, str) and index.isdecimal():
                    index = int(index)
                call_id = call.get("id", "")
                name = function.get("name", "")
                arguments = function.get("arguments", "")
                if call_id is None:
                    call_id = ""
                if name is None:
                    name = ""
                if arguments is None:
                    arguments = ""
                elif isinstance(arguments, Mapping):
                    arguments = _wire_json(arguments)
                if (
                    not isinstance(index, int)
                    or isinstance(index, bool)
                    or index < 0
                    or not isinstance(call_id, str)
                    or not isinstance(name, str)
                    or not isinstance(arguments, str)
                ):
                    raise TypeError
                parts.append(
                    ModelProviderStreamPart(
                        "tool_call",
                        {
                            "index": index,
                            "call_id_delta": call_id,
                            "name_delta": name,
                            "arguments_delta": arguments,
                        },
                    )
                )
            compatible_call = delta.get("function_call")
            if compatible_call is not None:
                if not isinstance(compatible_call, dict):
                    raise TypeError
                name = compatible_call.get("name", "")
                arguments = compatible_call.get("arguments", "")
                if name is None:
                    name = ""
                if arguments is None:
                    arguments = ""
                elif isinstance(arguments, Mapping):
                    arguments = _wire_json(arguments)
                if not isinstance(name, str) or not isinstance(arguments, str):
                    raise TypeError
                parts.append(
                    ModelProviderStreamPart(
                        "tool_call",
                        {
                            "index": 0,
                            "call_id_delta": "",
                            "name_delta": name,
                            "arguments_delta": arguments,
                        },
                    )
                )
            finish_reason = choice.get("finish_reason")
            if finish_reason is not None:
                if not isinstance(finish_reason, str):
                    raise TypeError
                finish: dict[str, object] = {
                    "finish_reason": _normalized_finish_reason(finish_reason)
                }
                request_id = body.get("id")
                if isinstance(request_id, str) and request_id:
                    finish["provider_request_id"] = request_id
                parts.append(ModelProviderStreamPart("finish", finish))
        except (KeyError, TypeError) as exc:
            raise ModelProviderError(
                ModelErrorKind.INVALID_RESPONSE,
                "provider SSE event has an invalid completion shape",
                reason_code=ModelFailureReason.STREAM_EVENT_INVALID,
            ) from exc
        return tuple(parts)

    def normalize_error(self, error: BaseException) -> ModelErrorKind:
        return _normalize_openai_error(error)


class _OpenAIStreamDecoder:
    def __init__(
        self, adapter: OpenAICompatibleAdapter, request: ModelProviderRequest
    ) -> None:
        self._adapter = adapter
        self._request = request
        self._completed = False
        self._reasoning_parts: list[str] = []
        self._continuation_emitted = False

    def feed(
        self, payload: FrozenJsonObject
    ) -> tuple[ModelProviderStreamPart, ...]:
        parts = self._adapter._decode_stream_payload(self._request, payload)
        if self._request.model.provider == "deepseek":
            for part in parts:
                if part.kind == ModelProviderStreamKind.REASONING:
                    text = cast(FrozenJsonObject, part.data).get("text")
                    if isinstance(text, str) and text:
                        self._reasoning_parts.append(text)
            if (
                self._reasoning_parts
                and not self._continuation_emitted
                and any(
                    part.kind == ModelProviderStreamKind.FINISH for part in parts
                )
            ):
                continuation = ModelProviderStreamPart(
                    "continuation",
                    {
                        "provider": "deepseek",
                        "protocol": self._adapter.protocol,
                        "data": {
                            "version": 1,
                            "reasoning_content": "".join(self._reasoning_parts),
                        },
                    },
                )
                finish_index = next(
                    index
                    for index, part in enumerate(parts)
                    if part.kind == ModelProviderStreamKind.FINISH
                )
                parts = (*parts[:finish_index], continuation, *parts[finish_index:])
                self._continuation_emitted = True
        if any(part.kind == ModelProviderStreamKind.FINISH for part in parts):
            self._completed = True
        return parts

    def finish(self) -> tuple[ModelProviderStreamPart, ...]:
        if not self._completed:
            raise ModelProviderError(
                ModelErrorKind.INVALID_RESPONSE,
                "model stream ended before a completion marker",
                reason_code=ModelFailureReason.STREAM_INCOMPLETE,
            )
        return ()


def openai_compatible_adapters() -> dict[str, OpenAICompatibleAdapter]:
    """Return the built-in protocol codec."""

    return {"openai_chat_completions": OpenAICompatibleAdapter()}


def _validate_catalog_timeout(timeout: float | None) -> None:
    if timeout is None:
        return
    if (
        not isinstance(timeout, (int, float))
        or isinstance(timeout, bool)
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("timeout must be finite and positive, or None")


def _wire_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _provider_status_error(
    status: int, raw_body: str | bytes | None
) -> ModelProviderError:
    kind = _kind_for_http_status(status)
    reason = _reason_for_provider_failure(status, raw_body)
    return ModelProviderError(
        kind,
        f"model provider request failed: {reason.value}",
        reason_code=reason,
        http_status=status,
    )


def _kind_for_http_status(status: int) -> ModelErrorKind:
    if status in (401, 403):
        return ModelErrorKind.AUTHENTICATION
    if status == 429:
        return ModelErrorKind.RATE_LIMIT
    if status in (408, 504):
        return ModelErrorKind.TIMEOUT
    if 400 <= status < 500:
        return ModelErrorKind.INVALID_REQUEST
    if status >= 500:
        return ModelErrorKind.UNAVAILABLE
    return ModelErrorKind.UNKNOWN


def _reason_for_provider_failure(
    status: int, raw_body: str | bytes | None
) -> ModelFailureReason:
    if raw_body:
        body_bytes = (
            raw_body.encode("utf-8", errors="replace")
            if isinstance(raw_body, str)
            else raw_body
        )[:_MAX_PROVIDER_ERROR_BYTES]
        try:
            payload = json.loads(body_bytes)
        except (TypeError, ValueError):
            payload = None
        if isinstance(payload, Mapping):
            error = payload.get("error")
            if isinstance(error, Mapping):
                for key in ("code", "type"):
                    value = error.get(key)
                    if isinstance(value, str):
                        reason = _PROVIDER_ERROR_REASONS.get(value.strip().lower())
                        if reason is not None:
                            return reason
    if status == 401:
        return ModelFailureReason.AUTHENTICATION_FAILED
    if status == 403:
        return ModelFailureReason.PERMISSION_DENIED
    if status == 404:
        return ModelFailureReason.RESOURCE_NOT_FOUND
    if status == 429:
        return ModelFailureReason.RATE_LIMITED
    if status in (408, 504):
        return ModelFailureReason.PROVIDER_TIMEOUT
    if 400 <= status < 500:
        return ModelFailureReason.INVALID_PARAMETER
    if status >= 500:
        return ModelFailureReason.PROVIDER_UNAVAILABLE
    return ModelFailureReason.UNKNOWN_PROVIDER_FAILURE


def _normalize_openai_error(error: BaseException) -> ModelErrorKind:
    if isinstance(error, ModelProviderError):
        return error.kind
    if isinstance(error, (TimeoutError, httpx.TimeoutException)):
        return ModelErrorKind.TIMEOUT
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        if status in (401, 403):
            return ModelErrorKind.AUTHENTICATION
        if status == 429:
            return ModelErrorKind.RATE_LIMIT
        if status in (408, 504):
            return ModelErrorKind.TIMEOUT
        if 400 <= status < 500:
            return ModelErrorKind.INVALID_REQUEST
        if status >= 500:
            return ModelErrorKind.UNAVAILABLE
    if isinstance(error, httpx.TransportError):
        return ModelErrorKind.UNAVAILABLE
    return ModelErrorKind.UNKNOWN


def _encode_messages(
    message: Message,
    wire_names: Mapping[str, str] | None = None,
    *,
    model: ModelSpec | None = None,
) -> list[dict[str, object]]:
    if isinstance(message, ToolMessage) and message.results:
        encoded_results: list[dict[str, object]] = [
            {
                "role": "tool",
                "tool_call_id": result.call_id,
                "name": (wire_names or {}).get(result.name, result.name),
                "content": _encode_tool_result_content(result),
            }
            for result in message.results
        ]
        if message.content:
            prior = cast(str, encoded_results[-1]["content"])
            encoded_results[-1]["content"] = f"{prior}\n{message.content}"
        return encoded_results
    encoded: dict[str, object] = {"role": message.role, "content": message.content}
    if isinstance(message, AIMessage) and model is not None:
        continuation = message.continuation
        if (
            continuation is not None
            and continuation.provider == model.provider
            and continuation.protocol == model.protocol
            and model.provider == "deepseek"
        ):
            data = cast(FrozenJsonObject, continuation.data)
            if (
                set(data) != {"version", "reasoning_content"}
                or type(data["version"]) is not int
                or data["version"] != 1
                or not isinstance(data["reasoning_content"], str)
            ):
                raise ModelProviderError(
                    ModelErrorKind.INVALID_REQUEST,
                    "DeepSeek continuation has an invalid shape",
                )
            encoded["reasoning_content"] = data["reasoning_content"]
    if isinstance(message, AIMessage) and message.tool_calls:
        encoded["tool_calls"] = [
            {
                "id": call.call_id,
                "type": "function",
                "function": {
                    "name": (wire_names or {}).get(call.name, call.name),
                    "arguments": json.dumps(
                        freeze_json_object(call.arguments).to_dict(),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            }
            for call in message.tool_calls
        ]
    return [encoded]


def _encode_tool_result_content(result: ToolResult) -> str:
    content: dict[str, object] = {
        "status": result.status,
        "output": thaw_json(result.output),
        "error": result.error,
    }
    if result.error_kind is not None:
        content["error_kind"] = result.error_kind
    if result.error_code is not None:
        content["error_code"] = result.error_code
    return json.dumps(content, ensure_ascii=False, separators=(",", ":"))


def _decode_text_content(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise TypeError("message content must be text or text parts")
    parts: list[str] = []
    recognized = False
    for item in value:
        if isinstance(item, str):
            parts.append(item)
            recognized = True
            continue
        if not isinstance(item, Mapping):
            continue
        text = item.get("text")
        if isinstance(text, str):
            parts.append(text)
            recognized = True
    if value and not recognized:
        raise TypeError("message content parts contain no text")
    return "".join(parts)


def _decode_json(value: str) -> object:
    candidate = value.strip()
    fenced = _JSON_FENCE.fullmatch(candidate)
    if fenced is not None:
        candidate = fenced.group(1).strip()
    return json.loads(candidate)


def _decode_tool_arguments(value: object) -> Mapping[str, object]:
    if value is None:
        return {}
    if isinstance(value, str):
        value = _decode_json(value or "{}")
    if not isinstance(value, Mapping):
        raise TypeError("tool call arguments must be an object")
    return cast(Mapping[str, object], value)


def _synthetic_tool_call_id(
    *,
    model_key: str,
    provider_request_id: str | None,
    index: int,
    name: str,
    arguments: Mapping[str, object],
) -> str:
    seed = _wire_json(
        {
            "request": provider_request_id or model_key,
            "index": index,
            "name": name,
            "arguments": arguments,
        }
    )
    return f"call_{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:24]}"


def _decode_tool_calls(
    value: object,
    wire_names: Mapping[str, str] | None = None,
    *,
    model_key: str = "provider",
    provider_request_id: str | None = None,
) -> tuple[ToolCall, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TypeError("tool_calls must be a list")
    calls: list[ToolCall] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise TypeError("tool call must be an object")
        function = item.get("function")
        if function is None and (
            "name" in item or "arguments" in item
        ):
            function = item
        if not isinstance(function, dict):
            raise TypeError("tool call function must be an object")
        arguments = _decode_tool_arguments(function.get("arguments"))
        raw_name = function["name"]
        if not isinstance(raw_name, str) or not raw_name:
            raise TypeError("tool call name must be non-empty")
        call_id = item.get("id")
        if call_id is None or call_id == "":
            call_id = _synthetic_tool_call_id(
                model_key=model_key,
                provider_request_id=provider_request_id,
                index=index,
                name=raw_name,
                arguments=arguments,
            )
        if not isinstance(call_id, str):
            raise TypeError("tool call id must be a string")
        calls.append(
            ToolCall(
                call_id=call_id,
                name=(wire_names or {}).get(raw_name, raw_name),
                arguments=arguments,
            )
        )
    return tuple(calls)


@lru_cache(maxsize=512)
def _openai_tool_name(name: str) -> str:
    """Map a portable tool name to a deterministic OpenAI wire name."""

    if _OPENAI_TOOL_NAME.fullmatch(name):
        return name
    prefix = re.sub(r"[^A-Za-z0-9_-]", "_", name).strip("_") or "tool"
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
    return f"{prefix[:49]}__{digest}"


@lru_cache(maxsize=128)
def _schema_projection(schema: FrozenJsonObject) -> dict[str, object]:
    """Cache deployment-static schema projection, never request content."""

    return schema.to_dict()


@lru_cache(maxsize=128)
def _response_format_projection(
    schema: FrozenJsonObject, name: str
) -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": _schema_projection(schema),
        },
    }


@lru_cache(maxsize=512)
def _tool_projection(tool: ToolDefinition, wire_name: str) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": wire_name,
            "description": tool.description,
            "parameters": cast(FrozenJsonObject, tool.parameters).to_dict(),
        },
    }


def _original_tool_name(wire_name: str, tools: tuple[ToolDefinition, ...]) -> str:
    by_wire = {_openai_tool_name(tool.name): tool.name for tool in tools}
    return by_wire.get(wire_name, wire_name)
