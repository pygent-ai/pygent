"""Provider SPI, OpenAI-compatible transport, and retry/fallback execution."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, NoReturn, Protocol, Self, cast

import httpx
import jsonschema  # type: ignore[import-untyped]

from pygent.core import (
    AIMessage,
    Context,
    FrozenJsonObject,
    JsonObjectInput,
    Message,
    ToolMessage,
    freeze_json_object,
    thaw_json,
)
from pygent.tool import ToolCall, ToolDefinition

from .catalog import ModelCatalog, ModelInfo
from .types import (
    GenerationConfig,
    ModelAttempt,
    ModelCallError,
    ModelErrorKind,
    ModelGroupConfig,
    ModelProviderError,
    ModelRoute,
    RetryPolicy,
)

EventSink = Callable[[str, FrozenJsonObject], Awaitable[None]]

_CANCELLATION_CLEANUP_GRACE_SECONDS = 1.0
_CANCELLATION_CLEANUP_REASON = "cancellation_cleanup_timeout"


class ModelEventKind(str, Enum):
    """Closed public event vocabulary for one model execution."""

    STARTED = "model.started"
    ATTEMPT_STARTED = "model.attempt.started"
    REASONING_DELTA = "model.reasoning.delta"
    TEXT_DELTA = "model.text.delta"
    TOOL_CALL_STARTED = "model.tool_call.started"
    TOOL_CALL_DELTA = "model.tool_call.delta"
    TOOL_CALL_COMPLETED = "model.tool_call.completed"
    USAGE = "model.usage"
    ATTEMPT_SUCCEEDED = "model.attempt.succeeded"
    ATTEMPT_FAILED = "model.attempt.failed"
    COMPLETED = "model.completed"
    FAILED = "model.failed"
    CANCELLED = "model.cancelled"


class ModelProviderStreamKind(str, Enum):
    """Closed provider-neutral vocabulary before public event reduction."""

    REASONING = "reasoning"
    TEXT = "text"
    TOOL_CALL = "tool_call"
    USAGE = "usage"
    FINISH = "finish"


@dataclass(frozen=True, slots=True)
class ModelProviderRequest:
    """One provider-independent request before wire-format conversion."""

    route: ModelRoute
    message: Message
    context: Context
    generation: GenerationConfig
    tools: tuple[ToolDefinition, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelProviderResponse:
    """Normalized successful response returned by a provider adapter."""

    message: AIMessage
    usage: JsonObjectInput = ()
    provider_request_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "usage", _validated_canonical_usage(self.usage))


@dataclass(frozen=True, slots=True)
class ModelStreamEvent:
    """One validated public model event owned by ``ModelExecution``."""

    kind: ModelEventKind | str
    data: JsonObjectInput = ()

    def __post_init__(self) -> None:
        try:
            kind = ModelEventKind(self.kind)
        except ValueError as exc:
            raise ValueError(f"unsupported model event kind: {self.kind}") from exc
        data = freeze_json_object(self.data)
        _validate_public_model_event(kind, data)
        object.__setattr__(self, "kind", kind.value)
        object.__setattr__(self, "data", data)


@dataclass(frozen=True, slots=True)
class ModelProviderStreamPart:
    """One provider-neutral increment; one wire chunk may produce many parts."""

    kind: ModelProviderStreamKind | str
    data: JsonObjectInput = ()

    def __post_init__(self) -> None:
        try:
            kind = ModelProviderStreamKind(self.kind)
        except ValueError as exc:
            raise ValueError(f"unsupported provider stream part: {self.kind}") from exc
        object.__setattr__(self, "kind", kind.value)
        object.__setattr__(self, "data", freeze_json_object(self.data))


class ModelProviderClient(Protocol):
    """Transport client whose lifecycle may be owned by Runtime."""

    async def invoke(
        self, route: ModelRoute, payload: FrozenJsonObject
    ) -> FrozenJsonObject: ...

    def stream(
        self, route: ModelRoute, payload: FrozenJsonObject
    ) -> AsyncIterator[FrozenJsonObject]: ...

    async def aclose(self) -> None: ...


class ModelProviderAdapter(Protocol):
    """Provider wire conversion and error normalization boundary."""

    provider: str

    def build_request(self, request: ModelProviderRequest) -> FrozenJsonObject: ...

    def parse_response(
        self, request: ModelProviderRequest, payload: FrozenJsonObject
    ) -> ModelProviderResponse: ...

    def parse_stream_events(
        self, request: ModelProviderRequest, payload: FrozenJsonObject
    ) -> tuple[ModelProviderStreamPart, ...]: ...

    def normalize_error(self, error: BaseException) -> ModelErrorKind: ...


@dataclass(frozen=True, slots=True)
class ModelProviderCapabilities:
    """Transport capabilities declared by one provider deployment."""

    streaming: bool = True


class _ProviderStreamOwner:
    """Sole task owner of one provider iterator from open through close."""

    def __init__(
        self,
        client: ModelProviderClient,
        route: ModelRoute,
        payload: FrozenJsonObject,
    ) -> None:
        self._client = client
        self._route = route
        self._payload = payload
        self._items: asyncio.Queue[FrozenJsonObject] = asyncio.Queue(maxsize=1)
        self._stopping = False
        self._task = asyncio.create_task(self._run(), name="pygent-model-stream-owner")

    @property
    def task(self) -> asyncio.Task[None]:
        return self._task

    @property
    def done(self) -> bool:
        return self._task.done()

    def cancel(self) -> bool:
        self._stopping = True
        if self._task.done():
            return False
        self._task.cancel()
        return True

    async def next(self) -> FrozenJsonObject:
        if not self._items.empty():
            return self._items.get_nowait()
        if self._task.done():
            await self._task
            raise StopAsyncIteration
        item_task = asyncio.create_task(
            self._items.get(), name="pygent-model-stream-item"
        )
        try:
            done, _ = await asyncio.wait(
                {item_task, self._task}, return_when=asyncio.FIRST_COMPLETED
            )
            if item_task in done:
                return item_task.result()
            item_task.cancel()
            await asyncio.gather(item_task, return_exceptions=True)
            if not self._items.empty():
                return self._items.get_nowait()
            await self._task
            raise StopAsyncIteration
        finally:
            if not item_task.done():
                item_task.cancel()
                await asyncio.gather(item_task, return_exceptions=True)

    async def _run(self) -> None:
        iterator = self._client.stream(self._route, self._payload).__aiter__()
        try:
            async for item in iterator:
                if self._stopping:
                    return
                await self._items.put(item)
                if self._stopping:
                    return
        finally:
            close = getattr(iterator, "aclose", None)
            if callable(close):
                await close()


class _ModelExecutionSubscription:
    def __init__(self, execution: ModelExecution, after: int | None) -> None:
        self._execution = execution
        self._next = 0 if after is None else after + 1

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def __aiter__(self) -> AsyncIterator[ModelStreamEvent]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[ModelStreamEvent]:
        while True:
            async with self._execution._condition:
                while (
                    self._next >= len(self._execution._events)
                    and not self._execution._task.done()
                ):
                    await self._execution._condition.wait()
                available = tuple(self._execution._events[self._next :])
                terminal = self._execution._task.done()
            for event in available:
                self._next += 1
                yield event
            if terminal and self._next >= len(self._execution._events):
                return


class ModelExecution:
    """Single owner of one model operation, its events, result, and cancellation."""

    def __init__(
        self,
        operation: Callable[[EventSink], Awaitable[ModelProviderResponse]],
    ) -> None:
        self._events: list[ModelStreamEvent] = []
        self._condition = asyncio.Condition()

        async def run() -> ModelProviderResponse:
            return await operation(self._publish)

        self._task: asyncio.Task[ModelProviderResponse] = asyncio.create_task(
            run(), name="pygent-model-execution"
        )
        self._task.add_done_callback(lambda _: asyncio.create_task(self._notify()))

    async def _publish(self, kind: str, data: FrozenJsonObject) -> None:
        async with self._condition:
            self._events.append(ModelStreamEvent(kind, data))
            self._condition.notify_all()

    async def _notify(self) -> None:
        async with self._condition:
            self._condition.notify_all()

    async def result(self) -> ModelProviderResponse:
        return await self._task

    async def cancel(self) -> bool:
        if self._task.done():
            return False
        self._task.cancel()
        return True

    def subscribe(self, *, after: int | None = None) -> _ModelExecutionSubscription:
        if after is not None and (isinstance(after, bool) or after < -1):
            raise ValueError("after must be a non-negative sequence or -1")
        return _ModelExecutionSubscription(self, after)


class ModelInvoker(Protocol):
    """Single model execution boundary."""

    def execute(
        self,
        *,
        model_group: ModelGroupConfig,
        retry_policy: RetryPolicy,
        generation: GenerationConfig,
        message: Message,
        context: Context,
        tools: tuple[ToolDefinition, ...] = (),
        deadline: float | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> ModelExecution: ...


class OpenAICompatibleClient:
    """Small HTTP/SSE client for OpenAI, GLM, Qwen, and compatible endpoints."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        headers: Mapping[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url must be non-empty")
        request_headers = dict(headers or {})
        if api_key is not None:
            request_headers.setdefault("Authorization", f"Bearer {api_key}")
        api_root = base_url.rstrip("/")
        self._endpoint = f"{api_root}/chat/completions"
        self._models_endpoint = f"{api_root}/models"
        self._request_headers = request_headers
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=api_root, headers=request_headers, timeout=None
        )
        self._models: ModelCatalog = _OpenAICompatibleModelCatalog(self)
        self._closed = False

    @property
    def models(self) -> ModelCatalog:
        """Models visible to this endpoint and credential at query time."""

        return self._models

    async def invoke(
        self, route: ModelRoute, payload: FrozenJsonObject
    ) -> FrozenJsonObject:
        self._ensure_open()
        response = await self._client.post(
            self._endpoint,
            json=payload.to_dict(),
            headers=self._request_headers or None,
        )
        response.raise_for_status()
        try:
            body = response.json()
        except (TypeError, ValueError) as exc:
            raise ModelProviderError(
                ModelErrorKind.INVALID_RESPONSE, "provider returned invalid JSON"
            ) from exc
        if not isinstance(body, Mapping):
            raise ModelProviderError(
                ModelErrorKind.INVALID_RESPONSE, "provider response must be an object"
            )
        return freeze_json_object(cast(Mapping[str, object], body))

    async def stream(
        self, route: ModelRoute, payload: FrozenJsonObject
    ) -> AsyncIterator[FrozenJsonObject]:
        self._ensure_open()
        body = payload.to_dict()
        body["stream"] = True
        async with self._client.stream(
            "POST",
            self._endpoint,
            json=body,
            headers={**self._request_headers, "Accept": "text/event-stream"},
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                line = line.strip()
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    yield freeze_json_object({"done": True})
                    return
                try:
                    item = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise ModelProviderError(
                        ModelErrorKind.INVALID_RESPONSE,
                        "provider returned an invalid SSE event",
                    ) from exc
                if not isinstance(item, Mapping):
                    raise ModelProviderError(
                        ModelErrorKind.INVALID_RESPONSE,
                        "provider SSE event must be an object",
                    )
                yield freeze_json_object(cast(Mapping[str, object], item))

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> Self:
        self._ensure_open()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("model provider client is closed")

    async def _list_models_payload(
        self, *, timeout: float | None
    ) -> FrozenJsonObject:
        self._ensure_open()
        try:
            response = await self._client.get(
                self._models_endpoint,
                headers=self._request_headers or None,
                timeout=timeout,
            )
            response.raise_for_status()
            body = response.json()
        except asyncio.CancelledError:
            raise
        except (TypeError, ValueError) as exc:
            raise ModelProviderError(
                ModelErrorKind.INVALID_RESPONSE,
                "model catalog returned invalid JSON",
            ) from exc
        except Exception as exc:  # noqa: BLE001 - provider transport boundary
            kind = _normalize_openai_error(exc)
            raise ModelProviderError(kind, "model catalog request failed") from None
        if not isinstance(body, Mapping):
            raise ModelProviderError(
                ModelErrorKind.INVALID_RESPONSE,
                "model catalog response must be an object",
            )
        return freeze_json_object(cast(Mapping[str, object], body))


class _OpenAICompatibleModelCatalog:
    def __init__(self, client: OpenAICompatibleClient) -> None:
        self._client = client

    async def list(
        self, *, timeout: float | None = 10.0
    ) -> tuple[ModelInfo, ...]:
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

    def __init__(self, provider: str = "openai") -> None:
        if not provider:
            raise ValueError("provider must be non-empty")
        self.provider = provider

    def build_request(self, request: ModelProviderRequest) -> FrozenJsonObject:
        messages: list[dict[str, object]] = []
        wire_names = {tool.name: _openai_tool_name(tool.name) for tool in request.tools}
        if request.context.system_prompt:
            messages.append(
                {"role": "system", "content": request.context.system_prompt}
            )
        for item in request.context.messages:
            messages.extend(_encode_messages(item, wire_names))
        messages.extend(_encode_messages(request.message, wire_names))
        body: dict[str, object] = {
            "model": request.route.model,
            "messages": messages,
        }
        generation = request.generation
        if generation.temperature is not None:
            body["temperature"] = generation.temperature
        if generation.max_output_tokens is not None:
            body["max_tokens"] = generation.max_output_tokens
        if generation.response_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": generation.response_schema_name,
                    "strict": True,
                    "schema": freeze_json_object(generation.response_schema).to_dict(),
                },
            }
        if request.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": wire_names[tool.name],
                        "description": tool.description,
                        "parameters": freeze_json_object(tool.parameters).to_dict(),
                    },
                }
                for tool in request.tools
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
        return freeze_json_object(body)

    def parse_response(
        self, request: ModelProviderRequest, payload: FrozenJsonObject
    ) -> ModelProviderResponse:
        body = payload.to_dict()
        try:
            choices = body["choices"]
            if not isinstance(choices, list) or not choices:
                raise TypeError
            choice = choices[0]
            if not isinstance(choice, dict):
                raise TypeError
            raw_message = choice["message"]
            if not isinstance(raw_message, dict):
                raise TypeError
            content_value = raw_message.get("content")
            content = "" if content_value is None else content_value
            if not isinstance(content, str):
                raise TypeError
            tool_calls = _decode_tool_calls(
                raw_message.get("tool_calls", []),
                {_openai_tool_name(tool.name): tool.name for tool in request.tools},
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelProviderError(
                ModelErrorKind.INVALID_RESPONSE,
                "provider response has an invalid completion shape",
            ) from exc

        if request.generation.response_schema is not None:
            try:
                value = json.loads(content)
                jsonschema.validate(
                    value,
                    freeze_json_object(request.generation.response_schema).to_dict(),
                )
            except (json.JSONDecodeError, jsonschema.ValidationError) as exc:
                raise ModelProviderError(
                    ModelErrorKind.INVALID_RESPONSE,
                    "model output does not match the declared JSON schema",
                ) from exc

        request_id = body.get("id")
        if request_id is not None and not isinstance(request_id, str):
            request_id = None
        usage = _canonical_usage(body.get("usage"))
        message = AIMessage(
            content=content,
            tool_calls=tool_calls,
            metadata={"route_id": request.route.route_id},
        )
        return ModelProviderResponse(
            message=message,
            usage=usage,
            provider_request_id=request_id,
        )

    def parse_stream_events(
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
                if isinstance(usage, Mapping):
                    return tuple(parts)
                raise TypeError
            choice = choices[0]
            if not isinstance(choice, dict):
                raise TypeError
            delta = choice.get("delta", {})
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
            content = delta.get("content")
            if isinstance(content, str) and content:
                parts.append(ModelProviderStreamPart("text", {"text": content}))
            tool_calls = delta.get("tool_calls")
            if tool_calls is not None and not isinstance(tool_calls, list):
                raise TypeError
            for call in tool_calls or ():
                if not isinstance(call, dict):
                    raise TypeError
                function = call.get("function", {})
                if not isinstance(function, dict):
                    raise TypeError
                index = call.get("index", 0)
                call_id = call.get("id", "")
                name = function.get("name", "")
                arguments = function.get("arguments", "")
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
            ) from exc
        return tuple(parts)

    def normalize_error(self, error: BaseException) -> ModelErrorKind:
        return _normalize_openai_error(error)


class DefaultModelInvoker:
    """Bounded model executor with deterministic route/retry/fallback order."""

    def __init__(
        self,
        *,
        adapters: Mapping[str, ModelProviderAdapter],
        clients: Mapping[str, ModelProviderClient],
        capabilities: Mapping[str, ModelProviderCapabilities] | None = None,
    ) -> None:
        self._adapters = dict(adapters)
        self._clients = dict(clients)
        self._capabilities = dict(capabilities or {})
        self._quarantined_tasks: dict[int, set[asyncio.Future[Any]]] = {}
        self._active_executions: set[asyncio.Task[Any]] = set()
        self._stream_owner_tasks: set[asyncio.Task[None]] = set()
        self._close_task: asyncio.Task[None] | None = None
        self._closing = False

    def execute(
        self,
        *,
        model_group: ModelGroupConfig,
        retry_policy: RetryPolicy,
        generation: GenerationConfig,
        message: Message,
        context: Context,
        tools: tuple[ToolDefinition, ...] = (),
        deadline: float | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> ModelExecution:
        return ModelExecution(
            lambda event_sink: self._execute_registered(
                model_group=model_group,
                retry_policy=retry_policy,
                generation=generation,
                message=message,
                context=context,
                tools=tools,
                deadline=deadline,
                cancel_event=cancel_event,
                event_sink=event_sink,
            )
        )

    async def _execute_registered(self, **kwargs: Any) -> ModelProviderResponse:
        if self._closing:
            raise ModelCallError(
                "model invoker is closing", kind=ModelErrorKind.OUTCOME_UNKNOWN
            )
        task = asyncio.current_task()
        if task is None:
            return await self._execute_with_lifecycle(**kwargs)
        self._active_executions.add(task)
        try:
            return await self._execute_with_lifecycle(**kwargs)
        finally:
            self._active_executions.discard(task)

    async def _execute_with_lifecycle(
        self,
        *,
        model_group: ModelGroupConfig,
        retry_policy: RetryPolicy,
        generation: GenerationConfig,
        message: Message,
        context: Context,
        tools: tuple[ToolDefinition, ...],
        deadline: float | None,
        cancel_event: asyncio.Event | None,
        event_sink: EventSink,
    ) -> ModelProviderResponse:
        await _emit(
            event_sink,
            ModelEventKind.STARTED,
            {"model_group": model_group.name},
        )
        try:
            return await self._reduce_stream(
                model_group=model_group,
                retry_policy=retry_policy,
                generation=generation,
                message=message,
                context=context,
                tools=tools,
                deadline=deadline,
                cancel_event=cancel_event,
                event_sink=event_sink,
            )
        except asyncio.CancelledError:
            await _emit(event_sink, ModelEventKind.CANCELLED, {})
            raise
        except ModelCallError as exc:
            await _emit(
                event_sink,
                ModelEventKind.FAILED,
                {
                    "error_kind": exc.kind.value,
                    "partial_output": exc.partial_output,
                },
            )
            raise
        except Exception as exc:  # noqa: BLE001 - provider SPI boundary
            kind = (
                exc.kind
                if isinstance(exc, ModelProviderError)
                else ModelErrorKind.UNKNOWN
            )
            error = ModelCallError("model call failed", kind=kind)
            await _emit(
                event_sink,
                ModelEventKind.FAILED,
                {"error_kind": kind.value, "partial_output": False},
            )
            raise error from None

    async def aclose(self) -> None:
        """Strictly join execution and stream owners before closing clients."""

        self._closing = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(
                self._close_all(), name="pygent-model-invoker-close"
            )
        await asyncio.shield(self._close_task)

    async def _close_all(self) -> None:
        active = tuple(task for task in self._active_executions if not task.done())
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)
        while True:
            owners = tuple(task for task in self._stream_owner_tasks if not task.done())
            quarantined = tuple(
                task
                for tasks in self._quarantined_tasks.values()
                for task in tasks
                if not task.done()
            )
            pending = tuple(dict.fromkeys((*owners, *quarantined)))
            if not pending:
                break
            for pending_task in pending:
                pending_task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        clients = {id(client): client for client in self._clients.values()}
        await asyncio.gather(*(client.aclose() for client in clients.values()))

    def _quarantine(
        self, client: ModelProviderClient, task: asyncio.Future[Any]
    ) -> None:
        key = id(client)
        tasks = self._quarantined_tasks.setdefault(key, set())
        tasks.add(task)

        def release(completed: asyncio.Future[Any]) -> None:
            _consume_task_result(completed)
            current = self._quarantined_tasks.get(key)
            if current is None:
                return
            current.discard(completed)
            if not current:
                self._quarantined_tasks.pop(key, None)

        task.add_done_callback(release)

    def _ensure_client_available(self, client: ModelProviderClient) -> None:
        if self._closing:
            raise ModelProviderError(
                ModelErrorKind.OUTCOME_UNKNOWN,
                "model invoker is closing",
            )
        if self._quarantined_tasks.get(id(client)):
            raise ModelProviderError(
                ModelErrorKind.OUTCOME_UNKNOWN,
                "model provider client is awaiting cancellation cleanup",
            )

    def _open_stream_owner(
        self,
        client: ModelProviderClient,
        route: ModelRoute,
        payload: FrozenJsonObject,
    ) -> _ProviderStreamOwner:
        owner = _ProviderStreamOwner(client, route, payload)
        self._stream_owner_tasks.add(owner.task)

        def release(completed: asyncio.Task[None]) -> None:
            _consume_task_result(completed)
            self._stream_owner_tasks.discard(completed)

        owner.task.add_done_callback(release)
        return owner

    async def _stream_events(
        self,
        *,
        model_group: ModelGroupConfig,
        retry_policy: RetryPolicy,
        generation: GenerationConfig,
        message: Message,
        context: Context,
        tools: tuple[ToolDefinition, ...] = (),
        deadline: float | None = None,
        cancel_event: asyncio.Event | None = None,
        event_sink: EventSink | None = None,
    ) -> AsyncIterator[ModelProviderStreamPart]:
        _validate_deadline(deadline)
        routes = {route.route_id: route for route in model_group.routes}
        order = model_group.fallback.order or tuple(routes)
        attempts: list[ModelAttempt] = []
        last_kind = ModelErrorKind.UNKNOWN
        for route_id in order:
            route = routes[route_id]
            adapter, client = self._resolve(route)
            request = ModelProviderRequest(
                route=route,
                message=message,
                context=context,
                generation=generation,
                tools=tuple(tools),
            )
            payload = adapter.build_request(request)
            for number in range(1, retry_policy.max_attempts_per_route + 1):
                attempt_deadline = _earliest_deadline(
                    deadline, retry_policy.attempt_timeout_seconds
                )
                emitted = False
                completed = False
                attempt_usage = freeze_json_object()
                await _emit(
                    event_sink,
                    ModelEventKind.ATTEMPT_STARTED,
                    {"route_id": route_id, "attempt": number},
                )
                try:
                    self._ensure_client_available(client)
                    async for part in self._transport_events(
                        route=route,
                        adapter=adapter,
                        client=client,
                        request=request,
                        payload=payload,
                        deadline=attempt_deadline,
                        cancel_event=cancel_event,
                    ):
                        part_payload = cast(FrozenJsonObject, part.data).to_dict()
                        part_payload.update({"route_id": route_id, "attempt": number})
                        part = ModelProviderStreamPart(part.kind, part_payload)
                        if part.kind == ModelProviderStreamKind.USAGE:
                            raw_usage = cast(FrozenJsonObject, part.data).to_dict()
                            raw_usage.pop("route_id", None)
                            raw_usage.pop("attempt", None)
                            attempt_usage = _validated_canonical_usage(raw_usage)
                        emitted = emitted or part.kind != ModelProviderStreamKind.FINISH
                        completed = (
                            completed or part.kind == ModelProviderStreamKind.FINISH
                        )
                        yield part
                    if not completed:
                        raise ModelProviderError(
                            ModelErrorKind.INVALID_RESPONSE,
                            "model stream ended before a completion marker",
                        )
                    return
                except asyncio.CancelledError:
                    attempts.append(ModelAttempt(route_id, "cancelled", attempt=number))
                    raise
                except Exception as exc:  # noqa: BLE001 - provider SPI boundary
                    kind = adapter.normalize_error(exc)
                    last_kind = kind
                    attempts.append(
                        ModelAttempt(route_id, "failed", kind, attempt=number)
                    )
                    await _emit(
                        event_sink,
                        ModelEventKind.USAGE,
                        _usage_event_payload(
                            attempt_usage,
                            route_id=route_id,
                            attempt=number,
                            final=True,
                        ),
                    )
                    await _emit(
                        event_sink,
                        ModelEventKind.ATTEMPT_FAILED,
                        _attempt_failed_payload(
                            route_id=route_id,
                            attempt=number,
                            kind=kind,
                        ),
                    )
                    if kind is ModelErrorKind.OUTCOME_UNKNOWN:
                        raise ModelCallError(
                            "model provider outcome is unknown after cancellation",
                            kind=kind,
                            attempts=tuple(attempts),
                        ) from None
                    if emitted:
                        raise ModelCallError(
                            "model stream failed after output was emitted",
                            kind=kind,
                            attempts=tuple(attempts),
                            partial_output=True,
                        ) from None
                    if (
                        kind not in retry_policy.retry_on
                        or number >= retry_policy.max_attempts_per_route
                    ):
                        break
                    await _sleep_budget(
                        retry_policy.backoff.delay(number - 1),
                        deadline=deadline,
                        cancel_event=cancel_event,
                    )
        raise ModelCallError(
            "model stream failed after retry and fallback",
            kind=last_kind,
            attempts=tuple(attempts),
        )

    async def _transport_events(
        self,
        *,
        route: ModelRoute,
        adapter: ModelProviderAdapter,
        client: ModelProviderClient,
        request: ModelProviderRequest,
        payload: FrozenJsonObject,
        deadline: float | None,
        cancel_event: asyncio.Event | None,
    ) -> AsyncIterator[ModelProviderStreamPart]:
        capabilities = self._capabilities.get(
            route.route_id,
            self._capabilities.get(route.provider, ModelProviderCapabilities()),
        )
        if capabilities.streaming:
            owner = self._open_stream_owner(client, route, payload)
            try:
                while True:
                    try:
                        raw = await _await_stream_owner(
                            owner,
                            client=client,
                            on_cleanup_stuck=self._quarantine,
                            deadline=deadline,
                            cancel_event=cancel_event,
                        )
                    except StopAsyncIteration:
                        return
                    for part in adapter.parse_stream_events(request, raw):
                        yield part
            finally:
                if not owner.done and not self._is_quarantined(client, owner.task):
                    owner.cancel()
                    cleaned = await _await_cancellation_cleanup(owner.task)
                    if not cleaned:
                        self._quarantine(client, owner.task)
        else:
            raw = await _await_budget(
                client.invoke(route, payload),
                deadline=deadline,
                cancel_event=cancel_event,
                on_cleanup_stuck=lambda task: self._quarantine(client, task),
            )
            response = adapter.parse_response(request, raw)
            if response.message.content:
                yield ModelProviderStreamPart(
                    "text", {"text": response.message.content}
                )
            for index, call in enumerate(response.message.tool_calls):
                yield ModelProviderStreamPart(
                    "tool_call",
                    {
                        "index": index,
                        "call_id_delta": call.call_id,
                        "name_delta": call.name,
                        "arguments_delta": json.dumps(
                            freeze_json_object(call.arguments).to_dict(),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                )
            if response.usage:
                yield ModelProviderStreamPart("usage", response.usage)
            yield ModelProviderStreamPart(
                "finish",
                {
                    "finish_reason": (
                        "tool_calls" if response.message.tool_calls else "stop"
                    ),
                    "provider_request_id": response.provider_request_id,
                },
            )

    def _is_quarantined(
        self, client: ModelProviderClient, task: asyncio.Future[Any]
    ) -> bool:
        return task in self._quarantined_tasks.get(id(client), ())

    async def _reduce_stream(
        self,
        *,
        model_group: ModelGroupConfig,
        retry_policy: RetryPolicy,
        generation: GenerationConfig,
        message: Message,
        context: Context,
        tools: tuple[ToolDefinition, ...],
        deadline: float | None,
        cancel_event: asyncio.Event | None,
        event_sink: EventSink | None,
    ) -> ModelProviderResponse:
        text_parts: list[str] = []
        usage = freeze_json_object()
        calls: dict[int, dict[str, str]] = {}
        selected_route: str | None = None
        selected_attempt: int | None = None
        finish_reason = "other"
        provider_request_id: str | None = None
        async for part in self._stream_events(
            model_group=model_group,
            retry_policy=retry_policy,
            generation=generation,
            message=message,
            context=context,
            tools=tools,
            deadline=deadline,
            cancel_event=cancel_event,
            event_sink=event_sink,
        ):
            data = freeze_json_object(part.data)
            route_value = data.get("route_id")
            if isinstance(route_value, str):
                selected_route = route_value
            attempt_value = data.get("attempt")
            if isinstance(attempt_value, int) and not isinstance(attempt_value, bool):
                selected_attempt = attempt_value
            if part.kind == ModelProviderStreamKind.REASONING:
                await _emit(event_sink, ModelEventKind.REASONING_DELTA, data)
            elif part.kind == ModelProviderStreamKind.TEXT:
                value = data.get("text", "")
                if isinstance(value, str):
                    text_parts.append(value)
                await _emit(event_sink, ModelEventKind.TEXT_DELTA, data)
            elif part.kind == ModelProviderStreamKind.TOOL_CALL:
                index = data.get("index", 0)
                if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                    await _raise_invalid_model_response(
                        event_sink,
                        "model stream returned an invalid tool-call index",
                        route_id=selected_route,
                        attempt=selected_attempt,
                        usage=usage,
                    )
                if index not in calls:
                    calls[index] = {"call_id": "", "name": "", "arguments": ""}
                    await _emit(
                        event_sink,
                        ModelEventKind.TOOL_CALL_STARTED,
                        _tool_item_payload(data, index=index),
                    )
                current = calls[index]
                for source, target in (
                    ("call_id_delta", "call_id"),
                    ("name_delta", "name"),
                    ("arguments_delta", "arguments"),
                ):
                    value = data.get(source, "")
                    if isinstance(value, str):
                        current[target] += value
                await _emit(
                    event_sink,
                    ModelEventKind.TOOL_CALL_DELTA,
                    _tool_delta_payload(data, index=index),
                )
            elif part.kind == ModelProviderStreamKind.USAGE:
                usage_data = data.to_dict()
                usage_data.pop("route_id", None)
                usage_data.pop("attempt", None)
                usage = _validated_canonical_usage(usage_data)
            elif part.kind == ModelProviderStreamKind.FINISH:
                raw_reason = data.get("finish_reason")
                if isinstance(raw_reason, str):
                    normalized_reason = _normalized_finish_reason(raw_reason)
                    if normalized_reason != "other" or finish_reason == "other":
                        finish_reason = normalized_reason
                raw_request_id = data.get("provider_request_id")
                if isinstance(raw_request_id, str) and raw_request_id:
                    provider_request_id = raw_request_id

        content = "".join(text_parts)
        if generation.response_schema is not None:
            try:
                jsonschema.validate(
                    json.loads(content),
                    freeze_json_object(generation.response_schema).to_dict(),
                )
            except (json.JSONDecodeError, jsonschema.ValidationError):
                await _raise_invalid_model_response(
                    event_sink,
                    "model output does not match the declared JSON schema",
                    route_id=selected_route,
                    attempt=selected_attempt,
                    usage=usage,
                )
        tool_calls: list[ToolCall] = []
        for index in sorted(calls):
            call = calls[index]
            try:
                arguments = json.loads(call["arguments"] or "{}")
                if not isinstance(arguments, Mapping):
                    raise TypeError
                tool_calls.append(
                    ToolCall(
                        call_id=call["call_id"],
                        name=_original_tool_name(call["name"], tools),
                        arguments=cast(Mapping[str, object], arguments),
                    )
                )
                await _emit(
                    event_sink,
                    ModelEventKind.TOOL_CALL_COMPLETED,
                    {
                        "item_id": f"tool-{index}",
                        "index": index,
                        "call_id": call["call_id"],
                        "name": _original_tool_name(call["name"], tools),
                        "arguments": cast(Mapping[str, object], arguments),
                        "route_id": selected_route,
                        "attempt": selected_attempt,
                    },
                )
            except (json.JSONDecodeError, TypeError, ValueError):
                await _raise_invalid_model_response(
                    event_sink,
                    "model stream returned an invalid tool call",
                    route_id=selected_route,
                    attempt=selected_attempt,
                    usage=usage,
                )
        if selected_route is None or selected_attempt is None:
            raise ModelCallError(
                "model stream completed without attempt identity",
                kind=ModelErrorKind.INVALID_RESPONSE,
            )
        if finish_reason == "other":
            finish_reason = "tool_calls" if tool_calls else "stop"
        await _emit(
            event_sink,
            ModelEventKind.USAGE,
            _usage_event_payload(
                usage,
                route_id=selected_route,
                attempt=selected_attempt,
                final=True,
            ),
        )
        await _emit(
            event_sink,
            ModelEventKind.ATTEMPT_SUCCEEDED,
            {"route_id": selected_route, "attempt": selected_attempt},
        )
        await _emit(
            event_sink,
            ModelEventKind.COMPLETED,
            {
                "route_id": selected_route,
                "attempt": selected_attempt,
                "finish_reason": finish_reason,
                "provider_request_id": provider_request_id,
            },
        )
        metadata = {"route_id": selected_route}
        return ModelProviderResponse(
            message=AIMessage(
                content=content, tool_calls=tuple(tool_calls), metadata=metadata
            ),
            usage=usage,
            provider_request_id=provider_request_id,
        )

    def _resolve(
        self, route: ModelRoute
    ) -> tuple[ModelProviderAdapter, ModelProviderClient]:
        adapter = self._adapters.get(route.provider)
        client = self._clients.get(route.route_id, self._clients.get(route.provider))
        if adapter is None or client is None:
            raise ModelCallError(
                f"model route {route.route_id!r} has no local provider binding",
                kind=ModelErrorKind.INVALID_REQUEST,
            )
        return adapter, client


def openai_compatible_adapters() -> dict[str, OpenAICompatibleAdapter]:
    """Return codecs for the initially supported compatible provider names."""

    return {name: OpenAICompatibleAdapter(name) for name in ("openai", "glm", "qwen")}


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
    message: Message, wire_names: Mapping[str, str] | None = None
) -> list[dict[str, object]]:
    if isinstance(message, ToolMessage) and message.results:
        return [
            {
                "role": "tool",
                "tool_call_id": result.call_id,
                "name": (wire_names or {}).get(result.name, result.name),
                "content": json.dumps(
                    {
                        "status": result.status,
                        "output": thaw_json(result.output),
                        "error": result.error,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
            for result in message.results
        ]
    encoded: dict[str, object] = {"role": message.role, "content": message.content}
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


def _decode_tool_calls(
    value: object, wire_names: Mapping[str, str] | None = None
) -> tuple[ToolCall, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TypeError("tool_calls must be a list")
    calls: list[ToolCall] = []
    for item in value:
        if not isinstance(item, dict):
            raise TypeError("tool call must be an object")
        function = item.get("function")
        if not isinstance(function, dict):
            raise TypeError("tool call function must be an object")
        arguments = function.get("arguments", "{}")
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, Mapping):
            raise TypeError("tool call arguments must be an object")
        raw_name = cast(str, function["name"])
        calls.append(
            ToolCall(
                call_id=cast(str, item["id"]),
                name=(wire_names or {}).get(raw_name, raw_name),
                arguments=cast(Mapping[str, object], arguments),
            )
        )
    return tuple(calls)


_OPENAI_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _openai_tool_name(name: str) -> str:
    """Map a portable tool name to a deterministic OpenAI wire name."""

    if _OPENAI_TOOL_NAME.fullmatch(name):
        return name
    prefix = re.sub(r"[^A-Za-z0-9_-]", "_", name).strip("_") or "tool"
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
    return f"{prefix[:49]}__{digest}"


def _original_tool_name(wire_name: str, tools: tuple[ToolDefinition, ...]) -> str:
    by_wire = {_openai_tool_name(tool.name): tool.name for tool in tools}
    return by_wire.get(wire_name, wire_name)


_CANONICAL_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_input_tokens",
    "reasoning_tokens",
)


def _counter(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _canonical_usage(value: object) -> dict[str, int]:
    """Map compatible-provider usage onto the fixed public counter names."""

    if not isinstance(value, Mapping):
        return {}
    prompt_details = value.get("prompt_tokens_details")
    completion_details = value.get("completion_tokens_details")
    cached = value.get("cached_input_tokens", value.get("cached_tokens"))
    reasoning = value.get("reasoning_tokens")
    if isinstance(prompt_details, Mapping):
        cached = prompt_details.get("cached_tokens", cached)
    if isinstance(completion_details, Mapping):
        reasoning = completion_details.get("reasoning_tokens", reasoning)
    candidates = {
        "input_tokens": value.get("input_tokens", value.get("prompt_tokens")),
        "output_tokens": value.get("output_tokens", value.get("completion_tokens")),
        "total_tokens": value.get("total_tokens"),
        "cached_input_tokens": cached,
        "reasoning_tokens": reasoning,
    }
    return {
        key: counter
        for key, raw in candidates.items()
        if (counter := _counter(raw)) is not None
    }


def _validated_canonical_usage(raw_usage: JsonObjectInput) -> FrozenJsonObject:
    usage = freeze_json_object(raw_usage)
    unknown = set(usage) - set(_CANONICAL_USAGE_FIELDS)
    if unknown:
        raise ValueError(f"unsupported model usage counters: {sorted(unknown)}")
    for key, counter_value in usage.items():
        if _counter(counter_value) is None:
            raise ValueError(f"{key} must be a non-negative integer")
    return usage


def _usage_event_payload(
    usage: JsonObjectInput,
    *,
    route_id: str,
    attempt: int,
    final: bool,
) -> dict[str, object]:
    canonical = _validated_canonical_usage(usage)
    return {
        "route_id": route_id,
        "attempt": attempt,
        "mode": "cumulative",
        "final": final,
        "available": bool(canonical),
        **{key: canonical.get(key) for key in _CANONICAL_USAGE_FIELDS},
    }


def _normalized_finish_reason(value: str) -> str:
    return {
        "stop": "stop",
        "tool_calls": "tool_calls",
        "function_call": "tool_calls",
        "length": "length",
        "max_tokens": "length",
        "content_filter": "content_filter",
    }.get(value, "other")


def _tool_item_payload(data: FrozenJsonObject, *, index: int) -> dict[str, object]:
    return {
        "item_id": f"tool-{index}",
        "index": index,
        "route_id": data.get("route_id"),
        "attempt": data.get("attempt"),
    }


def _tool_delta_payload(data: FrozenJsonObject, *, index: int) -> dict[str, object]:
    return {
        **_tool_item_payload(data, index=index),
        "call_id_delta": data.get("call_id_delta", ""),
        "name_delta": data.get("name_delta", ""),
        "arguments_delta": data.get("arguments_delta", ""),
    }


def _attempt_failed_payload(
    *, route_id: str, attempt: int, kind: ModelErrorKind
) -> dict[str, object]:
    payload: dict[str, object] = {
        "route_id": route_id,
        "attempt": attempt,
        "error_kind": kind.value,
    }
    if kind is ModelErrorKind.OUTCOME_UNKNOWN:
        payload["reason"] = _CANCELLATION_CLEANUP_REASON
    return payload


def _validate_public_model_event(kind: ModelEventKind, data: FrozenJsonObject) -> None:
    common_attempt = {"route_id", "attempt"}
    required: dict[ModelEventKind, set[str]] = {
        ModelEventKind.STARTED: {"model_group"},
        ModelEventKind.ATTEMPT_STARTED: common_attempt,
        ModelEventKind.REASONING_DELTA: common_attempt | {"text"},
        ModelEventKind.TEXT_DELTA: common_attempt | {"text"},
        ModelEventKind.TOOL_CALL_STARTED: common_attempt | {"item_id", "index"},
        ModelEventKind.TOOL_CALL_DELTA: common_attempt
        | {
            "item_id",
            "index",
            "call_id_delta",
            "name_delta",
            "arguments_delta",
        },
        ModelEventKind.TOOL_CALL_COMPLETED: common_attempt
        | {"item_id", "index", "call_id", "name", "arguments"},
        ModelEventKind.USAGE: common_attempt
        | {
            "mode",
            "final",
            "available",
            *_CANONICAL_USAGE_FIELDS,
        },
        ModelEventKind.ATTEMPT_SUCCEEDED: common_attempt,
        ModelEventKind.ATTEMPT_FAILED: common_attempt | {"error_kind"},
        ModelEventKind.COMPLETED: common_attempt
        | {"finish_reason", "provider_request_id"},
        ModelEventKind.FAILED: {"error_kind", "partial_output"},
        ModelEventKind.CANCELLED: set(),
    }
    expected = required[kind]
    if (
        kind is ModelEventKind.ATTEMPT_FAILED
        and data.get("error_kind") == ModelErrorKind.OUTCOME_UNKNOWN.value
    ):
        expected = expected | {"reason"}
    if set(data) != expected:
        raise ValueError(f"{kind.value} data fields must be exactly {sorted(expected)}")
    if kind is ModelEventKind.STARTED:
        _require_string(data, "model_group")
        return
    if kind in (ModelEventKind.FAILED, ModelEventKind.CANCELLED):
        if kind is ModelEventKind.FAILED:
            _require_string(data, "error_kind")
            if not isinstance(data["partial_output"], bool):
                raise ValueError("partial_output must be a bool")
        return
    _require_string(data, "route_id")
    attempt = data["attempt"]
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
        raise ValueError("attempt must be a positive integer")
    if kind in (ModelEventKind.REASONING_DELTA, ModelEventKind.TEXT_DELTA):
        _require_string(data, "text", allow_empty=False)
    elif kind in (
        ModelEventKind.TOOL_CALL_STARTED,
        ModelEventKind.TOOL_CALL_DELTA,
        ModelEventKind.TOOL_CALL_COMPLETED,
    ):
        _require_string(data, "item_id")
        index = data["index"]
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError("tool-call index must be a non-negative integer")
        if kind is ModelEventKind.TOOL_CALL_DELTA:
            for key in ("call_id_delta", "name_delta", "arguments_delta"):
                _require_string(data, key, allow_empty=True)
        elif kind is ModelEventKind.TOOL_CALL_COMPLETED:
            _require_string(data, "call_id")
            _require_string(data, "name")
            if not isinstance(data["arguments"], Mapping):
                raise ValueError("tool-call arguments must be an object")
    elif kind is ModelEventKind.USAGE:
        if data["mode"] != "cumulative":
            raise ValueError("usage mode must be cumulative")
        if not isinstance(data["final"], bool) or not isinstance(
            data["available"], bool
        ):
            raise ValueError("usage final and available must be bools")
        for key in _CANONICAL_USAGE_FIELDS:
            value = data[key]
            if value is not None and _counter(value) is None:
                raise ValueError(f"{key} must be null or a non-negative integer")
    elif kind is ModelEventKind.ATTEMPT_FAILED:
        _require_string(data, "error_kind")
        if (
            data["error_kind"] == ModelErrorKind.OUTCOME_UNKNOWN.value
            and data["reason"] != _CANCELLATION_CLEANUP_REASON
        ):
            raise ValueError("outcome-unknown reason is invalid")
    elif kind is ModelEventKind.COMPLETED:
        if data["finish_reason"] not in {
            "stop",
            "tool_calls",
            "length",
            "content_filter",
            "other",
        }:
            raise ValueError("unsupported model finish reason")
        request_id = data["provider_request_id"]
        if request_id is not None and (
            not isinstance(request_id, str) or not request_id
        ):
            raise ValueError("provider_request_id must be null or non-empty")


def _require_string(
    data: FrozenJsonObject, key: str, *, allow_empty: bool = False
) -> None:
    value = data[key]
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"{key} must be a string{' or empty' if allow_empty else ''}")


async def _raise_invalid_model_response(
    sink: EventSink | None,
    message: str,
    *,
    route_id: str | None,
    attempt: int | None,
    usage: JsonObjectInput,
) -> NoReturn:
    attempts: tuple[ModelAttempt, ...] = ()
    if route_id is not None and attempt is not None:
        await _emit(
            sink,
            ModelEventKind.USAGE,
            _usage_event_payload(usage, route_id=route_id, attempt=attempt, final=True),
        )
        await _emit(
            sink,
            ModelEventKind.ATTEMPT_FAILED,
            {
                "route_id": route_id,
                "attempt": attempt,
                "error_kind": ModelErrorKind.INVALID_RESPONSE.value,
            },
        )
        attempts = (
            ModelAttempt(
                route_id,
                "failed",
                ModelErrorKind.INVALID_RESPONSE,
                attempt=attempt,
            ),
        )
    raise ModelCallError(
        message,
        kind=ModelErrorKind.INVALID_RESPONSE,
        attempts=attempts,
        partial_output=route_id is not None and attempt is not None,
    )


async def _emit(
    sink: EventSink | None,
    kind: ModelEventKind | str,
    data: JsonObjectInput = (),
) -> None:
    if sink is not None:
        await sink(
            kind.value if isinstance(kind, ModelEventKind) else kind,
            freeze_json_object(data),
        )


def _earliest_deadline(
    request_deadline: float | None, attempt_timeout_seconds: float | None
) -> float | None:
    if attempt_timeout_seconds is None:
        return request_deadline
    attempt_deadline = time.monotonic() + attempt_timeout_seconds
    if request_deadline is None:
        return attempt_deadline
    return min(request_deadline, attempt_deadline)


def _validate_deadline(deadline: float | None) -> None:
    if deadline is not None and (
        not isinstance(deadline, (int, float))
        or not deadline > 0
        or deadline == float("inf")
    ):
        raise ValueError("deadline must be a finite absolute monotonic time")


def _check_budget(deadline: float | None, cancel_event: asyncio.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise asyncio.CancelledError
    if deadline is not None and time.monotonic() >= deadline:
        raise ModelProviderError(ModelErrorKind.TIMEOUT, "model deadline exceeded")


async def _await_budget(
    awaitable: Awaitable[Any],
    *,
    deadline: float | None,
    cancel_event: asyncio.Event | None,
    on_cleanup_stuck: Callable[[asyncio.Future[Any]], None] | None = None,
) -> Any:
    try:
        _check_budget(deadline, cancel_event)
    except BaseException:
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()
        raise
    task = asyncio.ensure_future(awaitable)
    cancel_task = (
        asyncio.create_task(cancel_event.wait()) if cancel_event is not None else None
    )
    timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
    waiters = {task}
    if cancel_task is not None:
        waiters.add(cancel_task)
    try:
        done, _ = await asyncio.wait(
            waiters, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
        )
        if task in done:
            return await task
        task.cancel()
        cleaned = await _await_cancellation_cleanup(task)
        if cancel_task is not None and cancel_task in done:
            if not cleaned and on_cleanup_stuck is not None:
                on_cleanup_stuck(task)
            raise asyncio.CancelledError
        if not cleaned:
            if on_cleanup_stuck is not None:
                on_cleanup_stuck(task)
            raise ModelProviderError(
                ModelErrorKind.OUTCOME_UNKNOWN,
                "model provider outcome is unknown after cancellation",
            )
        raise ModelProviderError(ModelErrorKind.TIMEOUT, "model deadline exceeded")
    except asyncio.CancelledError:
        task.cancel()
        cleaned = await _await_cancellation_cleanup(task)
        if not cleaned and on_cleanup_stuck is not None:
            on_cleanup_stuck(task)
        raise
    finally:
        if cancel_task is not None:
            cancel_task.cancel()
            await asyncio.gather(cancel_task, return_exceptions=True)


async def _await_stream_owner(
    owner: _ProviderStreamOwner,
    *,
    client: ModelProviderClient,
    on_cleanup_stuck: Callable[[ModelProviderClient, asyncio.Future[Any]], None],
    deadline: float | None,
    cancel_event: asyncio.Event | None,
) -> FrozenJsonObject:
    try:
        _check_budget(deadline, cancel_event)
    except asyncio.CancelledError:
        owner.cancel()
        cleaned = await _await_cancellation_cleanup(owner.task)
        if not cleaned:
            on_cleanup_stuck(client, owner.task)
        raise
    except ModelProviderError:
        owner.cancel()
        cleaned = await _await_cancellation_cleanup(owner.task)
        if not cleaned:
            on_cleanup_stuck(client, owner.task)
            raise ModelProviderError(
                ModelErrorKind.OUTCOME_UNKNOWN,
                "model provider outcome is unknown after cancellation",
            ) from None
        raise
    next_task = asyncio.create_task(owner.next(), name="pygent-model-stream-next")
    cancel_task = (
        asyncio.create_task(cancel_event.wait()) if cancel_event is not None else None
    )
    waiters: set[asyncio.Future[Any]] = {next_task}
    if cancel_task is not None:
        waiters.add(cancel_task)
    timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
    try:
        done, _ = await asyncio.wait(
            waiters, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
        )
        if next_task in done:
            return await next_task
        owner.cancel()
        cleaned = await _await_cancellation_cleanup(owner.task)
        if not cleaned:
            on_cleanup_stuck(client, owner.task)
        if cancel_task is not None and cancel_task in done:
            raise asyncio.CancelledError
        if not cleaned:
            raise ModelProviderError(
                ModelErrorKind.OUTCOME_UNKNOWN,
                "model provider outcome is unknown after cancellation",
            )
        raise ModelProviderError(ModelErrorKind.TIMEOUT, "model deadline exceeded")
    except asyncio.CancelledError:
        owner.cancel()
        cleaned = await _await_cancellation_cleanup(owner.task)
        if not cleaned:
            on_cleanup_stuck(client, owner.task)
        raise
    finally:
        if not next_task.done():
            next_task.cancel()
        if cancel_task is not None and not cancel_task.done():
            cancel_task.cancel()
        await asyncio.gather(
            next_task,
            *((cancel_task,) if cancel_task is not None else ()),
            return_exceptions=True,
        )


async def _await_cancellation_cleanup(task: asyncio.Future[Any]) -> bool:
    """Give cancellation its own bounded acknowledgement window.

    The operation deadline has already expired when this helper runs. Reusing
    it would collapse cooperative cleanup to a zero-length poll and make the
    error classification depend on event-loop scheduling.
    """

    grace = _CANCELLATION_CLEANUP_GRACE_SECONDS
    if task.done():
        _consume_task_result(task)
        return True
    if grace > 0:
        done, _ = await asyncio.wait({task}, timeout=grace)
        if task in done:
            _consume_task_result(task)
            return True
    task.add_done_callback(_consume_task_result)
    return False


def _consume_task_result(task: asyncio.Future[Any]) -> None:
    if not task.done():
        return
    try:
        task.exception()
    except asyncio.CancelledError:
        return


async def _sleep_budget(
    delay: float,
    *,
    deadline: float | None,
    cancel_event: asyncio.Event | None,
) -> None:
    if delay <= 0:
        _check_budget(deadline, cancel_event)
        return
    await _await_budget(
        asyncio.sleep(delay), deadline=deadline, cancel_event=cancel_event
    )


__all__ = [
    "DefaultModelInvoker",
    "EventSink",
    "ModelEventKind",
    "ModelExecution",
    "ModelInvoker",
    "ModelProviderAdapter",
    "ModelProviderCapabilities",
    "ModelProviderClient",
    "ModelProviderRequest",
    "ModelProviderResponse",
    "ModelProviderStreamKind",
    "ModelProviderStreamPart",
    "ModelStreamEvent",
    "OpenAICompatibleAdapter",
    "OpenAICompatibleClient",
    "openai_compatible_adapters",
]
