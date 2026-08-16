"""Provider extension contracts and canonical public model events."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, NoReturn, Protocol, runtime_checkable

from pygent.core import (
    AIMessage,
    Context,
    FrozenJsonObject,
    JsonObjectInput,
    Message,
    freeze_json_object,
)
from pygent.tool import ToolDefinition

from .types import (
    GenerationConfig,
    ModelAttempt,
    ModelCallError,
    ModelErrorKind,
    ModelGroupConfig,
    ModelRoute,
    RetryPolicy,
)

EventSink = Callable[[str, FrozenJsonObject], Awaitable[None]]

if TYPE_CHECKING:
    from ._model_execution import ModelExecution

_CANCELLATION_CLEANUP_REASON = "cancellation_cleanup_timeout"

_CANONICAL_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_input_tokens",
    "reasoning_tokens",
)

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
        data = (
            self.data
            if isinstance(self.data, FrozenJsonObject)
            else freeze_json_object(self.data)
        )
        _validate_public_model_event(kind, data)
        object.__setattr__(self, "kind", kind.value)
        object.__setattr__(self, "data", data)


def _trusted_model_stream_event(
    kind: str, data: FrozenJsonObject
) -> ModelStreamEvent:
    """Construct one event already normalized by the built-in invoker."""

    event = object.__new__(ModelStreamEvent)
    object.__setattr__(event, "kind", kind)
    object.__setattr__(event, "data", data)
    return event


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
        if not isinstance(self.data, FrozenJsonObject):
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


@runtime_checkable
class ModelProviderRouteValidator(Protocol):
    """Optional provider-owned validation for non-empty route options."""

    def validate_route(self, route: ModelRoute) -> None: ...


@dataclass(frozen=True, slots=True)
class ModelProviderCapabilities:
    """Transport capabilities declared by one provider deployment."""

    streaming: bool = True


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
