"""Execution identity, lifecycle, event, and effect outcome contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar, cast

from .json_values import (
    FrozenJsonObject,
    JsonObjectInput,
    JsonValue,
    freeze_json,
    freeze_json_object,
    thaw_json,
)


@dataclass(frozen=True, slots=True)
class ExecutionFailure:
    """Portable, sanitized failure information across execution boundaries."""

    domain: str
    kind: str
    message: str
    retryable: bool = False
    outcome_unknown: bool = False
    partial_output: bool = False
    details: JsonObjectInput | None = None

    def __post_init__(self) -> None:
        for name in ("domain", "kind", "message"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("retryable", "outcome_unknown", "partial_output"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        details = {} if self.details is None else self.details
        object.__setattr__(self, "details", freeze_json_object(details))

    def to_dict(self) -> dict[str, object]:
        return {
            "domain": self.domain,
            "kind": self.kind,
            "message": self.message,
            "retryable": self.retryable,
            "outcome_unknown": self.outcome_unknown,
            "partial_output": self.partial_output,
            "details": thaw_json(cast(JsonValue, self.details)),
        }

    @classmethod
    def from_dict(cls, value: object) -> ExecutionFailure:
        if not isinstance(value, Mapping):
            raise TypeError("execution failure must be an object")
        allowed = {
            "domain",
            "kind",
            "message",
            "retryable",
            "outcome_unknown",
            "partial_output",
            "details",
        }
        if set(value) != allowed:
            raise ValueError("execution failure fields are invalid")
        details = value["details"]
        if not isinstance(details, Mapping):
            raise TypeError("execution failure details must be an object")
        return cls(
            domain=cast(str, value["domain"]),
            kind=cast(str, value["kind"]),
            message=cast(str, value["message"]),
            retryable=cast(bool, value["retryable"]),
            outcome_unknown=cast(bool, value["outcome_unknown"]),
            partial_output=cast(bool, value["partial_output"]),
            details=cast(Mapping[str, Any], details),
        )


class ExecutionFailureError(RuntimeError):
    """Exception carrying a portable failure without exposing local causes."""

    def __init__(self, failure: ExecutionFailure) -> None:
        if not isinstance(failure, ExecutionFailure):
            raise TypeError("failure must be an ExecutionFailure")
        super().__init__(failure.message)
        self.failure = failure


class ExecutionStatus(str, Enum):
    """Stable externally observable state of one root execution."""

    QUEUED = "queued"
    RUNNING = "running"
    WAITING_CHILD = "waiting_child"
    WAITING_RESUME = "waiting_resume"
    WAITING_EXTERNAL = "waiting_external"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DEADLINE_EXCEEDED = "deadline_exceeded"

    @property
    def terminal(self) -> bool:
        return self in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
            ExecutionStatus.DEADLINE_EXCEEDED,
        }


@dataclass(frozen=True, slots=True)
class ExecutionOptions:
    """Per-execution transport, trace, and lifecycle options."""

    request_id: str | None = None
    idempotency_key: str | None = None
    identity: str | None = None
    context_ref: str | None = None
    deadline: float | None = None
    execution_id: str | None = None
    trace_id: str | None = None
    parent_execution_id: str | None = None
    parent_span_id: str | None = None
    model_calls: FrozenJsonObject | Mapping[str, object] = field(
        default_factory=FrozenJsonObject
    )

    def __post_init__(self) -> None:
        for name in (
            "request_id",
            "idempotency_key",
            "identity",
            "context_ref",
            "execution_id",
            "trace_id",
            "parent_execution_id",
            "parent_span_id",
        ):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be non-empty when provided")
        if self.deadline is not None and (
            isinstance(self.deadline, bool)
            or not isinstance(self.deadline, (int, float))
            or not math.isfinite(self.deadline)
            or self.deadline <= 0
        ):
            raise ValueError("deadline must be finite and greater than zero")
        raw_calls = self.model_calls
        if not isinstance(raw_calls, Mapping):
            raise TypeError("model_calls must be a mapping")
        prepared: dict[str, object] = {}
        for group_name, raw in raw_calls.items():
            if not isinstance(group_name, str) or not group_name:
                raise ValueError("model_calls keys must be non-empty strings")
            to_dict = getattr(raw, "to_dict", None)
            value = to_dict() if callable(to_dict) else raw
            if not isinstance(value, Mapping):
                raise TypeError("model_calls values must be ModelCallOptions or mappings")
            prepared[group_name] = dict(value)
        object.__setattr__(self, "model_calls", freeze_json_object(prepared))


@dataclass(frozen=True, slots=True)
class ExecutionEvent:
    """One portable event in an observable execution journal."""

    schema_version: str
    event_id: str
    execution_id: str
    trace_id: str
    span_id: str
    parent_span_id: str | None
    sequence: int
    timestamp_unix_ns: int
    module_path: str
    kind: str
    data: JsonObjectInput

    def __post_init__(self) -> None:
        for name in (
            "schema_version",
            "event_id",
            "execution_id",
            "trace_id",
            "span_id",
            "module_path",
            "kind",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.parent_span_id is not None and (
            not isinstance(self.parent_span_id, str) or not self.parent_span_id
        ):
            raise ValueError("parent_span_id must be non-empty when provided")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 0:
            raise ValueError("sequence must be a non-negative integer")
        if (
            isinstance(self.timestamp_unix_ns, bool)
            or not isinstance(self.timestamp_unix_ns, int)
            or self.timestamp_unix_ns <= 0
        ):
            raise ValueError("timestamp_unix_ns must be a positive integer")
        object.__setattr__(self, "data", freeze_json_object(self.data))


class EffectDisposition(str, Enum):
    EXECUTED = "executed"
    REPLAYED = "replayed"
    RETRIED = "retried"


EffectValueT = TypeVar("EffectValueT", bound=JsonValue)


@dataclass(frozen=True, slots=True)
class EffectOutcome(Generic[EffectValueT]):
    """Value plus the recovery disposition of one managed effect occurrence."""

    value: EffectValueT
    disposition: EffectDisposition
    effect_id: str
    attempt: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", freeze_json(self.value))
        if not isinstance(self.disposition, EffectDisposition):
            object.__setattr__(self, "disposition", EffectDisposition(self.disposition))
        if not isinstance(self.effect_id, str) or not self.effect_id:
            raise ValueError("effect_id must be a non-empty string")
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int) or self.attempt < 1:
            raise ValueError("attempt must be a positive integer")


EXECUTION_EVENT_SCHEMA_VERSION = "0.2"


__all__ = [
    "EXECUTION_EVENT_SCHEMA_VERSION",
    "EffectDisposition",
    "EffectOutcome",
    "ExecutionEvent",
    "ExecutionFailure",
    "ExecutionFailureError",
    "ExecutionOptions",
    "ExecutionStatus",
]
