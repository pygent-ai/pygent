"""Portable tool values shared by Core messages and the Tool domain."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, cast

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

from .json_values import (
    FrozenJsonObject,
    JsonObjectInput,
    JsonValue,
    freeze_json,
    freeze_json_object,
    thaw_json,
)


class ToolSideEffect(str, Enum):
    PURE = "pure"
    READ = "read"
    WRITE = "write"
    EXTERNAL = "external"


class IdempotencyPolicy(str, Enum):
    INHERENT = "inherent"
    REQUIRES_KEY = "requires_key"
    NOT_IDEMPOTENT = "not_idempotent"


class ToolTaskState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


ToolResultStatus = Literal[
    "succeeded", "rejected", "failed", "cancelled", "unknown", "detached"
]


def _non_empty(value: object, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must be non-empty")


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """The model-visible name and JSON Schema of a tool."""

    name: str
    description: str
    parameters: JsonObjectInput
    output_schema: JsonObjectInput | None = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ToolDefinition cannot be subclassed")

    def __post_init__(self) -> None:
        _non_empty(self.name, "tool definition name")
        if not isinstance(self.description, str):
            raise TypeError("tool definition description must be a string")
        object.__setattr__(self, "parameters", freeze_json_object(self.parameters))
        Draft202012Validator.check_schema(
            cast(dict[str, Any], thaw_json(cast(FrozenJsonObject, self.parameters)))
        )
        if self.output_schema is not None:
            object.__setattr__(
                self, "output_schema", freeze_json_object(self.output_schema)
            )
            Draft202012Validator.check_schema(
                cast(
                    dict[str, Any],
                    thaw_json(cast(FrozenJsonObject, self.output_schema)),
                )
            )


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One model-requested invocation; this is not an admitted task."""

    call_id: str
    name: str
    arguments: JsonObjectInput
    tool_id: str | None = None
    tool_version: str | None = None
    idempotency_key: str | None = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ToolCall cannot be subclassed")

    def __post_init__(self) -> None:
        _non_empty(self.call_id, "tool call_id")
        _non_empty(self.name, "tool name")
        for field_name in ("tool_id", "tool_version", "idempotency_key"):
            value = getattr(self, field_name)
            if value is not None:
                _non_empty(value, field_name)
        object.__setattr__(self, "arguments", freeze_json_object(self.arguments))


@dataclass(frozen=True, slots=True)
class ToolTask:
    """Immutable public snapshot of one admitted tool execution."""

    task_id: str
    call_id: str
    tool_id: str
    version: str
    state: ToolTaskState
    job_id: str | None = None
    metadata: JsonObjectInput = ()

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ToolTask cannot be subclassed")

    def __post_init__(self) -> None:
        for name in ("task_id", "call_id", "tool_id", "version"):
            _non_empty(getattr(self, name), name)
        if not isinstance(self.state, ToolTaskState):
            object.__setattr__(self, "state", ToolTaskState(self.state))
        if self.job_id is not None:
            _non_empty(self.job_id, "job_id")
        object.__setattr__(self, "metadata", freeze_json_object(self.metadata))


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Terminal result, admission rejection, or detached-task acknowledgment."""

    call_id: str
    name: str
    status: ToolResultStatus
    task: ToolTask | None = None
    output: JsonValue = None
    error: str | None = None
    error_kind: str | None = None
    error_code: str | None = None
    retryable: bool = False
    side_effect_committed: bool | None = None
    tool_id: str | None = None
    tool_version: str | None = None
    missing_capabilities: tuple[str, ...] = ()

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ToolResult cannot be subclassed")

    def __post_init__(self) -> None:
        _non_empty(self.call_id, "tool result call_id")
        _non_empty(self.name, "tool result name")
        if self.status not in (
            "succeeded",
            "rejected",
            "failed",
            "cancelled",
            "unknown",
            "detached",
        ):
            raise ValueError(f"unsupported tool result status: {self.status!r}")
        if self.task is not None and type(self.task) is not ToolTask:
            raise TypeError("task must be a ToolTask or None")
        if self.status == "detached" and self.task is None:
            raise ValueError("a detached result must include a ToolTask snapshot")
        if self.status == "rejected" and self.task is not None:
            raise ValueError("an authorization rejection cannot include a ToolTask")
        for name in ("error", "error_kind", "error_code", "tool_id", "tool_version"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be a string or None")
        if not isinstance(self.retryable, bool):
            raise TypeError("retryable must be a bool")
        if self.side_effect_committed is not None and not isinstance(
            self.side_effect_committed, bool
        ):
            raise TypeError("side_effect_committed must be a bool or None")
        capabilities = tuple(self.missing_capabilities)
        if any(not isinstance(value, str) or not value for value in capabilities):
            raise ValueError("missing_capabilities must contain non-empty strings")
        if len(capabilities) != len(set(capabilities)):
            raise ValueError("missing_capabilities must be unique")
        object.__setattr__(self, "missing_capabilities", capabilities)
        object.__setattr__(self, "output", freeze_json(self.output))


__all__ = [
    "IdempotencyPolicy",
    "ToolCall",
    "ToolDefinition",
    "ToolResult",
    "ToolResultStatus",
    "ToolSideEffect",
    "ToolTask",
    "ToolTaskState",
]
