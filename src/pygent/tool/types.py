"""Portable tool declarations, admission values, and task snapshots."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, cast

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

from pygent.core import (
    FrozenJsonObject,
    JsonObjectInput,
    JsonValue,
    Message,
    freeze_json,
    freeze_json_object,
    thaw_json,
)
from pygent.core.values import _MESSAGE_SUBCLASS_TOKEN


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


ToolLifecycle = Literal["sync", "detach"]
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
class ToolSpec:
    """Portable execution semantics without a handler, secret, or live resource."""

    tool_id: str
    version: str
    definition: ToolDefinition
    side_effect: ToolSideEffect = ToolSideEffect.PURE
    idempotency: IdempotencyPolicy = IdempotencyPolicy.INHERENT
    timeout: float | None = None
    resource_key: str | None = None
    sandbox_profile: str | None = None
    required_permissions: tuple[str, ...] = ()

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ToolSpec cannot be subclassed")

    def __post_init__(self) -> None:
        _non_empty(self.tool_id, "tool_id")
        _non_empty(self.version, "version")
        if type(self.definition) is not ToolDefinition:
            raise TypeError("definition must be a ToolDefinition")
        if not isinstance(self.side_effect, ToolSideEffect):
            object.__setattr__(self, "side_effect", ToolSideEffect(self.side_effect))
        if not isinstance(self.idempotency, IdempotencyPolicy):
            object.__setattr__(self, "idempotency", IdempotencyPolicy(self.idempotency))
        if self.timeout is not None and (
            isinstance(self.timeout, bool)
            or not isinstance(self.timeout, (int, float))
            or not math.isfinite(self.timeout)
            or self.timeout <= 0
        ):
            raise ValueError("tool timeout must be finite and greater than zero")
        for name in ("resource_key", "sandbox_profile"):
            value = getattr(self, name)
            if value is not None:
                _non_empty(value, name)
        permissions = tuple(self.required_permissions)
        if any(not isinstance(item, str) or not item for item in permissions):
            raise ValueError("required_permissions must contain non-empty strings")
        if len(permissions) != len(set(permissions)):
            raise ValueError("required_permissions contains duplicates")
        object.__setattr__(self, "required_permissions", permissions)


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


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolAuthorizationRequest(
    Message, _framework_token=_MESSAGE_SUBCLASS_TOKEN
):
    """Immutable facts passed to an application-owned authorization Module."""

    call: ToolCall
    spec: ToolSpec
    permissions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        Message.__post_init__(self)
        if type(self.call) is not ToolCall:
            raise TypeError("call must be a ToolCall")
        if type(self.spec) is not ToolSpec:
            raise TypeError("spec must be a ToolSpec")
        permissions = tuple(self.permissions)
        if any(not isinstance(item, str) or not item for item in permissions):
            raise ValueError("permissions must contain non-empty strings")
        if len(permissions) != len(set(permissions)):
            raise ValueError("permissions contains duplicates")
        object.__setattr__(self, "permissions", permissions)


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolAuthorizationDecision(
    Message, _framework_token=_MESSAGE_SUBCLASS_TOKEN
):
    """Application decision made before ToolTask admission."""

    call_id: str
    allowed: bool
    reason_code: str
    lifecycle: ToolLifecycle = "sync"

    def __post_init__(self) -> None:
        Message.__post_init__(self)
        _non_empty(self.call_id, "call_id")
        if not isinstance(self.allowed, bool):
            raise TypeError("allowed must be a bool")
        _non_empty(self.reason_code, "reason_code")
        if self.lifecycle not in ("sync", "detach"):
            raise ValueError("lifecycle must be 'sync' or 'detach'")


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
        object.__setattr__(self, "output", freeze_json(self.output))


__all__ = [
    "IdempotencyPolicy",
    "ToolAuthorizationDecision",
    "ToolAuthorizationRequest",
    "ToolCall",
    "ToolDefinition",
    "ToolLifecycle",
    "ToolResult",
    "ToolResultStatus",
    "ToolSideEffect",
    "ToolSpec",
    "ToolTask",
    "ToolTaskState",
]
