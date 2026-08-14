"""Portable tool declarations, admission values, and task snapshots."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from pygent.core import Message
from pygent.core._tool_values import (
    IdempotencyPolicy,
    ToolCall,
    ToolDefinition,
    ToolResult,
    ToolResultStatus,
    ToolSideEffect,
    ToolTask,
    ToolTaskState,
    _non_empty,
)
from pygent.core.values import _MESSAGE_SUBCLASS_TOKEN

ToolLifecycle = Literal["sync", "detach"]


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
