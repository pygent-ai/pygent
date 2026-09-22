"""Portable tool declarations, admission values, and task snapshots."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from pygent.core import JsonValue, Message
from pygent.core._tool_values import (
    IdempotencyPolicy,
    MediaBlock,
    MediaSource,
    ToolCall,
    ToolDefinition,
    ToolResult,
    ToolResultContent,
    ToolResultJson,
    ToolResultStatus,
    ToolResultText,
    ToolSideEffect,
    ToolTask,
    ToolTaskState,
    _non_empty,
)
from pygent.core.values import _MESSAGE_SUBCLASS_TOKEN

from ._waiting import resolve_wait_timeout

ToolLifecycle = Literal["sync", "detach"]


@dataclass(frozen=True, slots=True)
class ToolOutput:
    """Explicit business output and model-visible structured Tool content."""

    output: JsonValue = None
    content: tuple[ToolResultContent, ...] = ()

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ToolOutput cannot be subclassed")

    def __post_init__(self) -> None:
        from pygent.core import freeze_json

        content = tuple(self.content)
        if not content:
            raise ValueError("ToolOutput content must be non-empty")
        if any(
            type(value) not in (ToolResultText, ToolResultJson, MediaBlock)
            for value in content
        ):
            raise TypeError("ToolOutput content contains an unsupported value")
        object.__setattr__(self, "output", freeze_json(self.output))
        object.__setattr__(self, "content", content)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Portable execution semantics without a handler, secret, or live resource."""

    tool_id: str
    version: str
    definition: ToolDefinition
    side_effect: ToolSideEffect = ToolSideEffect.PURE
    idempotency: IdempotencyPolicy = IdempotencyPolicy.INHERENT
    timeout: float | None = None
    wait_timeout: float | None = None
    wait_timeout_parameter: str | None = None
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
        for name in ("resource_key", "sandbox_profile", "wait_timeout_parameter"):
            value = getattr(self, name)
            if value is not None:
                _non_empty(value, name)
        if self.wait_timeout_parameter is not None and self.wait_timeout is None:
            raise ValueError("wait_timeout_parameter requires a default wait_timeout")
        if self.wait_timeout is not None and (
            isinstance(self.wait_timeout, bool)
            or not isinstance(self.wait_timeout, (int, float))
            or not math.isfinite(self.wait_timeout)
            or self.wait_timeout < 0
        ):
            raise ValueError("tool wait_timeout must be finite and non-negative")
        permissions = tuple(self.required_permissions)
        if any(not isinstance(item, str) or not item for item in permissions):
            raise ValueError("required_permissions must contain non-empty strings")
        if len(permissions) != len(set(permissions)):
            raise ValueError("required_permissions contains duplicates")
        object.__setattr__(self, "required_permissions", permissions)

    def resolve_wait_timeout(self, arguments: Mapping[str, JsonValue]) -> float | None:
        override = (
            arguments.get(self.wait_timeout_parameter)
            if self.wait_timeout_parameter is not None
            else None
        )
        return resolve_wait_timeout(
            self.wait_timeout,
            override,
            is_background=arguments.get("is_background") is True,
        )


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
    "MediaBlock",
    "MediaSource",
    "ToolAuthorizationDecision",
    "ToolAuthorizationRequest",
    "ToolCall",
    "ToolDefinition",
    "ToolLifecycle",
    "ToolOutput",
    "ToolResult",
    "ToolResultContent",
    "ToolResultJson",
    "ToolResultStatus",
    "ToolResultText",
    "ToolSideEffect",
    "ToolSpec",
    "ToolTask",
    "ToolTaskState",
]
