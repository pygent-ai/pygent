"""Immutable, provider-neutral declarations for model calls."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Protocol, cast, runtime_checkable

from pygent.core import (
    ExecutionFailure,
    ExecutionFailureError,
    FrozenJsonObject,
    JsonObjectInput,
    JsonValueError,
    freeze_json_object,
)


class ModelErrorKind(str, Enum):
    TIMEOUT = "timeout"
    OUTCOME_UNKNOWN = "outcome_unknown"
    RATE_LIMIT = "rate_limit"
    UNAVAILABLE = "unavailable"
    INCOMPLETE_RESPONSE = "incomplete_response"
    AUTHENTICATION = "authentication"
    INVALID_REQUEST = "invalid_request"
    INVALID_RESPONSE = "invalid_response"
    UNKNOWN = "unknown"


class ModelFailureReason(str, Enum):
    """Closed, Provider-neutral reasons safe to expose at public boundaries."""

    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_IDLE_TIMEOUT = "provider_idle_timeout"
    PROVIDER_OUTCOME_UNKNOWN = "provider_outcome_unknown"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXHAUSTED = "quota_exhausted"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    AUTHENTICATION_FAILED = "authentication_failed"
    PERMISSION_DENIED = "permission_denied"
    MODEL_NOT_FOUND = "model_not_found"
    RESOURCE_NOT_FOUND = "resource_not_found"
    CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
    INVALID_PARAMETER = "invalid_parameter"
    CONTENT_POLICY_REJECTED = "content_policy_rejected"
    INVALID_PROVIDER_RESPONSE = "invalid_provider_response"
    PROVIDER_PAYLOAD_INVALID = "provider_payload_invalid"
    COMPLETION_SHAPE_INVALID = "completion_shape_invalid"
    STREAM_EVENT_INVALID = "stream_event_invalid"
    GENERATION_SCHEMA_INVALID = "generation_schema_invalid"
    TOOL_CALL_INVALID = "tool_call_invalid"
    STREAM_INCOMPLETE = "stream_incomplete"
    OUTPUT_LIMIT_REACHED = "output_limit_reached"
    UNKNOWN_PROVIDER_FAILURE = "unknown_provider_failure"


def _default_failure_reason(kind: ModelErrorKind) -> ModelFailureReason:
    return {
        ModelErrorKind.TIMEOUT: ModelFailureReason.PROVIDER_TIMEOUT,
        ModelErrorKind.OUTCOME_UNKNOWN: ModelFailureReason.PROVIDER_OUTCOME_UNKNOWN,
        ModelErrorKind.RATE_LIMIT: ModelFailureReason.RATE_LIMITED,
        ModelErrorKind.UNAVAILABLE: ModelFailureReason.PROVIDER_UNAVAILABLE,
        ModelErrorKind.INCOMPLETE_RESPONSE: ModelFailureReason.OUTPUT_LIMIT_REACHED,
        ModelErrorKind.AUTHENTICATION: ModelFailureReason.AUTHENTICATION_FAILED,
        ModelErrorKind.INVALID_REQUEST: ModelFailureReason.INVALID_PARAMETER,
        ModelErrorKind.INVALID_RESPONSE: ModelFailureReason.INVALID_PROVIDER_RESPONSE,
        ModelErrorKind.UNKNOWN: ModelFailureReason.UNKNOWN_PROVIDER_FAILURE,
    }[kind]


class ModelGroupResolution(str, Enum):
    CONCRETE = "concrete"
    DEFERRED = "deferred"


class ModelResourceOwnership(str, Enum):
    BORROWED = "borrowed"
    OWNED = "owned"


@dataclass(frozen=True, slots=True)
class ModelRoute:
    route_id: str
    provider: str
    model: str
    provider_options: JsonObjectInput = field(
        default_factory=dict,
        kw_only=True,
        repr=False,
        metadata={"pygent_omit_if_empty": True},
    )

    def __post_init__(self) -> None:
        for name in ("route_id", "provider", "model"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.provider_options, Mapping):
            raise JsonValueError("provider_options must be a JSON object")
        options = (
            self.provider_options
            if isinstance(self.provider_options, FrozenJsonObject)
            else freeze_json_object(self.provider_options)
        )
        object.__setattr__(self, "provider_options", options)


@dataclass(frozen=True, slots=True)
class FallbackPolicy:
    order: tuple[str, ...]

    def __post_init__(self) -> None:
        order = tuple(self.order)
        if any(not isinstance(item, str) or not item for item in order):
            raise ValueError("fallback order must contain non-empty route IDs")
        if len(order) != len(set(order)):
            raise ValueError("fallback order contains duplicate route IDs")
        object.__setattr__(self, "order", order)


@dataclass(frozen=True, slots=True)
class ModelGroupConfig:
    name: str
    routes: tuple[ModelRoute, ...]
    fallback: FallbackPolicy
    max_concurrency: int | None = None
    capacity_key: str | None = None
    resolution: ModelGroupResolution = ModelGroupResolution.CONCRETE

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("model group name must be a non-empty string")
        if not isinstance(self.resolution, ModelGroupResolution):
            raise TypeError("resolution must be a ModelGroupResolution")
        routes = tuple(self.routes)
        if self.resolution is ModelGroupResolution.CONCRETE and not routes:
            raise ValueError("concrete model group routes must be non-empty")
        if self.resolution is ModelGroupResolution.DEFERRED and routes:
            raise ValueError("deferred model group routes must be empty")
        if any(not isinstance(route, ModelRoute) for route in routes):
            raise TypeError("model group routes must contain ModelRoute values")
        route_ids = tuple(route.route_id for route in routes)
        if len(route_ids) != len(set(route_ids)):
            raise ValueError("model group route IDs must be unique")
        unknown = set(self.fallback.order) - set(route_ids)
        if unknown:
            raise ValueError("fallback order references an unknown route")
        if self.max_concurrency is not None and (
            not isinstance(self.max_concurrency, int)
            or isinstance(self.max_concurrency, bool)
            or self.max_concurrency <= 0
        ):
            raise ValueError("max_concurrency must be greater than zero")
        if self.capacity_key is not None and (
            not isinstance(self.capacity_key, str) or not self.capacity_key
        ):
            raise ValueError("capacity_key must be non-empty when provided")
        object.__setattr__(self, "routes", routes)

    @classmethod
    def deferred(
        cls,
        *,
        name: str,
        max_concurrency: int | None = None,
        capacity_key: str | None = None,
    ) -> ModelGroupConfig:
        return cls(
            name=name,
            routes=(),
            fallback=FallbackPolicy(()),
            max_concurrency=max_concurrency,
            capacity_key=capacity_key,
            resolution=ModelGroupResolution.DEFERRED,
        )

    @property
    def is_deferred(self) -> bool:
        return self.resolution is ModelGroupResolution.DEFERRED


_OVERRIDABLE_GENERATION_FIELDS = frozenset(
    {"temperature", "max_output_tokens"}
)


@dataclass(frozen=True, slots=True)
class ModelCallPolicy:
    allow_profile_override: bool = False
    overridable_generation: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not isinstance(self.allow_profile_override, bool):
            raise TypeError("allow_profile_override must be a bool")
        values = frozenset(self.overridable_generation)
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError("overridable_generation must contain non-empty strings")
        unknown = values - _OVERRIDABLE_GENERATION_FIELDS
        if unknown:
            raise ValueError(
                "unsupported generation override fields: "
                + ", ".join(sorted(unknown))
            )
        object.__setattr__(self, "overridable_generation", values)


@dataclass(frozen=True, slots=True)
class ModelCallOptions:
    profile: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.profile is not None and (
            not isinstance(self.profile, str) or not self.profile
        ):
            raise ValueError("profile must be non-empty when provided")
        GenerationConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ModelCallOptions:
        unknown = set(value) - {"profile", "temperature", "max_output_tokens"}
        if unknown:
            raise ValueError(
                "unknown model call option fields: " + ", ".join(sorted(unknown))
            )
        return cls(
            profile=cast(str | None, value.get("profile")),
            temperature=cast(float | None, value.get("temperature")),
            max_output_tokens=cast(int | None, value.get("max_output_tokens")),
        )


@dataclass(frozen=True, slots=True)
class ModelResourceRef:
    resolver_id: str
    resource_id: str
    revision: str
    capacity_owner_id: str
    coordinator_domain: str

    def __post_init__(self) -> None:
        for name in (
            "resolver_id",
            "resource_id",
            "revision",
            "capacity_owner_id",
            "coordinator_domain",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")

    def to_dict(self) -> dict[str, str]:
        return {
            "resolver_id": self.resolver_id,
            "resource_id": self.resource_id,
            "revision": self.revision,
            "capacity_owner_id": self.capacity_owner_id,
            "coordinator_domain": self.coordinator_domain,
        }


@dataclass(frozen=True, slots=True)
class ModelResourceBundle:
    resolver_id: str
    route_resources: tuple[tuple[str, ModelResourceRef], ...]
    capacity_owner_id: str
    coordinator_domain: str

    def __post_init__(self) -> None:
        for name in ("resolver_id", "capacity_owner_id", "coordinator_domain"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        items = tuple(self.route_resources)
        route_ids = tuple(route_id for route_id, _ in items)
        if not items or len(route_ids) != len(set(route_ids)):
            raise ValueError("route_resources must contain unique route IDs")
        for route_id, ref in items:
            if not isinstance(route_id, str) or not route_id:
                raise ValueError("route resource IDs must be non-empty")
            if not isinstance(ref, ModelResourceRef):
                raise TypeError("route resources must contain ModelResourceRef values")
            if ref.resolver_id != self.resolver_id:
                raise ValueError("all route resources must use the bundle resolver")
            if (
                ref.capacity_owner_id != self.capacity_owner_id
                or ref.coordinator_domain != self.coordinator_domain
            ):
                raise ValueError("route resources must share the bundle capacity owner")
        object.__setattr__(self, "route_resources", items)

    @classmethod
    def shared(
        cls, routes: tuple[ModelRoute, ...], ref: ModelResourceRef
    ) -> ModelResourceBundle:
        return cls(
            resolver_id=ref.resolver_id,
            route_resources=tuple((route.route_id, ref) for route in routes),
            capacity_owner_id=ref.capacity_owner_id,
            coordinator_domain=ref.coordinator_domain,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "resolver_id": self.resolver_id,
            "route_resources": [
                {"route_id": route_id, "resource": ref.to_dict()}
                for route_id, ref in self.route_resources
            ],
            "capacity_owner_id": self.capacity_owner_id,
            "coordinator_domain": self.coordinator_domain,
        }


@dataclass(frozen=True, slots=True)
class ModelProfileSnapshot:
    deployment_scope_id: str
    group_name: str
    profile: str
    snapshot_id: str
    digest: str
    resource_bundle_digest: str | None
    model_group: ModelGroupConfig
    resources: ModelResourceBundle | None = None

    def __post_init__(self) -> None:
        for name in (
            "deployment_scope_id",
            "group_name",
            "profile",
            "snapshot_id",
            "digest",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.resource_bundle_digest is not None and (
            not isinstance(self.resource_bundle_digest, str)
            or not self.resource_bundle_digest
        ):
            raise ValueError("resource_bundle_digest must be non-empty when provided")
        if self.model_group.is_deferred:
            raise ValueError("profile snapshots require a concrete model group")
        if self.model_group.name != self.group_name:
            raise ValueError("profile snapshot group name does not match model group")
        if (self.resources is None) != (self.resource_bundle_digest is None):
            raise ValueError("resource bundle and digest must be present together")


@runtime_checkable
class ModelResourceResolver(Protocol):
    resolver_id: str

    async def validate(
        self, model_group: ModelGroupConfig, resources: ModelResourceBundle
    ) -> None: ...

    def acquire(
        self, model_group: ModelGroupConfig, resources: ModelResourceBundle
    ) -> Any: ...


class ModelGroupError(RuntimeError):
    """Base class for dynamic model-group control-plane failures."""


class ModelGroupConfigurationError(ModelGroupError):
    pass


class ModelProfileSelectionError(ModelGroupError):
    pass


class ModelDeploymentUnavailableError(ModelGroupError):
    pass


class ModelDeploymentConflictError(ModelGroupError):
    pass


@dataclass(frozen=True, slots=True)
class ExponentialBackoff:
    initial: float = 0.0
    maximum: float = 0.0
    multiplier: float = 2.0

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in (self.initial, self.maximum, self.multiplier)
        ):
            raise ValueError("backoff values must be finite numbers")
        if self.initial < 0 or self.maximum < self.initial:
            raise ValueError("backoff requires 0 <= initial <= maximum")
        if self.multiplier < 1:
            raise ValueError("backoff multiplier must be at least one")

    def delay(self, retry_index: int) -> float:
        if retry_index < 0:
            raise ValueError("retry_index must be non-negative")
        return min(self.maximum, self.initial * self.multiplier**retry_index)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts_per_route: int = 2
    retry_on: tuple[ModelErrorKind, ...] = (
        ModelErrorKind.TIMEOUT,
        ModelErrorKind.RATE_LIMIT,
        ModelErrorKind.UNAVAILABLE,
        ModelErrorKind.INCOMPLETE_RESPONSE,
    )
    backoff: ExponentialBackoff = ExponentialBackoff()
    attempt_idle_timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_attempts_per_route, int)
            or isinstance(self.max_attempts_per_route, bool)
            or self.max_attempts_per_route < 1
        ):
            raise ValueError("max_attempts_per_route must be at least one")
        retry_on = tuple(self.retry_on)
        if any(not isinstance(kind, ModelErrorKind) for kind in retry_on):
            raise TypeError("retry_on must contain ModelErrorKind values")
        if len(retry_on) != len(set(retry_on)):
            raise ValueError("retry_on contains duplicates")
        if self.attempt_idle_timeout_seconds is not None and (
            not isinstance(self.attempt_idle_timeout_seconds, (int, float))
            or isinstance(self.attempt_idle_timeout_seconds, bool)
            or not math.isfinite(self.attempt_idle_timeout_seconds)
            or self.attempt_idle_timeout_seconds <= 0
        ):
            raise ValueError(
                "attempt_idle_timeout_seconds must be finite and positive"
            )
        object.__setattr__(self, "retry_on", retry_on)


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    temperature: float | None = None
    max_output_tokens: int | None = None
    response_schema: JsonObjectInput | None = None
    response_schema_name: str = "response"
    tool_choice: str | None = None

    def __post_init__(self) -> None:
        if self.temperature is not None and (
            not isinstance(self.temperature, (int, float))
            or isinstance(self.temperature, bool)
            or not math.isfinite(self.temperature)
            or self.temperature < 0
        ):
            raise ValueError("temperature must be a finite non-negative number")
        if self.max_output_tokens is not None and (
            not isinstance(self.max_output_tokens, int)
            or isinstance(self.max_output_tokens, bool)
            or self.max_output_tokens <= 0
        ):
            raise ValueError("max_output_tokens must be greater than zero")
        if not isinstance(self.response_schema_name, str) or not self.response_schema_name:
            raise ValueError("response_schema_name must be a non-empty string")
        if self.tool_choice is not None and (
            not isinstance(self.tool_choice, str) or not self.tool_choice
        ):
            raise ValueError("tool_choice must be non-empty when provided")
        if self.response_schema is not None:
            object.__setattr__(
                self, "response_schema", freeze_json_object(self.response_schema)
            )


@dataclass(frozen=True, slots=True)
class ModelAttempt:
    route_id: str
    status: Literal["succeeded", "failed", "cancelled"]
    error_kind: ModelErrorKind | None = None
    attempt: int = 1
    reason_code: ModelFailureReason | None = None
    http_status: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.route_id, str) or not self.route_id:
            raise ValueError("attempt route_id must be non-empty")
        if self.status not in ("succeeded", "failed", "cancelled"):
            raise ValueError("invalid model attempt status")
        if (
            not isinstance(self.attempt, int)
            or isinstance(self.attempt, bool)
            or self.attempt < 1
        ):
            raise ValueError("attempt number must be at least one")
        if self.error_kind is not None and not isinstance(
            self.error_kind, ModelErrorKind
        ):
            raise TypeError("error_kind must be a ModelErrorKind or None")
        if self.reason_code is not None and not isinstance(
            self.reason_code, ModelFailureReason
        ):
            raise TypeError("reason_code must be a ModelFailureReason or None")
        if self.status == "failed" and self.error_kind is not None:
            if self.reason_code is None:
                object.__setattr__(
                    self, "reason_code", _default_failure_reason(self.error_kind)
                )
        elif self.reason_code is not None or self.http_status is not None:
            raise ValueError("only failed model attempts may contain diagnostics")
        if self.http_status is not None and (
            not isinstance(self.http_status, int)
            or isinstance(self.http_status, bool)
            or not 100 <= self.http_status <= 599
        ):
            raise ValueError("http_status must be an HTTP status integer")


class ModelCallError(ExecutionFailureError):
    """Sanitized terminal model failure after retry/fallback is exhausted."""

    def __init__(
        self,
        message: str,
        *,
        kind: ModelErrorKind = ModelErrorKind.UNKNOWN,
        attempts: tuple[ModelAttempt, ...] = (),
        partial_output: bool = False,
    ) -> None:
        if not isinstance(partial_output, bool):
            raise TypeError("partial_output must be a bool")
        if not isinstance(kind, ModelErrorKind):
            raise TypeError("kind must be a ModelErrorKind")
        self.kind = kind
        self.attempts = tuple(attempts)
        self.partial_output = partial_output
        super().__init__(
            ExecutionFailure(
                domain="model",
                kind=kind.value,
                message=message,
                retryable=kind
                in {
                    ModelErrorKind.TIMEOUT,
                    ModelErrorKind.RATE_LIMIT,
                    ModelErrorKind.UNAVAILABLE,
                    ModelErrorKind.INCOMPLETE_RESPONSE,
                },
                outcome_unknown=kind is ModelErrorKind.OUTCOME_UNKNOWN,
                partial_output=partial_output,
                details={
                    "attempts": [
                        {
                            "route_id": attempt.route_id,
                            "status": attempt.status,
                            "error_kind": (
                                None
                                if attempt.error_kind is None
                                else attempt.error_kind.value
                            ),
                            "attempt": attempt.attempt,
                            "reason_code": (
                                None
                                if attempt.reason_code is None
                                else attempt.reason_code.value
                            ),
                            "http_status": attempt.http_status,
                        }
                        for attempt in self.attempts
                    ]
                },
            )
        )

    @classmethod
    def from_failure(cls, failure: ExecutionFailure) -> ModelCallError:
        if failure.domain != "model":
            raise ValueError("execution failure is not a model failure")
        raw_attempts = cast(Mapping[str, Any], failure.details).get("attempts", ())
        if not isinstance(raw_attempts, (list, tuple)):
            raise TypeError("model failure attempts must be an array")
        attempts: list[ModelAttempt] = []
        for item in raw_attempts:
            if not isinstance(item, Mapping):
                raise TypeError("model failure attempt must be an object")
            raw_kind = item.get("error_kind")
            attempts.append(
                ModelAttempt(
                    route_id=cast(str, item.get("route_id")),
                    status=cast(Any, item.get("status")),
                    error_kind=(
                        None if raw_kind is None else ModelErrorKind(cast(str, raw_kind))
                    ),
                    attempt=cast(int, item.get("attempt", 1)),
                    reason_code=(
                        None
                        if item.get("reason_code") is None
                        else ModelFailureReason(cast(str, item.get("reason_code")))
                    ),
                    http_status=cast(int | None, item.get("http_status")),
                )
            )
        return cls(
            failure.message,
            kind=ModelErrorKind(failure.kind),
            attempts=tuple(attempts),
            partial_output=failure.partial_output,
        )


class ModelProviderError(RuntimeError):
    """Sanitized adapter failure with a normalized kind."""

    def __init__(
        self,
        kind: ModelErrorKind,
        message: str = "model provider failed",
        *,
        reason_code: ModelFailureReason | None = None,
        http_status: int | None = None,
    ) -> None:
        if not isinstance(kind, ModelErrorKind):
            raise TypeError("kind must be a ModelErrorKind")
        if reason_code is not None and not isinstance(reason_code, ModelFailureReason):
            raise TypeError("reason_code must be a ModelFailureReason or None")
        if http_status is not None and (
            not isinstance(http_status, int)
            or isinstance(http_status, bool)
            or not 100 <= http_status <= 599
        ):
            raise ValueError("http_status must be an HTTP status integer")
        super().__init__(message)
        self.kind = kind
        self.reason_code = reason_code or _default_failure_reason(kind)
        self.http_status = http_status


__all__ = [
    "ExponentialBackoff",
    "FallbackPolicy",
    "GenerationConfig",
    "ModelAttempt",
    "ModelCallError",
    "ModelCallOptions",
    "ModelCallPolicy",
    "ModelDeploymentConflictError",
    "ModelDeploymentUnavailableError",
    "ModelErrorKind",
    "ModelFailureReason",
    "ModelGroupConfig",
    "ModelGroupConfigurationError",
    "ModelGroupError",
    "ModelGroupResolution",
    "ModelProfileSelectionError",
    "ModelProfileSnapshot",
    "ModelProviderError",
    "ModelResourceBundle",
    "ModelResourceOwnership",
    "ModelResourceRef",
    "ModelResourceResolver",
    "ModelRoute",
    "RetryPolicy",
]
