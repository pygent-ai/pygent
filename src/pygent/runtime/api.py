"""Runtime interfaces; no scheduler implementation lives in this module."""

from __future__ import annotations

import math
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from enum import Enum
from types import TracebackType
from typing import Any, Protocol, TypeVar

from pygent.core import (
    CapacityPermit,
    Context,
    ExecutionEvent,
    ExecutionInputDelivery,
    ExecutionOptions,
    ExecutionOutcome,
    ExecutionOwnerState,
    ExecutionPhase,
    ExecutionSnapshot,
    ExecutionStatus,
    JsonValue,
    Message,
    Module,
)

InputMessageT = TypeVar("InputMessageT", bound=Message)
OutputMessageT = TypeVar("OutputMessageT", bound=Message)
BoundInputMessageT_contra = TypeVar(
    "BoundInputMessageT_contra", bound=Message, contravariant=True
)
BoundOutputMessageT_co = TypeVar(
    "BoundOutputMessageT_co", bound=Message, covariant=True
)


def _event_cursor(after: int | None) -> int:
    if after is not None and (
        not isinstance(after, int) or isinstance(after, bool) or after < -1
    ):
        raise ValueError("event cursor must be -1 or a non-negative integer")
    return -1 if after is None else after


class CapacityScope(str, Enum):
    """Where one declared capacity total is coordinated."""

    DEPLOYMENT = "deployment"
    RUNTIME_INSTANCE = "runtime_instance"
    EXTERNAL_RESOURCE = "external_resource"


class DurabilityMode(str, Enum):
    """How strongly a Binding requests durable execution facilities."""

    DISABLED = "disabled"
    PREFERRED = "preferred"
    REQUIRED = "required"


class JobState(str, Enum):
    """Stable lifecycle of an independently admitted durable Job."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class JobRef:
    """Portable reference to a durable Job and the ToolTask it carries."""

    job_id: str
    task_id: str

    def __post_init__(self) -> None:
        for name in ("job_id", "task_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class JobSnapshot:
    """Immutable public state of a durable, independently governed Job."""

    job_id: str
    task_id: str
    logical_key: str
    state: JobState
    binding_id: str
    plan_id: str
    resource_key: str | None
    required_capabilities: tuple[str, ...]
    attempt: int = 1

    def __post_init__(self) -> None:
        for name in ("job_id", "task_id", "logical_key", "binding_id", "plan_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.state, JobState):
            object.__setattr__(self, "state", JobState(self.state))
        if self.resource_key is not None and (
            not isinstance(self.resource_key, str) or not self.resource_key
        ):
            raise ValueError("resource_key must be non-empty when provided")
        capabilities = tuple(self.required_capabilities)
        if any(not isinstance(value, str) or not value for value in capabilities):
            raise ValueError("required_capabilities must contain non-empty strings")
        if len(capabilities) != len(set(capabilities)):
            raise ValueError("required_capabilities must be unique")
        object.__setattr__(self, "required_capabilities", capabilities)
        if (
            not isinstance(self.attempt, int)
            or isinstance(self.attempt, bool)
            or self.attempt < 1
        ):
            raise ValueError("attempt must be a positive integer")

    @property
    def ref(self) -> JobRef:
        return JobRef(self.job_id, self.task_id)


@dataclass(frozen=True, slots=True)
class DurabilityPolicy:
    """Immutable Binding-level durability eligibility request."""

    mode: DurabilityMode = DurabilityMode.PREFERRED

    def __post_init__(self) -> None:
        if not isinstance(self.mode, DurabilityMode):
            object.__setattr__(self, "mode", DurabilityMode(self.mode))


@dataclass(frozen=True, slots=True)
class DurabilityReport:
    """Effective, bind-time durability capability report."""

    requested_mode: DurabilityMode
    effective_capabilities: tuple[str, ...]
    missing_capabilities: tuple[str, ...]
    recovery_level: str
    checkpoint_policy: str
    replay_policy: str
    event_reconnect: bool
    capacity_scope: CapacityScope
    degraded_reasons: tuple[str, ...] = ()
    recovery_undeclared_modules: tuple[str, ...] = ()
    effect_unverified_modules: tuple[str, ...] = ()
    external_side_effect_guarantee: str = "at_least_once"
    arbitrary_coroutine_recovery: bool = False
    exactly_once_external_side_effects: bool = False
    detached_tool_gaps: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.requested_mode, DurabilityMode):
            raise TypeError("requested_mode must be a DurabilityMode")
        for name in (
            "effective_capabilities",
            "missing_capabilities",
            "degraded_reasons",
            "recovery_undeclared_modules",
            "effect_unverified_modules",
            "detached_tool_gaps",
        ):
            values = tuple(getattr(self, name))
            if any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"{name} must contain non-empty strings")
            object.__setattr__(self, name, values)
        for name in (
            "recovery_level",
            "checkpoint_policy",
            "replay_policy",
            "external_side_effect_guarantee",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.capacity_scope, CapacityScope):
            raise TypeError("capacity_scope must be a CapacityScope")
        for name in (
            "event_reconnect",
            "arbitrary_coroutine_recovery",
            "exactly_once_external_side_effects",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_non_negative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionCapacityPolicy:
    """Admission and execution-tree limits shared by one Binding."""

    scope: CapacityScope
    max_live_executions: int
    max_runnable_executions: int
    max_queue_size: int
    max_waiters: int
    max_child_depth: int
    max_children_per_execution: int
    max_external_wait_seconds: float = 300.0

    def __post_init__(self) -> None:
        for name in (
            "max_live_executions",
            "max_runnable_executions",
            "max_child_depth",
            "max_children_per_execution",
        ):
            _require_positive_int(name, getattr(self, name))
        for name in ("max_queue_size", "max_waiters"):
            _require_non_negative_int(name, getattr(self, name))
        if self.max_runnable_executions > self.max_live_executions:
            raise ValueError("max_runnable_executions cannot exceed max_live_executions")
        if (
            isinstance(self.max_external_wait_seconds, bool)
            or not isinstance(self.max_external_wait_seconds, (int, float))
            or not math.isfinite(self.max_external_wait_seconds)
            or self.max_external_wait_seconds <= 0
        ):
            raise ValueError(
                "max_external_wait_seconds must be finite and greater than zero"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class CapacityPolicy:
    """Optional cross-execution capacity for one physical resource class."""

    scope: CapacityScope = CapacityScope.EXTERNAL_RESOURCE
    max_concurrency: int | None = None
    max_queue_size: int | None = None
    capacity_key: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scope, CapacityScope):
            object.__setattr__(self, "scope", CapacityScope(self.scope))
        if self.capacity_key is not None and (
            not isinstance(self.capacity_key, str) or not self.capacity_key
        ):
            raise ValueError("capacity_key must be non-empty when provided")
        if self.max_concurrency is None:
            if self.max_queue_size is not None:
                raise ValueError("max_queue_size requires a limited max_concurrency")
            return
        if self.scope is CapacityScope.EXTERNAL_RESOURCE:
            raise ValueError(
                "external_resource capacity is owned outside Runtime and must "
                "use passthrough"
            )
        if self.scope is CapacityScope.DEPLOYMENT and self.capacity_key is None:
            raise ValueError("deployment capacity requires a stable capacity_key")
        _require_positive_int("max_concurrency", self.max_concurrency)
        if self.max_queue_size is not None:
            _require_non_negative_int("max_queue_size", self.max_queue_size)

    @classmethod
    def passthrough(
        cls, *, capacity_key: str | None = None
    ) -> CapacityPolicy:
        return cls(
            scope=CapacityScope.EXTERNAL_RESOURCE,
            capacity_key=capacity_key,
        )

    @classmethod
    def limited(
        cls,
        *,
        max_concurrency: int,
        max_queue_size: int,
        capacity_key: str | None = None,
        scope: CapacityScope = CapacityScope.RUNTIME_INSTANCE,
    ) -> CapacityPolicy:
        return cls(
            scope=scope,
            max_concurrency=max_concurrency,
            max_queue_size=max_queue_size,
            capacity_key=capacity_key,
        )


class RunnableCapacityGate(Protocol):
    """Execution-lease gate supplied by a capacity coordinator."""

    async def acquire(self, *, resume: bool = False) -> None: ...

    def release(self) -> None: ...


class ResourceCapacityGate(Protocol):
    """Model/Tool permit gate supplied by a capacity coordinator."""

    def permit(self) -> AbstractAsyncContextManager[CapacityPermit]: ...


class ExecutionCapacityState(Protocol):
    """Deployment execution capacity state supplied by a coordinator."""

    @property
    def runnable(self) -> RunnableCapacityGate: ...

    async def admit(self) -> None: ...

    async def release_live(self) -> None: ...

    def waiter_slot(self) -> AbstractAsyncContextManager[CapacityPermit]: ...


class CapacityCoordinator(Protocol):
    """Injectable owner for deployment-scoped execution/Model/Tool capacity."""

    def execution_state(
        self, name: str, policy: ExecutionCapacityPolicy
    ) -> ExecutionCapacityState: ...

    def resource_gate(
        self,
        kind: str,
        capacity_key: str,
        policy: CapacityPolicy,
    ) -> ResourceCapacityGate: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class Binding:
    """Deployment-time policy attached to a reusable Module graph."""

    name: str
    execution_capacity: ExecutionCapacityPolicy
    model_capacity: CapacityPolicy
    tool_capacity: CapacityPolicy
    durability: DurabilityPolicy = DurabilityPolicy()
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("binding name must be non-empty")
        if not isinstance(self.durability, DurabilityPolicy):
            raise TypeError("durability must be a DurabilityPolicy")
        object.__setattr__(self, "metadata", tuple(self.metadata))


class RuntimeErrorBase(RuntimeError):
    """Base class for managed execution failures."""


class RuntimeClosedError(RuntimeErrorBase):
    """Raised when new work is submitted to a closed Runtime."""


class ExecutionAdmissionError(RuntimeErrorBase):
    """Raised before execution when a bounded admission queue is full."""


class ExecutionDeadlineExceeded(RuntimeErrorBase, TimeoutError):
    """Raised when a managed execution reaches its effective deadline."""


class ExternalWaitError(RuntimeErrorBase):
    """Base class for bounded external-wait failures."""


class ExternalWaitRejected(ExternalWaitError):
    """Raised when a waiter is invalid, duplicate, or over capacity."""


class ExternalWaitNotFound(ExternalWaitError):
    """Raised when feedback has no matching live waiter."""


class ExecutionStream(Protocol[BoundOutputMessageT_co]):
    async def __aenter__(self) -> ExecutionStream[BoundOutputMessageT_co]: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...

    def __aiter__(self) -> AsyncIterator[ExecutionEvent]: ...

    async def final_result(self) -> tuple[BoundOutputMessageT_co, Context]: ...


class ExecutionSubscription(Protocol):
    async def __aenter__(self) -> AsyncIterator[ExecutionEvent]: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...


class ExecutionBackend(Protocol[BoundOutputMessageT_co]):
    """Storage/transport boundary used by stable Execution handles."""

    async def snapshot(self, execution_id: str) -> ExecutionSnapshot: ...

    async def result(
        self, execution_id: str
    ) -> tuple[BoundOutputMessageT_co, Context]: ...

    async def request_cancel(self, execution_id: str) -> bool: ...

    async def send_input(
        self, execution_id: str, *, input_id: str, kind: str, value: JsonValue
    ) -> ExecutionInputDelivery: ...

    def subscribe(
        self, execution_id: str, *, after: int | None = None
    ) -> ExecutionSubscription: ...


class ExecutionHandle(Protocol[BoundOutputMessageT_co]):
    """Advanced control plane for a managed root execution."""

    @property
    def execution_id(self) -> str: ...

    @property
    def trace_id(self) -> str: ...

    @property
    def status(self) -> ExecutionStatus: ...

    async def snapshot(self) -> ExecutionSnapshot: ...

    async def outcome(self) -> ExecutionOutcome: ...

    async def result(self) -> tuple[BoundOutputMessageT_co, Context]: ...

    async def cancel(self) -> bool: ...

    async def send_input(
        self, *, input_id: str, kind: str, value: JsonValue
    ) -> ExecutionInputDelivery: ...

    def subscribe(self, *, after: int | None = None) -> ExecutionSubscription: ...


class BoundModule(Protocol[BoundInputMessageT_contra, BoundOutputMessageT_co]):
    """Executable handle returned when a Module is attached to a Runtime."""

    @property
    def durability(self) -> DurabilityReport: ...

    @property
    def model_groups(self) -> Any: ...

    async def invoke(
        self,
        message: BoundInputMessageT_contra,
        context: Context,
        *,
        execution: ExecutionOptions | None = None,
    ) -> tuple[BoundOutputMessageT_co, Context]: ...

    def stream(
        self,
        message: BoundInputMessageT_contra,
        context: Context,
        *,
        execution: ExecutionOptions | None = None,
    ) -> ExecutionStream[BoundOutputMessageT_co]: ...

    async def start(
        self,
        message: BoundInputMessageT_contra,
        context: Context,
        *,
        execution: ExecutionOptions | None = None,
    ) -> ExecutionHandle[BoundOutputMessageT_co]: ...


class Runtime(Protocol):
    """Logical execution domain implemented by local or distributed runtimes."""

    def bind(
        self,
        module: Module[InputMessageT, OutputMessageT],
        *,
        binding: Binding | None = None,
    ) -> BoundModule[InputMessageT, OutputMessageT]: ...

    def create_binding(
        self,
        *,
        name: str,
        execution_capacity: ExecutionCapacityPolicy,
        model_capacity: CapacityPolicy,
        tool_capacity: CapacityPolicy,
        durability: DurabilityPolicy | None = None,
        metadata: tuple[tuple[str, str], ...] = (),
    ) -> Any: ...

    async def get_job(self, job_id: str) -> JobSnapshot | None: ...

    async def get_execution_handle(
        self, execution_id: str
    ) -> ExecutionHandle[Message]: ...

    async def purge_execution(self, execution_id: str) -> None: ...


__all__ = [
    "Binding",
    "BoundModule",
    "CapacityPolicy",
    "CapacityScope",
    "DurabilityMode",
    "DurabilityPolicy",
    "DurabilityReport",
    "ExecutionAdmissionError",
    "ExecutionBackend",
    "ExecutionCapacityPolicy",
    "ExecutionDeadlineExceeded",
    "ExecutionEvent",
    "ExecutionHandle",
    "ExecutionOptions",
    "ExecutionOutcome",
    "ExecutionOwnerState",
    "ExecutionPhase",
    "ExecutionSnapshot",
    "ExecutionStatus",
    "ExecutionStream",
    "ExecutionSubscription",
    "ExternalWaitError",
    "ExternalWaitNotFound",
    "ExternalWaitRejected",
    "JobRef",
    "JobSnapshot",
    "JobState",
    "Runtime",
    "RuntimeClosedError",
    "RuntimeErrorBase",
]
