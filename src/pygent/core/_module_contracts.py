"""Module dependency, infrastructure, effect, and recovery contracts."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import (
    AbstractAsyncContextManager,
    contextmanager,
)
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from .execution import (
    EffectOutcome,
)
from .json_values import (
    JsonValue,
)
from .values import Context, Message

if TYPE_CHECKING:
    from ._module_definition import Module
InputMessageT = TypeVar("InputMessageT", bound=Message)
OutputMessageT = TypeVar("OutputMessageT", bound=Message)
DependencyInputMessageT_contra = TypeVar(
    "DependencyInputMessageT_contra", bound=Message, contravariant=True
)
DependencyOutputMessageT_co = TypeVar(
    "DependencyOutputMessageT_co", bound=Message, covariant=True
)

class EffectSideEffect(str, Enum):
    PURE = "pure"
    READ = "read"
    WRITE = "write"
    EXTERNAL = "external"


class EffectIdempotency(str, Enum):
    INHERENT = "inherent"
    REQUIRES_KEY = "requires_key"
    NOT_IDEMPOTENT = "not_idempotent"


class EffectRetryPolicy(str, Enum):
    REPLAY_SAFE = "replay_safe"
    FAIL_CLOSED = "fail_closed"


@dataclass(frozen=True, slots=True, kw_only=True)
class EffectSpec:
    """Portable recovery contract for one managed infrastructure effect."""

    effect_type: str
    side_effect: EffectSideEffect
    idempotency: EffectIdempotency
    retry_policy: EffectRetryPolicy
    idempotency_key: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.effect_type, str) or not self.effect_type:
            raise ValueError("effect_type must be a non-empty string")
        for name, enum_type in (
            ("side_effect", EffectSideEffect),
            ("idempotency", EffectIdempotency),
            ("retry_policy", EffectRetryPolicy),
        ):
            value = getattr(self, name)
            if not isinstance(value, enum_type):
                object.__setattr__(self, name, enum_type(value))
        if self.idempotency_key is not None and (
            not isinstance(self.idempotency_key, str) or not self.idempotency_key
        ):
            raise ValueError("idempotency_key must be non-empty when provided")
        if (
            self.idempotency is EffectIdempotency.REQUIRES_KEY
            and self.idempotency_key is None
        ):
            raise ValueError("requires_key effects need a stable idempotency_key")
        if self.retry_policy is EffectRetryPolicy.REPLAY_SAFE and not self.replay_safe:
            raise ValueError("replay_safe policy requires a replay-safe effect")

    @property
    def replay_safe(self) -> bool:
        return (
            self.side_effect in (EffectSideEffect.PURE, EffectSideEffect.READ)
            or self.idempotency is EffectIdempotency.INHERENT
            or (
                self.idempotency is EffectIdempotency.REQUIRES_KEY
                and self.idempotency_key is not None
            )
        )


class EffectRecoveryUnknown(RuntimeError):
    """An incomplete non-replayable effect may already have committed."""


@dataclass(frozen=True, slots=True)
class CapacityPermit:
    """Public resource-capacity ownership proof.

    ``fencing_token`` is present only when the coordinator can issue a stable,
    monotonically increasing token.  An external resource provides strong
    fencing only when it validates this token atomically while committing the
    protected operation.
    """

    owner_key: str | None = None
    fencing_token: int | None = None


def current_capacity_permit() -> CapacityPermit | None:
    """Return the permit visible to the current managed resource operation."""

    return _capacity_permit.get()


@runtime_checkable
class ModuleDependency(
    Protocol[DependencyInputMessageT_contra, DependencyOutputMessageT_co]
):
    """Callable Child contract shared by raw, bound, and remote Modules."""

    async def __call__(
        self, message: DependencyInputMessageT_contra, context: Context
    ) -> tuple[DependencyOutputMessageT_co, Context]: ...


@runtime_checkable
class Infrastructure(Protocol):
    """Public services available to infrastructure-oriented Modules.

    Provider and executor behavior remains outside Runtime.  Runtime supplies
    only governance, replay, stable resource resolution, and identity.
    """

    @property
    def deadline(self) -> float | None: ...

    @property
    def managed_execution_id(self) -> str | None: ...

    def model_permit(
        self,
        resource_key: str | None = None,
        *,
        max_concurrency: int | None = None,
    ) -> AbstractAsyncContextManager[CapacityPermit]: ...

    def tool_permit(
        self, resource_key: str | None = None
    ) -> AbstractAsyncContextManager[CapacityPermit]: ...

    def resolve_model_invoker(self, model_group: str) -> object: ...

    def model_call_options(self, model_group: str) -> Mapping[str, JsonValue]: ...

    def resolve_model_deployment(self, model_group: str) -> object: ...

    def model_deployment_lease(
        self, deployment: object
    ) -> AbstractAsyncContextManager[object]: ...

    def resolve_tool_registry(self) -> object: ...

    def tool_idempotency_key(self, call_id: str) -> str | None: ...

    async def gather(
        self, operations: tuple[Callable[[], Awaitable[Any]], ...]
    ) -> tuple[Any, ...]: ...

    async def execute_effect(
        self,
        *,
        spec: EffectSpec,
        request: Mapping[str, JsonValue],
        operation: Callable[[], Awaitable[JsonValue]],
    ) -> EffectOutcome[JsonValue]: ...

    async def submit_tool_task(self, spec: Any, call: Any) -> object | None: ...


class ExecutionScope(Infrastructure, Protocol):
    async def invoke_module(
        self,
        module: ModuleDependency[Any, Any],
        message: Message,
        context: Context,
    ) -> tuple[Message, Context]: ...

    async def emit_event(
        self,
        module: Module[Any, Any],
        kind: str,
        data: Mapping[str, JsonValue],
    ) -> None: ...

    async def wait_external(
        self,
        *,
        kind: str,
        key: str,
        request: Mapping[str, JsonValue],
        timeout: float | None,
    ) -> Mapping[str, JsonValue]: ...


def active_infrastructure() -> Infrastructure | None:
    """Return the active infrastructure SPI, or ``None`` outside execution."""

    return _execution_scope.get()


def current_infrastructure() -> Infrastructure:
    """Return the active infrastructure SPI or fail outside Module execution."""

    infrastructure = active_infrastructure()
    if infrastructure is None:
        raise RuntimeError(
            "infrastructure services require an active Module execution"
        )
    return infrastructure


@contextmanager
def independent_execution() -> Iterator[None]:
    """Prevent a newly admitted independent task from inheriting Parent scope."""

    token = _execution_scope.set(None)
    try:
        yield
    finally:
        _execution_scope.reset(token)


class DirectExecutionError(RuntimeError):
    """Raised when a dependency is unavailable to direct execution."""


class RecoverySafety(str, Enum):
    """Whether a Module definition promises safe boundary re-execution."""

    UNDECLARED = "undeclared"
    MODULE_BOUNDARY_RETRY = "module_boundary_retry"


class EffectSafety(str, Enum):
    """How a replayable Module constrains nondeterminism and side effects."""

    UNDECLARED = "undeclared"
    EFFECT_FREE = "effect_free"
    MANAGED_EFFECTS = "managed_effects"


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionRequirements:
    """Generic managed-execution capabilities declared by a Module definition."""

    requires_finite_deadline: bool = False
    required_capabilities: tuple[str, ...] = ()
    recovery_safety: RecoverySafety = RecoverySafety.UNDECLARED
    effect_safety: EffectSafety = EffectSafety.UNDECLARED

    def __post_init__(self) -> None:
        capabilities = tuple(self.required_capabilities)
        if any(not isinstance(item, str) or not item for item in capabilities):
            raise ValueError("required capabilities must be non-empty strings")
        if len(capabilities) != len(set(capabilities)):
            raise ValueError("required capabilities must be unique")
        if not isinstance(self.recovery_safety, RecoverySafety):
            raise TypeError("recovery_safety must be a RecoverySafety")
        if not isinstance(self.effect_safety, EffectSafety):
            raise TypeError("effect_safety must be an EffectSafety")
        object.__setattr__(self, "required_capabilities", capabilities)


_capacity_permit: ContextVar[CapacityPermit | None] = ContextVar(
    "pygent_capacity_permit", default=None
)

_execution_scope: ContextVar[ExecutionScope | None] = ContextVar(
    "pygent_execution_scope", default=None
)
