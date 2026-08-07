"""PyTorch-like Module definition contract."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from contextlib import (
    AbstractAsyncContextManager,
    asynccontextmanager,
    contextmanager,
    suppress,
)
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Protocol,
    Self,
    TypeVar,
    cast,
    runtime_checkable,
)
from uuid import uuid4

from .definition import module_definition_config
from .execution import (
    EXECUTION_EVENT_SCHEMA_VERSION,
    EffectDisposition,
    EffectOutcome,
    ExecutionEvent,
    ExecutionOptions,
    ExecutionStatus,
)
from .json_values import (
    FrozenJsonObject,
    JsonObjectInput,
    JsonValue,
    freeze_json,
    freeze_json_object,
)
from .values import Context, Message

if TYPE_CHECKING:
    from pygent.runtime import Binding, BoundModule, Runtime


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


_capacity_permit: ContextVar[CapacityPermit | None] = ContextVar(
    "pygent_capacity_permit", default=None
)


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


_execution_scope: ContextVar[ExecutionScope | None] = ContextVar(
    "pygent_execution_scope", default=None
)


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


@dataclass(slots=True)
class _DirectExecutionRecord(Generic[OutputMessageT]):
    """Single owner of direct execution state and its non-blocking journal."""

    execution_id: str
    trace_id: str
    root_span_id: str
    module: Module[Any, OutputMessageT]
    message: Message
    context: Context
    options: ExecutionOptions
    status: ExecutionStatus = ExecutionStatus.QUEUED
    events: list[ExecutionEvent] = field(default_factory=list)
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    task: asyncio.Task[tuple[OutputMessageT, Context]] | None = None
    module_paths: dict[int, str] = field(default_factory=dict)

    @property
    def terminal(self) -> bool:
        return self.status.terminal

    def ensure_started(self) -> None:
        if self.task is None:
            self.task = asyncio.create_task(
                _run_direct_execution(self),
                name=f"pygent-execution-{self.execution_id}",
            )

    async def emit(
        self,
        *,
        span_id: str,
        parent_span_id: str | None,
        module_path: str,
        kind: str,
        data: Mapping[str, JsonValue],
    ) -> ExecutionEvent:
        if not isinstance(kind, str) or not kind:
            raise ValueError("event kind must be a non-empty string")
        async with self.condition:
            event = ExecutionEvent(
                schema_version=EXECUTION_EVENT_SCHEMA_VERSION,
                event_id=str(uuid4()),
                execution_id=self.execution_id,
                trace_id=self.trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                sequence=len(self.events),
                timestamp_unix_ns=time.time_ns(),
                module_path=module_path,
                kind=kind,
                data=data,
            )
            self.events.append(event)
            self.condition.notify_all()
            return event

    async def notify_terminal(self) -> None:
        async with self.condition:
            self.condition.notify_all()


_direct_span: ContextVar[tuple[str, str | None, str] | None] = ContextVar(
    "pygent_direct_span", default=None
)


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


class _DirectExecutionScope:
    """Ephemeral local scope used by one unbound Root invocation."""

    def __init__(self, record: _DirectExecutionRecord[Any]) -> None:
        self.execution_id = record.execution_id
        self.trace_id = record.trace_id
        self._record = record
        self._active = True

    def close(self) -> None:
        self._active = False

    @property
    def deadline(self) -> None:
        return None

    async def invoke_module(
        self,
        module: ModuleDependency[Any, Any],
        message: Message,
        context: Context,
    ) -> tuple[Message, Context]:
        if not self._active:
            raise RuntimeError("the direct execution scope is already closed")
        if isinstance(module, RemoteModule):
            raise DirectExecutionError(
                "RemoteModule requires a managed executiontime execution scope"
            )
        if not isinstance(module, Module):
            raise TypeError("direct execution can only invoke local Module children")
        current = _direct_span.get()
        if current is None:
            span_id = self._record.root_span_id
            parent_span_id = None
            module_path = "root"
        else:
            span_id = str(uuid4())
            parent_span_id = current[0]
            module_path = self._record.module_paths.get(
                id(module), f"{current[2]}.{type(module).__qualname__}"
            )
        await self._record.emit(
            span_id=span_id,
            parent_span_id=parent_span_id,
            module_path=module_path,
            kind="span.started",
            data={},
        )
        token = _direct_span.set((span_id, parent_span_id, module_path))
        try:
            result = _validate_result(await module.forward(message, context))
        except asyncio.CancelledError:
            await self._record.emit(
                span_id=span_id,
                parent_span_id=parent_span_id,
                module_path=module_path,
                kind="span.cancelled",
                data={},
            )
            raise
        except BaseException as exc:
            await self._record.emit(
                span_id=span_id,
                parent_span_id=parent_span_id,
                module_path=module_path,
                kind="span.failed",
                data={"error_type": type(exc).__name__, "message": str(exc)},
            )
            raise
        else:
            await self._record.emit(
                span_id=span_id,
                parent_span_id=parent_span_id,
                module_path=module_path,
                kind="span.completed",
                data={},
            )
            return result
        finally:
            _direct_span.reset(token)

    async def invoke_bound(
        self,
        bound: object,
        message: Message,
        context: Context,
    ) -> tuple[Message, Context]:
        """Bridge a pre-bound dependency into its managed executiontime.

        The direct Parent remains Binding-free.  The explicitly pre-bound
        Child is admitted as an independent managed Root in its own deployment
        domain, and cancellation of the direct Parent is propagated to that
        managed owner before the direct scope unwinds.
        """

        if not self._active:
            raise RuntimeError("the direct execution scope is already closed")
        start = getattr(bound, "start", None)
        if not callable(start):
            raise TypeError("pre-bound Child has no managed start() entrypoint")
        handle = await start(message, context)
        result = getattr(handle, "result", None)
        cancel = getattr(handle, "cancel", None)
        if not callable(result) or not callable(cancel):
            raise TypeError("pre-bound Child start() returned an invalid ExecutionHandle")
        try:
            return _validate_result(await result())
        except asyncio.CancelledError:
            with suppress(BaseException):
                await asyncio.shield(cancel())
            raise

    async def emit_event(
        self,
        module: Module[Any, Any],
        kind: str,
        data: Mapping[str, JsonValue],
    ) -> None:
        if not self._active:
            raise RuntimeError("the direct execution scope is already closed")
        span = _direct_span.get()
        if span is None:
            raise RuntimeError("Module event has no active span")
        await self._record.emit(
            span_id=span[0],
            parent_span_id=span[1],
            module_path=span[2],
            kind=kind,
            data=data,
        )

    async def wait_external(
        self,
        *,
        kind: str,
        key: str,
        request: Mapping[str, JsonValue],
        timeout: float | None,
    ) -> Mapping[str, JsonValue]:
        raise DirectExecutionError(
            "wait_external() requires a managed executiontime execution"
        )

    @asynccontextmanager
    async def model_permit(
        self,
        resource_key: str | None = None,
        *,
        max_concurrency: int | None = None,
    ) -> AsyncIterator[CapacityPermit]:
        permit = CapacityPermit(owner_key=resource_key)
        token = _capacity_permit.set(permit)
        try:
            yield permit
        finally:
            _capacity_permit.reset(token)

    @asynccontextmanager
    async def tool_permit(
        self, resource_key: str | None = None
    ) -> AsyncIterator[CapacityPermit]:
        permit = CapacityPermit(owner_key=resource_key)
        token = _capacity_permit.set(permit)
        try:
            yield permit
        finally:
            _capacity_permit.reset(token)

    async def execute_effect(
        self,
        *,
        spec: EffectSpec,
        request: Mapping[str, JsonValue],
        operation: Callable[[], Awaitable[JsonValue]],
    ) -> EffectOutcome[JsonValue]:
        if not isinstance(spec, EffectSpec):
            raise TypeError("spec must be an EffectSpec")
        value = freeze_json(await operation())
        span = _direct_span.get()
        effect_id = f"{self.execution_id}:effect:{uuid4()}"
        if span is not None:
            await self._record.emit(
                span_id=span[0],
                parent_span_id=span[1],
                module_path=span[2],
                kind="effect.executed",
                data={"effect_id": effect_id, "attempt": 1},
            )
        return EffectOutcome(
            value=value,
            disposition=EffectDisposition.EXECUTED,
            effect_id=effect_id,
            attempt=1,
        )

    async def submit_tool_task(self, spec: Any, call: Any) -> None:
        return None

    def resolve_model_invoker(self, model_group: str) -> object:
        raise DirectExecutionError(
            "ModelCallLayer has no local ModelInvoker for direct execution"
        )

    def model_call_options(self, model_group: str) -> Mapping[str, JsonValue]:
        value = self._record.options.model_calls.get(model_group)
        return value if isinstance(value, FrozenJsonObject) else FrozenJsonObject()

    def resolve_model_deployment(self, model_group: str) -> object:
        raise DirectExecutionError(
            "deferred ModelGroup requires a managed Runtime admission"
        )

    @asynccontextmanager
    async def model_deployment_lease(self, deployment: object) -> AsyncIterator[object]:
        del deployment
        raise DirectExecutionError(
            "deferred ModelGroup requires a managed Runtime admission"
        )
        yield object()  # pragma: no cover

    def resolve_tool_registry(self) -> object:
        return None

    def tool_idempotency_key(self, call_id: str) -> None:
        return None

    async def gather(
        self, operations: tuple[Callable[[], Awaitable[Any]], ...]
    ) -> tuple[Any, ...]:
        return tuple(await asyncio.gather(*(operation() for operation in operations)))


def _validate_result(result: object) -> tuple[Message, Context]:
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("Module.forward() must return a (message, context) tuple")
    message, context = result
    if not isinstance(message, Message):
        raise TypeError("Module.forward() result message must be a Message")
    if not isinstance(context, Context):
        raise TypeError("Module.forward() result context must be a Context")
    return message, context


class _DirectExecutionSubscription:
    """Independent cursor over a direct execution journal."""

    def __init__(self, record: _DirectExecutionRecord[Any], after: int | None) -> None:
        self._record = record
        self._cursor = -1 if after is None else after
        self._claimed = False

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    def __aiter__(self) -> AsyncIterator[ExecutionEvent]:
        if self._claimed:
            raise RuntimeError("an ExecutionSubscription supports one iterator")
        self._claimed = True
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[ExecutionEvent]:
        self._record.ensure_started()
        while True:
            async with self._record.condition:
                while len(self._record.events) <= self._cursor + 1 and not self._record.terminal:
                    await self._record.condition.wait()
                available = tuple(self._record.events[self._cursor + 1 :])
                terminal = self._record.terminal
            for event in available:
                self._cursor = event.sequence
                yield event
            if terminal and len(self._record.events) <= self._cursor + 1:
                return


class _DirectExecutionHandle(Generic[OutputMessageT]):
    """Control plane for one direct execution owner."""

    def __init__(self, record: _DirectExecutionRecord[OutputMessageT]) -> None:
        self._record = record

    @property
    def execution_id(self) -> str:
        return self._record.execution_id

    @property
    def trace_id(self) -> str:
        return self._record.trace_id

    @property
    def status(self) -> ExecutionStatus:
        return self._record.status

    async def result(self) -> tuple[OutputMessageT, Context]:
        self._record.ensure_started()
        assert self._record.task is not None
        return await self._record.task

    async def cancel(self) -> bool:
        self._record.ensure_started()
        task = self._record.task
        if task is None or task.done():
            return False
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        return True

    def subscribe(self, *, after: int | None = None) -> _DirectExecutionSubscription:
        if after is not None and (isinstance(after, bool) or after < -1):
            raise ValueError("after must be a non-negative sequence or -1")
        return _DirectExecutionSubscription(self._record, after)


class DirectExecutionStream(Generic[OutputMessageT]):
    """Owned stream projection over one direct ExecutionHandle."""

    def __init__(self, handle: _DirectExecutionHandle[OutputMessageT]) -> None:
        self._handle = handle
        self._subscription = handle.subscribe()
        self._result_consumed = False
        self._closed = False

    async def __aenter__(self) -> DirectExecutionStream[OutputMessageT]:
        self._handle._record.ensure_started()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._closed = True
        if not self._result_consumed and not self._handle.status.terminal:
            await self._handle.cancel()

    def __aiter__(self) -> AsyncIterator[ExecutionEvent]:
        if self._closed:
            raise RuntimeError("the execution stream is closed")
        return self._subscription.__aiter__()

    async def final_result(self) -> tuple[OutputMessageT, Context]:
        if self._closed:
            raise RuntimeError("the execution stream is closed")
        try:
            return await self._handle.result()
        finally:
            self._result_consumed = True


def _direct_module_paths(root: Module[Any, Any]) -> dict[int, str]:
    paths: dict[int, str] = {id(root): "root"}
    pending: list[tuple[str, Module[Any, Any]]] = [("root", root)]
    while pending:
        parent_path, parent = pending.pop(0)
        for name, child in parent.named_children():
            if id(child) in paths:
                continue
            path = f"{parent_path}.{name}"
            paths[id(child)] = path
            pending.append((path, child))
    return paths


async def _run_direct_execution(
    record: _DirectExecutionRecord[OutputMessageT],
) -> tuple[OutputMessageT, Context]:
    record.module._freeze_definition()
    scope = _DirectExecutionScope(record)
    token = _execution_scope.set(scope)
    record.status = ExecutionStatus.RUNNING
    await record.emit(
        span_id=record.root_span_id,
        parent_span_id=None,
        module_path="root",
        kind="execution.started",
        data={},
    )
    try:
        message, context = await scope.invoke_module(
            record.module, record.message, record.context
        )
    except asyncio.CancelledError:
        record.status = ExecutionStatus.CANCELLED
        await record.emit(
            span_id=record.root_span_id,
            parent_span_id=None,
            module_path="root",
            kind="execution.cancelled",
            data={},
        )
        raise
    except BaseException as exc:
        record.status = ExecutionStatus.FAILED
        await record.emit(
            span_id=record.root_span_id,
            parent_span_id=None,
            module_path="root",
            kind="execution.failed",
            data={"error_type": type(exc).__name__, "message": str(exc)},
        )
        raise
    else:
        record.status = ExecutionStatus.SUCCEEDED
        await record.emit(
            span_id=record.root_span_id,
            parent_span_id=None,
            module_path="root",
            kind="execution.completed",
            data={},
        )
        return cast(OutputMessageT, message), context
    finally:
        scope.close()
        _execution_scope.reset(token)
        await record.notify_terminal()


class PlacementMode(str, Enum):
    """Stable physical-placement strategy for a declared Child target."""

    INHERIT = "inherit"
    PINNED = "pinned"
    ADAPTIVE = "adaptive"


@dataclass(frozen=True, slots=True)
class PlacementPolicy:
    mode: PlacementMode
    target_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, PlacementMode):
            object.__setattr__(self, "mode", PlacementMode(self.mode))
        if self.mode is PlacementMode.PINNED:
            if not isinstance(self.target_id, str) or not self.target_id:
                raise ValueError("pinned placement requires a stable target_id")
        elif self.target_id is not None:
            raise ValueError("only pinned placement accepts target_id")

    @classmethod
    def inherit(cls) -> PlacementPolicy:
        return cls(PlacementMode.INHERIT)

    @classmethod
    def pinned(cls, target_id: str) -> PlacementPolicy:
        return cls(PlacementMode.PINNED, target_id)

    @classmethod
    def adaptive(cls) -> PlacementPolicy:
        return cls(PlacementMode.ADAPTIVE)


@dataclass(frozen=True, slots=True, kw_only=True)
class RemoteModule(Generic[InputMessageT, OutputMessageT]):
    """Stable logical deployment reference resolved only by a managed executiontime."""

    binding_ref: str
    plan_id: str | None = None
    graph_hash: str | None = None
    required_capabilities: tuple[str, ...] = ()
    placement: PlacementPolicy = field(default_factory=PlacementPolicy.adaptive)

    def __post_init__(self) -> None:
        if not isinstance(self.binding_ref, str) or not self.binding_ref:
            raise ValueError("binding_ref must be a non-empty string")
        if (self.plan_id is None) != (self.graph_hash is None):
            raise ValueError("plan_id and graph_hash must be declared together")
        if self.graph_hash is not None:
            if (
                len(self.graph_hash) != 64
                or any(character not in "0123456789abcdef" for character in self.graph_hash)
            ):
                raise ValueError("graph_hash must be a lowercase SHA-256 digest")
            if self.plan_id != f"sha256:{self.graph_hash}":
                raise ValueError("plan_id must identify the declared graph_hash")
        capabilities = tuple(self.required_capabilities)
        if any(not isinstance(item, str) or not item for item in capabilities):
            raise ValueError("required capabilities must be non-empty strings")
        if len(capabilities) != len(set(capabilities)):
            raise ValueError("required capabilities must be unique")
        object.__setattr__(self, "required_capabilities", capabilities)
        if not isinstance(self.placement, PlacementPolicy):
            raise TypeError("placement must be a PlacementPolicy")
        if self.placement.mode is PlacementMode.INHERIT:
            raise ValueError("RemoteModule placement cannot be inherit")

    async def __call__(
        self, message: InputMessageT, context: Context
    ) -> tuple[OutputMessageT, Context]:
        scope = _execution_scope.get()
        if scope is None:
            raise RuntimeError(
                "RemoteModule calls require an active managed execution; "
                "use a bound Root entrypoint"
            )
        output, next_context = await scope.invoke_module(self, message, context)
        return cast(OutputMessageT, output), next_context


class Module(Generic[InputMessageT, OutputMessageT]):
    """Reusable definition whose subclasses implement one ``forward`` method."""

    _children: dict[str, Module[Any, Any]]
    _dependencies: dict[str, ModuleDependency[Any, Any]]
    _definition_frozen: bool
    _definition_config_snapshot: FrozenJsonObject | None
    execution_requirements = ExecutionRequirements()
    trusted_live_resource_attributes: tuple[str, ...] = ()

    def __init__(self) -> None:
        object.__setattr__(self, "_children", {})
        object.__setattr__(self, "_dependencies", {})
        object.__setattr__(self, "_definition_frozen", False)
        object.__setattr__(self, "_definition_config_snapshot", None)

    def __setattr__(self, name: str, value: Any) -> None:
        if self.__dict__.get("_definition_frozen", False):
            raise RuntimeError(
                f"{type(self).__name__} definition is frozen; Module attributes "
                "cannot be rebound after invoke(), stream(), or bind()"
            )
        children = self.__dict__.get("_children")
        dependencies = self.__dict__.get("_dependencies")
        if children is not None:
            if isinstance(value, Module):
                children[name] = value
            else:
                children.pop(name, None)
        if dependencies is not None:
            is_bound_dependency = (
                callable(value)
                and isinstance(getattr(value, "module", None), Module)
                and getattr(value, "binding", None) is not None
                and getattr(value, "runtime", None) is not None
            )
            if isinstance(value, (Module, RemoteModule)) or is_bound_dependency:
                dependencies[name] = cast(ModuleDependency[Any, Any], value)
            else:
                dependencies.pop(name, None)
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if self.__dict__.get("_definition_frozen", False):
            raise RuntimeError(
                f"{type(self).__name__} definition is frozen; Module attributes "
                "cannot be deleted after invoke(), stream(), or bind()"
            )
        children = self.__dict__.get("_children")
        dependencies = self.__dict__.get("_dependencies")
        if children is not None:
            children.pop(name, None)
        if dependencies is not None:
            dependencies.pop(name, None)
        object.__delattr__(self, name)

    @property
    def definition_frozen(self) -> bool:
        """Whether this definition has entered an executable lifecycle."""

        return self._definition_frozen

    def _freeze_definition(self) -> None:
        """Recursively close the raw Module graph against definition drift.

        Stored definition values are validated before any node is frozen.
        Mutable or unknown state is rejected unless its attribute is explicitly
        declared as a trusted deployment live resource.
        """

        discovered: list[Module[Any, Any]] = []
        visited: set[int] = set()
        pending: list[Module[Any, Any]] = [self]
        while pending:
            current = pending.pop()
            identity = id(current)
            if identity in visited:
                continue
            visited.add(identity)
            discovered.append(current)
            pending.extend(
                dependency
                for dependency in current._dependencies.values()
                if isinstance(dependency, Module)
            )

        # Validate the complete graph before changing any node, avoiding a
        # partially frozen definition when a later child is invalid.
        snapshots: list[tuple[Module[Any, Any], FrozenJsonObject]] = []
        for current in discovered:
            snapshot = freeze_json_object(module_definition_config(current))
            previous = current._definition_config_snapshot
            if previous is not None and previous != snapshot:
                raise RuntimeError(
                    f"{type(current).__name__} definition changed after freeze; "
                    "create a new Module definition"
                )
            snapshots.append((current, snapshot))
        for current, snapshot in snapshots:
            object.__setattr__(current, "_definition_config_snapshot", snapshot)
            object.__setattr__(current, "_definition_frozen", True)

    def named_children(self) -> tuple[tuple[str, Module[Any, Any]], ...]:
        return tuple(self._children.items())

    def named_dependencies(
        self,
    ) -> tuple[tuple[str, ModuleDependency[Any, Any]], ...]:
        """Return declared raw, pre-bound, and remote Child dependencies."""

        return tuple(self._dependencies.items())

    async def __call__(
        self, message: InputMessageT, context: Context
    ) -> tuple[OutputMessageT, Context]:
        scope = _execution_scope.get()
        if scope is None:
            raise RuntimeError(
                "Module calls require an active execution scope; "
                "use module.invoke() or module.stream() for a Root call"
            )
        output, next_context = await scope.invoke_module(self, message, context)
        return cast(OutputMessageT, output), next_context

    async def invoke(
        self,
        message: InputMessageT,
        context: Context,
        *,
        execution: ExecutionOptions | None = None,
    ) -> tuple[OutputMessageT, Context]:
        """Return the result projection of one direct execution."""

        handle = await self.start(message, context, execution=execution)
        return await handle.result()

    def _start_direct(
        self,
        message: InputMessageT,
        context: Context,
        execution: ExecutionOptions | None,
        *,
        start_now: bool = True,
    ) -> _DirectExecutionHandle[OutputMessageT]:
        """Create the direct owner synchronously for stream() and start()."""

        if _execution_scope.get() is not None:
            raise RuntimeError(
                "start() cannot create a Root inside an execution scope; "
                "call the Module directly to create a Child"
            )
        self._freeze_definition()
        options = execution or ExecutionOptions()
        if options.model_calls:
            declared_model_groups: set[str] = set()
            pending: list[Module[Any, Any]] = [self]
            visited: set[int] = set()
            while pending:
                current = pending.pop()
                if id(current) in visited:
                    continue
                visited.add(id(current))
                model_group = getattr(current, "model_group", None)
                group_name = getattr(model_group, "name", None)
                if isinstance(group_name, str) and group_name:
                    declared_model_groups.add(group_name)
                pending.extend(
                    dependency
                    for dependency in current._dependencies.values()
                    if isinstance(dependency, Module)
                )
            unknown = set(options.model_calls) - declared_model_groups
            if unknown:
                raise DirectExecutionError(
                    "model_calls references undeclared groups: "
                    + ", ".join(sorted(unknown))
                )
        execution_id = options.execution_id or f"direct-{uuid4()}"
        trace_id = options.trace_id or str(uuid4())
        record = _DirectExecutionRecord(
            execution_id=execution_id,
            trace_id=trace_id,
            root_span_id=str(uuid4()),
            module=self,
            message=message,
            context=context,
            options=options,
            module_paths=_direct_module_paths(self),
        )
        if start_now:
            record.ensure_started()
        return _DirectExecutionHandle(record)

    async def start(
        self,
        message: InputMessageT,
        context: Context,
        *,
        execution: ExecutionOptions | None = None,
    ) -> _DirectExecutionHandle[OutputMessageT]:
        """Start one unbound Module execution and return its control plane."""

        return self._start_direct(message, context, execution)

    def stream(
        self,
        message: InputMessageT,
        context: Context,
        *,
        execution: ExecutionOptions | None = None,
    ) -> DirectExecutionStream[OutputMessageT]:
        """Return an owned event projection over one direct execution."""

        return DirectExecutionStream(
            self._start_direct(message, context, execution, start_now=False)
        )

    async def forward(
        self, message: InputMessageT, context: Context
    ) -> tuple[OutputMessageT, Context]:
        raise NotImplementedError

    async def emit(self, *, kind: str, data: JsonObjectInput) -> None:
        """Publish custom content through the current execution channel."""

        scope = _execution_scope.get()
        if scope is None:
            raise RuntimeError("Module events require a bound Runtime execution")
        await scope.emit_event(self, kind, freeze_json_object(data))

    async def wait_external(
        self,
        *,
        kind: str,
        key: str,
        request: JsonObjectInput = (),
        timeout: float | None = None,
    ) -> FrozenJsonObject:
        """Wait for one bounded external signal in a managed execution."""

        scope = _execution_scope.get()
        if scope is None:
            raise RuntimeError("wait_external() requires an active execution scope")
        result = await scope.wait_external(
            kind=kind,
            key=key,
            request=freeze_json_object(request),
            timeout=timeout,
        )
        return freeze_json_object(result)

    async def gather(self, *awaitables: Awaitable[Any]) -> tuple[Any, ...]:
        """Await a structured parallel group in stable input order."""

        scope = _execution_scope.get()
        if scope is None:
            for awaitable in awaitables:
                close = getattr(awaitable, "close", None)
                if callable(close):
                    close()
            raise RuntimeError("gather() requires an active execution scope")
        operations = tuple(
            (lambda awaitable=awaitable: awaitable) for awaitable in awaitables
        )
        return await scope.gather(operations)

    def bind(
        self, runtime: Runtime, *, binding: Binding | None = None
    ) -> BoundModule[InputMessageT, OutputMessageT]:
        self._freeze_definition()
        return runtime.bind(self, binding=binding)


class Agent(
    Module[InputMessageT, OutputMessageT],
    Generic[InputMessageT, OutputMessageT],
):
    """Optional semantic name with exactly the same contract as Module."""


__all__ = [
    "Agent",
    "CapacityPermit",
    "DirectExecutionStream",
    "EffectIdempotency",
    "EffectRecoveryUnknown",
    "EffectRetryPolicy",
    "EffectSafety",
    "EffectSideEffect",
    "EffectSpec",
    "ExecutionRequirements",
    "ExecutionScope",
    "Infrastructure",
    "Module",
    "ModuleDependency",
    "PlacementMode",
    "PlacementPolicy",
    "RecoverySafety",
    "RemoteModule",
    "active_infrastructure",
    "current_capacity_permit",
    "current_infrastructure",
    "independent_execution",
]
