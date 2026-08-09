"""Direct execution state, scope, handles, and streams."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import (
    asynccontextmanager,
    suppress,
)
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Self,
    cast,
)
from uuid import uuid4

from ._module_contracts import (
    CapacityPermit,
    DirectExecutionError,
    EffectSpec,
    ModuleDependency,
    OutputMessageT,
    _capacity_permit,
    _execution_scope,
)
from .execution import (
    EXECUTION_EVENT_SCHEMA_VERSION,
    EffectDisposition,
    EffectOutcome,
    ExecutionEvent,
    ExecutionFailure,
    ExecutionOptions,
    ExecutionOutcome,
    ExecutionOwnerState,
    ExecutionPhase,
    ExecutionSnapshot,
    ExecutionStatus,
)
from .json_values import (
    FrozenJsonObject,
    JsonValue,
    freeze_json,
)
from .values import Context, Message

if TYPE_CHECKING:
    from ._module_definition import Module
_direct_span: ContextVar[tuple[str, str | None, str] | None] = ContextVar(
    "pygent_direct_span", default=None
)

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
    attempt_id: str = field(default_factory=lambda: str(uuid4()))
    status: ExecutionStatus = ExecutionStatus.PENDING
    phase: ExecutionPhase = ExecutionPhase.SUBMITTING
    submitted_at_unix_ns: int = field(default_factory=time.time_ns)
    updated_at_unix_ns: int = field(default_factory=time.time_ns)
    terminal_sequence: int | None = None
    outcome: ExecutionOutcome | None = None
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
                attempt_id=self.attempt_id,
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

    async def notify_terminal(
        self, status: ExecutionStatus, failure: ExecutionFailure | None = None
    ) -> None:
        async with self.condition:
            self.status = status
            self.phase = ExecutionPhase.TERMINAL
            self.updated_at_unix_ns = time.time_ns()
            self.terminal_sequence = self.events[-1].sequence
            self.outcome = ExecutionOutcome(
                execution_id=self.execution_id,
                status=self.status,
                attempt_id=self.attempt_id,
                terminal_sequence=self.terminal_sequence,
                error=failure,
            )
            self.condition.notify_all()

    def snapshot(self) -> ExecutionSnapshot:
        return ExecutionSnapshot(
            execution_id=self.execution_id,
            trace_id=self.trace_id,
            status=self.status,
            phase=self.phase,
            owner_state=(
                ExecutionOwnerState.TERMINAL
                if self.status.terminal
                else ExecutionOwnerState.ACTIVE
            ),
            attempt_id=self.attempt_id,
            last_sequence=len(self.events) - 1,
            terminal_sequence=self.terminal_sequence,
            submitted_at_unix_ns=self.submitted_at_unix_ns,
            updated_at_unix_ns=self.updated_at_unix_ns,
        )


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

    @property
    def managed_execution_id(self) -> None:
        return None

    async def invoke_module(
        self,
        module: ModuleDependency[Any, Any],
        message: Message,
        context: Context,
    ) -> tuple[Message, Context]:
        from ._module_definition import Module, RemoteModule

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
                while (
                    len(self._record.events) <= self._cursor + 1
                    and self._record.terminal_sequence is None
                ):
                    await self._record.condition.wait()
                available = tuple(self._record.events[self._cursor + 1 :])
                terminal_sequence = self._record.terminal_sequence
            for event in available:
                self._cursor = event.sequence
                yield event
            if terminal_sequence is not None and self._cursor >= terminal_sequence:
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

    async def snapshot(self) -> ExecutionSnapshot:
        return self._record.snapshot()

    async def outcome(self) -> ExecutionOutcome:
        self._record.ensure_started()
        assert self._record.task is not None
        await asyncio.gather(self._record.task, return_exceptions=True)
        if self._record.outcome is None:
            raise RuntimeError("execution has no terminal outcome")
        return self._record.outcome

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
    record.phase = ExecutionPhase.RUNNING
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
        await record.emit(
            span_id=record.root_span_id,
            parent_span_id=None,
            module_path="root",
            kind="execution.cancelled",
            data={},
        )
        await record.notify_terminal(
            ExecutionStatus.CANCELLED,
            ExecutionFailure(
                domain="runtime",
                kind="cancelled",
                message="execution was cancelled",
            ),
        )
        raise
    except BaseException as exc:
        await record.emit(
            span_id=record.root_span_id,
            parent_span_id=None,
            module_path="root",
            kind="execution.failed",
            data={"error_type": type(exc).__name__, "message": str(exc)},
        )
        await record.notify_terminal(
            ExecutionStatus.FAILED,
            ExecutionFailure(
                domain="runtime",
                kind=type(exc).__name__,
                message=str(exc) or type(exc).__name__,
            ),
        )
        raise
    else:
        await record.emit(
            span_id=record.root_span_id,
            parent_span_id=None,
            module_path="root",
            kind="execution.completed",
            data={},
        )
        await record.notify_terminal(ExecutionStatus.SUCCEEDED)
        return cast(OutputMessageT, message), context
    finally:
        scope.close()
        _execution_scope.reset(token)
