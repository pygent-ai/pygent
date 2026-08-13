"""Local binding, execution handle, subscription, and stream adapters."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import TracebackType
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from pygent.core import Context, ExecutionFailure, JsonValue, Message, Module, thaw_json
from pygent.core._module_contracts import _execution_scope
from pygent.llm import ModelGroupConfig

from .._history_store import SQLiteHistoryStore
from ..api import (
    Binding,
    DurabilityReport,
    ExecutionEvent,
    ExecutionOptions,
    ExecutionOutcome,
    ExecutionOwnerState,
    ExecutionPhase,
    ExecutionSnapshot,
    ExecutionStatus,
)
from ..codec import invocation_from_dict
from ..context_codec import ContextCodecRegistry
from ..plan import ExecutionPlan
from .state import _ExecutionRecord

if TYPE_CHECKING:
    from ..local import LocalRuntime

InputMessageT = TypeVar("InputMessageT", bound=Message)
OutputMessageT = TypeVar("OutputMessageT", bound=Message)


class RuntimeBinding:
    """A Binding declaration associated with one Runtime instance."""

    __slots__ = ("deployment_scope_id", "model_groups", "policy", "runtime")

    def __init__(
        self, runtime: LocalRuntime, policy: Binding, deployment_scope_id: str
    ) -> None:
        self.runtime = runtime
        self.policy = policy
        self.deployment_scope_id = deployment_scope_id
        from .model_groups import ModelGroupCollection

        self.model_groups = ModelGroupCollection(runtime, deployment_scope_id)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.policy, name)

    def bind(
        self, module: Module[InputMessageT, OutputMessageT]
    ) -> _LocalBoundModule[InputMessageT, OutputMessageT]:
        return self.runtime.bind(module, binding=self.policy)


class _ExecutionSubscription:
    def __init__(self, record: _ExecutionRecord, after: int | None) -> None:
        self._record = record
        self._next = record.event_base_sequence if after is None else after + 1
        self._entered = False

    async def __aenter__(self) -> AsyncIterator[ExecutionEvent]:
        if self._entered:
            raise RuntimeError("execution subscription is already active")
        self._entered = True
        self._record.active_subscribers += 1
        return self._iterate()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._entered:
            self._entered = False
            self._record.active_subscribers -= 1

    async def _iterate(self) -> AsyncIterator[ExecutionEvent]:
        while True:
            while self._next <= self._record.committed_sequence:
                offset = self._next - self._record.event_base_sequence
                if offset < 0 or offset >= len(self._record.events):
                    break
                event = self._record.events[offset]
                self._next += 1
                yield event
            if (
                self._record.terminal_sequence is not None
                and self._next > self._record.terminal_sequence
            ):
                return
            async with self._record.event_condition:
                await self._record.event_condition.wait_for(
                    lambda: (
                        self._next <= self._record.committed_sequence
                        or (
                            self._record.terminal_sequence is not None
                            and self._next > self._record.terminal_sequence
                        )
                    )
                )


def _event_from_frozen(frozen: JsonValue) -> ExecutionEvent:
    item = thaw_json(frozen)
    if not isinstance(item, dict):
        raise TypeError("durable event must be a JSON object")
    return ExecutionEvent(
        schema_version=cast(str, item.get("schema_version")),
        event_id=cast(str, item.get("event_id")),
        execution_id=cast(str, item.get("execution_id")),
        attempt_id=cast(str, item.get("attempt_id")),
        trace_id=cast(str, item.get("trace_id")),
        span_id=cast(str, item.get("span_id")),
        parent_span_id=cast(str | None, item.get("parent_span_id")),
        sequence=cast(int, item.get("sequence")),
        timestamp_unix_ns=cast(int, item.get("timestamp_unix_ns")),
        module_path=cast(str, item.get("module_path")),
        kind=cast(str, item.get("kind")),
        data=cast(dict[str, JsonValue], item.get("data", {})),
    )


class _LocalExecutionHandle(Generic[OutputMessageT]):
    def __init__(self, record: _ExecutionRecord) -> None:
        self._record = record

    @property
    def execution_id(self) -> str:
        return self._record.execution_id

    @property
    def status(self) -> ExecutionStatus:
        return self._record.status

    @property
    def trace_id(self) -> str:
        return self._record.trace_id

    async def snapshot(self) -> ExecutionSnapshot:
        return self._record.snapshot()

    async def outcome(self) -> ExecutionOutcome:
        task = self._record.task
        if task is not None and not task.done():
            await asyncio.gather(task, return_exceptions=True)
        outcome = self._record.outcome
        if outcome is None:
            raise RuntimeError("execution has no terminal outcome")
        return outcome

    @property
    def request_id(self) -> str:
        return self._record.request_id

    @property
    def plan(self) -> ExecutionPlan:
        return self._record.plan

    async def result(self) -> tuple[OutputMessageT, Context]:
        task = self._record.task
        if task is None:  # pragma: no cover - construction invariant
            raise RuntimeError("Execution has no owner task")
        scope = _execution_scope.get()
        wait_handle = getattr(scope, "wait_handle", None)
        if callable(wait_handle):
            message, context = await wait_handle(task)
        else:
            message, context = await task
        return cast(OutputMessageT, message), context

    async def cancel(self) -> bool:
        task = self._record.task
        if task is None or task.done():
            return False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        if self._record.status is ExecutionStatus.PENDING:
            self._record.status = ExecutionStatus.CANCELLED
            await self._record.emit(
                parent_execution_id=self._record.parent_execution_id,
                module_path=self._record.plan.root,
                kind="execution.cancelled",
                data={},
            )
            await self._record.notify_terminal()
        return True

    def subscribe(self, *, after: int | None = None) -> _ExecutionSubscription:
        if after is not None and (isinstance(after, bool) or after < -1):
            raise ValueError("event cursor must be -1 or a non-negative integer")
        if after is not None and after + 1 < self._record.event_base_sequence:
            raise ValueError("event cursor is outside the retained execution segment")
        return _ExecutionSubscription(self._record, after)


class _DurableExecutionSubscription:
    def __init__(
        self, history: SQLiteHistoryStore, execution_id: str, after: int | None
    ) -> None:
        self._history = history
        self._execution_id = execution_id
        self._next = 0 if after is None else after + 1

    async def __aenter__(self) -> AsyncIterator[ExecutionEvent]:
        return self._iterate()

    async def __aexit__(self, *args: object) -> None:
        return None

    async def _iterate(self) -> AsyncIterator[ExecutionEvent]:
        while True:
            page = await self._history.events_after(
                execution_id=self._execution_id,
                after=self._next - 1,
                limit=256,
            )
            for frozen in page:
                event = _event_from_frozen(frozen)
                if event.sequence != self._next:
                    raise RuntimeError("durable event journal is not contiguous")
                self._next += 1
                yield event
            stored = await self._history.get_execution(self._execution_id)
            if stored is None:
                raise KeyError(f"unknown execution {self._execution_id!r}")
            if (
                stored.terminal_sequence is not None
                and self._next > stored.terminal_sequence
            ):
                return
            await asyncio.sleep(0.02)


class _DurableExecutionHandle(Generic[OutputMessageT]):
    """Task-independent attachment backed only by the durable journal."""

    def __init__(
        self,
        history: SQLiteHistoryStore,
        execution_id: str,
        context_codec_registry: ContextCodecRegistry,
    ) -> None:
        self._history = history
        self._execution_id = execution_id
        self._context_codec_registry = context_codec_registry

    @property
    def execution_id(self) -> str:
        return self._execution_id

    async def snapshot(self) -> ExecutionSnapshot:
        stored = await self._history.get_execution(self._execution_id)
        if stored is None:
            raise KeyError(f"unknown execution {self._execution_id!r}")
        status = ExecutionStatus(stored.status)
        phase = ExecutionPhase(stored.phase)
        events = await self._history.events_tail(execution_id=self._execution_id, limit=1)
        last_sequence = -1
        if events:
            value = thaw_json(events[0])
            if isinstance(value, dict):
                last_sequence = cast(int, value.get("sequence", -1))
        return ExecutionSnapshot(
            execution_id=stored.execution_id,
            trace_id=stored.trace_id,
            status=status,
            phase=phase,
            owner_state=(
                ExecutionOwnerState.TERMINAL
                if status.terminal
                else ExecutionOwnerState.UNOWNED
            ),
            attempt_id=stored.attempt_id,
            last_sequence=last_sequence,
            terminal_sequence=stored.terminal_sequence,
            submitted_at_unix_ns=stored.submitted_at_unix_ns,
            updated_at_unix_ns=stored.updated_at_unix_ns,
        )

    async def result(self) -> tuple[OutputMessageT, Context]:
        while True:
            stored = await self._history.get_execution(self._execution_id)
            if stored is None:
                raise KeyError(f"unknown execution {self._execution_id!r}")
            status = ExecutionStatus(stored.status)
            if status is ExecutionStatus.SUCCEEDED and stored.output is not None:
                message, context = invocation_from_dict(
                    stored.output, registry=self._context_codec_registry
                )
                return cast(OutputMessageT, message), context
            if status.terminal:
                raise RuntimeError(
                    f"execution {self._execution_id} ended with {status.value}: {stored.error!r}"
                )
            await asyncio.sleep(0.02)

    async def outcome(self) -> ExecutionOutcome:
        while True:
            stored = await self._history.get_execution(self._execution_id)
            if stored is None:
                raise KeyError(f"unknown execution {self._execution_id!r}")
            status = ExecutionStatus(stored.status)
            if status.terminal and stored.terminal_sequence is not None:
                if stored.attempt_id is None:
                    raise RuntimeError("terminal execution has no attempt identity")
                return ExecutionOutcome(
                    execution_id=stored.execution_id,
                    status=status,
                    attempt_id=stored.attempt_id,
                    terminal_sequence=stored.terminal_sequence,
                    error=(
                        ExecutionFailure.from_dict(thaw_json(stored.error))
                        if stored.error is not None
                        else None
                    ),
                )
            await asyncio.sleep(0.02)

    async def cancel(self) -> bool:
        return False

    def subscribe(self, *, after: int | None = None) -> _DurableExecutionSubscription:
        return _DurableExecutionSubscription(self._history, self._execution_id, after)


class _LocalExecutionStream(Generic[OutputMessageT]):
    def __init__(
        self,
        bound: _LocalBoundModule[Any, OutputMessageT],
        message: Message,
        context: Context,
        execution: ExecutionOptions | None,
    ) -> None:
        self._bound = bound
        self._message = message
        self._context = context
        self._options = execution
        self._handle: _LocalExecutionHandle[OutputMessageT] | None = None
        self._consumed_result = False

    async def _start(self) -> _LocalExecutionHandle[OutputMessageT]:
        if self._handle is None:
            self._handle = await self._bound.start(
                self._message, self._context, execution=self._options
            )
        return self._handle

    async def __aenter__(self) -> _LocalExecutionStream[OutputMessageT]:
        await self._start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._handle is not None and not self._consumed_result:
            await self._handle.cancel()

    def __aiter__(self) -> AsyncIterator[ExecutionEvent]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[ExecutionEvent]:
        handle = await self._start()
        async with handle.subscribe() as events:
            async for event in events:
                yield event

    async def final_result(self) -> tuple[OutputMessageT, Context]:
        handle = await self._start()
        result = await handle.result()
        self._consumed_result = True
        return result


class _LocalBoundModule(Generic[InputMessageT, OutputMessageT]):
    def __init__(
        self,
        runtime: LocalRuntime,
        module: Module[InputMessageT, OutputMessageT],
        binding: Binding,
        plan: ExecutionPlan,
        graph: dict[str, Module[Any, Any]],
        durability: DurabilityReport,
        deployment_scope_id: str,
    ) -> None:
        self.runtime = runtime
        self.module = module
        self.binding = binding
        self.plan = plan
        self.graph = graph
        self.durability = durability
        self.deployment_scope_id = deployment_scope_id
        from .model_groups import ModelGroupCollection

        requirements = {
            model_group.name: model_group
            for model_group in (
                getattr(item, "model_group", None) for item in graph.values()
            )
            if isinstance(model_group, ModelGroupConfig)
            and model_group.is_deferred
        }
        self.model_groups = ModelGroupCollection(
            runtime, deployment_scope_id, requirements
        )

    async def __call__(
        self, message: InputMessageT, context: Context
    ) -> tuple[OutputMessageT, Context]:
        scope = _execution_scope.get()
        if scope is None:
            raise RuntimeError(
                "BoundModule Child calls require an active execution scope; "
                "use invoke() for a root execution"
            )
        invoke_bound = getattr(scope, "invoke_bound", None)
        if invoke_bound is None:
            raise RuntimeError("active execution scope cannot invoke a BoundModule")
        output, next_context = await invoke_bound(self, message, context)
        return cast(OutputMessageT, output), next_context

    async def start(
        self,
        message: InputMessageT,
        context: Context,
        *,
        execution: ExecutionOptions | None = None,
    ) -> _LocalExecutionHandle[OutputMessageT]:
        return await self.runtime.start(self, message, context, execution=execution)

    async def invoke(
        self,
        message: InputMessageT,
        context: Context,
        *,
        execution: ExecutionOptions | None = None,
    ) -> tuple[OutputMessageT, Context]:
        handle = await self.start(message, context, execution=execution)
        return await handle.result()

    def stream(
        self,
        message: InputMessageT,
        context: Context,
        *,
        execution: ExecutionOptions | None = None,
    ) -> _LocalExecutionStream[OutputMessageT]:
        return _LocalExecutionStream(self, message, context, execution)


__all__ = [
    "RuntimeBinding",
    "_ExecutionSubscription",
    "_LocalBoundModule",
    "_LocalExecutionHandle",
    "_LocalExecutionStream",
]
