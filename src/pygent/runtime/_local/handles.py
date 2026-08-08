"""Local binding, execution handle, subscription, and stream adapters."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import TracebackType
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from pygent.core import Context, Message, Module
from pygent.core._module_contracts import _execution_scope
from pygent.llm import ModelGroupConfig

from ..api import (
    Binding,
    DurabilityReport,
    ExecutionEvent,
    ExecutionOptions,
    ExecutionStatus,
)
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
        self._next = 0 if after is None else after + 1

    async def __aenter__(self) -> AsyncIterator[ExecutionEvent]:
        return self._iterate()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    async def _iterate(self) -> AsyncIterator[ExecutionEvent]:
        while True:
            while self._next < len(self._record.events):
                event = self._record.events[self._next]
                self._next += 1
                yield event
            if self._record.terminal:
                return
            async with self._record.event_condition:
                await self._record.event_condition.wait_for(
                    lambda: (
                        self._next < len(self._record.events) or self._record.terminal
                    )
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
        if self._record.status is ExecutionStatus.QUEUED:
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
        return _ExecutionSubscription(self._record, after)


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
