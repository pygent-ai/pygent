# mypy: disable-error-code="attr-defined"
"""Root execution scheduling, external waits, and shutdown lifecycle."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Mapping
from types import TracebackType
from typing import Any, Self, TypeVar

from pygent.core import Context, JsonValue, Message, Module, freeze_json_object
from pygent.core._module_contracts import _execution_scope
from pygent.llm import ModelCallLayer, ModelCallOptions, ModelProfileSelectionError

from ..api import (
    ExecutionAdmissionError,
    ExecutionDeadlineExceeded,
    ExecutionOptions,
    ExecutionStatus,
    ExternalWaitNotFound,
    ExternalWaitRejected,
    RuntimeClosedError,
)
from ..codec import invocation_to_dict
from .handles import _LocalBoundModule, _LocalExecutionHandle
from .policies import _apply_binding_policy, _finite_deadline_requirement
from .scope import _ManagedScope
from .state import _execution_frame, _ExecutionFrame, _ExecutionRecord

InputMessageT = TypeVar("InputMessageT", bound=Message)
OutputMessageT = TypeVar("OutputMessageT", bound=Message)


class _LifecycleMixin:
    _closed: bool

    async def start(
        self,
        bound: _LocalBoundModule[InputMessageT, OutputMessageT],
        message: InputMessageT,
        context: Context,
        *,
        execution: ExecutionOptions | None = None,
    ) -> _LocalExecutionHandle[OutputMessageT]:
        if self._closed:
            raise RuntimeClosedError("Runtime is closed")
        current_plan = _apply_binding_policy(
            self._compile_plan(bound.module), bound.binding
        )
        if current_plan.graph_hash != bound.plan.graph_hash:
            raise ExecutionAdmissionError(
                "Module definition changed after binding; create a new Binding "
                "snapshot before starting another execution"
            )
        if not isinstance(message, Message):
            raise TypeError("message must be a Message")
        if not isinstance(context, Context):
            raise TypeError("context must be a Context")
        options = execution or ExecutionOptions()
        deadline_requirement = _finite_deadline_requirement(bound.module)
        if deadline_requirement is not None and options.deadline is None:
            raise ExecutionAdmissionError(
                f"bound Module graph contains {type(deadline_requirement).__name__}, "
                "which requires a finite execution deadline"
            )
        request_id = options.request_id or str(uuid.uuid4())
        invocation = invocation_to_dict(message, context)
        execution_id = options.execution_id or str(uuid.uuid4())
        history = (
            self.history
            if "durability.sqlite" in bound.durability.effective_capabilities
            else None
        )
        has_deferred_models = any(
            isinstance(module, ModelCallLayer) and module.model_group.is_deferred
            for module in bound.graph.values()
        )
        if history is not None:
            stored, created = await history.begin_execution(
                execution_id=execution_id,
                request_id=request_id,
                plan_id=bound.plan.plan_id,
                input=invocation,
                binding_id=bound.binding.name,
                identity=options.identity or "",
                idempotency_key=options.idempotency_key,
                model_calls=options.model_calls,
                model_admission_status=(
                    "preparing" if has_deferred_models else "none"
                ),
            )
            if not created:
                return await self._recover_stored(
                    bound, stored, deadline=options.deadline
                )
        try:
            model_admission = await self._prepare_model_admission(
                bound, options, admission_id=execution_id
            )
            if history is not None and model_admission is not None:
                await history.commit_model_admission(
                    execution_id,
                    admission_id=model_admission.admission_id,
                    manifest_digest=model_admission.digest,
                )
        except BaseException:
            if history is not None and has_deferred_models:
                await history.abort_model_admission(execution_id)
            raise
        record = _ExecutionRecord(
            execution_id=execution_id,
            trace_id=options.trace_id or str(uuid.uuid4()),
            root_span_id=str(uuid.uuid4()),
            request_id=request_id,
            binding_state=self._state_for(bound.binding),
            plan=bound.plan,
            graph=bound.graph,
            deadline=options.deadline,
            history=history,
            idempotency_key=options.idempotency_key,
            model_calls=options.model_calls,
            model_admission=model_admission,
            parent_execution_id=options.parent_execution_id,
            parent_span_id=options.parent_span_id,
        )
        self._executions[execution_id] = record
        task = asyncio.create_task(
            self._run_root(record, bound.module, message, context),
            name=f"pygent-execution-{execution_id}",
        )
        record.task = task
        return _LocalExecutionHandle(record)

    async def _prepare_model_admission(
        self,
        bound: _LocalBoundModule[Any, Any],
        options: ExecutionOptions,
        *,
        admission_id: str,
    ) -> Any:
        layers: dict[str, ModelCallLayer] = {}
        for module in bound.graph.values():
            if not isinstance(module, ModelCallLayer):
                continue
            previous = layers.get(module.model_group.name)
            if previous is not None and (
                previous.model_group != module.model_group
                or previous.policy != module.policy
            ):
                raise ExecutionAdmissionError(
                    f"model group {module.model_group.name!r} has conflicting declarations"
                )
            layers[module.model_group.name] = module
        unknown = set(options.model_calls) - set(layers)
        if unknown:
            raise ExecutionAdmissionError(
                "model_calls references undeclared groups: " + ", ".join(sorted(unknown))
            )
        selections: dict[str, str | None] = {}
        for group_name, layer in layers.items():
            raw = options.model_calls.get(group_name)
            call_options = (
                ModelCallOptions()
                if raw is None
                else ModelCallOptions.from_dict(raw)  # type: ignore[arg-type]
            )
            if call_options.profile is not None:
                if not layer.model_group.is_deferred:
                    raise ExecutionAdmissionError(
                        f"fixed model group {group_name!r} cannot select a profile"
                    )
                if not layer.policy.allow_profile_override:
                    raise ModelProfileSelectionError(
                        f"model group {group_name!r} does not allow profile override"
                    )
            for field_name in ("temperature", "max_output_tokens"):
                if (
                    getattr(call_options, field_name) is not None
                    and field_name not in layer.policy.overridable_generation
                ):
                    raise ExecutionAdmissionError(
                        f"model group {group_name!r} does not allow {field_name} override"
                    )
            if layer.model_group.is_deferred:
                selections[group_name] = call_options.profile
        if not selections:
            return None
        await self._ensure_model_store_open()
        admission = await self.model_deployment_store.admit(
            bound.deployment_scope_id,
            tuple(sorted(selections)),
            selections,
            admission_id=admission_id,
        )
        if (
            "durability.sqlite" in bound.durability.effective_capabilities
            and any(snapshot.resources is None for _, snapshot in admission.snapshots)
        ):
            await self.model_deployment_store.release_admission(
                admission.admission_id, recoverable=False
            )
            raise ExecutionAdmissionError(
                "durable dynamic model execution requires reconstructable resources"
            )
        return admission

    async def _run_root(
        self,
        record: _ExecutionRecord,
        module: Module[Any, Any],
        message: Message,
        context: Context,
    ) -> tuple[Message, Context]:
        admitted = False
        deadline_timer = (
            None
            if record.deadline is None
            else asyncio.get_running_loop().call_at(
                record.deadline, setattr, record, "deadline_fired", True
            )
        )
        try:
            await self._await_with_deadline(record, record.binding_state.admit())
            admitted = True
            await self._await_with_deadline(
                record, record.binding_state.runnable.acquire()
            )
            record.runnable_held = True
            record.status = ExecutionStatus.RUNNING
            if record.history is not None:
                await record.history.update_execution(
                    record.execution_id, status=ExecutionStatus.RUNNING.value
                )
            await record.emit(
                parent_execution_id=record.parent_execution_id,
                module_path=record.plan.root,
                kind="execution.started",
                data={"request_id": record.request_id, "plan_id": record.plan.plan_id},
            )
            await record.emit(
                parent_execution_id=record.parent_execution_id,
                module_path=record.plan.root,
                kind="span.started",
                data={},
            )
            scope = _ManagedScope(self, record)
            token = _execution_scope.set(scope)
            root_frame = _ExecutionFrame(
                execution_id=record.execution_id,
                parent_execution_id=record.parent_execution_id,
                span_id=record.root_span_id,
                parent_span_id=record.parent_span_id,
                module_path=record.plan.root,
                module_occurrence=scope._next_module_occurrence(record.plan.root),
                runtime=self,
                binding_state=record.binding_state,
                deadline=record.deadline,
                runnable_held=True,
                model_calls=record.model_calls,
                model_admission=record.model_admission,
            )
            frame_token = _execution_frame.set(root_frame)
            try:
                result = await self._await_with_deadline(
                    record, scope.invoke_module(module, message, context)
                )
            finally:
                record.runnable_held = root_frame.runnable_held
                _execution_frame.reset(frame_token)
                _execution_scope.reset(token)
            output, next_context = result
            if not isinstance(output, Message) or not isinstance(next_context, Context):
                raise TypeError("Module.forward() must return (Message, Context)")
            record.status = ExecutionStatus.SUCCEEDED
            if record.history is not None:
                await record.history.update_execution(
                    record.execution_id,
                    status=ExecutionStatus.SUCCEEDED.value,
                    output=invocation_to_dict(output, next_context),
                )
            await record.emit(
                parent_execution_id=record.parent_execution_id,
                module_path=record.plan.root,
                kind="span.completed",
                data={},
            )
            await record.emit(
                parent_execution_id=record.parent_execution_id,
                module_path=record.plan.root,
                kind="execution.completed",
                data={},
            )
            return output, next_context
        except TimeoutError as exc:
            record.status = ExecutionStatus.DEADLINE_EXCEEDED
            if record.history is not None:
                await record.history.update_execution(
                    record.execution_id,
                    status=ExecutionStatus.DEADLINE_EXCEEDED.value,
                    error={"type": "deadline_exceeded"},
                )
            await record.emit(
                parent_execution_id=record.parent_execution_id,
                module_path=record.plan.root,
                kind="span.deadline_exceeded",
                data={},
            )
            await record.emit(
                parent_execution_id=record.parent_execution_id,
                module_path=record.plan.root,
                kind="execution.deadline_exceeded",
                data={},
            )
            raise ExecutionDeadlineExceeded(
                f"Execution {record.execution_id} exceeded its deadline"
            ) from exc
        except asyncio.CancelledError:
            record.status = ExecutionStatus.CANCELLED
            if record.history is not None:
                await record.history.update_execution(
                    record.execution_id,
                    status=ExecutionStatus.CANCELLED.value,
                    error={"type": "cancelled"},
                )
            await record.emit(
                parent_execution_id=record.parent_execution_id,
                module_path=record.plan.root,
                kind="span.cancelled",
                data={},
            )
            await record.emit(
                parent_execution_id=record.parent_execution_id,
                module_path=record.plan.root,
                kind="execution.cancelled",
                data={},
            )
            raise
        except BaseException as exc:
            record.status = ExecutionStatus.FAILED
            if record.history is not None:
                await record.history.update_execution(
                    record.execution_id,
                    status=ExecutionStatus.FAILED.value,
                    error={"type": type(exc).__name__, "message": str(exc)},
                )
            await record.emit(
                parent_execution_id=record.parent_execution_id,
                module_path=record.plan.root,
                kind="span.failed",
                data={"error_type": type(exc).__name__, "message": str(exc)},
            )
            await record.emit(
                parent_execution_id=record.parent_execution_id,
                module_path=record.plan.root,
                kind="execution.failed",
                data={"error_type": type(exc).__name__, "message": str(exc)},
            )
            raise
        finally:
            if deadline_timer is not None:
                deadline_timer.cancel()
            deferred, record.deferred_tool_tasks = record.deferred_tool_tasks, []
            for manager, task_id in deferred:
                await manager.start(task_id)
            if record.runnable_held:
                record.runnable_held = False
                record.binding_state.runnable.release()
            if admitted:
                await record.binding_state.release_live()
            if record.model_admission is not None:
                await self.model_deployment_store.release_admission(
                    record.model_admission.admission_id,
                    recoverable=record.history is not None,
                )
            await self._remove_waiters_for(record)
            await record.notify_terminal()

    async def _await_with_deadline(self, record: _ExecutionRecord, awaitable: Any) -> Any:
        return await self._await_until(record.deadline, awaitable)

    async def _await_until(self, deadline: float | None, awaitable: Any) -> Any:
        if deadline is None:
            return await awaitable
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            if hasattr(awaitable, "close"):
                awaitable.close()
            raise TimeoutError
        async with asyncio.timeout(remaining):
            return await awaitable

    async def _wait_external(
        self,
        record: _ExecutionRecord,
        frame: _ExecutionFrame,
        *,
        kind: str,
        key: str,
        request: Mapping[str, JsonValue],
        timeout: float | None,
    ) -> Mapping[str, JsonValue]:
        if not kind or not key:
            raise ExternalWaitRejected("external waiter kind and key are required")
        freeze_json_object(request)
        if timeout is not None and timeout <= 0:
            raise ExternalWaitRejected("external waiter timeout must be positive")
        deadline = frame.deadline
        local_deadline = time.monotonic() + timeout if timeout is not None else None
        if deadline is None and local_deadline is None:
            raise ExternalWaitRejected("external waiter requires a finite deadline")
        policy_deadline = (
            time.monotonic()
            + frame.binding_state.policy.execution_capacity.max_external_wait_seconds
        )
        effective = min(
            value
            for value in (deadline, local_deadline, policy_deadline)
            if value is not None
        )
        identity = (kind, key)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Mapping[str, JsonValue]] = loop.create_future()
        state = frame.binding_state
        try:
            waiter_context = state.execution.waiter_slot()
            async with waiter_context:
                async with self._external_lock:
                    if identity in self._external_waiters:
                        raise ExternalWaitRejected("external waiter already exists")
                    self._external_waiters[identity] = (record, future)
                old_status = record.status
                record.status = ExecutionStatus.WAITING_EXTERNAL
                try:
                    return await asyncio.wait_for(
                        future,
                        timeout=max(0.0, effective - time.monotonic()),
                    )
                finally:
                    if not record.terminal:
                        record.status = old_status
                    async with self._external_lock:
                        current = self._external_waiters.get(identity)
                        if current is not None and current[1] is future:
                            del self._external_waiters[identity]
        except ExecutionAdmissionError as exc:
            raise ExternalWaitRejected("external waiter capacity is full") from exc

    async def deliver_external(
        self,
        *,
        kind: str,
        key: str,
        value: Mapping[str, JsonValue],
    ) -> bool:
        frozen = freeze_json_object(value)
        async with self._external_lock:
            waiter = self._external_waiters.get((kind, key))
            if waiter is None:
                raise ExternalWaitNotFound(f"no waiter for {kind!r}/{key!r}")
            future = waiter[1]
            if future.done():
                raise ExternalWaitRejected("external waiter is already completed")
            future.set_result(frozen)
        return True

    async def _remove_waiters_for(self, record: _ExecutionRecord) -> None:
        async with self._external_lock:
            identities = [
                identity
                for identity, (owner, _) in self._external_waiters.items()
                if owner is record
            ]
            for identity in identities:
                _, future = self._external_waiters.pop(identity)
                if not future.done():
                    future.cancel()

    def get_execution(self, execution_id: str) -> _LocalExecutionHandle[Message]:
        try:
            return _LocalExecutionHandle(self._executions[execution_id])
        except KeyError as exc:
            raise KeyError(f"unknown execution {execution_id!r}") from exc

    async def purge_execution(self, execution_id: str) -> None:
        """Delete durable execution history and release its recoverable model pin."""

        active = self._executions.get(execution_id)
        if active is not None and not active.terminal:
            raise ExecutionAdmissionError("cannot purge an active execution")
        admission_id: str | None = None
        admission_ids: tuple[str, ...] = ()
        if self.history is not None:
            stored = await self.history.get_execution(execution_id)
            if stored is not None:
                admission_id = stored.model_admission_id
                admission_ids = await self.history.list_model_admission_refs(
                    execution_id
                )
                await self.history.delete_execution(execution_id)
        retained = set(admission_ids)
        if admission_id is not None:
            retained.add(admission_id)
        if retained:
            await self._ensure_model_store_open()
            for item in retained:
                await self.model_deployment_store.release_admission(
                    item, recoverable=False
                )
        self._executions.pop(execution_id, None)

    async def close(self, *, cancel: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        tasks = [record.task for record in self._executions.values() if record.task]
        pending = [task for task in tasks if task is not None and not task.done()]
        if cancel:
            for task in pending:
                task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if self._tool_tasks is not None:
            await self._tool_tasks.close(cancel=cancel)
        closed: set[int] = set()
        for invoker in self._model_invokers.values():
            if id(invoker) in closed:
                continue
            closed.add(id(invoker))
            aclose = getattr(invoker, "aclose", None)
            if callable(aclose):
                await aclose()
        for invoker, ownership in self._resident_model_invokers.values():
            if ownership.value != "owned" or id(invoker) in closed:
                continue
            closed.add(id(invoker))
            aclose = getattr(invoker, "aclose", None)
            if callable(aclose):
                await aclose()
        self._resident_model_invokers.clear()
        await self.model_deployment_store.close()

    async def __aenter__(self) -> Self:
        if self._closed:
            raise RuntimeClosedError("Runtime is closed")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()


__all__ = ["_LifecycleMixin"]
