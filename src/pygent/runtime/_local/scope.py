"""Managed execution scope for child calls, resources, and effects."""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from pygent.core import (
    CapacityPermit,
    Context,
    EffectDisposition,
    EffectOutcome,
    EffectRecoveryUnknown,
    EffectRetryPolicy,
    EffectSpec,
    JsonValue,
    Message,
    Module,
    ModuleDependency,
    RemoteModule,
    freeze_json,
)
from pygent.core._direct_execution import _validate_result
from pygent.core._module_contracts import ExecutionScope, _capacity_permit
from pygent.tool import (
    SandboxExecutorSupport,
    ToolCall,
    ToolExecutionError,
    ToolSpec,
    ToolTaskAdmission,
)
from pygent.tool.executors import validate_executor_sandbox

from ..api import (
    CapacityPolicy,
    ExecutionAdmissionError,
    ExecutionOptions,
    ExecutionStatus,
    ResourceCapacityGate,
    RuntimeClosedError,
)
from .capacity import _BindingState
from .handles import _LocalBoundModule
from .policies import _collect_dependency_paths
from .state import _execution_frame, _ExecutionFrame, _ExecutionRecord, _module_stack


class _ManagedScope(ExecutionScope):
    def __init__(self, runtime: Any, record: _ExecutionRecord) -> None:
        self.runtime = runtime
        self.record = record
        self._paths = {id(module): path for path, module in record.graph.items()}
        self._parallel_tasks: set[asyncio.Task[Any]] = set()
        self._dependency_paths = _collect_dependency_paths(record.graph["root"])

    @property
    def deadline(self) -> float | None:
        return self.record.deadline

    @property
    def managed_execution_id(self) -> str:
        return self.record.execution_id

    def _path_for(self, module: object) -> str:
        return self._paths.get(
            id(module),
            f"dynamic:{type(module).__module__}.{type(module).__qualname__}",
        )

    def _require_owner_task(self) -> None:
        current = asyncio.current_task()
        if current is not self.record.task and current not in self._parallel_tasks:
            raise ExecutionAdmissionError(
                "Module Child calls from an unregistered asyncio Task are not "
                "structured; use the Runtime parallel Child API"
            )

    async def gather(
        self, operations: tuple[Callable[[], Awaitable[Any]], ...]
    ) -> tuple[Any, ...]:
        """Run a bounded structured parallel group under the current lineage."""

        self._require_owner_task()
        frame = _execution_frame.get()
        if frame is None:
            raise RuntimeError("managed parallel group has no execution frame")
        if not operations:
            return ()
        parent_had_lease = frame.runnable_held
        if parent_had_lease:
            self._release_runnable(frame)
        self.record.status = ExecutionStatus.WAITING_CHILD

        async def run(operation: Callable[[], Awaitable[Any]]) -> Any:
            return await operation()

        tasks = tuple(
            asyncio.create_task(run(operation), name="pygent-parallel-child")
            for operation in operations
        )
        self._parallel_tasks.update(tasks)
        try:
            return tuple(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            self._parallel_tasks.difference_update(tasks)
            current = asyncio.current_task()
            if parent_had_lease and (current is None or current.cancelling() == 0):
                self.record.status = ExecutionStatus.WAITING_RESUME
                await self._resume_runnable(frame)
                self.record.status = ExecutionStatus.RUNNING

    async def wait_handle(self, task: asyncio.Task[Any]) -> Any:
        """Wait for an independent execution without holding this flow's lease."""

        self._require_owner_task()
        frame = _execution_frame.get()
        if frame is None:  # pragma: no cover - managed scope invariant
            raise RuntimeError("managed Handle wait has no execution frame")
        if task is asyncio.current_task():
            raise RuntimeError("a managed execution cannot wait for its own Handle")
        if task.done():
            return await task
        parent_had_lease = frame.runnable_held
        if parent_had_lease:
            self._release_runnable(frame)
        self.record.status = ExecutionStatus.WAITING_CHILD
        try:
            return await task
        finally:
            current = asyncio.current_task()
            if parent_had_lease and (current is None or current.cancelling() == 0):
                self.record.status = ExecutionStatus.WAITING_RESUME
                await self._resume_runnable(frame)
                self.record.status = ExecutionStatus.RUNNING

    async def invoke_module(
        self,
        module: ModuleDependency[Any, Any],
        message: Message,
        context: Context,
    ) -> tuple[Message, Context]:
        self._require_owner_task()
        stack = _module_stack.get()
        capacity = self.record.binding_state.policy.execution_capacity
        if len(stack) > capacity.max_child_depth:
            raise ExecutionAdmissionError("maximum managed Child depth exceeded")
        if stack:
            if self.record.child_calls >= capacity.max_children_per_execution:
                raise ExecutionAdmissionError("maximum managed Child fan-out exceeded")
            self.record.child_calls += 1
        if isinstance(module, RemoteModule):
            try:
                bound = self.runtime._remote_modules[module.binding_ref]
            except KeyError as exc:
                raise RuntimeError(
                    f"unresolved RemoteModule binding_ref {module.binding_ref!r}"
                ) from exc
            if isinstance(bound, _LocalBoundModule):
                return await self.invoke_bound(bound, message, context)
            invoke = getattr(bound, "invoke", None)
            if invoke is None:
                raise TypeError("registered remote target has no invoke method")
            return await self._invoke_remote_child(
                module, bound, invoke, message, context
            )
        if not isinstance(module, Module):
            raise TypeError("managed execution can only invoke Module dependencies")
        path = self._path_for(module)
        if stack:
            frame = _execution_frame.get()
            if frame is None:  # pragma: no cover - managed scope invariant
                raise RuntimeError("managed Child has no active execution frame")
            return await self._invoke_local_child(
                module,
                message,
                context,
                runtime=frame.runtime,
                binding_state=frame.binding_state,
                path=path,
                occurrence=self._next_module_occurrence(path),
            )
        token = _module_stack.set(stack + (path,))
        try:
            return _validate_result(await module.forward(message, context))
        finally:
            _module_stack.reset(token)

    async def invoke_bound(
        self,
        bound: _LocalBoundModule[Any, Any],
        message: Message,
        context: Context,
    ) -> tuple[Message, Context]:
        self._require_owner_task()
        stack = _module_stack.get()
        capacity = self.record.binding_state.policy.execution_capacity
        if len(stack) > capacity.max_child_depth:
            raise ExecutionAdmissionError("maximum managed Child depth exceeded")
        if self.record.child_calls >= capacity.max_children_per_execution:
            raise ExecutionAdmissionError("maximum managed Child fan-out exceeded")
        self.record.child_calls += 1
        if bound.runtime._closed:
            raise RuntimeClosedError("Child Runtime is closed")
        path = self._dependency_paths.get(id(bound), self._path_for(bound.module))
        return await self._invoke_local_child(
            bound.module,
            message,
            context,
            runtime=bound.runtime,
            binding_state=bound.runtime._state_for(bound.binding),
            path=path,
            occurrence=self._next_module_occurrence(path),
            bound=bound,
        )

    def _next_module_occurrence(self, path: str) -> int:
        occurrence = self.record.module_calls.get(path, 0)
        self.record.module_calls[path] = occurrence + 1
        return occurrence

    async def _invoke_local_child(
        self,
        module: Module[Any, Any],
        message: Message,
        context: Context,
        *,
        runtime: Any,
        binding_state: _BindingState,
        path: str,
        occurrence: int,
        bound: _LocalBoundModule[Any, Any] | None = None,
    ) -> tuple[Message, Context]:
        parent = _execution_frame.get()
        if parent is None:  # pragma: no cover - managed scope invariant
            raise RuntimeError("managed Child has no Parent execution frame")
        child_execution_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"pygent:{self.record.execution_id}:{path}:{occurrence}:"
                f"{binding_state.deployment_scope_id}",
            )
        )
        child = _ExecutionFrame(
            execution_id=child_execution_id,
            parent_execution_id=parent.execution_id,
            span_id=str(uuid.uuid4()),
            parent_span_id=parent.span_id,
            module_path=path,
            module_occurrence=occurrence,
            runtime=runtime,
            binding_state=binding_state,
            deadline=parent.deadline,
            model_calls=parent.model_calls,
            model_admission=parent.model_admission,
        )
        separate_live_admission = binding_state is not parent.binding_state
        admitted = False
        child_model_admission = None
        parent_had_lease = parent.runnable_held
        if parent_had_lease:
            self._release_runnable(parent)
        self.record.status = ExecutionStatus.WAITING_CHILD
        try:
            if separate_live_admission:
                if bound is not None:
                    child.model_calls = ExecutionOptions().model_calls
                    child_model_admission = await runtime._prepare_model_admission(
                        bound,
                        ExecutionOptions(),
                        admission_id=child.execution_id,
                    )
                    child.model_admission = child_model_admission
                    if (
                        child_model_admission is not None
                        and self.record.history is not None
                    ):
                        await self.record.history.add_model_admission_ref(
                            self.record.execution_id,
                            child_model_admission.admission_id,
                        )
                await runtime._await_until(child.deadline, binding_state.admit())
                admitted = True
            await runtime._await_until(
                child.deadline,
                binding_state.runnable.acquire(),
            )
            child.runnable_held = True
            await self.record.emit(
                execution_id=child.execution_id,
                parent_execution_id=child.parent_execution_id,
                span_id=child.span_id,
                parent_span_id=child.parent_span_id,
                module_path=path,
                kind="span.started",
                data={},
            )
            frame_token = _execution_frame.set(child)
            stack = _module_stack.get()
            stack_token = _module_stack.set(stack + (path,))
            try:
                result = await runtime._await_until(
                    child.deadline, module.forward(message, context)
                )
                validated = _validate_result(result)
            except TimeoutError:
                await self.record.emit(
                    execution_id=child.execution_id,
                    parent_execution_id=child.parent_execution_id,
                    module_path=path,
                    kind="span.deadline_exceeded",
                    data={},
                )
                raise
            except asyncio.CancelledError:
                deadline_expired = self.record.deadline_fired
                await self.record.emit(
                    execution_id=child.execution_id,
                    parent_execution_id=child.parent_execution_id,
                    module_path=path,
                    kind=(
                        "span.deadline_exceeded"
                        if deadline_expired
                        else "span.cancelled"
                    ),
                    data={},
                )
                raise
            except BaseException as exc:
                await self.record.emit(
                    execution_id=child.execution_id,
                    parent_execution_id=child.parent_execution_id,
                    module_path=path,
                    kind="span.failed",
                    data={"error_type": type(exc).__name__, "message": str(exc)},
                )
                raise
            else:
                await self.record.emit(
                    execution_id=child.execution_id,
                    parent_execution_id=child.parent_execution_id,
                    module_path=path,
                    kind="span.completed",
                    data={},
                )
                return validated
            finally:
                _module_stack.reset(stack_token)
                _execution_frame.reset(frame_token)
        finally:
            task = asyncio.current_task()
            can_resume = task is None or task.cancelling() == 0
            if child.runnable_held:
                child.runnable_held = False
                binding_state.runnable.release()
            if admitted:
                await binding_state.release_live()
            if child_model_admission is not None:
                await runtime.model_deployment_store.release_admission(
                    child_model_admission.admission_id,
                    recoverable=self.record.history is not None,
                )
            if parent_had_lease and can_resume:
                self.record.status = ExecutionStatus.WAITING_RESUME
                await self._resume_runnable(parent)

    async def _invoke_remote_child(
        self,
        module: RemoteModule[Any, Any],
        bound: object,
        invoke: Callable[..., Awaitable[tuple[Message, Context]]],
        message: Message,
        context: Context,
    ) -> tuple[Message, Context]:
        parent = _execution_frame.get()
        if parent is None:  # pragma: no cover - managed scope invariant
            raise RuntimeError("managed remote Child has no Parent execution frame")
        child_execution_id = str(uuid.uuid4())
        child_span_id = str(uuid.uuid4())
        path = self._dependency_paths.get(
            id(module), f"{self.record.plan.root}.remote[{module.binding_ref}]"
        )
        parent_had_lease = parent.runnable_held
        if parent_had_lease:
            self._release_runnable(parent)
        self.record.status = ExecutionStatus.WAITING_CHILD
        await self.record.emit(
            execution_id=child_execution_id,
            parent_execution_id=parent.execution_id,
            span_id=child_span_id,
            parent_span_id=parent.span_id,
            module_path=path,
            kind="span.started",
            data={"remote": True, "binding_ref": module.binding_ref},
        )
        try:
            invoke_remote_child = getattr(bound, "invoke_remote_child", None)
            if callable(invoke_remote_child):
                async def relay_remote_event(event: Any) -> None:
                    payload = event.data.to_dict()
                    payload["origin_execution_id"] = event.execution_id
                    payload["origin_sequence"] = event.sequence
                    await self.record.emit(
                        execution_id=event.execution_id,
                        parent_execution_id=parent.execution_id,
                        span_id=event.span_id,
                        parent_span_id=event.parent_span_id,
                        event_id=event.event_id,
                        timestamp_unix_ns=event.timestamp_unix_ns,
                        module_path=path,
                        kind=event.kind,
                        data=payload,
                    )

                result = await invoke_remote_child(
                    message,
                    context,
                    plan_id=module.plan_id,
                    graph_hash=module.graph_hash,
                    required_capabilities=module.required_capabilities,
                    placement_mode=module.placement.mode.value,
                    pinned_target_id=module.placement.target_id,
                    deadline=self.record.deadline,
                    trace_id=self.record.trace_id,
                    parent_execution_id=parent.execution_id,
                    parent_span_id=child_span_id,
                    event_sink=relay_remote_event,
                    child_execution_id=child_execution_id,
                    attempt=self.record.attempt,
                    idempotency_key=(
                        f"{self.record.idempotency_key or self.record.execution_id}:"
                        f"remote:{module.binding_ref}:{self.record.child_calls}"
                    ),
                )
            else:
                result = await invoke(message, context, deadline=self.record.deadline)
            validated = _validate_result(result)
        except asyncio.CancelledError:
            await self.record.emit(
                execution_id=child_execution_id,
                parent_execution_id=parent.execution_id,
                span_id=child_span_id,
                parent_span_id=parent.span_id,
                module_path=path,
                kind="span.cancelled",
                data={"remote": True},
            )
            raise
        except BaseException as exc:
            await self.record.emit(
                execution_id=child_execution_id,
                parent_execution_id=parent.execution_id,
                span_id=child_span_id,
                parent_span_id=parent.span_id,
                module_path=path,
                kind="span.failed",
                data={"remote": True, "error_type": type(exc).__name__},
            )
            raise
        else:
            await self.record.emit(
                execution_id=child_execution_id,
                parent_execution_id=parent.execution_id,
                span_id=child_span_id,
                parent_span_id=parent.span_id,
                module_path=path,
                kind="span.completed",
                data={"remote": True},
            )
            return validated
        finally:
            task = asyncio.current_task()
            if parent_had_lease and (task is None or task.cancelling() == 0):
                self.record.status = ExecutionStatus.WAITING_RESUME
                await self._resume_runnable(parent)
                self.record.status = ExecutionStatus.RUNNING

    async def emit_event(
        self,
        module: Module[Any, Any],
        kind: str,
        data: Mapping[str, JsonValue],
    ) -> None:
        if not isinstance(kind, str) or not kind:
            raise ValueError("event kind must be a non-empty string")
        frame = _execution_frame.get()
        await self.record.emit(
            execution_id=frame.execution_id if frame is not None else None,
            parent_execution_id=(
                frame.parent_execution_id if frame is not None else self.record.parent_execution_id
            ),
            module_path=(
                frame.module_path if frame is not None else self._path_for(module)
            ),
            kind=kind,
            data=data,
        )

    @asynccontextmanager
    async def model_permit(
        self,
        resource_key: str | None = None,
        *,
        max_concurrency: int | None = None,
    ) -> AsyncIterator[CapacityPermit]:
        frame = _execution_frame.get()
        if frame is None:  # pragma: no cover - managed scope invariant
            raise RuntimeError("managed model wait has no execution frame")
        gates = [frame.binding_state.model]
        binding_policy = frame.binding_state.policy.model_capacity
        if (
            resource_key is not None
            and max_concurrency is not None
            and binding_policy.capacity_key != resource_key
        ):
            gates.append(
                frame.runtime._shared_resource_gate(
                    "model",
                    resource_key,
                    CapacityPolicy.limited(
                        max_concurrency=max_concurrency,
                        max_queue_size=binding_policy.max_queue_size or 0,
                    ),
                )
            )
        async with self._resource_permits(gates) as permit:
            yield permit

    @asynccontextmanager
    async def tool_permit(
        self, resource_key: str | None = None
    ) -> AsyncIterator[CapacityPermit]:
        frame = _execution_frame.get()
        if frame is None:  # pragma: no cover - managed scope invariant
            raise RuntimeError("managed tool wait has no execution frame")
        gates = frame.runtime._tool_gates(frame.binding_state, resource_key)
        async with self._resource_permits(gates) as permit:
            yield permit

    def resolve_model_invoker(self, model_group: str) -> object:
        frame = _execution_frame.get()
        runtime = self.runtime if frame is None else frame.runtime
        try:
            return runtime._model_invokers[model_group]
        except KeyError as exc:
            raise RuntimeError(
                f"no managed ModelInvoker registered for group {model_group!r}"
            ) from exc

    def model_call_options(self, model_group: str) -> Mapping[str, JsonValue]:
        frame = _execution_frame.get()
        calls = self.record.model_calls if frame is None else frame.model_calls
        if isinstance(calls, Mapping):
            value = calls.get(model_group)
            if isinstance(value, Mapping):
                return value
        return {}

    def resolve_model_deployment(self, model_group: str) -> object:
        frame = _execution_frame.get()
        admission = self.record.model_admission if frame is None else frame.model_admission
        if admission is None:
            raise ExecutionAdmissionError(
                f"model group {model_group!r} has no active admission"
            )
        return admission.snapshot(model_group)

    def model_deployment_lease(self, deployment: object) -> Any:
        frame = _execution_frame.get()
        runtime = self.runtime if frame is None else frame.runtime
        return runtime._model_deployment_lease(deployment)

    def resolve_tool_registry(self) -> object:
        frame = _execution_frame.get()
        runtime = self.runtime if frame is None else frame.runtime
        registry = runtime._tool_registry
        if registry is None:
            raise RuntimeError("no managed ExecutorRegistry is attached")
        return registry

    def tool_idempotency_key(self, call_id: str) -> str:
        frame = _execution_frame.get()
        module_path = frame.module_path if frame is not None else self.record.plan.root
        occurrence = frame.module_occurrence if frame is not None else 0
        return f"{self.record.execution_id}:{module_path}:{occurrence}:{call_id}"

    async def submit_tool_task(self, spec: ToolSpec, call: ToolCall) -> ToolTaskAdmission:
        """Submit a detached ToolTask through the active deployment Runtime."""

        frame = _execution_frame.get()
        if frame is None:  # pragma: no cover - managed scope invariant
            raise RuntimeError("managed ToolTask submission has no execution frame")
        manager = frame.runtime._tool_tasks
        registry = frame.runtime._tool_registry
        if manager is None or registry is None:
            return ToolTaskAdmission(
                error="detach requires a managed ToolTaskManager and ExecutorRegistry",
                error_kind="capability_error",
                error_code="detach_unavailable",
            )
        binding_state = frame.binding_state
        try:
            executor = registry.resolve(spec.tool_id, spec.version)
            support = validate_executor_sandbox(
                spec, executor, durable=self.record.history is not None
            )
        except ToolExecutionError as exc:
            return ToolTaskAdmission(
                error=str(exc),
                error_kind=exc.kind,
                error_code=exc.code,
                missing_capabilities=exc.missing_capabilities,
            )
        except LookupError:
            return ToolTaskAdmission(
                error=f"no executor registered for {spec.tool_id}@{spec.version}",
                error_kind="capability_error",
                error_code="executor_unavailable",
            )
        execute = frame.runtime._detached_tool_execution(binding_state, registry)
        if self.record.history is not None:
            prepare_job = getattr(manager, "prepare_job", None)
            if (
                not callable(prepare_job)
                or getattr(manager, "history", None) is not self.record.history
            ):
                return ToolTaskAdmission(
                    error="durable detach requires a matching durable ToolTaskManager",
                    error_kind="capability_error",
                    error_code="durability_unavailable",
                )
            capabilities = ["durability.sqlite"]
            if spec.sandbox_profile is not None:
                capabilities.append(f"tool.sandbox.{spec.sandbox_profile}")
                if isinstance(support, SandboxExecutorSupport):
                    fingerprint = support.capability_for_fingerprint()
                    if fingerprint is not None:
                        capabilities.append(fingerprint)
            required_capabilities = tuple(capabilities)
            if not set(required_capabilities) <= frame.runtime.capabilities:
                missing = tuple(sorted(set(required_capabilities) - frame.runtime.capabilities))
                return ToolTaskAdmission(
                    error="durable detach is missing required capabilities",
                    error_kind="capability_error",
                    error_code="missing_sandbox_capability",
                    missing_capabilities=missing,
                )
            stack = _module_stack.get()
            module_path = stack[-1] if stack else self.record.plan.root
            frame = _execution_frame.get()
            module_occurrence = frame.module_occurrence if frame is not None else 0
            logical_key = (
                "sha256:"
                + hashlib.sha256(
                    json.dumps(
                        {
                            "execution_id": self.record.execution_id,
                            "root": self.record.plan.root,
                            "module_path": module_path,
                            "module_occurrence": module_occurrence,
                            "call_id": call.call_id,
                            "idempotency_key": call.idempotency_key,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
            )
            task = await prepare_job(
                spec,
                call,
                logical_key=logical_key,
                binding_id=binding_state.policy.name,
                plan_id=self.record.plan.plan_id,
                required_capabilities=required_capabilities,
                execution=execute,
            )
        else:
            task = await manager.prepare(spec, call, execution=execute)
        self.record.deferred_tool_tasks.append((manager, task.task_id))
        return ToolTaskAdmission(task=task)

    @asynccontextmanager
    async def _resource_permits(
        self, gates: list[ResourceCapacityGate]
    ) -> AsyncIterator[CapacityPermit]:
        """Handoff execution capacity while an external resource owns execution."""

        frame = _execution_frame.get()
        if frame is None:  # pragma: no cover - managed scope invariant
            raise RuntimeError("managed resource wait has no execution frame")
        self._release_runnable(frame)
        try:
            unique_gates = tuple(dict.fromkeys(gates))
            async with AsyncExitStack() as stack:
                permits: list[CapacityPermit] = []
                for gate in unique_gates:
                    permits.append(await stack.enter_async_context(gate.permit()))
                permit = next(
                    (
                        item
                        for item in reversed(permits)
                        if item.fencing_token is not None
                    ),
                    permits[-1],
                )
                capacity_token = _capacity_permit.set(permit)
                try:
                    yield permit
                finally:
                    _capacity_permit.reset(capacity_token)
        finally:
            await self._resume_runnable(frame)

    def _release_runnable(self, frame: _ExecutionFrame) -> None:
        if frame.runnable_held:
            frame.runnable_held = False
            frame.binding_state.runnable.release()
            if frame.parent_execution_id is None:
                self.record.runnable_held = False

    async def _resume_runnable(self, frame: _ExecutionFrame) -> None:
        if not frame.runnable_held and not self.record.terminal:
            await frame.runtime._await_until(
                frame.deadline,
                frame.binding_state.runnable.acquire(resume=True),
            )
            frame.runnable_held = True
            if frame.parent_execution_id is None:
                self.record.runnable_held = True

    async def wait_external(
        self,
        *,
        kind: str,
        key: str,
        request: Mapping[str, JsonValue],
        timeout: float | None,
    ) -> Mapping[str, JsonValue]:
        frame = _execution_frame.get()
        if frame is None:  # pragma: no cover - managed scope invariant
            raise RuntimeError("managed external wait has no execution frame")
        self._release_runnable(frame)
        try:
            return await frame.runtime._wait_external(
                self.record,
                frame,
                kind=kind,
                key=key,
                request=request,
                timeout=timeout,
            )
        finally:
            await self._resume_runnable(frame)

    async def execute_effect(
        self,
        *,
        spec: EffectSpec,
        request: Mapping[str, JsonValue],
        operation: Callable[[], Awaitable[JsonValue]],
    ) -> EffectOutcome[JsonValue]:
        """Run or deterministically replay one provider/executor side effect."""

        if not isinstance(spec, EffectSpec):
            raise TypeError("spec must be an EffectSpec")
        stack = _module_stack.get()
        module_path = stack[-1] if stack else self.record.plan.root
        call_index = self.record.effect_calls.get(module_path, 0)
        self.record.effect_calls[module_path] = call_index + 1
        history = self.record.history
        disposition = EffectDisposition.EXECUTED
        if history is None:
            result = freeze_json(await operation())
        else:
            spec_value = {
                "side_effect": spec.side_effect.value,
                "idempotency": spec.idempotency.value,
                "retry_policy": spec.retry_policy.value,
                "idempotency_key": spec.idempotency_key,
            }
            stored, created = await history.begin_effect(
                execution_id=self.record.execution_id,
                module_path=module_path,
                call_index=call_index,
                effect_type=spec.effect_type,
                request=request,
                spec=spec_value,
            )
            if stored.status == "completed":
                if stored.result is None:  # pragma: no cover - store invariant
                    raise RuntimeError("completed effect has no result")
                result = stored.result
                disposition = EffectDisposition.REPLAYED
            elif stored.status == "unknown":
                raise EffectRecoveryUnknown(
                    f"effect {spec.effect_type!r} has an unknown committed state"
                )
            elif created or spec.retry_policy is EffectRetryPolicy.REPLAY_SAFE:
                if not created:
                    disposition = EffectDisposition.RETRIED
                result = freeze_json(await operation())
                await history.complete_effect(
                    execution_id=self.record.execution_id,
                    module_path=module_path,
                    call_index=call_index,
                    result=result,
                )
            else:
                await history.mark_effect_unknown(
                    execution_id=self.record.execution_id,
                    module_path=module_path,
                    call_index=call_index,
                )
                raise EffectRecoveryUnknown(
                    f"effect {spec.effect_type!r} may already have committed"
                )
        await self.record.emit(
            parent_execution_id=self.record.parent_execution_id,
            module_path=module_path,
            kind=f"effect.{disposition.value}",
            data={
                "effect_type": spec.effect_type,
                "call_index": call_index,
                "effect_id": f"{self.record.execution_id}:{module_path}:{call_index}",
                "attempt": self.record.attempt,
            },
        )
        return EffectOutcome(
            value=result,
            disposition=disposition,
            effect_id=f"{self.record.execution_id}:{module_path}:{call_index}",
            attempt=self.record.attempt,
        )


__all__ = ["_ManagedScope"]
