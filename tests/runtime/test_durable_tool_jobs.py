from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import FrozenInstanceError

import pytest

from pygent import (
    AIMessage,
    Context,
    Module,
    ToolAuthorizationDecision,
    ToolCall,
    ToolCallLayer,
    ToolDefinition,
    ToolMessage,
    ToolSideEffect,
    ToolSpec,
)
from pygent.core import (
    EffectSafety,
    ExecutionRequirements,
    RecoverySafety,
    current_capacity_permit,
    thaw_json,
)
from pygent.runtime import (
    CapacityPolicy,
    CapacityScope,
    DurabilityMode,
    DurabilityPolicy,
    ExecutionAdmissionError,
    ExecutionCapacityPolicy,
    HistoryConflictError,
    JobRef,
    JobSnapshot,
    JobState,
    LocalRuntime,
    SQLiteCapacityCoordinator,
    SQLiteHistoryStore,
)
from pygent.runtime.codec import invocation_to_dict
from pygent.runtime.tasks import DurableToolTaskManager
from pygent.tool import (
    ExecutorRegistry,
    IdempotencyPolicy,
    LocalToolExecutor,
    SandboxExecutorSupport,
)


def _sandbox_executor(
    profile: str = "restricted", *, fingerprint: str = "sandbox:test"
) -> LocalToolExecutor:
    executor = LocalToolExecutor(lambda arguments: arguments)
    executor.sandbox_support = SandboxExecutorSupport(  # type: ignore[attr-defined]
        profiles=(profile,),
        durable_reconnect=True,
        deployment_fingerprint=fingerprint,
    )
    return executor


def _spec(
    *,
    version: str = "1",
    side_effect: ToolSideEffect = ToolSideEffect.PURE,
    idempotency: IdempotencyPolicy = IdempotencyPolicy.INHERENT,
    sandbox_profile: str | None = None,
) -> ToolSpec:
    return ToolSpec(
        tool_id="durable.echo",
        version=version,
        definition=ToolDefinition(
            name="durable_echo",
            description="durable echo",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            },
        ),
        side_effect=side_effect,
        idempotency=idempotency,
        resource_key="durable-backend",
        sandbox_profile=sandbox_profile,
    )


def _call(value: int, *, key: str | None = None) -> ToolCall:
    return ToolCall(
        call_id=f"call-{value}",
        name="durable_echo",
        arguments={"value": value},
        idempotency_key=key,
    )


class _AllowDetach(Module):
    execution_requirements = ExecutionRequirements(
        recovery_safety=RecoverySafety.MODULE_BOUNDARY_RETRY,
        effect_safety=EffectSafety.EFFECT_FREE,
    )

    async def forward(self, message, context):
        return (
            ToolAuthorizationDecision(
                call_id=message.call.call_id,
                allowed=True,
                reason_code="allowed",
                lifecycle="detach",
            ),
            context,
        )


class _TwoRoundDetach(Module):
    execution_requirements = ExecutionRequirements(
        recovery_safety=RecoverySafety.MODULE_BOUNDARY_RETRY,
        effect_safety=EffectSafety.EFFECT_FREE,
    )

    def __init__(self, spec: ToolSpec) -> None:
        super().__init__()
        self.tools = ToolCallLayer(tools=(spec,), authorization=_AllowDetach())

    async def forward(self, message, context):
        first, context = await self.tools(
            AIMessage(tool_calls=(message.tool_calls[0],)), context
        )
        second, context = await self.tools(
            AIMessage(tool_calls=(message.tool_calls[1],)), context
        )
        return ToolMessage(results=first.results + second.results), context


def _runtime_binding(runtime: LocalRuntime, *, name: str):
    return runtime.create_binding(
        name=name,
        execution_capacity=ExecutionCapacityPolicy(
            scope=CapacityScope.RUNTIME_INSTANCE,
            max_live_executions=4,
            max_runnable_executions=1,
            max_queue_size=2,
            max_waiters=4,
            max_child_depth=4,
            max_children_per_execution=8,
        ),
        model_capacity=CapacityPolicy.passthrough(),
        tool_capacity=CapacityPolicy.limited(
            max_concurrency=1,
            max_queue_size=2,
            capacity_key="durable-tool-account",
        ),
        durability=DurabilityPolicy(DurabilityMode.REQUIRED),
    )


def _binding(runtime: LocalRuntime, spec: ToolSpec, *, name: str = "jobs"):
    return _runtime_binding(runtime, name=name).bind(
        ToolCallLayer(
            tools=(spec,),
            authorization=_AllowDetach(),
        )
    )


def _logical_key(
    *,
    execution_id: str,
    root: str,
    module_path: str,
    module_occurrence: int,
    call: ToolCall,
) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(
            {
                "execution_id": execution_id,
                "root": root,
                "module_path": module_path,
                "module_occurrence": module_occurrence,
                "call_id": call.call_id,
                "idempotency_key": call.idempotency_key,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _attach(
    runtime: LocalRuntime,
    history: SQLiteHistoryStore,
    registry: ExecutorRegistry,
) -> DurableToolTaskManager:
    manager = DurableToolTaskManager(history, registry)
    runtime.attach_executor_registry(registry)
    runtime.attach_tool_task_manager(manager)
    return manager


async def _prepare_job(
    manager: DurableToolTaskManager,
    bound,
    spec: ToolSpec,
    call: ToolCall,
    *,
    capabilities: tuple[str, ...] = ("durability.sqlite",),
):
    async def unreachable(_spec, _call):
        raise AssertionError("a persisted Job must not retain this callback")

    return await manager.prepare_job(
        spec,
        call,
        logical_key=f"seed:{bound.binding.name}:{call.call_id}",
        binding_id=bound.binding.name,
        plan_id=bound.plan.plan_id,
        required_capabilities=capabilities,
        execution=unreachable,
    )


@pytest.mark.asyncio
async def test_managed_durable_detach_admits_job_and_stable_task_atomically(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    registry = ExecutorRegistry()
    spec = _spec()
    registry.register(
        spec.tool_id,
        spec.version,
        LocalToolExecutor(lambda arguments: arguments["value"] * 2),
    )
    async with SQLiteHistoryStore(path) as history:
        runtime = LocalRuntime(history=history)
        _attach(runtime, history, registry)
        bound = _binding(runtime, spec)
        message, _ = await bound.invoke(
            AIMessage(tool_calls=(_call(3),)),
            Context(tools=(spec.definition,)),
        )
        detached = message.results[0]
        assert detached.status == "detached" and detached.task is not None
        assert detached.task.job_id is not None
        job_ref = JobRef(detached.task.job_id, detached.task.task_id)
        final = await runtime.get_tool_result(job_ref.task_id, wait=True)
        job = await runtime.get_job(job_ref.job_id)
        stored = await history.get_job(job_ref.job_id)

        assert final is not None and final.output == 6
        assert final.task is not None and final.task.job_id == job_ref.job_id
        assert job is not None and job.state is JobState.SUCCEEDED
        assert job.ref == job_ref
        assert stored is not None and stored.task_id == job_ref.task_id
        assert stored.binding_id == bound.binding.name
        assert stored.plan_id == bound.plan.plan_id
        assert stored.resource_key == spec.resource_key
        persisted = thaw_json(stored.request)
        assert "execution" not in repr(persisted)
        assert "handler" not in repr(persisted)
        with pytest.raises(HistoryConflictError):
            await history.create_tool_job(
                job_id="job-conflicting-admission",
                task_id=stored.task_id,
                logical_key="conflicting-logical-key",
                binding_id=stored.binding_id,
                plan_id=stored.plan_id,
                resource_key=stored.resource_key,
                required_capabilities=stored.required_capabilities,
                request=stored.request,
            )
        assert await history.get_job("job-conflicting-admission") is None
        with pytest.raises(FrozenInstanceError):
            job.attempt = 2  # type: ignore[misc]
        await runtime.close()


@pytest.mark.parametrize("second_value", (1, 2))
@pytest.mark.asyncio
async def test_cross_round_reused_call_id_admits_independent_jobs(
    tmp_path, second_value: int
):
    path = tmp_path / f"cross-round-{second_value}.sqlite3"
    spec = _spec()
    calls = (
        ToolCall("reused", "durable_echo", {"value": 1}),
        ToolCall("reused", "durable_echo", {"value": second_value}),
    )
    executed: list[int] = []
    registry = ExecutorRegistry()
    registry.register(
        spec.tool_id,
        spec.version,
        LocalToolExecutor(
            lambda arguments: executed.append(arguments["value"])
            or arguments["value"]
        ),
    )
    async with SQLiteHistoryStore(path) as history:
        runtime = LocalRuntime(history=history)
        _attach(runtime, history, registry)
        bound = _runtime_binding(runtime, name="cross-round").bind(
            _TwoRoundDetach(spec)
        )
        message, _ = await bound.invoke(
            AIMessage(tool_calls=calls), Context(tools=(spec.definition,))
        )

        tasks = [result.task for result in message.results]
        assert all(task is not None for task in tasks)
        task_ids = [task.task_id for task in tasks if task is not None]
        job_ids = [task.job_id for task in tasks if task is not None]
        assert len(set(task_ids)) == len(set(job_ids)) == 2
        results = [
            await runtime.get_tool_result(task_id, wait=True) for task_id in task_ids
        ]
        assert [result.output for result in results if result is not None] == [
            1,
            second_value,
        ]
        assert executed == [1, second_value]
        jobs = await history.list_jobs(binding_id=bound.binding.name)
        assert len(jobs) == 2
        assert len({job.logical_key for job in jobs}) == 2
        await runtime.close()


@pytest.mark.asyncio
async def test_parent_recovery_reuses_job_committed_before_run_result(tmp_path):
    path = tmp_path / "parent-gap.sqlite3"
    execution_id = "parent-gap-run"
    spec = _spec()
    call = _call(9)
    registry = ExecutorRegistry()
    executions = 0

    def execute(arguments):
        nonlocal executions
        executions += 1
        return arguments["value"]

    registry.register(spec.tool_id, spec.version, LocalToolExecutor(execute))
    async with SQLiteHistoryStore(path) as history:
        runtime = LocalRuntime(history=history)
        manager = _attach(runtime, history, registry)
        bound = _binding(runtime, spec, name="parent-gap")
        message = AIMessage(tool_calls=(call,))
        context = Context(tools=(spec.definition,))
        await history.create_execution(
            execution_id=execution_id,
            request_id="parent-gap-request",
            plan_id=bound.plan.plan_id,
            input=invocation_to_dict(message, context),
            status="running",
        )
        logical_key = _logical_key(
            execution_id=execution_id,
            root=bound.plan.root,
            module_path=bound.plan.root,
            module_occurrence=0,
            call=call,
        )
        async def unreachable(_spec, _call):
            raise AssertionError("recovery must replace the process-local callback")

        admitted = await manager.prepare_job(
            spec,
            call,
            logical_key=logical_key,
            binding_id=bound.binding.name,
            plan_id=bound.plan.plan_id,
            required_capabilities=("durability.sqlite",),
            execution=unreachable,
        )

        recovered = await runtime.recover(bound, execution_id)
        output, _ = await recovered.result()
        detached = output.results[0]
        assert detached.task is not None
        assert detached.task.task_id == admitted.task_id
        jobs = await history.list_jobs(binding_id=bound.binding.name)
        assert len(jobs) == 1 and jobs[0].logical_key == logical_key
        result = await runtime.get_tool_result(admitted.task_id, wait=True)
        assert result is not None and result.status == "succeeded"
        assert executions == 1
        await runtime.close()


@pytest.mark.asyncio
async def test_parent_recovery_reuses_each_cross_round_occurrence(tmp_path):
    path = tmp_path / "cross-round-recovery.sqlite3"
    execution_id = "cross-round-recovery-run"
    spec = _spec()
    calls = (
        ToolCall("reused", "durable_echo", {"value": 1}),
        ToolCall("reused", "durable_echo", {"value": 2}),
    )
    executed: list[int] = []
    registry = ExecutorRegistry()
    registry.register(
        spec.tool_id,
        spec.version,
        LocalToolExecutor(
            lambda arguments: executed.append(arguments["value"])
            or arguments["value"]
        ),
    )
    async with SQLiteHistoryStore(path) as history:
        runtime = LocalRuntime(history=history)
        manager = _attach(runtime, history, registry)
        bound = _runtime_binding(runtime, name="cross-round-recovery").bind(
            _TwoRoundDetach(spec)
        )
        message = AIMessage(tool_calls=calls)
        context = Context(tools=(spec.definition,))
        await history.create_execution(
            execution_id=execution_id,
            request_id="cross-round-recovery-request",
            plan_id=bound.plan.plan_id,
            input=invocation_to_dict(message, context),
            status="running",
        )

        async def unreachable(_spec, _call):
            raise AssertionError("recovery must replace process-local execution")

        admitted = []
        for occurrence, call in enumerate(calls):
            task = await manager.prepare_job(
                spec,
                call,
                logical_key=_logical_key(
                    execution_id=execution_id,
                    root=bound.plan.root,
                    module_path="root.tools",
                    module_occurrence=occurrence,
                    call=call,
                ),
                binding_id=bound.binding.name,
                plan_id=bound.plan.plan_id,
                required_capabilities=("durability.sqlite",),
                execution=unreachable,
            )
            admitted.append(task)

        recovered = await runtime.recover(bound, execution_id)
        output, _ = await recovered.result()
        recovered_ids = [
            result.task.task_id
            for result in output.results
            if result.task is not None
        ]
        assert recovered_ids == [task.task_id for task in admitted]
        finals = [
            await runtime.get_tool_result(task.task_id, wait=True)
            for task in admitted
        ]
        assert [result.output for result in finals if result is not None] == [1, 2]
        assert executed == [1, 2]
        jobs = await history.list_jobs(binding_id=bound.binding.name)
        assert len(jobs) == 2
        await runtime.close()


@pytest.mark.asyncio
async def test_recovery_reenters_binding_tool_capacity_and_cleans_cancelled_queue(
    tmp_path,
):
    path = tmp_path / "capacity.sqlite3"
    spec = _spec()
    seed_registry = ExecutorRegistry()
    seed_registry.register(
        spec.tool_id, spec.version, LocalToolExecutor(lambda arguments: arguments)
    )
    async with SQLiteHistoryStore(path) as history:
        seed_runtime = LocalRuntime(history=history)
        seed_manager = _attach(seed_runtime, history, seed_registry)
        seed_bound = _binding(seed_runtime, spec, name="capacity-jobs")
        first = await _prepare_job(seed_manager, seed_bound, spec, _call(1))
        second = await _prepare_job(seed_manager, seed_bound, spec, _call(2))

    entered = 0
    peak = 0
    first_entered = asyncio.Event()
    second_entered = asyncio.Event()
    release = asyncio.Event()
    active_value: int | None = None

    async def execute(arguments):
        nonlocal active_value, entered, peak
        entered += 1
        peak = max(peak, entered)
        try:
            if not first_entered.is_set():
                active_value = arguments["value"]
                first_entered.set()
                await release.wait()
            else:
                second_entered.set()
            return arguments["value"]
        finally:
            entered -= 1

    registry = ExecutorRegistry()
    registry.register(spec.tool_id, spec.version, LocalToolExecutor(execute))
    async with SQLiteHistoryStore(path) as history:
        runtime = LocalRuntime(history=history)
        _attach(runtime, history, registry)
        bound = _binding(runtime, spec, name="capacity-jobs")
        recovered = await runtime.recover_tool_jobs(bound)
        assert {job.task_id for job in recovered} == {first.task_id, second.task_id}
        await first_entered.wait()
        await asyncio.sleep(0)
        assert not second_entered.is_set()

        assert active_value in (1, 2)
        queued_id = second.task_id if active_value == 1 else first.task_id
        assert await runtime.cancel_tool_task(queued_id)
        cancelled = await runtime.get_tool_result(queued_id)
        assert cancelled is not None and cancelled.status == "cancelled"
        release.set()
        running_id = first.task_id if active_value == 1 else second.task_id
        completed = await runtime.get_tool_result(running_id, wait=True)
        assert completed is not None and completed.status == "succeeded"
        assert peak == 1
        await runtime.close()


@pytest.mark.asyncio
async def test_recovery_rejects_plan_capability_and_tool_version_mismatch(tmp_path):
    path = tmp_path / "mismatch.sqlite3"
    spec = _spec(sandbox_profile="restricted")
    registry = ExecutorRegistry()
    registry.register(spec.tool_id, spec.version, _sandbox_executor())
    async with SQLiteHistoryStore(path) as history:
        seed_runtime = LocalRuntime(history=history)
        manager = _attach(seed_runtime, history, registry)
        bound = _binding(seed_runtime, spec, name="mismatch-jobs")
        await _prepare_job(
            manager,
            bound,
            spec,
            _call(1),
            capabilities=("durability.sqlite", "tool.sandbox.restricted"),
        )

    async with SQLiteHistoryStore(path) as history:
        missing_runtime = LocalRuntime(history=history)
        unsupported_registry = ExecutorRegistry()
        unsupported_registry.register(
            spec.tool_id,
            spec.version,
            LocalToolExecutor(lambda arguments: arguments),
        )
        _attach(missing_runtime, history, unsupported_registry)
        missing_bound = _binding(missing_runtime, spec, name="mismatch-jobs")
        with pytest.raises(ExecutionAdmissionError, match="missing required capabilities"):
            await missing_runtime.recover_tool_jobs(missing_bound)
        await missing_runtime.close()

    async with SQLiteHistoryStore(path) as history:
        missing_version_runtime = LocalRuntime(history=history)
        empty_registry = ExecutorRegistry()
        _attach(missing_version_runtime, history, empty_registry)
        matching_bound = _binding(
            missing_version_runtime, spec, name="mismatch-jobs"
        )
        with pytest.raises(ExecutionAdmissionError, match="missing required capabilities"):
            await missing_version_runtime.recover_tool_jobs(matching_bound)
        await missing_version_runtime.close()

    async with SQLiteHistoryStore(path) as history:
        changed_runtime = LocalRuntime(history=history)
        changed_registry = ExecutorRegistry()
        changed_spec = _spec(version="2", sandbox_profile="restricted")
        changed_registry.register(
            changed_spec.tool_id, changed_spec.version, _sandbox_executor()
        )
        _attach(changed_runtime, history, changed_registry)
        changed_bound = _binding(
            changed_runtime, changed_spec, name="mismatch-jobs"
        )
        with pytest.raises(ExecutionAdmissionError, match="ExecutionPlan"):
            await changed_runtime.recover_tool_jobs(changed_bound)
        await changed_runtime.close()


@pytest.mark.asyncio
async def test_running_job_recovery_obeys_idempotency_and_unknown_side_effects(
    tmp_path,
):
    path = tmp_path / "idempotency.sqlite3"
    unsafe = _spec(
        side_effect=ToolSideEffect.EXTERNAL,
        idempotency=IdempotencyPolicy.NOT_IDEMPOTENT,
    )
    safe = _spec(
        side_effect=ToolSideEffect.EXTERNAL,
        idempotency=IdempotencyPolicy.REQUIRES_KEY,
    )
    seed_registry = ExecutorRegistry()
    seed_registry.register(
        unsafe.tool_id,
        unsafe.version,
        LocalToolExecutor(lambda arguments: arguments),
    )
    async with SQLiteHistoryStore(path) as history:
        seed_runtime = LocalRuntime(history=history)
        manager = _attach(seed_runtime, history, seed_registry)
        bound = _binding(seed_runtime, unsafe, name="idempotency-jobs")
        unsafe_task = await _prepare_job(manager, bound, unsafe, _call(1))
        safe_task = await _prepare_job(
            manager, bound, safe, _call(2, key="stable-key")
        )
        await history.update_tool_job(
            unsafe_task.job_id, status=JobState.RUNNING.value
        )
        await history.update_tool_job(safe_task.job_id, status=JobState.RUNNING.value)

    executed: list[int] = []
    registry = ExecutorRegistry()
    registry.register(
        safe.tool_id,
        safe.version,
        LocalToolExecutor(lambda arguments: executed.append(arguments["value"]) or "ok"),
    )
    async with SQLiteHistoryStore(path) as history:
        runtime = LocalRuntime(history=history)
        _attach(runtime, history, registry)
        bound = _binding(runtime, unsafe, name="idempotency-jobs")
        await runtime.recover_tool_jobs(bound)
        unsafe_result = await runtime.get_tool_result(unsafe_task.task_id)
        safe_result = await runtime.get_tool_result(safe_task.task_id, wait=True)

        assert unsafe_result is not None and unsafe_result.status == "unknown"
        assert unsafe_result.side_effect_committed is None
        assert safe_result is not None and safe_result.status == "succeeded"
        assert executed == [2]
        safe_job = await runtime.get_job(safe_task.job_id)
        assert isinstance(safe_job, JobSnapshot) and safe_job.attempt == 2
        await runtime.close()


@pytest.mark.asyncio
async def test_durable_detached_executor_receives_sqlite_fencing_permit(tmp_path):
    coordinator = SQLiteCapacityCoordinator(tmp_path / "capacity.sqlite3")
    registry = ExecutorRegistry()
    spec = _spec()
    observed = []

    async def execute(arguments):
        permit = current_capacity_permit()
        assert permit is not None
        observed.append(permit)
        return arguments["value"]

    registry.register(spec.tool_id, spec.version, LocalToolExecutor(execute))
    async with SQLiteHistoryStore(tmp_path / "fenced-jobs.sqlite3") as history:
        runtime = LocalRuntime(history=history, capacity_coordinator=coordinator)
        _attach(runtime, history, registry)
        binding = runtime.create_binding(
            name="fenced-jobs",
            execution_capacity=ExecutionCapacityPolicy(
                scope=CapacityScope.RUNTIME_INSTANCE,
                max_live_executions=2,
                max_runnable_executions=1,
                max_queue_size=1,
                max_waiters=1,
                max_child_depth=4,
                max_children_per_execution=8,
            ),
            model_capacity=CapacityPolicy.passthrough(),
            tool_capacity=CapacityPolicy.limited(
                scope=CapacityScope.DEPLOYMENT,
                max_concurrency=1,
                max_queue_size=1,
                capacity_key="durable-backend",
            ),
            durability=DurabilityPolicy(DurabilityMode.REQUIRED),
        )
        bound = binding.bind(
            ToolCallLayer(tools=(spec,), authorization=_AllowDetach())
        )
        message, _ = await bound.invoke(
            AIMessage(tool_calls=(_call(7),)),
            Context(tools=(spec.definition,)),
        )
        task = message.results[0].task
        assert task is not None
        result = await runtime.get_tool_result(task.task_id, wait=True)
        assert result is not None and result.status == "succeeded"
        await runtime.close()
    await coordinator.close()

    assert len(observed) == 1
    assert observed[0].owner_key == "tool:durable-backend"
    assert observed[0].fencing_token is not None
