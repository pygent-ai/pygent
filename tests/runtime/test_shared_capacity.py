from __future__ import annotations

import asyncio
import threading
import time

import pytest

from pygent import (
    AIMessage,
    CapacityPolicy,
    CapacityScope,
    Context,
    ExecutionAdmissionError,
    ExecutionCapacityPolicy,
    ExecutionOptions,
    ExecutionStatus,
    ExternalWaitRejected,
    FallbackPolicy,
    GenerationConfig,
    InMemoryCapacityCoordinator,
    LocalRuntime,
    ModelCallLayer,
    ModelGroupConfig,
    ModelProviderResponse,
    ModelRoute,
    Module,
    RetryPolicy,
    SQLiteCapacityCoordinator,
    ToolAuthorizationDecision,
    ToolCall,
    ToolCallLayer,
    ToolDefinition,
    ToolSpec,
    UserMessage,
)
from pygent.core.module import _execution_scope
from pygent.tool import ExecutorRegistry, InMemoryToolTaskManager, LocalToolExecutor


def execution_capacity(
    *, scope: CapacityScope = CapacityScope.RUNTIME_INSTANCE
) -> ExecutionCapacityPolicy:
    return ExecutionCapacityPolicy(
        scope=scope,
        max_live_executions=4,
        max_runnable_executions=4,
        max_queue_size=0,
        max_waiters=4,
        max_child_depth=4,
        max_children_per_execution=8,
    )


class UsesModelPermit(Module[UserMessage, AIMessage]):
    trusted_live_resource_attributes = ("_entered", "_release")

    def __init__(self, entered: asyncio.Event, release: asyncio.Event) -> None:
        super().__init__()
        self._entered = entered
        self._release = release

    async def forward(self, message, context):
        scope = _execution_scope.get()
        assert scope is not None
        async with scope.model_permit():
            self._entered.set()
            await self._release.wait()
        return AIMessage(content=message.content), context


@pytest.mark.asyncio
async def test_capacity_key_shares_model_gate_across_bindings() -> None:
    runtime = LocalRuntime()
    policy = CapacityPolicy.limited(
        max_concurrency=1,
        max_queue_size=1,
        capacity_key="provider-account-a",
    )
    first_binding = runtime.create_binding(
        name="first",
        execution_capacity=execution_capacity(),
        model_capacity=policy,
        tool_capacity=CapacityPolicy.passthrough(),
    )
    second_binding = runtime.create_binding(
        name="second",
        execution_capacity=execution_capacity(),
        model_capacity=policy,
        tool_capacity=CapacityPolicy.passthrough(),
    )
    first_entered = asyncio.Event()
    first_release = asyncio.Event()
    second_entered = asyncio.Event()
    second_release = asyncio.Event()
    first = first_binding.bind(UsesModelPermit(first_entered, first_release))
    second = second_binding.bind(UsesModelPermit(second_entered, second_release))

    first_handle = await first.start(UserMessage(content="one"), Context())
    await first_entered.wait()
    second_handle = await second.start(UserMessage(content="two"), Context())
    await asyncio.sleep(0)
    assert not second_entered.is_set()

    first_release.set()
    await first_handle.result()
    await second_entered.wait()
    second_release.set()
    await second_handle.result()
    await runtime.close()


@pytest.mark.asyncio
async def test_deployment_scope_reuses_stable_binding_capacity_identity() -> None:
    coordinator = InMemoryCapacityCoordinator()
    first_runtime = LocalRuntime(capacity_coordinator=coordinator)
    second_runtime = LocalRuntime(capacity_coordinator=coordinator)

    def create_binding():
        runtime = (
            first_runtime if not hasattr(create_binding, "called") else second_runtime
        )
        create_binding.called = True
        return runtime.create_binding(
            name="shared-deployment",
            execution_capacity=ExecutionCapacityPolicy(
                scope=CapacityScope.DEPLOYMENT,
                max_live_executions=1,
                max_runnable_executions=1,
                max_queue_size=0,
                max_waiters=1,
                max_child_depth=4,
                max_children_per_execution=8,
            ),
            model_capacity=CapacityPolicy.passthrough(),
            tool_capacity=CapacityPolicy.passthrough(),
        )

    first_binding = create_binding()
    second_binding = create_binding()
    entered = asyncio.Event()
    release = asyncio.Event()
    first = first_binding.bind(UsesModelPermit(entered, release))
    second = second_binding.bind(UsesModelPermit(asyncio.Event(), asyncio.Event()))

    first_handle = await first.start(UserMessage(), Context())
    await entered.wait()
    second_handle = await second.start(UserMessage(), Context())
    with pytest.raises(ExecutionAdmissionError):
        await second_handle.result()

    release.set()
    await first_handle.result()
    await first_runtime.close()
    await second_runtime.close()


def test_deployment_capacity_fails_closed_without_shared_coordinator() -> None:
    runtime = LocalRuntime()
    with pytest.raises(ExecutionAdmissionError, match="capacity_coordinator"):
        runtime.create_binding(
            name="not-global",
            execution_capacity=execution_capacity(scope=CapacityScope.DEPLOYMENT),
            model_capacity=CapacityPolicy.passthrough(),
            tool_capacity=CapacityPolicy.passthrough(),
        )


@pytest.mark.asyncio
async def test_deployment_model_gate_is_shared_across_runtime_instances() -> None:
    coordinator = InMemoryCapacityCoordinator()
    first_runtime = LocalRuntime(capacity_coordinator=coordinator)
    second_runtime = LocalRuntime(capacity_coordinator=coordinator)
    policy = CapacityPolicy.limited(
        scope=CapacityScope.DEPLOYMENT,
        max_concurrency=1,
        max_queue_size=1,
        capacity_key="provider-account-global",
    )

    def bind(runtime, name, entered, release):
        return runtime.create_binding(
            name=name,
            execution_capacity=execution_capacity(),
            model_capacity=policy,
            tool_capacity=CapacityPolicy.passthrough(),
        ).bind(UsesModelPermit(entered, release))

    first_entered, first_release = asyncio.Event(), asyncio.Event()
    second_entered, second_release = asyncio.Event(), asyncio.Event()
    first = bind(first_runtime, "first", first_entered, first_release)
    second = bind(second_runtime, "second", second_entered, second_release)

    first_handle = await first.start(UserMessage(), Context())
    await first_entered.wait()
    second_handle = await second.start(UserMessage(), Context())
    await asyncio.sleep(0)
    assert not second_entered.is_set()
    first_release.set()
    await first_handle.result()
    await second_entered.wait()
    second_release.set()
    await second_handle.result()
    await first_runtime.close()
    await second_runtime.close()


@pytest.mark.asyncio
async def test_sqlite_deployment_execution_capacity_is_shared_across_coordinators(
    tmp_path,
) -> None:
    database = tmp_path / "shared-capacity.sqlite3"
    first_coordinator = SQLiteCapacityCoordinator(database)
    second_coordinator = SQLiteCapacityCoordinator(database)
    first_runtime = LocalRuntime(capacity_coordinator=first_coordinator)
    second_runtime = LocalRuntime(capacity_coordinator=second_coordinator)
    deployment = ExecutionCapacityPolicy(
        scope=CapacityScope.DEPLOYMENT,
        max_live_executions=1,
        max_runnable_executions=1,
        max_queue_size=1,
        max_waiters=2,
        max_child_depth=4,
        max_children_per_execution=8,
    )

    def bind(runtime, entered, release):
        return runtime.create_binding(
            name="sqlite-shared-deployment",
            execution_capacity=deployment,
            model_capacity=CapacityPolicy.passthrough(),
            tool_capacity=CapacityPolicy.passthrough(),
        ).bind(UsesModelPermit(entered, release))

    first_entered, first_release = asyncio.Event(), asyncio.Event()
    second_entered, second_release = asyncio.Event(), asyncio.Event()
    first = bind(first_runtime, first_entered, first_release)
    second = bind(second_runtime, second_entered, second_release)
    first_handle = await first.start(UserMessage(), Context())
    await asyncio.wait_for(first_entered.wait(), timeout=1)
    second_handle = await second.start(UserMessage(), Context())
    await asyncio.sleep(0.05)
    assert not second_entered.is_set()

    first_release.set()
    await first_handle.result()
    await asyncio.wait_for(second_entered.wait(), timeout=1)
    second_release.set()
    await second_handle.result()
    await first_runtime.close()
    await second_runtime.close()
    await first_coordinator.close()
    await second_coordinator.close()


@pytest.mark.asyncio
async def test_sqlite_deployment_max_waiters_is_global_across_runtimes(
    tmp_path,
) -> None:
    class Waiting(Module[UserMessage, AIMessage]):
        async def forward(self, message, context):
            value = await self.wait_external(
                kind="approval",
                key=message.content,
                request={"question": "continue?"},
                timeout=2,
            )
            return AIMessage(content=str(value["decision"])), context

    database = tmp_path / "shared-waiters.sqlite3"
    first_coordinator = SQLiteCapacityCoordinator(database)
    second_coordinator = SQLiteCapacityCoordinator(database)
    first_runtime = LocalRuntime(capacity_coordinator=first_coordinator)
    second_runtime = LocalRuntime(capacity_coordinator=second_coordinator)
    policy = ExecutionCapacityPolicy(
        scope=CapacityScope.DEPLOYMENT,
        max_live_executions=4,
        max_runnable_executions=2,
        max_queue_size=0,
        max_waiters=1,
        max_child_depth=4,
        max_children_per_execution=8,
        max_external_wait_seconds=2,
    )

    def bind(runtime):
        return runtime.create_binding(
            name="sqlite-global-waiters",
            execution_capacity=policy,
            model_capacity=CapacityPolicy.passthrough(),
            tool_capacity=CapacityPolicy.passthrough(),
        ).bind(Waiting())

    first = bind(first_runtime)
    second = bind(second_runtime)
    first_handle = await first.start(
        UserMessage(content="first"),
        Context(),
        execution=ExecutionOptions(),
    )
    while first_handle.status is not ExecutionStatus.WAITING_EXTERNAL:
        await asyncio.sleep(0)

    second_handle = await second.start(
        UserMessage(content="second"),
        Context(),
        execution=ExecutionOptions(),
    )
    with pytest.raises(ExternalWaitRejected, match="waiter capacity"):
        await asyncio.wait_for(second_handle.result(), timeout=1)

    await first_runtime.deliver_external(
        kind="approval", key="first", value={"decision": "approved"}
    )
    assert (await first_handle.result())[0].content == "approved"
    await first_runtime.close()
    await second_runtime.close()
    await first_coordinator.close()
    await second_coordinator.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["model", "tool"])
async def test_sqlite_deployment_resource_capacity_is_shared_and_fifo(
    tmp_path,
    kind,
) -> None:
    database = tmp_path / f"shared-{kind}.sqlite3"
    first_coordinator = SQLiteCapacityCoordinator(database)
    second_coordinator = SQLiteCapacityCoordinator(database)
    policy = CapacityPolicy.limited(
        scope=CapacityScope.DEPLOYMENT,
        max_concurrency=1,
        max_queue_size=2,
        capacity_key="shared-account",
    )
    first_gate = first_coordinator.resource_gate(kind, "shared-account", policy)
    second_gate = second_coordinator.resource_gate(kind, "shared-account", policy)
    entered: list[str] = []
    release_first = asyncio.Event()

    async def first() -> None:
        async with first_gate.permit():
            entered.append("first")
            await release_first.wait()

    async def queued(label: str) -> None:
        async with second_gate.permit():
            entered.append(label)

    first_task = asyncio.create_task(first())
    while entered != ["first"]:
        await asyncio.sleep(0)
    second_task = asyncio.create_task(queued("second"))
    await asyncio.sleep(0.02)
    third_task = asyncio.create_task(queued("third"))
    await asyncio.sleep(0.05)
    assert entered == ["first"]
    release_first.set()
    await asyncio.gather(first_task, second_task, third_task)
    assert entered == ["first", "second", "third"]
    await first_coordinator.close()
    await second_coordinator.close()


@pytest.mark.asyncio
async def test_sqlite_capacity_fifo_is_stable_under_concurrent_waiters(
    tmp_path,
) -> None:
    database = tmp_path / "concurrent-fifo.sqlite3"
    coordinators = (
        SQLiteCapacityCoordinator(database, poll_interval=0.005),
        SQLiteCapacityCoordinator(database, poll_interval=0.005),
    )
    policy = CapacityPolicy.limited(
        scope=CapacityScope.DEPLOYMENT,
        max_concurrency=1,
        max_queue_size=12,
        capacity_key="fifo-owner",
    )
    gates = tuple(
        coordinator.resource_gate("model", "fifo-owner", policy)
        for coordinator in coordinators
    )

    async def exercise_round() -> None:
        blocker_entered = asyncio.Event()
        release_blocker = asyncio.Event()
        entered: list[int] = []

        async def block() -> None:
            async with gates[0].permit():
                blocker_entered.set()
                await release_blocker.wait()

        async def queued(index: int) -> None:
            async with gates[index % 2].permit():
                entered.append(index)

        blocker = asyncio.create_task(block())
        await blocker_entered.wait()
        waiters = [asyncio.create_task(queued(index)) for index in range(12)]
        # Let every acquire pass through the path-wide submission lock and
        # persist its FIFO ticket before making capacity available.
        await asyncio.sleep(0.1)
        release_blocker.set()
        await asyncio.gather(blocker, *waiters)
        assert entered == list(range(12))

    for _round in range(4):
        await exercise_round()
    await asyncio.gather(*(coordinator.close() for coordinator in coordinators))


@pytest.mark.asyncio
async def test_sqlite_capacity_serializes_writers_across_coordinators(
    tmp_path,
) -> None:
    database = tmp_path / "shared-writer.sqlite3"
    coordinators = (
        SQLiteCapacityCoordinator(database),
        SQLiteCapacityCoordinator(database),
    )
    state_lock = threading.Lock()
    active = 0
    peak = 0

    def operation(_connection) -> None:
        nonlocal active, peak
        with state_lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with state_lock:
            active -= 1

    await asyncio.gather(
        *(
            asyncio.to_thread(coordinator._transaction, operation)
            for coordinator in coordinators
        )
    )

    assert peak == 1
    await asyncio.gather(*(coordinator.close() for coordinator in coordinators))


@pytest.mark.asyncio
async def test_sqlite_capacity_cancel_removes_waiter_and_frees_queue(
    tmp_path,
) -> None:
    database = tmp_path / "cancel-capacity.sqlite3"
    first_coordinator = SQLiteCapacityCoordinator(database)
    second_coordinator = SQLiteCapacityCoordinator(database)
    policy = CapacityPolicy.limited(
        scope=CapacityScope.DEPLOYMENT,
        max_concurrency=1,
        max_queue_size=1,
        capacity_key="cancel-cleanup",
    )
    first_gate = first_coordinator.resource_gate("tool", "cancel-cleanup", policy)
    second_gate = second_coordinator.resource_gate("tool", "cancel-cleanup", policy)
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    third_entered = asyncio.Event()

    async def hold() -> None:
        async with first_gate.permit():
            first_entered.set()
            await release_first.wait()

    async def wait(gate, entered=None) -> None:
        async with gate.permit():
            if entered is not None:
                entered.set()

    holder = asyncio.create_task(hold())
    await first_entered.wait()
    cancelled = asyncio.create_task(wait(second_gate))
    await asyncio.sleep(0.05)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled

    replacement = asyncio.create_task(wait(second_gate, third_entered))
    await asyncio.sleep(0.05)
    assert not third_entered.is_set()
    release_first.set()
    await asyncio.gather(holder, replacement)
    assert third_entered.is_set()
    await first_coordinator.close()
    await second_coordinator.close()


@pytest.mark.asyncio
async def test_sqlite_capacity_reclaims_expired_crash_lease(tmp_path) -> None:
    database = tmp_path / "expired-capacity.sqlite3"
    first_coordinator = SQLiteCapacityCoordinator(
        database, lease_ttl=0.12, poll_interval=0.01
    )
    second_coordinator = SQLiteCapacityCoordinator(
        database, lease_ttl=0.12, poll_interval=0.01
    )
    policy = CapacityPolicy.limited(
        scope=CapacityScope.DEPLOYMENT,
        max_concurrency=1,
        max_queue_size=1,
        capacity_key="crash-recovery",
    )
    first_gate = first_coordinator.resource_gate("model", "crash-recovery", policy)
    second_gate = second_coordinator.resource_gate("model", "crash-recovery", policy)
    first_entered = asyncio.Event()
    never_release = asyncio.Event()

    async def crashed_owner() -> None:
        async with first_gate.permit():
            first_entered.set()
            await never_release.wait()

    stale_task = asyncio.create_task(crashed_owner())
    await first_entered.wait()
    await first_coordinator.close(release_leases=False)
    recovered = asyncio.Event()

    async def recover() -> None:
        async with second_gate.permit():
            recovered.set()

    await asyncio.wait_for(recover(), timeout=1)
    assert recovered.is_set()
    stale_task.cancel()
    await asyncio.gather(stale_task, return_exceptions=True)
    await second_coordinator.close()


def test_sqlite_capacity_rejects_cross_process_policy_conflict(tmp_path) -> None:
    database = tmp_path / "policy-conflict.sqlite3"
    first = SQLiteCapacityCoordinator(database)
    second = SQLiteCapacityCoordinator(database)
    first.resource_gate(
        "model",
        "stable-owner",
        CapacityPolicy.limited(
            scope=CapacityScope.DEPLOYMENT,
            max_concurrency=1,
            max_queue_size=1,
            capacity_key="stable-owner",
        ),
    )
    with pytest.raises(ValueError, match="conflicting policies"):
        second.resource_gate(
            "model",
            "stable-owner",
            CapacityPolicy.limited(
                scope=CapacityScope.DEPLOYMENT,
                max_concurrency=2,
                max_queue_size=1,
                capacity_key="stable-owner",
            ),
        )


class BlockingInvoker:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    def execute(self, **kwargs):
        from pygent import ModelExecution

        async def operation(emit):
            self.entered.set()
            await self.release.wait()
            return ModelProviderResponse(AIMessage(content="ok"))

        return ModelExecution(operation)


@pytest.mark.asyncio
async def test_model_group_max_concurrency_is_enforced_by_managed_runtime() -> None:
    runtime = LocalRuntime()
    invoker = BlockingInvoker()
    runtime.register_model_invoker("logical-a", invoker)
    runtime.register_model_invoker("logical-b", invoker)

    def layer(name: str) -> ModelCallLayer:
        return ModelCallLayer(
            model_group=ModelGroupConfig(
                name=name,
                routes=(ModelRoute("only", "openai", "test"),),
                fallback=FallbackPolicy(("only",)),
                max_concurrency=1,
                capacity_key="shared-model",
            ),
            retry_policy=RetryPolicy(),
            generation=GenerationConfig(),
        )

    binding = runtime.create_binding(
        name="model-group",
        execution_capacity=execution_capacity(),
        model_capacity=CapacityPolicy.passthrough(),
        tool_capacity=CapacityPolicy.passthrough(),
    )
    first_bound = binding.bind(layer("logical-a"))
    second_bound = binding.bind(layer("logical-b"))

    first = await first_bound.start(
        UserMessage(content="one"),
        Context(),
        execution=ExecutionOptions(deadline=time.monotonic() + 3),
    )
    await invoker.entered.wait()
    second = await second_bound.start(
        UserMessage(content="two"),
        Context(),
        execution=ExecutionOptions(deadline=time.monotonic() + 3),
    )
    with pytest.raises(ExecutionAdmissionError, match="model:shared-model"):
        await second.result()

    invoker.release.set()
    await first.result()
    await runtime.close()


def test_capacity_scope_contract_rejects_false_ownership_claims() -> None:
    external = CapacityPolicy.passthrough(capacity_key="provider-account-a")
    assert external.scope is CapacityScope.EXTERNAL_RESOURCE
    with pytest.raises(ValueError, match="must use passthrough"):
        CapacityPolicy.limited(
            scope=CapacityScope.EXTERNAL_RESOURCE,
            max_concurrency=1,
            max_queue_size=0,
        )
    with pytest.raises(ValueError, match="stable capacity_key"):
        CapacityPolicy.limited(
            scope=CapacityScope.DEPLOYMENT,
            max_concurrency=1,
            max_queue_size=0,
        )


def test_binding_capacity_policy_participates_in_bound_plan_identity() -> None:
    def bind(limit: int, capacity_key: str):
        runtime = LocalRuntime()
        binding = runtime.create_binding(
            name=f"policy-{limit}-{capacity_key}",
            execution_capacity=execution_capacity(),
            model_capacity=CapacityPolicy.limited(
                max_concurrency=limit,
                max_queue_size=1,
                capacity_key=capacity_key,
            ),
            tool_capacity=CapacityPolicy.passthrough(),
        )
        return binding.bind(UsesModelPermit(asyncio.Event(), asyncio.Event()))

    baseline = bind(1, "provider-a")
    changed_limit = bind(2, "provider-a")
    changed_key = bind(1, "provider-b")

    assert baseline.plan.graph_hash != changed_limit.plan.graph_hash
    assert baseline.plan.graph_hash != changed_key.plan.graph_hash
    root = next(item for item in baseline.plan.modules if item.path == "root")
    assert "model:provider-a" in root.resource_keys
    assert dict(root.metadata)["binding_policy_ref"].startswith("sha256:")


@pytest.mark.asyncio
async def test_tool_resource_key_shares_gate_across_bindings() -> None:
    runtime = LocalRuntime()
    registry = ExecutorRegistry()
    entered_count = 0
    first_entered = asyncio.Event()
    second_entered = asyncio.Event()
    first_release = asyncio.Event()

    async def execute(arguments):
        nonlocal entered_count
        entered_count += 1
        if entered_count == 1:
            first_entered.set()
            await first_release.wait()
        else:
            second_entered.set()
        return arguments["value"]

    tool = ToolSpec(
        tool_id="shared.echo",
        version="1.0.0",
        definition=ToolDefinition(
            name="shared_echo",
            description="echo",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            },
        ),
        resource_key="shared-backend",
    )
    registry.register(tool.tool_id, tool.version, LocalToolExecutor(execute))
    runtime.attach_executor_registry(registry)

    def authorize(request, context):
        return ToolAuthorizationDecision(
            call_id=request.call.call_id,
            allowed=True,
            reason_code="allowed",
        )

    layer_one = ToolCallLayer(tools=(tool,), authorization_adapter=authorize)
    layer_two = ToolCallLayer(tools=(tool,), authorization_adapter=authorize)
    tool_policy = CapacityPolicy.limited(
        max_concurrency=1,
        max_queue_size=1,
    )

    def bind(name, layer):
        binding = runtime.create_binding(
            name=name,
            execution_capacity=execution_capacity(),
            model_capacity=CapacityPolicy.passthrough(),
            tool_capacity=tool_policy,
        )
        return binding.bind(layer)

    first = bind("tool-one", layer_one)
    second = bind("tool-two", layer_two)
    context = Context(tools=(tool.definition,))
    first_handle = await first.start(
        AIMessage(tool_calls=(ToolCall("first", "shared_echo", {"value": 1}),)),
        context,
    )
    await first_entered.wait()
    second_handle = await second.start(
        AIMessage(tool_calls=(ToolCall("second", "shared_echo", {"value": 2}),)),
        context,
    )
    await asyncio.sleep(0)
    assert not second_entered.is_set()

    first_release.set()
    assert (await first_handle.result())[0].results[0].status == "succeeded"
    await second_entered.wait()
    assert (await second_handle.result())[0].results[0].status == "succeeded"
    await runtime.close()


@pytest.mark.asyncio
async def test_managed_detach_does_not_deadlock_with_one_runnable_lease() -> None:
    runtime = LocalRuntime()
    registry = ExecutorRegistry()
    entered = asyncio.Event()

    async def execute(arguments):
        entered.set()
        return arguments["value"]

    tool = ToolSpec(
        tool_id="detached.echo",
        version="1.0.0",
        definition=ToolDefinition(
            name="detached_echo",
            description="echo",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            },
        ),
        resource_key="detached-backend",
    )
    registry.register(tool.tool_id, tool.version, LocalToolExecutor(execute))
    manager = InMemoryToolTaskManager(registry)
    runtime.attach_executor_registry(registry)
    runtime.attach_tool_task_manager(manager)
    binding = runtime.create_binding(
        name="detached-one-runnable",
        execution_capacity=ExecutionCapacityPolicy(
            scope=CapacityScope.RUNTIME_INSTANCE,
            max_live_executions=2,
            max_runnable_executions=1,
            max_queue_size=1,
            max_waiters=2,
            max_child_depth=4,
            max_children_per_execution=8,
        ),
        model_capacity=CapacityPolicy.passthrough(),
        tool_capacity=CapacityPolicy.limited(max_concurrency=1, max_queue_size=1),
    )
    layer = binding.bind(
        ToolCallLayer(
            tools=(tool,),
            authorization_adapter=lambda request, _context: ToolAuthorizationDecision(
                call_id=request.call.call_id,
                allowed=True,
                reason_code="allowed",
                lifecycle="detach",
            ),
        )
    )

    message, _ = await asyncio.wait_for(
        layer.invoke(
            AIMessage(
                tool_calls=(ToolCall("detached", "detached_echo", {"value": 1}),)
            ),
            Context(tools=(tool.definition,)),
        ),
        timeout=1,
    )
    detached = message.results[0]
    assert detached.status == "detached" and detached.task is not None
    await asyncio.wait_for(entered.wait(), timeout=1)
    final = await asyncio.wait_for(
        runtime.get_tool_result(detached.task.task_id, wait=True), timeout=1
    )
    assert final is not None and final.status == "succeeded"
    await runtime.close()


@pytest.mark.asyncio
async def test_detached_tasks_share_binding_tool_capacity_and_cancel_queue_cleanly() -> (
    None
):
    runtime = LocalRuntime()
    registry = ExecutorRegistry()
    first_entered = asyncio.Event()
    second_entered = asyncio.Event()
    third_entered = asyncio.Event()
    release_first = asyncio.Event()
    inflight = 0
    peak_inflight = 0

    async def execute(arguments):
        nonlocal inflight, peak_inflight
        value = arguments["value"]
        inflight += 1
        peak_inflight = max(peak_inflight, inflight)
        try:
            if value == 1:
                first_entered.set()
                await release_first.wait()
            elif value == 2:
                second_entered.set()
            else:
                third_entered.set()
            return value
        finally:
            inflight -= 1

    tool = ToolSpec(
        tool_id="detached.shared",
        version="1.0.0",
        definition=ToolDefinition(
            name="detached_shared",
            description="shared detached capacity",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            },
        ),
        resource_key="physical-detached-backend",
    )
    registry.register(tool.tool_id, tool.version, LocalToolExecutor(execute))
    manager = InMemoryToolTaskManager(registry)
    runtime.attach_executor_registry(registry)
    runtime.attach_tool_task_manager(manager)
    policy = CapacityPolicy.limited(
        max_concurrency=1,
        max_queue_size=2,
        capacity_key="shared-detached-capacity",
    )

    def bind(name: str):
        binding = runtime.create_binding(
            name=name,
            execution_capacity=execution_capacity(),
            model_capacity=CapacityPolicy.passthrough(),
            tool_capacity=policy,
        )
        return binding.bind(
            ToolCallLayer(
                tools=(tool,),
                authorization_adapter=lambda request, _context: (
                    ToolAuthorizationDecision(
                        call_id=request.call.call_id,
                        allowed=True,
                        reason_code="allowed",
                        lifecycle="detach",
                    )
                ),
            )
        )

    first_layer = bind("detached-capacity-one")
    second_layer = bind("detached-capacity-two")
    context = Context(tools=(tool.definition,))

    async def submit(layer, call_id: str, value: int):
        message, _ = await layer.invoke(
            AIMessage(
                tool_calls=(ToolCall(call_id, "detached_shared", {"value": value}),)
            ),
            context,
        )
        task = message.results[0].task
        assert task is not None
        return task

    first_task = await submit(first_layer, "first", 1)
    await first_entered.wait()
    second_task = await submit(second_layer, "second", 2)
    await asyncio.sleep(0)
    assert not second_entered.is_set()

    assert await runtime.cancel_tool_task(second_task.task_id)
    cancelled = await runtime.get_tool_result(second_task.task_id)
    assert cancelled is not None and cancelled.status == "cancelled"

    release_first.set()
    first_result = await runtime.get_tool_result(first_task.task_id, wait=True)
    assert first_result is not None and first_result.status == "succeeded"

    third_task = await submit(second_layer, "third", 3)
    await asyncio.wait_for(third_entered.wait(), timeout=1)
    third_result = await runtime.get_tool_result(third_task.task_id, wait=True)
    assert third_result is not None and third_result.status == "succeeded"
    assert peak_inflight == 1
    await runtime.close()
