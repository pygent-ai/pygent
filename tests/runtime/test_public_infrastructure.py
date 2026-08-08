from __future__ import annotations

import asyncio

import pytest

from pygent import (
    AIMessage,
    Context,
    Module,
    UserMessage,
)
from pygent.core import (
    EffectIdempotency,
    EffectRecoveryUnknown,
    EffectRetryPolicy,
    EffectSafety,
    EffectSideEffect,
    EffectSpec,
    ExecutionRequirements,
    RecoverySafety,
    active_infrastructure,
    current_capacity_permit,
    current_infrastructure,
)
from pygent.runtime import (
    CapacityPolicy,
    CapacityScope,
    ExecutionCapacityPolicy,
    LocalRuntime,
    SQLiteCapacityCoordinator,
    SQLiteHistoryStore,
)
from pygent.runtime.codec import invocation_to_dict


class _CallCounter:
    def __init__(self) -> None:
        self.calls = 0


class PublicEffectModule(Module[UserMessage, AIMessage]):
    execution_requirements = ExecutionRequirements(
        recovery_safety=RecoverySafety.MODULE_BOUNDARY_RETRY,
        effect_safety=EffectSafety.MANAGED_EFFECTS,
    )
    trusted_live_resource_attributes = ("counter",)

    def __init__(self, counter: _CallCounter) -> None:
        super().__init__()
        self.counter = counter

    async def forward(self, message, context):
        infrastructure = current_infrastructure()

        async def operation():
            self.counter.calls += 1
            return {"content": message.content.upper()}

        result = await infrastructure.execute_effect(
            spec=EffectSpec(
                effect_type="application.lookup",
                side_effect=EffectSideEffect.READ,
                idempotency=EffectIdempotency.INHERENT,
                retry_policy=EffectRetryPolicy.REPLAY_SAFE,
            ),
            request={"content": message.content},
            operation=operation,
        )
        content = result.value["content"]
        assert isinstance(content, str)
        return AIMessage(content=content), context


class UnsafeWriteEffectModule(Module[UserMessage, AIMessage]):
    execution_requirements = ExecutionRequirements(
        recovery_safety=RecoverySafety.MODULE_BOUNDARY_RETRY,
        effect_safety=EffectSafety.MANAGED_EFFECTS,
    )
    trusted_live_resource_attributes = ("counter",)

    def __init__(self, counter: _CallCounter) -> None:
        super().__init__()
        self.counter = counter

    async def forward(self, message, context):
        async def operation():
            self.counter.calls += 1
            return {"content": "must-not-replay"}

        result = await current_infrastructure().execute_effect(
            spec=EffectSpec(
                effect_type="application.write",
                side_effect=EffectSideEffect.WRITE,
                idempotency=EffectIdempotency.NOT_IDEMPOTENT,
                retry_policy=EffectRetryPolicy.FAIL_CLOSED,
            ),
            request={"content": message.content},
            operation=operation,
        )
        return AIMessage(content=str(result.value["content"])), context


class _PermitProbe:
    def __init__(self) -> None:
        self.active = 0
        self.peak = 0

    async def run(self) -> None:
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.01)
        finally:
            self.active -= 1


class PublicToolPermitModule(Module[UserMessage, AIMessage]):
    trusted_live_resource_attributes = ("_probe",)
    execution_requirements = ExecutionRequirements(
        recovery_safety=RecoverySafety.MODULE_BOUNDARY_RETRY,
        effect_safety=EffectSafety.EFFECT_FREE,
    )

    def __init__(self, probe: _PermitProbe) -> None:
        super().__init__()
        self._probe = probe

    async def forward(self, message, context):
        async with current_infrastructure().tool_permit("application.database"):
            await self._probe.run()
        return AIMessage(content=message.content), context


class _TokenRecorder:
    def __init__(self) -> None:
        self.tokens: list[int] = []


class PublicFencingPermitModule(Module[UserMessage, AIMessage]):
    trusted_live_resource_attributes = ("_recorder",)

    def __init__(self, recorder: _TokenRecorder) -> None:
        super().__init__()
        self._recorder = recorder

    async def forward(self, message, context):
        async with current_infrastructure().tool_permit(
            "application.database"
        ) as permit:
            assert current_capacity_permit() == permit
            assert permit.owner_key == "tool:application.database"
            assert permit.fencing_token is not None
            self._recorder.tokens.append(permit.fencing_token)
        assert current_capacity_permit() is None
        return AIMessage(content=message.content), context


class PublicResourceModule(Module[UserMessage, AIMessage]):
    trusted_live_resource_attributes = ("_expected",)

    def __init__(self, expected: object) -> None:
        super().__init__()
        self._expected = expected

    async def forward(self, message, context):
        infrastructure = current_infrastructure()
        resolved = infrastructure.resolve_model_invoker("application-resource")
        key = infrastructure.tool_idempotency_key("operation-1")
        return AIMessage(content=f"{resolved is self._expected}:{key}"), context


def test_effect_spec_rejects_unsafe_replay_and_missing_stable_key():
    with pytest.raises(ValueError, match="replay_safe"):
        EffectSpec(
            effect_type="application.write",
            side_effect=EffectSideEffect.WRITE,
            idempotency=EffectIdempotency.NOT_IDEMPOTENT,
            retry_policy=EffectRetryPolicy.REPLAY_SAFE,
        )
    with pytest.raises(ValueError, match="stable idempotency_key"):
        EffectSpec(
            effect_type="application.write",
            side_effect=EffectSideEffect.WRITE,
            idempotency=EffectIdempotency.REQUIRES_KEY,
            retry_policy=EffectRetryPolicy.FAIL_CLOSED,
        )


@pytest.mark.asyncio
async def test_custom_infrastructure_module_replays_managed_effect(tmp_path):
    counter = _CallCounter()
    async with SQLiteHistoryStore(tmp_path / "history.sqlite3") as history:
        runtime = LocalRuntime(history=history)
        bound = runtime.bind(PublicEffectModule(counter))
        await history.create_execution(
            execution_id="public-effect-run",
            request_id="public-effect-request",
            plan_id=bound.plan.plan_id,
            input=invocation_to_dict(UserMessage(content="once"), Context()),
            status="running",
        )
        await history.record_effect(
            execution_id="public-effect-run",
            module_path=bound.plan.root,
            call_index=0,
            effect_type="application.lookup",
            request={"content": "once"},
            result={"content": "REPLAYED"},
        )

        recovered = await runtime.recover(bound, "public-effect-run")
        output, _ = await recovered.result()
        await runtime.close()

    assert output.content == "REPLAYED"
    assert counter.calls == 0


@pytest.mark.asyncio
async def test_started_non_idempotent_effect_becomes_unknown_without_replay(tmp_path):
    counter = _CallCounter()
    async with SQLiteHistoryStore(tmp_path / "unknown.sqlite3") as history:
        runtime = LocalRuntime(history=history)
        bound = runtime.bind(UnsafeWriteEffectModule(counter))
        message = UserMessage(content="write-once")
        context = Context()
        await history.create_execution(
            execution_id="unknown-effect-run",
            request_id="unknown-effect-request",
            plan_id=bound.plan.plan_id,
            input=invocation_to_dict(message, context),
            status="running",
        )
        policy = {
            "side_effect": "write",
            "idempotency": "not_idempotent",
            "retry_policy": "fail_closed",
            "idempotency_key": None,
        }
        await history.begin_effect(
            execution_id="unknown-effect-run",
            module_path=bound.plan.root,
            call_index=0,
            effect_type="application.write",
            request={"content": "write-once"},
            spec=policy,
        )

        recovered = await runtime.recover(bound, "unknown-effect-run")
        with pytest.raises(EffectRecoveryUnknown):
            await recovered.result()
        stored = await history.replay_effect(
            execution_id="unknown-effect-run",
            module_path=bound.plan.root,
            call_index=0,
            effect_type="application.write",
            request={"content": "write-once"},
            spec=policy,
        )
        assert stored.status == "unknown" and stored.result is None
        assert counter.calls == 0
        await runtime.close()


@pytest.mark.asyncio
async def test_started_read_effect_replays_under_explicit_safe_policy(tmp_path):
    counter = _CallCounter()
    async with SQLiteHistoryStore(tmp_path / "safe-replay.sqlite3") as history:
        runtime = LocalRuntime(history=history)
        bound = runtime.bind(PublicEffectModule(counter))
        message = UserMessage(content="again")
        context = Context()
        await history.create_execution(
            execution_id="safe-effect-run",
            request_id="safe-effect-request",
            plan_id=bound.plan.plan_id,
            input=invocation_to_dict(message, context),
            status="running",
        )
        policy = {
            "side_effect": "read",
            "idempotency": "inherent",
            "retry_policy": "replay_safe",
            "idempotency_key": None,
        }
        await history.begin_effect(
            execution_id="safe-effect-run",
            module_path=bound.plan.root,
            call_index=0,
            effect_type="application.lookup",
            request={"content": "again"},
            spec=policy,
        )
        recovered = await runtime.recover(bound, "safe-effect-run")
        output, _ = await recovered.result()
        stored = await history.replay_effect(
            execution_id="safe-effect-run",
            module_path=bound.plan.root,
            call_index=0,
            effect_type="application.lookup",
            request={"content": "again"},
            spec=policy,
        )
        assert output.content == "AGAIN"
        assert counter.calls == 1
        assert stored.status == "completed" and stored.result is not None
        await runtime.close()


@pytest.mark.asyncio
async def test_custom_infrastructure_module_uses_managed_resource_permit():
    probe = _PermitProbe()
    runtime = LocalRuntime()
    binding = runtime.create_binding(
        name="public-infrastructure",
        execution_capacity=ExecutionCapacityPolicy(
            max_live_executions=8,
            scope=CapacityScope.RUNTIME_INSTANCE,
            max_runnable_executions=4,
            max_queue_size=8,
            max_waiters=8,
            max_child_depth=4,
            max_children_per_execution=8,
        ),
        model_capacity=CapacityPolicy.passthrough(),
        tool_capacity=CapacityPolicy.limited(
            max_concurrency=1,
            max_queue_size=8,
        ),
    )
    bound = binding.bind(PublicToolPermitModule(probe))
    handles = [
        await bound.start(UserMessage(content=str(index)), Context())
        for index in range(6)
    ]
    await asyncio.gather(*(handle.result() for handle in handles))
    await runtime.close()

    assert probe.peak == 1


@pytest.mark.asyncio
async def test_public_capacity_permit_exposes_sqlite_fencing_token(tmp_path):
    coordinator = SQLiteCapacityCoordinator(tmp_path / "fencing.sqlite3")
    runtime = LocalRuntime(capacity_coordinator=coordinator)
    binding = runtime.create_binding(
        name="public-fencing",
        execution_capacity=ExecutionCapacityPolicy(
            max_live_executions=2,
            scope=CapacityScope.RUNTIME_INSTANCE,
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
            capacity_key="application.database",
        ),
    )
    recorder = _TokenRecorder()
    bound = binding.bind(PublicFencingPermitModule(recorder))

    await bound.invoke(UserMessage(content="first"), Context())
    await bound.invoke(UserMessage(content="second"), Context())

    assert len(recorder.tokens) == 2
    assert recorder.tokens[1] > recorder.tokens[0]
    await runtime.close()
    await coordinator.close()


@pytest.mark.asyncio
async def test_public_infrastructure_direct_boundary_and_resource_resolution():
    counter = _CallCounter()
    direct = PublicEffectModule(counter)
    first, _ = await direct.invoke(UserMessage(content="one"), Context())
    second, _ = await direct.invoke(UserMessage(content="two"), Context())
    assert (first.content, second.content) == ("ONE", "TWO")
    assert counter.calls == 2
    assert active_infrastructure() is None
    with pytest.raises(RuntimeError, match="active Module execution"):
        current_infrastructure()

    class Marker:
        def execute(self):
            raise AssertionError("resource resolution must not invoke the adapter")

    marker = Marker()
    resource_module = PublicResourceModule(marker)
    with pytest.raises(RuntimeError, match="no local ModelInvoker"):
        await resource_module.invoke(UserMessage(content="direct"), Context())

    runtime = LocalRuntime()
    runtime.register_model_invoker("application-resource", marker)
    managed, _ = await runtime.bind(resource_module).invoke(
        UserMessage(content="managed"), Context()
    )
    await runtime.close()
    prefix, key = managed.content.split(":", 1)
    assert prefix == "True"
    assert key.endswith(":operation-1")
