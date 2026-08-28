from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from pygent import (
    AIMessage,
    Context,
    Message,
    ModelCallLayer,
    Module,
    PygentAgent,
    ReActLayer,
    ToolCallLayer,
    ToolMessage,
    UserMessage,
)
from pygent.core import (
    EffectSafety,
    ExecutionRequirements,
    RecoverySafety,
)
from pygent.runtime import (
    CapacityPolicy,
    CapacityScope,
    DurabilityMode,
    DurabilityPolicy,
    ExecutionAdmissionError,
    ExecutionCapacityPolicy,
    LocalRuntime,
    SQLiteHistoryStore,
    compile_execution_plan,
)


class Echo(Module[UserMessage, AIMessage]):
    async def forward(self, message, context):
        return AIMessage(content=message.content), context


class NeedsSQLiteDurability(Echo):
    execution_requirements = ExecutionRequirements(
        required_capabilities=("durability.sqlite",),
        recovery_safety=RecoverySafety.MODULE_BOUNDARY_RETRY,
        effect_safety=EffectSafety.EFFECT_FREE,
    )


class NeedsManagedReplay(Echo):
    execution_requirements = ExecutionRequirements(
        required_capabilities=(
            "durability.sqlite",
            "durability.managed-replay",
        ),
        recovery_safety=RecoverySafety.MODULE_BOUNDARY_RETRY,
        effect_safety=EffectSafety.MANAGED_EFFECTS,
    )


class RetryDeclaredButEffectsUnverified(Echo):
    execution_requirements = ExecutionRequirements(
        recovery_safety=RecoverySafety.MODULE_BOUNDARY_RETRY,
    )


def binding(runtime: LocalRuntime, mode: DurabilityMode = DurabilityMode.PREFERRED):
    return runtime.create_binding(
        name=f"durability-{mode.value}",
        execution_capacity=ExecutionCapacityPolicy(
            scope=CapacityScope.RUNTIME_INSTANCE,
            max_live_executions=2,
            max_runnable_executions=1,
            max_queue_size=1,
            max_waiters=1,
            max_child_depth=4,
            max_children_per_execution=4,
        ),
        model_capacity=CapacityPolicy.passthrough(),
        tool_capacity=CapacityPolicy.passthrough(),
        durability=DurabilityPolicy(mode),
    )


def test_durability_policy_and_report_are_immutable_and_default_is_compatible():
    policy = DurabilityPolicy()
    assert policy.mode is DurabilityMode.PREFERRED
    with pytest.raises(FrozenInstanceError):
        policy.mode = DurabilityMode.REQUIRED  # type: ignore[misc]

    runtime = LocalRuntime()
    bound = runtime.bind(Echo())
    report = bound.durability
    assert report.requested_mode is DurabilityMode.PREFERRED
    assert report.effective_capabilities == ()
    assert report.missing_capabilities == ("durability.sqlite",)
    assert report.recovery_level == "none"
    assert report.checkpoint_policy == "none"
    assert report.replay_policy == "none"
    assert report.event_reconnect is False
    assert report.capacity_scope is CapacityScope.RUNTIME_INSTANCE
    assert report.degraded_reasons
    assert report.recovery_undeclared_modules == ("root",)
    assert report.effect_unverified_modules == ("root",)
    with pytest.raises(FrozenInstanceError):
        report.recovery_level = "arbitrary_coroutine"  # type: ignore[misc]


def test_execution_requirements_safety_is_strict_immutable_hashable_and_hashed():
    declared = ExecutionRequirements(
        recovery_safety=RecoverySafety.MODULE_BOUNDARY_RETRY,
        effect_safety=EffectSafety.EFFECT_FREE,
    )
    assert isinstance(hash(declared), int)
    with pytest.raises(FrozenInstanceError):
        declared.effect_safety = EffectSafety.UNDECLARED  # type: ignore[misc]
    with pytest.raises(TypeError, match="RecoverySafety"):
        ExecutionRequirements(recovery_safety="module_boundary_retry")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="EffectSafety"):
        ExecutionRequirements(effect_safety="effect_free")  # type: ignore[arg-type]

    class DeclaredEcho(Echo):
        execution_requirements = declared

    undeclared_plan = compile_execution_plan(Echo())
    declared_plan = compile_execution_plan(DeclaredEcho())
    metadata = dict(declared_plan.modules[0].metadata)
    assert undeclared_plan.graph_hash != declared_plan.graph_hash
    assert metadata["recovery_safety"] == "module_boundary_retry"
    assert metadata["effect_safety"] == "effect_free"


def test_builtin_layers_declare_only_their_verifiable_effect_boundaries():
    assert (
        ReActLayer.execution_requirements.effect_safety
        is EffectSafety.MANAGED_EFFECTS
    )
    assert (
        ModelCallLayer.execution_requirements.effect_safety
        is EffectSafety.MANAGED_EFFECTS
    )
    assert (
        ToolCallLayer.execution_requirements.effect_safety
        is EffectSafety.MANAGED_EFFECTS
    )

    adapter_layer = ToolCallLayer(
        tools=(),
        authorization_adapter=lambda request, context: object(),  # type: ignore[arg-type,return-value]
    )
    assert adapter_layer.execution_requirements.effect_safety is (
        EffectSafety.UNDECLARED
    )


@pytest.mark.asyncio
async def test_pygent_agent_native_graph_declares_managed_boundary_retry(
    tmp_path,
):
    class SafeModel(Module[Message, AIMessage]):
        execution_requirements = ExecutionRequirements(
            recovery_safety=RecoverySafety.MODULE_BOUNDARY_RETRY,
            effect_safety=EffectSafety.MANAGED_EFFECTS,
        )

        async def forward(self, message, context):
            return AIMessage(content="done"), context

    class SafeTools(Module[AIMessage, ToolMessage]):
        execution_requirements = ExecutionRequirements(
            recovery_safety=RecoverySafety.MODULE_BOUNDARY_RETRY,
            effect_safety=EffectSafety.MANAGED_EFFECTS,
        )

        async def forward(self, message, context):
            return ToolMessage(content=""), context

    agent = PygentAgent(
        system_prompt="system",
        compression_prompt="compress",
        model=SafeModel(),
        compressor=SafeModel(),
        tools=SafeTools(),
        context_window_tokens=4096,
    )
    plan = compile_execution_plan(agent)

    assert all(
        dict(spec.metadata)["recovery_safety"] == "module_boundary_retry"
        for spec in plan.modules
    )
    assert all(
        dict(spec.metadata)["effect_safety"] in ("effect_free", "managed_effects")
        for spec in plan.modules
    )

    async with SQLiteHistoryStore(tmp_path / "history.sqlite3") as history:
        runtime = LocalRuntime(history=history)
        bound = binding(runtime).bind(agent)

        assert bound.durability.recovery_level == "module_boundary_retry"
        assert bound.durability.recovery_undeclared_modules == ()
        assert bound.durability.effect_unverified_modules == ()
        await runtime.close()


@pytest.mark.asyncio
async def test_required_rejects_unverified_effect_even_with_sqlite(tmp_path):
    async with SQLiteHistoryStore(tmp_path / "history.sqlite3") as history:
        runtime = LocalRuntime(history=history)
        preferred = binding(runtime).bind(RetryDeclaredButEffectsUnverified())
        assert preferred.durability.recovery_undeclared_modules == ()
        assert preferred.durability.effect_unverified_modules == ("root",)
        assert preferred.durability.recovery_level == "none"

        with pytest.raises(ExecutionAdmissionError, match="unverified unmanaged effects"):
            binding(runtime, DurabilityMode.REQUIRED).bind(
                RetryDeclaredButEffectsUnverified()
            )
        await runtime.close()


@pytest.mark.asyncio
async def test_required_validates_every_module_in_the_declared_graph(tmp_path):
    class SafeParent(Echo):
        execution_requirements = ExecutionRequirements(
            recovery_safety=RecoverySafety.MODULE_BOUNDARY_RETRY,
            effect_safety=EffectSafety.EFFECT_FREE,
        )

        def __init__(self) -> None:
            super().__init__()
            self.child = Echo()

    async with SQLiteHistoryStore(tmp_path / "history.sqlite3") as history:
        runtime = LocalRuntime(history=history)
        preferred = binding(runtime).bind(SafeParent())
        assert preferred.durability.recovery_undeclared_modules == (
            "root.child",
        )
        assert preferred.durability.effect_unverified_modules == ("root.child",)

        with pytest.raises(ExecutionAdmissionError, match="root.child"):
            binding(runtime, DurabilityMode.REQUIRED).bind(SafeParent())
        await runtime.close()


def test_required_rejects_missing_but_preferred_reports_plan_downgrade():
    runtime = LocalRuntime()
    preferred = binding(runtime).bind(NeedsSQLiteDurability())
    assert preferred.durability.missing_capabilities == ("durability.sqlite",)
    assert preferred.durability.degraded_reasons

    with pytest.raises(ExecutionAdmissionError, match="durability.sqlite"):
        binding(runtime, DurabilityMode.REQUIRED).bind(NeedsSQLiteDurability())

    with pytest.raises(ExecutionAdmissionError, match="disables durability"):
        binding(runtime, DurabilityMode.DISABLED).bind(NeedsSQLiteDurability())


@pytest.mark.asyncio
async def test_sqlite_report_is_explicit_and_disabled_binding_does_not_persist(
    tmp_path,
):
    async with SQLiteHistoryStore(tmp_path / "history.sqlite3") as history:
        runtime = LocalRuntime(history=history)
        preferred_unqualified = binding(runtime).bind(Echo())
        assert preferred_unqualified.durability.effective_capabilities == (
            "durability.sqlite",
        )
        assert preferred_unqualified.durability.recovery_level == "none"
        assert preferred_unqualified.durability.checkpoint_policy == (
            "run_history_only"
        )
        assert preferred_unqualified.durability.replay_policy == "none"
        assert preferred_unqualified.durability.recovery_undeclared_modules == (
            "root",
        )
        assert preferred_unqualified.durability.effect_unverified_modules == (
            "root",
        )
        with pytest.raises(ExecutionAdmissionError, match="module-boundary retry safety"):
            binding(runtime, DurabilityMode.REQUIRED).bind(Echo())

        required = binding(runtime, DurabilityMode.REQUIRED).bind(
            NeedsSQLiteDurability()
        )
        report = required.durability
        assert report.effective_capabilities == ("durability.sqlite",)
        assert report.missing_capabilities == ()
        assert report.recovery_level == "module_boundary_retry"
        assert report.checkpoint_policy == "run_and_module_boundaries"
        assert report.replay_policy == "recorded_managed_effects"
        assert report.event_reconnect is True
        assert report.external_side_effect_guarantee == "at_least_once"
        assert report.arbitrary_coroutine_recovery is False
        assert report.exactly_once_external_side_effects is False

        preferred = binding(runtime).bind(NeedsManagedReplay())
        assert preferred.durability.effective_capabilities == (
            "durability.sqlite",
        )
        assert preferred.durability.missing_capabilities == (
            "durability.managed-replay",
        )
        with pytest.raises(ExecutionAdmissionError, match="durability.managed-replay"):
            binding(runtime, DurabilityMode.REQUIRED).bind(NeedsManagedReplay())

        disabled = binding(runtime, DurabilityMode.DISABLED).bind(Echo())
        handle = await disabled.start(UserMessage(content="local"), Context())
        await handle.result()
        assert await history.get_execution(handle.execution_id) is None
        with pytest.raises(RuntimeError, match="no effective durable recovery"):
            await runtime.recover(disabled, handle.execution_id)

        await runtime.close()
