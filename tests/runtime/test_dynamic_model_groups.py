from __future__ import annotations

import asyncio
import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest

from pygent import (
    AIMessage,
    Context,
    FallbackPolicy,
    GenerationConfig,
    ModelCallLayer,
    ModelCallOptions,
    ModelCallPolicy,
    ModelGroupConfig,
    ModelRoute,
    RetryPolicy,
    UserMessage,
)
from pygent.core import FrozenJsonObject, Module, RemoteModule
from pygent.llm import (
    ModelDeploymentConflictError,
    ModelDeploymentUnavailableError,
    ModelExecution,
    ModelProfileSelectionError,
    ModelResourceOwnership,
    ModelResourceRef,
)
from pygent.llm.spi import ModelProviderResponse
from pygent.runtime import (
    CapacityPolicy,
    CapacityScope,
    ExecutionAdmissionError,
    ExecutionCapacityPolicy,
    ExecutionOptions,
    HTTPWorkerApp,
    HTTPWorkerClient,
    LocalRuntime,
    SQLiteHistoryStore,
    SQLiteModelDeploymentStore,
    WorkerRegistry,
    WorkerTarget,
    WorkerUnavailableError,
)


class _Invoker:
    def __init__(
        self,
        value: str,
        *,
        entered: asyncio.Event | None = None,
        release: asyncio.Event | None = None,
    ) -> None:
        self.value = value
        self.entered = entered
        self.release = release
        self.generations: list[GenerationConfig] = []

    def execute(self, **kwargs: object) -> ModelExecution:
        self.generations.append(kwargs["generation"])  # type: ignore[arg-type]

        async def operation(emit: object) -> ModelProviderResponse:
            del emit
            if self.entered is not None:
                self.entered.set()
            if self.release is not None:
                await self.release.wait()
            return ModelProviderResponse(AIMessage(content=self.value), {})

        return ModelExecution(operation)


def _requirement(name: str = "assistant") -> ModelGroupConfig:
    return ModelGroupConfig.deferred(
        name=name, max_concurrency=2, capacity_key=f"capacity:{name}"
    )


def _layer(
    requirement: ModelGroupConfig,
    *,
    allow_profile: bool = True,
) -> ModelCallLayer:
    return ModelCallLayer(
        model_group=requirement,
        retry_policy=RetryPolicy(),
        generation=GenerationConfig(temperature=0.7),
        policy=ModelCallPolicy(
            allow_profile_override=allow_profile,
            overridable_generation=frozenset({"temperature"}),
        ),
    )


async def _configure(handle: object, profile: str, invoker: _Invoker) -> None:
    await handle.configure(  # type: ignore[attr-defined]
        profile=profile,
        routes=(ModelRoute("main", "test", profile),),
        fallback=FallbackPolicy(("main",)),
        invoker=invoker,
    )


@pytest.mark.spec("dynamic-model-group", "DMG-TYPE-001")
def test_deferred_and_concrete_group_invariants_and_frozen_options() -> None:
    deferred = _requirement()
    assert deferred.is_deferred
    with pytest.raises(ValueError, match="concrete model group routes"):
        ModelGroupConfig("empty", (), FallbackPolicy(()))
    with pytest.raises(ValueError, match="deferred model group routes"):
        ModelGroupConfig(
            "bad",
            (ModelRoute("main", "test", "x"),),
            FallbackPolicy(("main",)),
            resolution=deferred.resolution,
        )

    options = ExecutionOptions(
        model_calls={"assistant": ModelCallOptions(profile="fast", temperature=0.2)}
    )
    assert isinstance(options.model_calls, FrozenJsonObject)
    with pytest.raises(TypeError):
        options.model_calls["assistant"] = {}  # type: ignore[index]


@pytest.mark.asyncio
@pytest.mark.spec("dynamic-model-group", "DMG-LOCAL-001")
async def test_default_and_per_execution_profile_with_generation_override() -> None:
    requirement = _requirement()
    runtime = LocalRuntime()
    bound = runtime.bind(_layer(requirement))
    group = bound.model_groups.get(requirement)
    standard = _Invoker("standard")
    fast = _Invoker("fast")
    await _configure(group, "standard", standard)
    await _configure(group, "fast", fast)
    await group.set_default("standard")

    first, _ = await bound.invoke(
        UserMessage(content="hello"),
        Context(),
        execution=ExecutionOptions(deadline=time.monotonic() + 2),
    )
    second, _ = await bound.invoke(
        UserMessage(content="hello"),
        Context(),
        execution=ExecutionOptions(
            deadline=time.monotonic() + 2,
            model_calls={
                "assistant": ModelCallOptions(profile="fast", temperature=0.2)
            },
        ),
    )

    assert first.content == "standard"
    assert second.content == "fast"
    assert fast.generations[-1].temperature == 0.2
    assert (await group.current()).profile == "standard"
    await runtime.close()


@pytest.mark.asyncio
@pytest.mark.spec("dynamic-model-group", "DMG-ADMISSION-001")
async def test_profile_update_does_not_change_an_admitted_execution() -> None:
    requirement = _requirement()
    runtime = LocalRuntime()
    bound = runtime.bind(_layer(requirement))
    group = bound.model_groups.get(requirement)
    entered, release = asyncio.Event(), asyncio.Event()
    old = _Invoker("old", entered=entered, release=release)
    await _configure(group, "default", old)
    await group.set_default("default")

    handle = await bound.start(
        UserMessage(content="hello"),
        Context(),
        execution=ExecutionOptions(deadline=time.monotonic() + 2),
    )
    await entered.wait()
    await _configure(group, "default", _Invoker("new"))
    release.set()

    output, _ = await handle.result()
    assert output.content == "old"
    await runtime.close()


@pytest.mark.asyncio
@pytest.mark.spec("dynamic-model-group", "DMG-FAIL-CLOSED-001")
async def test_profile_and_generation_policy_fail_closed() -> None:
    requirement = _requirement()
    runtime = LocalRuntime()
    bound = runtime.bind(_layer(requirement, allow_profile=False))
    group = bound.model_groups.get(requirement)
    await _configure(group, "default", _Invoker("default"))
    await group.set_default("default")

    with pytest.raises(ModelProfileSelectionError, match="does not allow profile"):
        await bound.start(
            UserMessage(content="x"),
            Context(),
            execution=ExecutionOptions(
                deadline=time.monotonic() + 2,
                model_calls={"assistant": ModelCallOptions(profile="default")},
            ),
        )
    with pytest.raises(ExecutionAdmissionError, match="max_output_tokens"):
        await bound.start(
            UserMessage(content="x"),
            Context(),
            execution=ExecutionOptions(
                deadline=time.monotonic() + 2,
                model_calls={
                    "assistant": ModelCallOptions(max_output_tokens=12)
                },
            ),
        )
    await runtime.close()


@pytest.mark.asyncio
@pytest.mark.spec("dynamic-model-group", "DMG-BINDING-001")
async def test_binding_scopes_are_isolated_for_the_same_requirement() -> None:
    requirement = _requirement()
    runtime = LocalRuntime()
    execution_capacity = ExecutionCapacityPolicy(
        scope=CapacityScope.RUNTIME_INSTANCE,
        max_live_executions=2,
        max_runnable_executions=2,
        max_queue_size=2,
        max_waiters=2,
        max_child_depth=4,
        max_children_per_execution=4,
    )
    first = runtime.create_binding(
        name="first",
        execution_capacity=execution_capacity,
        model_capacity=CapacityPolicy.passthrough(),
        tool_capacity=CapacityPolicy.passthrough(),
    ).bind(_layer(requirement))
    second = runtime.create_binding(
        name="second",
        execution_capacity=execution_capacity,
        model_capacity=CapacityPolicy.passthrough(),
        tool_capacity=CapacityPolicy.passthrough(),
    ).bind(_layer(requirement))
    first_group = first.model_groups.get(requirement)
    second_group = second.model_groups.get(requirement)
    await _configure(first_group, "default", _Invoker("first"))
    await first_group.set_default("default")
    assert first_group.deployment_scope_id != second_group.deployment_scope_id
    with pytest.raises(
        ModelDeploymentUnavailableError, match="no selected or default profile"
    ):
        await second.start(
            UserMessage(content="x"),
            Context(),
            execution=ExecutionOptions(deadline=time.monotonic() + 2),
        )
    await runtime.close()


@pytest.mark.asyncio
@pytest.mark.spec("dynamic-model-group", "DMG-SQLITE-001")
async def test_history_v3_migrates_to_v4_and_keeps_execution(tmp_path: Path) -> None:
    path = tmp_path / "history.sqlite3"
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE executions (execution_id TEXT PRIMARY KEY,request_id TEXT NOT NULL,"
        "status TEXT NOT NULL,plan_id TEXT NOT NULL,input_json TEXT NOT NULL,"
        "output_json TEXT,error_json TEXT,attempt INTEGER NOT NULL DEFAULT 1,"
        "binding_id TEXT NOT NULL DEFAULT '',identity TEXT NOT NULL DEFAULT '',"
        "idempotency_key TEXT,updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
    )
    db.execute(
        "INSERT INTO executions(execution_id,request_id,status,plan_id,input_json) "
        "VALUES('run','request','queued','plan','{}')"
    )
    db.execute("PRAGMA user_version=3")
    db.commit()
    db.close()

    async with SQLiteHistoryStore(path) as history:
        stored = await history.get_execution("run")
        assert stored is not None
        assert stored.model_admission_status == "none"
    db = sqlite3.connect(path)
    assert db.execute("PRAGMA user_version").fetchone()[0] == 4
    db.close()


@pytest.mark.asyncio
@pytest.mark.spec("dynamic-model-group", "DMG-STORE-001")
async def test_sqlite_store_reuses_exact_admission_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "models.sqlite3"
    requirement = _requirement()
    first = await SQLiteModelDeploymentStore(path).open()
    second = await SQLiteModelDeploymentStore(path).open()
    runtime = LocalRuntime(model_deployment_store=first)
    bound = runtime.bind(_layer(requirement))
    group = bound.model_groups.get(requirement)
    await _configure(group, "default", _Invoker("resident"))
    await group.set_default("default")
    admission = await first.admit(
        group.deployment_scope_id,
        (requirement.name,),
        {requirement.name: None},
        admission_id="stable",
    )
    restored = await second.get_admission("stable")
    assert restored is not None
    assert restored.digest == admission.digest
    assert restored.snapshot(requirement.name).snapshot_id == admission.snapshot(
        requirement.name
    ).snapshot_id
    await runtime.close()
    await second.close()


class _TwoModels(Module[UserMessage, AIMessage]):
    def __init__(self, first: ModelCallLayer, second: ModelCallLayer) -> None:
        super().__init__()
        self.first = first
        self.second = second

    async def forward(
        self, message: UserMessage, context: Context
    ) -> tuple[AIMessage, Context]:
        answer, context = await self.first(message, context)
        return await self.second(answer, context)


@pytest.mark.asyncio
@pytest.mark.spec("dynamic-model-group", "DMG-ADMISSION-002")
async def test_multi_group_admission_fails_before_any_model_call() -> None:
    first_requirement = _requirement("first")
    second_requirement = _requirement("second")
    runtime = LocalRuntime()
    bound = runtime.bind(
        _TwoModels(_layer(first_requirement), _layer(second_requirement))
    )
    invoker = _Invoker("unused")
    first = bound.model_groups.get(first_requirement)
    await _configure(first, "default", invoker)
    await first.set_default("default")

    with pytest.raises(
        ModelDeploymentUnavailableError, match="no selected or default profile"
    ):
        await bound.start(
            UserMessage(content="x"),
            Context(),
            execution=ExecutionOptions(deadline=time.monotonic() + 2),
        )
    assert invoker.generations == []
    await runtime.close()


@pytest.mark.asyncio
@pytest.mark.spec("dynamic-model-group", "DMG-DIRECT-001")
async def test_direct_deferred_and_fixed_profile_override_fail_closed() -> None:
    deferred = _layer(_requirement())
    with pytest.raises(RuntimeError, match="deferred ModelGroup requires a managed"):
        await deferred.invoke(UserMessage(content="x"), Context())

    fixed = ModelCallLayer(
        model_group=ModelGroupConfig(
            "fixed",
            (ModelRoute("main", "test", "fixed"),),
            FallbackPolicy(("main",)),
        ),
        retry_policy=RetryPolicy(),
        generation=GenerationConfig(),
        policy=ModelCallPolicy(allow_profile_override=True),
        invoker=_Invoker("fixed"),
    )
    with pytest.raises(ValueError, match="fixed model group.*cannot select"):
        await fixed.invoke(
            UserMessage(content="x"),
            Context(),
            execution=ExecutionOptions(
                model_calls={"fixed": ModelCallOptions(profile="other")}
            ),
        )


class _Resolver:
    resolver_id = "test-resolver"
    coordinator_domain = "test-domain"

    def __init__(self, invoker: _Invoker) -> None:
        self.invoker = invoker
        self.validated = 0
        self.leases = 0

    async def validate(self, model_group: ModelGroupConfig, resources: object) -> None:
        del model_group, resources
        self.validated += 1

    @asynccontextmanager
    async def acquire(self, model_group: ModelGroupConfig, resources: object):
        del model_group, resources
        self.leases += 1
        yield self.invoker


@pytest.mark.asyncio
@pytest.mark.spec("dynamic-model-group", "DMG-DURABLE-001")
async def test_durable_manifest_is_committed_and_purge_releases_it(
    tmp_path: Path,
) -> None:
    history_path = tmp_path / "history.sqlite3"
    models_path = tmp_path / "models.sqlite3"
    requirement = _requirement()
    async with SQLiteHistoryStore(history_path) as history:
        store = await SQLiteModelDeploymentStore(models_path).open()
        runtime = LocalRuntime(history=history, model_deployment_store=store)
        resolver = _Resolver(_Invoker("durable"))
        runtime.register_model_resource_resolver(resolver)
        bound = runtime.bind(_layer(requirement))
        group = bound.model_groups.get(requirement)
        ref = ModelResourceRef(
            resolver_id=resolver.resolver_id,
            resource_id="credential-alias",
            revision="revision-1",
            capacity_owner_id="owner-1",
            coordinator_domain=resolver.coordinator_domain,
        )
        await group.configure(
            profile="default",
            routes=(ModelRoute("main", "test", "durable"),),
            fallback=FallbackPolicy(("main",)),
            resource_ref=ref,
        )
        await group.set_default("default")
        handle = await bound.start(
            UserMessage(content="x"),
            Context(),
            execution=ExecutionOptions(deadline=time.monotonic() + 2),
        )
        output, _ = await handle.result()
        stored = await history.get_execution(handle.execution_id)
        assert output.content == "durable"
        assert stored is not None
        assert stored.model_admission_status == "committed"
        assert stored.model_admission_id == handle.execution_id
        assert resolver.validated == 1
        assert resolver.leases == 1

        await runtime.purge_execution(handle.execution_id)
        assert await store.get_admission(handle.execution_id) is None
        assert await history.get_execution(handle.execution_id) is None
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.spec("dynamic-model-group", "DMG-RETIRE-001")
async def test_retiring_default_requires_atomic_replacement() -> None:
    requirement = _requirement()
    runtime = LocalRuntime()
    bound = runtime.bind(_layer(requirement))
    group = bound.model_groups.get(requirement)
    await _configure(group, "old", _Invoker("old"))
    await _configure(group, "new", _Invoker("new"))
    await group.set_default("old")
    with pytest.raises(ModelDeploymentConflictError, match="without a replacement"):
        await group.retire("old")
    await group.retire("old", replacement_default="new")
    assert await group.list_profiles() == ("new",)
    assert (await group.current()).profile == "new"
    await runtime.close()


class _RemoteCaller(Module[UserMessage, AIMessage]):
    def __init__(self, child: RemoteModule[UserMessage, AIMessage]) -> None:
        super().__init__()
        self.child = child

    async def forward(
        self, message: UserMessage, context: Context
    ) -> tuple[AIMessage, Context]:
        return await self.child(message, context)


@pytest.mark.asyncio
@pytest.mark.spec("dynamic-model-group", "DMG-CHILD-001")
async def test_independent_binding_child_uses_its_own_model_admission() -> None:
    requirement = _requirement()
    runtime = LocalRuntime()
    capacity = ExecutionCapacityPolicy(
        scope=CapacityScope.RUNTIME_INSTANCE,
        max_live_executions=2,
        max_runnable_executions=2,
        max_queue_size=2,
        max_waiters=2,
        max_child_depth=4,
        max_children_per_execution=4,
    )
    child = runtime.create_binding(
        name="child",
        execution_capacity=capacity,
        model_capacity=CapacityPolicy.passthrough(),
        tool_capacity=CapacityPolicy.passthrough(),
    ).bind(_layer(requirement))
    parent_binding = runtime.create_binding(
        name="parent",
        execution_capacity=capacity,
        model_capacity=CapacityPolicy.passthrough(),
        tool_capacity=CapacityPolicy.passthrough(),
    )
    runtime.register_remote("dynamic-child", child)
    parent = parent_binding.bind(
        _RemoteCaller(
            RemoteModule[UserMessage, AIMessage](binding_ref="dynamic-child")
        )
    )
    group = child.model_groups.get(requirement)
    await _configure(group, "default", _Invoker("child-default"))
    await group.set_default("default")

    output, _ = await parent.invoke(
        UserMessage(content="x"),
        Context(),
        execution=ExecutionOptions(deadline=time.monotonic() + 2),
    )
    assert output.content == "child-default"
    await runtime.close()


@pytest.mark.asyncio
@pytest.mark.spec("dynamic-model-group", "DMG-WORKER-001")
async def test_worker_exact_pin_extension_round_trips_and_old_worker_rejects() -> None:
    observed: dict[str, object] = {}

    async def handler(invocation: object, event_sink: object) -> object:
        del event_sink
        observed["model_calls"] = invocation.model_calls  # type: ignore[attr-defined]
        observed["admission"] = invocation.model_admission_ref  # type: ignore[attr-defined]
        return {"ok": True}

    capability = "model.deferred.exact-pin.v1"
    worker = HTTPWorkerApp(
        handler,
        capabilities=("local", capability),
        model_store_namespace="store:test",
    )
    registry = WorkerRegistry()
    target = WorkerTarget("new", "http://worker", ("local", capability))
    registry.publish("dynamic", (target,))
    graph_hash = "0" * 64
    async with HTTPWorkerClient(
        registry, transport=httpx.ASGITransport(app=worker.app)
    ) as client:
        _, result = await client.invoke(
            "dynamic",
            {"input": "x"},
            request_id="stable-request",
            plan_id=f"sha256:{graph_hash}",
            graph_hash=graph_hash,
            required_capabilities=(capability,),
            model_calls={"assistant": {"profile": "quality"}},
            model_admission_ref="stable-request",
        )
    assert result["ok"] is True  # type: ignore[index]
    assert observed["admission"] == "stable-request"
    assert observed["model_calls"]["assistant"]["profile"] == "quality"  # type: ignore[index]

    old_worker = HTTPWorkerApp(handler, capabilities=("local",))
    old_registry = WorkerRegistry()
    old_registry.publish(
        "dynamic",
        (WorkerTarget("old", "http://old", ("local", capability)),),
    )
    async with HTTPWorkerClient(
        old_registry, transport=httpx.ASGITransport(app=old_worker.app)
    ) as client:
        with pytest.raises(WorkerUnavailableError):
            await client.start(
                "dynamic",
                {},
                request_id="stable-request",
                plan_id=f"sha256:{graph_hash}",
                graph_hash=graph_hash,
                required_capabilities=(capability,),
                model_admission_ref="stable-request",
            )


class _ClosableInvoker(_Invoker):
    def __init__(self, value: str) -> None:
        super().__init__(value)
        self.closes = 0

    async def aclose(self) -> None:
        self.closes += 1


@pytest.mark.asyncio
@pytest.mark.spec("dynamic-model-group", "DMG-LIFECYCLE-001")
async def test_runtime_closes_owned_resident_once_and_never_closes_borrowed() -> None:
    requirement = _requirement()
    runtime = LocalRuntime()
    bound = runtime.bind(_layer(requirement))
    group = bound.model_groups.get(requirement)
    owned = _ClosableInvoker("owned")
    borrowed = _ClosableInvoker("borrowed")
    routes = (ModelRoute("main", "test", "x"),)
    await group.configure(
        profile="owned-a",
        routes=routes,
        fallback=FallbackPolicy(("main",)),
        invoker=owned,
        ownership=ModelResourceOwnership.OWNED,
    )
    await group.configure(
        profile="owned-b",
        routes=routes,
        fallback=FallbackPolicy(("main",)),
        invoker=owned,
        ownership=ModelResourceOwnership.OWNED,
    )
    await group.configure(
        profile="borrowed",
        routes=routes,
        fallback=FallbackPolicy(("main",)),
        invoker=borrowed,
    )
    await runtime.close()
    assert owned.closes == 1
    assert borrowed.closes == 0
