"""Contract tests for portable Runtime execution plans."""

from __future__ import annotations

import json
import threading
from dataclasses import FrozenInstanceError

import pytest

from pygent import (
    AIMessage,
    ExponentialBackoff,
    FallbackPolicy,
    GenerationConfig,
    IdempotencyPolicy,
    ModelCallLayer,
    ModelErrorKind,
    ModelGroupConfig,
    ModelRoute,
    Module,
    ReActLayer,
    RetryPolicy,
    ToolCallLayer,
    ToolDefinition,
    ToolSideEffect,
    ToolSpec,
    UserMessage,
)
from pygent.core import (
    ExecutionRequirements,
    PlacementPolicy,
    RemoteModule,
)
from pygent.runtime import compile_execution_plan
from pygent.runtime.plan import (
    CodeArtifactSpec,
    ExecutionPlan,
    ModuleSpec,
    PlanIntegrityError,
    PlanValidationError,
    PlanVersionError,
)
from pygent.tool import ExecutorRegistry


def _portable_plan() -> ExecutionPlan:
    model = ModuleSpec(
        path="agent.model",
        type_name="ModelCallLayer",
        definition_id="pygent.llm.ModelCallLayer",
        input_schema="schema://message/user@1",
        output_schema="schema://message/ai@1",
        serializer="json",
        resource_keys=("model:assistant",),
        required_capabilities=("llm.openai-compatible",),
        placement_constraints=("pool=networked",),
        retry_policy_ref="policy://model/default@1",
        checkpoint_policy_ref="policy://checkpoint/module-boundary@1",
    )
    root = ModuleSpec(
        path="agent",
        type_name="CoordinatorAgent",
        definition_id="example.CoordinatorAgent",
        children=("agent.model",),
        input_schema="schema://message/user@1",
        output_schema="schema://message/ai@1",
        serializer="json",
    )
    return ExecutionPlan(
        root="agent",
        modules=(root, model),
        runtime_api_version="0.4",
        artifact=CodeArtifactSpec(
            package="pygent-example",
            version="1.4.0",
            digest="sha256:0123456789abcdef",
            entrypoint="examples.service:build_agent",
        ),
        metadata=(("environment", "test"),),
    )


def test_execution_plan_is_immutable_portable_and_json_serializable():
    plan = _portable_plan()

    assert plan.is_portable
    assert plan.plan_id == f"sha256:{plan.graph_hash}"
    assert json.loads(json.dumps(plan.to_dict()))["root"] == "agent"

    with pytest.raises(FrozenInstanceError):
        plan.root = "other"  # type: ignore[misc]


def test_execution_plan_round_trip_preserves_integrity():
    plan = _portable_plan()

    restored = ExecutionPlan.from_dict(plan.to_dict())

    assert restored == plan
    assert restored.graph_hash == plan.graph_hash


def test_graph_hash_covers_executable_contract_but_not_metadata():
    plan = _portable_plan()
    changed_metadata = ExecutionPlan(
        root=plan.root,
        modules=plan.modules,
        runtime_api_version=plan.runtime_api_version,
        artifact=plan.artifact,
        metadata=(("environment", "production"),),
    )
    changed_artifact = ExecutionPlan(
        root=plan.root,
        modules=plan.modules,
        runtime_api_version=plan.runtime_api_version,
        artifact=CodeArtifactSpec(
            package="pygent-example",
            version="1.4.1",
            digest="sha256:fedcba9876543210",
            entrypoint="examples.service:build_agent",
        ),
    )

    assert changed_metadata.graph_hash == plan.graph_hash
    assert changed_artifact.graph_hash != plan.graph_hash


def test_local_minimal_plan_remains_valid_but_is_not_portable():
    plan = ExecutionPlan(
        root="agent",
        modules=(ModuleSpec(path="agent", type_name="Agent"),),
    )

    assert not plan.is_portable


def test_constructor_normalizes_sequence_values_to_immutable_tuples():
    module = ModuleSpec(
        path="agent",
        type_name="Agent",
        resource_keys=["resource:a"],  # type: ignore[arg-type]
        metadata=[["owner", "team-a"]],  # type: ignore[arg-type]
    )
    plan = ExecutionPlan(
        root="agent",
        modules=[module],  # type: ignore[arg-type]
        metadata=[["environment", "test"]],  # type: ignore[arg-type]
    )

    assert module.resource_keys == ("resource:a",)
    assert module.metadata == (("owner", "team-a"),)
    assert plan.modules == (module,)
    assert plan.metadata == (("environment", "test"),)


def test_execution_plan_allows_shared_definition_nodes():
    shared = ModuleSpec("root.shared", "SharedLayer")
    left = ModuleSpec("root.left", "LeftAgent", children=("root.shared",))
    right = ModuleSpec("root.right", "RightAgent", children=("root.shared",))
    root = ModuleSpec(
        "root",
        "CoordinatorAgent",
        children=("root.left", "root.right"),
    )

    plan = ExecutionPlan(root="root", modules=(root, left, right, shared))

    assert plan.modules[-1] is shared


@pytest.mark.parametrize(
    "root, modules, match",
    [
        ("missing", (ModuleSpec("agent", "Agent"),), "root"),
        (
            "agent",
            (ModuleSpec("agent", "Agent"), ModuleSpec("agent", "Other")),
            "duplicate",
        ),
        (
            "agent",
            (ModuleSpec("agent", "Agent", children=("missing",)),),
            "unknown child",
        ),
        (
            "agent",
            (
                ModuleSpec("agent", "Agent", children=("agent.child",)),
                ModuleSpec("agent.child", "Child", children=("agent",)),
            ),
            "cycle",
        ),
    ],
)
def test_execution_plan_rejects_invalid_graphs(root, modules, match):
    with pytest.raises(PlanValidationError, match=match):
        ExecutionPlan(root=root, modules=modules)


def test_execution_plan_rejects_unsupported_schema_version():
    payload = _portable_plan().to_dict()
    payload["schema_version"] = 999

    with pytest.raises(PlanVersionError):
        ExecutionPlan.from_dict(payload)


def test_execution_plan_rejects_tampered_payload():
    payload = _portable_plan().to_dict()
    payload["modules"][0]["type_name"] = "InjectedAgent"

    with pytest.raises(PlanIntegrityError):
        ExecutionPlan.from_dict(payload)


def test_execution_plan_rejects_unknown_schema_fields():
    payload = _portable_plan().to_dict()
    payload["worker_command"] = "run-unhashed-command"

    with pytest.raises(PlanValidationError, match="unknown fields"):
        ExecutionPlan.from_dict(payload)


class _LiveInvoker:
    def __init__(self, secret: str) -> None:
        self.secret = secret
        self.lock = threading.Lock()


def _compiled_agent(
    *,
    model_name: str = "primary-model",
    temperature: float = 0.2,
    retry_attempts: int = 2,
    tool_resource: str = "tool:calculator",
    max_steps: int = 4,
    invoker_secret: str = "not-part-of-the-plan",
) -> ExecutionPlan:
    definition = ToolDefinition(
        name="calculate",
        description="Perform a deterministic calculation.",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    )
    tool = ToolSpec(
        tool_id="calculator",
        version="1",
        definition=definition,
        side_effect=ToolSideEffect.PURE,
        idempotency=IdempotencyPolicy.INHERENT,
        timeout=2.0,
        resource_key=tool_resource,
        sandbox_profile="restricted",
        required_permissions=("calculate",),
    )
    model = ModelCallLayer(
        model_group=ModelGroupConfig(
            name="assistant",
            routes=(ModelRoute("primary", "openai", model_name),),
            fallback=FallbackPolicy(("primary",)),
            max_concurrency=3,
        ),
        retry_policy=RetryPolicy(
            max_attempts_per_route=retry_attempts,
            retry_on=(ModelErrorKind.UNAVAILABLE,),
            backoff=ExponentialBackoff(0.1, 0.5),
        ),
        generation=GenerationConfig(
            temperature=temperature,
            max_output_tokens=128,
            response_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        ),
        tools=(definition,),
        invoker=_LiveInvoker(invoker_secret),  # type: ignore[arg-type]
    )
    tools = ToolCallLayer(
        tools=(tool,),
        executor_registry=ExecutorRegistry(),
        max_concurrency=2,
    )
    return compile_execution_plan(
        ReActLayer(
            model=model,
            tools=tools,
            max_steps=max_steps,
            max_model_calls=4,
            max_tool_calls=8,
        )
    )


@pytest.mark.parametrize(
    "changed",
    [
        {"model_name": "fallback-model"},
        {"temperature": 0.7},
        {"retry_attempts": 3},
        {"tool_resource": "tool:calculator-v2"},
        {"max_steps": 5},
    ],
)
def test_compiler_hash_covers_builtin_module_declarations(changed):
    baseline = _compiled_agent()

    different = _compiled_agent(**changed)

    assert different.graph_hash != baseline.graph_hash
    assert json.loads(json.dumps(different.to_dict()))["graph_hash"] == (
        different.graph_hash
    )


def test_compiler_identity_is_deterministic_and_excludes_live_adapters():
    first = _compiled_agent(invoker_secret="first-secret")
    second = _compiled_agent(invoker_secret="second-secret")
    serialized = json.dumps(first.to_dict(), sort_keys=True)

    assert first.graph_hash == second.graph_hash
    assert "first-secret" not in serialized
    assert "_LiveInvoker" not in serialized
    assert "ExecutorRegistry" not in serialized
    assert next(
        item for item in first.modules if item.type_name == "ModelCallLayer"
    ).resource_keys == ("model:assistant",)
    assert next(
        item for item in first.modules if item.type_name == "ToolCallLayer"
    ).resource_keys == ("tool:calculator",)


def test_compiler_requires_complete_portable_contract_and_hashes_it():
    class Echo(Module[UserMessage, AIMessage]):
        async def forward(self, message, context):
            return AIMessage(content=message.content), context

    artifact = CodeArtifactSpec(
        package="pygent-test",
        version="0.2.0",
        digest="sha256:artifact-one",
        entrypoint="tests.runtime.test_execution_plan:build_echo",
    )
    contract = {
        "artifact": artifact,
        "input_schema": "schema://message-context-input@0.3",
        "output_schema": "schema://message-context-output@0.3",
        "serializer": "pygent-json-v2",
    }
    plan = compile_execution_plan(Echo(), **contract)
    changed = compile_execution_plan(
        Echo(),
        **{
            **contract,
            "artifact": CodeArtifactSpec(
                package="pygent-test",
                version="0.2.0",
                digest="sha256:artifact-two",
                entrypoint="tests.runtime.test_execution_plan:build_echo",
            ),
        },
    )

    assert plan.is_portable
    assert all(module.is_portable for module in plan.modules)
    assert plan.artifact == artifact
    assert plan.graph_hash != changed.graph_hash
    with pytest.raises(ValueError, match="portable execution requires"):
        compile_execution_plan(Echo(), artifact=artifact)
    with pytest.raises(ValueError, match="wire schemas require"):
        compile_execution_plan(Echo(), input_schema="schema://input")


@pytest.mark.parametrize("mutable", [{"mode": "fast"}, ["fast"]])
def test_compiler_rejects_mutable_public_config_instead_of_ignoring(mutable):
    class MutableConfig(Module[UserMessage, AIMessage]):
        def __init__(self) -> None:
            super().__init__()
            self.settings = mutable

        async def forward(self, message, context):
            return AIMessage(content="ok"), context

    with pytest.raises(TypeError, match=r"MutableConfig\.settings.*immutable portable"):
        compile_execution_plan(MutableConfig())


def test_compiler_rejects_unknown_public_object_instead_of_ignoring():
    class UnknownConfig(Module[UserMessage, AIMessage]):
        def __init__(self) -> None:
            super().__init__()
            self.backend = object()

        async def forward(self, message, context):
            return AIMessage(content="ok"), context

    with pytest.raises(TypeError, match=r"UnknownConfig\.backend"):
        compile_execution_plan(UnknownConfig())


def test_explicit_live_resource_names_remain_outside_plan_identity():
    class UsesClient(Module[UserMessage, AIMessage]):
        trusted_live_resource_attributes = ("client",)

        def __init__(self, client: object) -> None:
            super().__init__()
            self.client = client
            self.mode = "stable"

        async def forward(self, message, context):
            return AIMessage(content=self.mode), context

    first = compile_execution_plan(UsesClient(object()))
    second = compile_execution_plan(UsesClient(object()))

    assert first.graph_hash == second.graph_hash


class _RequirementModule(Module[UserMessage, AIMessage]):
    def __init__(self, *, mode: str) -> None:
        super().__init__()
        self.mode = mode

    async def forward(self, message, context):
        return AIMessage(content=self.mode), context


def test_compiler_hash_covers_public_config_and_execution_requirements():
    baseline = _RequirementModule(mode="safe")
    changed_config = _RequirementModule(mode="fast")
    changed_deadline = _RequirementModule(mode="safe")
    changed_deadline.execution_requirements = ExecutionRequirements(
        requires_finite_deadline=True
    )
    changed_capability = _RequirementModule(mode="safe")
    changed_capability.execution_requirements = ExecutionRequirements(
        required_capabilities=("accelerator.gpu",)
    )

    baseline_hash = compile_execution_plan(baseline).graph_hash

    assert compile_execution_plan(changed_config).graph_hash != baseline_hash
    assert compile_execution_plan(changed_deadline).graph_hash != baseline_hash
    assert compile_execution_plan(changed_capability).graph_hash != baseline_hash


class _HookedModule(Module[UserMessage, AIMessage]):
    trusted_live_resource_attributes = ("client",)

    def __init__(self, version: int) -> None:
        super().__init__()
        self._version = version
        self.client = _LiveInvoker("ignored")

    def execution_plan_config(self):
        return {
            "version": self._version,
            "features": ["a", "b"],
            "routing": {"mode": "stable"},
        }

    async def forward(self, message, context):
        return AIMessage(content="ok"), context


def test_explicit_execution_plan_config_hook_is_strict_and_hashed():
    first = compile_execution_plan(_HookedModule(1))
    second = compile_execution_plan(_HookedModule(2))

    assert first.graph_hash != second.graph_hash

    class InvalidHook(_HookedModule):
        def execution_plan_config(self):
            return {"lock": threading.Lock()}

    with pytest.raises(TypeError, match="strict immutable JSON"):
        compile_execution_plan(InvalidHook(1))


def test_execution_plan_hook_does_not_hide_ordinary_immutable_attributes():
    class PartiallyDeclared(Module[UserMessage, AIMessage]):
        def __init__(self, mode: str) -> None:
            super().__init__()
            self.mode = mode

        def execution_plan_config(self):
            return {"protocol": "stable"}

        async def forward(self, message, context):
            return AIMessage(content=self.mode), context

    stable = compile_execution_plan(PartiallyDeclared("stable"))
    fast = compile_execution_plan(PartiallyDeclared("fast"))

    assert stable.graph_hash != fast.graph_hash


def test_remote_declaration_identity_and_capabilities_are_hashed():
    graph_hash = "1" * 64

    class Caller(Module[UserMessage, AIMessage]):
        def __init__(
            self,
            *,
            capabilities: tuple[str, ...],
            digest: str,
            placement: PlacementPolicy | None = None,
        ) -> None:
            super().__init__()
            self.remote = RemoteModule[UserMessage, AIMessage](
                binding_ref="reviewer",
                plan_id=f"sha256:{digest}",
                graph_hash=digest,
                required_capabilities=capabilities,
                placement=placement or PlacementPolicy.adaptive(),
            )

    baseline = compile_execution_plan(
        Caller(capabilities=("durable",), digest=graph_hash)
    )
    changed_capabilities = compile_execution_plan(
        Caller(capabilities=("durable", "gpu"), digest=graph_hash)
    )
    changed_plan = compile_execution_plan(
        Caller(capabilities=("durable",), digest="2" * 64)
    )
    changed_placement = compile_execution_plan(
        Caller(
            capabilities=("durable",),
            digest=graph_hash,
            placement=PlacementPolicy.pinned("reviewer-1"),
        )
    )
    remote = next(item for item in baseline.modules if item.path == "root.remote")

    assert baseline.graph_hash != changed_capabilities.graph_hash
    assert baseline.graph_hash != changed_plan.graph_hash
    assert baseline.graph_hash != changed_placement.graph_hash
    assert remote.required_capabilities == ("durable",)
    assert remote.definition_id == f"remote:reviewer@{graph_hash}"
    assert dict(remote.metadata)["remote_plan_id"] == f"sha256:{graph_hash}"
    assert remote.placement_constraints == ("mode=adaptive",)
