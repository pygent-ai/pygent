"""Asyncio reference Runtime for local managed execution.

The implementation is split by responsibility under pygent.runtime._local;
this module remains the stable import facade and owns Runtime composition.
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from collections.abc import Awaitable, Callable, Iterable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, TypeVar

from pygent.core import CapacityPermit, JsonValue, Message, Module
from pygent.core._module_contracts import _capacity_permit
from pygent.llm import (
    ModelDeploymentUnavailableError,
    ModelProfileSnapshot,
    ModelResourceOwnership,
)
from pygent.tool import (
    ExecutorRegistry,
    ToolCall,
    ToolExecutionContext,
    ToolExecutor,
    ToolSpec,
    ToolTaskManager,
)
from pygent.tool.executors import validate_executor_sandbox

from ._history_store import SQLiteHistoryStore
from ._local.capacity import (
    InMemoryCapacityCoordinator,
    _BindingState,
    _ExecutionCapacityState,
    _ResourceGate,
    _RunnableGate,
)
from ._local.handles import RuntimeBinding, _LocalBoundModule
from ._local.lifecycle import _LifecycleMixin
from ._local.policies import _apply_binding_policy, _collect_graph
from ._local.recovery import _RecoveryMixin
from ._local.state import _ExecutionRecord
from ._local.tool_jobs import _ToolJobsMixin
from .api import (
    Binding,
    CapacityCoordinator,
    CapacityPolicy,
    CapacityScope,
    DurabilityMode,
    DurabilityPolicy,
    DurabilityReport,
    ExecutionAdmissionError,
    ExecutionCapacityPolicy,
    ResourceCapacityGate,
    RuntimeClosedError,
)
from .compiler import compile_execution_plan
from .model_deployment import InMemoryModelDeploymentStore, ModelDeploymentStore
from .plan import CodeArtifactSpec, ExecutionPlan

InputMessageT = TypeVar("InputMessageT", bound=Message)
OutputMessageT = TypeVar("OutputMessageT", bound=Message)


class LocalRuntime(_LifecycleMixin, _RecoveryMixin, _ToolJobsMixin):
    """Process-local asyncio Runtime with bounded admission and execution handles."""

    _closed: bool
    _base_capabilities: frozenset[str]

    def __init__(
        self,
        *,
        history: SQLiteHistoryStore | None = None,
        capabilities: tuple[str, ...] = (),
        capacity_coordinator: CapacityCoordinator | None = None,
        code_artifact: CodeArtifactSpec | None = None,
        input_schema: str | None = None,
        output_schema: str | None = None,
        serializer: str | None = None,
        model_deployment_store: ModelDeploymentStore | None = None,
        deployment_namespace: str = "default",
        max_retained_executions: int = 1024,
    ) -> None:
        wire_values = (input_schema, output_schema, serializer)
        if code_artifact is None and any(value is not None for value in wire_values):
            raise ValueError("wire schemas require an explicit code_artifact")
        if code_artifact is not None and any(value is None for value in wire_values):
            raise ValueError(
                "portable Runtime requires input_schema, output_schema, and serializer"
            )
        if (
            not isinstance(max_retained_executions, int)
            or isinstance(max_retained_executions, bool)
            or max_retained_executions <= 0
        ):
            raise ValueError("max_retained_executions must be a positive integer")
        self._closed = False
        self.max_retained_executions = max_retained_executions
        self._binding_states: dict[object, _BindingState] = {}
        self._shared_resource_gates: dict[
            tuple[str, str], tuple[CapacityPolicy, _ResourceGate]
        ] = {}
        self._executions: dict[str, _ExecutionRecord] = {}
        self._idempotency_records: dict[
            tuple[str, str, str], tuple[str, _ExecutionRecord]
        ] = {}
        self._remote_modules: dict[str, Any] = {}
        self._external_waiters: dict[
            tuple[str, str], tuple[_ExecutionRecord, asyncio.Future[Mapping[str, JsonValue]]]
        ] = {}
        self._external_lock = asyncio.Lock()
        self._tool_tasks: ToolTaskManager | None = None
        self._model_invokers: dict[str, Any] = {}
        self.model_deployment_store = (
            model_deployment_store or InMemoryModelDeploymentStore()
        )
        if not isinstance(deployment_namespace, str) or not deployment_namespace:
            raise ValueError("deployment_namespace must be a non-empty string")
        self.deployment_namespace = deployment_namespace
        self._model_store_opened = False
        self._model_store_open_task: asyncio.Task[None] | None = None
        self._profile_publications: dict[
            tuple[str, str, str, str], asyncio.Task[Any]
        ] = {}
        self._model_resource_resolvers: dict[str, Any] = {}
        self._resident_model_invokers: dict[
            str, tuple[Any, ModelResourceOwnership]
        ] = {}
        self._tool_registry: Any | None = None
        self.history = history
        self.capacity_coordinator = capacity_coordinator
        self.code_artifact = code_artifact
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.serializer = serializer
        self._recovery_owner_id = uuid.uuid4().hex
        self._recovery_lease_ttl = 30.0
        built_in = {"runtime.asyncio", "runtime.local", "external.wait"}
        if history is not None:
            built_in.add("durability.sqlite")
        if any(value.startswith("tool.sandbox.") for value in capabilities):
            raise ValueError(
                "sandbox capabilities are derived from registered executors; "
                "use runtime.register_tool()"
            )
        built_in.update(capabilities)
        self._base_capabilities = frozenset(built_in)

    @property
    def capabilities(self) -> frozenset[str]:
        sandbox = getattr(self._tool_registry, "sandbox_capabilities", ())
        return self._base_capabilities | frozenset(sandbox)

    def create_binding(
        self,
        *,
        name: str,
        execution_capacity: ExecutionCapacityPolicy,
        model_capacity: CapacityPolicy,
        tool_capacity: CapacityPolicy,
        durability: DurabilityPolicy | None = None,
        metadata: tuple[tuple[str, str], ...] = (),
    ) -> RuntimeBinding:
        if self._closed:
            raise RuntimeClosedError("Runtime is closed")
        policy = Binding(
            name=name,
            execution_capacity=execution_capacity,
            model_capacity=model_capacity,
            tool_capacity=tool_capacity,
            durability=durability or DurabilityPolicy(),
            metadata=metadata,
        )
        state = self._state_for(policy)
        return RuntimeBinding(self, policy, state.deployment_scope_id)

    def _deployment_scope_id(self, binding: Binding) -> str:
        payload = f"{self.deployment_namespace}\0{binding!r}".encode()
        return "model-scope:" + hashlib.sha256(payload).hexdigest()

    async def _ensure_model_store_open(self) -> None:
        if self._model_store_opened:
            return
        task = self._model_store_open_task
        if task is None:
            async def open_once() -> None:
                open_store = getattr(self.model_deployment_store, "open", None)
                if callable(open_store):
                    await open_store()

            task = asyncio.create_task(open_once(), name="pygent-model-store-open")
            self._model_store_open_task = task
        try:
            await asyncio.shield(task)
        except BaseException:
            if task.done() and self._model_store_open_task is task:
                self._model_store_open_task = None
            raise
        self._model_store_opened = True

    def register_model_resource_resolver(self, resolver: Any) -> None:
        resolver_id = getattr(resolver, "resolver_id", None)
        if not isinstance(resolver_id, str) or not resolver_id:
            raise TypeError("model resource resolver requires a resolver_id")
        current = self._model_resource_resolvers.get(resolver_id)
        if current is not None and current is not resolver:
            raise ValueError(
                f"model resource resolver {resolver_id!r} is already registered"
            )
        self._model_resource_resolvers[resolver_id] = resolver

    @asynccontextmanager
    async def _model_deployment_lease(self, deployment: object) -> Any:
        if not isinstance(deployment, ModelProfileSnapshot):
            raise TypeError("deployment must be a ModelProfileSnapshot")
        resident = self._resident_model_invokers.get(deployment.snapshot_id)
        if resident is not None:
            yield resident[0]
            return
        resources = deployment.resources
        if resources is None:
            raise ModelDeploymentUnavailableError(
                f"model profile {deployment.profile!r} has no live invoker"
            )
        resolver = self._model_resource_resolvers.get(resources.resolver_id)
        if resolver is None:
            raise ModelDeploymentUnavailableError(
                f"model resource resolver {resources.resolver_id!r} is unavailable"
            )
        lease = resolver.acquire(deployment.model_group, resources)
        async with lease as invoker:
            yield invoker

    def _state_for(self, binding: Binding) -> _BindingState:
        key: object = (
            ("deployment", binding.name)
            if binding.execution_capacity.scope is CapacityScope.DEPLOYMENT
            else ("runtime_instance", id(binding))
        )
        state = self._binding_states.get(key)
        if state is None:
            if binding.execution_capacity.scope is CapacityScope.DEPLOYMENT:
                coordinator = self.capacity_coordinator
                if coordinator is None:
                    raise ExecutionAdmissionError(
                        "deployment execution capacity requires a shared capacity_coordinator"
                    )
                execution_state = coordinator.execution_state(binding.name, binding.execution_capacity)
            else:
                execution_state = _ExecutionCapacityState(binding.name, binding.execution_capacity)
            state = _BindingState(
                binding,
                self._deployment_scope_id(binding),
                execution_state,
                self._capacity_gate("model", binding.model_capacity),
                self._capacity_gate("tool", binding.tool_capacity),
            )
            self._binding_states[key] = state
        elif state.policy != binding:
            raise ValueError(
                f"Binding {binding.name!r} reuses a deployment identity "
                "with different capacity policy"
            )
        return state

    def _capacity_gate(self, kind: str, policy: CapacityPolicy) -> ResourceCapacityGate:
        if policy.scope is CapacityScope.DEPLOYMENT:
            coordinator = self.capacity_coordinator
            if coordinator is None:
                raise ExecutionAdmissionError(
                    f"deployment {kind} capacity requires a shared capacity_coordinator"
                )
            assert policy.capacity_key is not None
            return coordinator.resource_gate(kind, policy.capacity_key, policy)
        if (
            policy.scope is CapacityScope.RUNTIME_INSTANCE
            and policy.capacity_key is not None
        ):
            return self._shared_resource_gate(kind, policy.capacity_key, policy)
        return _ResourceGate(policy, kind)

    def _tool_gates(
        self, binding_state: _BindingState, resource_key: str | None
    ) -> list[ResourceCapacityGate]:
        """Resolve the same Binding/resource Tool gates for sync and detached work."""

        gates = [binding_state.tool]
        policy = binding_state.policy.tool_capacity
        if (
            resource_key is not None
            and policy.max_concurrency is not None
            and policy.capacity_key is None
        ):
            gates.append(self._shared_resource_gate("tool", resource_key, policy))
        return gates

    def _detached_tool_execution(
        self, binding_state: _BindingState, registry: Any
    ) -> Callable[[ToolSpec, ToolCall, ToolExecutionContext], Awaitable[object]]:
        async def execute(
            spec: ToolSpec, call: ToolCall, context: ToolExecutionContext
        ) -> object:
            # A Job/ToolTask is independent from the former Parent's runnable
            # lease, but it must re-enter the admitting Binding's Tool plane.
            gates = self._tool_gates(binding_state, spec.resource_key)
            async with AsyncExitStack() as stack:
                permits: list[CapacityPermit] = []
                for gate in tuple(dict.fromkeys(gates)):
                    permits.append(await stack.enter_async_context(gate.permit()))
                permit = next(
                    (
                        item
                        for item in reversed(permits)
                        if item.fencing_token is not None
                    ),
                    permits[-1],
                )
                token = _capacity_permit.set(permit)
                try:
                    managed_context = ToolExecutionContext(
                        deadline=context.deadline,
                        emit=context.emit,
                        execution_id=context.execution_id
                        or f"detached:{context.task_id or call.call_id}",
                        task_id=context.task_id,
                        recovery=context.recovery,
                    )
                    return await registry.execute(spec, call, managed_context)
                finally:
                    _capacity_permit.reset(token)

        return execute

    def _shared_resource_gate(
        self,
        kind: str,
        capacity_key: str,
        policy: CapacityPolicy,
    ) -> ResourceCapacityGate:
        if policy.scope is CapacityScope.DEPLOYMENT:
            coordinator = self.capacity_coordinator
            if coordinator is None:
                raise ExecutionAdmissionError(
                    f"deployment {kind} capacity requires a shared capacity_coordinator"
                )
            return coordinator.resource_gate(kind, capacity_key, policy)
        if policy.scope is CapacityScope.EXTERNAL_RESOURCE:
            return _ResourceGate(policy, f"external:{kind}:{capacity_key}")
        key = (kind, capacity_key)
        current = self._shared_resource_gates.get(key)
        if current is None:
            gate = _ResourceGate(policy, f"{kind}:{capacity_key}")
            self._shared_resource_gates[key] = (policy, gate)
            return gate
        current_policy, gate = current
        if current_policy != policy:
            raise ValueError(
                f"shared {kind} capacity {capacity_key!r} has conflicting policies"
            )
        return gate

    def _compile_plan(self, module: Module[Any, Any]) -> ExecutionPlan:
        return compile_execution_plan(
            module,
            artifact=self.code_artifact,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            serializer=self.serializer,
        )

    def bind(
        self,
        module: Module[InputMessageT, OutputMessageT],
        *,
        binding: Binding | RuntimeBinding | None = None,
    ) -> _LocalBoundModule[InputMessageT, OutputMessageT]:
        if self._closed:
            raise RuntimeClosedError("Runtime is closed")
        if not isinstance(module, Module):
            raise TypeError("module must be a Module")
        module._freeze_definition()
        if isinstance(binding, RuntimeBinding):
            if binding.runtime is not self:
                raise ValueError("managed Binding belongs to another Runtime")
            binding = binding.policy
        if binding is None:
            binding = Binding(
                name=f"local:{type(module).__name__}",
                execution_capacity=ExecutionCapacityPolicy(
                    scope=CapacityScope.RUNTIME_INSTANCE,
                    max_live_executions=128,
                    max_runnable_executions=32,
                    max_queue_size=128,
                    max_waiters=128,
                    max_child_depth=32,
                    max_children_per_execution=1024,
                ),
                model_capacity=CapacityPolicy.passthrough(),
                tool_capacity=CapacityPolicy.passthrough(),
            )
        plan = self._compile_plan(module)
        plan = _apply_binding_policy(plan, binding)
        required = {
            capability
            for spec in plan.modules
            for capability in spec.required_capabilities
            if not dict(spec.metadata).get("binding")
            and not dict(spec.metadata).get("binding_ref")
        }
        durability_required = {
            capability
            for capability in required
            if capability.startswith("durability.")
        }
        hard_missing = sorted((required - durability_required) - self.capabilities)
        if hard_missing:
            raise ExecutionAdmissionError(
                "Runtime does not provide required capabilities: "
                + ", ".join(hard_missing)
            )
        sandbox_gaps = self._sandbox_binding_gaps(module)
        durability_report = self._durability_report(
            binding,
            durability_required,
            plan,
            sandbox_gaps=sandbox_gaps,
        )
        if binding.durability.mode is DurabilityMode.DISABLED and durability_required:
            raise ExecutionAdmissionError(
                "Binding disables durability required by the ExecutionPlan: "
                + ", ".join(sorted(durability_required))
            )
        if binding.durability.mode is DurabilityMode.REQUIRED and (
            durability_report.missing_capabilities
            or durability_report.recovery_undeclared_modules
            or durability_report.effect_unverified_modules
        ):
            gaps = [
                *(
                    f"missing capability {item}"
                    for item in durability_report.missing_capabilities
                ),
                *(
                    f"{item} does not declare module-boundary retry safety"
                    for item in durability_report.recovery_undeclared_modules
                ),
                *(
                    f"{item} has unverified unmanaged effects"
                    for item in durability_report.effect_unverified_modules
                ),
            ]
            raise ExecutionAdmissionError(
                "Binding cannot satisfy required durability eligibility: "
                + "; ".join(gaps)
            )
        binding_state = self._state_for(binding)
        return _LocalBoundModule(
            self,
            module,
            binding,
            plan,
            _collect_graph(module),
            durability_report,
            binding_state.deployment_scope_id,
        )

    def _durability_report(
        self,
        binding: Binding,
        plan_requirements: set[str],
        plan: ExecutionPlan,
        *,
        sandbox_gaps: tuple[str, ...] = (),
    ) -> DurabilityReport:
        mode = binding.durability.mode
        recovery_undeclared = tuple(
            spec.path
            for spec in plan.modules
            if dict(spec.metadata).get("recovery_safety") != "module_boundary_retry"
        )
        effect_unverified = tuple(
            spec.path
            for spec in plan.modules
            if dict(spec.metadata).get("effect_safety")
            not in ("effect_free", "managed_effects")
        )
        if mode is DurabilityMode.DISABLED:
            return DurabilityReport(
                requested_mode=mode,
                effective_capabilities=(),
                missing_capabilities=tuple(sorted(plan_requirements)),
                recovery_level="none",
                checkpoint_policy="none",
                replay_policy="none",
                event_reconnect=False,
                capacity_scope=binding.execution_capacity.scope,
                recovery_undeclared_modules=recovery_undeclared,
                effect_unverified_modules=effect_unverified,
                detached_tool_gaps=sandbox_gaps,
            )

        requested = {"durability.sqlite", *plan_requirements}
        available = set(self.capabilities)
        if self.history is None:
            available.discard("durability.sqlite")
        effective = tuple(sorted(requested & available))
        missing = tuple(sorted(requested - available))
        degraded_reasons = (
            *(f"missing capability: {capability}" for capability in missing),
            *(
                f"module {path} has not declared module-boundary retry safety"
                for path in recovery_undeclared
            ),
            *(
                f"module {path} has unverified unmanaged effects"
                for path in effect_unverified
            ),
        )
        sqlite_enabled = "durability.sqlite" in effective
        graph_eligible = not recovery_undeclared and not effect_unverified
        recovery_enabled = sqlite_enabled and graph_eligible
        return DurabilityReport(
            requested_mode=mode,
            effective_capabilities=effective,
            missing_capabilities=missing,
            recovery_level=("module_boundary_retry" if recovery_enabled else "none"),
            checkpoint_policy=(
                "run_and_module_boundaries"
                if recovery_enabled
                else ("run_history_only" if sqlite_enabled else "none")
            ),
            replay_policy=("recorded_managed_effects" if recovery_enabled else "none"),
            event_reconnect=sqlite_enabled,
            capacity_scope=binding.execution_capacity.scope,
            degraded_reasons=degraded_reasons,
            recovery_undeclared_modules=recovery_undeclared,
            effect_unverified_modules=effect_unverified,
            detached_tool_gaps=sandbox_gaps,
        )

    def _sandbox_binding_gaps(self, module: Module[Any, Any]) -> tuple[str, ...]:
        """Report sandbox wiring gaps without rejecting sync-only bindings."""

        gaps: list[str] = []
        for path, item in _collect_graph(module).items():
            layer_registry = getattr(item, "executor_registry", None)
            registry = layer_registry or self._tool_registry
            specs = getattr(item, "tools", ())
            if not isinstance(specs, tuple):
                continue
            for spec in specs:
                if not isinstance(spec, ToolSpec) or spec.sandbox_profile is None:
                    continue
                capability = f"tool.sandbox.{spec.sandbox_profile}"
                try:
                    if registry is None:
                        raise LookupError
                    executor = registry.resolve(spec.tool_id, spec.version)
                    validate_executor_sandbox(spec, executor)
                except Exception as exc:  # report-only preflight
                    if isinstance(exc, LookupError) or getattr(exc, "code", None) == (
                        "missing_sandbox_capability"
                    ):
                        gaps.append(
                            f"{path}:{spec.tool_id}@{spec.version} missing {capability}"
                        )
                    else:
                        raise
                if self._tool_registry is not registry:
                    try:
                        if self._tool_registry is None:
                            raise LookupError
                        detached_executor = self._tool_registry.resolve(
                            spec.tool_id, spec.version
                        )
                        validate_executor_sandbox(spec, detached_executor)
                    except Exception as exc:  # report-only detached preflight
                        if isinstance(exc, LookupError) or getattr(exc, "code", None) == (
                            "missing_sandbox_capability"
                        ):
                            gaps.append(
                                f"{path}:{spec.tool_id}@{spec.version} detached missing "
                                f"{capability}"
                            )
                        else:
                            raise
        return tuple(dict.fromkeys(gaps))

    def register_remote(
        self, binding_ref: str, bound: _LocalBoundModule[Any, Any]
    ) -> None:
        """Register a declared logical target for local placement tests."""

        if not binding_ref:
            raise ValueError("binding_ref must be non-empty")
        if binding_ref in self._remote_modules:
            raise ValueError(f"duplicate binding_ref {binding_ref!r}")
        self._remote_modules[binding_ref] = bound

    def register_remote_target(self, binding_ref: str, target: Any) -> None:
        """Register a declared distributed target for RemoteModule resolution."""

        if not binding_ref:
            raise ValueError("binding_ref must be non-empty")
        if binding_ref in self._remote_modules:
            raise ValueError(f"duplicate binding_ref {binding_ref!r}")
        if not callable(getattr(target, "invoke", None)):
            raise TypeError("remote target must expose invoke()")
        self._remote_modules[binding_ref] = target

    def attach_tool_task_manager(self, manager: ToolTaskManager) -> None:
        """Attach the application-selected process-local or durable task manager."""

        if self._tool_tasks is not None:
            raise RuntimeError("a ToolTaskManager is already attached")
        self._tool_tasks = manager

    def register_model_invoker(self, model_group: str, invoker: Any) -> None:
        """Register a deployment-owned invoker for managed model calls."""

        if not isinstance(model_group, str) or not model_group:
            raise ValueError("model_group must be a non-empty string")
        if model_group in self._model_invokers:
            raise ValueError(f"duplicate model group {model_group!r}")
        if not callable(getattr(invoker, "execute", None)):
            raise TypeError("model invoker must expose execute()")
        self._model_invokers[model_group] = invoker

    def attach_executor_registry(self, registry: Any) -> None:
        """Attach the deployment executor registry used by managed tools."""

        if self._tool_registry is not None:
            raise RuntimeError("an ExecutorRegistry is already attached")
        if not isinstance(registry, ExecutorRegistry):
            raise TypeError("registry must be an ExecutorRegistry")
        self._tool_registry = registry

    def register_tool(
        self,
        spec: ToolSpec,
        executor: ToolExecutor,
        *,
        replace_existing: bool = False,
    ) -> None:
        """Register an exact ToolSpec/executor pair and derive sandbox capability."""

        self.register_tools(
            ((spec, executor),),
            replace_existing=replace_existing,
        )

    def register_tools(
        self,
        registrations: Iterable[tuple[ToolSpec, ToolExecutor]],
        *,
        replace_existing: bool = False,
    ) -> None:
        """Atomically validate and register exact ToolSpec/executor pairs."""

        items = tuple(registrations)
        identities: list[tuple[str, str]] = []
        for spec, executor in items:
            if not isinstance(spec, ToolSpec):
                raise TypeError("spec must be a ToolSpec")
            if not isinstance(executor, ToolExecutor):
                raise TypeError("executor must implement ToolExecutor")
            identity = (spec.tool_id, spec.version)
            if identity in identities:
                raise ValueError(
                    f"duplicate Tool registration for {spec.tool_id}@{spec.version}"
                )
            identities.append(identity)
            if spec.sandbox_profile is not None:
                try:
                    validate_executor_sandbox(spec, executor)
                except Exception as exc:
                    raise ValueError(str(exc)) from exc

        registry = self._tool_registry
        if registry is None:
            registry = ExecutorRegistry()
        if not isinstance(registry, ExecutorRegistry):  # pragma: no cover - attach invariant
            raise TypeError("attached registry must be an ExecutorRegistry")
        if not replace_existing:
            for tool_id, version in identities:
                if registry.contains(tool_id, version):
                    raise ValueError(
                        f"executor already registered for {tool_id}@{version}"
                    )
        for (spec, executor) in items:
            registry.register(
                spec.tool_id,
                spec.version,
                executor,
                replace_existing=replace_existing,
            )
        self._tool_registry = registry


# Preserve the historical facade identity for public and test-visible classes.
RuntimeBinding.__module__ = __name__
InMemoryCapacityCoordinator.__module__ = __name__
_RunnableGate.__module__ = __name__

__all__ = [
    "InMemoryCapacityCoordinator",
    "LocalRuntime",
    "RuntimeBinding",
    "_RunnableGate",
]
