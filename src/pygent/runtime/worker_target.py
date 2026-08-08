"""Bound-module Worker handler and remote Module target."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast
from uuid import uuid4

import httpx

from pygent.core import (
    Context,
    JsonValue,
    Message,
    PlacementMode,
    PlacementPolicy,
    thaw_json,
)
from pygent.llm import ModelCallOptions

from ._worker_protocol import (
    WorkerArtifactResolver,
    WorkerDeploymentManifest,
    WorkerEventSink,
    WorkerHandler,
    WorkerInvocation,
    WorkerProtocolError,
    WorkerRemoteError,
    _validate_plan_identity,
    _worker_failure,
)
from .api import BoundModule, ExecutionOptions
from .codec import invocation_from_dict, invocation_to_dict
from .worker_client import HTTPWorkerClient


def bound_module_worker_handler(
    bindings: Mapping[str, BoundModule[Any, Any]],
    *,
    artifact_resolver: WorkerArtifactResolver | None = None,
) -> WorkerHandler:
    """Adapt declared BoundModules to the portable HTTP Worker payload."""

    declared = dict(bindings)
    for binding_ref, bound in declared.items():
        plan = getattr(bound, "plan", None)
        if plan is None or not plan.is_portable:
            raise WorkerProtocolError(
                "HTTP Worker binding "
                f"{binding_ref!r} requires a portable ExecutionPlan with "
                "artifact, entrypoint, schemas, and serializer"
            )
        if artifact_resolver is None or plan.artifact is None:
            raise WorkerProtocolError(
                "HTTP Worker requires a deployment artifact resolver"
            )
        manifest = artifact_resolver(plan.artifact)
        if not isinstance(manifest, WorkerDeploymentManifest):
            raise WorkerProtocolError("artifact resolver returned an invalid manifest")
        entrypoint_name = (
            f"{manifest.entrypoint.__module__}:{manifest.entrypoint.__qualname__}"
        )
        root_spec = next(spec for spec in plan.modules if spec.path == plan.root)
        if (
            manifest.artifact != plan.artifact
            or manifest.verified_digest != plan.artifact.digest
            or entrypoint_name != plan.artifact.entrypoint
            or manifest.input_schema != root_spec.input_schema
            or manifest.output_schema != root_spec.output_schema
            or manifest.serializer != root_spec.serializer
        ):
            raise WorkerProtocolError(
                f"Worker deployment manifest does not verify {binding_ref!r}"
            )

    async def invoke(
        request: WorkerInvocation, event_sink: WorkerEventSink
    ) -> JsonValue:
        try:
            bound = declared[request.binding_ref]
        except KeyError as exc:
            raise WorkerRemoteError(
                _worker_failure(
                    "binding_not_found",
                    f"undeclared Worker binding_ref {request.binding_ref!r}",
                )
            ) from exc
        plan = getattr(bound, "plan", None)
        if plan is None:
            raise WorkerRemoteError(
                _worker_failure(
                    "invalid_execution_plan",
                    f"Worker binding {request.binding_ref!r} has no ExecutionPlan",
                )
            )
        if request.plan_id != plan.plan_id or request.graph_hash != plan.graph_hash:
            raise WorkerRemoteError(
                _worker_failure(
                    "invalid_execution_plan",
                    "remote ExecutionPlan identity does not match the declared "
                    f"Worker binding {request.binding_ref!r}",
                )
            )
        has_deferred_models = any(spec.model_requirements for spec in plan.modules)
        if has_deferred_models:
            if "model.deferred.exact-pin.v1" not in request.required_capabilities:
                raise WorkerRemoteError(
                    _worker_failure(
                        "capability_mismatch",
                        "dynamic model execution requires model.deferred.exact-pin.v1",
                    )
                )
            if request.model_admission_ref != request.request_id:
                raise WorkerRemoteError(
                    _worker_failure(
                        "invalid_model_admission",
                        "dynamic model admission reference must equal the stable request identity",
                    )
                )
            runtime = getattr(bound, "runtime", None)
            scope_id = getattr(bound, "deployment_scope_id", None)
            store = getattr(runtime, "model_deployment_store", None)
            ensure_open = getattr(runtime, "_ensure_model_store_open", None)
            resolvers = getattr(runtime, "_model_resource_resolvers", None)
            if (
                not isinstance(scope_id, str)
                or store is None
                or not callable(ensure_open)
                or not isinstance(resolvers, dict)
            ):
                raise WorkerRemoteError(
                    _worker_failure(
                        "model_deployment_unavailable",
                        "Worker cannot validate an exact model deployment manifest",
                    )
                )
            store_namespace = getattr(store, "namespace_id", None)
            if (
                not isinstance(store_namespace, str)
                or request.model_store_namespace != store_namespace
            ):
                raise WorkerRemoteError(
                    _worker_failure(
                        "model_store_namespace_mismatch",
                        "Worker model deployment store namespace does not match the coordinator",
                    )
                )
            requirements = tuple(
                sorted(
                    {
                        requirement.group_name
                        for spec in plan.modules
                        for requirement in spec.model_requirements
                    }
                )
            )
            selections: dict[str, str | None] = {}
            for group_name in requirements:
                raw = request.model_calls.get(group_name)
                options = (
                    ModelCallOptions()
                    if raw is None
                    else ModelCallOptions.from_dict(cast(Mapping[str, object], raw))
                )
                selections[group_name] = options.profile
            await ensure_open()
            admission = await store.admit(
                scope_id,
                requirements,
                selections,
                admission_id=request.model_admission_ref,
            )
            for _, snapshot in admission.snapshots:
                resources = snapshot.resources
                if resources is None or resources.resolver_id not in resolvers:
                    raise WorkerRemoteError(
                        _worker_failure(
                            "model_deployment_unavailable",
                            "Worker cannot reconstruct the exact model deployment",
                        )
                    )
                resolver = resolvers[resources.resolver_id]
                resolver_domain = getattr(resolver, "coordinator_domain", None)
                if (
                    resolver_domain is not None
                    and resolver_domain != resources.coordinator_domain
                ):
                    raise WorkerRemoteError(
                        _worker_failure(
                            "model_capacity_domain_mismatch",
                            "Worker model resolver uses a different capacity domain",
                        )
                    )
        message, context = invocation_from_dict(request.input)
        handle = await bound.start(
            message,
            context,
            execution=ExecutionOptions(
                request_id=request.request_id,
                execution_id=request.request_id,
                trace_id=request.trace_id,
                parent_execution_id=request.parent_execution_id,
                parent_span_id=request.parent_span_id,
                idempotency_key=request.idempotency_key,
                context_ref=(
                    None
                    if request.parent_execution_id is None
                    else f"parent-run:{request.parent_execution_id}"
                ),
                deadline=request.deadline,
                model_calls=request.model_calls,
            ),
        )

        async def relay() -> None:
            async with handle.subscribe() as events:
                async for event in events:
                    await event_sink(event)

        relay_task = asyncio.create_task(relay(), name="pygent-worker-event-relay")
        try:
            output, next_context = await handle.result()
        except asyncio.CancelledError:
            await handle.cancel()
            relay_task.cancel()
            await asyncio.gather(relay_task, return_exceptions=True)
            raise
        except BaseException:
            relay_task.cancel()
            await asyncio.gather(relay_task, return_exceptions=True)
            raise
        else:
            await relay_task
        return invocation_to_dict(output, next_context)

    dynamic_namespaces = {
        namespace
        for bound in declared.values()
        if any(
            spec.model_requirements
            for spec in cast(Any, bound).plan.modules
        )
        for namespace in (
            getattr(
                getattr(getattr(bound, "runtime", None), "model_deployment_store", None),
                "namespace_id",
                None,
            ),
        )
        if isinstance(namespace, str) and namespace
    }
    if len(dynamic_namespaces) > 1:
        raise WorkerProtocolError(
            "dynamic Worker bindings must share one model store namespace"
        )
    if dynamic_namespaces:
        cast(Any, invoke)._pygent_model_store_namespace = next(iter(dynamic_namespaces))
    return invoke


@dataclass(frozen=True, slots=True)
class HTTPRemoteModuleTarget:
    """Managed RemoteModule target backed by the HTTP Worker protocol."""

    client: HTTPWorkerClient
    binding_ref: str
    plan_id: str
    graph_hash: str
    required_capabilities: tuple[str, ...] = ()
    placement: PlacementPolicy = field(default_factory=PlacementPolicy.adaptive)

    def __post_init__(self) -> None:
        if not self.binding_ref:
            raise ValueError("binding_ref must be non-empty")
        _validate_plan_identity(self.plan_id, self.graph_hash)
        capabilities = tuple(self.required_capabilities)
        if any(not isinstance(item, str) or not item for item in capabilities):
            raise ValueError("required capabilities must be non-empty strings")
        if len(capabilities) != len(set(capabilities)):
            raise ValueError("required capabilities must be unique")
        object.__setattr__(self, "required_capabilities", capabilities)
        if not isinstance(self.placement, PlacementPolicy):
            raise TypeError("placement must be a PlacementPolicy")
        if self.placement.mode is PlacementMode.INHERIT:
            raise ValueError("HTTP remote target placement cannot be inherit")

    async def invoke(
        self, message: Message, context: Context, *, deadline: float | None
    ) -> tuple[Message, Context]:
        return await self.invoke_remote_child(
            message,
            context,
            plan_id=self.plan_id,
            graph_hash=self.graph_hash,
            required_capabilities=self.required_capabilities,
            placement_mode=self.placement.mode.value,
            pinned_target_id=self.placement.target_id,
            deadline=deadline,
            trace_id=None,
            parent_execution_id=None,
            parent_span_id=None,
            event_sink=None,
            child_execution_id=None,
            attempt=1,
            idempotency_key=None,
        )

    async def invoke_remote_child(
        self,
        message: Message,
        context: Context,
        *,
        plan_id: str | None,
        graph_hash: str | None,
        required_capabilities: tuple[str, ...],
        placement_mode: str,
        pinned_target_id: str | None,
        deadline: float | None,
        trace_id: str | None,
        parent_execution_id: str | None,
        parent_span_id: str | None,
        event_sink: WorkerEventSink | None,
        child_execution_id: str | None,
        attempt: int,
        idempotency_key: str | None,
    ) -> tuple[Message, Context]:
        if plan_id is None or graph_hash is None:
            raise WorkerProtocolError(
                "distributed RemoteModule requires a stable ExecutionPlan identity"
            )
        _validate_plan_identity(plan_id, graph_hash)
        if (
            plan_id != self.plan_id
            or graph_hash != self.graph_hash
            or tuple(required_capabilities) != self.required_capabilities
        ):
            raise WorkerProtocolError(
                "registered remote target identity differs from the RemoteModule "
                "declaration; compile a new caller ExecutionPlan"
            )
        placement = PlacementPolicy(PlacementMode(placement_mode), pinned_target_id)
        if placement != self.placement:
            raise WorkerProtocolError(
                "registered remote target placement differs from the "
                "RemoteModule declaration; compile a new caller ExecutionPlan"
            )
        payload = invocation_to_dict(message, context)
        request_id = child_execution_id or str(uuid4())
        logical_key = idempotency_key or (
            None if trace_id is None else f"remote-child:{trace_id}:{self.binding_ref}"
        )
        ref = await self.client.start(
            self.binding_ref,
            cast(Mapping[str, Any], thaw_json(payload)),
            request_id=request_id,
            plan_id=plan_id,
            graph_hash=graph_hash,
            placement=placement,
            deadline=deadline,
            required_capabilities=required_capabilities,
            idempotency_key=logical_key,
            trace_id=trace_id,
            parent_execution_id=parent_execution_id,
            parent_span_id=parent_span_id,
            attempt=attempt,
            model_admission_ref=(
                request_id
                if "model.deferred.exact-pin.v1" in required_capabilities
                else None
            ),
        )
        relay_task: asyncio.Task[None] | None = None
        if event_sink is not None:

            async def relay() -> None:
                async with ref.subscribe() as events:
                    async for event in events:
                        await event_sink(event)

            relay_task = asyncio.create_task(relay(), name="pygent-remote-event-relay")
        try:
            result = await self.client.result(ref, deadline=deadline)
        except (asyncio.CancelledError, TimeoutError):
            try:
                await asyncio.shield(self.client.cancel(ref))
            except (WorkerProtocolError, httpx.HTTPError):
                pass
            if relay_task is not None:
                relay_task.cancel()
                await asyncio.gather(relay_task, return_exceptions=True)
            raise
        except BaseException:
            if relay_task is not None:
                relay_task.cancel()
                await asyncio.gather(relay_task, return_exceptions=True)
            raise
        else:
            if relay_task is not None:
                await relay_task
        return invocation_from_dict(result)
