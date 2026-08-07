"""Deterministic compilation of an in-process Module graph."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from typing import Any

from pygent.core import Module, RemoteModule
from pygent.core.definition import module_definition_config
from pygent.llm import ModelGroupConfig

from .plan import CodeArtifactSpec, ExecutionPlan, ModelRequirement, ModuleSpec


def _module_config(module: Module[Any, Any]) -> dict[str, object]:
    """Build the portable part of one Module definition.

    A user Module can provide ``execution_plan_config()`` when its immutable
    declaration is not represented by ordinary public attributes.  The hook is
    intentionally duck-typed so it does not become a second execution API.
    """

    requirements = module.execution_requirements
    result: dict[str, object] = {
        "execution_requirements": {
            "requires_finite_deadline": requirements.requires_finite_deadline,
            "required_capabilities": list(requirements.required_capabilities),
            "recovery_safety": requirements.recovery_safety.value,
            "effect_safety": requirements.effect_safety.value,
        }
    }
    result.update(module_definition_config(module))
    return result


def _config_ref(module: Module[Any, Any]) -> str:
    encoded = json.dumps(
        _module_config(module),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _resource_keys(module: Module[Any, Any]) -> tuple[str, ...]:
    keys: set[str] = set()
    model_group = getattr(module, "model_group", None)
    group_name = getattr(model_group, "name", None)
    capacity_key = getattr(model_group, "capacity_key", None) or group_name
    if isinstance(capacity_key, str) and capacity_key:
        keys.add(f"model:{capacity_key}")
    tools = getattr(module, "tools", ())
    if isinstance(tools, tuple):
        for tool in tools:
            resource_key = getattr(tool, "resource_key", None)
            if isinstance(resource_key, str) and resource_key:
                keys.add(resource_key)
    return tuple(sorted(keys))


def compile_execution_plan(
    module: Module[Any, Any],
    *,
    artifact: CodeArtifactSpec | None = None,
    input_schema: str | None = None,
    output_schema: str | None = None,
    serializer: str | None = None,
) -> ExecutionPlan:
    """Compile a Module definition graph into a stable local execution plan.

    Attribute names, not object addresses or construction order, define paths.
    A shared definition is emitted once and every parent references its canonical
    (lexicographically first) path.
    """

    if not isinstance(module, Module):
        raise TypeError("module must be a Module")
    wire_values = (input_schema, output_schema, serializer)
    if artifact is None and any(value is not None for value in wire_values):
        raise ValueError("wire schemas require an explicit code artifact")
    if artifact is not None and any(value is None for value in wire_values):
        raise ValueError(
            "portable execution requires input_schema, output_schema, and serializer"
        )

    canonical: dict[int, str] = {id(module): "root"}
    objects: dict[str, object] = {"root": module}
    pending: deque[str] = deque(("root",))
    child_edges: dict[str, tuple[str, ...]] = {}

    while pending:
        path = pending.popleft()
        current = objects[path]
        definition = (
            current
            if isinstance(current, Module)
            else getattr(current, "module", None)
        )
        edges: list[str] = []
        dependencies = (
            definition.named_dependencies()
            if isinstance(definition, Module)
            else ()
        )
        for name, child in sorted(dependencies, key=lambda item: item[0]):
            candidate = f"{path}.{name}"
            child_path = canonical.get(id(child))
            if child_path is None:
                child_path = candidate
                canonical[id(child)] = child_path
                objects[child_path] = child
                pending.append(child_path)
            if child_path not in edges:
                edges.append(child_path)
        child_edges[path] = tuple(edges)

    specs: list[ModuleSpec] = []
    for path in sorted(objects):
        dependency = objects[path]
        model_requirements: tuple[ModelRequirement, ...] = ()
        definition = (
            dependency
            if isinstance(dependency, Module)
            else getattr(dependency, "module", None)
        )
        metadata: tuple[tuple[str, str], ...]
        if isinstance(dependency, RemoteModule):
            type_name = "RemoteModule"
            definition_id = (
                f"remote:{dependency.binding_ref}"
                if dependency.graph_hash is None
                else f"remote:{dependency.binding_ref}@{dependency.graph_hash}"
            )
            requirements = dependency.required_capabilities
            encoded = json.dumps(
                {
                    "binding_ref": dependency.binding_ref,
                    "plan_id": dependency.plan_id,
                    "graph_hash": dependency.graph_hash,
                    "required_capabilities": list(requirements),
                    "placement": dependency.placement.mode.value,
                    "pinned_target_id": dependency.placement.target_id,
                },
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            config_ref = f"sha256:{hashlib.sha256(encoded).hexdigest()}"
            resource_keys: tuple[str, ...] = ()
            placement_constraints = (
                f"mode={dependency.placement.mode.value}",
                *(
                    (f"target_id={dependency.placement.target_id}",)
                    if dependency.placement.target_id is not None
                    else ()
                ),
            )
            metadata = tuple(
                (key, value)
                for key, value in (
                    ("binding_ref", dependency.binding_ref),
                    ("remote_plan_id", dependency.plan_id),
                    ("remote_graph_hash", dependency.graph_hash),
                    ("recovery_safety", "undeclared"),
                    ("effect_safety", "undeclared"),
                )
                if value is not None
            )
        elif isinstance(definition, Module):
            type_name = type(definition).__name__
            definition_id = (
                f"{type(definition).__module__}.{type(definition).__qualname__}"
            )
            requirements = definition.execution_requirements.required_capabilities
            config_ref = _config_ref(definition)
            resource_keys = _resource_keys(definition)
            binding = getattr(dependency, "binding", None)
            placement_constraints = (
                ("mode=pinned", f"binding={binding.name}")
                if binding is not None
                else (("mode=inherit",) if path != "root" else ())
            )
            metadata = (
                (("binding", str(binding.name)),)
                if binding is not None
                else ()
            ) + (
                ("recovery_safety", definition.execution_requirements.recovery_safety.value),
                ("effect_safety", definition.execution_requirements.effect_safety.value),
            )
            model_group = getattr(definition, "model_group", None)
            policy = getattr(definition, "policy", None)
            model_requirements = (
                (
                    ModelRequirement(
                        group_name=model_group.name,
                        capacity_key=model_group.capacity_key or model_group.name,
                        max_concurrency=model_group.max_concurrency,
                        allow_profile_override=bool(
                            getattr(policy, "allow_profile_override", False)
                        ),
                        overridable_generation=tuple(
                            sorted(getattr(policy, "overridable_generation", ()))
                        ),
                    ),
                )
                if isinstance(model_group, ModelGroupConfig)
                and model_group.is_deferred
                else ()
            )
        else:  # pragma: no cover - dependency registration invariant
            raise TypeError(f"unsupported Module dependency at {path!r}")
        specs.append(
            ModuleSpec(
                path=path,
                type_name=type_name,
                children=child_edges[path],
                definition_id=definition_id,
                config_ref=config_ref,
                resource_keys=resource_keys,
                placement_constraints=placement_constraints,
                required_capabilities=requirements,
                input_schema=input_schema,
                output_schema=output_schema,
                serializer=serializer,
                metadata=metadata,
                model_requirements=model_requirements,
            )
        )
    declarations: dict[str, ModelRequirement] = {}
    for spec in specs:
        for requirement in spec.model_requirements:
            previous = declarations.get(requirement.group_name)
            if previous is not None and previous != requirement:
                raise ValueError(
                    f"model group {requirement.group_name!r} has conflicting requirements"
                )
            declarations[requirement.group_name] = requirement
    return ExecutionPlan(root="root", modules=tuple(specs), artifact=artifact)


__all__ = ["compile_execution_plan"]
