"""Pure helpers for graph discovery and binding policy projection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any

from pygent.core import Module

from ..api import Binding
from ..plan import ExecutionPlan


def _collect_graph(module: Module[Any, Any]) -> dict[str, Module[Any, Any]]:
    result: dict[str, Module[Any, Any]] = {"root": module}
    canonical: dict[int, str] = {id(module): "root"}
    pending = ["root"]
    while pending:
        path = pending.pop(0)
        for name, child in sorted(
            result[path].named_children(), key=lambda item: item[0]
        ):
            if id(child) in canonical:
                continue
            child_path = f"{path}.{name}"
            canonical[id(child)] = child_path
            result[child_path] = child
            pending.append(child_path)
    return result


def _finite_deadline_requirement(
    module: Module[Any, Any],
) -> Module[Any, Any] | None:
    """Return the first declared dependency that requires a finite deadline."""

    pending: list[Module[Any, Any]] = [module]
    visited: set[int] = set()
    while pending:
        current = pending.pop(0)
        if id(current) in visited:
            continue
        visited.add(id(current))
        if current.execution_requirements.requires_finite_deadline:
            return current
        for _, dependency in sorted(
            current.named_dependencies(), key=lambda item: item[0]
        ):
            definition = (
                dependency
                if isinstance(dependency, Module)
                else getattr(dependency, "module", None)
            )
            if isinstance(definition, Module):
                pending.append(definition)
    return None


def _apply_binding_policy(plan: ExecutionPlan, binding: Binding) -> ExecutionPlan:
    """Include execution-affecting Binding policy in the immutable plan ID."""

    payload = {
        "run": {
            "scope": binding.execution_capacity.scope.value,
            "max_live_executions": binding.execution_capacity.max_live_executions,
            "max_runnable_executions": binding.execution_capacity.max_runnable_executions,
            "max_queue_size": binding.execution_capacity.max_queue_size,
            "max_waiters": binding.execution_capacity.max_waiters,
            "max_child_depth": binding.execution_capacity.max_child_depth,
            "max_children_per_execution": binding.execution_capacity.max_children_per_execution,
            "max_external_wait_seconds": (
                binding.execution_capacity.max_external_wait_seconds
            ),
        },
        "model": {
            "scope": binding.model_capacity.scope.value,
            "max_concurrency": binding.model_capacity.max_concurrency,
            "max_queue_size": binding.model_capacity.max_queue_size,
            "capacity_key": binding.model_capacity.capacity_key,
        },
        "tool": {
            "scope": binding.tool_capacity.scope.value,
            "max_concurrency": binding.tool_capacity.max_concurrency,
            "max_queue_size": binding.tool_capacity.max_queue_size,
            "capacity_key": binding.tool_capacity.capacity_key,
        },
        "durability": binding.durability.mode.value,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    policy_ref = f"sha256:{hashlib.sha256(encoded).hexdigest()}"
    modules = tuple(
        replace(
            spec,
            metadata=tuple(
                sorted((*spec.metadata, ("binding_policy_ref", policy_ref)))
            ),
            resource_keys=tuple(
                sorted(
                    {
                        *spec.resource_keys,
                        *(
                            (f"model:{binding.model_capacity.capacity_key}",)
                            if spec.path == plan.root
                            and binding.model_capacity.capacity_key is not None
                            else ()
                        ),
                        *(
                            (f"tool:{binding.tool_capacity.capacity_key}",)
                            if spec.path == plan.root
                            and binding.tool_capacity.capacity_key is not None
                            else ()
                        ),
                    }
                )
            ),
        )
        if spec.path == plan.root
        else spec
        for spec in plan.modules
    )
    return replace(plan, modules=modules)


def _collect_dependency_paths(
    module: Module[Any, Any],
) -> dict[int, str]:
    """Map every declared dependency handle to its canonical plan path."""

    paths: dict[int, str] = {id(module): "root"}
    pending: list[tuple[str, Module[Any, Any]]] = [("root", module)]
    while pending:
        parent_path, parent = pending.pop(0)
        for name, dependency in sorted(
            parent.named_dependencies(), key=lambda item: item[0]
        ):
            if id(dependency) in paths:
                continue
            path = f"{parent_path}.{name}"
            paths[id(dependency)] = path
            definition = (
                dependency
                if isinstance(dependency, Module)
                else getattr(dependency, "module", None)
            )
            if isinstance(definition, Module):
                paths.setdefault(id(definition), path)
                pending.append((path, definition))
    return paths


__all__ = [
    "_apply_binding_policy",
    "_collect_dependency_paths",
    "_collect_graph",
    "_finite_deadline_requirement",
]
