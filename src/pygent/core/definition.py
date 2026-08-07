"""Shared fail-closed validation for executable Module definitions."""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

from .json_values import FrozenJsonObject

UNSUPPORTED_DEFINITION_VALUE = object()
_INTERNAL_MODULE_ATTRIBUTES = frozenset(
    {
        "_children",
        "_dependencies",
        "_definition_frozen",
        "_definition_config_snapshot",
    }
)
_FRAMEWORK_CLASS_ATTRIBUTES = frozenset(
    {"execution_requirements", "trusted_live_resource_attributes"}
)


def _qualified_name(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def portable_definition_value(
    value: object, *, allow_snapshot_containers: bool = False
) -> object:
    """Return deterministic JSON or an unsupported-value sentinel.

    Ordinary stored definition state must already be immutable.  The explicit
    execution-plan hook is a snapshot boundary and may return fresh dict/list
    containers, which are copied immediately.
    """

    if isinstance(value, Enum):
        return portable_definition_value(
            value.value, allow_snapshot_containers=allow_snapshot_containers
        )
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else UNSUPPORTED_DEFINITION_VALUE
    if isinstance(value, FrozenJsonObject):
        converted_items = {
            key: portable_definition_value(value[key]) for key in sorted(value)
        }
        if any(
            item is UNSUPPORTED_DEFINITION_VALUE
            for item in converted_items.values()
        ):
            return UNSUPPORTED_DEFINITION_VALUE
        return converted_items
    if isinstance(value, tuple):
        converted = tuple(
            portable_definition_value(
                item, allow_snapshot_containers=allow_snapshot_containers
            )
            for item in value
        )
        if any(item is UNSUPPORTED_DEFINITION_VALUE for item in converted):
            return UNSUPPORTED_DEFINITION_VALUE
        return list(converted)
    if isinstance(value, frozenset):
        converted = tuple(
            portable_definition_value(
                item, allow_snapshot_containers=allow_snapshot_containers
            )
            for item in value
        )
        if any(item is UNSUPPORTED_DEFINITION_VALUE for item in converted):
            return UNSUPPORTED_DEFINITION_VALUE
        return sorted(converted, key=repr)
    if allow_snapshot_containers and isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            return UNSUPPORTED_DEFINITION_VALUE
        converted_items = {
            key: portable_definition_value(
                value[key], allow_snapshot_containers=True
            )
            for key in sorted(value)
        }
        if any(
            item is UNSUPPORTED_DEFINITION_VALUE
            for item in converted_items.values()
        ):
            return UNSUPPORTED_DEFINITION_VALUE
        return converted_items
    if allow_snapshot_containers and isinstance(value, list):
        converted = tuple(
            portable_definition_value(item, allow_snapshot_containers=True)
            for item in value
        )
        if any(item is UNSUPPORTED_DEFINITION_VALUE for item in converted):
            return UNSUPPORTED_DEFINITION_VALUE
        return list(converted)
    if is_dataclass(value):
        parameters = getattr(type(value), "__dataclass_params__", None)
        if parameters is None or not parameters.frozen:
            return UNSUPPORTED_DEFINITION_VALUE
        converted_fields = {
            item.name: portable_definition_value(
                getattr(value, item.name),
                allow_snapshot_containers=allow_snapshot_containers,
            )
            for item in fields(value)
        }
        if any(
            item is UNSUPPORTED_DEFINITION_VALUE
            for item in converted_fields.values()
        ):
            return UNSUPPORTED_DEFINITION_VALUE
        return {"$type": _qualified_name(value), **converted_fields}
    return UNSUPPORTED_DEFINITION_VALUE


def module_definition_config(module: Any) -> dict[str, object]:
    """Validate stored Module state and return its portable identity payload."""

    live_names = tuple(getattr(module, "trusted_live_resource_attributes", ()))
    if any(not isinstance(name, str) or not name for name in live_names):
        raise TypeError(
            "trusted_live_resource_attributes must contain non-empty strings"
        )
    if len(live_names) != len(set(live_names)):
        raise ValueError("trusted_live_resource_attributes contains duplicates")
    missing_live = tuple(name for name in live_names if not hasattr(module, name))
    if missing_live:
        raise TypeError(
            "trusted live-resource attributes are not defined: "
            + ", ".join(missing_live)
        )
    for name in live_names:
        value = getattr(module, name)
        if isinstance(value, (dict, list, set, bytearray)):
            raise TypeError(
                f"trusted live-resource {type(module).__qualname__}.{name} must "
                "be an opaque adapter/resource, not a raw mutable container"
            )

    hook = getattr(module, "execution_plan_config", None)
    if hook is not None and not callable(hook):
        raise TypeError("execution_plan_config must be callable")

    dependencies = {id(value) for _, value in module.named_dependencies()}
    attributes: dict[str, object] = {}
    for name, value in sorted(vars(module).items()):
        if (
            name in _INTERNAL_MODULE_ATTRIBUTES
            or name in live_names
            or id(value) in dependencies
        ):
            continue
        if callable(value):
            if hook is not None:
                # Callable implementation objects have no portable structural
                # identity.  A Module that stores one must either declare it as
                # a trusted deployment resource or provide an explicit strict
                # declaration snapshot that identifies its behavior.
                continue
            raise TypeError(
                f"stored callable Module state {type(module).__qualname__}.{name} "
                "must be declared as a trusted live-resource attribute or "
                "described by execution_plan_config()"
            )
        converted = portable_definition_value(value)
        if converted is UNSUPPORTED_DEFINITION_VALUE:
            raise TypeError(
                f"stored Module state {type(module).__qualname__}.{name} must "
                "be a strict immutable portable value; use FrozenJsonObject/"
                "tuple/frozen dataclass, declare a trusted live-resource "
                "attribute, or move test instrumentation behind such an adapter"
            )
        attributes[name] = converted

    # User Module classes commonly use class attributes for immutable policy.
    # They are effective definition state just like instance attributes.  Walk
    # the user MRO up to (but excluding) the framework Module base, preserving
    # normal nearest-class shadowing while ignoring methods and descriptors.
    from .module import Module

    class_attributes: dict[str, object] = {}
    effective_class_values: dict[str, object] = {}
    for owner in type(module).__mro__:
        if owner is Module:
            break
        for name, value in vars(owner).items():
            effective_class_values.setdefault(name, value)
    for name, value in sorted(effective_class_values.items()):
        if (
            name.startswith("__")
            or name in _FRAMEWORK_CLASS_ATTRIBUTES
            or name in live_names
            or name in vars(module)
            or isinstance(value, (staticmethod, classmethod, property, type))
            or inspect.isroutine(value)
            or inspect.isdatadescriptor(value)
        ):
            continue
        converted = portable_definition_value(value)
        if converted is UNSUPPORTED_DEFINITION_VALUE:
            raise TypeError(
                f"stored Module class state {type(module).__qualname__}.{name} "
                "must be a strict immutable portable value; mutable/callable "
                "class state cannot participate in an executable definition"
            )
        class_attributes[name] = converted

    result: dict[str, object] = {}
    if hook is not None:
        declared = portable_definition_value(
            hook(), allow_snapshot_containers=True
        )
        if declared is UNSUPPORTED_DEFINITION_VALUE or not isinstance(
            declared, dict
        ):
            raise TypeError(
                "execution_plan_config must return a strict immutable JSON snapshot"
            )
        result["declaration"] = declared
    if attributes:
        result["attributes"] = attributes
    if class_attributes:
        result["class_attributes"] = class_attributes
    return result


__all__ = ["module_definition_config", "portable_definition_value"]
