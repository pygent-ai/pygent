"""Canonical portable projection for named model specifications."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from pygent.core import FrozenJsonObject, JsonObjectInput, thaw_json

from .configuration import ModelCapabilities, ModelEntry, ModelSpec

_SPEC_FIELDS = frozenset(
    {"provider", "model_id", "protocol", "provider_options", "capabilities"}
)
_ENTRY_FIELDS = frozenset({"name", "spec"})


def model_spec_value(model: ModelSpec) -> dict[str, object]:
    return {
        "provider": model.provider,
        "model_id": model.model_id,
        "protocol": model.protocol,
        "provider_options": thaw_json(cast(FrozenJsonObject, model.provider_options)),
        "capabilities": model.capabilities.to_mapping(),
    }


def model_spec_from_value(value: Mapping[str, object]) -> ModelSpec:
    unknown = set(value) - _SPEC_FIELDS
    missing = _SPEC_FIELDS - set(value)
    if unknown or missing:
        raise ValueError("stored model spec fields do not match the current schema")
    options = value["provider_options"]
    capabilities = value["capabilities"]
    if not isinstance(options, Mapping) or not isinstance(capabilities, Mapping):
        raise TypeError("stored model spec objects are invalid")
    return ModelSpec(
        provider=cast(str, value["provider"]),
        model_id=cast(str, value["model_id"]),
        protocol=cast(str, value["protocol"]),
        provider_options=cast(JsonObjectInput, options),
        capabilities=ModelCapabilities.from_mapping(capabilities),
    )


def model_entry_value(entry: ModelEntry) -> dict[str, object]:
    return {"name": entry.name, "spec": model_spec_value(entry.spec)}


def model_entry_from_value(value: Mapping[str, object]) -> ModelEntry:
    if set(value) != _ENTRY_FIELDS or not isinstance(value["spec"], Mapping):
        raise ValueError("stored model entry fields do not match the current schema")
    return ModelEntry(
        name=cast(str, value["name"]),
        spec=model_spec_from_value(cast(Mapping[str, object], value["spec"])),
    )


__all__ = [
    "model_entry_from_value",
    "model_entry_value",
    "model_spec_from_value",
    "model_spec_value",
]
