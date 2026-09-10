"""Offline provider, model-capability, and capability-preset catalogs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from types import MappingProxyType
from typing import cast

from pygent.core import FrozenJsonObject, freeze_json_object

from .configuration import (
    ModelCapabilities,
    ModelLimits,
    ModelModalities,
    ModelReasoningCapabilities,
    ModelStreamingCapabilities,
    ModelStructuredOutputCapabilities,
    ModelToolCapabilities,
    _exact_fields,
    _non_empty,
    _object,
    _string_tuple,
    _validated_url,
)

_SCHEMA_VERSION = 1


def _builtin_mapping(filename: str) -> Mapping[str, object]:
    resource = files("pygent.llm.data").joinpath(filename)
    value = json.loads(resource.read_text(encoding="utf-8"))
    return _object(value, filename)


def _schema_version(value: Mapping[str, object], label: str) -> None:
    if value.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(f"unsupported {label} schema_version")


@dataclass(frozen=True, slots=True)
class ProviderPreset:
    provider: str
    display_name: str
    protocols: tuple[str, ...]
    default_protocol: str
    base_url: str
    authentication: str
    api_key_env: str | None
    provider_options_schema: FrozenJsonObject

    @classmethod
    def from_mapping(cls, provider: str, value: Mapping[str, object]) -> ProviderPreset:
        _exact_fields(
            value,
            frozenset(
                {
                    "display_name",
                    "protocols",
                    "default_protocol",
                    "base_url",
                    "authentication",
                    "api_key_env",
                    "provider_options_schema",
                }
            ),
            "provider preset",
        )
        protocols = _string_tuple(value["protocols"], "provider protocols")
        default_protocol = _non_empty(value["default_protocol"], "default_protocol")
        if default_protocol not in protocols:
            raise ValueError("default_protocol must be present in protocols")
        authentication = _non_empty(value["authentication"], "authentication")
        if authentication not in {"bearer", "none"}:
            raise ValueError("authentication must be bearer or none")
        api_key_env = value["api_key_env"]
        if authentication == "bearer":
            api_key_env = _non_empty(api_key_env, "api_key_env")
        elif api_key_env is not None:
            raise ValueError("api_key_env must be null when authentication is none")
        schema = _object(value["provider_options_schema"], "provider_options_schema")
        return cls(
            provider=_non_empty(provider, "provider"),
            display_name=_non_empty(value["display_name"], "display_name"),
            protocols=protocols,
            default_protocol=default_protocol,
            base_url=_validated_url(value["base_url"], "base_url"),
            authentication=authentication,
            api_key_env=cast(str | None, api_key_env),
            provider_options_schema=freeze_json_object(schema),
        )


@dataclass(frozen=True, slots=True)
class ProviderCatalog:
    providers: Mapping[str, ProviderPreset]

    def __post_init__(self) -> None:
        object.__setattr__(self, "providers", MappingProxyType(dict(self.providers)))

    @classmethod
    def builtin(cls) -> ProviderCatalog:
        return cls.from_mapping(_builtin_mapping("providers.json"))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ProviderCatalog:
        raw = _object(value, "provider catalog")
        _exact_fields(raw, frozenset({"schema_version", "providers"}), "provider catalog")
        _schema_version(raw, "provider catalog")
        providers = _object(raw["providers"], "providers")
        return cls(
            {
                name: ProviderPreset.from_mapping(name, _object(item, "provider preset"))
                for name, item in providers.items()
            }
        )


@dataclass(frozen=True, slots=True)
class ModelCapabilityCatalog:
    models: Mapping[tuple[str, str, str], ModelCapabilities]

    def __post_init__(self) -> None:
        object.__setattr__(self, "models", MappingProxyType(dict(self.models)))

    @classmethod
    def builtin(cls) -> ModelCapabilityCatalog:
        return cls.from_mapping(_builtin_mapping("model_capabilities.json"))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ModelCapabilityCatalog:
        raw = _object(value, "model capability catalog")
        _exact_fields(
            raw,
            frozenset({"schema_version", "models"}),
            "model capability catalog",
        )
        _schema_version(raw, "model capability catalog")
        items = raw["models"]
        if not isinstance(items, list):
            raise TypeError("models must be an array")
        models: dict[tuple[str, str, str], ModelCapabilities] = {}
        for item in items:
            model = _object(item, "model capability entry")
            _exact_fields(
                model,
                frozenset({"provider", "model_id", "protocol", "capabilities"}),
                "model capability entry",
            )
            key = (
                _non_empty(model["provider"], "provider"),
                _non_empty(model["model_id"], "model_id"),
                _non_empty(model["protocol"], "protocol"),
            )
            if key in models:
                raise ValueError("model capability catalog contains a duplicate key")
            models[key] = ModelCapabilities.from_mapping(
                _object(model["capabilities"], "capabilities")
            )
        return cls(models)


@dataclass(frozen=True, slots=True)
class CapabilityPreset:
    name: str
    modalities: ModelModalities
    streaming: ModelStreamingCapabilities
    tools: ModelToolCapabilities
    structured_output: ModelStructuredOutputCapabilities
    reasoning: ModelReasoningCapabilities

    def materialize(
        self, *, context_tokens: int, max_output_tokens: int
    ) -> ModelCapabilities:
        return ModelCapabilities(
            modalities=self.modalities,
            streaming=self.streaming,
            tools=self.tools,
            structured_output=self.structured_output,
            reasoning=self.reasoning,
            limits=ModelLimits(context_tokens, max_output_tokens),
        )

    @classmethod
    def from_mapping(cls, name: str, value: Mapping[str, object]) -> CapabilityPreset:
        _exact_fields(
            value,
            frozenset({"modalities", "streaming", "tools", "structured_output", "reasoning"}),
            "capability preset",
        )
        return cls(
            name=_non_empty(name, "capability preset name"),
            modalities=ModelModalities.from_mapping(_object(value["modalities"], "modalities")),
            streaming=ModelStreamingCapabilities.from_mapping(
                _object(value["streaming"], "streaming")
            ),
            tools=ModelToolCapabilities.from_mapping(_object(value["tools"], "tools")),
            structured_output=ModelStructuredOutputCapabilities.from_mapping(
                _object(value["structured_output"], "structured_output")
            ),
            reasoning=ModelReasoningCapabilities.from_mapping(
                _object(value["reasoning"], "reasoning")
            ),
        )


@dataclass(frozen=True, slots=True)
class CapabilityPresetCatalog:
    presets: Mapping[str, CapabilityPreset]

    def __post_init__(self) -> None:
        object.__setattr__(self, "presets", MappingProxyType(dict(self.presets)))

    @classmethod
    def builtin(cls) -> CapabilityPresetCatalog:
        return cls.from_mapping(_builtin_mapping("capability_presets.json"))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> CapabilityPresetCatalog:
        raw = _object(value, "capability preset catalog")
        _exact_fields(
            raw,
            frozenset({"schema_version", "presets"}),
            "capability preset catalog",
        )
        _schema_version(raw, "capability preset catalog")
        presets = _object(raw["presets"], "presets")
        return cls(
            {
                name: CapabilityPreset.from_mapping(name, _object(item, "capability preset"))
                for name, item in presets.items()
            }
        )


__all__ = [
    "CapabilityPreset",
    "CapabilityPresetCatalog",
    "ModelCapabilityCatalog",
    "ProviderCatalog",
    "ProviderPreset",
]
