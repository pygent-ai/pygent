"""Immutable model semantics and user configuration parsing."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast
from urllib.parse import urlsplit

from pygent.core import FrozenJsonObject, JsonObjectInput, freeze_json_object

from .protocols import BuiltinModelProtocol
from .types import ModelGroupResolution

_CAPABILITY_FIELDS = frozenset(
    {"modalities", "streaming", "tools", "structured_output", "reasoning", "limits"}
)
_MODEL_FIELDS = frozenset(
    {
        "provider",
        "model_id",
        "protocol",
        "connection",
        "provider_options",
        "capabilities",
    }
)
_CONNECTION_FIELDS = frozenset({"base_url", "credential", "verify_ssl", "proxy"})
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TOOL_CHOICES = frozenset({"none", "auto", "required", "named"})


def _object(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return cast(Mapping[str, object], value)


def _exact_fields(value: Mapping[str, object], fields: frozenset[str], label: str) -> None:
    unknown = set(value) - fields
    if unknown:
        raise ValueError(f"unknown {label} fields: " + ", ".join(sorted(unknown)))
    missing = fields - set(value)
    if missing:
        raise ValueError(f"missing {label} fields: " + ", ".join(sorted(missing)))


def _non_empty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{label} must be an array")
    items = tuple(value)
    if any(not isinstance(item, str) or not item for item in items):
        raise ValueError(f"{label} must contain non-empty strings")
    if len(items) != len(set(items)):
        raise ValueError(f"{label} must not contain duplicates")
    return cast(tuple[str, ...], items)


def _bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a bool")
    return value


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class ModelModalities:
    input: tuple[str, ...]
    output: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "input", _string_tuple(self.input, "modalities.input"))
        object.__setattr__(self, "output", _string_tuple(self.output, "modalities.output"))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ModelModalities:
        _exact_fields(value, frozenset({"input", "output"}), "modalities")
        return cls(
            input=_string_tuple(value["input"], "modalities.input"),
            output=_string_tuple(value["output"], "modalities.output"),
        )


@dataclass(frozen=True, slots=True)
class ModelStreamingCapabilities:
    text: bool

    def __post_init__(self) -> None:
        _bool(self.text, "streaming.text")

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> ModelStreamingCapabilities:
        _exact_fields(value, frozenset({"text"}), "streaming")
        return cls(text=_bool(value["text"], "streaming.text"))


@dataclass(frozen=True, slots=True)
class ModelToolCapabilities:
    call: bool
    choice: tuple[str, ...]
    parallel: bool

    def __post_init__(self) -> None:
        _bool(self.call, "tools.call")
        choices = _string_tuple(self.choice, "tools.choice")
        unknown = set(choices) - _TOOL_CHOICES
        if unknown:
            raise ValueError(
                "unsupported tools.choice values: " + ", ".join(sorted(unknown))
            )
        _bool(self.parallel, "tools.parallel")
        object.__setattr__(self, "choice", choices)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ModelToolCapabilities:
        _exact_fields(value, frozenset({"call", "choice", "parallel"}), "tools")
        choices = _string_tuple(value["choice"], "tools.choice")
        unknown = set(choices) - _TOOL_CHOICES
        if unknown:
            raise ValueError("unsupported tools.choice values: " + ", ".join(sorted(unknown)))
        return cls(
            call=_bool(value["call"], "tools.call"),
            choice=choices,
            parallel=_bool(value["parallel"], "tools.parallel"),
        )


@dataclass(frozen=True, slots=True)
class ModelStructuredOutputCapabilities:
    json_object: bool
    json_schema: bool

    def __post_init__(self) -> None:
        _bool(self.json_object, "structured_output.json_object")
        _bool(self.json_schema, "structured_output.json_schema")

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> ModelStructuredOutputCapabilities:
        _exact_fields(value, frozenset({"json_object", "json_schema"}), "structured_output")
        return cls(
            json_object=_bool(value["json_object"], "structured_output.json_object"),
            json_schema=_bool(value["json_schema"], "structured_output.json_schema"),
        )


@dataclass(frozen=True, slots=True)
class ModelReasoningCapabilities:
    supported: bool
    controllable: bool

    def __post_init__(self) -> None:
        _bool(self.supported, "reasoning.supported")
        _bool(self.controllable, "reasoning.controllable")
        if self.controllable and not self.supported:
            raise ValueError("reasoning.controllable requires reasoning.supported")

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> ModelReasoningCapabilities:
        _exact_fields(value, frozenset({"supported", "controllable"}), "reasoning")
        supported = _bool(value["supported"], "reasoning.supported")
        controllable = _bool(value["controllable"], "reasoning.controllable")
        if controllable and not supported:
            raise ValueError("reasoning.controllable requires reasoning.supported")
        return cls(supported=supported, controllable=controllable)


@dataclass(frozen=True, slots=True)
class ModelLimits:
    context_tokens: int
    max_output_tokens: int

    def __post_init__(self) -> None:
        _positive_int(self.context_tokens, "limits.context_tokens")
        _positive_int(self.max_output_tokens, "limits.max_output_tokens")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ModelLimits:
        _exact_fields(value, frozenset({"context_tokens", "max_output_tokens"}), "limits")
        return cls(
            context_tokens=_positive_int(value["context_tokens"], "limits.context_tokens"),
            max_output_tokens=_positive_int(
                value["max_output_tokens"], "limits.max_output_tokens"
            ),
        )


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    modalities: ModelModalities
    streaming: ModelStreamingCapabilities
    tools: ModelToolCapabilities
    structured_output: ModelStructuredOutputCapabilities
    reasoning: ModelReasoningCapabilities
    limits: ModelLimits

    def __post_init__(self) -> None:
        expected = (
            ("modalities", ModelModalities),
            ("streaming", ModelStreamingCapabilities),
            ("tools", ModelToolCapabilities),
            ("structured_output", ModelStructuredOutputCapabilities),
            ("reasoning", ModelReasoningCapabilities),
            ("limits", ModelLimits),
        )
        for name, value_type in expected:
            if not isinstance(getattr(self, name), value_type):
                raise TypeError(f"capabilities.{name} must be {value_type.__name__}")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ModelCapabilities:
        _exact_fields(value, _CAPABILITY_FIELDS, "capabilities")
        return cls(
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
            limits=ModelLimits.from_mapping(_object(value["limits"], "limits")),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "modalities": {
                "input": list(self.modalities.input),
                "output": list(self.modalities.output),
            },
            "streaming": {"text": self.streaming.text},
            "tools": {
                "call": self.tools.call,
                "choice": list(self.tools.choice),
                "parallel": self.tools.parallel,
            },
            "structured_output": {
                "json_object": self.structured_output.json_object,
                "json_schema": self.structured_output.json_schema,
            },
            "reasoning": {
                "supported": self.reasoning.supported,
                "controllable": self.reasoning.controllable,
            },
            "limits": {
                "context_tokens": self.limits.context_tokens,
                "max_output_tokens": self.limits.max_output_tokens,
            },
        }


@dataclass(frozen=True, slots=True)
class ModelSpec:
    provider: str
    model_id: str
    protocol: str
    provider_options: JsonObjectInput = field(default_factory=dict, repr=False)
    capabilities: ModelCapabilities = field(kw_only=True)

    def __post_init__(self) -> None:
        for name in ("provider", "model_id"):
            _non_empty(getattr(self, name), name)
        protocol = _non_empty(self.protocol, "protocol")
        if isinstance(self.protocol, BuiltinModelProtocol):
            protocol = self.protocol.value
        object.__setattr__(self, "protocol", protocol)
        if not isinstance(self.capabilities, ModelCapabilities):
            raise TypeError("capabilities must be ModelCapabilities")
        if not isinstance(self.provider_options, Mapping):
            raise TypeError("provider_options must be an object")
        if not isinstance(self.provider_options, FrozenJsonObject):
            object.__setattr__(self, "provider_options", freeze_json_object(self.provider_options))


@dataclass(frozen=True, slots=True)
class ModelEntry:
    name: str
    spec: ModelSpec

    def __post_init__(self) -> None:
        _non_empty(self.name, "model entry name")
        if not isinstance(self.spec, ModelSpec):
            raise TypeError("model entry spec must be ModelSpec")


@dataclass(frozen=True, slots=True)
class ModelGroup:
    name: str
    models: tuple[ModelEntry, ...]
    resolution: ModelGroupResolution = ModelGroupResolution.CONCRETE

    def __post_init__(self) -> None:
        _non_empty(self.name, "model group name")
        if not isinstance(self.resolution, ModelGroupResolution):
            raise TypeError("resolution must be ModelGroupResolution")
        models = tuple(self.models)
        if any(not isinstance(model, ModelEntry) for model in models):
            raise TypeError("model group models must contain ModelEntry values")
        names = tuple(model.name for model in models)
        if len(names) != len(set(names)):
            raise ValueError("model group contains duplicate models")
        if self.resolution is ModelGroupResolution.CONCRETE and not models:
            raise ValueError("concrete model group models must be non-empty")
        if self.resolution is ModelGroupResolution.DEFERRED and models:
            raise ValueError("deferred model group models must be empty")
        object.__setattr__(self, "models", models)

    @classmethod
    def deferred(cls, *, name: str) -> ModelGroup:
        return cls(name=name, models=(), resolution=ModelGroupResolution.DEFERRED)

    @property
    def is_deferred(self) -> bool:
        return self.resolution is ModelGroupResolution.DEFERRED


@dataclass(frozen=True, slots=True)
class CredentialRef:
    _environment_variable: str | None = field(default=None, repr=True)
    _without_authentication: bool = field(default=False, repr=True)

    def __post_init__(self) -> None:
        if (self._environment_variable is None) == (not self._without_authentication):
            raise ValueError("credential must select exactly one of env or none")
        if self._environment_variable is not None and not _ENV_NAME.fullmatch(
            self._environment_variable
        ):
            raise ValueError("credential env must be a valid environment variable name")

    @classmethod
    def environment(cls, name: str) -> CredentialRef:
        return cls(_environment_variable=name)

    @classmethod
    def none(cls) -> CredentialRef:
        return cls(_without_authentication=True)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> CredentialRef:
        unknown = set(value) - {"env", "none"}
        if unknown:
            raise ValueError("unknown credential fields: " + ", ".join(sorted(unknown)))
        if set(value) == {"env"}:
            return cls.environment(_non_empty(value["env"], "credential env"))
        if set(value) == {"none"} and value["none"] is True:
            return cls.none()
        raise ValueError("credential must select exactly one of env or none")

    def resolve(self, environ: Mapping[str, str] | None = None) -> str | None:
        if self._without_authentication:
            return None
        source = os.environ if environ is None else environ
        assert self._environment_variable is not None
        try:
            return source[self._environment_variable]
        except KeyError:
            raise LookupError(
                f"credential environment variable {self._environment_variable!r} is not set"
            ) from None


def _validated_url(value: object, label: str) -> str:
    url = _non_empty(value, label)
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{label} must be an absolute HTTP or HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{label} must not contain embedded credentials")
    if parsed.query or parsed.fragment:
        raise ValueError(f"{label} must not contain a query or fragment")
    return url.rstrip("/")


@dataclass(frozen=True, slots=True)
class ModelConnection:
    base_url: str
    credential: CredentialRef
    verify_ssl: bool = True
    proxy: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", _validated_url(self.base_url, "base_url"))
        if not isinstance(self.credential, CredentialRef):
            raise TypeError("credential must be CredentialRef")
        if not isinstance(self.verify_ssl, bool):
            raise TypeError("verify_ssl must be a bool")
        if self.proxy is not None:
            object.__setattr__(self, "proxy", _validated_url(self.proxy, "proxy"))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ModelConnection:
        unknown = set(value) - _CONNECTION_FIELDS
        if unknown:
            raise ValueError("unknown connection fields: " + ", ".join(sorted(unknown)))
        missing = {"base_url", "credential"} - set(value)
        if missing:
            raise ValueError("missing connection fields: " + ", ".join(sorted(missing)))
        return cls(
            base_url=cast(str, value["base_url"]),
            credential=CredentialRef.from_mapping(_object(value["credential"], "credential")),
            verify_ssl=_bool(value.get("verify_ssl", True), "verify_ssl"),
            proxy=(None if value.get("proxy") is None else cast(str, value["proxy"])),
        )


@dataclass(frozen=True, slots=True)
class ModelConfig:
    models: Mapping[str, ModelEntry]
    model_groups: Mapping[str, ModelGroup]
    connections: Mapping[str, ModelConnection]

    def __post_init__(self) -> None:
        object.__setattr__(self, "models", MappingProxyType(dict(self.models)))
        object.__setattr__(self, "model_groups", MappingProxyType(dict(self.model_groups)))
        object.__setattr__(self, "connections", MappingProxyType(dict(self.connections)))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ModelConfig:
        raw = _object(value, "model config")
        unknown = set(raw) - {"models", "model_groups"}
        if unknown:
            raise ValueError("unknown model config fields: " + ", ".join(sorted(unknown)))
        models_value = _object(raw.get("models"), "models")
        if not models_value:
            raise ValueError("models must be non-empty")
        entries: dict[str, ModelEntry] = {}
        connections: dict[str, ModelConnection] = {}
        for name, item in models_value.items():
            _non_empty(name, "model name")
            model = _object(item, f"model {name!r}")
            unknown_model = set(model) - _MODEL_FIELDS
            if unknown_model:
                raise ValueError("unknown model fields: " + ", ".join(sorted(unknown_model)))
            missing = {"provider", "model_id", "protocol", "connection", "capabilities"} - set(model)
            if missing:
                raise ValueError("missing model fields: " + ", ".join(sorted(missing)))
            spec = ModelSpec(
                provider=_non_empty(model["provider"], "provider"),
                model_id=_non_empty(model["model_id"], "model_id"),
                protocol=_non_empty(model["protocol"], "protocol"),
                provider_options=cast(JsonObjectInput, model.get("provider_options", {})),
                capabilities=ModelCapabilities.from_mapping(
                    _object(model["capabilities"], "capabilities")
                ),
            )
            entry = ModelEntry(name=name, spec=spec)
            entries[name] = entry
            connections[name] = ModelConnection.from_mapping(
                _object(model["connection"], "connection")
            )
        groups_value = _object(raw.get("model_groups", {}), "model_groups")
        groups: dict[str, ModelGroup] = {}
        for name, item in groups_value.items():
            _non_empty(name, "model group name")
            group = _object(item, f"model group {name!r}")
            _exact_fields(group, frozenset({"models"}), "model group")
            raw_model_names = group["models"]
            if (
                isinstance(raw_model_names, (list, tuple))
                and len(raw_model_names) != len(set(raw_model_names))
            ):
                raise ValueError("model group contains duplicate models")
            model_names = _string_tuple(raw_model_names, "model group models")
            unknown_models = set(model_names) - set(entries)
            if unknown_models:
                raise ValueError(
                    "model group references an unknown model: "
                    + ", ".join(sorted(unknown_models))
                )
            groups[name] = ModelGroup(
                name=name, models=tuple(entries[model_name] for model_name in model_names)
            )
        return cls(models=entries, model_groups=groups, connections=connections)


__all__ = [
    "CredentialRef",
    "ModelCapabilities",
    "ModelConfig",
    "ModelConnection",
    "ModelEntry",
    "ModelGroup",
    "ModelLimits",
    "ModelModalities",
    "ModelReasoningCapabilities",
    "ModelSpec",
    "ModelStreamingCapabilities",
    "ModelStructuredOutputCapabilities",
    "ModelToolCapabilities",
]
