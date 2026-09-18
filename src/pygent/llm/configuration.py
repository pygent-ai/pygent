"""Immutable model semantics and user configuration parsing."""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast
from urllib.parse import urlsplit

from pygent.core import FrozenJsonObject, JsonObjectInput, freeze_json_object, thaw_json

from .protocols import BuiltinModelProtocol
from .types import ModelGroupResolution

_REQUIRED_CAPABILITY_FIELDS = frozenset(
    {"modalities", "streaming", "tools", "structured_output", "reasoning", "limits"}
)
_CAPABILITY_FIELDS = _REQUIRED_CAPABILITY_FIELDS | {"media_input"}
_MODEL_FIELDS = frozenset(
    {
        "connection",
        "model_id",
        "protocol",
        "provider_options",
        "capabilities",
    }
)
_CONNECTION_FIELDS = frozenset(
    {"provider", "credential", "protocols", "verify_ssl", "proxy"}
)
_PROTOCOL_ENDPOINT_FIELDS = frozenset({"base_url"})
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TOOL_CHOICES = frozenset({"none", "auto", "required", "named"})
_MODEL_INPUT_MODALITIES = frozenset({"text", "image", "audio", "video"})
_MODEL_OUTPUT_MODALITIES = _MODEL_INPUT_MODALITIES | {"embedding"}
_MEDIA_RESOLUTION_MODES = frozenset({"low", "high", "original"})
_MIME_TYPE = re.compile(r"^[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*$")


def _object(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return cast(Mapping[str, object], value)


def _exact_fields(
    value: Mapping[str, object], fields: frozenset[str], label: str
) -> None:
    unknown = set(value) - fields
    if unknown:
        raise ValueError(f"unknown {label} fields: " + ", ".join(sorted(unknown)))
    missing = fields - set(value)
    if missing:
        raise ValueError(f"missing {label} fields: " + ", ".join(sorted(missing)))


def _capability_fields(value: Mapping[str, object]) -> None:
    unknown = set(value) - _CAPABILITY_FIELDS
    if unknown:
        raise ValueError("unknown capabilities fields: " + ", ".join(sorted(unknown)))
    missing = _REQUIRED_CAPABILITY_FIELDS - set(value)
    if missing:
        raise ValueError("missing capabilities fields: " + ", ".join(sorted(missing)))


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


def _model_key_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError("model group models must be an array")
    items = tuple(value)
    if any(not isinstance(item, str) or not item for item in items):
        raise ValueError("model group models must contain non-empty strings")
    if len(items) != len(set(items)):
        raise ValueError("model group contains duplicate models")
    return cast(tuple[str, ...], items)


def _bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a bool")
    return value


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _optional_positive_int(value: object, label: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, label)


def _optional_positive_number(value: object, label: str) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{label} must be a positive number")
    return float(value)


def _optional_bool(value: object, label: str) -> bool | None:
    if value is None:
        return None
    return _bool(value, label)


def _mime_types(value: object, label: str, prefix: str) -> tuple[str, ...]:
    items = _string_tuple(value, label)
    if any(
        _MIME_TYPE.fullmatch(item) is None or not item.startswith(prefix)
        for item in items
    ):
        raise ValueError(f"{label} must contain valid {prefix.rstrip('/')} MIME types")
    return items


def _modalities(
    value: object, label: str, allowed: frozenset[str] | set[str]
) -> tuple[str, ...]:
    items = _string_tuple(value, label)
    unknown = set(items) - allowed
    if unknown:
        raise ValueError(f"unsupported {label} values: " + ", ".join(sorted(unknown)))
    return items


@dataclass(frozen=True, slots=True)
class ModelModalities:
    input: tuple[str, ...]
    output: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "input",
            _modalities(self.input, "modalities.input", _MODEL_INPUT_MODALITIES),
        )
        object.__setattr__(
            self,
            "output",
            _modalities(self.output, "modalities.output", _MODEL_OUTPUT_MODALITIES),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ModelModalities:
        _exact_fields(value, frozenset({"input", "output"}), "modalities")
        return cls(
            input=_modalities(
                value["input"], "modalities.input", _MODEL_INPUT_MODALITIES
            ),
            output=_modalities(
                value["output"], "modalities.output", _MODEL_OUTPUT_MODALITIES
            ),
        )


@dataclass(frozen=True, slots=True)
class ModelStreamingCapabilities:
    output: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "output",
            _modalities(self.output, "streaming.output", _MODEL_OUTPUT_MODALITIES),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ModelStreamingCapabilities:
        _exact_fields(value, frozenset({"output"}), "streaming")
        return cls(
            output=_modalities(
                value["output"], "streaming.output", _MODEL_OUTPUT_MODALITIES
            )
        )


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
            raise ValueError(
                "unsupported tools.choice values: " + ", ".join(sorted(unknown))
            )
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
        _exact_fields(
            value, frozenset({"json_object", "json_schema"}), "structured_output"
        )
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
    def from_mapping(cls, value: Mapping[str, object]) -> ModelReasoningCapabilities:
        _exact_fields(value, frozenset({"supported", "controllable"}), "reasoning")
        supported = _bool(value["supported"], "reasoning.supported")
        controllable = _bool(value["controllable"], "reasoning.controllable")
        if controllable and not supported:
            raise ValueError("reasoning.controllable requires reasoning.supported")
        return cls(supported=supported, controllable=controllable)


@dataclass(frozen=True, slots=True)
class ModelLimits:
    context_tokens: int | None
    max_output_tokens: int | None

    def __post_init__(self) -> None:
        _optional_positive_int(self.context_tokens, "limits.context_tokens")
        _optional_positive_int(self.max_output_tokens, "limits.max_output_tokens")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ModelLimits:
        _exact_fields(
            value, frozenset({"context_tokens", "max_output_tokens"}), "limits"
        )
        return cls(
            context_tokens=_optional_positive_int(
                value["context_tokens"], "limits.context_tokens"
            ),
            max_output_tokens=_optional_positive_int(
                value["max_output_tokens"], "limits.max_output_tokens"
            ),
        )


@dataclass(frozen=True, slots=True)
class ModelImageInputCapabilities:
    """Machine-readable image constraints for one model."""

    mime_types: tuple[str, ...] = ()
    max_bytes: int | None = None
    max_width: int | None = None
    max_height: int | None = None
    max_pixels: int | None = None
    resolution_modes: tuple[str, ...] = ()
    animated: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mime_types",
            _mime_types(self.mime_types, "media_input.image.mime_types", "image/"),
        )
        for name in ("max_bytes", "max_width", "max_height", "max_pixels"):
            _optional_positive_int(getattr(self, name), f"media_input.image.{name}")
        modes = _string_tuple(
            self.resolution_modes, "media_input.image.resolution_modes"
        )
        unknown = set(modes) - _MEDIA_RESOLUTION_MODES
        if unknown:
            raise ValueError(
                "unsupported media_input.image.resolution_modes values: "
                + ", ".join(sorted(unknown))
            )
        _optional_bool(self.animated, "media_input.image.animated")
        object.__setattr__(self, "resolution_modes", modes)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ModelImageInputCapabilities:
        fields = frozenset(
            {
                "mime_types",
                "max_bytes",
                "max_width",
                "max_height",
                "max_pixels",
                "resolution_modes",
                "animated",
            }
        )
        _exact_fields(value, fields, "media_input.image")
        return cls(
            mime_types=_mime_types(
                value["mime_types"], "media_input.image.mime_types", "image/"
            ),
            max_bytes=_optional_positive_int(
                value["max_bytes"], "media_input.image.max_bytes"
            ),
            max_width=_optional_positive_int(
                value["max_width"], "media_input.image.max_width"
            ),
            max_height=_optional_positive_int(
                value["max_height"], "media_input.image.max_height"
            ),
            max_pixels=_optional_positive_int(
                value["max_pixels"], "media_input.image.max_pixels"
            ),
            resolution_modes=_string_tuple(
                value["resolution_modes"], "media_input.image.resolution_modes"
            ),
            animated=_optional_bool(value["animated"], "media_input.image.animated"),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "mime_types": list(self.mime_types),
            "max_bytes": self.max_bytes,
            "max_width": self.max_width,
            "max_height": self.max_height,
            "max_pixels": self.max_pixels,
            "resolution_modes": list(self.resolution_modes),
            "animated": self.animated,
        }


@dataclass(frozen=True, slots=True)
class ModelVideoInputCapabilities:
    """Machine-readable native-video constraints for one model."""

    native: bool | None = None
    mime_types: tuple[str, ...] = ()
    max_bytes: int | None = None
    max_duration_seconds: float | None = None
    max_width: int | None = None
    max_height: int | None = None
    max_fps: float | None = None
    audio: bool | None = None
    delivery_modes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _optional_bool(self.native, "media_input.video.native")
        object.__setattr__(
            self,
            "mime_types",
            _mime_types(self.mime_types, "media_input.video.mime_types", "video/"),
        )
        for name in ("max_bytes", "max_width", "max_height"):
            _optional_positive_int(getattr(self, name), f"media_input.video.{name}")
        _optional_positive_number(
            self.max_duration_seconds, "media_input.video.max_duration_seconds"
        )
        _optional_positive_number(self.max_fps, "media_input.video.max_fps")
        _optional_bool(self.audio, "media_input.video.audio")
        delivery_modes = tuple(self.delivery_modes)
        if any(value not in ("video_url", "image_frames") for value in delivery_modes):
            raise ValueError(
                "media_input.video.delivery_modes contains an unsupported value"
            )
        if len(delivery_modes) != len(set(delivery_modes)):
            raise ValueError(
                "media_input.video.delivery_modes must not contain duplicates"
            )
        object.__setattr__(self, "delivery_modes", delivery_modes)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ModelVideoInputCapabilities:
        fields = frozenset(
            {
                "native",
                "mime_types",
                "max_bytes",
                "max_duration_seconds",
                "max_width",
                "max_height",
                "max_fps",
                "audio",
                "delivery_modes",
            }
        )
        normalized = {"delivery_modes": (), **value}
        _exact_fields(normalized, fields, "media_input.video")
        return cls(
            native=_optional_bool(value["native"], "media_input.video.native"),
            mime_types=_mime_types(
                value["mime_types"], "media_input.video.mime_types", "video/"
            ),
            max_bytes=_optional_positive_int(
                value["max_bytes"], "media_input.video.max_bytes"
            ),
            max_duration_seconds=_optional_positive_number(
                value["max_duration_seconds"],
                "media_input.video.max_duration_seconds",
            ),
            max_width=_optional_positive_int(
                value["max_width"], "media_input.video.max_width"
            ),
            max_height=_optional_positive_int(
                value["max_height"], "media_input.video.max_height"
            ),
            max_fps=_optional_positive_number(
                value["max_fps"], "media_input.video.max_fps"
            ),
            audio=_optional_bool(value["audio"], "media_input.video.audio"),
            delivery_modes=_string_tuple(
                normalized["delivery_modes"], "media_input.video.delivery_modes"
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        value: dict[str, object] = {
            "native": self.native,
            "mime_types": list(self.mime_types),
            "max_bytes": self.max_bytes,
            "max_duration_seconds": self.max_duration_seconds,
            "max_width": self.max_width,
            "max_height": self.max_height,
            "max_fps": self.max_fps,
            "audio": self.audio,
        }
        if self.delivery_modes:
            value["delivery_modes"] = list(self.delivery_modes)
        return value


@dataclass(frozen=True, slots=True)
class ModelMediaInputCapabilities:
    """Optional detailed media input limits layered on model modalities."""

    image: ModelImageInputCapabilities | None = None
    video: ModelVideoInputCapabilities | None = None

    def __bool__(self) -> bool:
        return self.image is not None or self.video is not None

    def __post_init__(self) -> None:
        if self.image is not None and not isinstance(
            self.image, ModelImageInputCapabilities
        ):
            raise TypeError(
                "media_input.image must be ModelImageInputCapabilities or None"
            )
        if self.video is not None and not isinstance(
            self.video, ModelVideoInputCapabilities
        ):
            raise TypeError(
                "media_input.video must be ModelVideoInputCapabilities or None"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ModelMediaInputCapabilities:
        _exact_fields(value, frozenset({"image", "video"}), "media_input")
        image = value["image"]
        video = value["video"]
        return cls(
            image=(
                None
                if image is None
                else ModelImageInputCapabilities.from_mapping(
                    _object(image, "media_input.image")
                )
            ),
            video=(
                None
                if video is None
                else ModelVideoInputCapabilities.from_mapping(
                    _object(video, "media_input.video")
                )
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "image": None if self.image is None else self.image.to_mapping(),
            "video": None if self.video is None else self.video.to_mapping(),
        }


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    modalities: ModelModalities
    streaming: ModelStreamingCapabilities
    tools: ModelToolCapabilities
    structured_output: ModelStructuredOutputCapabilities
    reasoning: ModelReasoningCapabilities
    limits: ModelLimits
    media_input: ModelMediaInputCapabilities = field(
        default_factory=ModelMediaInputCapabilities,
        metadata={"pygent_omit_if_empty": True},
    )

    def __post_init__(self) -> None:
        expected = (
            ("modalities", ModelModalities),
            ("streaming", ModelStreamingCapabilities),
            ("tools", ModelToolCapabilities),
            ("structured_output", ModelStructuredOutputCapabilities),
            ("reasoning", ModelReasoningCapabilities),
            ("limits", ModelLimits),
            ("media_input", ModelMediaInputCapabilities),
        )
        for name, value_type in expected:
            if not isinstance(getattr(self, name), value_type):
                raise TypeError(f"capabilities.{name} must be {value_type.__name__}")
        unsupported_streams = set(self.streaming.output) - set(self.modalities.output)
        if unsupported_streams:
            raise ValueError(
                "streaming.output must be a subset of modalities.output: "
                + ", ".join(sorted(unsupported_streams))
            )
        if self.media_input.image is not None and "image" not in self.modalities.input:
            raise ValueError("media_input.image requires modalities.input image")
        if self.media_input.video is not None and "video" not in self.modalities.input:
            raise ValueError("media_input.video requires modalities.input video")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ModelCapabilities:
        _capability_fields(value)
        return cls(
            modalities=ModelModalities.from_mapping(
                _object(value["modalities"], "modalities")
            ),
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
            media_input=(
                ModelMediaInputCapabilities()
                if "media_input" not in value
                else ModelMediaInputCapabilities.from_mapping(
                    _object(value["media_input"], "media_input")
                )
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        value: dict[str, object] = {
            "modalities": {
                "input": list(self.modalities.input),
                "output": list(self.modalities.output),
            },
            "streaming": {"output": list(self.streaming.output)},
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
        if self.media_input != ModelMediaInputCapabilities():
            value["media_input"] = self.media_input.to_mapping()
        return value


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
            object.__setattr__(
                self, "provider_options", freeze_json_object(self.provider_options)
            )


@dataclass(frozen=True, slots=True)
class EnabledModelConfig:
    connection_key: str
    model_id: str
    protocol: str
    capabilities: ModelCapabilities
    provider_options: JsonObjectInput = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for value, label in (
            (self.connection_key, "connection"),
            (self.model_id, "model_id"),
            (self.protocol, "protocol"),
        ):
            _non_empty(value, label)
        if not isinstance(self.capabilities, ModelCapabilities):
            raise TypeError("capabilities must be ModelCapabilities")
        if not isinstance(self.provider_options, Mapping):
            raise TypeError("provider_options must be an object")
        if not isinstance(self.provider_options, FrozenJsonObject):
            object.__setattr__(
                self,
                "provider_options",
                freeze_json_object(cast(JsonObjectInput, self.provider_options)),
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> EnabledModelConfig:
        unknown = set(value) - _MODEL_FIELDS
        if unknown:
            raise ValueError("unknown model fields: " + ", ".join(sorted(unknown)))
        missing = {"connection", "model_id", "protocol", "capabilities"} - set(value)
        if missing:
            raise ValueError("missing model fields: " + ", ".join(sorted(missing)))
        return cls(
            connection_key=_non_empty(value["connection"], "connection"),
            model_id=_non_empty(value["model_id"], "model_id"),
            protocol=_non_empty(value["protocol"], "protocol"),
            provider_options=cast(
                JsonObjectInput,
                _object(value.get("provider_options", {}), "provider_options"),
            ),
            capabilities=ModelCapabilities.from_mapping(
                _object(value["capabilities"], "capabilities")
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "connection": self.connection_key,
            "model_id": self.model_id,
            "protocol": self.protocol,
            "provider_options": thaw_json(
                cast(FrozenJsonObject, self.provider_options)
            ),
            "capabilities": self.capabilities.to_mapping(),
        }


@dataclass(frozen=True, slots=True)
class ModelEntry:
    key: str
    spec: ModelSpec

    def __post_init__(self) -> None:
        _non_empty(self.key, "model entry key")
        if not isinstance(self.spec, ModelSpec):
            raise TypeError("model entry spec must be ModelSpec")


@dataclass(frozen=True, slots=True)
class ModelGroupConfig:
    model_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        model_keys = _model_key_tuple(self.model_keys)
        if not model_keys:
            raise ValueError("model group models must be non-empty")
        object.__setattr__(self, "model_keys", model_keys)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ModelGroupConfig:
        _exact_fields(value, frozenset({"models"}), "model group")
        return cls(model_keys=_model_key_tuple(value["models"]))

    def to_mapping(self) -> dict[str, object]:
        return {"models": list(self.model_keys)}


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
        keys = tuple(model.key for model in models)
        if len(keys) != len(set(keys)):
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

    def to_mapping(self) -> dict[str, object]:
        if self._without_authentication:
            return {"none": True}
        assert self._environment_variable is not None
        return {"env": self._environment_variable}


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
class ConnectionConfig:
    provider: str
    credential: CredentialRef
    protocols: Mapping[str, str]
    verify_ssl: bool = True
    proxy: str | None = None

    def __post_init__(self) -> None:
        _non_empty(self.provider, "provider")
        if not isinstance(self.credential, CredentialRef):
            raise TypeError("credential must be CredentialRef")
        if not isinstance(self.protocols, Mapping):
            raise TypeError("protocols must be a mapping")
        protocols: dict[str, str] = {}
        for protocol, base_url in self.protocols.items():
            protocol = _non_empty(protocol, "protocol")
            protocols[protocol] = _validated_url(base_url, "base_url")
        if not protocols:
            raise ValueError("protocols must be non-empty")
        if not isinstance(self.verify_ssl, bool):
            raise TypeError("verify_ssl must be a bool")
        if self.proxy is not None:
            object.__setattr__(self, "proxy", _validated_url(self.proxy, "proxy"))
        object.__setattr__(self, "protocols", MappingProxyType(protocols))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ConnectionConfig:
        unknown = set(value) - _CONNECTION_FIELDS
        if unknown:
            raise ValueError("unknown connection fields: " + ", ".join(sorted(unknown)))
        missing = {"provider", "credential", "protocols"} - set(value)
        if missing:
            raise ValueError("missing connection fields: " + ", ".join(sorted(missing)))
        raw_protocols = _object(value["protocols"], "protocols")
        protocols: dict[str, str] = {}
        for protocol, item in raw_protocols.items():
            protocol = _non_empty(protocol, "protocol")
            endpoint = _object(item, f"protocol endpoint {protocol!r}")
            _exact_fields(
                endpoint,
                _PROTOCOL_ENDPOINT_FIELDS,
                "protocol endpoint",
            )
            protocols[protocol] = _validated_url(endpoint["base_url"], "base_url")
        return cls(
            provider=_non_empty(value["provider"], "provider"),
            credential=CredentialRef.from_mapping(
                _object(value["credential"], "credential")
            ),
            protocols=protocols,
            verify_ssl=_bool(value.get("verify_ssl", True), "verify_ssl"),
            proxy=(None if value.get("proxy") is None else cast(str, value["proxy"])),
        )

    def to_mapping(self) -> dict[str, object]:
        value: dict[str, object] = {
            "provider": self.provider,
            "credential": self.credential.to_mapping(),
            "protocols": {
                protocol: {"base_url": base_url}
                for protocol, base_url in self.protocols.items()
            },
            "verify_ssl": self.verify_ssl,
        }
        if self.proxy is not None:
            value["proxy"] = self.proxy
        return value


@dataclass(frozen=True, slots=True)
class ResolvedModelConnection:
    connection_key: str
    provider: str
    protocol: str
    base_url: str
    credential: CredentialRef
    verify_ssl: bool = True
    proxy: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("connection_key", "provider", "protocol"):
            _non_empty(getattr(self, field_name), field_name)
        object.__setattr__(self, "base_url", _validated_url(self.base_url, "base_url"))
        if not isinstance(self.credential, CredentialRef):
            raise TypeError("credential must be CredentialRef")
        if not isinstance(self.verify_ssl, bool):
            raise TypeError("verify_ssl must be a bool")
        if self.proxy is not None:
            object.__setattr__(self, "proxy", _validated_url(self.proxy, "proxy"))


@dataclass(frozen=True, slots=True, init=False)
class ModelConfig:
    connections: Mapping[str, ConnectionConfig]
    models: Mapping[str, ModelEntry]
    model_groups: Mapping[str, ModelGroup]
    _model_connections: Mapping[str, str] = field(repr=False)

    def __init__(self) -> None:
        raise TypeError(
            "ModelConfig must be created with from_mapping() or from_components()"
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ModelConfig:
        raw = _object(value, "model config")
        unknown = set(raw) - {"connections", "models", "model_groups"}
        if unknown:
            raise ValueError(
                "unknown model config fields: " + ", ".join(sorted(unknown))
            )
        missing = {"connections", "models"} - set(raw)
        if missing:
            raise ValueError(
                "missing model config fields: " + ", ".join(sorted(missing))
            )
        connections_value = _object(raw["connections"], "connections")
        if not connections_value:
            raise ValueError("connections must be non-empty")
        connections: dict[str, ConnectionConfig] = {}
        for name, item in connections_value.items():
            _non_empty(name, "connection name")
            connections[name] = ConnectionConfig.from_mapping(
                _object(item, f"connection {name!r}")
            )
        models_value = _object(raw.get("models"), "models")
        if not models_value:
            raise ValueError("models must be non-empty")
        models: dict[str, EnabledModelConfig] = {}
        for name, item in models_value.items():
            _non_empty(name, "model key")
            models[name] = EnabledModelConfig.from_mapping(
                _object(item, f"model {name!r}")
            )
        groups_value = _object(raw.get("model_groups", {}), "model_groups")
        groups: dict[str, ModelGroupConfig] = {}
        for name, item in groups_value.items():
            _non_empty(name, "model group name")
            groups[name] = ModelGroupConfig.from_mapping(
                _object(item, f"model group {name!r}")
            )
        return cls.from_components(
            connections=connections,
            models=models,
            model_groups=groups,
        )

    @classmethod
    def from_components(
        cls,
        *,
        connections: Mapping[str, ConnectionConfig],
        models: Mapping[str, EnabledModelConfig],
        model_groups: Mapping[str, ModelGroupConfig] | None = None,
    ) -> ModelConfig:
        if not isinstance(connections, Mapping):
            raise TypeError("connections must be a mapping")
        if not isinstance(models, Mapping):
            raise TypeError("models must be a mapping")
        if model_groups is not None and not isinstance(model_groups, Mapping):
            raise TypeError("model_groups must be a mapping")
        configured_connections = dict(connections)
        configured_models = dict(models)
        configured_groups = dict(model_groups or {})
        if not configured_connections:
            raise ValueError("connections must be non-empty")
        if not configured_models:
            raise ValueError("models must be non-empty")
        for key, connection in configured_connections.items():
            _non_empty(key, "connection name")
            if not isinstance(connection, ConnectionConfig):
                raise TypeError("connections must contain ConnectionConfig values")
        entries: dict[str, ModelEntry] = {}
        model_connections: dict[str, str] = {}
        for key, configured_model in configured_models.items():
            _non_empty(key, "model key")
            if not isinstance(configured_model, EnabledModelConfig):
                raise TypeError("models must contain EnabledModelConfig values")
            connection_name = configured_model.connection_key
            try:
                connection = configured_connections[connection_name]
            except KeyError:
                raise ValueError(
                    f"model {key!r} references unknown connection {connection_name!r}"
                ) from None
            protocol = configured_model.protocol
            if protocol not in connection.protocols:
                raise ValueError(
                    f"connection {connection_name!r} does not provide protocol {protocol!r}"
                )
            spec = ModelSpec(
                provider=connection.provider,
                model_id=configured_model.model_id,
                protocol=protocol,
                provider_options=cast(
                    JsonObjectInput, configured_model.provider_options
                ),
                capabilities=configured_model.capabilities,
            )
            entry = ModelEntry(key=key, spec=spec)
            entries[key] = entry
            model_connections[key] = connection_name
        groups: dict[str, ModelGroup] = {}
        for key, configured_group in configured_groups.items():
            _non_empty(key, "model group name")
            if not isinstance(configured_group, ModelGroupConfig):
                raise TypeError("model_groups must contain ModelGroupConfig values")
            model_names = configured_group.model_keys
            unknown_models = set(model_names) - set(entries)
            if unknown_models:
                raise ValueError(
                    "model group references an unknown model: "
                    + ", ".join(sorted(unknown_models))
                )
            groups[key] = ModelGroup(
                name=key,
                models=tuple(entries[model_name] for model_name in model_names),
            )
        result = cls.__new__(cls)
        object.__setattr__(
            result,
            "connections",
            MappingProxyType(configured_connections),
        )
        object.__setattr__(result, "models", MappingProxyType(entries))
        object.__setattr__(result, "model_groups", MappingProxyType(groups))
        object.__setattr__(
            result,
            "_model_connections",
            MappingProxyType(model_connections),
        )
        return result

    def connection_for(self, model_key: str) -> ResolvedModelConnection:
        entry = self.models[model_key]
        connection_name = self._model_connections[model_key]
        connection = self.connections[connection_name]
        protocol = entry.spec.protocol
        return ResolvedModelConnection(
            connection_key=connection_name,
            provider=connection.provider,
            protocol=protocol,
            base_url=connection.protocols[protocol],
            credential=connection.credential,
            verify_ssl=connection.verify_ssl,
            proxy=connection.proxy,
        )


__all__ = [
    "ConnectionConfig",
    "CredentialRef",
    "EnabledModelConfig",
    "ModelCapabilities",
    "ModelConfig",
    "ModelEntry",
    "ModelGroup",
    "ModelGroupConfig",
    "ModelImageInputCapabilities",
    "ModelLimits",
    "ModelMediaInputCapabilities",
    "ModelModalities",
    "ModelReasoningCapabilities",
    "ModelSpec",
    "ModelStreamingCapabilities",
    "ModelStructuredOutputCapabilities",
    "ModelToolCapabilities",
    "ModelVideoInputCapabilities",
    "ResolvedModelConnection",
]
