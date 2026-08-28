"""Portable codecs for concrete Context value types."""

from __future__ import annotations

import hashlib
import json
import types
from collections.abc import Mapping
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from types import MappingProxyType
from typing import Any, Union, cast, get_args, get_origin, get_type_hints

from pygent.core import Context, FrozenJsonObject, Message, freeze_json, thaw_json
from pygent.tool import ToolDefinition

CONTEXT_CODEC_NAME = "pygent-dataclass-json-v1"
_BASE_ANNOTATIONS: dict[str, object] = {
    "system_prompt": str,
    "messages": tuple[Message, ...],
    "tools": tuple[ToolDefinition, ...],
    "metadata": FrozenJsonObject,
    "projection_revision": int,
}


class ContextCodecError(ValueError):
    """Raised when a Context type or value violates the portable codec contract."""


def _annotation_schema(annotation: object) -> object:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if annotation in (str, int, float, bool, type(None)):
        return {"type": getattr(annotation, "__name__", "null")}
    if annotation in (Any, object):
        raise ContextCodecError("Any and object are not portable Context annotations")
    if annotation is FrozenJsonObject:
        return {"type": "strict-json-object"}
    if isinstance(annotation, type) and issubclass(annotation, Message):
        return {"type": "pygent-message"}
    if annotation is ToolDefinition:
        return {"type": "pygent-tool-definition"}
    if origin is tuple:
        if len(args) != 2 or args[1] is not Ellipsis:
            raise ContextCodecError("Context tuple annotations must be homogeneous")
        return {"type": "array", "items": _annotation_schema(args[0])}
    if origin in (Union, types.UnionType):
        return {"oneOf": [_annotation_schema(item) for item in args]}
    if isinstance(annotation, type) and is_dataclass(annotation):
        parameters = getattr(annotation, "__dataclass_params__", None)
        if parameters is None or not parameters.frozen or "__slots__" not in annotation.__dict__:
            raise ContextCodecError("nested dataclasses must be frozen and use slots")
        hints = get_type_hints(annotation)
        return {
            "type": "dataclass",
            "fields": [
                {"name": item.name, "schema": _annotation_schema(hints[item.name])}
                for item in fields(annotation)
            ],
        }
    raise ContextCodecError(f"unsupported Context annotation: {annotation!r}")


def _context_hints(context_type: type[Context]) -> dict[str, object]:
    if context_type is Context:
        return dict(_BASE_ANNOTATIONS)
    annotations = dict(getattr(context_type, "__annotations__", {}))
    module_globals = vars(__import__(context_type.__module__, fromlist=["*"]))
    resolved: dict[str, object] = {}
    for name, annotation in annotations.items():
        if isinstance(annotation, str):
            try:
                annotation = eval(annotation, module_globals, vars(context_type))
            except (NameError, TypeError) as exc:
                raise ContextCodecError(
                    f"cannot resolve Context annotation for {name!r}"
                ) from exc
        resolved[name] = annotation
    return {**resolved, **_BASE_ANNOTATIONS}


def _encode_value(value: object, annotation: object) -> object:
    from .codec import message_to_dict, tool_definition_to_dict

    origin = get_origin(annotation)
    args = get_args(annotation)
    if annotation in (str, int, float, bool, type(None)):
        if annotation is int and isinstance(value, bool):
            raise ContextCodecError("bool is not a valid int Context value")
        if not isinstance(value, cast(type[Any], annotation)):
            raise ContextCodecError(f"expected {annotation}, got {type(value).__name__}")
        return value
    if annotation is FrozenJsonObject:
        if not isinstance(value, FrozenJsonObject):
            raise ContextCodecError("expected FrozenJsonObject")
        return thaw_json(value)
    if isinstance(annotation, type) and issubclass(annotation, Message):
        if not isinstance(value, cast(type[Any], annotation)):
            raise ContextCodecError("Context Message value has the wrong type")
        return message_to_dict(value)
    if annotation is ToolDefinition:
        if type(value) is not ToolDefinition:
            raise ContextCodecError("expected ToolDefinition")
        return tool_definition_to_dict(value)
    if origin is tuple:
        if not isinstance(value, tuple):
            raise ContextCodecError("expected tuple")
        return [_encode_value(item, args[0]) for item in value]
    if origin in (Union, types.UnionType):
        successes: list[object] = []
        for candidate in args:
            try:
                successes.append(_encode_value(value, candidate))
            except (ContextCodecError, TypeError, ValueError):
                pass
        if len(successes) != 1:
            raise ContextCodecError("Context union value is ambiguous or invalid")
        return successes[0]
    if isinstance(annotation, type) and is_dataclass(annotation):
        if type(value) is not annotation:
            raise ContextCodecError("nested Context dataclass has the wrong type")
        hints = get_type_hints(annotation)
        return {
            item.name: _encode_value(getattr(value, item.name), hints[item.name])
            for item in fields(annotation)
        }
    raise ContextCodecError(f"unsupported Context annotation: {annotation!r}")


def _decode_value(value: object, annotation: object) -> object:
    from .codec import message_from_dict, tool_definition_from_dict

    origin = get_origin(annotation)
    args = get_args(annotation)
    if annotation in (str, int, float, bool, type(None)):
        if annotation is int and isinstance(value, bool):
            raise ContextCodecError("bool is not a valid int Context value")
        if not isinstance(value, cast(type[Any], annotation)):
            raise ContextCodecError(f"expected {annotation}, got {type(value).__name__}")
        return value
    if annotation is FrozenJsonObject:
        if not isinstance(value, Mapping):
            raise ContextCodecError("expected JSON object")
        frozen = freeze_json(value)
        if not isinstance(frozen, FrozenJsonObject):
            raise ContextCodecError("expected JSON object")
        return frozen
    if isinstance(annotation, type) and issubclass(annotation, Message):
        decoded = message_from_dict(value)
        if not isinstance(decoded, annotation):
            raise ContextCodecError("decoded Message has the wrong type")
        return decoded
    if annotation is ToolDefinition:
        return tool_definition_from_dict(value)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ContextCodecError("expected array")
        return tuple(_decode_value(item, args[0]) for item in value)
    if origin in (Union, types.UnionType):
        successes: list[object] = []
        for candidate in args:
            try:
                successes.append(_decode_value(value, candidate))
            except (ContextCodecError, TypeError, ValueError):
                pass
        if len(successes) != 1:
            raise ContextCodecError("Context union value is ambiguous or invalid")
        return successes[0]
    if isinstance(annotation, type) and is_dataclass(annotation):
        if not isinstance(value, Mapping):
            raise ContextCodecError("expected dataclass object")
        declared = {item.name for item in fields(annotation)}
        if set(value) != declared:
            raise ContextCodecError("nested dataclass fields do not match schema")
        hints = get_type_hints(annotation)
        return annotation(
            **{
                item.name: _decode_value(value[item.name], hints[item.name])
                for item in fields(annotation)
            }
        )
    raise ContextCodecError(f"unsupported Context annotation: {annotation!r}")


@dataclass(frozen=True, slots=True)
class ContextCodec:
    context_type: type[Context]
    schema: str
    version: int
    codec: str
    codec_digest: str
    canonical_schema: FrozenJsonObject
    _hints: Mapping[str, object] = field(repr=False, compare=False)

    @classmethod
    def dataclass(cls, context_type: type[Context]) -> ContextCodec:
        if not isinstance(context_type, type) or not issubclass(context_type, Context):
            raise ContextCodecError("context_type must be a Context subclass")
        if not is_dataclass(context_type):
            raise ContextCodecError("Context type must be a dataclass")
        schema = getattr(context_type, "context_schema", None)
        version = getattr(context_type, "context_schema_version", None)
        if not isinstance(schema, str) or not schema:
            raise ContextCodecError("Context type requires context_schema")
        if not isinstance(version, int) or isinstance(version, bool) or version <= 0:
            raise ContextCodecError("Context type requires a positive schema version")
        hints = _context_hints(context_type)
        field_schema = []
        for item in fields(context_type):
            if item.name not in hints:
                raise ContextCodecError(f"Context field {item.name!r} lacks an annotation")
            if item.default_factory is not MISSING:  # type: ignore[misc]
                factory = item.default_factory  # type: ignore[misc]
                if factory in (list, dict, set):
                    raise ContextCodecError("mutable Context default factory")
            field_schema.append(
                {"name": item.name, "schema": _annotation_schema(hints[item.name])}
            )
        canonical = {
            "schema": schema,
            "version": version,
            "codec": CONTEXT_CODEC_NAME,
            "fields": field_schema,
        }
        encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        digest = f"sha256:{hashlib.sha256(encoded).hexdigest()}"
        frozen = freeze_json(canonical)
        assert isinstance(frozen, FrozenJsonObject)
        return cls(
            context_type,
            schema,
            version,
            CONTEXT_CODEC_NAME,
            digest,
            frozen,
            MappingProxyType(hints),
        )

    @property
    def identity(self) -> tuple[str, int, str, str]:
        return self.schema, self.version, self.codec, self.codec_digest

    def encode(self, value: Context) -> dict[str, object]:
        if type(value) is not self.context_type:
            raise ContextCodecError("Context value does not match codec type")
        return {
            item.name: _encode_value(getattr(value, item.name), self._hints[item.name])
            for item in fields(self.context_type)
        }

    def decode(self, value: object) -> Context:
        if not isinstance(value, Mapping):
            raise ContextCodecError("Context data must be an object")
        declared = {item.name for item in fields(self.context_type)}
        if set(value) != declared:
            raise ContextCodecError("Context data fields do not match codec schema")
        constructor = cast(Any, self.context_type)
        return constructor(
            **{
                item.name: _decode_value(value[item.name], self._hints[item.name])
                for item in fields(self.context_type)
            }
        )


class ContextCodecRegistry:
    def __init__(self, codecs: tuple[ContextCodec, ...] = ()) -> None:
        values = (BASE_CONTEXT_CODEC, *tuple(codecs))
        by_identity: dict[tuple[str, int, str, str], ContextCodec] = {}
        by_type: dict[type[Context], ContextCodec] = {}
        by_schema: dict[tuple[str, int], ContextCodec] = {}
        for item in values:
            if not isinstance(item, ContextCodec):
                raise TypeError("context_codecs must contain ContextCodec values")
            key = (item.schema, item.version)
            if item.context_type in by_type or item.identity in by_identity or key in by_schema:
                raise ContextCodecError("duplicate or conflicting Context codec")
            by_type[item.context_type] = item
            by_identity[item.identity] = item
            by_schema[key] = item
        self._by_type = by_type
        self._by_identity = by_identity

    def _validate_registration(self, codec: ContextCodec) -> ContextCodec:
        if not isinstance(codec, ContextCodec):
            raise TypeError("codec must be a ContextCodec")
        current = self._by_type.get(codec.context_type)
        if current is not None:
            if current == codec:
                return current
            raise ContextCodecError("duplicate or conflicting Context codec")
        schema_key = (codec.schema, codec.version)
        if codec.identity in self._by_identity or any(
            (item.schema, item.version) == schema_key
            for item in self._by_type.values()
        ):
            raise ContextCodecError("duplicate or conflicting Context codec")
        return codec

    def _register(self, codec: ContextCodec) -> ContextCodec:
        """Register one deployment-local codec, accepting an identical repeat."""

        codec = self._validate_registration(codec)
        current = self._by_type.get(codec.context_type)
        if current is not None:
            return current
        self._by_type[codec.context_type] = codec
        self._by_identity[codec.identity] = codec
        return codec

    @property
    def codecs(self) -> tuple[ContextCodec, ...]:
        return tuple(self._by_type.values())

    def for_value(self, value: Context) -> ContextCodec:
        try:
            return self._by_type[type(value)]
        except KeyError as exc:
            raise ContextCodecError("unregistered Context type") from exc

    def for_type(self, context_type: type[Context]) -> ContextCodec:
        try:
            return self._by_type[context_type]
        except KeyError as exc:
            raise ContextCodecError("unregistered Context type") from exc

    def for_identity(self, identity: tuple[str, int, str, str]) -> ContextCodec:
        try:
            return self._by_identity[identity]
        except KeyError as exc:
            raise ContextCodecError("unknown or incompatible Context codec") from exc


BASE_CONTEXT_CODEC = ContextCodec.dataclass(Context)
DEFAULT_CONTEXT_CODECS = ContextCodecRegistry()


__all__ = [
    "BASE_CONTEXT_CODEC",
    "CONTEXT_CODEC_NAME",
    "DEFAULT_CONTEXT_CODECS",
    "ContextCodec",
    "ContextCodecError",
    "ContextCodecRegistry",
]
