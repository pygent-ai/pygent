"""Reuse validators for immutable portable Tool schemas."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

from jsonschema import ValidationError
from jsonschema.validators import validator_for

from pygent.core import FrozenJsonObject, thaw_json

from ._schema import validate_draft202012, validate_instance


@dataclass(frozen=True, slots=True)
class _IntegerMapValidator:
    fallback: Any

    def validate(self, value: object) -> None:
        if type(value) is FrozenJsonObject:
            if all(type(item) is int for _, item in value._items):
                return
            value = thaw_json(value)
        if type(value) is dict and all(type(item) is int for item in value.values()):
            return
        self.fallback.validate(value)


@dataclass(frozen=True, slots=True)
class _IntegerPropertiesValidator:
    fallback: Any
    properties: frozenset[str]
    required: frozenset[str]

    def validate(self, value: object) -> None:
        if type(value) is FrozenJsonObject:
            keys = {name for name, _ in value._items}
            if (
                self.required <= keys <= self.properties
                and all(type(item) is int for _, item in value._items)
            ):
                return
            value = thaw_json(value)
        if (
            type(value) is dict
            and self.required.issubset(value)
            and value.keys() <= self.properties
            and all(type(item) is int for item in value.values())
        ):
            return
        self.fallback.validate(value)


def _simple_integer_object_validator(schema: dict[str, Any], fallback: Any) -> Any:
    if schema.get("type") != "object":
        return fallback
    if (
        schema.keys() <= {"type", "additionalProperties", "title", "description"}
        and schema.get("additionalProperties") == {"type": "integer"}
    ):
        return _IntegerMapValidator(fallback)
    if not schema.keys() <= {
        "type", "properties", "required", "additionalProperties", "title", "description"
    } or schema.get("additionalProperties") is not False:
        return fallback
    properties = schema.get("properties")
    required = schema.get("required", [])
    if not isinstance(properties, dict) or not isinstance(required, list):
        return fallback
    if not all(
        isinstance(name, str)
        and isinstance(prop, dict)
        and prop.get("type") == "integer"
        and prop.keys() <= {"type", "title", "description"}
        for name, prop in properties.items()
    ):
        return fallback
    if any(not isinstance(name, str) or name not in properties for name in required):
        return fallback
    return _IntegerPropertiesValidator(
        fallback, frozenset(properties), frozenset(required)
    )


@lru_cache(maxsize=256)
def validator_for_tool_schema(schema: FrozenJsonObject) -> Any:
    """Compile a Tool schema once while preserving its declared draft."""

    plain_schema = cast(dict[str, Any], thaw_json(schema))
    validator_type = validator_for(plain_schema)
    validator_type.check_schema(plain_schema)
    validator = validator_type(plain_schema)
    return _simple_integer_object_validator(plain_schema, validator)


def validate_tool_value(
    schema: FrozenJsonObject, value: object, *, draft202012: bool = False
) -> None:
    """Validate a frozen value without copying common integer tool payloads."""

    validator = validator_for_tool_schema(schema)
    if isinstance(validator, (_IntegerMapValidator, _IntegerPropertiesValidator)):
        try:
            validator.validate(value)
        except ValidationError:
            # Preserve the caller's draft and error-selection behavior for
            # invalid values; only the successful common case uses the fast path.
            plain_value = thaw_json(cast(Any, value))
            if draft202012:
                validate_draft202012(schema, plain_value)
            else:
                validate_instance(schema, plain_value)
    else:
        plain_value = thaw_json(cast(Any, value))
        if draft202012:
            validate_draft202012(schema, plain_value)
        else:
            validate_instance(schema, plain_value)
