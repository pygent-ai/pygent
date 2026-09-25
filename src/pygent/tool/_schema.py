"""Compiled JSON Schema validators reused across tool calls.

``jsonschema.validate`` re-checks a schema against its meta-schema and rebuilds
a validator for every validated instance. Declared schemas are frozen and
immutable, so the compiled validator and its thawed document are cached per
schema value and reused instead.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, cast

from jsonschema import Draft202012Validator, exceptions
from jsonschema.protocols import Validator
from jsonschema.validators import validator_for

from pygent.core import FrozenJsonObject, thaw_json

# Schemas are declared per tool and per generation request; the bound only
# protects long-lived processes that keep declaring new schemas.
_MAX_CACHED_SCHEMAS = 512


@lru_cache(maxsize=_MAX_CACHED_SCHEMAS)
def _draft202012(schema: FrozenJsonObject) -> Validator:
    return Draft202012Validator(cast(dict[str, Any], thaw_json(schema)))


@lru_cache(maxsize=_MAX_CACHED_SCHEMAS)
def _declared(schema: FrozenJsonObject) -> Validator:
    document = cast(dict[str, Any], thaw_json(schema))
    validator_class = validator_for(document)
    validator_class.check_schema(document)
    return validator_class(document)


def validate_draft202012(schema: FrozenJsonObject, instance: object) -> None:
    """Validate one instance under Draft 2020-12 rules.

    Raises the first error, matching ``Draft202012Validator.validate``: the
    caller already knows the schema is a Draft 2020-12 sub-schema.
    """

    _draft202012(schema).validate(cast(Any, instance))


def validate_instance(schema: FrozenJsonObject, instance: object) -> None:
    """Validate one instance the way ``jsonschema.validate`` does.

    The validator class follows the schema's ``$schema`` keyword, the schema is
    meta-checked once instead of on every call, and the raised error is the
    best-match error rather than the first one.
    """

    error = exceptions.best_match(_declared(schema).iter_errors(cast(Any, instance)))
    if error is not None:
        raise error


__all__ = ["validate_draft202012", "validate_instance"]
