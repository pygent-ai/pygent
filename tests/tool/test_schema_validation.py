from __future__ import annotations

import pytest
from jsonschema import Draft202012Validator, ValidationError

from pygent.core import freeze_json_object
from pygent.tool._schema_validation import (
    validate_tool_value,
    validator_for_tool_schema,
)


@pytest.mark.parametrize(
    ("schema", "values"),
    [
        (
            {"type": "object", "additionalProperties": {"type": "integer"}},
            [{"step": 1}, {}, {"step": True}, {"step": 1.0}, {"step": "1"}],
        ),
        (
            {
                "type": "object",
                "title": "Arguments",
                "properties": {"step": {"type": "integer", "title": "Step"}},
                "required": ["step"],
                "additionalProperties": False,
            },
            [
                {"step": 1},
                {},
                {"step": True},
                {"step": 1.0},
                {"step": "1"},
                {"step": 1, "extra": 2},
            ],
        ),
    ],
)
def test_simple_tool_schema_fast_path_matches_jsonschema(schema, values) -> None:
    frozen_schema = freeze_json_object(schema)
    validator = validator_for_tool_schema(frozen_schema)
    reference = Draft202012Validator(schema)
    assert validator is validator_for_tool_schema(freeze_json_object(schema))
    for value in values:
        try:
            reference.validate(value)
        except ValidationError as expected:
            with pytest.raises(ValidationError) as actual:
                validator.validate(value)
            assert actual.value.message == expected.message
            with pytest.raises(ValidationError) as frozen_actual:
                validate_tool_value(frozen_schema, freeze_json_object(value))
            assert frozen_actual.value.message == expected.message
        else:
            validator.validate(value)
            validate_tool_value(frozen_schema, freeze_json_object(value))
