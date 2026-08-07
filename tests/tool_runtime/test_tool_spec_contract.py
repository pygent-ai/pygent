"""Portable ToolSpec and result-safety contracts."""

import pytest

from pygent import (
    IdempotencyPolicy,
    JsonValueError,
    ToolAuthorizationDecision,
    ToolAuthorizationRequest,
    ToolCall,
    ToolCallLayer,
    ToolDefinition,
    ToolResult,
    ToolSideEffect,
    ToolSpec,
    ToolTask,
    thaw_json,
)


@pytest.mark.parametrize(
    "portable_type",
    [ToolDefinition, ToolSpec, ToolCall, ToolTask, ToolResult],
)
def test_portable_tool_values_are_sealed(portable_type):
    with pytest.raises(TypeError, match="cannot be subclassed"):
        type("UnsafePortableValue", (portable_type,), {})


def _weather_spec() -> ToolSpec:
    return ToolSpec(
        tool_id="weather.lookup",
        version="1.0.0",
        definition=ToolDefinition(
            name="weather.lookup",
            description="Look up weather",
            parameters={"type": "object"},
        ),
        side_effect=ToolSideEffect.READ,
        idempotency=IdempotencyPolicy.INHERENT,
        timeout=5.0,
        resource_key="weather-api",
        required_permissions=("weather:read",),
    )


def test_tool_layer_exposes_definitions_without_execution_policy():
    spec = _weather_spec()
    layer = ToolCallLayer(tools=(spec,), max_concurrency=8)

    assert layer.tools == (spec,)
    assert layer.definitions == (spec.definition,)


def test_tool_arguments_and_results_are_strict_json_values():
    call = ToolCall(
        call_id="call-1",
        name="weather.lookup",
        arguments={"city": "Shanghai"},
        tool_id="weather.lookup",
        tool_version="1.0.0",
    )
    result = ToolResult(
        call_id="call-1",
        name="weather.lookup",
        status="succeeded",
        output={"temperature": 28},
        side_effect_committed=False,
    )

    assert call.arguments["city"] == "Shanghai"
    assert thaw_json(result.output) == {"temperature": 28}

    with pytest.raises(JsonValueError):
        ToolResult(
            call_id="call-2",
            name="bad",
            status="failed",
            output=object(),
        )


def test_capacity_belongs_to_the_tool_layer_not_tool_spec():
    definition = _weather_spec().definition

    assert "max_concurrency" not in ToolSpec.__dataclass_fields__
    with pytest.raises(ValueError, match="layer max_concurrency"):
        ToolCallLayer(tools=(_weather_spec(),), max_concurrency=0)

    with pytest.raises(ValueError):
        ToolSpec(
            tool_id="weather.lookup",
            version="1.0.0",
            definition=definition,
            timeout=0,
        )


def test_authorization_values_preserve_call_identity_and_lifecycle():
    spec = _weather_spec()
    call = ToolCall(
        call_id="call-1",
        name=spec.definition.name,
        arguments={"city": "Shanghai"},
    )
    request = ToolAuthorizationRequest(
        call=call,
        spec=spec,
        permissions=("weather:read",),
    )
    decision = ToolAuthorizationDecision(
        call_id=call.call_id,
        allowed=True,
        reason_code="allowed",
        lifecycle="sync",
    )

    assert request.call is call
    assert request.permissions == ("weather:read",)
    assert decision.call_id == call.call_id

    with pytest.raises(ValueError, match="lifecycle"):
        ToolAuthorizationDecision(
            call_id=call.call_id,
            allowed=True,
            reason_code="allowed",
            lifecycle="background",  # type: ignore[arg-type]
        )
