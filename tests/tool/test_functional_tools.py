from __future__ import annotations

from dataclasses import replace
from typing import Annotated, Literal, TypedDict

import pytest
from pydantic import BaseModel, Field

from pygent import (
    AIMessage,
    Context,
    IdempotencyPolicy,
    LocalRuntime,
    ToolAuthorizationDecision,
    ToolCall,
    ToolCallLayer,
    ToolDefinition,
    ToolKit,
    ToolSideEffect,
    compile_execution_plan,
    tool,
)
from pygent.tool import ExecutorRegistry


def allow(request, _context):
    return ToolAuthorizationDecision(
        call_id=request.call.call_id,
        allowed=True,
        reason_code="allowed",
    )


class Point(TypedDict):
    x: int
    y: int


class Measurement(BaseModel):
    value: int
    unit: Literal["c"] = "c"


@tool(
    tool_id="weather.measure",
    version="1.0.0",
    side_effect=ToolSideEffect.READ,
    timeout=2,
    resource_key="weather-api",
    required_permissions=("weather:read",),
)
async def measure(
    city: Annotated[str, Field(description="City selected explicitly")],
    point: Point,
    units: Literal["c", "f"] = "c",
) -> Measurement:
    """Measure the weather at a point.

    Args:
        city: City from the Google-style docstring.
        point: Coordinates to inspect.
        units: Desired temperature units.
    """

    assert city
    return Measurement(value=point["x"] + point["y"], unit="c")


def test_toolkit_compiles_pydantic_schemas_and_portable_policy() -> None:
    toolkit = ToolKit(measure)
    definition = toolkit.definitions[0]
    spec = toolkit.specs[0]
    parameters = definition.parameters.to_dict()
    output_schema = definition.output_schema.to_dict()

    assert definition.name == "measure"
    assert definition.description == "Measure the weather at a point."
    assert parameters["additionalProperties"] is False
    assert parameters["properties"]["city"]["description"] == (
        "City selected explicitly"
    )
    assert parameters["properties"]["point"]["description"] == (
        "Coordinates to inspect."
    )
    assert parameters["properties"]["units"]["default"] == "c"
    assert output_schema["properties"]["value"]["type"] == "integer"
    assert spec.tool_id == "weather.measure"
    assert spec.version == "1.0.0"
    assert spec.side_effect is ToolSideEffect.READ
    assert spec.idempotency is IdempotencyPolicy.INHERENT
    assert spec.timeout == 2
    assert spec.resource_key == "weather-api"
    assert spec.required_permissions == ("weather:read",)


@pytest.mark.asyncio
async def test_local_layer_executes_async_function_and_serializes_model() -> None:
    toolkit = ToolKit(measure)
    layer = toolkit.local_layer(authorization_adapter=allow)
    context = toolkit.make_visible_in(Context())

    message, returned_context = await layer.invoke(
        AIMessage(
            tool_calls=(
                ToolCall(
                    call_id="measure-1",
                    name="measure",
                    arguments={"city": "Shanghai", "point": {"x": 2, "y": 3}},
                ),
            )
        ),
        context,
    )

    assert message.results[0].status == "succeeded"
    assert message.results[0].output.to_dict() == {"value": 5, "unit": "c"}
    assert returned_context is context


@pytest.mark.asyncio
async def test_sync_function_and_bound_method_are_supported() -> None:
    @tool(tool_id="math.add", version="1", side_effect=ToolSideEffect.PURE)
    def add(left: int, right: int = 1) -> int:
        """Add two integers."""

        return left + right

    class Calculator:
        def __init__(self, offset: int) -> None:
            self.offset = offset

        @tool(tool_id="math.offset", version="1", side_effect=ToolSideEffect.PURE)
        def apply(self, value: int) -> int:
            """Apply the configured offset."""

            return value + self.offset

    toolkit = ToolKit(add, Calculator(4).apply)

    assert [item.name for item in toolkit.definitions] == ["add", "apply"]
    assert "self" not in toolkit.definitions[1].parameters["properties"]
    message, _ = await toolkit.local_layer(authorization_adapter=allow).invoke(
        AIMessage(
            tool_calls=(
                ToolCall("add-1", "add", {"left": 2}),
                ToolCall("apply-1", "apply", {"value": 3}),
            )
        ),
        toolkit.make_visible_in(Context()),
    )
    assert [result.output for result in message.results] == [3, 7]


@pytest.mark.asyncio
async def test_invalid_return_uses_existing_output_validation_result() -> None:
    @tool(tool_id="bad.output", version="1", side_effect=ToolSideEffect.PURE)
    def bad(value: int) -> int:
        """Return an invalid output for verification."""

        return "wrong"  # type: ignore[return-value]

    toolkit = ToolKit(bad)
    message, _ = await toolkit.local_layer(authorization_adapter=allow).invoke(
        AIMessage(tool_calls=(ToolCall("bad-1", "bad", {"value": 1}),)),
        toolkit.make_visible_in(Context()),
    )

    result = message.results[0]
    assert result.status == "failed"
    assert result.error_code == "invalid_output"
    assert result.side_effect_committed is True


@pytest.mark.asyncio
async def test_extra_input_is_rejected_before_handler() -> None:
    calls = 0

    @tool(tool_id="strict", version="1", side_effect=ToolSideEffect.PURE)
    def strict(value: int) -> int:
        """Accept one strict input."""

        nonlocal calls
        calls += 1
        return value

    toolkit = ToolKit(strict)
    message, _ = await toolkit.local_layer(authorization_adapter=allow).invoke(
        AIMessage(
            tool_calls=(ToolCall("strict-1", "strict", {"value": 1, "extra": 2}),)
        ),
        toolkit.make_visible_in(Context()),
    )

    assert calls == 0
    assert message.results[0].error_code == "invalid_arguments"


@pytest.mark.asyncio
async def test_none_return_and_handler_exception_keep_existing_boundaries() -> None:
    @tool(tool_id="none", version="1", side_effect=ToolSideEffect.PURE)
    def returns_none(value: int) -> None:
        """Consume a value without an output object."""

    @tool(tool_id="raises", version="1", side_effect=ToolSideEffect.PURE)
    def raises(value: int) -> int:
        """Raise from the local handler."""

        raise RuntimeError(str(value))

    toolkit = ToolKit(returns_none, raises)
    message, _ = await toolkit.local_layer(authorization_adapter=allow).invoke(
        AIMessage(
            tool_calls=(
                ToolCall("none-1", "returns_none", {"value": 1}),
                ToolCall("raises-1", "raises", {"value": 2}),
            )
        ),
        toolkit.make_visible_in(Context()),
    )

    assert message.results[0].status == "succeeded"
    assert message.results[0].output is None
    assert message.results[1].status == "failed"
    assert message.results[1].error_code == "RuntimeError"


def test_registry_build_and_registration_are_explicit() -> None:
    toolkit = ToolKit(measure)
    built = toolkit.build_registry()
    target = ExecutorRegistry()

    assert built.resolve("weather.measure", "1.0.0") is not None
    assert toolkit.register_into(target) is target
    assert target.resolve("weather.measure", "1.0.0") is not None
    with pytest.raises(ValueError, match="already registered"):
        toolkit.register_into(target)
    toolkit.register_into(target, replace_existing=True)


def test_make_visible_in_is_immutable_idempotent_and_conflict_safe() -> None:
    toolkit = ToolKit(measure)
    original = Context(system_prompt="stable")
    visible = toolkit.make_visible_in(original)

    assert original.tools == ()
    assert visible.tools == toolkit.definitions
    assert toolkit.make_visible_in(visible) is visible

    conflicting = replace(
        original,
        tools=(
            ToolDefinition(
                name="measure",
                description="different",
                parameters={"type": "object"},
            ),
        ),
    )
    with pytest.raises(ValueError, match="different ToolDefinition"):
        toolkit.make_visible_in(conflicting)


def test_local_layer_without_authorization_remains_fail_closed() -> None:
    toolkit = ToolKit(measure)
    layer = toolkit.local_layer()
    assert layer.authorization is None
    assert layer.authorization_adapter is None


def test_numpy_and_sphinx_parameter_descriptions_are_parsed() -> None:
    @tool(tool_id="numpy", version="1", side_effect=ToolSideEffect.PURE)
    def numpy_style(value: int) -> int:
        """Use NumPy documentation.

        Parameters
        ----------
        value : int
            NumPy parameter description.
        """

        return value

    @tool(tool_id="sphinx", version="1", side_effect=ToolSideEffect.PURE)
    def sphinx_style(value: int) -> int:
        """Use Sphinx documentation.

        :param value: Sphinx parameter description.
        """

        return value

    numpy_schema = ToolKit(numpy_style).definitions[0].parameters
    sphinx_schema = ToolKit(sphinx_style).definitions[0].parameters

    assert numpy_schema["properties"]["value"]["description"] == (
        "NumPy parameter description."
    )
    assert sphinx_schema["properties"]["value"]["description"] == (
        "Sphinx parameter description."
    )


def test_explicit_name_and_description_override_inferred_values() -> None:
    @tool(
        tool_id="override",
        version="1",
        side_effect=ToolSideEffect.PURE,
        name="public_name",
        description="Explicit description.",
    )
    def internal(value: int) -> int:
        """Inferred description.

        Args:
            value: Input value.
        """

        return value

    definition = ToolKit(internal).definitions[0]
    assert definition.name == "public_name"
    assert definition.description == "Explicit description."


def test_definition_errors_fail_early() -> None:
    with pytest.raises(ValueError, match="explicit idempotency"):
        tool(tool_id="write", version="1", side_effect=ToolSideEffect.WRITE)

    @tool(tool_id="missing.annotation", version="1", side_effect=ToolSideEffect.PURE)
    def missing_annotation(value) -> int:
        """Missing an annotation."""

        return int(value)

    with pytest.raises(TypeError, match="type annotation"):
        ToolKit(missing_annotation)

    @tool(tool_id="missing.description", version="1", side_effect=ToolSideEffect.PURE)
    def missing_description(value: int) -> int:
        return value

    with pytest.raises(ValueError, match="requires a description"):
        ToolKit(missing_description)


def test_invalid_signature_and_duplicate_declaration_errors() -> None:
    declaration = tool(
        tool_id="duplicate.decorator",
        version="1",
        side_effect=ToolSideEffect.PURE,
    )

    @declaration
    def decorated(value: int) -> int:
        """Decorated once."""

        return value

    with pytest.raises(ValueError, match="already decorated"):
        declaration(decorated)

    with pytest.raises(TypeError, match="generator"):

        @tool(tool_id="generator", version="1", side_effect=ToolSideEffect.PURE)
        def generator(value: int):
            """Yield values."""

            yield value

    @tool(tool_id="positional", version="1", side_effect=ToolSideEffect.PURE)
    def positional(value: int, /) -> int:
        """Use a positional-only parameter."""

        return value

    with pytest.raises(TypeError, match="positional-only"):
        ToolKit(positional)

    @tool(tool_id="variadic", version="1", side_effect=ToolSideEffect.PURE)
    def variadic(*values: int) -> int:
        """Use variadic parameters."""

        return sum(values)

    with pytest.raises(TypeError, match=r"\*args"):
        ToolKit(variadic)


def test_duplicate_name_and_identity_are_rejected() -> None:
    @tool(tool_id="first", version="1", side_effect=ToolSideEffect.PURE, name="same")
    def first(value: int) -> int:
        """First tool."""

        return value

    @tool(tool_id="second", version="1", side_effect=ToolSideEffect.PURE, name="same")
    def second(value: int) -> int:
        """Second tool."""

        return value

    with pytest.raises(ValueError, match="duplicate model-visible"):
        ToolKit(first, second)

    @tool(tool_id="identity", version="1", side_effect=ToolSideEffect.PURE)
    def identity_one(value: int) -> int:
        """First identity."""

        return value

    @tool(tool_id="identity", version="1", side_effect=ToolSideEffect.PURE)
    def identity_two(value: int) -> int:
        """Second identity."""

        return value

    with pytest.raises(ValueError, match=r"duplicate \(tool_id, version\)"):
        ToolKit(identity_one, identity_two)


def test_execution_plan_hashes_generated_contract_but_excludes_local_objects() -> None:
    class Client:
        def __init__(self, marker: str) -> None:
            self.marker = marker

        @tool(
            tool_id="client.read",
            version="1",
            side_effect=ToolSideEffect.READ,
            resource_key="client-api",
        )
        def read(self, value: int) -> int:
            """Read through a deployment-owned client."""

            return value

    first = ToolKit(Client("first-secret").read)
    second = ToolKit(Client("second-secret").read)
    first_plan = compile_execution_plan(ToolCallLayer(tools=first.specs))
    second_plan = compile_execution_plan(ToolCallLayer(tools=second.specs))
    serialized = str(first_plan.to_dict())

    assert first_plan.graph_hash == second_plan.graph_hash
    assert "first-secret" not in serialized
    assert "Client" not in serialized
    assert "TypeAdapter" not in serialized

    @tool(tool_id="client.read", version="2", side_effect=ToolSideEffect.READ)
    def changed_read(value: int) -> int:
        """Read a changed contract."""

        return value

    changed_plan = compile_execution_plan(
        ToolCallLayer(tools=ToolKit(changed_read).specs)
    )
    assert changed_plan.graph_hash != first_plan.graph_hash


@pytest.mark.asyncio
async def test_managed_runtime_executes_toolkit_registry() -> None:
    @tool(tool_id="managed.double", version="1", side_effect=ToolSideEffect.PURE)
    def double(value: int) -> int:
        """Double a managed value."""

        return value * 2

    toolkit = ToolKit(double)
    runtime = LocalRuntime()
    runtime.attach_executor_registry(toolkit.build_registry())
    bound = runtime.bind(
        ToolCallLayer(tools=toolkit.specs, authorization_adapter=allow)
    )

    message, _ = await bound.invoke(
        AIMessage(tool_calls=(ToolCall("managed-1", "double", {"value": 3}),)),
        toolkit.make_visible_in(Context()),
    )

    assert message.results[0].status == "succeeded"
    assert message.results[0].output == 6
    await runtime.close()
