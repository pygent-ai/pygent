from __future__ import annotations

import asyncio

import pytest
from jsonschema import SchemaError

from pygent import (
    AIMessage,
    Context,
    thaw_json,
)
from pygent.tool import (
    ExecutorRegistry,
    InMemoryToolTaskManager,
    LocalToolExecutor,
    ToolAuthorizationDecision,
    ToolCall,
    ToolCallLayer,
    ToolDefinition,
    ToolSideEffect,
    ToolSpec,
)


def definition(
    name: str = "convert",
    *,
    output_schema: dict[str, object] | None = None,
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description="convert a bounded value",
        parameters={
            "type": "object",
            "properties": {
                "value": {"type": "integer", "minimum": 1, "maximum": 3},
                "mode": {
                    "type": "string",
                    "enum": ["decimal", "hex"],
                    "default": "decimal",
                },
            },
            "required": ["value"],
            "additionalProperties": False,
        },
        output_schema=output_schema,
    )


def spec(
    name: str = "convert",
    *,
    side_effect: ToolSideEffect = ToolSideEffect.PURE,
    timeout: float | None = None,
    output_schema: dict[str, object] | None = None,
) -> ToolSpec:
    return ToolSpec(
        tool_id=f"test.{name}",
        version="1",
        definition=definition(name, output_schema=output_schema),
        side_effect=side_effect,
        timeout=timeout,
    )


def visible_context(*specs: ToolSpec) -> Context:
    return Context(tools=tuple(item.definition for item in specs))


def allow(request, _context):
    return ToolAuthorizationDecision(
        call_id=request.call.call_id,
        allowed=True,
        reason_code="allowed",
    )


def layer_for(tool: ToolSpec, handler) -> ToolCallLayer:
    registry = ExecutorRegistry()
    registry.register(tool.tool_id, tool.version, LocalToolExecutor(handler))
    return ToolCallLayer(
        tools=(tool,),
        authorization_adapter=allow,
        executor_registry=registry,
    )


def test_tool_definition_freezes_nested_schema_and_rejects_invalid_schema() -> None:
    parameters: dict[str, object] = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
    }
    tool = ToolDefinition("stable", "", parameters)
    properties = parameters["properties"]
    assert isinstance(properties, dict)
    properties["value"] = {"type": "string"}

    frozen = thaw_json(tool.parameters)
    assert frozen["properties"]["value"]["type"] == "integer"

    with pytest.raises(SchemaError):
        ToolDefinition("bad", "", {"type": "not-a-json-schema-type"})


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"value": 0},
        {"value": 4},
        {"value": 1, "mode": "binary"},
        {"value": 1, "unexpected": True},
    ],
)
@pytest.mark.asyncio
async def test_schema_corner_cases_reject_before_authorization_and_execution(
    arguments: dict[str, object],
) -> None:
    tool = spec()
    authorized = 0
    executed = 0

    def authorize(request, context):
        nonlocal authorized
        authorized += 1
        return allow(request, context)

    async def execute(_arguments):
        nonlocal executed
        executed += 1
        return 1

    registry = ExecutorRegistry()
    registry.register(tool.tool_id, tool.version, LocalToolExecutor(execute))
    message, _ = await ToolCallLayer(
        tools=(tool,),
        authorization_adapter=authorize,
        executor_registry=registry,
    ).invoke(
        AIMessage(tool_calls=(ToolCall("bad", "convert", arguments),)),
        visible_context(tool),
    )

    assert message.results[0].status == "rejected"
    assert message.results[0].error_code == "invalid_arguments"
    assert message.results[0].task is None
    assert authorized == executed == 0


@pytest.mark.asyncio
async def test_optional_default_is_schema_metadata_not_an_injected_argument() -> None:
    tool = spec()
    observed = None

    async def capture(arguments):
        nonlocal observed
        observed = thaw_json(arguments)
        return 2

    message, _ = await layer_for(tool, capture).invoke(
        AIMessage(tool_calls=(ToolCall("ok", "convert", {"value": 2}),)),
        visible_context(tool),
    )

    assert message.results[0].status == "succeeded"
    assert observed == {"value": 2}


@pytest.mark.asyncio
async def test_visibility_identity_and_version_fail_before_authorization() -> None:
    tool = spec()
    authorized: list[str] = []

    def authorize(request, context):
        authorized.append(request.call.call_id)
        return allow(request, context)

    layer = ToolCallLayer(tools=(tool,), authorization_adapter=authorize)
    calls = (
        ToolCall("hidden", "convert", {"value": 1}),
        ToolCall("identity", "convert", {"value": 1}, tool_id="somewhere.else"),
        ToolCall("version", "convert", {"value": 1}, tool_version="2"),
    )
    hidden, _ = await layer.invoke(AIMessage(tool_calls=(calls[0],)), Context())
    mismatches, _ = await layer.invoke(
        AIMessage(tool_calls=calls[1:]), visible_context(tool)
    )

    results = (*hidden.results, *mismatches.results)
    assert [item.error_code for item in results] == [
        "tool_not_visible",
        "tool_identity_mismatch",
        "tool_version_mismatch",
    ]
    assert all(item.task is None for item in results)
    assert authorized == []


@pytest.mark.asyncio
async def test_authorization_is_fail_closed_and_call_id_is_correlated() -> None:
    tool = spec()
    call = ToolCall("auth", "convert", {"value": 1})
    no_policy, _ = await ToolCallLayer(tools=(tool,)).invoke(
        AIMessage(tool_calls=(call,)), visible_context(tool)
    )

    def wrong_call_id(_request, _context):
        return ToolAuthorizationDecision(
            call_id="another-call", allowed=True, reason_code="allowed"
        )

    mismatch, _ = await ToolCallLayer(
        tools=(tool,), authorization_adapter=wrong_call_id
    ).invoke(AIMessage(tool_calls=(call,)), visible_context(tool))

    assert no_policy.results[0].error_code == "authorization_not_configured"
    assert mismatch.results[0].error_code == "authorization_call_id_mismatch"
    assert no_policy.results[0].task is mismatch.results[0].task is None


@pytest.mark.parametrize("second_arguments", ({"value": 1}, {"value": 2}))
@pytest.mark.asyncio
async def test_duplicate_call_ids_reject_before_authorization_and_admission(
    second_arguments: dict[str, object],
) -> None:
    tool = spec()
    authorized = 0
    executed = 0

    def authorize(request, context):
        nonlocal authorized
        authorized += 1
        return allow(request, context)

    def execute(_arguments):
        nonlocal executed
        executed += 1
        return 1

    registry = ExecutorRegistry()
    registry.register(tool.tool_id, tool.version, LocalToolExecutor(execute))
    message, _ = await ToolCallLayer(
        tools=(tool,),
        authorization_adapter=authorize,
        executor_registry=registry,
    ).invoke(
        AIMessage(
            tool_calls=(
                ToolCall("duplicate", "convert", {"value": 1}),
                ToolCall("duplicate", "convert", second_arguments),
            )
        ),
        visible_context(tool),
    )

    assert [result.error_code for result in message.results] == [
        "duplicate_call_id",
        "duplicate_call_id",
    ]
    assert all(result.status == "rejected" for result in message.results)
    assert all(result.error_kind == "validation_error" for result in message.results)
    assert all(result.task is None for result in message.results)
    assert authorized == executed == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["raise", "wrong-type"])
async def test_authorization_failure_rejects_without_admitting_task(invalid: str) -> None:
    tool = spec()
    executed = 0

    async def execute(_arguments):
        nonlocal executed
        executed += 1
        return 1

    def authorize(request, context):
        if invalid == "raise":
            raise RuntimeError("private authorization backend detail")
        return object()

    registry = ExecutorRegistry()
    registry.register(tool.tool_id, tool.version, LocalToolExecutor(execute))
    layer = ToolCallLayer(
        tools=(tool,),
        authorization_adapter=authorize,
        executor_registry=registry,
    )
    message, _ = await layer.invoke(
        AIMessage(tool_calls=(ToolCall("auth-failure", "convert", {"value": 1}),)),
        visible_context(tool),
    )

    result = message.results[0]
    assert result.status == "rejected"
    assert result.error_code == "authorization_failed"
    assert result.error is None
    assert executed == 0


@pytest.mark.asyncio
async def test_mixed_batch_is_isolated_and_preserves_model_order() -> None:
    good = spec("good")
    bad = spec("bad", side_effect=ToolSideEffect.EXTERNAL)
    registry = ExecutorRegistry()

    async def slow_success(arguments):
        await asyncio.sleep(0.02)
        return arguments["value"]

    async def opaque_failure(_arguments):
        raise RuntimeError("secret backend details")

    registry.register(good.tool_id, good.version, LocalToolExecutor(slow_success))
    registry.register(bad.tool_id, bad.version, LocalToolExecutor(opaque_failure))
    message, _ = await ToolCallLayer(
        tools=(good, bad),
        authorization_adapter=allow,
        executor_registry=registry,
        max_concurrency=3,
    ).invoke(
        AIMessage(
            tool_calls=(
                ToolCall("first", "good", {"value": 1}),
                ToolCall("second", "missing", {}),
                ToolCall("third", "bad", {"value": 2}),
            )
        ),
        visible_context(good, bad),
    )

    assert [item.call_id for item in message.results] == ["first", "second", "third"]
    assert [item.status for item in message.results] == [
        "succeeded",
        "rejected",
        "unknown",
    ]
    assert message.results[1].task is None
    assert message.results[2].error == "tool executor failed"


@pytest.mark.asyncio
async def test_invalid_executor_output_is_terminal_after_side_effect() -> None:
    tool = spec(output_schema={"type": "integer"})
    message, _ = await layer_for(tool, lambda _arguments: "wrong-type").invoke(
        AIMessage(tool_calls=(ToolCall("output", "convert", {"value": 1}),)),
        visible_context(tool),
    )

    result = message.results[0]
    assert result.status == "failed"
    assert result.error_code == "invalid_output"
    assert result.side_effect_committed is True
    assert result.task is not None and result.task.state.value == "failed"


@pytest.mark.parametrize(
    ("side_effect", "expected_status"),
    [
        (ToolSideEffect.PURE, "failed"),
        (ToolSideEffect.EXTERNAL, "unknown"),
    ],
)
@pytest.mark.asyncio
async def test_timeout_does_not_claim_external_side_effect_was_uncommitted(
    side_effect: ToolSideEffect, expected_status: str
) -> None:
    tool = spec(side_effect=side_effect, timeout=0.01)

    async def slow(_arguments):
        await asyncio.sleep(10)

    message, _ = await layer_for(tool, slow).invoke(
        AIMessage(tool_calls=(ToolCall("timeout", "convert", {"value": 1}),)),
        visible_context(tool),
    )

    result = message.results[0]
    assert result.status == expected_status
    assert result.error_code == "tool_timeout"
    assert result.retryable is True
    assert result.side_effect_committed is None


def test_layer_and_registry_reject_ambiguous_registration() -> None:
    first = spec("first")
    same_name = ToolSpec(
        tool_id="other.identity",
        version="1",
        definition=definition("first"),
    )
    with pytest.raises(ValueError, match="duplicate model-visible names"):
        ToolCallLayer(tools=(first, same_name), authorization_adapter=allow)

    registry = ExecutorRegistry()
    original = LocalToolExecutor(lambda _arguments: 1)
    replacement = LocalToolExecutor(lambda _arguments: 2)
    registry.register(first.tool_id, first.version, original)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(first.tool_id, first.version, replacement)
    registry.register(first.tool_id, first.version, replacement, replace_existing=True)
    assert registry.resolve(first.tool_id, first.version) is replacement
    registry.unregister(first.tool_id, first.version)
    with pytest.raises(LookupError):
        registry.resolve(first.tool_id, first.version)


@pytest.mark.asyncio
async def test_local_executor_accepts_sync_and_async_handlers() -> None:
    tool = spec()
    call = ToolCall("local", "convert", {"value": 2})
    sync_executor = LocalToolExecutor(lambda arguments: arguments["value"])

    async def async_handler(arguments):
        return arguments["value"] + 1

    from pygent.tool import ToolExecutionContext

    assert await sync_executor.execute(tool, call, ToolExecutionContext()) == 2
    assert await LocalToolExecutor(async_handler).execute(
        tool, call, ToolExecutionContext()
    ) == 3


@pytest.mark.asyncio
async def test_task_manager_close_cancels_pending_work_and_is_idempotent() -> None:
    tool = spec()
    registry = ExecutorRegistry()
    started = asyncio.Event()

    async def forever(_arguments):
        started.set()
        await asyncio.Event().wait()

    registry.register(tool.tool_id, tool.version, LocalToolExecutor(forever))
    manager = InMemoryToolTaskManager(registry)
    task = await manager.submit(tool, ToolCall("close", "convert", {"value": 1}))
    await started.wait()
    await manager.close(cancel=True)
    await manager.close(cancel=True)

    result = await manager.get_result(task.task_id)
    assert result is not None and result.status == "cancelled"
    assert await manager.cancel(task.task_id) is False
