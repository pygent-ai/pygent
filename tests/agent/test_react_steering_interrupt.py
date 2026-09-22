"""Immediate steering (interrupt) behavior of the managed ReAct layer."""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace as dataclass_replace

import pytest

from pygent.agent import (
    REACT_PROJECTION_OPERATION_KIND,
    TOOL_BATCH_INTERRUPT_NOTICE,
    ReActBudgetExceeded,
    ReActLayer,
    ReplaceMessageProjection,
    StandaloneUserMessage,
    SteeringMode,
    encode_react_projection_operation,
)
from pygent.agent.react_projection_operations import (
    decode_react_projection_operation,
)
from pygent.core import (
    AIMessage,
    Context,
    ExecutionInput,
    Message,
    Module,
    ToolMessage,
    UserMessage,
)
from pygent.runtime import (
    CapacityPolicy,
    CapacityScope,
    ExecutionCapacityPolicy,
    ExecutionOptions,
    LocalRuntime,
)
from pygent.tool import (
    ExecutorRegistry,
    LocalToolExecutor,
    ToolAuthorizationDecision,
    ToolCall,
    ToolCallLayer,
    ToolDefinition,
    ToolSideEffect,
    ToolSpec,
)


class ModelRunState:
    def __init__(self, answers: tuple[AIMessage, ...]) -> None:
        self.answers = answers
        self.messages: list[Message] = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.aborted = False


class SteeringTestModel(Module[Message, AIMessage]):
    """First call records its input, then blocks until aborted or released."""

    trusted_live_resource_attributes = ("state",)

    def __init__(
        self,
        answers: tuple[AIMessage, ...],
        *,
        first_call_delay: float = 30.0,
        gated: bool = False,
    ) -> None:
        super().__init__()
        self.state = ModelRunState(answers)
        self.first_call_delay = first_call_delay
        self.gated = gated

    async def forward(self, message: Message, context: Context):
        index = len(self.state.messages)
        self.state.messages.append(message)
        if index == 0:
            self.state.entered.set()
            try:
                if self.gated:
                    await self.state.release.wait()
                else:
                    await asyncio.sleep(self.first_call_delay)
            except asyncio.CancelledError:
                self.state.aborted = True
                raise
        return self.state.answers[min(index, len(self.state.answers) - 1)], context


class EmptyTools(Module[AIMessage, ToolMessage]):
    async def forward(self, message: AIMessage, context: Context):
        raise AssertionError("tools should not be called")


def bind(runtime: LocalRuntime, react: ReActLayer):
    binding = runtime.create_binding(
        name="steering-test",
        execution_capacity=ExecutionCapacityPolicy(
            scope=CapacityScope.RUNTIME_INSTANCE,
            max_live_executions=2,
            max_runnable_executions=2,
            max_queue_size=2,
            max_waiters=2,
            max_child_depth=8,
            max_children_per_execution=16,
        ),
        model_capacity=CapacityPolicy.passthrough(),
        tool_capacity=CapacityPolicy.passthrough(),
    )
    return binding.bind(react)


def options() -> ExecutionOptions:
    return ExecutionOptions(deadline=time.monotonic() + 30)


def steer(mode: SteeringMode, content: str = "new direction") -> dict:
    operation = StandaloneUserMessage(UserMessage(content=content), mode=mode)
    return encode_react_projection_operation(operation)


def _allow(request, context) -> ToolAuthorizationDecision:
    return ToolAuthorizationDecision(
        call_id=request.call.call_id,
        allowed=True,
        reason_code="allowed",
        lifecycle="sync",
    )


# ---------------------------------------------------------------------------
# Wire compatibility
# ---------------------------------------------------------------------------


def test_decode_without_mode_defaults_to_wait() -> None:
    encoded = encode_react_projection_operation(
        StandaloneUserMessage(UserMessage(content="hello"))
    )
    legacy = {"type": "standalone_user_message", "message": encoded["message"]}
    operation = decode_react_projection_operation(legacy)
    assert isinstance(operation, StandaloneUserMessage)
    assert operation.mode is SteeringMode.WAIT


def test_encode_decode_round_trips_immediate_mode() -> None:
    encoded = encode_react_projection_operation(
        StandaloneUserMessage(
            UserMessage(content="hello"), mode=SteeringMode.IMMEDIATE
        )
    )
    assert encoded["mode"] == "immediate"
    operation = decode_react_projection_operation(encoded)
    assert isinstance(operation, StandaloneUserMessage)
    assert operation.mode is SteeringMode.IMMEDIATE


def test_decode_rejects_unknown_mode() -> None:
    encoded = encode_react_projection_operation(
        StandaloneUserMessage(UserMessage(content="hello"))
    )
    broken = dict(encoded)
    broken["mode"] = "eventually"
    with pytest.raises(ValueError):
        decode_react_projection_operation(broken)


# ---------------------------------------------------------------------------
# Managed interrupt behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_immediate_message_aborts_model_call_and_becomes_next_input() -> None:
    model = SteeringTestModel(
        (AIMessage(content="stale"), AIMessage(content="final"))
    )
    runtime = LocalRuntime()
    handle = await bind(runtime, ReActLayer(model=model, tools=EmptyTools())).start(
        UserMessage(content="initial"), Context(), execution=options()
    )
    await model.state.entered.wait()
    delivery = await handle.send_input(
        input_id="steer-1",
        kind=REACT_PROJECTION_OPERATION_KIND,
        value=steer(SteeringMode.IMMEDIATE),
    )
    assert delivery.status == "accepted"

    collected: list[object] = []

    async def observe() -> None:
        async with handle.subscribe(after=None) as events:
            async for event in events:
                collected.append(event)

    (answer, _), _ = await asyncio.gather(handle.result(), observe())
    assert answer.content == "final"
    assert model.state.aborted is True
    assert [message.content for message in model.state.messages] == [
        "initial",
        "new direction",
    ]
    interrupted = [
        event
        for event in collected
        if getattr(event, "kind", None) == "react.interrupted"
    ]
    assert len(interrupted) == 1
    assert dict(getattr(interrupted[0], "data", {}))["point"] == "model"
    await runtime.close()


@pytest.mark.asyncio
async def test_immediate_message_voids_returned_tool_calls() -> None:
    model = SteeringTestModel(
        (
            AIMessage(
                content="",
                tool_calls=(
                    ToolCall(call_id="call_1", name="noop", arguments={"value": 1}),
                ),
            ),
            AIMessage(content="final"),
        ),
        gated=True,
    )
    runtime = LocalRuntime()
    handle = await bind(runtime, ReActLayer(model=model, tools=EmptyTools())).start(
        UserMessage(content="initial"), Context(), execution=options()
    )
    await model.state.entered.wait()
    delivery = await handle.send_input(
        input_id="steer-1",
        kind=REACT_PROJECTION_OPERATION_KIND,
        value=steer(SteeringMode.IMMEDIATE),
    )
    assert delivery.status == "accepted"
    # Let the watcher absorb the steering before the model call returns so
    # the void happens at the model-return boundary (or as an abort; both
    # must discard the tool calls).
    await asyncio.sleep(0.2)
    model.state.release.set()
    answer, _ = await handle.result()
    assert answer.content == "final"
    assert [message.content for message in model.state.messages] == [
        "initial",
        "new direction",
    ]
    await runtime.close()


@pytest.mark.asyncio
async def test_immediate_message_interrupts_tool_batch_and_keeps_finished_results() -> None:
    tool = ToolSpec(
        tool_id="math.double",
        version="1",
        definition=ToolDefinition(
            name="double",
            description="double an integer",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            output_schema={"type": "integer"},
        ),
        side_effect=ToolSideEffect.PURE,
    )
    registry = ExecutorRegistry()
    slow_entered = asyncio.Event()

    async def double(arguments):
        if arguments["value"] == 1:
            return 2
        slow_entered.set()
        await asyncio.sleep(30)
        return 4

    registry.register(tool.tool_id, tool.version, LocalToolExecutor(double))
    tools = ToolCallLayer(
        tools=(tool,),
        authorization=None,
        executor_registry=registry,
        authorization_adapter=_allow,
        max_concurrency=2,
    )
    model = SteeringTestModel(
        (
            AIMessage(
                content="",
                tool_calls=(
                    ToolCall(call_id="fast", name="double", arguments={"value": 1}),
                    ToolCall(call_id="slow", name="double", arguments={"value": 2}),
                ),
            ),
            AIMessage(content="final"),
        ),
        first_call_delay=0.0,
    )
    runtime = LocalRuntime()
    handle = await bind(runtime, ReActLayer(model=model, tools=tools)).start(
        UserMessage(content="initial"),
        Context(
            tools=(tool.definition,),
            metadata={"permissions": ["tool:use"]},
        ),
        execution=options(),
    )
    await slow_entered.wait()
    delivery = await handle.send_input(
        input_id="steer-1",
        kind=REACT_PROJECTION_OPERATION_KIND,
        value=steer(SteeringMode.IMMEDIATE),
    )
    assert delivery.status == "accepted"
    answer, context = await handle.result()
    assert answer.content == "final"
    assert model.state.messages[1].content == "new direction"

    tool_messages = [
        message for message in context.messages if isinstance(message, ToolMessage)
    ]
    assert len(tool_messages) == 1
    results = {result.call_id: result for result in tool_messages[0].results}
    assert results["fast"].status == "succeeded"
    assert results["fast"].output == 2
    assert results["slow"].status == "cancelled"
    assert results["slow"].error_code == "interrupted_by_steering"
    assert TOOL_BATCH_INTERRUPT_NOTICE in tool_messages[0].content
    await runtime.close()


@pytest.mark.asyncio
async def test_multiple_immediate_messages_apply_in_sequence_last_wins() -> None:
    model = SteeringTestModel(
        (AIMessage(content="stale"), AIMessage(content="final"))
    )
    runtime = LocalRuntime()
    handle = await bind(runtime, ReActLayer(model=model, tools=EmptyTools())).start(
        UserMessage(content="initial"), Context(), execution=options()
    )
    await model.state.entered.wait()
    await handle.send_input(
        input_id="steer-1",
        kind=REACT_PROJECTION_OPERATION_KIND,
        value=steer(SteeringMode.IMMEDIATE, "first direction"),
    )
    await handle.send_input(
        input_id="steer-2",
        kind=REACT_PROJECTION_OPERATION_KIND,
        value=steer(SteeringMode.IMMEDIATE, "second direction"),
    )
    answer, context = await handle.result()
    assert answer.content == "final"
    # Both messages apply in input sequence; only the last one becomes the
    # next model input, the first is committed to history.
    assert [message.content for message in model.state.messages] == [
        "initial",
        "second direction",
    ]
    committed = [message.content for message in context.messages]
    assert "first direction" in committed
    assert "second direction" in committed
    await runtime.close()

@pytest.mark.asyncio
async def test_interrupted_model_call_consumes_model_call_budget() -> None:
    model = SteeringTestModel(
        (AIMessage(content="stale"), AIMessage(content="final"))
    )
    runtime = LocalRuntime()
    handle = await bind(
        runtime,
        ReActLayer(model=model, tools=EmptyTools(), max_model_calls=1),
    ).start(UserMessage(content="initial"), Context(), execution=options())
    await model.state.entered.wait()
    await handle.send_input(
        input_id="steer-1",
        kind=REACT_PROJECTION_OPERATION_KIND,
        value=steer(SteeringMode.IMMEDIATE),
    )
    # The aborted call consumes the only model-call slot; the next admission
    # must fail instead of silently granting a free retry.
    with pytest.raises(ReActBudgetExceeded):
        await handle.result()
    await runtime.close()


class GatedModelWithReplacement(Module[Message, AIMessage]):
    """First call waits for release, then returns tool calls plus a replaced projection."""

    trusted_live_resource_attributes = ("state",)

    def __init__(self, answers: tuple[AIMessage, ...]) -> None:
        super().__init__()
        self.state = ModelRunState(answers)

    async def forward(self, message: Message, context: Context):
        index = len(self.state.messages)
        self.state.messages.append(message)
        if index == 0:
            self.state.entered.set()
            await self.state.release.wait()
            replaced = dataclass_replace(
                context,
                messages=(UserMessage(content="compressed"),),
                projection_revision=context.projection_revision + 1,
            )
            return self.state.answers[0], replaced
        return self.state.answers[min(index, len(self.state.answers) - 1)], context


@pytest.mark.asyncio
async def test_voided_tool_calls_keep_model_projection_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The model_returned void resolves a same-tick race (the model call
    # completes as the interrupt fires) and cannot be timed deterministically
    # from outside. Pin the race in favor of "completed" by stubbing the
    # interrupt signal while the watcher still fills the pending queue.
    import pygent.agent.react as react_module

    class CompletedRaceState(react_module._SteeringState):
        def absorb(self, item, operation):
            self.pending.append((item, operation))

    monkeypatch.setattr(react_module, "_SteeringState", CompletedRaceState)
    model = GatedModelWithReplacement(
        (
            AIMessage(
                content="",
                tool_calls=(
                    ToolCall(call_id="call_1", name="noop", arguments={"value": 1}),
                ),
            ),
            AIMessage(content="final"),
        ),
    )
    runtime = LocalRuntime()
    handle = await bind(runtime, ReActLayer(model=model, tools=EmptyTools())).start(
        UserMessage(content="initial"), Context(), execution=options()
    )
    await model.state.entered.wait()
    delivery = await handle.send_input(
        input_id="steer-1",
        kind=REACT_PROJECTION_OPERATION_KIND,
        value=steer(SteeringMode.IMMEDIATE),
    )
    assert delivery.status == "accepted"
    await asyncio.sleep(0.2)  # let the watcher absorb into pending
    model.state.release.set()

    collected: list[object] = []

    async def observe() -> None:
        async with handle.subscribe(after=None) as events:
            async for event in events:
                collected.append(event)

    (answer, context), _ = await asyncio.gather(handle.result(), observe())
    assert answer.content == "final"
    # The voided answer's tool calls never ran, but the model module's
    # explicit projection replacement (compression) is accepted.
    assert [message.content for message in model.state.messages] == [
        "initial",
        "new direction",
    ]
    committed = [message.content for message in context.messages]
    assert "compressed" in committed
    assert "new direction" in committed
    points = [
        dict(getattr(event, "data", {})).get("point")
        for event in collected
        if getattr(event, "kind", None) == "react.interrupted"
    ]
    assert points == ["model_returned"]
    monkeypatch.undo()
    await runtime.close()


# ---------------------------------------------------------------------------
# Signal ownership, race and ordering regressions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_immediate_absorbed_during_drain_keeps_interrupt_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: an immediate absorbed while the loop-top drain runs must
    # not lose its interrupt signal to the drain-following event.clear().
    import pygent.agent.react as react_module

    model = SteeringTestModel(
        (AIMessage(content="stale"), AIMessage(content="final")), gated=True
    )
    react = ReActLayer(model=model, tools=EmptyTools())
    operation = decode_react_projection_operation(steer(SteeringMode.IMMEDIATE))
    raced = {"done": False}

    class RaceState(react_module._SteeringState):
        def take_pending(self):
            taken = super().take_pending()
            if not raced["done"]:
                raced["done"] = True
                # Simulate a new immediate arriving while the drain runs:
                # the watcher consumed it and set the interrupt signal.
                self.absorb(
                    ExecutionInput(
                        "steer-race", 99, REACT_PROJECTION_OPERATION_KIND, {}
                    ),
                    operation,
                )
            return taken

    monkeypatch.setattr(react_module, "_SteeringState", RaceState)
    runtime = LocalRuntime()
    handle = await bind(runtime, react).start(
        UserMessage(content="initial"), Context(), execution=options()
    )
    await model.state.entered.wait()
    answer, _ = await asyncio.wait_for(handle.result(), timeout=10)
    assert model.state.aborted is True
    assert answer.content == "final"
    assert [message.content for message in model.state.messages] == [
        "initial",
        "new direction",
    ]
    monkeypatch.undo()
    await runtime.close()


@pytest.mark.asyncio
async def test_execution_cancellation_stops_run_and_watcher() -> None:
    # The watcher is auxiliary control flow and must never be the reason an
    # execution cannot finish, including on explicit cancellation.
    model = SteeringTestModel((AIMessage(content="final"),), gated=True)
    runtime = LocalRuntime()
    handle = await bind(runtime, ReActLayer(model=model, tools=EmptyTools())).start(
        UserMessage(content="initial"), Context(), execution=options()
    )
    await model.state.entered.wait()
    assert await handle.cancel() is True
    raised = None
    try:
        await asyncio.wait_for(handle.result(), timeout=10)
    except BaseException as exc:  # noqa: BLE001 - cancellation surfaces here
        raised = exc
    assert raised is not None
    await runtime.close()


@pytest.mark.asyncio
async def test_mixed_wait_and_immediate_sequence_applies_in_input_order() -> None:
    # The event only wakes the interrupt; ordering always comes from the
    # ExecutionInput sequence.
    model = SteeringTestModel(
        (AIMessage(content="stale"), AIMessage(content="final")), gated=True
    )
    runtime = LocalRuntime()
    handle = await bind(runtime, ReActLayer(model=model, tools=EmptyTools())).start(
        UserMessage(content="initial"), Context(), execution=options()
    )
    await model.state.entered.wait()
    await handle.send_input(
        input_id="s1", kind=REACT_PROJECTION_OPERATION_KIND,
        value=steer(SteeringMode.WAIT, "wait #1"),
    )
    await handle.send_input(
        input_id="s2", kind=REACT_PROJECTION_OPERATION_KIND,
        value=steer(SteeringMode.IMMEDIATE, "immediate #2"),
    )
    await handle.send_input(
        input_id="s3", kind=REACT_PROJECTION_OPERATION_KIND,
        value=steer(SteeringMode.WAIT, "wait #3"),
    )
    await asyncio.sleep(0.2)
    answer, context = await asyncio.wait_for(handle.result(), timeout=10)
    assert answer.content == "final"
    # Batch semantics: every message is applied in input sequence, the last
    # one becomes the next model input, earlier ones are committed history.
    assert [message.content for message in model.state.messages] == [
        "initial",
        "wait #3",
    ]
    assert [message.content for message in context.messages] == [
        "initial",
        "wait #1",
        "immediate #2",
        "wait #3",
        "final",
    ]
    await runtime.close()


@pytest.mark.asyncio
async def test_replace_and_immediate_apply_in_sequence_order() -> None:
    model = SteeringTestModel(
        (AIMessage(content="stale"), AIMessage(content="final")), gated=True
    )
    runtime = LocalRuntime()
    handle = await bind(runtime, ReActLayer(model=model, tools=EmptyTools())).start(
        UserMessage(content="initial"), Context(), execution=options()
    )
    await model.state.entered.wait()
    replacement = encode_react_projection_operation(
        ReplaceMessageProjection(
            messages=(UserMessage(content="replaced"),), expected_revision=1
        )
    )
    await handle.send_input(input_id="s1", kind=REACT_PROJECTION_OPERATION_KIND, value=replacement)
    await handle.send_input(
        input_id="s2", kind=REACT_PROJECTION_OPERATION_KIND,
        value=steer(SteeringMode.IMMEDIATE, "steered"),
    )
    await asyncio.sleep(0.2)
    answer, context = await asyncio.wait_for(handle.result(), timeout=10)
    assert answer.content == "final"
    assert [message.content for message in model.state.messages] == [
        "initial",
        "steered",
    ]
    committed = [message.content for message in context.messages]
    assert "replaced" in committed
    assert "steered" in committed
    await runtime.close()
