from __future__ import annotations

import asyncio
import time

import pytest

from pygent.agent import (
    REACT_PROJECTION_OPERATION_KIND,
    AppendToolResultContent,
    PygentAgentContext,
    ReActLayer,
    ReplaceMessageProjection,
    StandaloneUserMessage,
    encode_react_projection_operation,
)
from pygent.core import AIMessage, Context, Message, Module, ToolMessage, UserMessage
from pygent.runtime import (
    CapacityPolicy,
    CapacityScope,
    ContextCodec,
    ExecutionCapacityPolicy,
    ExecutionOptions,
    LocalRuntime,
)
from pygent.tool import ToolCall, ToolResult


class ModelState:
    def __init__(self, answers: tuple[AIMessage, ...]) -> None:
        self.answers = answers
        self.messages: list[Message] = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()


class BlockingFirstModel(Module[Message, AIMessage]):
    trusted_live_resource_attributes = ("state",)

    def __init__(self, answers: tuple[AIMessage, ...]) -> None:
        super().__init__()
        self.state = ModelState(answers)

    async def forward(self, message: Message, context: Context):
        index = len(self.state.messages)
        self.state.messages.append(message)
        if index == 0:
            self.state.entered.set()
            await self.state.release.wait()
        return self.state.answers[index], context


class ToolState:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()


class BlockingTools(Module[AIMessage, ToolMessage]):
    trusted_live_resource_attributes = ("state",)

    def __init__(self) -> None:
        super().__init__()
        self.state = ToolState()

    async def forward(self, message: AIMessage, context: Context):
        self.state.entered.set()
        await self.state.release.wait()
        call = message.tool_calls[0]
        return (
            ToolMessage(
                results=(
                    ToolResult(
                        call_id=call.call_id,
                        name=call.name,
                        status="succeeded",
                        output={"original": True},
                    ),
                )
            ),
            context,
        )


class EmptyTools(Module[AIMessage, ToolMessage]):
    async def forward(self, message: AIMessage, context: Context):
        raise AssertionError("tools should not be called")


class GatedReAct(Module[UserMessage, AIMessage]):
    trusted_live_resource_attributes = ("entered", "release")

    def __init__(self, react: ReActLayer) -> None:
        super().__init__()
        self.react = react
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def forward(self, message: UserMessage, context: Context):
        self.entered.set()
        await self.release.wait()
        return await self.react(message, context)


def bind(runtime: LocalRuntime, react: ReActLayer):
    binding = runtime.create_binding(
        name="react-input-test",
        execution_capacity=ExecutionCapacityPolicy(
            scope=CapacityScope.RUNTIME_INSTANCE,
            max_live_executions=1,
            max_runnable_executions=1,
            max_queue_size=1,
            max_waiters=1,
            max_child_depth=8,
            max_children_per_execution=16,
        ),
        model_capacity=CapacityPolicy.passthrough(),
        tool_capacity=CapacityPolicy.passthrough(),
    )
    return binding.bind(react)


def options() -> ExecutionOptions:
    return ExecutionOptions(deadline=time.monotonic() + 5)


@pytest.mark.asyncio
async def test_standalone_user_message_arriving_during_final_model_is_consumed() -> None:
    model = BlockingFirstModel(
        (AIMessage(content="stale"), AIMessage(content="after steering"))
    )
    runtime = LocalRuntime()
    handle = await bind(runtime, ReActLayer(model=model, tools=EmptyTools())).start(
        UserMessage(content="initial"), Context(), execution=options()
    )
    await model.state.entered.wait()
    delivery = await handle.send_input(
        input_id="steer-1",
        kind=REACT_PROJECTION_OPERATION_KIND,
        value=encode_react_projection_operation(
            StandaloneUserMessage(UserMessage(content="new direction"))
        ),
    )
    assert delivery.status == "accepted"
    model.state.release.set()

    answer, context = await handle.result()
    assert answer.content == "after steering"
    assert [message.content for message in model.state.messages] == [
        "initial",
        "new direction",
    ]
    assert context.projection_revision == 4
    await runtime.close()


@pytest.mark.asyncio
async def test_replace_during_final_model_does_not_duplicate_committed_history() -> None:
    first = AIMessage(content="first answer")
    second = AIMessage(content="after replacement")
    model = BlockingFirstModel((first, second))
    runtime = LocalRuntime(
        context_codecs=(ContextCodec.dataclass(PygentAgentContext),)
    )
    handle = await bind(runtime, ReActLayer(model=model, tools=EmptyTools())).start(
        UserMessage(content="initial"),
        PygentAgentContext(),
        execution=options(),
    )
    await model.state.entered.wait()
    await handle.send_input(
        input_id="replace-final",
        kind=REACT_PROJECTION_OPERATION_KIND,
        value=encode_react_projection_operation(
            ReplaceMessageProjection(
                messages=(UserMessage(content="snapshot"),),
                expected_revision=2,
            )
        ),
    )
    model.state.release.set()

    _, context = await handle.result()
    assert isinstance(context, PygentAgentContext)
    assert [message.content for message in context.full_history] == [
        "initial",
        "first answer",
        "snapshot",
        "after replacement",
    ]
    await runtime.close()


@pytest.mark.asyncio
async def test_replace_preserving_pending_current_commits_it_once() -> None:
    current = UserMessage(content="current request")
    model = BlockingFirstModel((AIMessage(content="after replacement"),))
    model.state.release.set()
    gated = GatedReAct(ReActLayer(model=model, tools=EmptyTools()))
    runtime = LocalRuntime(
        context_codecs=(ContextCodec.dataclass(PygentAgentContext),)
    )
    handle = await bind(runtime, gated).start(
        current,
        PygentAgentContext(
            messages=(
                UserMessage(content="earlier request"),
                AIMessage(content="earlier answer"),
            ),
            full_history=(
                UserMessage(content="earlier request"),
                AIMessage(content="earlier answer"),
            ),
        ),
        execution=options(),
    )
    await gated.entered.wait()
    await handle.send_input(
        input_id="replace-preserving-current",
        kind=REACT_PROJECTION_OPERATION_KIND,
        value=encode_react_projection_operation(
            ReplaceMessageProjection(
                messages=(
                    UserMessage(content="snapshot"),
                    current,
                ),
                expected_revision=1,
            )
        ),
    )
    gated.release.set()

    _, context = await handle.result()
    assert isinstance(context, PygentAgentContext)
    assert [message.content for message in context.full_history] == [
        "earlier request",
        "earlier answer",
        "current request",
        "after replacement",
    ]
    await runtime.close()


@pytest.mark.asyncio
async def test_tool_result_reminder_preserves_original_result() -> None:
    first = AIMessage(
        tool_calls=(ToolCall(call_id="call-1", name="lookup", arguments={}),)
    )
    model = BlockingFirstModel((first, AIMessage(content="done")))
    model.state.release.set()
    tools = BlockingTools()
    runtime = LocalRuntime()
    handle = await bind(runtime, ReActLayer(model=model, tools=tools)).start(
        UserMessage(content="initial"), Context(), execution=options()
    )
    await tools.state.entered.wait()
    await handle.send_input(
        input_id="reminder-1",
        kind=REACT_PROJECTION_OPERATION_KIND,
        value=encode_react_projection_operation(AppendToolResultContent("workspace changed")),
    )
    await handle.send_input(
        input_id="reminder-2",
        kind=REACT_PROJECTION_OPERATION_KIND,
        value=encode_react_projection_operation(AppendToolResultContent("capability changed")),
    )
    tools.state.release.set()

    _, context = await handle.result()
    projected_tool = model.state.messages[1]
    assert isinstance(projected_tool, ToolMessage)
    assert projected_tool.content == "workspace changed\ncapability changed"
    assert projected_tool.results[0].output["original"] is True
    assert context.projection_revision == 6
    await runtime.close()


@pytest.mark.asyncio
async def test_replace_message_projection_rebases_only_complete_appended_messages() -> None:
    first = AIMessage(
        content="working",
        tool_calls=(ToolCall(call_id="call-1", name="lookup", arguments={}),),
    )
    model = BlockingFirstModel((first, AIMessage(content="done")))
    model.state.release.set()
    tools = BlockingTools()
    runtime = LocalRuntime()
    handle = await bind(runtime, ReActLayer(model=model, tools=tools)).start(
        UserMessage(content="initial"), Context(), execution=options()
    )
    await tools.state.entered.wait()
    await handle.send_input(
        input_id="replace-1",
        kind=REACT_PROJECTION_OPERATION_KIND,
        value=encode_react_projection_operation(
            ReplaceMessageProjection(
                messages=(UserMessage(content="snapshot"),),
                expected_revision=1,
                rebase_appended=True,
            )
        ),
    )
    tools.state.release.set()

    await handle.result()
    second_input = model.state.messages[1]
    assert isinstance(second_input, ToolMessage)
    assert second_input.results[0].output["original"] is True
    await runtime.close()


@pytest.mark.asyncio
async def test_non_append_change_rejects_later_rebase_without_failing_execution() -> None:
    first = AIMessage(
        tool_calls=(ToolCall(call_id="call-1", name="lookup", arguments={}),)
    )
    model = BlockingFirstModel((first, AIMessage(content="done")))
    model.state.release.set()
    tools = BlockingTools()
    runtime = LocalRuntime()
    handle = await bind(runtime, ReActLayer(model=model, tools=tools)).start(
        UserMessage(content="initial"), Context(), execution=options()
    )
    await tools.state.entered.wait()
    await handle.send_input(
        input_id="reminder",
        kind=REACT_PROJECTION_OPERATION_KIND,
        value=encode_react_projection_operation(AppendToolResultContent("changed")),
    )
    await handle.send_input(
        input_id="replace",
        kind=REACT_PROJECTION_OPERATION_KIND,
        value=encode_react_projection_operation(
            ReplaceMessageProjection(
                messages=(UserMessage(content="snapshot"),),
                expected_revision=3,
                rebase_appended=True,
            )
        ),
    )
    tools.state.release.set()

    answer, _ = await handle.result()
    assert answer.content == "done"
    async with handle.subscribe() as events:
        rejected = [
            event
            async for event in events
            if event.kind == "react.projection_operation.rejected"
        ]
    assert len(rejected) == 1
    assert rejected[0].data["input_id"] == "replace"
    assert rejected[0].data["reason"] == "revision_conflict"
    await runtime.close()


@pytest.mark.asyncio
async def test_replace_rejects_empty_and_assistant_current() -> None:
    first = AIMessage(
        tool_calls=(ToolCall(call_id="call-1", name="lookup", arguments={}),)
    )
    model = BlockingFirstModel((first, AIMessage(content="done")))
    model.state.release.set()
    tools = BlockingTools()
    runtime = LocalRuntime()
    handle = await bind(runtime, ReActLayer(model=model, tools=tools)).start(
        UserMessage(content="initial"), Context(), execution=options()
    )
    await tools.state.entered.wait()
    for input_id, messages in (
        ("empty", ()),
        ("assistant", (AIMessage(content="not a valid current"),)),
    ):
        await handle.send_input(
            input_id=input_id,
            kind=REACT_PROJECTION_OPERATION_KIND,
            value=encode_react_projection_operation(
                ReplaceMessageProjection(messages=messages, expected_revision=3)
            ),
        )
    tools.state.release.set()
    await handle.result()

    async with handle.subscribe() as events:
        rejected = [
            event
            async for event in events
            if event.kind == "react.projection_operation.rejected"
        ]
    assert [event.data["input_id"] for event in rejected] == ["empty", "assistant"]
    assert {event.data["reason"] for event in rejected} == {"invalid_replacement"}
    await runtime.close()
