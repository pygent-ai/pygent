from __future__ import annotations

import asyncio
import time
from typing import cast

import pytest

from pygent import (
    AIMessage,
    CapacityPolicy,
    CapacityScope,
    Context,
    ExecutionAdmissionError,
    ExecutionCapacityPolicy,
    ExecutionOptions,
    ExecutionStatus,
    ExternalWaitRejected,
    LocalRuntime,
    Module,
    ToolAuthorizationDecision,
    ToolCall,
    ToolCallLayer,
    ToolDefinition,
    ToolSpec,
    UserMessage,
)
from pygent.core import FrozenJsonObject
from pygent.tool import ExecutorRegistry, LocalToolExecutor


def _binding(
    runtime: LocalRuntime,
    *,
    name: str,
    live: int = 20,
    runnable: int = 5,
    queue: int = 20,
    waiters: int = 20,
    child_depth: int = 8,
    children: int = 64,
    tool_capacity: CapacityPolicy | None = None,
):
    return runtime.create_binding(
        name=name,
        execution_capacity=ExecutionCapacityPolicy(
            scope=CapacityScope.RUNTIME_INSTANCE,
            max_live_executions=live,
            max_runnable_executions=runnable,
            max_queue_size=queue,
            max_waiters=waiters,
            max_child_depth=child_depth,
            max_children_per_execution=children,
        ),
        model_capacity=CapacityPolicy.passthrough(),
        tool_capacity=tool_capacity or CapacityPolicy.passthrough(),
    )


class _Echo(Module[UserMessage, AIMessage]):
    async def forward(
        self, message: UserMessage, context: Context
    ) -> tuple[AIMessage, Context]:
        output = AIMessage(content=message.content.upper())
        return output, context + message + output


@pytest.mark.asyncio
async def test_same_bound_module_twenty_executions_isolate_context_events_and_cursors():
    class Probe(Module[UserMessage, AIMessage]):
        async def forward(self, message, context):
            request_marker = cast(FrozenJsonObject, context.metadata)["marker"]
            await self.emit(
                kind="probe.observed",
                data={"marker": request_marker, "content": message.content},
            )
            await asyncio.sleep(0)
            output = AIMessage(content=f"answer:{message.content}")
            return output, context + message + output

    runtime = LocalRuntime()
    bound = _binding(runtime, name="twenty-isolated").bind(Probe())
    handles = [
        await bound.start(
            UserMessage(content=f"request-{index}"),
            Context(metadata={"marker": f"marker-{index}"}),
            execution=ExecutionOptions(request_id=f"transport-{index}"),
        )
        for index in range(20)
    ]

    results = await asyncio.gather(*(handle.result() for handle in handles))
    assert len({handle.execution_id for handle in handles}) == 20

    for index, (handle, (output, context)) in enumerate(zip(handles, results)):
        expected_content = f"request-{index}"
        expected_marker = f"marker-{index}"
        assert output.content == f"answer:{expected_content}"
        assert cast(FrozenJsonObject, context.metadata)["marker"] == expected_marker
        assert [message.content for message in context.messages] == [
            expected_content,
            f"answer:{expected_content}",
        ]

        async with handle.subscribe() as subscription:
            events = [event async for event in subscription]
        assert [event.sequence for event in events] == list(range(len(events)))
        assert {event.execution_id for event in events} == {handle.execution_id}
        assert cast(FrozenJsonObject, events[0].data)["request_id"] == (
            f"transport-{index}"
        )
        observed = next(event for event in events if event.kind == "probe.observed")
        assert cast(FrozenJsonObject, observed.data).to_dict() == {
            "marker": expected_marker,
            "content": expected_content,
        }

        async with handle.subscribe(after=0) as cursor_subscription:
            after_zero = [event async for event in cursor_subscription]
        assert [event.sequence for event in after_zero] == list(range(1, len(events)))

    await runtime.close()


@pytest.mark.asyncio
async def test_managed_stream_early_exit_cancels_and_releases_live_capacity():
    entered = asyncio.Event()
    cleaned = asyncio.Event()

    class StreamingBlock(Module[UserMessage, AIMessage]):
        async def forward(self, message, context):
            try:
                entered.set()
                await self.emit(kind="stream.ready", data={})
                await asyncio.Event().wait()
            finally:
                cleaned.set()
            return AIMessage(content="unreachable"), context

    runtime = LocalRuntime()
    binding = _binding(
        runtime,
        name="stream-cancel",
        live=1,
        runnable=1,
        queue=0,
    )
    blocked = binding.bind(StreamingBlock())
    echo = binding.bind(_Echo())

    async with blocked.stream(UserMessage(content="stop"), Context()) as stream:
        async for event in stream:
            if event.kind == "stream.ready":
                break

    assert entered.is_set()
    await asyncio.wait_for(cleaned.wait(), timeout=1)
    output, _ = await echo.invoke(UserMessage(content="next"), Context())
    assert output.content == "NEXT"
    await runtime.close()


@pytest.mark.asyncio
async def test_child_depth_and_total_fanout_limits_are_hard_boundaries():
    class Chain(Module[UserMessage, AIMessage]):
        def __init__(self, remaining: int) -> None:
            super().__init__()
            self.child = Chain(remaining - 1) if remaining else None

        async def forward(self, message, context):
            if self.child is None:
                return AIMessage(content="leaf"), context
            return await self.child(message, context)

    class Fanout(Module[UserMessage, AIMessage]):
        def __init__(self) -> None:
            super().__init__()
            self.child = _Echo()

        async def forward(self, message, context):
            for _ in range(3):
                await self.child(message, context)
            return AIMessage(content="done"), context

    runtime = LocalRuntime()
    depth_bound = _binding(
        runtime, name="depth-limit", child_depth=2, children=20
    ).bind(Chain(3))
    fanout_bound = _binding(
        runtime, name="fanout-limit", child_depth=8, children=2
    ).bind(Fanout())

    with pytest.raises(ExecutionAdmissionError, match="Child depth"):
        await depth_bound.invoke(UserMessage(content="deep"), Context())
    with pytest.raises(ExecutionAdmissionError, match="Child fan-out"):
        await fanout_bound.invoke(UserMessage(content="wide"), Context())
    await runtime.close()


@pytest.mark.asyncio
async def test_external_waiter_limit_rejects_then_recovers_after_cancel():
    class Waiting(Module[UserMessage, AIMessage]):
        async def forward(self, message, context):
            value = await self.wait_external(
                kind="approval",
                key=message.content,
                request={"key": message.content},
                timeout=2,
            )
            return AIMessage(content=cast(str, value["decision"])), context

    runtime = LocalRuntime()
    waiting = _binding(
        runtime,
        name="waiter-limit",
        live=2,
        runnable=2,
        waiters=1,
    ).bind(Waiting())
    first = await waiting.start(
        UserMessage(content="first"),
        Context(),
        execution=ExecutionOptions(deadline=time.monotonic() + 3),
    )
    while first.status is not ExecutionStatus.WAITING_EXTERNAL:
        await asyncio.sleep(0)

    with pytest.raises(ExternalWaitRejected, match="capacity is full"):
        await waiting.invoke(
            UserMessage(content="second"),
            Context(),
            execution=ExecutionOptions(deadline=time.monotonic() + 3),
        )

    assert await first.cancel()
    third = await waiting.start(
        UserMessage(content="third"),
        Context(),
        execution=ExecutionOptions(deadline=time.monotonic() + 3),
    )
    while third.status is not ExecutionStatus.WAITING_EXTERNAL:
        await asyncio.sleep(0)
    await runtime.deliver_external(
        kind="approval", key="third", value={"decision": "approved"}
    )
    assert (await third.result())[0].content == "approved"
    await runtime.close()


@pytest.mark.asyncio
async def test_tool_permit_handoff_and_cancelled_queue_entry_are_cleaned_up():
    first_entered = asyncio.Event()
    first_release = asyncio.Event()
    executed: list[int] = []

    async def handler(arguments: FrozenJsonObject) -> object:
        value = cast(int, arguments["value"])
        executed.append(value)
        if value == 1:
            first_entered.set()
            await first_release.wait()
        return {"value": value}

    definition = ToolDefinition(
        name="boundary_tool",
        description="Exercise managed Tool capacity.",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    )
    spec = ToolSpec(
        tool_id="boundary.tool",
        version="1.0.0",
        definition=definition,
        resource_key="boundary-resource",
    )
    registry = ExecutorRegistry()
    registry.register("boundary.tool", "1.0.0", LocalToolExecutor(handler))
    layer = ToolCallLayer(
        tools=(spec,),
        authorization_adapter=lambda request, context: ToolAuthorizationDecision(
            call_id=request.call.call_id,
            allowed=True,
            reason_code="allowed",
        ),
    )

    def tool_message(value: int) -> AIMessage:
        return AIMessage(
            tool_calls=(
                ToolCall(
                    call_id=f"call-{value}",
                    name="boundary_tool",
                    arguments={"value": value},
                ),
            )
        )

    runtime = LocalRuntime()
    runtime.attach_executor_registry(registry)
    binding = _binding(
        runtime,
        name="tool-handoff",
        live=4,
        runnable=1,
        queue=4,
        tool_capacity=CapacityPolicy.limited(
            max_concurrency=1, max_queue_size=1
        ),
    )
    tools = binding.bind(layer)
    echo = binding.bind(_Echo())
    context = Context(tools=(definition,))

    first = await tools.start(tool_message(1), context)
    await first_entered.wait()
    queued = await tools.start(tool_message(2), context)

    # Both Tool Runs release their runnable lease while using/waiting for the
    # Tool permit, so an unrelated execution can complete with runnable=1.
    assert (await echo.invoke(UserMessage(content="free"), Context()))[0].content == (
        "FREE"
    )
    assert executed == [1]
    assert await queued.cancel()
    assert queued.status is ExecutionStatus.CANCELLED

    # max_queue_size=1: this third call is admitted only if cancellation
    # removed the second call's resource waiter.
    third = await tools.start(tool_message(3), context)
    await asyncio.sleep(0)
    first_release.set()
    assert (await first.result())[0].results[0].status == "succeeded"
    assert (await third.result())[0].results[0].status == "succeeded"
    assert executed == [1, 3]
    await runtime.close()
