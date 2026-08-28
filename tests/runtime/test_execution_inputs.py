from __future__ import annotations

import asyncio
import time

import pytest

from pygent.core import (
    AIMessage,
    Context,
    DirectExecutionError,
    ExecutionInput,
    ExecutionInputDelivery,
    Module,
    UserMessage,
)
from pygent.runtime import (
    CapacityPolicy,
    CapacityScope,
    ExecutionCapacityPolicy,
    ExecutionOptions,
    LocalRuntime,
    SQLiteHistoryStore,
)
from pygent.runtime._execution_inputs import ExecutionInputConsumerError


class InputReceiver(Module[UserMessage, AIMessage]):
    trusted_live_resource_attributes = ("state",)

    class State:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.received = []

    def __init__(self) -> None:
        super().__init__()
        self.state = self.State()

    async def forward(self, message: UserMessage, context: Context):
        self.state.started.set()
        await self.state.release.wait()
        self.state.received.extend(
            await self.receive_execution_inputs(kinds=("test.input",))
        )
        return AIMessage(content="done"), context


def bind(runtime: LocalRuntime, module: Module):
    binding = runtime.create_binding(
        name="input-test",
        execution_capacity=ExecutionCapacityPolicy(
            scope=CapacityScope.RUNTIME_INSTANCE,
            max_live_executions=1,
            max_runnable_executions=1,
            max_queue_size=1,
            max_waiters=1,
            max_child_depth=4,
            max_children_per_execution=4,
        ),
        model_capacity=CapacityPolicy.passthrough(),
        tool_capacity=CapacityPolicy.passthrough(),
    )
    return binding.bind(module)


@pytest.mark.asyncio
@pytest.mark.parametrize("durable", [False, True])
async def test_execution_input_is_ordered_idempotent_and_terminal(tmp_path, durable):
    history = (
        await SQLiteHistoryStore(tmp_path / "inputs.sqlite3").open()
        if durable
        else None
    )
    runtime = LocalRuntime(history=history)
    receiver = InputReceiver()
    handle = await bind(runtime, receiver).start(
        UserMessage(content="go"),
        Context(),
        execution=ExecutionOptions(deadline=time.monotonic() + 5),
    )
    first = await handle.send_input(input_id="one", kind="test.input", value={"n": 1})
    duplicate = await handle.send_input(
        input_id="one", kind="test.input", value={"n": 999}
    )
    second = await handle.send_input(input_id="two", kind="test.input", value={"n": 2})
    assert (first.status, first.sequence) == ("accepted", 0)
    assert (duplicate.status, duplicate.sequence) == ("duplicate", 0)
    assert (second.status, second.sequence) == ("accepted", 1)

    await receiver.state.started.wait()
    receiver.state.release.set()
    await handle.result()
    assert [item.input_id for item in receiver.state.received] == ["one", "two"]
    if history is not None:
        replayed = await history.receive_execution_inputs(
            execution_id=handle.execution_id,
            module_path=handle._record.plan.root,
            receive_index=0,
            kinds=("test.input",),
            limit=16,
            seal_if_empty=False,
        )
        assert replayed == tuple(receiver.state.received)
    finished = await handle.send_input(
        input_id="three", kind="test.input", value={"n": 3}
    )
    assert finished.status == "execution_finished"
    await runtime.close()
    if history is not None:
        await history.close()


@pytest.mark.asyncio
async def test_direct_execution_rejects_send_and_receives_empty() -> None:
    receiver = InputReceiver()
    receiver.state.release.set()
    handle = await receiver.start(UserMessage(content="go"), Context())
    with pytest.raises(DirectExecutionError, match="direct executions"):
        await handle.send_input(input_id="one", kind="test.input", value={})
    await handle.result()
    assert receiver.state.received == []


def test_execution_input_portable_codecs_are_strict_and_frozen() -> None:
    value = {"nested": [1, 2]}
    item = ExecutionInput("input-1", 3, "kind", value)
    value["nested"].append(4)
    assert item.to_dict() == {
        "input_id": "input-1",
        "sequence": 3,
        "kind": "kind",
        "value": {"nested": [1, 2]},
    }
    assert ExecutionInput.from_dict(item.to_dict()) == item
    delivery = ExecutionInputDelivery("accepted", "execution-1", "input-1", 3)
    assert ExecutionInputDelivery.from_dict(delivery.to_dict()) == delivery
    with pytest.raises(ValueError, match="fields are invalid"):
        ExecutionInput.from_dict({**item.to_dict(), "extra": True})


@pytest.mark.asyncio
@pytest.mark.parametrize("durable", [False, True])
async def test_seal_if_empty_closes_the_send_finalization_window(
    tmp_path, durable
) -> None:
    class State:
        def __init__(self) -> None:
            self.sealed = asyncio.Event()
            self.release = asyncio.Event()

    class Sealer(Module[UserMessage, AIMessage]):
        trusted_live_resource_attributes = ("state",)

        def __init__(self) -> None:
            super().__init__()
            self.state = State()

        async def forward(self, message, context):
            assert await self.receive_execution_inputs(
                kinds=("test.input",), seal_if_empty=True
            ) == ()
            self.state.sealed.set()
            await self.state.release.wait()
            return AIMessage(content="done"), context

    history = (
        await SQLiteHistoryStore(tmp_path / "seal.sqlite3").open()
        if durable
        else None
    )
    runtime = LocalRuntime(history=history)
    sealer = Sealer()
    handle = await bind(runtime, sealer).start(
        UserMessage(content="go"), Context(), execution=ExecutionOptions(
            deadline=time.monotonic() + 5
        )
    )
    await sealer.state.sealed.wait()
    delivery = await handle.send_input(
        input_id="too-late", kind="test.input", value={}
    )
    assert delivery.status == "execution_finished"
    sealer.state.release.set()
    await handle.result()
    await runtime.close()
    if history is not None:
        await history.close()


@pytest.mark.asyncio
async def test_one_kind_cannot_be_consumed_by_two_module_paths() -> None:
    class Child(Module[UserMessage, AIMessage]):
        async def forward(self, message, context):
            await self.receive_execution_inputs(kinds=("owned",))
            return AIMessage(content="child"), context

    class Parent(Module[UserMessage, AIMessage]):
        def __init__(self) -> None:
            super().__init__()
            self.child = Child()

        async def forward(self, message, context):
            await self.receive_execution_inputs(kinds=("owned",))
            return await self.child(message, context)

    runtime = LocalRuntime()
    handle = await bind(runtime, Parent()).start(
        UserMessage(content="go"), Context(), execution=ExecutionOptions(
            deadline=time.monotonic() + 5
        )
    )
    with pytest.raises(ExecutionInputConsumerError, match="is owned by"):
        await handle.result()
    await runtime.close()


@pytest.mark.asyncio
async def test_execution_input_enforces_size_and_pending_capacity() -> None:
    receiver = InputReceiver()
    runtime = LocalRuntime()
    handle = await bind(runtime, receiver).start(
        UserMessage(content="go"), Context(), execution=ExecutionOptions(
            deadline=time.monotonic() + 5
        )
    )
    await receiver.state.started.wait()
    with pytest.raises(ValueError, match="64 KiB"):
        await handle.send_input(
            input_id="large", kind="test.input", value="x" * (64 * 1024)
        )
    for index in range(256):
        await handle.send_input(
            input_id=f"input-{index}", kind="test.input", value=index
        )
    with pytest.raises(OverflowError, match="inbox is full"):
        await handle.send_input(
            input_id="overflow", kind="test.input", value=257
        )
    receiver.state.release.set()
    await handle.result()
    await runtime.close()
