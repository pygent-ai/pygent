from __future__ import annotations

import asyncio

import pytest

from pygent.core import Module, RemoteModule
from pygent.core._module_contracts import _execution_scope
from pygent.core.values import AIMessage, Context, UserMessage
from pygent.runtime import LocalRuntime


class Child(Module[UserMessage, AIMessage]):
    async def forward(
        self, message: UserMessage, context: Context
    ) -> tuple[AIMessage, Context]:
        await self.emit(kind="child.delta", data={"text": message.content})
        output = AIMessage(content=f"child:{message.content}")
        return output, context + message + output


class Parent(Module[UserMessage, AIMessage]):
    def __init__(self, child: Child) -> None:
        super().__init__()
        self.child = child

    async def forward(
        self, message: UserMessage, context: Context
    ) -> tuple[AIMessage, Context]:
        await self.emit(kind="parent.start", data={"order": 0})
        output, context = await self.child(message, context)
        await self.emit(kind="parent.end", data={"order": 2})
        return output, context


@pytest.mark.asyncio
async def test_direct_invoke_executes_children_and_restores_scope() -> None:
    parent = Parent(Child())

    output, context = await parent.invoke(UserMessage(content="hello"), Context())

    assert output == AIMessage(content="child:hello")
    assert context.messages == (UserMessage(content="hello"), output)
    assert _execution_scope.get() is None


@pytest.mark.asyncio
async def test_direct_stream_observes_strict_order_and_same_final_value() -> None:
    parent = Parent(Child())
    input_message = UserMessage(content="hello")

    expected = await parent.invoke(input_message, Context())
    async with parent.stream(input_message, Context()) as stream:
        events = [event async for event in stream]
        actual = await stream.final_result()

    assert actual == expected
    assert [event.sequence for event in events] == list(range(len(events)))
    assert [event.kind for event in events] == [
        "execution.started",
        "span.started",
        "parent.start",
        "span.started",
        "child.delta",
        "span.completed",
        "parent.end",
        "span.completed",
        "execution.completed",
    ]
    assert events[0].trace_id == events[-1].trace_id
    assert len({event.event_id for event in events}) == len(events)


@pytest.mark.asyncio
async def test_direct_stream_final_result_can_drain_unobserved_events() -> None:
    class ManyEvents(Module[UserMessage, AIMessage]):
        async def forward(self, message: UserMessage, context: Context):
            for index in range(100):
                await self.emit(kind="delta", data={"index": index})
            return AIMessage(content="done"), context

    async with ManyEvents().stream(UserMessage(), Context()) as stream:
        output, context = await stream.final_result()

    assert output == AIMessage(content="done")
    assert context == Context()


@pytest.mark.asyncio
async def test_direct_start_has_one_owner_and_independent_subscribers() -> None:
    calls = 0

    class Observable(Module[UserMessage, AIMessage]):
        async def forward(self, message: UserMessage, context: Context):
            nonlocal calls
            calls += 1
            await self.emit(kind="delta", data={"text": message.content})
            return AIMessage(content=message.content), context

    handle = await Observable().start(UserMessage(content="once"), Context())

    async def collect(after: int | None = None):
        async with handle.subscribe(after=after) as subscription:
            return [event async for event in subscription]

    result, first, second = await asyncio.gather(
        handle.result(), collect(), collect(after=0)
    )

    assert calls == 1
    assert result == (AIMessage(content="once"), Context())
    assert [event.sequence for event in first] == list(range(len(first)))
    assert [event.sequence for event in second] == list(range(1, len(first)))
    assert [event.event_id for event in second] == [
        event.event_id for event in first[1:]
    ]


@pytest.mark.asyncio
async def test_direct_execution_without_subscribers_skips_condition_notifications() -> None:
    class CountingCondition(asyncio.Condition):
        def __init__(self) -> None:
            super().__init__()
            self.acquisitions = 0
            self.notifications = 0

        async def acquire(self) -> bool:
            self.acquisitions += 1
            return await super().acquire()

        def notify_all(self) -> None:
            self.notifications += 1
            super().notify_all()

    class ManyEvents(Module[UserMessage, AIMessage]):
        async def forward(self, message: UserMessage, context: Context):
            for index in range(100):
                await self.emit(kind="delta", data={"index": index})
            return AIMessage(content="done"), context

    handle = await ManyEvents().start(UserMessage(), Context())
    condition = CountingCondition()
    handle._record.condition = condition

    assert await handle.result() == (AIMessage(content="done"), Context())
    assert handle._record.active_subscribers == 0
    assert condition.acquisitions == 0
    assert condition.notifications == 0


@pytest.mark.asyncio
async def test_direct_subscription_releases_notification_interest_on_exit() -> None:
    release = asyncio.Event()

    class Waits(Module[UserMessage, AIMessage]):
        async def forward(self, message: UserMessage, context: Context):
            await self.emit(kind="ready", data={})
            await release.wait()
            return AIMessage(content="done"), context

    handle = await Waits().start(UserMessage(), Context())
    async with handle.subscribe() as subscription:
        assert handle._record.active_subscribers == 1
        iterator = subscription.__aiter__()
        event = await anext(iterator)
        assert event.kind == "execution.started"
        await iterator.aclose()
        assert handle._record.active_subscribers == 0

    assert handle._record.active_subscribers == 0
    release.set()
    assert await handle.result() == (AIMessage(content="done"), Context())


@pytest.mark.asyncio
async def test_cancelled_direct_subscription_releases_notification_interest() -> None:
    release = asyncio.Event()

    class Waits(Module[UserMessage, AIMessage]):
        async def forward(self, message: UserMessage, context: Context):
            await release.wait()
            return AIMessage(content="done"), context

    handle = await Waits().start(UserMessage(), Context())
    while len(handle._record.events) < 2:
        await asyncio.sleep(0)
    subscription = handle.subscribe(after=handle._record.events[-1].sequence)
    iterator = subscription.__aiter__()
    pending = asyncio.create_task(anext(iterator))
    await asyncio.sleep(0)
    assert handle._record.active_subscribers == 1

    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert handle._record.active_subscribers == 0

    release.set()
    assert await handle.result() == (AIMessage(content="done"), Context())


@pytest.mark.asyncio
async def test_early_stream_exit_cancels_and_cleans_up_execution() -> None:
    cancelled = asyncio.Event()

    class Running(Module[UserMessage, AIMessage]):
        async def forward(self, message: UserMessage, context: Context):
            try:
                await self.emit(kind="started", data={})
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    async with Running().stream(UserMessage(), Context()) as stream:
        async for event in stream:
            if event.kind == "started":
                break

    assert cancelled.is_set()
    assert _execution_scope.get() is None


@pytest.mark.asyncio
async def test_shared_module_has_isolated_concurrent_direct_scopes() -> None:
    class Echo(Module[UserMessage, AIMessage]):
        async def forward(self, message: UserMessage, context: Context):
            await asyncio.sleep(0)
            output = AIMessage(content=message.content)
            return output, context + message + output

    shared = Echo()
    results = await asyncio.gather(
        *(shared.invoke(UserMessage(content=str(index)), Context()) for index in range(20))
    )

    assert [output.content for output, _ in results] == [str(index) for index in range(20)]
    for index, (output, context) in enumerate(results):
        assert context.messages == (UserMessage(content=str(index)), output)


@pytest.mark.asyncio
async def test_root_and_remote_calls_fail_with_actionable_scope_errors() -> None:
    child = Child()
    with pytest.raises(RuntimeError, match=r"invoke\(\) or module\.stream\(\)"):
        await child(UserMessage(), Context())

    remote = RemoteModule[UserMessage, AIMessage](binding_ref="reviewer-service")
    with pytest.raises(RuntimeError, match="managed execution"):
        await remote(UserMessage(), Context())


@pytest.mark.asyncio
async def test_invalid_forward_result_is_rejected_at_execution_boundary() -> None:
    class Invalid(Module[UserMessage, AIMessage]):
        async def forward(self, message: UserMessage, context: Context):
            return "not-a-message", context

    with pytest.raises(TypeError, match="result message"):
        await Invalid().invoke(UserMessage(), Context())


@pytest.mark.asyncio
async def test_first_direct_execution_recursively_freezes_module_definition() -> None:
    child = Child()
    parent = Parent(child)

    await parent.invoke(UserMessage(content="freeze"), Context())

    assert parent.definition_frozen
    assert child.definition_frozen
    with pytest.raises(RuntimeError, match="definition is frozen"):
        parent.child = Child()
    with pytest.raises(RuntimeError, match="definition is frozen"):
        child.label = "request-local"
    with pytest.raises(RuntimeError, match="cannot be deleted"):
        del parent.child


def test_creating_direct_stream_freezes_before_execution_starts() -> None:
    module = Child()

    module.stream(UserMessage(content="freeze"), Context())

    assert module.definition_frozen
    with pytest.raises(RuntimeError, match="definition is frozen"):
        module.policy = "changed"


@pytest.mark.asyncio
async def test_forward_cannot_write_request_state_to_module_attributes() -> None:
    class Stateful(Module[UserMessage, AIMessage]):
        async def forward(self, message: UserMessage, context: Context):
            self.current_message = message
            return AIMessage(content="unreachable"), context

    with pytest.raises(RuntimeError, match="definition is frozen"):
        await Stateful().invoke(UserMessage(content="request"), Context())


@pytest.mark.asyncio
@pytest.mark.parametrize("attribute_name", ["history", "_history"])
async def test_direct_freeze_rejects_mutable_stored_state_before_forward(
    attribute_name: str,
) -> None:
    class Leaky(Module[UserMessage, AIMessage]):
        def __init__(self) -> None:
            super().__init__()
            setattr(self, attribute_name, [])

        async def forward(self, message: UserMessage, context: Context):
            history = getattr(self, attribute_name)
            history.append(message.content)
            return AIMessage(content=",".join(history)), context

    module = Leaky()

    with pytest.raises(TypeError, match=r"stored Module state .*history"):
        await module.invoke(UserMessage(content="request"), Context())
    assert not module.definition_frozen


@pytest.mark.asyncio
async def test_direct_freeze_validates_entire_child_graph_before_freezing() -> None:
    child = Child()
    child.mutable_config = {"mode": "unsafe"}
    parent = Parent(child)

    with pytest.raises(TypeError, match="mutable_config"):
        await parent.invoke(UserMessage(content="request"), Context())

    assert not parent.definition_frozen
    assert not child.definition_frozen


@pytest.mark.asyncio
async def test_trusted_live_resource_cannot_be_a_raw_mutable_container() -> None:
    class DisguisedState(Module[UserMessage, AIMessage]):
        trusted_live_resource_attributes = ("history",)

        def __init__(self) -> None:
            super().__init__()
            self.history: list[str] = []

        async def forward(self, message: UserMessage, context: Context):
            self.history.append(message.content)
            return AIMessage(content="ok"), context

    with pytest.raises(TypeError, match="not a raw mutable container"):
        await DisguisedState().invoke(UserMessage(content="request"), Context())


@pytest.mark.asyncio
async def test_direct_freeze_rejects_stateful_callable_without_explicit_boundary() -> None:
    class StatefulCallable:
        def __init__(self) -> None:
            self.history: list[str] = []

        def __call__(self, value: str) -> int:
            self.history.append(value)
            return len(self.history)

    class LeakyCallable(Module[UserMessage, AIMessage]):
        def __init__(self) -> None:
            super().__init__()
            self.handler = StatefulCallable()

        async def forward(self, message: UserMessage, context: Context):
            return AIMessage(content=str(self.handler(message.content))), context

    module = LeakyCallable()

    with pytest.raises(TypeError, match=r"callable Module state .*handler"):
        await module.invoke(UserMessage(content="request"), Context())
    assert not module.definition_frozen


@pytest.mark.asyncio
@pytest.mark.parametrize("class_state", [[], {}])
async def test_direct_freeze_rejects_mutable_user_class_state(class_state) -> None:
    class ClassState(Module[UserMessage, AIMessage]):
        history = class_state

        async def forward(self, message: UserMessage, context: Context):
            return AIMessage(content="unsafe"), context

    with pytest.raises(TypeError, match=r"Module class state .*history"):
        await ClassState().invoke(UserMessage(), Context())


@pytest.mark.asyncio
async def test_direct_freeze_rejects_callable_user_class_state() -> None:
    class StatefulCallable:
        def __call__(self) -> None:
            pass

    class ClassState(Module[UserMessage, AIMessage]):
        handler = StatefulCallable()

        async def forward(self, message: UserMessage, context: Context):
            return AIMessage(content="unsafe"), context

    with pytest.raises(TypeError, match=r"Module class state .*handler"):
        await ClassState().invoke(UserMessage(), Context())


@pytest.mark.asyncio
async def test_direct_revalidates_immutable_class_config_after_freeze() -> None:
    class ClassConfig(Module[UserMessage, AIMessage]):
        mode = "before"

        async def forward(self, message: UserMessage, context: Context):
            return AIMessage(content=self.mode), context

    module = ClassConfig()
    output, _ = await module.invoke(UserMessage(), Context())
    assert output.content == "before"

    ClassConfig.mode = "after"
    with pytest.raises(RuntimeError, match="definition changed after freeze"):
        await module.invoke(UserMessage(), Context())


@pytest.mark.asyncio
async def test_direct_stream_revalidates_class_config_when_execution_starts() -> None:
    class ClassConfig(Module[UserMessage, AIMessage]):
        mode = "before"

        async def forward(self, message: UserMessage, context: Context):
            return AIMessage(content=self.mode), context

    module = ClassConfig()
    stream = module.stream(UserMessage(), Context())
    ClassConfig.mode = "after"

    with pytest.raises(RuntimeError, match="definition changed after freeze"):
        await stream.final_result()


@pytest.mark.asyncio
async def test_direct_parent_invokes_prebound_child_in_managed_runtime() -> None:
    runtime = LocalRuntime()
    bound_child = runtime.bind(Child())

    class BoundParent(Module[UserMessage, AIMessage]):
        def __init__(self) -> None:
            super().__init__()
            self.child = bound_child

        async def forward(self, message: UserMessage, context: Context):
            return await self.child(message, context)

    output, context = await BoundParent().invoke(
        UserMessage(content="managed-child"), Context()
    )

    assert output.content == "child:managed-child"
    assert context.messages == (UserMessage(content="managed-child"), output)
    await runtime.close()


@pytest.mark.asyncio
async def test_direct_parent_cancellation_cleans_up_prebound_child() -> None:
    entered = asyncio.Event()
    finished = asyncio.Event()

    class BlockingChild(Module[UserMessage, AIMessage]):
        trusted_live_resource_attributes = ("entered", "finished")

        def __init__(self) -> None:
            super().__init__()
            self.entered = entered
            self.finished = finished

        async def forward(self, message: UserMessage, context: Context):
            self.entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.finished.set()

    runtime = LocalRuntime()
    bound_child = runtime.bind(BlockingChild())

    class BoundParent(Module[UserMessage, AIMessage]):
        def __init__(self) -> None:
            super().__init__()
            self.child = bound_child

        async def forward(self, message: UserMessage, context: Context):
            return await self.child(message, context)

    task = asyncio.create_task(BoundParent().invoke(UserMessage(), Context()))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert finished.is_set()
    await runtime.close()


@pytest.mark.asyncio
async def test_definition_freeze_does_not_deep_freeze_deployment_adapter() -> None:
    class Adapter:
        def __init__(self) -> None:
            self.calls = 0

        def use(self) -> int:
            self.calls += 1
            return self.calls

    class UsesAdapter(Module[UserMessage, AIMessage]):
        trusted_live_resource_attributes = ("adapter",)

        def __init__(self, adapter: Adapter) -> None:
            super().__init__()
            self.adapter = adapter

        async def forward(self, message: UserMessage, context: Context):
            return AIMessage(content=str(self.adapter.use())), context

    adapter = Adapter()
    module = UsesAdapter(adapter)

    first, _ = await module.invoke(UserMessage(), Context())
    second, _ = await module.invoke(UserMessage(), Context())

    assert (first.content, second.content) == ("1", "2")
    assert adapter.calls == 2
    with pytest.raises(RuntimeError, match="definition is frozen"):
        module.adapter = Adapter()
