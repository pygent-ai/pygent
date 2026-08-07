from __future__ import annotations

import asyncio
import time

import pytest

from pygent.agent import ReActLayer
from pygent.core import (
    AIMessage,
    Context,
    Message,
    Module,
    RemoteModule,
    ToolMessage,
    UserMessage,
)
from pygent.runtime import (
    CapacityPolicy,
    CapacityScope,
    ExecutionAdmissionError,
    ExecutionCapacityPolicy,
    ExecutionDeadlineExceeded,
    ExecutionOptions,
    ExecutionStatus,
    LocalRuntime,
    compile_execution_plan,
)
from pygent.runtime.local import _RunnableGate


class Echo(Module[UserMessage, AIMessage]):
    async def forward(
        self, message: UserMessage, context: Context
    ) -> tuple[AIMessage, Context]:
        await self.emit(kind="echo.progress", data={"text": message.content})
        output = AIMessage(content=message.content.upper())
        return output, context + message + output


class Blocking(Module[UserMessage, AIMessage]):
    trusted_live_resource_attributes = ("_entered", "_release")

    def __init__(self, entered: asyncio.Event, release: asyncio.Event) -> None:
        super().__init__()
        # Test-only synchronization is private instrumentation, not portable
        # Module configuration.
        self._entered = entered
        self._release = release

    async def forward(
        self, message: UserMessage, context: Context
    ) -> tuple[AIMessage, Context]:
        self._entered.set()
        await self._release.wait()
        return AIMessage(content=message.content), context


class RemoteCaller(Module[UserMessage, AIMessage]):
    def __init__(self, target: RemoteModule[UserMessage, AIMessage]) -> None:
        super().__init__()
        self.target = target

    async def forward(
        self, message: UserMessage, context: Context
    ) -> tuple[AIMessage, Context]:
        return await self.target(message, context)


def _capacity(*, queue: int = 2, live: int = 1) -> ExecutionCapacityPolicy:
    return ExecutionCapacityPolicy(
        scope=CapacityScope.RUNTIME_INSTANCE,
        max_live_executions=live,
        max_runnable_executions=1,
        max_queue_size=queue,
        max_waiters=2,
        max_child_depth=4,
        max_children_per_execution=8,
    )


def _binding(runtime: LocalRuntime, *, queue: int = 2):
    return runtime.create_binding(
        name="test",
        execution_capacity=_capacity(queue=queue),
        model_capacity=CapacityPolicy.passthrough(),
        tool_capacity=CapacityPolicy.passthrough(),
    )


@pytest.mark.asyncio
async def test_invoke_stream_and_run_handle_share_result_and_event_source():
    runtime = LocalRuntime()
    bound = _binding(runtime).bind(Echo())
    input_message = UserMessage(content="hello")

    invoked = await bound.invoke(input_message, Context())
    async with bound.stream(input_message, Context()) as stream:
        events = [event async for event in stream]
        streamed = await stream.final_result()

    assert streamed == invoked
    assert [event.kind for event in events] == [
        "execution.started",
        "span.started",
        "echo.progress",
        "span.completed",
        "execution.completed",
    ]
    assert [event.sequence for event in events] == list(range(5))
    await runtime.close()


@pytest.mark.asyncio
async def test_run_handle_supports_cursor_reconnect_without_cancelling_run():
    runtime = LocalRuntime()
    bound = _binding(runtime).bind(Echo())
    handle = await bound.start(UserMessage(content="hello"), Context())

    result = await handle.result()
    async with handle.subscribe(after=0) as subscription:
        remaining = [event async for event in subscription]

    assert result[0].content == "HELLO"
    assert handle.status is ExecutionStatus.SUCCEEDED
    assert [event.sequence for event in remaining] == [1, 2, 3, 4]
    assert await handle.cancel() is False
    await runtime.close()


@pytest.mark.asyncio
async def test_managed_handle_wait_hands_off_runnable_lease_before_resume():
    runtime = LocalRuntime()
    binding = runtime.create_binding(
        name="handle-handoff",
        execution_capacity=ExecutionCapacityPolicy(
            scope=CapacityScope.RUNTIME_INSTANCE,
            max_live_executions=2,
            max_runnable_executions=1,
            max_queue_size=1,
            max_waiters=1,
            max_child_depth=4,
            max_children_per_execution=4,
        ),
        model_capacity=CapacityPolicy.passthrough(),
        tool_capacity=CapacityPolicy.passthrough(),
    )
    target = binding.bind(Echo())

    class WaitsForHandle(Module[UserMessage, AIMessage]):
        async def forward(self, message, context):
            handle = await target.start(message, context)
            return await handle.result()

    parent = binding.bind(WaitsForHandle())
    output, context = await asyncio.wait_for(
        parent.invoke(UserMessage(content="handoff"), Context()),
        timeout=1,
    )

    assert output.content == "HANDOFF"
    assert [item.content for item in context.messages] == ["handoff", "HANDOFF"]
    await runtime.close()


@pytest.mark.asyncio
async def test_runnable_gate_bounds_resume_priority_without_starving_start():
    gate = _RunnableGate(1)
    await gate.acquire()
    order: list[str] = []

    async def waiter(label: str, *, resume: bool) -> None:
        await gate.acquire(resume=resume)
        order.append(label)
        await asyncio.sleep(0)
        gate.release()

    # START arrives first, but an already-waiting RESUME still gets priority.
    start = asyncio.create_task(waiter("start", resume=False))
    await asyncio.sleep(0)
    resumes = [
        asyncio.create_task(waiter(f"resume-{index}", resume=True))
        for index in range(24)
    ]
    await asyncio.sleep(0)
    gate.release()
    await asyncio.wait_for(asyncio.gather(start, *resumes), timeout=1)

    start_index = order.index("start")
    assert order[0] == "resume-0"
    assert start_index <= _RunnableGate._MAX_CONSECUTIVE_RESUMES
    assert start_index < len(resumes)
    assert [item for item in order if item.startswith("resume-")] == [
        f"resume-{index}" for index in range(24)
    ]


@pytest.mark.asyncio
async def test_binding_admission_queue_is_bounded_before_forward_starts():
    runtime = LocalRuntime()
    entered = asyncio.Event()
    release = asyncio.Event()
    binding = _binding(runtime, queue=0)
    first = binding.bind(Blocking(entered, release))
    second = binding.bind(Echo())
    first_handle = await first.start(UserMessage(content="one"), Context())
    await entered.wait()

    second_handle = await second.start(UserMessage(content="two"), Context())
    with pytest.raises(ExecutionAdmissionError):
        await second_handle.result()

    release.set()
    await first_handle.result()
    await runtime.close()


@pytest.mark.asyncio
async def test_deadline_and_explicit_cancel_have_distinct_terminal_states():
    runtime = LocalRuntime()
    entered = asyncio.Event()
    release = asyncio.Event()
    bound = _binding(runtime).bind(Blocking(entered, release))
    deadline_handle = await bound.start(
        UserMessage(content="deadline"),
        Context(),
        execution=ExecutionOptions(deadline=time.monotonic() + 0.02),
    )
    with pytest.raises(ExecutionDeadlineExceeded):
        await deadline_handle.result()
    assert deadline_handle.status is ExecutionStatus.DEADLINE_EXCEEDED

    entered.clear()
    cancel_handle = await bound.start(UserMessage(content="cancel"), Context())
    await entered.wait()
    assert await cancel_handle.cancel() is True
    assert cancel_handle.status is ExecutionStatus.CANCELLED
    await runtime.close()


@pytest.mark.asyncio
async def test_remote_module_is_resolved_by_declared_binding_reference():
    runtime = LocalRuntime()
    binding = _binding(runtime)
    target = binding.bind(Echo())
    runtime.register_remote("reviewer", target)
    caller = binding.bind(
        RemoteCaller(RemoteModule[UserMessage, AIMessage](binding_ref="reviewer"))
    )

    result = await caller.invoke(UserMessage(content="remote"), Context())

    assert result[0].content == "REMOTE"
    await runtime.close()


def test_execution_plan_compilation_is_deterministic_and_tracks_shared_nodes():
    shared = Echo()

    class Parent(Module[UserMessage, AIMessage]):
        def __init__(self) -> None:
            super().__init__()
            self.right = shared
            self.left = shared

    first = compile_execution_plan(Parent())
    second = compile_execution_plan(Parent())

    assert first.graph_hash == second.graph_hash
    assert first.modules[0].path == "root"
    assert first.modules[0].children == ("root.left",)


@pytest.mark.asyncio
async def test_generic_execution_requirement_enforces_finite_deadline():
    from pygent.core import ExecutionRequirements

    class DeadlineRequired(Echo):
        execution_requirements = ExecutionRequirements(
            requires_finite_deadline=True
        )

    runtime = LocalRuntime()
    bound = _binding(runtime).bind(DeadlineRequired())

    with pytest.raises(ExecutionAdmissionError, match="finite execution deadline"):
        await bound.start(UserMessage(content="missing"), Context())

    result = await bound.invoke(
        UserMessage(content="present"),
        Context(),
        execution=ExecutionOptions(deadline=time.monotonic() + 1),
    )
    assert result[0].content == "PRESENT"
    await runtime.close()


@pytest.mark.asyncio
async def test_nested_react_requires_finite_root_deadline():
    class Model(Module[Message, AIMessage]):
        async def forward(self, message, context):
            return AIMessage(content="done"), context

    class Tools(Module[AIMessage, ToolMessage]):
        async def forward(self, message, context):
            return ToolMessage(), context

    class Parent(Module[UserMessage, AIMessage]):
        def __init__(self) -> None:
            super().__init__()
            self.react = ReActLayer(model=Model(), tools=Tools())

        async def forward(self, message, context):
            return await self.react(message, context)

    runtime = LocalRuntime()
    bound = _binding(runtime).bind(Parent())

    with pytest.raises(ExecutionAdmissionError, match="ReActLayer.*finite execution deadline"):
        await bound.start(UserMessage(content="missing"), Context())

    result = await bound.invoke(
        UserMessage(content="present"),
        Context(),
        execution=ExecutionOptions(deadline=time.monotonic() + 1),
    )
    assert result[0].content == "done"
    await runtime.close()


@pytest.mark.asyncio
async def test_external_wait_releases_then_reacquires_runnable_lease():
    class Waiting(Module[UserMessage, AIMessage]):
        async def forward(self, message, context):
            value = await self.wait_external(
                kind="approval",
                key=message.content,
                request={"question": "continue?"},
                timeout=1,
            )
            return AIMessage(content=str(value["decision"])), context

    runtime = LocalRuntime()
    binding = runtime.create_binding(
        name="external-handoff",
        execution_capacity=_capacity(live=2),
        model_capacity=CapacityPolicy.passthrough(),
        tool_capacity=CapacityPolicy.passthrough(),
    )
    waiting = binding.bind(Waiting())
    echo = binding.bind(Echo())
    handle = await waiting.start(
        UserMessage(content="wait-1"),
        Context(),
        execution=ExecutionOptions(deadline=time.monotonic() + 2),
    )
    while handle.status is not ExecutionStatus.WAITING_EXTERNAL:
        await asyncio.sleep(0)

    # max_runnable_executions is one; this can finish only if the waiter released it.
    result = await echo.invoke(UserMessage(content="other"), Context())
    assert result[0].content == "OTHER"

    await runtime.deliver_external(
        kind="approval", key="wait-1", value={"decision": "approved"}
    )
    assert (await handle.result())[0].content == "approved"
    await runtime.close()


@pytest.mark.asyncio
async def test_external_wait_is_capped_by_deployment_policy_and_cleans_up():
    class Waiting(Module[UserMessage, AIMessage]):
        async def forward(self, message, context):
            value = await self.wait_external(
                kind="approval",
                key=message.content,
                request={"question": "continue?"},
                timeout=None,
            )
            return AIMessage(content=str(value["decision"])), context

    runtime = LocalRuntime()
    binding = runtime.create_binding(
        name="short-external-wait",
        execution_capacity=ExecutionCapacityPolicy(
            scope=CapacityScope.RUNTIME_INSTANCE,
            max_live_executions=1,
            max_runnable_executions=1,
            max_queue_size=0,
            max_waiters=1,
            max_child_depth=4,
            max_children_per_execution=4,
            max_external_wait_seconds=0.02,
        ),
        model_capacity=CapacityPolicy.passthrough(),
        tool_capacity=CapacityPolicy.passthrough(),
    )
    waiting = binding.bind(Waiting())
    far_future = time.monotonic() + 365 * 24 * 60 * 60

    expired = await waiting.start(
        UserMessage(content="same-key"),
        Context(),
        execution=ExecutionOptions(deadline=far_future),
    )
    with pytest.raises(ExecutionDeadlineExceeded):
        await asyncio.wait_for(expired.result(), timeout=1)

    # Timeout removes the old waiter, so the same stable key can be admitted
    # again and delivered without colliding with leaked state.
    retried = await waiting.start(
        UserMessage(content="same-key"),
        Context(),
        execution=ExecutionOptions(deadline=far_future),
    )
    while retried.status is not ExecutionStatus.WAITING_EXTERNAL:
        await asyncio.sleep(0)
    await runtime.deliver_external(
        kind="approval", key="same-key", value={"decision": "approved"}
    )
    assert (await retried.result())[0].content == "approved"
    await runtime.close()


@pytest.mark.asyncio
async def test_model_resource_handoff_does_not_hold_runnable_lease():
    from pygent.core.module import _execution_scope

    resource_entered = asyncio.Event()
    resource_release = asyncio.Event()

    class UsesModelResource(Module[UserMessage, AIMessage]):
        async def forward(self, message, context):
            scope = _execution_scope.get()
            assert scope is not None
            permit = scope.model_permit
            async with permit():
                resource_entered.set()
                await resource_release.wait()
            return AIMessage(content="done"), context

    runtime = LocalRuntime()
    managed = runtime.create_binding(
        name="resource-handoff",
        execution_capacity=_capacity(queue=2, live=2),
        model_capacity=CapacityPolicy.limited(
            max_concurrency=1, max_queue_size=1
        ),
        tool_capacity=CapacityPolicy.passthrough(),
    )
    first = managed.bind(UsesModelResource())
    echo = managed.bind(Echo())
    handle = await first.start(UserMessage(), Context())
    await resource_entered.wait()

    assert (await echo.invoke(UserMessage(content="free"), Context()))[0].content == "FREE"
    resource_release.set()
    assert (await handle.result())[0].content == "done"
    await runtime.close()


@pytest.mark.asyncio
async def test_managed_execution_validates_each_child_result_boundary():
    class InvalidChild(Module[UserMessage, AIMessage]):
        async def forward(self, message, context):
            return "not-a-message", context

    class MasksInvalidChild(Module[UserMessage, AIMessage]):
        def __init__(self) -> None:
            super().__init__()
            self.child = InvalidChild()

        async def forward(self, message, context):
            await self.child(message, context)
            return AIMessage(content="masked"), context

    runtime = LocalRuntime()
    bound = _binding(runtime).bind(MasksInvalidChild())

    with pytest.raises(TypeError, match="result message"):
        await bound.invoke(UserMessage(), Context())

    await runtime.close()


@pytest.mark.asyncio
async def test_managed_child_has_lineage_and_hands_off_runnable_lease():
    parent_ready = asyncio.Event()
    parent_go = asyncio.Event()
    child_entered = asyncio.Event()
    child_release = asyncio.Event()
    competitor_entered = asyncio.Event()
    competitor_release = asyncio.Event()

    class Parent(Module[UserMessage, AIMessage]):
        def __init__(self) -> None:
            super().__init__()
            self.child = Blocking(child_entered, child_release)

        async def forward(self, message, context):
            parent_ready.set()
            await parent_go.wait()
            return await self.child(message, context)

    runtime = LocalRuntime()
    binding = runtime.create_binding(
        name="child-lineage",
        execution_capacity=_capacity(queue=2, live=2),
        model_capacity=CapacityPolicy.passthrough(),
        tool_capacity=CapacityPolicy.passthrough(),
    )
    parent = binding.bind(Parent())
    competitor = binding.bind(Blocking(competitor_entered, competitor_release))

    # Queue the competitor behind the Parent before the Parent yields to Child.
    parent_handle = await parent.start(UserMessage(content="parent"), Context())
    await parent_ready.wait()
    competitor_handle = await competitor.start(
        UserMessage(content="competitor"), Context()
    )
    await asyncio.sleep(0)
    parent_go.set()
    await competitor_entered.wait()
    assert not child_entered.is_set()

    competitor_release.set()
    await competitor_handle.result()
    await child_entered.wait()
    child_release.set()
    await parent_handle.result()

    async with parent_handle.subscribe() as subscription:
        events = [event async for event in subscription]
    child_events = [
        event for event in events
        if event.module_path == "root.child" and event.kind.startswith("span.")
    ]
    assert [event.kind for event in child_events] == [
        "span.started",
        "span.completed",
    ]
    assert {event.execution_id for event in child_events} == {parent_handle.execution_id}
    assert child_events[0].parent_span_id is not None
    await runtime.close()


@pytest.mark.asyncio
async def test_prebound_child_uses_its_own_runtime_and_binding_capacity():
    child_entered = asyncio.Event()
    child_release = asyncio.Event()

    class Parent(Module[UserMessage, AIMessage]):
        def __init__(self, child) -> None:
            super().__init__()
            self.child = child

        async def forward(self, message, context):
            return await self.child(message, context)

    parent_runtime = LocalRuntime()
    child_runtime = LocalRuntime()
    parent_binding = parent_runtime.create_binding(
        name="parent",
        execution_capacity=_capacity(queue=2, live=2),
        model_capacity=CapacityPolicy.passthrough(),
        tool_capacity=CapacityPolicy.passthrough(),
    )
    child_binding = child_runtime.create_binding(
        name="child",
        execution_capacity=_capacity(queue=1),
        model_capacity=CapacityPolicy.passthrough(),
        tool_capacity=CapacityPolicy.passthrough(),
    )
    pinned = child_binding.bind(Blocking(child_entered, child_release))
    parent = parent_binding.bind(Parent(pinned))
    sibling = parent_binding.bind(Echo())

    handle = await parent.start(UserMessage(content="pinned"), Context())
    await child_entered.wait()
    assert (await sibling.invoke(UserMessage(content="free"), Context()))[0].content == "FREE"
    child_release.set()
    assert (await handle.result())[0].content == "pinned"
    async with handle.subscribe() as subscription:
        events = [event async for event in subscription]
    assert {
        event.module_path
        for event in events
        if event.kind.startswith("span.") and event.parent_span_id is not None
    } == {"root.child"}

    await parent_runtime.close()
    await child_runtime.close()


@pytest.mark.asyncio
async def test_child_deadline_and_cancel_propagate_with_terminal_events():
    entered = asyncio.Event()
    release = asyncio.Event()

    class Parent(Module[UserMessage, AIMessage]):
        def __init__(self) -> None:
            super().__init__()
            self.child = Blocking(entered, release)

        async def forward(self, message, context):
            return await self.child(message, context)

    runtime = LocalRuntime()
    bound = _binding(runtime).bind(Parent())
    deadline_handle = await bound.start(
        UserMessage(content="deadline"),
        Context(),
        execution=ExecutionOptions(deadline=time.monotonic() + 0.02),
    )
    with pytest.raises(ExecutionDeadlineExceeded):
        await deadline_handle.result()
    async with deadline_handle.subscribe() as subscription:
        deadline_events = [event async for event in subscription]
    assert "span.deadline_exceeded" in [event.kind for event in deadline_events]

    entered.clear()
    cancel_handle = await bound.start(UserMessage(content="cancel"), Context())
    await entered.wait()
    assert await cancel_handle.cancel()
    async with cancel_handle.subscribe() as subscription:
        cancel_events = [event async for event in subscription]
    assert "span.cancelled" in [event.kind for event in cancel_events]
    await runtime.close()


def test_runtime_rejects_missing_declared_module_capabilities():
    from pygent.core import ExecutionRequirements

    class NeedsGPU(Echo):
        execution_requirements = ExecutionRequirements(
            required_capabilities=("accelerator.gpu",)
        )

    runtime = LocalRuntime()
    with pytest.raises(ExecutionAdmissionError, match="accelerator.gpu"):
        runtime.bind(NeedsGPU())

    capable = LocalRuntime(capabilities=("accelerator.gpu",))
    bound = capable.bind(NeedsGPU())
    assert bound.plan.modules[0].required_capabilities == ("accelerator.gpu",)


def test_plan_includes_raw_prebound_and_remote_declared_dependencies():
    runtime = LocalRuntime()
    pinned = _binding(runtime).bind(Echo())

    class Composite(Module[UserMessage, AIMessage]):
        def __init__(self) -> None:
            super().__init__()
            self.raw = Echo()
            self.pinned = pinned
            self.remote = RemoteModule[UserMessage, AIMessage](
                binding_ref="reviewer-service"
            )

    plan = compile_execution_plan(Composite())
    root = next(spec for spec in plan.modules if spec.path == "root")
    pinned_spec = next(spec for spec in plan.modules if spec.path == "root.pinned")
    remote_spec = next(spec for spec in plan.modules if spec.path == "root.remote")

    assert root.children == ("root.pinned", "root.raw", "root.remote")
    assert dict(pinned_spec.metadata)["binding"] == "test"
    assert remote_spec.definition_id == "remote:reviewer-service"


@pytest.mark.asyncio
async def test_managed_child_rejects_unregistered_asyncio_task():
    class DetachedParent(Module[UserMessage, AIMessage]):
        def __init__(self) -> None:
            super().__init__()
            self.child = Echo()

        async def forward(self, message, context):
            task = asyncio.create_task(self.child(message, context))
            return await task

    runtime = LocalRuntime()
    bound = _binding(runtime).bind(DetachedParent())

    with pytest.raises(ExecutionAdmissionError, match="unregistered asyncio Task"):
        await bound.invoke(UserMessage(content="unsafe"), Context())
    await runtime.close()


@pytest.mark.asyncio
async def test_structured_gather_executions_children_with_one_runnable_lease():
    class Child(Module[UserMessage, AIMessage]):
        def __init__(self, suffix: str) -> None:
            super().__init__()
            self.suffix = suffix

        async def forward(self, message, context):
            await asyncio.sleep(0)
            return AIMessage(content=message.content + self.suffix), context

    class ParallelParent(Module[UserMessage, AIMessage]):
        def __init__(self) -> None:
            super().__init__()
            self.left = Child("-left")
            self.right = Child("-right")

        async def forward(self, message, context):
            left, right = await self.gather(
                self.left(message, context),
                self.right(message, context),
            )
            return AIMessage(content=f"{left[0].content}|{right[0].content}"), context

    runtime = LocalRuntime()
    handle = await _binding(runtime).bind(ParallelParent()).start(
        UserMessage(content="work"), Context()
    )
    output, _ = await handle.result()
    async with handle.subscribe() as subscription:
        events = [event async for event in subscription]
    await runtime.close()

    assert output.content == "work-left|work-right"
    assert [event.kind for event in events].count("span.completed") >= 3
