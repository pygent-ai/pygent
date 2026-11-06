"""Execution-level cancel mid tool batch: paired results, ordered artifact.

Pins the contract that a tool batch interrupted by ``handle.cancel()``
(or a deadline kill, which delivers the same task cancellation) finishes
with one paired ``ToolResult`` per original ``call_id``, registers a
``tool.batch.cancelled`` journal artifact ordered ahead of the terminal
event, and still terminates the execution with ``CancelledError``
semantics intact. The steering interrupt path has its own suite and must
stay byte-identical.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from pygent.agent import ReActLayer
from pygent.core import (
    AIMessage,
    Context,
    ExecutionStatus,
    Message,
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
from pygent.runtime.tasks import DurableToolTaskManager
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
    ToolTaskState,
)


class _ModelRunState:
    def __init__(self, answers: tuple[AIMessage, ...]) -> None:
        self.answers = answers
        self.inputs: list[Message] = []


class ScriptedModel(Module[Message, AIMessage]):
    trusted_live_resource_attributes = ("state",)

    def __init__(self, answers: tuple[AIMessage, ...]) -> None:
        super().__init__()
        self.state = _ModelRunState(answers)

    async def forward(self, message: Message, context: Context):
        self.state.inputs.append(message)
        answers = self.state.answers
        return answers[min(len(self.state.inputs) - 1, len(answers) - 1)], context


def _allow(request, context) -> ToolAuthorizationDecision:
    return ToolAuthorizationDecision(
        call_id=request.call.call_id,
        allowed=True,
        reason_code="allowed",
        lifecycle="sync",
    )


def _detach_allow(request, context) -> ToolAuthorizationDecision:
    return ToolAuthorizationDecision(
        call_id=request.call.call_id,
        allowed=True,
        reason_code="allowed",
        lifecycle="detach",
    )


def bind(runtime: LocalRuntime, react: ReActLayer):
    binding = runtime.create_binding(
        name="cancel-test",
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


def options(deadline: float | None = None) -> ExecutionOptions:
    return ExecutionOptions(
        deadline=deadline if deadline is not None else time.monotonic() + 30
    )


def double_spec(wait_timeout: float | None = None) -> ToolSpec:
    return ToolSpec(
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
        wait_timeout=wait_timeout,
    )


def both_calls_answer() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=(
            ToolCall(call_id="fast", name="double", arguments={"value": 1}),
            ToolCall(call_id="slow", name="double", arguments={"value": 2}),
        ),
    )


async def collect_events(handle) -> list[Any]:
    async with handle.subscribe() as stream:
        return [event async for event in stream]


def artifact_events(events: list[Any]) -> list[Any]:
    return [event for event in events if event.kind == "tool.batch.cancelled"]


def results_by_call(event: Any) -> dict[str, dict[str, Any]]:
    return {item["call_id"]: item for item in event.data["results"]}


@pytest.mark.asyncio
async def test_cancel_mid_sync_batch_keeps_completed_and_cancels_inflight() -> None:
    tool = double_spec()
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
        executor_registry=registry,
        authorization_adapter=_allow,
        max_concurrency=2,
    )
    model = ScriptedModel((both_calls_answer(), AIMessage(content="final")))
    runtime = LocalRuntime()
    handle = await bind(runtime, ReActLayer(model=model, tools=tools)).start(
        UserMessage(content="initial"),
        Context(tools=(tool.definition,), metadata={"permissions": ["tool:use"]}),
        execution=options(),
    )
    await slow_entered.wait()
    await asyncio.sleep(0.05)  # let the fast call finish deterministically
    assert await handle.cancel() is True
    outcome = await handle.outcome()
    assert outcome.status is ExecutionStatus.CANCELLED

    events = await collect_events(handle)
    artifacts = artifact_events(events)
    assert len(artifacts) == 1
    cancelled_index = events.index(
        next(event for event in events if event.kind == "execution.cancelled")
    )
    assert events.index(artifacts[0]) < cancelled_index
    assert list(artifacts[0].data["call_ids"]) == ["fast", "slow"]
    results = results_by_call(artifacts[0])
    assert results["fast"]["status"] == "succeeded"
    assert results["fast"]["output"] == 2
    assert results["slow"]["status"] == "cancelled"
    assert results["slow"]["error_code"] == "interrupted_by_execution_cancel"
    assert results["slow"]["side_effect_committed"] is None
    await runtime.close()


class GateAfterAdmissionManager(InMemoryToolTaskManager):
    """Parks inside the real admission flow: prepare -> [gate] -> start.

    The managed detach path calls ``manager.prepare`` then ``manager.start``
    separately (scope.submit_tool_task), so gating these two primitives
    reproduces the critical window exactly: the runtime records the task
    identity at the admission boundary while ``start`` has not completed,
    and a cancel fired while parked never lets the layer observe the return
    value. Modes are queued per ``start`` call: ``before_launch`` parks with
    the task persisted but not launched, ``after_launch`` parks with the
    task already running.
    """

    def __init__(self, registry: ExecutorRegistry) -> None:
        super().__init__(registry)
        self.admission_done = asyncio.Event()
        self.launched = asyncio.Event()
        self.parked = asyncio.Event()
        self.release = asyncio.Event()
        self.modes: list[str] = []

    async def prepare(self, spec, call, *, execution=None):
        task = await super().prepare(spec, call, execution=execution)
        self.admission_done.set()
        return task

    async def start(self, task_id):
        mode = self.modes.pop(0) if self.modes else None
        if mode in ("before_launch", "after_launch"):
            # A fresh gate per gated start, so one release cannot let a
            # later start through.
            self.release = asyncio.Event()
        if mode == "before_launch":
            self.parked.set()
            await self.release.wait()
        await super().start(task_id)
        if mode == "after_launch":
            self.launched.set()
            await self.release.wait()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,expected_state", [
    ("before_launch", ToolTaskState.PENDING),
    ("after_launch", ToolTaskState.RUNNING),
])
async def test_cancel_inside_submit_window_uses_admission_identity(
    mode: str, expected_state: ToolTaskState
) -> None:
    """Cancel between admission and start completing keeps the task identity.

    The manager has persisted the task but the layer never observed the
    return value, so the cleanup consults the execution's admission record
    plus the manager's launch boundary. A task that actually started
    (RUNNING) yields the detached reference; a merely admitted task
    (PENDING) follows the steering adjudication and reports cancelled with
    unknown side-effect state instead of a detached claim on work that
    never started.
    """

    tool = double_spec(wait_timeout=600.0)
    registry = ExecutorRegistry()

    async def double(arguments):
        await asyncio.sleep(30)
        return 4

    registry.register(tool.tool_id, tool.version, LocalToolExecutor(double))
    manager = GateAfterAdmissionManager(registry)
    manager.modes.append(mode)
    tools = ToolCallLayer(
        tools=(tool,),
        executor_registry=registry,
        task_manager=manager,
        authorization_adapter=_detach_allow,
    )
    model = ScriptedModel(
        (
            AIMessage(
                content="",
                tool_calls=(
                    ToolCall(call_id="slow", name="double", arguments={"value": 2}),
                ),
            ),
            AIMessage(content="final"),
        ),
    )
    runtime = LocalRuntime()
    runtime.attach_executor_registry(registry)
    runtime.attach_tool_task_manager(manager)
    handle = await bind(runtime, ReActLayer(model=model, tools=tools)).start(
        UserMessage(content="initial"),
        Context(tools=(tool.definition,), metadata={"permissions": ["tool:use"]}),
        execution=options(),
    )
    await manager.admission_done.wait()
    await (manager.launched.wait() if mode == "after_launch" else manager.parked.wait())
    assert await handle.cancel() is True
    outcome = await handle.outcome()
    assert outcome.status is ExecutionStatus.CANCELLED

    events = await collect_events(handle)
    artifacts = artifact_events(events)
    assert len(artifacts) == 1
    result = next(iter(artifacts[0].data["results"]))
    assert result["error_code"] == "interrupted_by_execution_cancel"
    if expected_state is ToolTaskState.RUNNING:
        assert result["status"] == "detached"
        assert result["side_effect_committed"] is None
        task_id = result["task"]["task_id"]
    else:
        assert result["status"] == "cancelled"
        assert result["side_effect_committed"] is None
        assert result["task"] is None
        by_call = await manager.get_task_by_call("slow")
        assert by_call is not None and by_call.state is ToolTaskState.PENDING
        task_id = by_call.task_id
    admitted = await manager.get_task(task_id)
    assert admitted is not None and admitted.state is expected_state
    await manager.cancel(task_id)
    await runtime.close()


@pytest.mark.asyncio
async def test_call_id_reuse_across_executions_does_not_cross_match() -> None:
    """The admission lookup is scoped to the execution, not the shared manager.

    Execution 1 admits a running detached task under call_id ``slow``; the
    task survives the cancel. Execution 2 reuses the same call_id and is
    cancelled inside its own admission window: its artifact must not pick
    up execution 1's still-running task.
    """

    tool = double_spec(wait_timeout=600.0)
    registry = ExecutorRegistry()

    async def double(arguments):
        await asyncio.sleep(30)
        return 4

    registry.register(tool.tool_id, tool.version, LocalToolExecutor(double))
    manager = GateAfterAdmissionManager(registry)
    manager.modes.extend(["after_launch", "before_launch"])
    tools = ToolCallLayer(
        tools=(tool,),
        executor_registry=registry,
        task_manager=manager,
        authorization_adapter=_detach_allow,
    )
    first_answer = AIMessage(
        content="",
        tool_calls=(ToolCall(call_id="slow", name="double", arguments={"value": 2}),),
    )
    # Both runs must dispatch the same tool call: the shared scripted model
    # hands out answers by call index, so repeat the tool-call answer.
    model = ScriptedModel((first_answer, first_answer, AIMessage(content="final")))
    runtime = LocalRuntime()
    runtime.attach_executor_registry(registry)
    runtime.attach_tool_task_manager(manager)
    bound = bind(runtime, ReActLayer(model=model, tools=tools))
    context = Context(
        tools=(tool.definition,), metadata={"permissions": ["tool:use"]}
    )

    first = await bound.start(
        UserMessage(content="initial"), context, execution=options(),
    )
    await manager.admission_done.wait()
    await manager.launched.wait()
    manager.release.set()  # run 1's start completes; the layer waits on the task
    assert await first.cancel() is True
    outcome = await first.outcome()
    assert outcome.status is ExecutionStatus.CANCELLED
    events = await collect_events(first)
    first_result = next(iter(artifact_events(events)[0].data["results"]))
    assert first_result["status"] == "detached"
    first_task_id = first_result["task"]["task_id"]
    running = await manager.get_task(first_task_id)
    assert running is not None and running.state is ToolTaskState.RUNNING

    second = await bound.start(
        UserMessage(content="second"), context, execution=options(),
    )
    await manager.parked.wait()  # run 2 parked before its own launch
    assert await second.cancel() is True
    outcome = await second.outcome()
    assert outcome.status is ExecutionStatus.CANCELLED
    events = await collect_events(second)
    second_result = next(iter(artifact_events(events)[0].data["results"]))
    assert second_result["status"] == "cancelled"
    assert second_result["task"] is None
    assert second_result["side_effect_committed"] is None
    still_running = await manager.get_task(first_task_id)
    assert still_running is not None and still_running.state is ToolTaskState.RUNNING

    await manager.cancel(first_task_id)
    await runtime.close()


@pytest.mark.asyncio
async def test_cancel_after_fast_completion_keeps_task_reference_and_result() -> None:
    """A task that starts AND completes inside the admission window survives.

    ``task_started`` reads the manager's live task table, which retains
    finished entries until retention pressure, so a task that runs to
    completion while ``submit_tool_task`` is still unwinding is not mistaken
    for unstarted work: the artifact keeps the detached reference and the
    final result stays queryable through it.
    """

    tool = double_spec(wait_timeout=600.0)
    registry = ExecutorRegistry()

    async def double(arguments):
        await asyncio.sleep(0.3)
        return 4

    registry.register(tool.tool_id, tool.version, LocalToolExecutor(double))
    manager = GateAfterAdmissionManager(registry)
    manager.modes.append("after_launch")
    tools = ToolCallLayer(
        tools=(tool,),
        executor_registry=registry,
        task_manager=manager,
        authorization_adapter=_detach_allow,
    )
    model = ScriptedModel(
        (
            AIMessage(
                content="",
                tool_calls=(
                    ToolCall(call_id="slow", name="double", arguments={"value": 2}),
                ),
            ),
            AIMessage(content="final"),
        ),
    )
    runtime = LocalRuntime()
    runtime.attach_executor_registry(registry)
    runtime.attach_tool_task_manager(manager)
    handle = await bind(runtime, ReActLayer(model=model, tools=tools)).start(
        UserMessage(content="initial"),
        Context(tools=(tool.definition,), metadata={"permissions": ["tool:use"]}),
        execution=options(),
    )
    await manager.launched.wait()
    await asyncio.sleep(0.6)  # the tool completes while submit is parked
    assert await handle.cancel() is True
    outcome = await handle.outcome()
    assert outcome.status is ExecutionStatus.CANCELLED

    events = await collect_events(handle)
    artifacts = artifact_events(events)
    assert len(artifacts) == 1
    result = next(iter(artifacts[0].data["results"]))
    assert result["status"] == "detached"
    assert result["side_effect_committed"] is None
    task_id = result["task"]["task_id"]
    finished = await manager.get_task(task_id)
    assert finished is not None and finished.state is ToolTaskState.SUCCEEDED
    final = await runtime.get_tool_result(task_id)
    assert final is not None and final.status == "succeeded"
    assert final.output == 4
    await runtime.close()


@pytest.mark.asyncio
async def test_cancel_mid_detached_wait_yields_task_reference() -> None:
    tool = double_spec(wait_timeout=600.0)
    registry = ExecutorRegistry()
    task_started = asyncio.Event()

    async def double(arguments):
        task_started.set()
        await asyncio.sleep(30)
        return 4

    registry.register(tool.tool_id, tool.version, LocalToolExecutor(double))
    manager = InMemoryToolTaskManager(registry)
    tools = ToolCallLayer(
        tools=(tool,),
        executor_registry=registry,
        task_manager=manager,
        authorization_adapter=_detach_allow,
    )
    model = ScriptedModel(
        (
            AIMessage(
                content="",
                tool_calls=(
                    ToolCall(call_id="slow", name="double", arguments={"value": 2}),
                ),
            ),
            AIMessage(content="final"),
        ),
    )
    runtime = LocalRuntime()
    runtime.attach_executor_registry(registry)
    runtime.attach_tool_task_manager(manager)
    handle = await bind(runtime, ReActLayer(model=model, tools=tools)).start(
        UserMessage(content="initial"),
        Context(tools=(tool.definition,), metadata={"permissions": ["tool:use"]}),
        execution=options(),
    )
    await task_started.wait()
    await asyncio.sleep(0.05)  # let the layer reach its admitted-task wait
    assert await handle.cancel() is True
    outcome = await handle.outcome()
    assert outcome.status is ExecutionStatus.CANCELLED

    events = await collect_events(handle)
    artifacts = artifact_events(events)
    assert len(artifacts) == 1
    result = next(iter(artifacts[0].data["results"]))
    assert result["status"] == "detached"
    assert result["error_code"] == "interrupted_by_execution_cancel"
    assert result["side_effect_committed"] is None
    task_id = result["task"]["task_id"]
    admitted = await manager.get_task(task_id)
    assert admitted is not None and admitted.state is ToolTaskState.RUNNING
    by_call = await manager.get_task_by_call("slow")
    assert by_call is not None and by_call.task_id == task_id
    assert await manager.cancel(task_id) is True
    await runtime.close()


@pytest.mark.asyncio
async def test_cancel_during_authorization_reports_not_started() -> None:
    tool = double_spec()
    registry = ExecutorRegistry()

    async def double(arguments):
        return arguments["value"] * 2

    registry.register(tool.tool_id, tool.version, LocalToolExecutor(double))
    auth_entered = asyncio.Event()

    async def gated_auth(request, context) -> ToolAuthorizationDecision:
        auth_entered.set()
        await asyncio.sleep(30)
        return _allow(request, context)

    tools = ToolCallLayer(
        tools=(tool,),
        executor_registry=registry,
        authorization_adapter=gated_auth,
    )
    model = ScriptedModel(
        (
            AIMessage(
                content="",
                tool_calls=(
                    ToolCall(call_id="slow", name="double", arguments={"value": 2}),
                ),
            ),
            AIMessage(content="final"),
        ),
    )
    runtime = LocalRuntime()
    handle = await bind(runtime, ReActLayer(model=model, tools=tools)).start(
        UserMessage(content="initial"),
        Context(tools=(tool.definition,), metadata={"permissions": ["tool:use"]}),
        execution=options(),
    )
    await auth_entered.wait()
    assert await handle.cancel() is True
    outcome = await handle.outcome()
    assert outcome.status is ExecutionStatus.CANCELLED

    events = await collect_events(handle)
    artifacts = artifact_events(events)
    assert len(artifacts) == 1
    result = next(iter(artifacts[0].data["results"]))
    assert result["status"] == "cancelled"
    assert result["error_code"] == "interrupted_by_execution_cancel"
    assert result["side_effect_committed"] is False
    await runtime.close()


@pytest.mark.asyncio
async def test_cancel_waiting_on_semaphore_reports_not_started() -> None:
    tool = double_spec()
    registry = ExecutorRegistry()
    first_entered = asyncio.Event()
    release_first = asyncio.Event()

    async def double(arguments):
        if arguments["value"] == 1:
            first_entered.set()
            await release_first.wait()
        return arguments["value"] * 2

    registry.register(tool.tool_id, tool.version, LocalToolExecutor(double))
    tools = ToolCallLayer(
        tools=(tool,),
        executor_registry=registry,
        authorization_adapter=_allow,
        max_concurrency=1,
    )
    model = ScriptedModel((both_calls_answer(), AIMessage(content="final")))
    runtime = LocalRuntime()
    handle = await bind(runtime, ReActLayer(model=model, tools=tools)).start(
        UserMessage(content="initial"),
        Context(tools=(tool.definition,), metadata={"permissions": ["tool:use"]}),
        execution=options(),
    )
    await first_entered.wait()
    await asyncio.sleep(0.05)  # let the queued call park on the semaphore
    assert await handle.cancel() is True
    outcome = await handle.outcome()
    assert outcome.status is ExecutionStatus.CANCELLED

    events = await collect_events(handle)
    artifacts = artifact_events(events)
    assert len(artifacts) == 1
    results = results_by_call(artifacts[0])
    assert results["fast"]["status"] == "cancelled"
    assert results["fast"]["side_effect_committed"] is None
    assert results["slow"]["status"] == "cancelled"
    assert results["slow"]["side_effect_committed"] is False
    await runtime.close()


@pytest.mark.asyncio
async def test_double_cancel_records_single_ordered_artifact() -> None:
    tool = double_spec()
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
        executor_registry=registry,
        authorization_adapter=_allow,
        max_concurrency=2,
    )
    model = ScriptedModel((both_calls_answer(), AIMessage(content="final")))
    runtime = LocalRuntime()
    handle = await bind(runtime, ReActLayer(model=model, tools=tools)).start(
        UserMessage(content="initial"),
        Context(tools=(tool.definition,), metadata={"permissions": ["tool:use"]}),
        execution=options(),
    )
    await slow_entered.wait()
    await asyncio.sleep(0.05)
    first = asyncio.create_task(handle.cancel())
    await asyncio.sleep(0.02)  # first cancel delivered; unwinding in progress
    second = asyncio.create_task(handle.cancel())
    assert await asyncio.gather(first, second) == [True, False]
    outcome = await handle.outcome()
    assert outcome.status is ExecutionStatus.CANCELLED

    events = await collect_events(handle)
    artifacts = artifact_events(events)
    # The registration is synchronous, so a re-delivered cancellation cannot
    # interrupt or duplicate it.
    assert len(artifacts) == 1
    cancelled_events = [
        event for event in events if event.kind == "execution.cancelled"
    ]
    assert len(cancelled_events) == 1
    assert events.index(artifacts[0]) < events.index(cancelled_events[0])
    results = results_by_call(artifacts[0])
    assert results["fast"]["status"] == "succeeded"
    assert results["slow"]["status"] == "cancelled"
    await runtime.close()


@pytest.mark.asyncio
async def test_normal_completion_has_no_cancel_artifact() -> None:
    tool = double_spec()
    registry = ExecutorRegistry()

    async def double(arguments):
        return arguments["value"] * 2

    registry.register(tool.tool_id, tool.version, LocalToolExecutor(double))
    tools = ToolCallLayer(
        tools=(tool,),
        executor_registry=registry,
        authorization_adapter=_allow,
    )
    model = ScriptedModel((both_calls_answer(), AIMessage(content="final")))
    runtime = LocalRuntime()
    handle = await bind(runtime, ReActLayer(model=model, tools=tools)).start(
        UserMessage(content="initial"),
        Context(tools=(tool.definition,), metadata={"permissions": ["tool:use"]}),
        execution=options(),
    )
    answer, _ = await handle.result()
    assert answer.content == "final"
    events = await collect_events(handle)
    assert artifact_events(events) == []
    tool_messages = [
        event for event in events if event.kind == "tool.batch.cancelled"
    ]
    assert tool_messages == []
    await runtime.close()


@pytest.mark.asyncio
async def test_deadline_expiry_mid_batch_records_paired_results() -> None:
    """Deadline kills deliver the same task cancellation mid batch.

    The in-flight call's effect machinery converts the deadline into its own
    failed envelope before the layer unwinds, so the artifact records that
    real outcome for the killed call (``side_effect_committed`` unknown) and
    the layer's cancelled mapping only for calls without a completed result.
    Either way the journal is paired and ordered ahead of the terminal event.
    """

    tool = double_spec()
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
        executor_registry=registry,
        authorization_adapter=_allow,
        max_concurrency=2,
    )
    model = ScriptedModel((both_calls_answer(), AIMessage(content="final")))
    runtime = LocalRuntime()
    handle = await bind(runtime, ReActLayer(model=model, tools=tools)).start(
        UserMessage(content="initial"),
        Context(tools=(tool.definition,), metadata={"permissions": ["tool:use"]}),
        execution=options(deadline=time.monotonic() + 0.3),
    )
    await slow_entered.wait()
    await asyncio.sleep(0.05)
    outcome = await handle.outcome()
    assert outcome.status is ExecutionStatus.DEADLINE_EXCEEDED

    events = await collect_events(handle)
    artifacts = artifact_events(events)
    assert len(artifacts) == 1
    results = results_by_call(artifacts[0])
    assert results["fast"]["status"] == "succeeded"
    assert results["slow"]["status"] == "failed"
    # The deadline race has two legitimate winners: the runtime's deadline
    # enforcement records the deadline exception, while the tool runner's own
    # execution-deadline watch can fire first on slower schedulers and envelope
    # the kill as a tool timeout. The paired-results contract under test does
    # not depend on which watcher wins; asserting one code made the test
    # platform-dependent. (Follow-up: the runner converting the EXECUTION
    # deadline into a retryable tool timeout is a product inconsistency to
    # resolve in the tool-cancel semantics.)
    assert results["slow"]["error_code"] in {
        "_ExecutionDeadlineExpired",
        "tool_timeout",
    }
    assert results["slow"]["side_effect_committed"] is None
    await runtime.close()


@pytest.mark.asyncio
async def test_durable_cancel_journals_artifact_before_terminal(tmp_path) -> None:
    tool = double_spec()
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
        executor_registry=registry,
        authorization_adapter=_allow,
        max_concurrency=2,
    )
    model = ScriptedModel((both_calls_answer(), AIMessage(content="final")))
    async with SQLiteHistoryStore(tmp_path / "history.sqlite3") as history:
        runtime = LocalRuntime(history=history)
        handle = await bind(runtime, ReActLayer(model=model, tools=tools)).start(
            UserMessage(content="initial"),
            Context(
                tools=(tool.definition,), metadata={"permissions": ["tool:use"]}
            ),
            execution=options(),
        )
        await slow_entered.wait()
        await asyncio.sleep(0.05)
        assert await handle.cancel() is True
        outcome = await handle.outcome()
        assert outcome.status is ExecutionStatus.CANCELLED

        events = await collect_events(handle)
        artifacts = artifact_events(events)
        assert len(artifacts) == 1
        cancelled_index = events.index(
            next(event for event in events if event.kind == "execution.cancelled")
        )
        assert events.index(artifacts[0]) < cancelled_index
        results = results_by_call(artifacts[0])
        assert results["fast"]["status"] == "succeeded"
        assert results["slow"]["status"] == "cancelled"
        await runtime.close()


class GatedDurableManager(DurableToolTaskManager):
    """Parks inside ``start`` after ``prepare_job`` persisted the Job.

    Reproduces the durable admission window through the real
    ``scope.submit_tool_task`` Job path: the Job and ToolTask are persisted
    (and recorded on the execution), while the launch has not happened.
    """

    def __init__(self, history, registry: ExecutorRegistry) -> None:
        super().__init__(history, registry)
        self.parked = asyncio.Event()
        self.release = asyncio.Event()

    async def start(self, task_id):
        self.parked.set()
        await self.release.wait()
        await super().start(task_id)


@pytest.mark.asyncio
async def test_durable_cancel_during_start_reports_cancelled_pending_job(
    tmp_path,
) -> None:
    """A durable Job persisted but never launched reports cancelled, not detached.

    The launch boundary has not passed, so the artifact carries no detached
    claim and unknown side-effect state; the persisted PENDING Job is
    rediscovered by the durable Job recovery rules instead.
    """

    tool = double_spec(wait_timeout=600.0)
    registry = ExecutorRegistry()

    async def double(arguments):
        await asyncio.sleep(30)
        return 4

    registry.register(tool.tool_id, tool.version, LocalToolExecutor(double))
    model = ScriptedModel(
        (
            AIMessage(
                content="",
                tool_calls=(
                    ToolCall(call_id="slow", name="double", arguments={"value": 2}),
                ),
            ),
            AIMessage(content="final"),
        ),
    )
    async with SQLiteHistoryStore(tmp_path / "history.sqlite3") as history:
        manager = GatedDurableManager(history, registry)
        runtime = LocalRuntime(history=history)
        runtime.attach_executor_registry(registry)
        runtime.attach_tool_task_manager(manager)
        tools = ToolCallLayer(
            tools=(tool,),
            executor_registry=registry,
            task_manager=manager,
            authorization_adapter=_detach_allow,
        )
        handle = await bind(runtime, ReActLayer(model=model, tools=tools)).start(
            UserMessage(content="initial"),
            Context(tools=(tool.definition,), metadata={"permissions": ["tool:use"]}),
            execution=options(),
        )
        await manager.parked.wait()
        assert await handle.cancel() is True
        outcome = await handle.outcome()
        assert outcome.status is ExecutionStatus.CANCELLED

        events = await collect_events(handle)
        artifacts = artifact_events(events)
        assert len(artifacts) == 1
        result = next(iter(artifacts[0].data["results"]))
        assert result["status"] == "cancelled"
        assert result["task"] is None
        assert result["side_effect_committed"] is None
        assert result["error_code"] == "interrupted_by_execution_cancel"
        await runtime.close()


__all__ = ["ScriptedModel"]
