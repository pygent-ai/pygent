import asyncio
from contextlib import asynccontextmanager

import pytest

from pygent import (
    AIMessage,
    Context,
    Module,
    UserMessage,
)
from pygent.core import (
    EffectDisposition,
    EffectOutcome,
)
from pygent.core._module_contracts import _execution_scope
from pygent.runtime import (
    CapacityPolicy,
    CapacityScope,
    ExecutionCapacityPolicy,
    LocalRuntime,
)
from pygent.tool import (
    AgentToolExecutor,
    ExecutorRegistry,
    IdempotencyPolicy,
    InMemoryToolTaskManager,
    LocalToolExecutor,
    ToolAuthorizationDecision,
    ToolAuthorizationRequest,
    ToolCall,
    ToolCallLayer,
    ToolDefinition,
    ToolExecutionError,
    ToolSideEffect,
    ToolSpec,
)


class Authorization(Module[ToolAuthorizationRequest, ToolAuthorizationDecision]):
    def __init__(self, *, detach: bool = False) -> None:
        super().__init__()
        self.detach = detach

    async def forward(self, message, context):
        allowed = "tool:use" in message.permissions
        return (
            ToolAuthorizationDecision(
                call_id=message.call.call_id,
                allowed=allowed,
                reason_code="allowed" if allowed else "missing_permission",
                lifecycle="detach" if self.detach else "sync",
            ),
            context,
        )


def spec(*, side_effect=ToolSideEffect.PURE) -> ToolSpec:
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
        side_effect=side_effect,
    )


def context(tool: ToolSpec) -> Context:
    return Context(
        tools=(tool.definition,),
        metadata={"permissions": ["tool:use"]},
    )


@pytest.mark.asyncio
async def test_local_execution_validates_and_preserves_order() -> None:
    tool = spec()
    registry = ExecutorRegistry()

    async def double(arguments):
        await asyncio.sleep(0.02 if arguments["value"] == 1 else 0)
        return arguments["value"] * 2

    registry.register(tool.tool_id, tool.version, LocalToolExecutor(double))
    layer = ToolCallLayer(
        tools=(tool,),
        authorization=Authorization(),
        executor_registry=registry,
        max_concurrency=2,
    )
    message, returned = await layer.invoke(
        AIMessage(
            tool_calls=(
                ToolCall(call_id="first", name="double", arguments={"value": 1}),
                ToolCall(call_id="second", name="double", arguments={"value": 2}),
            )
        ),
        context(tool),
    )
    assert returned == context(tool)
    assert [result.call_id for result in message.results] == ["first", "second"]
    assert [result.output for result in message.results] == [2, 4]
    assert all(result.task is not None for result in message.results)


@pytest.mark.asyncio
async def test_requires_key_is_rejected_direct_and_injected_managed() -> None:
    tool = ToolSpec(
        tool_id="external.write",
        version="1",
        definition=ToolDefinition(
            name="write",
            description="write once",
            parameters={"type": "object"},
        ),
        side_effect=ToolSideEffect.WRITE,
        idempotency=IdempotencyPolicy.REQUIRES_KEY,
    )
    observed: list[str | None] = []
    registry = ExecutorRegistry()

    class CaptureExecutor:
        async def execute(self, spec, call, execution_context):
            observed.append(call.idempotency_key)
            return {"ok": True}

    registry.register(tool.tool_id, tool.version, CaptureExecutor())
    layer = ToolCallLayer(
        tools=(tool,),
        authorization=Authorization(),
        executor_registry=registry,
    )
    input_message = AIMessage(
        tool_calls=(ToolCall(call_id="write-1", name="write", arguments={}),)
    )
    direct, _ = await layer.invoke(input_message, context(tool))
    assert direct.results[0].error_code == "idempotency_key_required"
    assert observed == []

    runtime = LocalRuntime()
    managed, _ = await runtime.bind(layer).invoke(input_message, context(tool))
    await runtime.close()
    assert managed.results[0].status == "succeeded"
    assert observed and observed[0].endswith(":root:0:write-1")


@pytest.mark.asyncio
async def test_rejection_happens_before_task_admission() -> None:
    tool = spec()
    layer = ToolCallLayer(tools=(tool,), authorization=Authorization())
    message, _ = await layer.invoke(
        AIMessage(
            tool_calls=(ToolCall(call_id="x", name="double", arguments={"value": 1}),)
        ),
        Context(tools=(tool.definition,)),
    )
    result = message.results[0]
    assert result.status == "rejected"
    assert result.task is None
    assert result.error_code == "missing_permission"


@pytest.mark.asyncio
async def test_bad_arguments_are_rejected_before_authorization() -> None:
    tool = spec()
    message, _ = await ToolCallLayer(
        tools=(tool,), authorization=Authorization()
    ).invoke(
        AIMessage(
            tool_calls=(ToolCall(call_id="x", name="double", arguments={"value": "1"}),)
        ),
        context(tool),
    )
    assert message.results[0].status == "rejected"
    assert message.results[0].error_code == "invalid_arguments"
    assert message.results[0].task is None


@pytest.mark.asyncio
async def test_unknown_side_effect_is_not_reported_uncommitted() -> None:
    tool = spec(side_effect=ToolSideEffect.EXTERNAL)
    registry = ExecutorRegistry()

    async def fail(_arguments):
        raise ToolExecutionError(
            "connection lost after send",
            kind="transport_error",
            retryable=True,
            side_effect_committed=None,
        )

    registry.register(tool.tool_id, tool.version, LocalToolExecutor(fail))
    message, _ = await ToolCallLayer(
        tools=(tool,),
        authorization=Authorization(),
        executor_registry=registry,
    ).invoke(
        AIMessage(
            tool_calls=(ToolCall(call_id="x", name="double", arguments={"value": 1}),)
        ),
        context(tool),
    )
    result = message.results[0]
    assert result.status == "unknown"
    assert result.side_effect_committed is None
    assert result.retryable


@pytest.mark.asyncio
async def test_unclassified_external_failure_has_unknown_commit_state() -> None:
    tool = spec(side_effect=ToolSideEffect.EXTERNAL)
    registry = ExecutorRegistry()

    async def fail(_arguments):
        raise RuntimeError("handler failed after an opaque external operation")

    registry.register(tool.tool_id, tool.version, LocalToolExecutor(fail))
    message, _ = await ToolCallLayer(
        tools=(tool,),
        authorization=Authorization(),
        executor_registry=registry,
    ).invoke(
        AIMessage(
            tool_calls=(ToolCall(call_id="x", name="double", arguments={"value": 1}),)
        ),
        context(tool),
    )

    result = message.results[0]
    assert result.status == "unknown"
    assert result.side_effect_committed is None
    assert "opaque external operation" not in (result.error or "")


@pytest.mark.asyncio
async def test_explicit_detach_returns_queryable_task() -> None:
    tool = spec()
    registry = ExecutorRegistry()
    registry.register(
        tool.tool_id,
        tool.version,
        LocalToolExecutor(lambda arguments: arguments["value"] * 2),
    )
    manager = InMemoryToolTaskManager(registry)
    message, _ = await ToolCallLayer(
        tools=(tool,),
        authorization=Authorization(detach=True),
        task_manager=manager,
    ).invoke(
        AIMessage(
            tool_calls=(ToolCall(call_id="x", name="double", arguments={"value": 2}),)
        ),
        context(tool),
    )
    detached = message.results[0]
    assert detached.status == "detached"
    assert detached.task is not None
    final = await manager.get_result(detached.task.task_id, wait=True)
    assert final is not None
    assert final.status == "succeeded"
    assert final.output == 4


@pytest.mark.asyncio
async def test_managed_detach_uses_runtime_task_manager_for_query_result_and_cancel() -> None:
    tool = spec()
    registry = ExecutorRegistry()
    blocking_started = asyncio.Event()

    async def execute(arguments):
        if arguments["value"] == 0:
            blocking_started.set()
            await asyncio.Event().wait()
        return arguments["value"] * 2

    registry.register(tool.tool_id, tool.version, LocalToolExecutor(execute))
    manager = InMemoryToolTaskManager(registry)
    runtime = LocalRuntime()
    layer = ToolCallLayer(
        tools=(tool,),
        authorization=Authorization(detach=True),
    )
    bound = runtime.bind(layer)
    runtime.attach_tool_task_manager(manager)

    unavailable_message, _ = await bound.invoke(
        AIMessage(
            tool_calls=(
                ToolCall(
                    call_id="managed-unavailable",
                    name="double",
                    arguments={"value": 2},
                ),
            )
        ),
        context(tool),
    )
    assert unavailable_message.results[0].error_code == "detach_unavailable"

    runtime.attach_executor_registry(registry)

    succeeded_message, _ = await bound.invoke(
        AIMessage(
            tool_calls=(
                ToolCall(call_id="managed-ok", name="double", arguments={"value": 2}),
            )
        ),
        context(tool),
    )
    succeeded_task = succeeded_message.results[0].task
    assert succeeded_task is not None
    assert await runtime.get_tool_task(succeeded_task.task_id) is not None
    succeeded = await runtime.get_tool_result(succeeded_task.task_id, wait=True)
    assert succeeded is not None
    assert succeeded.status == "succeeded"
    assert succeeded.output == 4

    cancelled_message, _ = await bound.invoke(
        AIMessage(
            tool_calls=(
                ToolCall(
                    call_id="managed-cancel",
                    name="double",
                    arguments={"value": 0},
                ),
            )
        ),
        context(tool),
    )
    cancelled_task = cancelled_message.results[0].task
    assert cancelled_task is not None
    await blocking_started.wait()
    assert await runtime.cancel_tool_task(cancelled_task.task_id)
    cancelled = await runtime.get_tool_result(cancelled_task.task_id)
    assert cancelled is not None
    assert cancelled.status == "cancelled"

    await runtime.close()


@pytest.mark.asyncio
async def test_detached_task_can_be_cancelled_before_handler_finishes() -> None:
    tool = spec()
    registry = ExecutorRegistry()

    async def wait_forever(_arguments):
        await asyncio.Event().wait()

    registry.register(tool.tool_id, tool.version, LocalToolExecutor(wait_forever))
    manager = InMemoryToolTaskManager(registry)
    task = await manager.submit(
        tool,
        ToolCall(call_id="cancel-me", name="double", arguments={"value": 2}),
    )
    assert await manager.cancel(task.task_id)
    result = await manager.get_result(task.task_id)
    assert result is not None
    assert result.status == "cancelled"
    assert result.side_effect_committed is None


@pytest.mark.asyncio
async def test_durable_effect_replay_preserves_task_identity_and_skips_executor() -> None:
    tool = spec()
    registry = ExecutorRegistry()
    calls = 0

    async def double(arguments):
        nonlocal calls
        calls += 1
        return arguments["value"] * 2

    registry.register(tool.tool_id, tool.version, LocalToolExecutor(double))
    layer = ToolCallLayer(
        tools=(tool,),
        authorization=Authorization(),
        executor_registry=registry,
    )

    class EffectScope:
        deadline = None

        def __init__(self) -> None:
            self.effect = None

        async def invoke_module(self, module, message, current_context):
            return await module.forward(message, current_context)

        async def emit_event(self, module, kind, data):
            return None

        @asynccontextmanager
        async def model_permit(self):
            yield

        @asynccontextmanager
        async def tool_permit(self, resource_key=None):
            yield

        async def execute_effect(self, *, spec, request, operation):
            if self.effect is None:
                self.effect = await operation()
                disposition = EffectDisposition.EXECUTED
            else:
                disposition = EffectDisposition.REPLAYED
            return EffectOutcome(
                value=self.effect,
                disposition=disposition,
                effect_id="test-effect",
            )

    scope = EffectScope()
    token = _execution_scope.set(scope)
    try:
        message = AIMessage(
            tool_calls=(ToolCall(call_id="x", name="double", arguments={"value": 2}),)
        )
        first, _ = await layer.forward(message, context(tool))
        second, _ = await layer.forward(message, context(tool))
    finally:
        _execution_scope.reset(token)

    assert calls == 1
    assert first.results[0].output == second.results[0].output == 4
    assert first.results[0].task == second.results[0].task


@pytest.mark.asyncio
async def test_durable_effect_replays_classified_executor_failure() -> None:
    tool = spec(side_effect=ToolSideEffect.EXTERNAL)
    registry = ExecutorRegistry()
    calls = 0

    async def fail(_arguments):
        nonlocal calls
        calls += 1
        raise ToolExecutionError(
            "request outcome is uncertain",
            kind="transport_error",
            retryable=True,
            side_effect_committed=None,
        )

    registry.register(tool.tool_id, tool.version, LocalToolExecutor(fail))
    layer = ToolCallLayer(
        tools=(tool,),
        authorization=Authorization(),
        executor_registry=registry,
    )

    class EffectScope:
        deadline = None

        def __init__(self) -> None:
            self.effect = None

        async def invoke_module(self, module, message, current_context):
            return await module.forward(message, current_context)

        async def emit_event(self, module, kind, data):
            return None

        @asynccontextmanager
        async def model_permit(self):
            yield

        @asynccontextmanager
        async def tool_permit(self, resource_key=None):
            yield

        async def execute_effect(self, *, spec, request, operation):
            if self.effect is None:
                self.effect = await operation()
                disposition = EffectDisposition.EXECUTED
            else:
                disposition = EffectDisposition.REPLAYED
            return EffectOutcome(
                value=self.effect,
                disposition=disposition,
                effect_id="test-effect",
            )

    scope = EffectScope()
    token = _execution_scope.set(scope)
    try:
        message = AIMessage(
            tool_calls=(ToolCall(call_id="x", name="double", arguments={"value": 2}),)
        )
        first, _ = await layer.forward(message, context(tool))
        second, _ = await layer.forward(message, context(tool))
    finally:
        _execution_scope.reset(token)

    assert calls == 1
    assert first.results[0].status == second.results[0].status == "unknown"
    assert first.results[0].task == second.results[0].task


@pytest.mark.asyncio
async def test_managed_scope_can_supply_deployment_executor_registry() -> None:
    tool = spec()
    registry = ExecutorRegistry()
    registry.register(
        tool.tool_id,
        tool.version,
        LocalToolExecutor(lambda arguments: arguments["value"] * 2),
    )
    layer = ToolCallLayer(tools=(tool,), authorization=Authorization())

    class ManagedScope:
        deadline = None

        async def invoke_module(self, module, message, current_context):
            return await module.forward(message, current_context)

        async def emit_event(self, module, kind, data):
            return None

        def resolve_tool_registry(self):
            return registry

        @asynccontextmanager
        async def model_permit(self):
            yield

        @asynccontextmanager
        async def tool_permit(self, resource_key=None):
            yield

        async def execute_effect(self, *, spec, request, operation):
            return EffectOutcome(
                value=await operation(),
                disposition=EffectDisposition.EXECUTED,
                effect_id="test-effect",
            )

    token = _execution_scope.set(ManagedScope())
    try:
        result, _ = await layer.forward(
            AIMessage(
                tool_calls=(
                    ToolCall(call_id="managed", name="double", arguments={"value": 3}),
                )
            ),
            context(tool),
        )
    finally:
        _execution_scope.reset(token)

    assert result.results[0].status == "succeeded"
    assert result.results[0].output == 6


def _managed_binding(runtime: LocalRuntime, name: str, *, runnable: int = 1):
    return runtime.create_binding(
        name=name,
        execution_capacity=ExecutionCapacityPolicy(
            scope=CapacityScope.RUNTIME_INSTANCE,
            max_live_executions=4,
            max_runnable_executions=runnable,
            max_queue_size=4,
            max_waiters=2,
            max_child_depth=8,
            max_children_per_execution=16,
        ),
        model_capacity=CapacityPolicy.passthrough(),
        tool_capacity=CapacityPolicy.passthrough(),
    )


def _allow_agent_tool(*, detach: bool = False):
    def authorize(request, _context):
        return ToolAuthorizationDecision(
            call_id=request.call.call_id,
            allowed=True,
            reason_code="allowed",
            lifecycle="detach" if detach else "sync",
        )

    return authorize


class _BlockingAgent(Module[UserMessage, AIMessage]):
    trusted_live_resource_attributes = (
        "_entered",
        "_release",
        "_cancelled",
    )

    def __init__(self) -> None:
        super().__init__()
        self._entered = asyncio.Event()
        self._release = asyncio.Event()
        self._cancelled = asyncio.Event()

    @property
    def entered(self) -> asyncio.Event:
        return self._entered

    @property
    def release(self) -> asyncio.Event:
        return self._release

    @property
    def cancelled(self) -> asyncio.Event:
        return self._cancelled

    async def forward(self, message, context):
        self._entered.set()
        try:
            await self._release.wait()
        except asyncio.CancelledError:
            self._cancelled.set()
            raise
        return AIMessage(content=str(int(message.content) * 2)), context


def _agent_executor(agent) -> AgentToolExecutor:
    return AgentToolExecutor(
        agent=agent,
        request_builder=lambda _spec, call: (
            UserMessage(content=str(call.arguments["value"])),
            Context(),
        ),
        result_builder=lambda message, _context: int(message.content),
    )


@pytest.mark.asyncio
async def test_managed_sync_agent_tool_is_structured_child_and_hands_off_lease():
    tool = spec()
    agent = _BlockingAgent()
    registry = ExecutorRegistry()
    registry.register(tool.tool_id, tool.version, _agent_executor(agent))
    layer = ToolCallLayer(
        tools=(tool,),
        authorization_adapter=_allow_agent_tool(),
        executor_registry=registry,
    )
    runtime = LocalRuntime()
    handle = await _managed_binding(runtime, "agent-tool-sync").bind(layer).start(
        AIMessage(
            tool_calls=(
                ToolCall(call_id="agent-sync", name="double", arguments={"value": 3}),
            )
        ),
        context(tool),
    )

    # max_runnable_executions is one. Reaching the Agent proves the Tool parent
    # released its lease while the structured Child acquired it.
    await asyncio.wait_for(agent.entered.wait(), timeout=1)
    agent.release.set()
    message, _ = await handle.result()
    async with handle.subscribe() as subscription:
        events = [event async for event in subscription]
    await runtime.close()

    assert message.results[0].output == 6
    assert [event.kind for event in events].count("span.started") >= 2
    assert [event.kind for event in events].count("span.completed") >= 2
    assert any(event.parent_span_id is not None for event in events)


@pytest.mark.asyncio
async def test_managed_sync_agent_tool_cancel_propagates_and_joins_child():
    tool = spec()
    agent = _BlockingAgent()
    registry = ExecutorRegistry()
    registry.register(tool.tool_id, tool.version, _agent_executor(agent))
    runtime = LocalRuntime()
    bound = _managed_binding(runtime, "agent-tool-cancel").bind(
        ToolCallLayer(
            tools=(tool,),
            authorization_adapter=_allow_agent_tool(),
            executor_registry=registry,
        )
    )
    handle = await bound.start(
        AIMessage(
            tool_calls=(
                ToolCall(
                    call_id="agent-cancel", name="double", arguments={"value": 3}
                ),
            )
        ),
        context(tool),
    )
    await agent.entered.wait()

    assert await handle.cancel()
    await agent.cancelled.wait()
    async with handle.subscribe() as subscription:
        events = [event async for event in subscription]
    await runtime.close()

    assert "span.cancelled" in [event.kind for event in events]


@pytest.mark.asyncio
async def test_detached_agent_tool_starts_independent_root_not_child():
    tool = spec()
    agent = _BlockingAgent()
    runtime = LocalRuntime()
    binding = _managed_binding(runtime, "agent-tool-detach", runnable=1)
    managed_agent = binding.bind(agent)
    registry = ExecutorRegistry()
    registry.register(tool.tool_id, tool.version, _agent_executor(managed_agent))
    manager = InMemoryToolTaskManager(registry)
    runtime.attach_executor_registry(registry)
    runtime.attach_tool_task_manager(manager)
    bound = binding.bind(
        ToolCallLayer(
            tools=(tool,),
            authorization_adapter=_allow_agent_tool(detach=True),
        )
    )
    handle = await bound.start(
        AIMessage(
            tool_calls=(
                ToolCall(
                    call_id="agent-detach", name="double", arguments={"value": 4}
                ),
            )
        ),
        context(tool),
    )

    try:
        message, _ = await asyncio.wait_for(handle.result(), timeout=1)
        detached = message.results[0]
        assert detached.status == "detached"
        assert detached.task is not None
        await asyncio.wait_for(agent.entered.wait(), timeout=1)
        async with handle.subscribe() as subscription:
            parent_events = [event async for event in subscription]
        assert not any(event.kind.startswith("child.") for event in parent_events)

        agent.release.set()
        final = await asyncio.wait_for(
            runtime.get_tool_result(detached.task.task_id, wait=True), timeout=1
        )
    finally:
        agent.release.set()
        await runtime.close()

    assert final is not None
    assert final.status == "succeeded"
    assert final.output == 8
