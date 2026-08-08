import asyncio

import pytest

from pygent import (
    AIMessage,
    Context,
    ToolAuthorizationDecision,
    ToolCall,
    ToolCallLayer,
    ToolDefinition,
    ToolSideEffect,
    ToolSpec,
)
from pygent.runtime import LocalRuntime
from pygent.tool import (
    ExecutorRegistry,
    InMemoryToolTaskManager,
    SandboxExecutorSupport,
    ToolExecutionContext,
    ToolExecutionError,
)
from pygent.tool.executors import validate_executor_sandbox


def _spec(profile: str = "workspace-write") -> ToolSpec:
    return ToolSpec(
        tool_id="shell.command",
        version="1",
        definition=ToolDefinition(
            name="bash",
            description="run a command",
            parameters={"type": "object"},
        ),
        side_effect=ToolSideEffect.WRITE,
        sandbox_profile=profile,
    )


class _SandboxExecutor:
    def __init__(
        self,
        *,
        profiles: tuple[str, ...] = ("workspace-write",),
        durable: bool = False,
        fingerprint: str | None = None,
    ) -> None:
        self.sandbox_support = SandboxExecutorSupport(
            profiles=profiles,
            durable_reconnect=durable,
            deployment_fingerprint=fingerprint,
        )
        self.contexts: list[ToolExecutionContext] = []

    async def execute(self, spec, call, context):
        await asyncio.sleep(0)
        self.contexts.append(context)
        return {"call_id": call.call_id}


def test_runtime_derives_sandbox_capability_from_registered_executor() -> None:
    spec = _spec()
    registry = ExecutorRegistry()
    runtime = LocalRuntime()
    runtime.attach_executor_registry(registry)
    runtime.register_tool(spec, _SandboxExecutor())

    assert "tool.sandbox.workspace-write" in runtime.capabilities
    assert registry.resolve(spec.tool_id, spec.version)


def test_runtime_rejects_manual_sandbox_capability_and_incompatible_executor() -> None:
    with pytest.raises(ValueError, match="derived from registered executors"):
        LocalRuntime(capabilities=("tool.sandbox.workspace-write",))

    runtime = LocalRuntime()
    runtime.attach_executor_registry(ExecutorRegistry())
    with pytest.raises(ValueError, match="requires tool.sandbox.workspace-write"):
        runtime.register_tool(_spec(), _SandboxExecutor(profiles=("network-read",)))


@pytest.mark.asyncio
async def test_registry_enforces_sandbox_only_for_managed_execution() -> None:
    spec = _spec()
    call = ToolCall("call-1", "bash", {})

    class LocalExecutor:
        async def execute(self, spec, call, context):
            return "local"

    registry = ExecutorRegistry()
    registry.register(spec.tool_id, spec.version, LocalExecutor())
    assert await registry.execute(spec, call, ToolExecutionContext()) == "local"

    with pytest.raises(ToolExecutionError) as caught:
        await registry.execute(
            spec,
            call,
            ToolExecutionContext(execution_id="execution-1", task_id="task-1"),
        )
    assert caught.value.code == "missing_sandbox_capability"
    assert caught.value.missing_capabilities == (
        "tool.sandbox.workspace-write",
    )


@pytest.mark.asyncio
async def test_bind_reports_gap_and_managed_result_keeps_exact_capability() -> None:
    spec = _spec()

    class LocalExecutor:
        async def execute(self, spec, call, context):
            return "local"

    registry = ExecutorRegistry()
    registry.register(spec.tool_id, spec.version, LocalExecutor())
    layer = ToolCallLayer(
        tools=(spec,),
        executor_registry=registry,
        authorization_adapter=lambda request, context: ToolAuthorizationDecision(
            call_id=request.call.call_id,
            allowed=True,
            reason_code="allowed",
        ),
    )
    runtime = LocalRuntime()
    bound = runtime.bind(layer)
    assert bound.durability.detached_tool_gaps == (
        "root:shell.command@1 missing tool.sandbox.workspace-write",
        "root:shell.command@1 detached missing tool.sandbox.workspace-write",
    )

    message, _ = await bound.invoke(
        AIMessage(tool_calls=(ToolCall("call-1", "bash", {}),)),
        Context(tools=(spec.definition,)),
    )
    result = message.results[0]
    assert result.status == "failed"
    assert result.error_kind == "capability_error"
    assert result.error_code == "missing_sandbox_capability"
    assert result.missing_capabilities == ("tool.sandbox.workspace-write",)
    await runtime.close()


def test_durable_sandbox_requires_reconnect_and_stable_fingerprint() -> None:
    spec = _spec()
    with pytest.raises(ToolExecutionError) as caught:
        validate_executor_sandbox(spec, _SandboxExecutor(), durable=True)
    assert caught.value.code == "sandbox_reconnect_unavailable"

    first = _SandboxExecutor(durable=True, fingerprint="e2b:template:v1")
    support = validate_executor_sandbox(spec, first, durable=True)
    assert support is not None
    expected = support.capability_for_fingerprint()
    assert expected is not None
    with pytest.raises(ToolExecutionError) as changed:
        validate_executor_sandbox(
            spec,
            _SandboxExecutor(durable=True, fingerprint="e2b:template:v2"),
            durable=True,
            required_capabilities=(expected,),
        )
    assert changed.value.code == "sandbox_deployment_changed"


@pytest.mark.asyncio
async def test_detached_context_is_isolated_per_concurrent_task() -> None:
    spec = _spec()
    executor = _SandboxExecutor()
    registry = ExecutorRegistry()
    registry.register(spec.tool_id, spec.version, executor)
    manager = InMemoryToolTaskManager(registry)

    tasks = await asyncio.gather(
        *(
            manager.submit(spec, ToolCall(f"call-{index}", "bash", {}))
            for index in range(8)
        )
    )
    await asyncio.gather(
        *(manager.get_result(task.task_id, wait=True) for task in tasks)
    )

    assert {context.task_id for context in executor.contexts} == {
        task.task_id for task in tasks
    }
    assert all(context.recovery is False for context in executor.contexts)
    await manager.close()
