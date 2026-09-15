import asyncio

import pytest

from pygent import AIMessage, Context, ToolAuthorizationDecision
from pygent.runtime.codec import _tool_spec_from_dict, _tool_spec_to_dict
from pygent.tool import ToolCall, ToolKit, ToolSideEffect, tool


def test_call_wait_override_is_explicit_and_survives_codec(tmp_path):
    from dataclasses import replace

    from pygent.tool import BashTools

    bash = BashTools(workspace_root=tmp_path, timeout=25)
    spec = _tool_spec_from_dict(_tool_spec_to_dict(bash.toolkit.specs[0]))
    assert spec.wait_timeout_parameter == "timeout"
    assert spec.resolve_wait_timeout({}) == 25
    assert spec.resolve_wait_timeout({"timeout": None}) == 25
    assert spec.resolve_wait_timeout({"timeout": 1.5}) == 1.5
    assert spec.resolve_wait_timeout({"timeout": 0}) == 0
    assert spec.resolve_wait_timeout({"timeout": 50, "is_background": True}) == 0
    # Other tools' same-named parameters retain their executor meaning.
    ordinary = replace(spec, wait_timeout_parameter=None)
    assert ordinary.resolve_wait_timeout({"timeout": 1}) == 25


@pytest.mark.asyncio
async def test_native_handle_is_projected_before_output_schema_and_effect_serialization():
    from pygent.core import thaw_json
    from pygent.tool import InMemoryToolTaskManager, ToolTaskHandle

    release = asyncio.Event()

    @tool(tool_id="test.handle.work", version="1", side_effect=ToolSideEffect.READ)
    async def work() -> str:
        """Wait for completion."""
        await release.wait()
        return "done"

    work_kit = ToolKit(work)
    manager = InMemoryToolTaskManager(work_kit.build_registry())
    task = await manager.submit(work_kit.specs[0], ToolCall("inner", "work", {}))
    handle = ToolTaskHandle(manager, task.task_id)

    @tool(tool_id="test.handle.view", version="1", side_effect=ToolSideEffect.READ)
    async def view() -> str | ToolTaskHandle:
        """Return an existing live task."""
        return handle

    kit = ToolKit(view)
    layer = kit.local_layer(
        authorization_adapter=lambda req, ctx: ToolAuthorizationDecision(
            call_id=req.call.call_id, allowed=True, reason_code="allowed"
        )
    )
    try:
        message, _ = await layer.invoke(
            AIMessage(tool_calls=(ToolCall("outer", "view", {}),)),
            kit.make_visible_in(Context()),
        )
        result = message.results[0]
        assert result.status == "detached"
        assert result.task.task_id == task.task_id
        assert thaw_json(result.output) is None
        assert kit.definitions[0].output_schema["type"] == "string"
    finally:
        release.set()
        await manager.close(cancel=True)


def test_background_wait_is_portable_assembly_configuration():
    @tool(
        tool_id="test.wait",
        version="1",
        side_effect=ToolSideEffect.READ,
        wait_timeout=600,
    )
    async def work() -> str:
        """Perform a task."""
        return "done"

    toolkit = ToolKit(work, wait_timeouts={"work": 0.01})
    spec = toolkit.specs[0]
    assert spec.wait_timeout == 0.01
    assert spec.timeout is None
    assert _tool_spec_from_dict(_tool_spec_to_dict(spec)) == spec


@pytest.mark.asyncio
async def test_detached_wait_returns_early_result_or_background_same_execution():
    from pygent.tool import InMemoryToolTaskManager

    entered = asyncio.Event()
    release = asyncio.Event()
    invocations = []

    @tool(
        tool_id="test.wait",
        version="1",
        side_effect=ToolSideEffect.READ,
        wait_timeout=0.1,
    )
    async def work() -> str:
        """Perform a task."""
        invocations.append(1)
        entered.set()
        await release.wait()
        return "done"

    toolkit = ToolKit(work)
    manager = InMemoryToolTaskManager(toolkit.build_registry())
    layer = toolkit.local_layer(
        task_manager=manager,
        authorization_adapter=lambda req, ctx: ToolAuthorizationDecision(
            call_id=req.call.call_id,
            allowed=True,
            reason_code="allowed",
            lifecycle="detach",
        ),
    )
    context = toolkit.make_visible_in(Context())
    message = AIMessage(tool_calls=(ToolCall("call", "work", {}),))
    try:
        result, _ = await layer.invoke(message, context)
        assert result.results[0].status == "detached"
        task = result.results[0].task
        assert task is not None
        assert entered.is_set()
        release.set()
        final = await manager.get_result(task.task_id, wait=True)
        assert final is not None and final.output == "done"
        assert invocations == [1]
        result, _ = await layer.invoke(message, context)
        assert result.results[0].status == "succeeded"
        assert result.results[0].output == "done"
    finally:
        await manager.close(cancel=True)


@pytest.mark.asyncio
async def test_managed_wait_starts_task_without_parent_return_and_releases_capacity():
    import time

    from pygent.runtime import (
        CapacityPolicy,
        CapacityScope,
        ExecutionCapacityPolicy,
        ExecutionOptions,
        LocalRuntime,
    )
    from pygent.tool import InMemoryToolTaskManager

    @tool(
        tool_id="test.managed.wait",
        version="1",
        side_effect=ToolSideEffect.READ,
        wait_timeout=1,
    )
    async def work() -> str:
        """Complete under independent ownership."""
        await asyncio.sleep(0)
        return "managed done"

    toolkit = ToolKit(work)
    registry = toolkit.build_registry()
    runtime = LocalRuntime()
    runtime.attach_executor_registry(registry)
    runtime.attach_tool_task_manager(InMemoryToolTaskManager(registry))
    binding = runtime.create_binding(
        name="background-test",
        execution_capacity=ExecutionCapacityPolicy(
            scope=CapacityScope.RUNTIME_INSTANCE,
            max_live_executions=4,
            max_runnable_executions=1,
            max_queue_size=4,
            max_waiters=4,
            max_child_depth=8,
            max_children_per_execution=16,
        ),
        model_capacity=CapacityPolicy.passthrough(),
        tool_capacity=CapacityPolicy.limited(max_concurrency=1, max_queue_size=4),
    )
    layer = toolkit.local_layer(
        authorization_adapter=lambda req, ctx: ToolAuthorizationDecision(
            call_id=req.call.call_id,
            allowed=True,
            reason_code="allowed",
            lifecycle="detach",
        ),
    )
    try:
        result, _ = await binding.bind(layer).invoke(
            AIMessage(tool_calls=(ToolCall("call", "work", {}),)),
            toolkit.make_visible_in(Context()),
            execution=ExecutionOptions(deadline=time.monotonic() + 3),
        )
        assert result.results[0].status == "succeeded"
        assert result.results[0].output == "managed done"
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_managed_bash_controls_work_when_shell_capacity_is_full(tmp_path):
    import time

    from pygent.runtime import (
        CapacityPolicy,
        CapacityScope,
        ExecutionCapacityPolicy,
        ExecutionOptions,
        LocalRuntime,
    )
    from pygent.tool import (
        InMemoryToolTaskManager,
        LocalToolExecutor,
        SandboxExecutorSupport,
    )
    from pygent.tool.standard import BashTools

    class WorkspaceExecutor(LocalToolExecutor):
        sandbox_support = SandboxExecutorSupport(profiles=("workspace",))

    bash = BashTools(workspace_root=tmp_path, timeout=600)
    runtime = LocalRuntime()
    kit = bash.toolkit
    registry = kit.build_registry()
    runtime.attach_executor_registry(registry)
    manager = InMemoryToolTaskManager(registry)
    runtime.attach_tool_task_manager(manager)
    layer = kit.managed_layer(
        runtime,
        executor_factory=lambda spec, handler: WorkspaceExecutor(handler),
        replace_existing=True,
        authorization_adapter=lambda req, ctx: ToolAuthorizationDecision(
            call_id=req.call.call_id,
            allowed=True,
            reason_code="allowed",
            lifecycle="detach" if req.spec.wait_timeout is not None else "sync",
        ),
    )
    binding = runtime.create_binding(
        name="bash-controls",
        execution_capacity=ExecutionCapacityPolicy(
            scope=CapacityScope.RUNTIME_INSTANCE,
            max_live_executions=4,
            max_runnable_executions=1,
            max_queue_size=4,
            max_waiters=4,
            max_child_depth=8,
            max_children_per_execution=16,
        ),
        model_capacity=CapacityPolicy.passthrough(),
        tool_capacity=CapacityPolicy.limited(max_concurrency=1, max_queue_size=4),
    )
    bound = binding.bind(layer)
    context = kit.make_visible_in(Context())

    async def invoke(name, arguments):
        message, _ = await bound.invoke(
            AIMessage(tool_calls=(ToolCall(name, name, arguments),)),
            context,
            execution=ExecutionOptions(deadline=time.monotonic() + 5),
        )
        return message.results[0]

    try:
        for invalid in (-1, True, "2"):
            rejected = await invoke("bash", {"command": "touch should-not-exist", "timeout": invalid})
            assert rejected.status == "rejected"
            assert rejected.task is None
        assert not (tmp_path / "should-not-exist").exists()
        result = await invoke("bash", {"command": "printf ready; sleep 30", "timeout": 0.05})
        assert result.status == "detached", result
        assert result.task is not None
        task_id = result.task.task_id
        for _ in range(100):
            if await manager.get_output(task_id):
                break
            await asyncio.sleep(0.02)
        assert await manager.get_output(task_id)
        queried = await invoke("tool_task_get", {"task_id": task_id})
        assert queried.status == "succeeded", queried
        stopped = await invoke("tool_task_stop", {"task_id": task_id})
        assert stopped.status == "succeeded", stopped
        final = await manager.get_result(task_id, wait=True)
        assert final is not None and final.status in ("unknown", "cancelled")
    finally:
        await runtime.close()
        await bash.aclose()


@pytest.mark.asyncio
async def test_real_bash_result_can_be_queried_after_history_reopens(tmp_path):
    from pygent.runtime import SQLiteHistoryStore
    from pygent.runtime.tasks import DurableToolTaskManager
    from pygent.tool import ExecutorRegistry
    from pygent.tool.standard import BashTools

    path = tmp_path / "bash-history.db"
    async with SQLiteHistoryStore(path) as history:
        manager = DurableToolTaskManager(history, ExecutorRegistry())
        async with BashTools(workspace_root=tmp_path, task_manager=manager) as bash:
            handle = await bash.bash("printf persisted", is_background=True)
            result = await handle.result()
            assert result.status == "succeeded"
            assert "persisted" in result.output
            task_id = handle.task_id
        await manager.close()

    async with SQLiteHistoryStore(path) as history:
        manager = DurableToolTaskManager(history, ExecutorRegistry())
        async with BashTools(workspace_root=tmp_path, task_manager=manager) as bash:
            snapshot = await bash.tool_task_get(task_id)
            assert snapshot["task"]["state"] == "succeeded"
            assert "persisted" in snapshot["output"]
            assert snapshot["result"]["status"] == "succeeded"
            assert (await bash.tool_task_stop(task_id))["cancel_requested"] is False
        await manager.close()
