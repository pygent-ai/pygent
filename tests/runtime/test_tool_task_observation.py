import asyncio

import pytest

from pygent.runtime import SQLiteHistoryStore
from pygent.runtime.tasks import DurableToolTaskManager
from pygent.tool import (
    ExecutorRegistry,
    InMemoryToolTaskManager,
    ToolCall,
    ToolDefinition,
    ToolSpec,
    executors,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("durable", [False, True])
async def test_observer_timeout_does_not_cancel_execution_and_output_survives(
    tmp_path, durable
):
    registry = ExecutorRegistry()
    async with SQLiteHistoryStore(tmp_path / "tasks.db") as history:
        manager = (
            DurableToolTaskManager(history, registry)
            if durable
            else InMemoryToolTaskManager(registry)
        )
        entered = asyncio.Event()
        release = asyncio.Event()

        async def execute(spec, call, context):
            assert executors.current_tool_execution() is context
            await context.publish_output({"stdout": "partial"})
            entered.set()
            await release.wait()
            return {"stdout": "complete"}

        spec = ToolSpec(
            tool_id="test",
            version="1",
            definition=ToolDefinition(
                name="test", description="Test", parameters={"type": "object"}
            ),
        )
        task = await manager.submit(
            spec, ToolCall(call_id="c", name="test", arguments={}), execution=execute
        )
        # Missing context publication must fail promptly, rather than leave this test hanging.
        await asyncio.wait_for(asyncio.shield(entered.wait()), 0.5)
        from pygent.tool.task_handle import ToolTaskHandle

        handle = ToolTaskHandle(manager, task.task_id)
        assert await handle.wait(0.001) is None
        assert (await handle.snapshot()).state.value == "running"
        assert (await manager.get_output(task.task_id))["stdout"] == "partial"
        release.set()
        result = await handle.result()
        assert result.status == "succeeded"
        assert (await manager.get_output(task.task_id))["stdout"] == "complete"
        await manager.close()
    if durable:
        async with SQLiteHistoryStore(tmp_path / "tasks.db") as history:
            manager = DurableToolTaskManager(history, registry)
            assert (await manager.get_output(task.task_id))["stdout"] == "complete"
            assert (await manager.get_result(task.task_id)).status == "succeeded"
            await manager.close()


@pytest.mark.asyncio
async def test_persistent_query_distinguishes_live_other_owner_from_expired_owner(
    tmp_path,
):
    registry = ExecutorRegistry()
    async with SQLiteHistoryStore(tmp_path / "owners.db") as history:
        first = DurableToolTaskManager(history, registry)
        second = DurableToolTaskManager(history, registry)
        spec = ToolSpec(
            tool_id="test",
            version="1",
            definition=ToolDefinition(
                name="test", description="Test", parameters={"type": "object"}
            ),
        )
        task = await first.prepare(
            spec, ToolCall(call_id="c", name="test", arguments={})
        )
        await first._publish_output(task.task_id, {"stdout": "before exit"})
        assert (await second.get_task(task.task_id)).state.value == "pending"
        # An expired persisted lease models an owner whose heartbeat has stopped.
        await history.renew_tool_observations(first._owner_id, -1)
        restored = await second.get_task(task.task_id)
        assert restored.state.value == "unknown"
        result = await second.get_result(task.task_id)
        assert result.status == "unknown"
        assert (await second.get_output(task.task_id))["stdout"] == "before exit"
        await first.close()
        await second.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("durable", [False, True])
async def test_cancelling_before_start_has_terminal_result(tmp_path, durable):
    registry = ExecutorRegistry()
    async with SQLiteHistoryStore(tmp_path / "cancel.db") as history:
        manager = (
            DurableToolTaskManager(history, registry)
            if durable
            else InMemoryToolTaskManager(registry)
        )
        spec = ToolSpec(
            tool_id="test",
            version="1",
            definition=ToolDefinition(
                name="test", description="Test", parameters={"type": "object"}
            ),
        )
        task = await manager.prepare(
            spec, ToolCall(call_id="c", name="test", arguments={})
        )
        assert await manager.cancel(task.task_id)
        await manager.start(task.task_id)
        assert (await manager.get_result(task.task_id)).status == "cancelled"
        await manager.close()


@pytest.mark.asyncio
async def test_expired_owner_cannot_overwrite_unknown_terminal(tmp_path):
    from pygent.runtime import HistoryConflictError

    registry = ExecutorRegistry()
    async with SQLiteHistoryStore(tmp_path / "fence.db") as history:
        first = DurableToolTaskManager(history, registry)
        second = DurableToolTaskManager(history, registry)
        spec = ToolSpec(
            tool_id="test",
            version="1",
            definition=ToolDefinition(
                name="test", description="Test", parameters={"type": "object"}
            ),
        )
        task = await first.prepare(
            spec, ToolCall(call_id="c", name="test", arguments={})
        )
        await history.renew_tool_observations(first._owner_id, -1)
        assert (await second.get_task(task.task_id)).state.value == "unknown"
        from pygent.tool import ToolResult

        with pytest.raises(HistoryConflictError):
            await first._store_terminal(
                task.task_id,
                spec,
                ToolCall(call_id="c", name="test", arguments={}),
                ToolResult(call_id="c", name="test", status="failed", task=task),
            )
        assert (await second.get_result(task.task_id)).status == "unknown"
        await first.close()
        await second.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("durable", [False, True])
async def test_task_runner_rejects_invalid_output_schema(tmp_path, durable):
    registry = ExecutorRegistry()
    async with SQLiteHistoryStore(tmp_path / "schema.db") as history:
        manager = (
            DurableToolTaskManager(history, registry)
            if durable
            else InMemoryToolTaskManager(registry)
        )
        spec = ToolSpec(
            tool_id="test",
            version="1",
            definition=ToolDefinition(
                name="test",
                description="Test",
                parameters={"type": "object"},
                output_schema={"type": "integer"},
            ),
        )

        async def execute(spec, call, context):
            return "wrong"

        task = await manager.submit(
            spec, ToolCall(call_id="c", name="test", arguments={}), execution=execute
        )
        result = await manager.get_result(task.task_id, wait=True)
        assert result.status == "failed"
        assert result.error_kind == "validation_error"
        assert result.error_code == "invalid_output"
        assert result.side_effect_committed is True
        await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("durable", [False, True])
async def test_close_cancels_admitted_tasks_before_execution_enters(tmp_path, durable):
    registry = ExecutorRegistry()
    async with SQLiteHistoryStore(tmp_path / "close.db") as history:
        manager = (
            DurableToolTaskManager(history, registry)
            if durable
            else InMemoryToolTaskManager(registry)
        )
        spec = ToolSpec(
            tool_id="test",
            version="1",
            definition=ToolDefinition(
                name="test", description="Test", parameters={"type": "object"}
            ),
        )
        task = await manager.submit(
            spec, ToolCall(call_id="c", name="test", arguments={})
        )
        await manager.close(cancel=True)
        result = await manager.get_result(task.task_id)
        assert result is not None
        assert result.status == "cancelled"


@pytest.mark.asyncio
@pytest.mark.parametrize("durable", [False, True])
async def test_close_terminalizes_prepared_admission(tmp_path, durable):
    registry = ExecutorRegistry()
    async with SQLiteHistoryStore(tmp_path / "prepared-close.db") as history:
        manager = (
            DurableToolTaskManager(history, registry)
            if durable
            else InMemoryToolTaskManager(registry)
        )
        spec = ToolSpec(
            tool_id="test",
            version="1",
            definition=ToolDefinition(
                name="test", description="Test", parameters={"type": "object"}
            ),
        )
        task = await manager.prepare(
            spec, ToolCall(call_id="c", name="test", arguments={})
        )
        await manager.close(cancel=True)
        result = await manager.get_result(task.task_id)
        assert result is not None and result.status == "cancelled"


@pytest.mark.asyncio
@pytest.mark.parametrize("same_owner", [False, True])
@pytest.mark.parametrize("unsafe", [False, True])
async def test_recovery_cannot_take_over_another_live_job_owner(
    tmp_path, same_owner, unsafe
):
    registry = ExecutorRegistry()
    async with SQLiteHistoryStore(tmp_path / "live-owner.db") as history:
        first = DurableToolTaskManager(history, registry)
        second = first if same_owner else DurableToolTaskManager(history, registry)
        entered = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def execute(spec, call, context):
            nonlocal calls
            calls += 1
            await context.publish_output({"stdout": "started"})
            entered.set()
            await release.wait()
            return "finished"

        spec = ToolSpec(
            tool_id="test",
            version="1",
            definition=ToolDefinition(
                name="test", description="Test", parameters={"type": "object"}
            ),
        )
        from dataclasses import replace

        from pygent.tool import IdempotencyPolicy, ToolSideEffect

        if unsafe:
            spec = replace(
                spec,
                side_effect=ToolSideEffect.EXTERNAL,
                idempotency=IdempotencyPolicy.NOT_IDEMPOTENT,
            )
        task = await first.prepare_job(
            spec,
            ToolCall(call_id="c", name="test", arguments={}),
            logical_key="live",
            binding_id="b",
            plan_id="p",
            required_capabilities=(),
            execution=execute,
        )
        await first.start(task.task_id)
        await entered.wait()
        stored = await history.get_job(task.job_id)
        try:
            await second.recover_job(stored, execution=execute)
            await asyncio.sleep(0)
            assert calls == 1
            assert (await first.get_task(task.task_id)).state.value == "running"
            # The original writer must retain its fence after the attempted recovery.
            await first._publish_output(task.task_id, {"stdout": "still owned"})
            release.set()
            assert (
                await first.get_result(task.task_id, wait=True)
            ).status == "succeeded"
        finally:
            release.set()
            await first.close(cancel=True)
            await second.close(cancel=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("job", [False, True])
async def test_delayed_concurrent_start_cannot_relaunch_completed_task(
    tmp_path, monkeypatch, job
):
    registry = ExecutorRegistry()
    async with SQLiteHistoryStore(tmp_path / "start-race.db") as history:
        manager = DurableToolTaskManager(history, registry)
        calls = 0

        async def execute(spec, call, context):
            nonlocal calls
            calls += 1
            return calls

        spec = ToolSpec(
            tool_id="test",
            version="1",
            definition=ToolDefinition(
                name="test", description="Test", parameters={"type": "object"}
            ),
        )
        call = ToolCall(call_id="c", name="test", arguments={})
        if job:
            task = await manager.prepare_job(
                spec,
                call,
                logical_key="race",
                binding_id="b",
                plan_id="p",
                required_capabilities=(),
                execution=execute,
            )
        else:
            task = await manager.prepare(spec, call, execution=execute)
        launch_name = "_launch_job" if job else "_launch"
        launch = getattr(manager, launch_name)
        delayed = asyncio.Event()
        release = asyncio.Event()

        async def delay_launch(*args, **kwargs):
            delayed.set()
            await release.wait()
            await launch(*args, **kwargs)

        monkeypatch.setattr(manager, launch_name, delay_launch)
        observer = asyncio.create_task(manager.start(task.task_id))
        await delayed.wait()
        monkeypatch.setattr(manager, launch_name, launch)
        await manager.start(task.task_id)
        await manager.get_result(task.task_id, wait=True)
        release.set()
        await observer
        await manager.get_result(task.task_id, wait=True)
        assert calls == 1
        await manager.close()
