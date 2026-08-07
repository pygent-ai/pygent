from __future__ import annotations

import asyncio

import pytest

from pygent.runtime import LocalRuntime, SQLiteHistoryStore
from pygent.runtime.tasks import DurableToolTaskManager, _request_to_dict
from pygent.tool import (
    ExecutorRegistry,
    IdempotencyPolicy,
    LocalToolExecutor,
    ToolCall,
    ToolDefinition,
    ToolSideEffect,
    ToolSpec,
    ToolTaskState,
)


def tool(
    *,
    side_effect: ToolSideEffect = ToolSideEffect.PURE,
    idempotency: IdempotencyPolicy = IdempotencyPolicy.INHERENT,
) -> ToolSpec:
    return ToolSpec(
        tool_id="tool.echo",
        version="1",
        definition=ToolDefinition(
            name="echo",
            description="Echo",
            parameters={"type": "object"},
        ),
        side_effect=side_effect,
        idempotency=idempotency,
    )


def call() -> ToolCall:
    return ToolCall(call_id="call-1", name="echo", arguments={"value": 1})


@pytest.mark.asyncio
async def test_durable_tool_task_persists_admission_and_terminal_result(tmp_path):
    registry = ExecutorRegistry()
    spec = tool()
    registry.register(
        spec.tool_id,
        spec.version,
        LocalToolExecutor(lambda arguments: {"echo": arguments["value"]}),
    )
    path = tmp_path / "history.sqlite3"
    async with SQLiteHistoryStore(path) as history:
        manager = DurableToolTaskManager(history, registry)
        runtime = LocalRuntime()
        runtime.attach_tool_task_manager(manager)
        snapshot = await manager.submit(spec, call())
        result = await runtime.get_tool_result(snapshot.task_id, wait=True)
        assert result is not None and result.status == "succeeded"
        assert result.output["echo"] == 1
        assert await runtime.get_tool_task(snapshot.task_id) is not None
        await runtime.close()

    async with SQLiteHistoryStore(path) as restored:
        manager = DurableToolTaskManager(restored, registry)
        snapshot = await manager.get_task(snapshot.task_id)
        result = await manager.get_result(snapshot.task_id)
        assert snapshot is not None and snapshot.state is ToolTaskState.SUCCEEDED
        assert result is not None and result.output["echo"] == 1


@pytest.mark.asyncio
async def test_legacy_recovery_refuses_pending_task_without_runtime_validation(tmp_path):
    executed = asyncio.Event()
    registry = ExecutorRegistry()
    spec = tool()

    def execute(arguments):
        executed.set()
        return "done"

    registry.register(spec.tool_id, spec.version, LocalToolExecutor(execute))
    async with SQLiteHistoryStore(tmp_path / "history.sqlite3") as history:
        await history.put_task(
            task_id="tool-pending",
            kind="tool_task",
            status="pending",
            request=_request_to_dict(spec, call()),
        )
        manager = DurableToolTaskManager(history, registry)
        with pytest.raises(RuntimeError, match="LocalRuntime.recover_tool_jobs"):
            await manager.recover()
        result = await manager.get_result("tool-pending")

    assert not executed.is_set()
    assert result is None


@pytest.mark.asyncio
async def test_legacy_recovery_refuses_running_task_without_runtime_validation(tmp_path):
    registry = ExecutorRegistry()
    spec = tool(
        side_effect=ToolSideEffect.EXTERNAL,
        idempotency=IdempotencyPolicy.NOT_IDEMPOTENT,
    )
    async with SQLiteHistoryStore(tmp_path / "history.sqlite3") as history:
        await history.put_task(
            task_id="tool-running",
            kind="tool_task",
            status="running",
            request=_request_to_dict(spec, call()),
        )
        manager = DurableToolTaskManager(history, registry)
        with pytest.raises(RuntimeError, match="LocalRuntime.recover_tool_jobs"):
            await manager.recover()
        result = await manager.get_result("tool-running")
        stored = await history.get_task("tool-running")

    assert result is None
    assert stored is not None and stored.status == "running"


@pytest.mark.asyncio
async def test_cancelling_started_side_effect_records_unknown_not_cancelled(tmp_path):
    entered = asyncio.Event()
    registry = ExecutorRegistry()
    spec = tool(
        side_effect=ToolSideEffect.EXTERNAL,
        idempotency=IdempotencyPolicy.NOT_IDEMPOTENT,
    )

    async def execute(arguments):
        entered.set()
        await asyncio.Event().wait()

    registry.register(spec.tool_id, spec.version, LocalToolExecutor(execute))
    async with SQLiteHistoryStore(tmp_path / "history.sqlite3") as history:
        manager = DurableToolTaskManager(history, registry)
        task = await manager.submit(spec, call())
        await entered.wait()
        assert await manager.cancel(task.task_id) is True
        result = await manager.get_result(task.task_id)

    assert result is not None and result.status == "unknown"
    assert result.side_effect_committed is None
    assert result.task is not None and result.task.state is ToolTaskState.UNKNOWN
