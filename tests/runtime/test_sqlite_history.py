from __future__ import annotations

import asyncio
import sqlite3

import pytest

from pygent.core import thaw_json
from pygent.runtime.history import (
    HistoryConflictError,
    HistoryStoreError,
    NonDeterministicReplayError,
    SQLiteHistoryStore,
)


@pytest.mark.asyncio
async def test_sqlite_run_and_task_state_survive_reopen(tmp_path):
    path = tmp_path / "history.sqlite3"
    async with SQLiteHistoryStore(path) as store:
        await store.create_execution(
            execution_id="run-1",
            request_id="request-1",
            plan_id="plan-1",
            input={"message": "hello"},
        )
        await store.update_execution(
            "run-1", status="succeeded", output={"message": "world"}
        )
        await store.put_task(
            task_id="tool-1",
            kind="tool_task",
            status="succeeded",
            request={"city": "Shanghai"},
            result={"temperature": 30},
        )

    async with SQLiteHistoryStore(path) as restored:
        run = await restored.get_execution("run-1")
        task = await restored.get_task("tool-1")

    assert run is not None and run.status == "succeeded"
    assert thaw_json(run.output) == {"message": "world"}
    assert task is not None and task.kind == "tool_task"
    assert thaw_json(task.result) == {"temperature": 30}


@pytest.mark.asyncio
async def test_idempotency_identity_and_effect_history_are_conflict_safe(tmp_path):
    async with SQLiteHistoryStore(tmp_path / "history.sqlite3") as store:
        first = await store.create_execution(
            execution_id="run-1",
            request_id="request-1",
            plan_id="plan-1",
            input={"value": 1},
            binding_id="binding-1",
            identity="user-1",
            idempotency_key="operation-1",
        )
        repeated = await store.create_execution(
            execution_id="run-2",
            request_id="request-2",
            plan_id="plan-1",
            input={"value": 1},
            binding_id="binding-1",
            identity="user-1",
            idempotency_key="operation-1",
        )
        assert repeated == first

        with pytest.raises(HistoryConflictError):
            await store.create_execution(
                execution_id="run-3",
                request_id="request-3",
                plan_id="plan-1",
                input={"value": 2},
                binding_id="binding-1",
                identity="user-1",
                idempotency_key="operation-1",
            )

        independent = await store.create_execution(
            execution_id="run-4",
            request_id="request-1",
            plan_id="plan-1",
            input={"value": 2},
        )
        assert independent.execution_id == "run-4"

        await store.record_effect(
            execution_id="run-1",
            module_path="root.model",
            call_index=0,
            effect_type="model",
            request={"prompt": "hello"},
            result={"content": "world"},
        )
        replay = await store.replay_effect(
            execution_id="run-1",
            module_path="root.model",
            call_index=0,
            effect_type="model",
            request={"prompt": "hello"},
        )
        assert thaw_json(replay.result) == {"content": "world"}

        with pytest.raises(NonDeterministicReplayError):
            await store.replay_effect(
                execution_id="run-1",
                module_path="root.model",
                call_index=0,
                effect_type="model",
                request={"prompt": "different"},
            )
        with pytest.raises(NonDeterministicReplayError):
            await store.replay_effect(
                execution_id="run-1",
                module_path="root.model",
                call_index=0,
                effect_type="tool",
                request={"prompt": "hello"},
            )
        with pytest.raises(NonDeterministicReplayError):
            await store.record_effect(
                execution_id="run-1",
                module_path="root.model",
                call_index=0,
                effect_type="tool",
                request={"prompt": "hello"},
                result={"content": "different effect"},
            )


@pytest.mark.asyncio
async def test_task_identity_rejects_kind_or_request_conflicts(tmp_path):
    async with SQLiteHistoryStore(tmp_path / "history.sqlite3") as store:
        await store.put_task(
            task_id="task-1",
            kind="tool_task",
            status="pending",
            request={"value": 1},
        )
        await store.put_task(
            task_id="task-1",
            kind="tool_task",
            status="running",
            request={"value": 1},
        )
        with pytest.raises(HistoryConflictError):
            await store.put_task(
                task_id="task-1",
                kind="tool_task",
                status="running",
                request={"value": 2},
            )
        with pytest.raises(HistoryConflictError):
            await store.put_task(
                task_id="task-1",
                kind="job",
                status="running",
                request={"value": 1},
            )


@pytest.mark.asyncio
async def test_open_rejects_legacy_effect_identity_schema(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE effects (execution_id TEXT NOT NULL,module_path TEXT NOT NULL,"
        "call_index INTEGER NOT NULL,effect_type TEXT NOT NULL,"
        "request_digest TEXT NOT NULL,result_json TEXT NOT NULL,"
        "PRIMARY KEY(execution_id,module_path,call_index,effect_type))"
    )
    db.commit()
    db.close()

    with pytest.raises(HistoryStoreError, match="Pygent 0.2"):
        await SQLiteHistoryStore(path).open()


@pytest.mark.asyncio
async def test_checkpoint_compatibility_and_event_cursor(tmp_path):
    async with SQLiteHistoryStore(tmp_path / "history.sqlite3") as store:
        await store.save_checkpoint(
            execution_id="run-1",
            checkpoint_index=2,
            graph_hash="graph-v1",
            state={"message": "checkpoint"},
        )
        checkpoint = await store.load_checkpoint(
            execution_id="run-1", graph_hash="graph-v1"
        )
        assert checkpoint is not None
        assert checkpoint[0] == 2
        assert thaw_json(checkpoint[1]) == {"message": "checkpoint"}

        with pytest.raises(NonDeterministicReplayError):
            await store.load_checkpoint(execution_id="run-1", graph_hash="graph-v2")

        for index in range(3):
            await store.append_event(
                execution_id="run-1", index=index, event={"index": index}
            )
        events = await store.events_after(execution_id="run-1", after=0)
        assert [thaw_json(event)["index"] for event in events] == [1, 2]
        tail = await store.events_tail(execution_id="run-1", limit=2)
        assert [thaw_json(event)["index"] for event in tail] == [1, 2]


@pytest.mark.asyncio
async def test_claim_transactions_serialize_with_concurrent_event_writes(tmp_path):
    async with SQLiteHistoryStore(tmp_path / "history.sqlite3") as store:
        async def write_and_claim(index: int) -> None:
            claim_id = f"claim-{index}"
            event_id = f"event-{index}"
            _, token = await asyncio.gather(
                store.append_event(
                    execution_id=event_id,
                    index=0,
                    event={"index": index},
                ),
                store.claim_execution(
                    execution_id=claim_id,
                    owner_id="worker",
                    lease_ttl=5.0,
                ),
            )
            assert token is not None
            await store.release_execution_claim(
                execution_id=claim_id,
                owner_id="worker",
                fencing_token=token,
            )

        await asyncio.gather(*(write_and_claim(index) for index in range(64)))
