from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3

import pytest

from pygent.core import JsonValueError, freeze_json_object, thaw_json
from pygent.runtime import (
    HistoryConflictError,
    HistoryStoreError,
    NonDeterministicReplayError,
    SQLiteHistoryStore,
)
from pygent.runtime._history_types import (
    _json_frozen,
    _json_frozen_object,
    _prepare_json,
    effect_digest,
)


@pytest.mark.parametrize("pending_batches", [0, -1, True])
def test_pending_event_batches_must_be_positive_integer(tmp_path, pending_batches):
    with pytest.raises(ValueError, match="max_pending_event_batches"):
        SQLiteHistoryStore(
            tmp_path / "history.sqlite3",
            max_pending_event_batches=pending_batches,
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("max_transaction_batch_size", 0),
        ("max_transaction_batch_size", True),
        ("max_pending_transactions", -1),
        ("max_pending_transactions", True),
    ],
)
def test_transaction_batch_bounds_must_be_positive_integers(tmp_path, name, value):
    with pytest.raises(ValueError, match=name):
        SQLiteHistoryStore(tmp_path / "history.sqlite3", **{name: value})


def test_validated_json_object_serialization_reuses_frozen_children():
    payload = freeze_json_object(
        {"nested": {"value": 1}, "sequence": [True, None, "上海"]}
    )

    assert _json_frozen_object({"data": payload, "sequence": 3}) == json.dumps(
        {"data": thaw_json(payload), "sequence": 3},
        sort_keys=True,
        separators=(",", ":"),
    )
    assert _json_frozen(payload) == json.dumps(
        thaw_json(payload),
        sort_keys=True,
        separators=(",", ":"),
    )


def test_prepared_json_reuses_frozen_root_and_preserves_digest_bytes():
    payload = freeze_json_object(
        {"message": "上海", "nested": {"enabled": True}, "sequence": [1, 2]}
    )
    expected = json.dumps(
        thaw_json(payload),
        sort_keys=True,
        separators=(",", ":"),
    )

    prepared = _prepare_json(payload)

    assert prepared.frozen is payload
    assert prepared.payload == expected
    assert effect_digest(payload) == hashlib.sha256(expected.encode()).hexdigest()
    with pytest.raises(JsonValueError):
        _prepare_json({"invalid": float("nan")})


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
async def test_new_effect_boundary_returns_without_replay_query(tmp_path):
    class NoReplayOnInsertHistory(SQLiteHistoryStore):
        async def replay_effect(self, **kwargs):
            raise AssertionError("new effect boundary must not be read back")

    async with NoReplayOnInsertHistory(tmp_path / "history.sqlite3") as store:
        stored, created = await store.begin_effect(
            execution_id="run-1",
            module_path="root.tool",
            call_index=0,
            effect_type="tool",
            request={"name": "search"},
            spec={"idempotency": "required"},
        )

    assert created is True
    assert stored.status == "started"
    assert thaw_json(stored.spec) == {"idempotency": "required"}


@pytest.mark.asyncio
async def test_completed_effect_with_json_null_result_remains_idempotent(tmp_path):
    async with SQLiteHistoryStore(tmp_path / "history.sqlite3") as store:
        first = await store.record_effect(
            execution_id="run-1",
            module_path="root.tool",
            call_index=0,
            effect_type="tool",
            request={"name": "noop"},
            result=None,
        )
        repeated = await store.record_effect(
            execution_id="run-1",
            module_path="root.tool",
            call_index=0,
            effect_type="tool",
            request={"name": "noop"},
            result=None,
        )

    assert first.result is None
    assert repeated == first


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

    with pytest.raises(HistoryStoreError, match="schema v7"):
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
        await store.append_event(
            execution_id="run-1", index=2, event={"index": 2}
        )
        with pytest.raises(HistoryConflictError):
            await store.append_event(
                execution_id="run-1", index=2, event={"index": "conflict"}
            )
        await store.append_event(
            execution_id="run-1", index=3, event={"index": 3}
        )


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


@pytest.mark.asyncio
async def test_concurrent_events_share_group_commit(tmp_path):
    class CountingHistory(SQLiteHistoryStore):
        def __init__(self, path):
            super().__init__(path, max_event_batch_size=64)
            self.batch_sizes = []

        async def _commit_event_batch(self, batch):
            self.batch_sizes.append(len(batch))
            await super()._commit_event_batch(batch)

    async with CountingHistory(tmp_path / "grouped.sqlite3") as store:
        await asyncio.gather(
            *(
                store.append_event(
                    execution_id=f"execution-{index}",
                    index=0,
                    event={"index": index},
                )
                for index in range(32)
            )
        )

        assert sum(store.batch_sizes) == 32
        assert len(store.batch_sizes) < 32


@pytest.mark.asyncio
async def test_full_event_batch_shares_one_commit_receipt(tmp_path):
    async with SQLiteHistoryStore(
        tmp_path / "grouped.sqlite3", max_event_batch_size=8
    ) as store:
        receipts = []
        for index in range(8):
            await store._reserve_event_slot()
            receipts.append(
                store._enqueue_reserved_event_payload(
                    f"execution-{index}", index, json.dumps({"index": index})
                )
            )

        assert len({id(receipt) for receipt in receipts}) == 1
        await asyncio.gather(*(asyncio.shield(receipt) for receipt in receipts))
        assert all(receipt.done() and receipt.exception() is None for receipt in receipts)


@pytest.mark.asyncio
async def test_cancelled_event_waiter_does_not_cancel_shared_commit(tmp_path):
    class GatedHistory(SQLiteHistoryStore):
        def __init__(self, path):
            super().__init__(path)
            self.commit_entered = asyncio.Event()
            self.release_commit = asyncio.Event()

        async def _commit_event_batch(self, batch):
            self.commit_entered.set()
            await self.release_commit.wait()
            await super()._commit_event_batch(batch)

    async with GatedHistory(tmp_path / "cancelled.sqlite3") as store:
        waiter = asyncio.create_task(
            store.append_event(execution_id="execution", index=0, event={"ok": True})
        )
        await asyncio.wait_for(store.commit_entered.wait(), timeout=1)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        store.release_commit.set()
        assert store._event_flush_task is not None
        await asyncio.shield(store._event_flush_task)
        persisted = await store.events_after(execution_id="execution", after=-1)
        assert [thaw_json(event) for event in persisted] == [{"ok": True}]


@pytest.mark.asyncio
async def test_pending_event_capacity_applies_backpressure(tmp_path):
    class GatedHistory(SQLiteHistoryStore):
        def __init__(self, path):
            super().__init__(
                path, max_event_batch_size=1, max_pending_event_batches=1
            )
            self.commit_entered = asyncio.Event()
            self.release_commit = asyncio.Event()

        async def _commit_event_batch(self, batch):
            self.commit_entered.set()
            await self.release_commit.wait()
            await super()._commit_event_batch(batch)

    async with GatedHistory(tmp_path / "bounded.sqlite3") as store:
        first = asyncio.create_task(
            store.append_event(execution_id="execution", index=0, event={"index": 0})
        )
        await asyncio.wait_for(store.commit_entered.wait(), timeout=1)
        second = asyncio.create_task(
            store.append_event(execution_id="execution", index=1, event={"index": 1})
        )
        await asyncio.sleep(0)
        assert second.done() is False

        store.release_commit.set()
        await asyncio.gather(first, second)
        persisted = await store.events_after(execution_id="execution", after=-1)
        assert [thaw_json(event) for event in persisted] == [
            {"index": 0},
            {"index": 1},
        ]


@pytest.mark.asyncio
async def test_concurrent_effect_boundaries_share_transactions(tmp_path):
    class CountingHistory(SQLiteHistoryStore):
        def __init__(self, path):
            super().__init__(path)
            self.transaction_sizes = []

        async def _run_transaction_batch(self, batch):
            self.transaction_sizes.append(len(batch))
            return await super()._run_transaction_batch(batch)

    async with CountingHistory(tmp_path / "effects.sqlite3") as store:
        started = await asyncio.gather(
            *(
                store.begin_effect(
                    execution_id=f"execution-{index}",
                    module_path="root",
                    call_index=0,
                    effect_type="test.effect",
                    request={"index": index},
                    spec={"side_effect": "read"},
                )
                for index in range(32)
            )
        )
        await asyncio.gather(
            *(
                store.complete_effect(
                    execution_id=f"execution-{index}",
                    module_path="root",
                    call_index=0,
                    result={"index": index},
                )
                for index in range(32)
            )
        )

        assert all(created for _, created in started)
        assert sum(store.transaction_sizes) == 64
        assert len(store.transaction_sizes) < 64


@pytest.mark.asyncio
async def test_homogeneous_transaction_runs_use_vectorized_sql(tmp_path):
    class CountingHistory(SQLiteHistoryStore):
        def __init__(self, path):
            super().__init__(path)
            self.vector_sizes = {
                "begin_execution": [],
                "update_execution": [],
                "begin_effect": [],
                "complete_effect": [],
                "finalize_execution": [],
            }

        async def _batch_begin_executions(self, db, payloads):
            self.vector_sizes["begin_execution"].append(len(payloads))
            return await super()._batch_begin_executions(db, payloads)

        async def _batch_update_executions(self, db, payloads):
            self.vector_sizes["update_execution"].append(len(payloads))
            return await super()._batch_update_executions(db, payloads)

        async def _batch_begin_effects(self, db, payloads):
            self.vector_sizes["begin_effect"].append(len(payloads))
            return await super()._batch_begin_effects(db, payloads)

        async def _batch_complete_effects(self, db, payloads):
            self.vector_sizes["complete_effect"].append(len(payloads))
            return await super()._batch_complete_effects(db, payloads)

        async def _batch_finalize_executions(self, db, payloads):
            self.vector_sizes["finalize_execution"].append(len(payloads))
            return await super()._batch_finalize_executions(db, payloads)

    count = 16
    async with CountingHistory(tmp_path / "vectorized.sqlite3") as store:
        await asyncio.gather(
            *(
                store.create_execution(
                    execution_id=f"execution-{index}",
                    request_id=f"request-{index}",
                    plan_id="plan",
                    input={"index": index},
                )
                for index in range(count)
            )
        )
        await asyncio.gather(
            *(
                store.update_execution(
                    f"execution-{index}", status="running", phase="running"
                )
                for index in range(count)
            )
        )
        await asyncio.gather(
            *(
                store.begin_effect(
                    execution_id=f"execution-{index}",
                    module_path="root",
                    call_index=0,
                    effect_type="test.effect",
                    request={"index": index},
                    spec={"side_effect": "read"},
                )
                for index in range(count)
            )
        )
        await asyncio.gather(
            *(
                store.complete_effect(
                    execution_id=f"execution-{index}",
                    module_path="root",
                    call_index=0,
                    result={"index": index},
                )
                for index in range(count)
            )
        )
        await asyncio.gather(
            *(
                store.finalize_execution(
                    f"execution-{index}",
                    status="succeeded",
                    output={"index": index},
                    error=None,
                    terminal_events=((0, {"kind": "execution.completed"}),),
                    terminal_sequence=0,
                )
                for index in range(count)
            )
        )

    assert all(
        sum(sizes) == count and max(sizes) > 1
        for sizes in store.vector_sizes.values()
    )


@pytest.mark.asyncio
async def test_failed_transaction_request_isolated_from_valid_peer(tmp_path):
    async with SQLiteHistoryStore(tmp_path / "isolated.sqlite3") as store:
        await store.create_execution(
            execution_id="valid",
            request_id="valid",
            plan_id="plan",
            input={},
        )

        failed, succeeded = await asyncio.gather(
            store.finalize_execution(
                "missing",
                status="succeeded",
                output={},
                error=None,
                terminal_events=((0, {"kind": "execution.completed"}),),
                terminal_sequence=0,
            ),
            store.finalize_execution(
                "valid",
                status="succeeded",
                output={"ok": True},
                error=None,
                terminal_events=((0, {"kind": "execution.completed"}),),
                terminal_sequence=0,
            ),
            return_exceptions=True,
        )

        assert isinstance(failed, HistoryConflictError)
        assert succeeded is None
        stored = await store.get_execution("valid")
        assert stored is not None and stored.status == "succeeded"
