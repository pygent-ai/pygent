from __future__ import annotations

import asyncio
import time

import pytest

from pygent.core import (
    AIMessage,
    Context,
    DirectExecutionError,
    ExecutionInput,
    ExecutionInputDelivery,
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
from pygent.runtime._execution_inputs import ExecutionInputConsumerError


class InputReceiver(Module[UserMessage, AIMessage]):
    trusted_live_resource_attributes = ("state",)

    class State:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.received = []

    def __init__(self) -> None:
        super().__init__()
        self.state = self.State()

    async def forward(self, message: UserMessage, context: Context):
        self.state.started.set()
        await self.state.release.wait()
        self.state.received.extend(
            await self.receive_execution_inputs(kinds=("test.input",))
        )
        return AIMessage(content="done"), context


def bind(runtime: LocalRuntime, module: Module):
    binding = runtime.create_binding(
        name="input-test",
        execution_capacity=ExecutionCapacityPolicy(
            scope=CapacityScope.RUNTIME_INSTANCE,
            max_live_executions=1,
            max_runnable_executions=1,
            max_queue_size=1,
            max_waiters=1,
            max_child_depth=4,
            max_children_per_execution=4,
        ),
        model_capacity=CapacityPolicy.passthrough(),
        tool_capacity=CapacityPolicy.passthrough(),
    )
    return binding.bind(module)


@pytest.mark.asyncio
@pytest.mark.parametrize("durable", [False, True])
async def test_execution_input_is_ordered_idempotent_and_terminal(tmp_path, durable):
    history = (
        await SQLiteHistoryStore(tmp_path / "inputs.sqlite3").open()
        if durable
        else None
    )
    runtime = LocalRuntime(history=history)
    receiver = InputReceiver()
    handle = await bind(runtime, receiver).start(
        UserMessage(content="go"),
        Context(),
        execution=ExecutionOptions(deadline=time.monotonic() + 5),
    )
    first = await handle.send_input(input_id="one", kind="test.input", value={"n": 1})
    duplicate = await handle.send_input(
        input_id="one", kind="test.input", value={"n": 999}
    )
    second = await handle.send_input(input_id="two", kind="test.input", value={"n": 2})
    assert (first.status, first.sequence) == ("accepted", 0)
    assert (duplicate.status, duplicate.sequence) == ("duplicate", 0)
    assert (second.status, second.sequence) == ("accepted", 1)

    await receiver.state.started.wait()
    receiver.state.release.set()
    await handle.result()
    assert [item.input_id for item in receiver.state.received] == ["one", "two"]
    if history is not None:
        replayed = await history.receive_execution_inputs(
            execution_id=handle.execution_id,
            module_path=handle._record.plan.root,
            receive_index=0,
            kinds=("test.input",),
            limit=16,
            seal_if_empty=False,
        )
        assert replayed == tuple(receiver.state.received)
    finished = await handle.send_input(
        input_id="three", kind="test.input", value={"n": 3}
    )
    assert finished.status == "execution_finished"
    await runtime.close()
    if history is not None:
        await history.close()


@pytest.mark.asyncio
async def test_direct_execution_rejects_send_and_receives_empty() -> None:
    receiver = InputReceiver()
    receiver.state.release.set()
    handle = await receiver.start(UserMessage(content="go"), Context())
    with pytest.raises(DirectExecutionError, match="direct executions"):
        await handle.send_input(input_id="one", kind="test.input", value={})
    await handle.result()
    assert receiver.state.received == []


def test_execution_input_portable_codecs_are_strict_and_frozen() -> None:
    value = {"nested": [1, 2]}
    item = ExecutionInput("input-1", 3, "kind", value)
    value["nested"].append(4)
    assert item.to_dict() == {
        "input_id": "input-1",
        "sequence": 3,
        "kind": "kind",
        "value": {"nested": [1, 2]},
    }
    assert ExecutionInput.from_dict(item.to_dict()) == item
    delivery = ExecutionInputDelivery("accepted", "execution-1", "input-1", 3)
    assert ExecutionInputDelivery.from_dict(delivery.to_dict()) == delivery
    with pytest.raises(ValueError, match="fields are invalid"):
        ExecutionInput.from_dict({**item.to_dict(), "extra": True})


@pytest.mark.asyncio
@pytest.mark.parametrize("durable", [False, True])
async def test_seal_if_empty_closes_the_send_finalization_window(
    tmp_path, durable
) -> None:
    class State:
        def __init__(self) -> None:
            self.sealed = asyncio.Event()
            self.release = asyncio.Event()

    class Sealer(Module[UserMessage, AIMessage]):
        trusted_live_resource_attributes = ("state",)

        def __init__(self) -> None:
            super().__init__()
            self.state = State()

        async def forward(self, message, context):
            assert await self.receive_execution_inputs(
                kinds=("test.input",), seal_if_empty=True
            ) == ()
            self.state.sealed.set()
            await self.state.release.wait()
            return AIMessage(content="done"), context

    history = (
        await SQLiteHistoryStore(tmp_path / "seal.sqlite3").open()
        if durable
        else None
    )
    runtime = LocalRuntime(history=history)
    sealer = Sealer()
    handle = await bind(runtime, sealer).start(
        UserMessage(content="go"), Context(), execution=ExecutionOptions(
            deadline=time.monotonic() + 5
        )
    )
    await sealer.state.sealed.wait()
    delivery = await handle.send_input(
        input_id="too-late", kind="test.input", value={}
    )
    assert delivery.status == "execution_finished"
    sealer.state.release.set()
    await handle.result()
    await runtime.close()
    if history is not None:
        await history.close()


@pytest.mark.asyncio
async def test_one_kind_cannot_be_consumed_by_two_module_paths() -> None:
    class Child(Module[UserMessage, AIMessage]):
        async def forward(self, message, context):
            await self.receive_execution_inputs(kinds=("owned",))
            return AIMessage(content="child"), context

    class Parent(Module[UserMessage, AIMessage]):
        def __init__(self) -> None:
            super().__init__()
            self.child = Child()

        async def forward(self, message, context):
            await self.receive_execution_inputs(kinds=("owned",))
            return await self.child(message, context)

    runtime = LocalRuntime()
    handle = await bind(runtime, Parent()).start(
        UserMessage(content="go"), Context(), execution=ExecutionOptions(
            deadline=time.monotonic() + 5
        )
    )
    with pytest.raises(ExecutionInputConsumerError, match="is owned by"):
        await handle.result()
    await runtime.close()


@pytest.mark.asyncio
async def test_execution_input_enforces_pending_capacity() -> None:
    receiver = InputReceiver()
    runtime = LocalRuntime()
    handle = await bind(runtime, receiver).start(
        UserMessage(content="go"), Context(), execution=ExecutionOptions(
            deadline=time.monotonic() + 5
        )
    )
    await receiver.state.started.wait()
    for index in range(256):
        await handle.send_input(
            input_id=f"input-{index}", kind="test.input", value=index
        )
    with pytest.raises(OverflowError, match="inbox is full"):
        await handle.send_input(
            input_id="overflow", kind="test.input", value=257
        )
    receiver.state.release.set()
    await handle.result()
    await runtime.close()

@pytest.mark.asyncio
@pytest.mark.parametrize("durable", [False, True])
async def test_large_execution_input_round_trips_without_truncation(tmp_path, durable):
    value = {"memory": "上下文" * 100_000, "history": ["x" * 200_000]}
    history = (
        await SQLiteHistoryStore(tmp_path / "large-input.sqlite3").open()
        if durable
        else None
    )
    runtime = LocalRuntime(history=history)
    receiver = InputReceiver()
    try:
        handle = await bind(runtime, receiver).start(
            UserMessage(content="go"),
            Context(),
            execution=ExecutionOptions(deadline=time.monotonic() + 20),
        )
        await receiver.state.started.wait()
        delivery = await handle.send_input(input_id="large", kind="test.input", value=value)
        assert delivery.status == "accepted"
        receiver.state.release.set()
        await handle.result()
        item = receiver.state.received[0]
        assert item.to_dict()["value"] == value
        assert ExecutionInput.from_dict(item.to_dict()) == item
    finally:
        await runtime.close()
        if history is not None:
            await history.close()


async def _create_inbox_execution(store, execution_id):
    await store.create_execution(
        execution_id=execution_id, request_id=execution_id, plan_id="plan", input={}
    )


async def _receive(
    store,
    execution_id,
    *,
    index=0,
    module="root",
    kinds=("a", "b"),
    limit=16,
    seal=False,
):
    return await store.receive_execution_inputs(
        execution_id=execution_id,
        module_path=module,
        receive_index=index,
        kinds=kinds,
        limit=limit,
        seal_if_empty=seal,
    )


@pytest.mark.asyncio
async def test_sqlite_receives_batch_queries_and_keep_empty_receipts(tmp_path):
    async with SQLiteHistoryStore(tmp_path / "batch.sqlite3") as store:
        identities = [str(i) for i in range(32)]
        await asyncio.gather(*(_create_inbox_execution(store, i) for i in identities))
        queries = []
        await store._db().set_trace_callback(queries.append)
        assert (
            await asyncio.gather(*(_receive(store, i) for i in identities)) == [()] * 32
        )
        await store._db().set_trace_callback(None)
        assert (
            len([q for q in queries if q.startswith("SELECT") and "json_each" in q])
            == 3
        )
        assert (
            await store._db().execute_fetchall(
                "SELECT COUNT(*) FROM execution_input_receives WHERE batch_json='[]'"
            )
        )[0][0] == 32
        await store.send_execution_input("0", input_id="late", kind="a", value=1)
        assert await _receive(store, "0") == ()
        assert [i.input_id for i in await _receive(store, "0", index=1)] == ["late"]


@pytest.mark.asyncio
async def test_sqlite_batched_receives_preserve_per_execution_limits_and_kind_cursors(
    tmp_path,
):
    async with SQLiteHistoryStore(tmp_path / "limits.sqlite3") as store:
        await asyncio.gather(
            *(_create_inbox_execution(store, i) for i in ("left", "right"))
        )
        for execution_id in ("left", "right"):
            for n, kind in enumerate(("a", "b", "a", "b")):
                await store.send_execution_input(
                    execution_id, input_id=str(n), kind=kind, value=n
                )
        left, right = await asyncio.gather(
            _receive(store, "left", limit=2),
            _receive(store, "right", kinds=("b",), limit=1),
        )
        assert [i.sequence for i in left] == [0, 1]
        assert [i.sequence for i in right] == [1]
        left, right = await asyncio.gather(
            _receive(store, "left", index=1), _receive(store, "right", index=1)
        )
        assert [i.sequence for i in left] == [2, 3]
        assert [i.sequence for i in right] == [0, 2, 3]


@pytest.mark.asyncio
async def test_sqlite_overlapping_receives_remain_ordered(tmp_path):
    async with SQLiteHistoryStore(tmp_path / "ordered.sqlite3") as store:
        await _create_inbox_execution(store, "one")
        for i in range(2):
            await store.send_execution_input("one", input_id=str(i), kind="a", value=i)
        first, second = await asyncio.gather(
            _receive(store, "one", index=0, limit=1),
            _receive(store, "one", index=1, limit=1),
        )
        assert [i.sequence for i in first + second] == [0, 1]


@pytest.mark.asyncio
async def test_sqlite_bad_receive_does_not_advance_valid_peer_twice(tmp_path):
    async with SQLiteHistoryStore(tmp_path / "isolated.sqlite3") as store:
        for identity in ("owned", "valid", "replayed"):
            await _create_inbox_execution(store, identity)
        await _receive(store, "owned", module="owner")
        await _receive(store, "replayed")
        await store.send_execution_input(
            "valid", input_id="once", kind="a", value={"v": 1}
        )
        owned, valid, replayed = await asyncio.gather(
            _receive(store, "owned", module="other"),
            _receive(store, "valid"),
            _receive(store, "replayed", limit=1),
            return_exceptions=True,
        )
        assert isinstance(owned, ExecutionInputConsumerError)
        assert isinstance(replayed, RuntimeError)
        assert [i.input_id for i in valid] == ["once"]
        assert await _receive(store, "valid") == valid
        assert await _receive(store, "valid", index=1) == ()


@pytest.mark.asyncio
async def test_sqlite_receive_batch_rolls_back_cursors_receipts_and_seals(
    tmp_path, monkeypatch
):
    async with SQLiteHistoryStore(tmp_path / "rollback.sqlite3") as store:
        for identity in ("input", "empty"):
            await _create_inbox_execution(store, identity)
        await store.send_execution_input("input", input_id="once", kind="a", value=1)
        original = store._batch_receive_inputs
        failed = False

        async def fail_after_write(db, payloads):
            nonlocal failed
            result = await original(db, payloads)
            if len(payloads) > 1:
                failed = True
                raise RuntimeError("injected after receipt and seal writes")
            return result

        monkeypatch.setattr(store, "_batch_receive_inputs", fail_after_write)
        received, empty = await asyncio.gather(
            _receive(store, "input"), _receive(store, "empty", seal=True)
        )
        assert failed and [i.input_id for i in received] == ["once"] and empty == ()
        assert (
            await store.send_execution_input(
                "empty", input_id="late", kind="a", value=1
            )
        ).status == "execution_finished"
        assert (
            await store._db().execute_fetchall(
                "SELECT COUNT(*) FROM execution_input_receives"
            )
        )[0][0] == 2


@pytest.mark.asyncio
async def test_sqlite_cross_connection_send_and_seal_share_atomic_boundary(tmp_path):
    path = tmp_path / "seal-race.sqlite3"
    async with SQLiteHistoryStore(path) as receiver, SQLiteHistoryStore(path) as sender:
        for i in range(8):
            identity = str(i)
            await _create_inbox_execution(receiver, identity)
            received, sent = await asyncio.gather(
                _receive(receiver, identity, seal=True),
                sender.send_execution_input(
                    identity, input_id="race", kind="a", value=i
                ),
            )
            assert (sent.status == "accepted") == bool(received)
            assert await _receive(receiver, identity, seal=True) == received


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 2])
async def test_sqlite_receive_limit_bounds_payload_materialization(tmp_path, count):
    async with SQLiteHistoryStore(tmp_path / "payload-limit.sqlite3") as store:
        for identity in map(str, range(count)):
            await _create_inbox_execution(store, identity)
            for i in range(8):
                await store.send_execution_input(identity, input_id=str(i), kind="a", value="x" * 4096)
        db = store._db()
        reads = []

        def read_payload(value):
            reads.append(value)
            return value

        await db.create_function("read_payload", 1, read_payload)
        await db.execute("ALTER TABLE execution_inputs RENAME TO stored_inputs")
        await db.execute(
            "CREATE VIEW execution_inputs AS SELECT execution_id,input_id,sequence,kind,"
            "read_payload(value_json) AS value_json FROM stored_inputs"
        )
        await db.commit()
        results = await asyncio.gather(*(_receive(store, str(i), limit=1) for i in range(count)))
        assert all([i.sequence for i in result] == [0] for result in results)
        assert len(reads) == count


@pytest.mark.asyncio
async def test_sqlite_stale_receive_cannot_consume_or_seal_and_valid_peer_survives(
    tmp_path,
):
    from pygent.runtime._history_ownership import owner_scope
    from pygent.runtime._history_types import HistoryConflictError

    async with SQLiteHistoryStore(tmp_path / "stale-receive.sqlite3") as store:
        for identity in ("stale", "valid"):
            await _create_inbox_execution(store, identity)
        fence = await store.claim_execution(
            execution_id="stale", owner_id="old", lease_ttl=30
        )
        await store._db().execute(
            "UPDATE execution_claims SET expires_at=0 WHERE execution_id='stale'"
        )
        await store._db().commit()
        replacement = await store.claim_execution(
            execution_id="stale", owner_id="new", lease_ttl=30
        )
        with owner_scope("stale", "old", fence):
            stale, valid = await asyncio.gather(
                _receive(store, "stale", seal=True),
                _receive(store, "valid"),
                return_exceptions=True,
            )
        assert isinstance(stale, HistoryConflictError) and valid == ()
        assert (
            await store.send_execution_input(
                "stale", input_id="still-open", kind="a", value=1
            )
        ).status == "accepted"
        with owner_scope("stale", "new", replacement):
            assert [i.input_id for i in await _receive(store, "stale")] == [
                "still-open"
            ]
