from __future__ import annotations

import asyncio
import time

import pytest

from pygent import AIMessage, Context, Module, UserMessage
from pygent.runtime import (
    ExecutionOptions,
    ExecutionStatus,
    HistoryConflictError,
    HistoryStoreError,
    LocalRuntime,
    SQLiteHistoryStore,
)
from pygent.runtime._history_ownership import owner_scope


@pytest.mark.asyncio
async def test_expired_writer_cannot_mutate_or_release_new_owner(tmp_path):
    path = tmp_path / "fences.sqlite3"
    async with SQLiteHistoryStore(path) as old, SQLiteHistoryStore(path) as new:
        await old.create_execution(
            execution_id="run", request_id="req", plan_id="plan", input={}
        )
        first = await old.claim_execution(
            execution_id="run", owner_id="old", lease_ttl=10
        )
        # Advance the persisted lease deterministically, without a timing race.
        await old._db().execute("UPDATE execution_claims SET expires_at=0")
        await old._db().commit()
        assert not await old.renew_execution_claim(
            execution_id="run", owner_id="old", fencing_token=first, lease_ttl=10
        )
        second = await new.claim_execution(
            execution_id="run", owner_id="new", lease_ttl=10
        )
        assert second > first
        with owner_scope("run", "old", first):
            operations = (
                old.update_execution("run", status="failed"),
                old.append_event(execution_id="run", index=0, event={"sequence": 0}),
                old.record_effect(
                    execution_id="run",
                    module_path="root",
                    call_index=0,
                    effect_type="test",
                    request={},
                    result={},
                ),
                old.finalize_execution(
                    "run",
                    status="succeeded",
                    output={},
                    error=None,
                    terminal_events=((0, {"sequence": 0}),),
                    terminal_sequence=0,
                ),
            )
            errors = await asyncio.gather(*operations, return_exceptions=True)
            assert all(isinstance(error, HistoryConflictError) for error in errors)
        assert await new.renew_execution_claim(
            execution_id="run", owner_id="new", fencing_token=second, lease_ttl=10
        )
        assert await old.events_after(execution_id="run") == ()
        with owner_scope("run", "new", second):
            await new.finalize_execution(
                "run",
                status="succeeded",
                output={},
                error=None,
                terminal_events=((0, {"sequence": 0}),),
                terminal_sequence=0,
            )
        assert (await new.get_execution("run")).status == "succeeded"
        assert (
            await new.claim_execution(
                execution_id="run", owner_id="third", lease_ttl=10
            )
            is None
        )


@pytest.mark.asyncio
async def test_event_batch_isolates_stale_writer_from_valid_writer(tmp_path):
    async with SQLiteHistoryStore(tmp_path / "batch.sqlite3") as store:
        token = await store.claim_execution(
            execution_id="stale", owner_id="new", lease_ttl=10
        )
        with owner_scope("stale", "old", token - 1):
            await store._reserve_event_slot()
            stale = store._enqueue_reserved_event_payload("stale", 0, '{"sequence":0}')
        await store._reserve_event_slot()
        valid = store._enqueue_reserved_event_payload("valid", 0, '{"sequence":0}')
        results = await asyncio.gather(stale, valid, return_exceptions=True)
        assert isinstance(results[0], HistoryConflictError)
        assert results[1] is None
        assert len(await store.events_after(execution_id="valid")) == 1
        assert not await store.events_after(execution_id="stale")


@pytest.mark.asyncio
async def test_uncommitted_event_is_not_visible_to_observers(tmp_path, monkeypatch):
    async with SQLiteHistoryStore(tmp_path / "atomic.sqlite3") as store:
        entered, release = asyncio.Event(), asyncio.Event()

        async def broken_commit():
            entered.set()
            await release.wait()
            raise HistoryStoreError("commit failed")

        monkeypatch.setattr(store._db(), "commit", broken_commit)
        append = asyncio.create_task(
            store.append_event(execution_id="run", index=0, event={"sequence": 0})
        )
        await entered.wait()
        read = asyncio.create_task(store.events_after(execution_id="run"))
        await asyncio.sleep(0)
        assert not read.done()
        release.set()
        with pytest.raises(HistoryStoreError):
            await append
        assert await read == ()


@pytest.mark.asyncio
async def test_terminal_commit_invalidates_later_write_in_same_batch(tmp_path):
    async with SQLiteHistoryStore(tmp_path / "terminal-fence.sqlite3") as history:
        await history.create_execution(
            execution_id="run", request_id="req", plan_id="plan", input={}
        )
        token = await history.claim_execution(
            execution_id="run", owner_id="owner", lease_ttl=10
        )
        with owner_scope("run", "owner", token):
            results = await asyncio.gather(
                history.finalize_execution(
                    "run",
                    status="succeeded",
                    output={},
                    error=None,
                    terminal_events=((0, {"sequence": 0}),),
                    terminal_sequence=0,
                ),
                history.update_execution("run", status="running"),
                return_exceptions=True,
            )
        assert results[0] is None
        assert isinstance(results[1], HistoryConflictError)
        assert (await history.get_execution("run")).status == "succeeded"


@pytest.mark.asyncio
async def test_worker_job_and_journal_share_the_same_fencing_authority(tmp_path):
    async with SQLiteHistoryStore(tmp_path / "worker-fence.sqlite3") as history:
        await history.put_task(task_id="job", kind="job", status="running", request={})
        token = await history.claim_execution(
            execution_id="worker-job:job", owner_id="owner", lease_ttl=10
        )
        with owner_scope("worker-job:job", "stale", token - 1, journal_id="job"):
            failures = await asyncio.gather(
                history.put_task(
                    task_id="job", kind="job", status="failed", request={}
                ),
                history.append_event(
                    execution_id="job", index=0, event={"sequence": 0}
                ),
                history._finalize_worker_job(
                    task_id="job",
                    status="succeeded",
                    result={},
                    error=None,
                    index=0,
                    event={"sequence": 0},
                ),
                return_exceptions=True,
            )
            assert all(isinstance(error, HistoryConflictError) for error in failures)
        assert (await history.get_task("job")).status == "running"
        assert not await history.events_after(execution_id="job")


@pytest.mark.asyncio
async def test_cancel_during_identity_commit_finishes_one_durable_outcome(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()

    class GatedHistory(SQLiteHistoryStore):
        async def begin_execution(self, **kwargs):
            stored = await super().begin_execution(**kwargs)
            entered.set()
            await release.wait()
            return stored

    class Unreachable(Module):
        async def forward(self, message, context):
            raise AssertionError("cancelled submission must not execute")

    async with (
        GatedHistory(tmp_path / "submission.sqlite3") as history,
        LocalRuntime(history=history) as runtime,
    ):
        handle = await runtime.bind(Unreachable()).start(
            UserMessage(content="hello"), Context()
        )
        await entered.wait()
        cancellation = asyncio.create_task(handle.cancel())
        await asyncio.sleep(0)
        release.set()
        assert await cancellation
        assert (await handle.outcome()).status is ExecutionStatus.CANCELLED
        stored = await history.get_execution(handle.execution_id)
        assert stored.status == "cancelled"
        assert stored.terminal_sequence is not None
        assert not (await history._execution_status(handle.execution_id)).owner_active


@pytest.mark.asyncio
@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("wait_method", ["result", "outcome"])
async def test_cancelled_observer_does_not_cancel_execution(managed, wait_method):
    entered, release, exited = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class Blocking(Module):
        async def forward(self, message, context):
            entered.set()
            try:
                await release.wait()
                return AIMessage(content="done"), context
            finally:
                exited.set()

    async with LocalRuntime() as runtime:
        module = runtime.bind(Blocking()) if managed else Blocking()
        handle = await module.start(UserMessage(content="hello"), Context())
        await entered.wait()
        observer = asyncio.create_task(getattr(handle, wait_method)())
        await asyncio.sleep(0)
        observer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observer
        assert not exited.is_set()
        release.set()
        assert (await handle.result())[0].content == "done"
        assert (await handle.outcome()).status is ExecutionStatus.SUCCEEDED


@pytest.mark.asyncio
@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("mode", ["invoke", "stream"])
async def test_cancelling_owned_call_joins_execution(managed, mode):
    entered, exited = asyncio.Event(), asyncio.Event()

    class Blocking(Module):
        async def forward(self, message, context):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                exited.set()

    async with LocalRuntime() as runtime:
        module = runtime.bind(Blocking()) if managed else Blocking()

        async def invoke():
            if mode == "invoke":
                await module.invoke(UserMessage(content="hello"), Context())
            else:
                async with module.stream(
                    UserMessage(content="hello"), Context()
                ) as stream:
                    await stream.final_result()

        caller = asyncio.create_task(invoke())
        await entered.wait()
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert exited.is_set()


@pytest.mark.asyncio
async def test_business_timeout_is_failure_including_empty_message():
    class Child(Module):
        async def forward(self, message, context):
            raise TimeoutError

    class Parent(Module):
        def __init__(self):
            super().__init__()
            self.child = Child()

        async def forward(self, message, context):
            return await self.child(message, context)

    async with LocalRuntime() as runtime:
        handle = await runtime.bind(Parent()).start(
            UserMessage(content="hello"), Context()
        )
        with pytest.raises(TimeoutError):
            await handle.result()
        outcome = await handle.outcome()
        assert outcome.status is ExecutionStatus.FAILED
        assert outcome.error.kind == "TimeoutError"
        assert outcome.error.message
        async with handle.subscribe() as events:
            assert not any(["deadline" in event.kind async for event in events])


@pytest.mark.asyncio
async def test_durable_cancel_and_state_polling_across_connections(
    tmp_path, monkeypatch
):
    entered = asyncio.Event()

    class Blocking(Module):
        async def forward(self, message, context):
            entered.set()
            await asyncio.Event().wait()

    path = tmp_path / "cancel.sqlite3"
    async with (
        SQLiteHistoryStore(path) as owner_store,
        SQLiteHistoryStore(path) as observer_store,
        LocalRuntime(history=owner_store) as runtime,
        LocalRuntime(history=observer_store) as observer,
    ):
        handle = await runtime.bind(Blocking()).start(
            UserMessage(content="large" * 10000), Context()
        )
        await entered.wait()
        attached = await observer.get_execution_handle(handle.execution_id)

        async def forbidden(*args, **kwargs):
            raise AssertionError("status observation must not decode the invocation")

        monkeypatch.setattr(observer_store, "get_execution", forbidden)
        assert (await attached.snapshot()).owner_state.value == "active"
        assert await attached.cancel()
        assert (
            await asyncio.wait_for(attached.outcome(), 1)
        ).status is ExecutionStatus.CANCELLED
        assert not await attached.cancel()


@pytest.mark.asyncio
async def test_failed_finalization_wakes_subscriber_without_fabricating_terminal(
    tmp_path,
):
    class BrokenHistory(SQLiteHistoryStore):
        async def finalize_execution(self, *args, **kwargs):
            raise HistoryStoreError("journal unavailable")

    class Echo(Module):
        async def forward(self, message, context):
            return AIMessage(content="done"), context

    async with (
        BrokenHistory(tmp_path / "broken.sqlite3") as store,
        LocalRuntime(history=store) as runtime,
    ):
        handle = await runtime.bind(Echo()).start(
            UserMessage(content="hello"), Context()
        )
        observed = []

        async def consume():
            async with handle.subscribe() as events:
                async for event in events:
                    observed.append(event)

        consumer = asyncio.create_task(consume())
        with pytest.raises(HistoryStoreError, match="journal unavailable"):
            await handle.result()
        with pytest.raises(HistoryStoreError, match="journal unavailable"):
            await asyncio.wait_for(consumer, 1)
        with pytest.raises(HistoryStoreError):
            await handle.outcome()
        assert not any(event.kind == "execution.completed" for event in observed)
        assert (await handle.snapshot()).terminal_sequence is None
        assert (
            await store.get_execution(handle.execution_id)
        ).terminal_sequence is None


@pytest.mark.asyncio
async def test_deadline_bounds_finalization_and_close_joins_pending_commit(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()

    class GatedHistory(SQLiteHistoryStore):
        async def finalize_execution(self, *args, **kwargs):
            entered.set()
            await release.wait()
            await super().finalize_execution(*args, **kwargs)

    class Echo(Module):
        async def forward(self, message, context):
            return AIMessage(content="done"), context

    async with GatedHistory(tmp_path / "slow.sqlite3") as store:
        runtime = LocalRuntime(history=store)
        handle = await runtime.bind(Echo()).start(
            UserMessage(content="hello"),
            Context(),
            execution=ExecutionOptions(deadline=time.monotonic() + 0.1),
        )
        await entered.wait()
        with pytest.raises(HistoryStoreError, match="cleanup"):
            await asyncio.wait_for(handle.result(), 2)
        assert (await handle.snapshot()).terminal_sequence is None
        closing = asyncio.create_task(runtime.close())
        await asyncio.sleep(0)
        assert not closing.done()
        release.set()
        await asyncio.wait_for(closing, 1)
        assert (await store.get_execution(handle.execution_id)).status == "succeeded"


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 200, 1000])
async def test_cancellation_queries_are_store_scoped(tmp_path, count):
    async with SQLiteHistoryStore(tmp_path / "watch.sqlite3") as store:
        queries = []
        await store._db().set_trace_callback(
            lambda sql: queries.append(sql) if "SELECT execution_id FROM execution_cancellations" in sql else None
        )
        owners = [asyncio.create_task(asyncio.Event().wait()) for _ in range(count)]
        try:
            for index, owner in enumerate(owners):
                store._watch_execution_cancel(str(index), owner)
            await asyncio.sleep(0.23)
            assert 1 <= len(queries) <= 6
            assert not any(owner.done() for owner in owners)
            for index, owner in enumerate(owners):
                store._unwatch_execution_cancel(str(index), owner)
            watcher = store._cancel_watcher
            await asyncio.wait_for(asyncio.shield(watcher), 1)
            total = len(queries)
            await asyncio.sleep(0.06)
            assert len(queries) == total
        finally:
            for owner in owners:
                owner.cancel()
            await asyncio.gather(*owners, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancel_watcher_failure_stops_owners_and_can_restart(tmp_path, monkeypatch):
    async with SQLiteHistoryStore(tmp_path / "broken-watch.sqlite3") as store:
        original = store._db().execute_fetchall

        async def broken(*args, **kwargs):
            raise HistoryStoreError("read failed")

        monkeypatch.setattr(store._db(), "execute_fetchall", broken)
        owners = [asyncio.create_task(asyncio.Event().wait()) for _ in range(4)]
        for index, owner in enumerate(owners):
            store._watch_execution_cancel(str(index), owner)
        await asyncio.wait_for(asyncio.gather(*owners, return_exceptions=True), 1)
        assert all(owner.cancelled() for owner in owners)
        assert not store._cancel_owners
        monkeypatch.setattr(store._db(), "execute_fetchall", original)
        owner = asyncio.create_task(asyncio.Event().wait())
        store._watch_execution_cancel("new", owner)
        await asyncio.sleep(0.06)
        assert not owner.done()
    assert store._cancel_watcher is None
    await asyncio.gather(owner, return_exceptions=True)
    assert owner.cancelled()


@pytest.mark.asyncio
async def test_rolled_back_cancel_does_not_cancel_owner(tmp_path):
    async with SQLiteHistoryStore(tmp_path / "rollback-watch.sqlite3") as store:
        owner = asyncio.create_task(asyncio.Event().wait())
        store._watch_execution_cancel("run", owner)
        try:
            async with store._write_lock:
                await store._db().execute("INSERT INTO execution_cancellations VALUES('run')")
                await asyncio.sleep(0.08)
                await store._db().rollback()
            await asyncio.sleep(0.08)
            assert not owner.done()
            # Removing a stale attempt must not unregister its replacement.
            stale = asyncio.create_task(asyncio.sleep(0))
            store._unwatch_execution_cancel("run", stale)
            assert store._cancel_owners["run"] is owner
            await stale
        finally:
            store._unwatch_execution_cancel("run", owner)
            owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)


@pytest.mark.asyncio
async def test_local_cancel_notifies_without_poll_and_watcher_drops_caller_context(tmp_path, monkeypatch):
    from contextvars import ContextVar

    tenant = ContextVar("test_tenant", default=None)
    async with SQLiteHistoryStore(tmp_path / "local-watch.sqlite3") as store:
        await store.create_execution(execution_id="run", request_id="req", plan_id="plan", input={})
        observed = asyncio.Event()
        original = store._db().execute_fetchall

        async def read(*args, **kwargs):
            assert tenant.get() is None
            observed.set()
            return await original(*args, **kwargs)

        monkeypatch.setattr(store._db(), "execute_fetchall", read)
        owner = asyncio.create_task(asyncio.Event().wait())
        token = tenant.set("tenant-specific-value")
        try:
            store._watch_execution_cancel("run", owner)
        finally:
            tenant.reset(token)
        await asyncio.wait_for(observed.wait(), 1)
        # Exclude the watcher so cancellation must be delivered synchronously
        # after the local request commits, without waiting for a polling tick.
        store._cancel_watcher.cancel()
        await asyncio.gather(store._cancel_watcher, return_exceptions=True)
        assert await store._request_execution_cancel("run")
        assert owner.cancelling()
        await asyncio.gather(owner, return_exceptions=True)
        assert owner.cancelled()


@pytest.mark.asyncio
async def test_store_close_joins_registered_owner_before_closing_connection(tmp_path):
    store = await SQLiteHistoryStore(tmp_path / "close-watch.sqlite3").open()
    entered = asyncio.Event()
    exited = asyncio.Event()

    async def owner():
        try:
            entered.set()
            await asyncio.Event().wait()
        finally:
            await store._db().execute_fetchall("SELECT 1")
            exited.set()

    task = asyncio.create_task(owner())
    await entered.wait()
    store._watch_execution_cancel("run", task)
    await store.close()
    assert exited.is_set() and task.cancelled()
    assert store._connection is None


@pytest.mark.asyncio
async def test_shared_watcher_receives_cancellation_from_another_process(tmp_path):
    import sys

    path = tmp_path / "process-watch.sqlite3"
    async with SQLiteHistoryStore(path) as store:
        owner = asyncio.create_task(asyncio.Event().wait())
        store._watch_execution_cancel("run", owner)
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c",
            "import sqlite3, sys; db = sqlite3.connect(sys.argv[1]); "
            "db.execute(\"INSERT INTO execution_cancellations VALUES('run')\"); "
            "db.commit(); db.close()",
            str(path),
        )
        try:
            assert await asyncio.wait_for(process.wait(), 5) == 0
            await asyncio.wait_for(asyncio.gather(owner, return_exceptions=True), 2)
            assert owner.cancelled()
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
            store._unwatch_execution_cancel("run", owner)
            owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)
