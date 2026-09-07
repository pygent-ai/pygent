"""SQLite history connection, schema lifecycle, and composition root."""

from __future__ import annotations

import asyncio
import json
from collections import Counter, deque
from collections.abc import Awaitable, Callable
from contextvars import Context
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, TypeVar, cast

import aiosqlite

from ._history_effects import EffectHistoryMixin
from ._history_executions import ExecutionHistoryMixin
from ._history_inputs import ExecutionInputHistoryMixin
from ._history_jobs import JobHistoryMixin
from ._history_ownership import WriteAuthority, validate_writers, write_authority
from ._history_types import HistoryStoreError

_T = TypeVar("_T")


@dataclass(slots=True)
class _QueuedEvent:
    execution_id: str
    index: int
    payload: str
    authority: WriteAuthority
    on_commit: Callable[[int], None] | None = None
    on_error: Callable[[BaseException], None] | None = None


@dataclass(slots=True)
class _EventBatch:
    events: list[_QueuedEvent]
    committed: dict[WriteAuthority, asyncio.Future[None]]


@dataclass(slots=True)
class _TransactionRequest:
    operation: Callable[[aiosqlite.Connection], Awaitable[object]]
    committed: asyncio.Future[object]
    authority: WriteAuthority | None
    batch_key: str | None = None
    batch_payload: object | None = None
    batch_operation: (
        Callable[[aiosqlite.Connection, list[object]], Awaitable[list[object]]] | None
    ) = None


class SQLiteHistoryStore(
    ExecutionHistoryMixin,
    JobHistoryMixin,
    EffectHistoryMixin,
    ExecutionInputHistoryMixin,
):
    """One serialized SQLite durability boundary for executions and effects."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_event_batch_size: int = 64,
        max_pending_event_batches: int = 16,
        max_transaction_batch_size: int = 64,
        max_pending_transactions: int = 1024,
    ) -> None:
        if (
            not isinstance(max_event_batch_size, int)
            or isinstance(max_event_batch_size, bool)
            or max_event_batch_size <= 0
        ):
            raise ValueError("max_event_batch_size must be a positive integer")
        if (
            not isinstance(max_pending_event_batches, int)
            or isinstance(max_pending_event_batches, bool)
            or max_pending_event_batches <= 0
        ):
            raise ValueError("max_pending_event_batches must be a positive integer")
        for name, value in (
            ("max_transaction_batch_size", max_transaction_batch_size),
            ("max_pending_transactions", max_pending_transactions),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.path = str(path)
        self._connection: aiosqlite.Connection | None = None
        self._cancel_owners: dict[str, asyncio.Task[Any]] = {}
        self._cancel_watcher: asyncio.Task[None] | None = None
        self._write_lock = asyncio.Lock()
        self._event_batches: deque[_EventBatch] = deque()
        self._event_flush_task: asyncio.Task[None] | None = None
        self._max_event_batch_size = max_event_batch_size
        self._event_capacity = asyncio.Semaphore(
            max_event_batch_size * max_pending_event_batches
        )
        self._transaction_queue: deque[_TransactionRequest] = deque()
        self._transaction_writer_task: asyncio.Task[None] | None = None
        self._transaction_capacity = asyncio.Semaphore(max_pending_transactions)
        self._max_transaction_batch_size = max_transaction_batch_size

    async def open(self) -> Self:
        if self._connection is not None:
            return self
        self._connection = await aiosqlite.connect(self.path)
        await self._connection.execute("PRAGMA journal_mode=WAL")
        await self._connection.execute("PRAGMA foreign_keys=ON")
        user_version_row = await (
            await self._connection.execute("PRAGMA user_version")
        ).fetchone()
        user_version = 0 if user_version_row is None else int(user_version_row[0])
        tables = {
            row[0]
            for row in await (
                await self._connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%'"
                )
            ).fetchall()
        }
        if tables and user_version != 7:
            await self.close()
            raise HistoryStoreError(
                "SQLite history schema is incompatible; this Runtime requires schema v7"
            )
        await self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS executions (
                execution_id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL,
                status TEXT NOT NULL,
                plan_id TEXT NOT NULL,
                input_json TEXT NOT NULL,
                output_json TEXT,
                error_json TEXT,
                attempt INTEGER NOT NULL DEFAULT 1,
                binding_id TEXT NOT NULL DEFAULT '',
                identity TEXT NOT NULL DEFAULT '',
                idempotency_key TEXT,
                model_calls_json TEXT NOT NULL DEFAULT '{}',
                model_admission_id TEXT,
                model_admission_digest TEXT,
                model_admission_status TEXT NOT NULL DEFAULT 'none',
                trace_id TEXT NOT NULL,
                phase TEXT NOT NULL,
                attempt_id TEXT,
                terminal_sequence INTEGER,
                submitted_at_unix_ns INTEGER NOT NULL,
                updated_at_unix_ns INTEGER NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL CHECK(kind IN ('job', 'tool_task')),
                status TEXT NOT NULL,
                request_json TEXT NOT NULL,
                result_json TEXT,
                error_json TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL UNIQUE,
                logical_key TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL,
                binding_id TEXT NOT NULL,
                plan_id TEXT NOT NULL,
                resource_key TEXT,
                required_capabilities_json TEXT NOT NULL,
                request_json TEXT NOT NULL,
                result_json TEXT,
                error_json TEXT,
                attempt INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS effects (
                execution_id TEXT NOT NULL,
                module_path TEXT NOT NULL,
                call_index INTEGER NOT NULL,
                effect_type TEXT NOT NULL,
                request_digest TEXT NOT NULL,
                spec_json TEXT,
                status TEXT NOT NULL,
                result_json TEXT,
                PRIMARY KEY(execution_id, module_path, call_index)
            );
            CREATE TABLE IF NOT EXISTS checkpoints (
                execution_id TEXT NOT NULL,
                checkpoint_index INTEGER NOT NULL,
                graph_hash TEXT NOT NULL,
                state_json TEXT NOT NULL,
                PRIMARY KEY(execution_id, checkpoint_index)
            );
            CREATE TABLE IF NOT EXISTS events (
                execution_id TEXT NOT NULL,
                event_index INTEGER NOT NULL,
                event_json TEXT NOT NULL,
                PRIMARY KEY(execution_id, event_index)
            );
            CREATE TABLE IF NOT EXISTS execution_fences (
                fencing_token INTEGER PRIMARY KEY AUTOINCREMENT
            );
            CREATE TABLE IF NOT EXISTS execution_claims (
                execution_id TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                fencing_token INTEGER NOT NULL,
                expires_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS execution_cancellations (
                execution_id TEXT PRIMARY KEY
            );
            CREATE TABLE IF NOT EXISTS execution_model_admissions (
                execution_id TEXT NOT NULL,
                admission_id TEXT NOT NULL,
                PRIMARY KEY(execution_id, admission_id)
            );
            CREATE TABLE IF NOT EXISTS execution_inboxes (
                execution_id TEXT PRIMARY KEY,
                next_sequence INTEGER NOT NULL,
                sealed INTEGER NOT NULL CHECK(sealed IN (0,1))
            );
            CREATE TABLE IF NOT EXISTS execution_inputs (
                execution_id TEXT NOT NULL,
                input_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                kind TEXT NOT NULL,
                value_json TEXT NOT NULL,
                PRIMARY KEY(execution_id,input_id),
                UNIQUE(execution_id,sequence)
            );
            CREATE TABLE IF NOT EXISTS execution_input_consumers (
                execution_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                module_path TEXT NOT NULL,
                last_sequence INTEGER NOT NULL,
                PRIMARY KEY(execution_id,kind)
            );
            CREATE TABLE IF NOT EXISTS execution_input_receives (
                execution_id TEXT NOT NULL,
                module_path TEXT NOT NULL,
                receive_index INTEGER NOT NULL,
                request_json TEXT NOT NULL,
                batch_json TEXT NOT NULL,
                PRIMARY KEY(execution_id,module_path,receive_index)
            );
            """
        )
        await self._connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS jobs_logical_key ON jobs(logical_key)"
        )
        await self._connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS executions_idempotency_scope "
            "ON executions(binding_id,identity,idempotency_key) "
            "WHERE idempotency_key IS NOT NULL"
        )
        await self._connection.execute("PRAGMA user_version=7")
        await self._connection.commit()
        return self

    def _watch_execution_cancel(self, execution_id: str, owner: asyncio.Task[Any]) -> None:
        self._db()
        self._cancel_owners[execution_id] = owner
        if self._cancel_watcher is None or self._cancel_watcher.done():
            self._cancel_watcher = asyncio.create_task(
                self._watch_cancellations(),
                name="pygent-sqlite-cancellations",
                context=Context(),
            )

    def _unwatch_execution_cancel(self, execution_id: str, owner: asyncio.Task[Any]) -> None:
        if self._cancel_owners.get(execution_id) is owner:
            del self._cancel_owners[execution_id]

    def _cancel_execution_owner(self, execution_id: str) -> None:
        owner = self._cancel_owners.pop(execution_id, None)
        if owner is not None:
            owner.cancel()

    async def _watch_cancellations(self) -> None:
        try:
            while self._cancel_owners:
                await asyncio.sleep(0.05)
                owners = self._cancel_owners.copy()
                if not owners:
                    return
                # One queue round-trip per Store tick, independent of owner count.
                # The read lock excludes cancellation requests that later roll back.
                async with self._write_lock:
                    rows = await self._db().execute_fetchall(
                        "SELECT execution_id FROM execution_cancellations "
                        "WHERE execution_id IN (SELECT value FROM json_each(?))",
                        (json.dumps(tuple(owners)),),
                    )
                for (execution_id,) in rows:
                    if self._cancel_owners.get(execution_id) is owners[execution_id]:
                        self._cancel_execution_owner(execution_id)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - stop owners when cancellation cannot be observed
            owners, self._cancel_owners = self._cancel_owners, {}
            for owner in owners.values():
                owner.cancel()

    async def close(self) -> None:
        if asyncio.current_task() in self._cancel_owners.values():
            raise HistoryStoreError("an execution cannot close its own history store")
        owners, self._cancel_owners = self._cancel_owners, {}
        for owner in owners.values():
            owner.cancel()
        watcher, self._cancel_watcher = self._cancel_watcher, None
        if watcher is not None:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
        if owners:
            await asyncio.gather(*owners.values(), return_exceptions=True)
        flush_task = self._event_flush_task
        if flush_task is not None:
            await asyncio.shield(flush_task)
        transaction_task = self._transaction_writer_task
        if transaction_task is not None:
            await asyncio.shield(transaction_task)
        connection, self._connection = self._connection, None
        if connection is not None:
            await connection.close()

    async def _queue_event(self, execution_id: str, index: int, payload: str) -> None:
        await self._reserve_event_slot()
        try:
            committed = self._enqueue_reserved_event_payload(
                execution_id, index, payload
            )
        except BaseException:
            self._event_capacity.release()
            raise
        await asyncio.shield(committed)

    async def _reserve_event_slot(self) -> None:
        self._db()
        await self._event_capacity.acquire()
        try:
            self._db()
        except BaseException:
            self._event_capacity.release()
            raise

    def _enqueue_reserved_event_payload(
        self,
        execution_id: str,
        index: int,
        payload: str,
        *,
        on_commit: Callable[[int], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> asyncio.Future[None]:
        """Queue an event synchronously and return its batch commit receipt."""

        self._db()
        loop = asyncio.get_running_loop()
        if (
            not self._event_batches
            or len(self._event_batches[-1].events) >= self._max_event_batch_size
        ):
            self._event_batches.append(_EventBatch([], {}))
        batch = self._event_batches[-1]
        authority = write_authority(execution_id)
        committed = batch.committed.get(authority)
        if committed is None:
            committed = loop.create_future()
            batch.committed[authority] = committed
        batch.events.append(
            _QueuedEvent(execution_id, index, payload, authority, on_commit, on_error)
        )
        task = self._event_flush_task
        if task is None or task.done():
            self._event_flush_task = asyncio.create_task(
                self._flush_event_batches(), name="pygent-sqlite-event-writer"
            )
        return committed

    async def _flush_event_batches(self) -> None:
        while True:
            if not self._event_batches:
                self._event_flush_task = None
                return
            batch = self._event_batches[0]
            if len(batch.events) < self._max_event_batch_size:
                await asyncio.sleep(0)
            batch = self._event_batches.popleft()
            try:
                await self._commit_event_batch(batch.events)
            except asyncio.CancelledError as exc:
                self._finish_event_batch(batch, exc)
                pending, self._event_batches = self._event_batches, deque()
                self._event_flush_task = None
                for queued in pending:
                    self._finish_event_batch(queued, exc)
                raise
            except Exception:  # noqa: BLE001 - isolate execution writers
                for authority, receipt in batch.committed.items():
                    isolated = _EventBatch(
                        [
                            event
                            for event in batch.events
                            if event.authority == authority
                        ],
                        {authority: receipt},
                    )
                    try:
                        await self._commit_event_batch(isolated.events)
                    except Exception as exc:  # noqa: BLE001 - report commit failure
                        self._finish_event_batch(isolated, exc)
                    else:
                        self._finish_event_batch(isolated)
            else:
                self._finish_event_batch(batch)

    def _finish_event_batch(
        self, batch: _EventBatch, error: BaseException | None = None
    ) -> None:
        for event in batch.events:
            self._event_capacity.release()
            if error is not None:
                if event.on_error is not None:
                    event.on_error(error)
            elif event.on_commit is not None:
                event.on_commit(event.index)
        for receipt in batch.committed.values():
            if not receipt.done():
                if error is None:
                    receipt.set_result(None)
                else:
                    receipt.set_exception(error)
                    # Observe unused receipts without scheduling a callback per writer.
                    receipt.exception()

    async def _commit_event_batch(self, batch: list[_QueuedEvent]) -> None:
        db = self._db()
        async with self._write_lock:
            await db.execute("BEGIN IMMEDIATE")
            try:
                await validate_writers(db, tuple(event.authority for event in batch))
                cursor = await db.executemany(
                    "INSERT INTO events(execution_id,event_index,event_json) "
                    "VALUES(?,?,?) ON CONFLICT(execution_id,event_index) DO NOTHING",
                    [
                        (event.execution_id, event.index, event.payload)
                        for event in batch
                    ],
                )
                if cursor.rowcount != len(batch):
                    for event in batch:
                        row = await (
                            await db.execute(
                                "SELECT event_json FROM events "
                                "WHERE execution_id=? AND event_index=?",
                                (event.execution_id, event.index),
                            )
                        ).fetchone()
                        if row is None or row[0] != event.payload:
                            from ._history_types import HistoryConflictError

                            raise HistoryConflictError(
                                "event cursor has conflicting content"
                            )
                await db.commit()
            except BaseException:
                await db.rollback()
                raise

    async def _queue_transaction(
        self,
        operation: Callable[[aiosqlite.Connection], Awaitable[_T]],
        *,
        execution_id: str | None = None,
        batch_key: str | None = None,
        batch_payload: object | None = None,
        batch_operation: (
            Callable[[aiosqlite.Connection, list[object]], Awaitable[list[object]]]
            | None
        ) = None,
    ) -> _T:
        if (batch_key is None) != (batch_operation is None):
            raise TypeError("batch_key and batch_operation must be provided together")
        self._db()
        await self._transaction_capacity.acquire()
        try:
            self._db()
            loop = asyncio.get_running_loop()
            committed: asyncio.Future[object] = loop.create_future()
            self._transaction_queue.append(
                _TransactionRequest(
                    cast(Callable[..., Awaitable[object]], operation),
                    committed,
                    None if execution_id is None else write_authority(execution_id),
                    batch_key,
                    batch_payload,
                    batch_operation,
                )
            )
            task = self._transaction_writer_task
            if task is None or task.done():
                self._transaction_writer_task = asyncio.create_task(
                    self._flush_transaction_batches(),
                    name="pygent-sqlite-transaction-writer",
                )
        except BaseException:
            self._transaction_capacity.release()
            raise
        return cast(_T, await asyncio.shield(committed))

    async def _flush_transaction_batches(self) -> None:
        while True:
            if not self._transaction_queue:
                self._transaction_writer_task = None
                return
            await asyncio.sleep(0)
            batch = [
                self._transaction_queue.popleft()
                for _ in range(
                    min(len(self._transaction_queue), self._max_transaction_batch_size)
                )
            ]
            try:
                results = await self._run_transaction_batch(batch)
            except asyncio.CancelledError as exc:
                self._finish_transaction_errors(batch, exc)
                pending, self._transaction_queue = self._transaction_queue, deque()
                self._finish_transaction_errors(list(pending), exc)
                self._transaction_writer_task = None
                raise
            except Exception:  # noqa: BLE001 - retry failed batch in isolation
                await self._retry_transaction_batch_individually(batch)
            else:
                self._finish_transaction_results(batch, results)

    async def _run_transaction_batch(
        self, batch: list[_TransactionRequest]
    ) -> list[object]:
        db = self._db()
        async with self._write_lock:
            await db.execute("BEGIN IMMEDIATE")
            try:
                authorities = Counter(
                    item.authority for item in batch if item.authority is not None
                )
                terminal_owners = {
                    item.authority
                    for item in batch
                    if item.batch_key == "finalize_execution"
                    and item.authority is not None
                }
                if terminal_owners and any(
                    authorities[owner] > 1
                    for owner in terminal_owners
                ):
                    # A terminal write ends this authority. Re-run such mixed
                    # requests in order, validating the fence before each one.
                    raise RuntimeError(
                        "terminal owner requires an isolated transaction"
                    )
                await validate_writers(
                    db,
                    tuple(authorities),
                )
                results: list[object] = []
                index = 0
                while index < len(batch):
                    request = batch[index]
                    if request.batch_key is None:
                        results.append(await request.operation(db))
                        index += 1
                        continue
                    end = index + 1
                    while (
                        end < len(batch) and batch[end].batch_key == request.batch_key
                    ):
                        end += 1
                    run = batch[index:end]
                    if len(run) == 1:
                        results.append(await request.operation(db))
                    else:
                        assert request.batch_operation is not None
                        batched = await request.batch_operation(
                            db,
                            [item.batch_payload for item in run],
                        )
                        if len(batched) != len(run):
                            raise RuntimeError(
                                "transaction batch operation returned the wrong result count"
                            )
                        results.extend(batched)
                    index = end
                await db.commit()
            except BaseException:
                await db.rollback()
                raise
        return results

    async def _retry_transaction_batch_individually(
        self, batch: list[_TransactionRequest]
    ) -> None:
        for request in batch:
            try:
                result = (await self._run_transaction_batch([request]))[0]
            except BaseException as exc:  # noqa: BLE001 - isolate failed request
                self._finish_transaction_errors([request], exc)
            else:
                self._finish_transaction_results([request], [result])

    def _finish_transaction_results(
        self, batch: list[_TransactionRequest], results: list[object]
    ) -> None:
        for request, result in zip(batch, results, strict=True):
            self._transaction_capacity.release()
            if not request.committed.done():
                request.committed.set_result(result)

    def _finish_transaction_errors(
        self, batch: list[_TransactionRequest], exc: BaseException
    ) -> None:
        for request in batch:
            self._transaction_capacity.release()
            if not request.committed.done():
                request.committed.set_exception(exc)
                request.committed.exception()

    async def __aenter__(self) -> Self:
        return await self.open()

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    def _db(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise HistoryStoreError("SQLiteHistoryStore is not open")
        return self._connection
