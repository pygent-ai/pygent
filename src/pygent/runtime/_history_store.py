"""SQLite history connection, schema lifecycle, and composition root."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Self

import aiosqlite

from ._history_effects import EffectHistoryMixin
from ._history_executions import ExecutionHistoryMixin
from ._history_jobs import JobHistoryMixin
from ._history_types import HistoryStoreError


class SQLiteHistoryStore(ExecutionHistoryMixin, JobHistoryMixin, EffectHistoryMixin):
    """One serialized SQLite durability boundary for executions and effects."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_event_batch_size: int = 64,
    ) -> None:
        if (
            not isinstance(max_event_batch_size, int)
            or isinstance(max_event_batch_size, bool)
            or max_event_batch_size <= 0
        ):
            raise ValueError("max_event_batch_size must be a positive integer")
        self.path = str(path)
        self._connection: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()
        self._event_batch_lock = asyncio.Lock()
        self._event_batch: list[tuple[str, int, str, asyncio.Future[None]]] = []
        self._event_flush_task: asyncio.Task[None] | None = None
        self._max_event_batch_size = max_event_batch_size

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
        if tables and user_version != 6:
            await self.close()
            raise HistoryStoreError(
                "SQLite history schema is incompatible; this Runtime requires schema v6"
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
            CREATE TABLE IF NOT EXISTS execution_model_admissions (
                execution_id TEXT NOT NULL,
                admission_id TEXT NOT NULL,
                PRIMARY KEY(execution_id, admission_id)
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
        await self._connection.execute("PRAGMA user_version=6")
        await self._connection.commit()
        return self

    async def close(self) -> None:
        flush_task = self._event_flush_task
        if flush_task is not None:
            await asyncio.shield(flush_task)
        connection, self._connection = self._connection, None
        if connection is not None:
            await connection.close()

    async def _queue_event(self, execution_id: str, index: int, payload: str) -> None:
        loop = asyncio.get_running_loop()
        committed: asyncio.Future[None] = loop.create_future()
        async with self._event_batch_lock:
            self._event_batch.append((execution_id, index, payload, committed))
            task = self._event_flush_task
            if task is None or task.done():
                task = asyncio.create_task(
                    self._flush_event_batches(), name="pygent-sqlite-event-writer"
                )
                self._event_flush_task = task
        await asyncio.shield(committed)

    async def _flush_event_batches(self) -> None:
        # One event-loop turn is enough to combine writes from concurrent
        # executions without imposing a timer on a single interactive stream.
        await asyncio.sleep(0)
        while True:
            async with self._event_batch_lock:
                batch = self._event_batch[: self._max_event_batch_size]
                del self._event_batch[: len(batch)]
                if not batch:
                    self._event_flush_task = None
                    return
            try:
                await self._commit_event_batch(batch)
            except BaseException as exc:  # noqa: BLE001 - fail every queued writer
                for *_, future in batch:
                    if not future.done():
                        future.set_exception(exc)
                async with self._event_batch_lock:
                    pending, self._event_batch = self._event_batch, []
                    self._event_flush_task = None
                for *_, future in pending:
                    if not future.done():
                        future.set_exception(exc)
                return
            else:
                for *_, future in batch:
                    if not future.done():
                        future.set_result(None)

    async def _commit_event_batch(
        self, batch: list[tuple[str, int, str, asyncio.Future[None]]]
    ) -> None:
        db = self._db()
        async with self._write_lock:
            await db.execute("BEGIN IMMEDIATE")
            try:
                cursor = await db.executemany(
                    "INSERT INTO events(execution_id,event_index,event_json) "
                    "VALUES(?,?,?) ON CONFLICT(execution_id,event_index) DO NOTHING",
                    [(execution_id, index, payload) for execution_id, index, payload, _ in batch],
                )
                if cursor.rowcount != len(batch):
                    for execution_id, index, payload, _ in batch:
                        row = await (
                            await db.execute(
                                "SELECT event_json FROM events "
                                "WHERE execution_id=? AND event_index=?",
                                (execution_id, index),
                            )
                        ).fetchone()
                        if row is None or row[0] != payload:
                            from ._history_types import HistoryConflictError

                            raise HistoryConflictError(
                                "event cursor has conflicting content"
                            )
                await db.commit()
            except BaseException:
                await db.rollback()
                raise

    async def __aenter__(self) -> Self:
        return await self.open()

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    def _db(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise HistoryStoreError("SQLiteHistoryStore is not open")
        return self._connection
