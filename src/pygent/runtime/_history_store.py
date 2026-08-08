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

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._connection: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()

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
        if tables and user_version not in {3, 4}:
            await self.close()
            raise HistoryStoreError(
                "SQLite history schema is incompatible; Pygent 0.2 requires schema v3 or v4"
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
        columns = {
            row[1] for row in await (await self._connection.execute(
                "PRAGMA table_info(executions)"
            )).fetchall()
        }
        for name, declaration in (
            ("binding_id", "TEXT NOT NULL DEFAULT ''"),
            ("identity", "TEXT NOT NULL DEFAULT ''"),
            ("idempotency_key", "TEXT"),
            ("model_calls_json", "TEXT NOT NULL DEFAULT '{}'"),
            ("model_admission_id", "TEXT"),
            ("model_admission_digest", "TEXT"),
            ("model_admission_status", "TEXT NOT NULL DEFAULT 'none'"),
        ):
            if name not in columns:
                await self._connection.execute(
                    f"ALTER TABLE executions ADD COLUMN {name} {declaration}"
                )
        effect_columns = await (
            await self._connection.execute("PRAGMA table_info(effects)")
        ).fetchall()
        effect_primary_key = tuple(
            row[1] for row in sorted(effect_columns, key=lambda row: row[5]) if row[5]
        )
        effect_column_names = {row[1] for row in effect_columns}
        result_not_null = next(
            (bool(row[3]) for row in effect_columns if row[1] == "result_json"),
            False,
        )
        if (
            effect_primary_key != ("execution_id", "module_path", "call_index")
            or "spec_json" not in effect_column_names
            or "status" not in effect_column_names
            or result_not_null
        ):
            await self._connection.executescript(
                """
                ALTER TABLE effects RENAME TO effects_legacy;
                CREATE TABLE effects (
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
                INSERT INTO effects(execution_id,module_path,call_index,effect_type,
                                    request_digest,spec_json,status,result_json)
                SELECT execution_id,module_path,call_index,effect_type,request_digest,
                       NULL,'completed',result_json
                FROM effects_legacy;
                DROP TABLE effects_legacy;
                """
            )
        await self._connection.execute("DROP INDEX IF EXISTS executions_request_id")
        job_columns = {
            row[1]
            for row in await (
                await self._connection.execute("PRAGMA table_info(jobs)")
            ).fetchall()
        }
        if "logical_key" not in job_columns:
            await self._connection.execute("ALTER TABLE jobs ADD COLUMN logical_key TEXT")
            await self._connection.execute(
                "UPDATE jobs SET logical_key='legacy:' || job_id WHERE logical_key IS NULL"
            )
        await self._connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS jobs_logical_key ON jobs(logical_key)"
        )
        await self._connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS executions_idempotency_scope "
            "ON executions(binding_id,identity,idempotency_key) "
            "WHERE idempotency_key IS NOT NULL"
        )
        await self._connection.execute("PRAGMA user_version=4")
        await self._connection.commit()
        return self

    async def close(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            await connection.close()

    async def __aenter__(self) -> Self:
        return await self.open()

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    def _db(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise HistoryStoreError("SQLiteHistoryStore is not open")
        return self._connection
