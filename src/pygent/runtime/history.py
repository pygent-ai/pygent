"""SQLite durable execution history and deterministic effect replay."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Concatenate, Literal, ParamSpec, Self, TypeVar, cast

import aiosqlite

from pygent.core import JsonValue, freeze_json, thaw_json


class HistoryStoreError(RuntimeError):
    """Base class for durable history failures."""


class HistoryConflictError(HistoryStoreError):
    """Raised when a stable identity is committed with different content."""


class NonDeterministicReplayError(HistoryStoreError):
    """Raised when replay reaches a boundary different from committed history."""


@dataclass(frozen=True, slots=True)
class StoredExecution:
    execution_id: str
    request_id: str
    status: str
    plan_id: str
    input: JsonValue
    output: JsonValue | None
    error: JsonValue | None
    attempt: int
    binding_id: str = ""
    identity: str = ""
    idempotency_key: str | None = None
    model_calls: JsonValue | None = None
    model_admission_id: str | None = None
    model_admission_digest: str | None = None
    model_admission_status: str = "none"


@dataclass(frozen=True, slots=True)
class StoredTask:
    task_id: str
    kind: Literal["job", "tool_task"]
    status: str
    request: JsonValue
    result: JsonValue | None
    error: JsonValue | None


@dataclass(frozen=True, slots=True)
class StoredJob:
    job_id: str
    task_id: str
    logical_key: str
    status: str
    binding_id: str
    plan_id: str
    resource_key: str | None
    required_capabilities: tuple[str, ...]
    request: JsonValue
    result: JsonValue | None
    error: JsonValue | None
    attempt: int


@dataclass(frozen=True, slots=True)
class StoredEffect:
    execution_id: str
    module_path: str
    call_index: int
    effect_type: str
    request_digest: str
    status: Literal["started", "completed", "unknown"]
    spec: JsonValue | None
    result: JsonValue | None


def _json(value: object) -> str:
    frozen = freeze_json(value)
    return json.dumps(thaw_json(frozen), sort_keys=True, separators=(",", ":"))


def _load(value: str | None) -> JsonValue | None:
    if value is None:
        return None
    return freeze_json(json.loads(value))


def effect_digest(request: object) -> str:
    """Return a stable digest for a provider-neutral effect request."""

    return hashlib.sha256(_json(request).encode("utf-8")).hexdigest()


_P = ParamSpec("_P")
_R = TypeVar("_R")


def _serialized_write(
    method: Callable[Concatenate[SQLiteHistoryStore, _P], Awaitable[_R]],
) -> Callable[Concatenate[SQLiteHistoryStore, _P], Awaitable[_R]]:
    """Serialize transactions on the Store's single writer connection."""

    @wraps(method)
    async def wrapped(
        self: SQLiteHistoryStore, /, *args: _P.args, **kwargs: _P.kwargs
    ) -> _R:
        async with self._write_lock:
            return await method(self, *args, **kwargs)

    return wrapped


class SQLiteHistoryStore:
    """Async SQLite store for Executions, tasks, effects, checkpoints and events."""

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

    async def claim_execution(
        self, *, execution_id: str, owner_id: str, lease_ttl: float
    ) -> int | None:
        """Atomically claim one durable recovery attempt across processes."""

        db = self._db()
        async with self._write_lock:
            await db.execute("BEGIN IMMEDIATE")
            try:
                await db.execute(
                    "DELETE FROM execution_claims "
                    "WHERE execution_id=? AND expires_at<=unixepoch('subsec')",
                    (execution_id,),
                )
                row = await (
                    await db.execute(
                        "SELECT owner_id FROM execution_claims WHERE execution_id=?", (execution_id,)
                    )
                ).fetchone()
                if row is not None:
                    await db.rollback()
                    return None
                cursor = await db.execute("INSERT INTO execution_fences DEFAULT VALUES")
                token = cursor.lastrowid
                assert token is not None
                await db.execute(
                    "INSERT INTO execution_claims VALUES(?,?,?,unixepoch('subsec')+?)",
                    (execution_id, owner_id, token, lease_ttl),
                )
                await db.commit()
                return int(token)
            except BaseException:
                await db.rollback()
                raise

    async def renew_execution_claim(
        self,
        *,
        execution_id: str,
        owner_id: str,
        fencing_token: int,
        lease_ttl: float,
    ) -> bool:
        async with self._write_lock:
            cursor = await self._db().execute(
                "UPDATE execution_claims SET expires_at=unixepoch('subsec')+? "
                "WHERE execution_id=? AND owner_id=? AND fencing_token=?",
                (lease_ttl, execution_id, owner_id, fencing_token),
            )
            await self._db().commit()
            return cursor.rowcount == 1

    async def release_execution_claim(
        self, *, execution_id: str, owner_id: str, fencing_token: int
    ) -> None:
        async with self._write_lock:
            await self._db().execute(
                "DELETE FROM execution_claims WHERE execution_id=? AND owner_id=? AND fencing_token=?",
                (execution_id, owner_id, fencing_token),
            )
            await self._db().commit()

    async def create_execution(
        self,
        *,
        execution_id: str,
        request_id: str,
        plan_id: str,
        input: object,
        status: str = "queued",
        binding_id: str = "",
        identity: str = "",
        idempotency_key: str | None = None,
        model_calls: object | None = None,
        model_admission_status: str = "none",
    ) -> StoredExecution:
        stored, _ = await self.begin_execution(
            execution_id=execution_id,
            request_id=request_id,
            plan_id=plan_id,
            input=input,
            status=status,
            binding_id=binding_id,
            identity=identity,
            idempotency_key=idempotency_key,
            model_calls=model_calls,
            model_admission_status=model_admission_status,
        )
        return stored

    @_serialized_write
    async def begin_execution(
        self,
        *,
        execution_id: str,
        request_id: str,
        plan_id: str,
        input: object,
        status: str = "queued",
        binding_id: str = "",
        identity: str = "",
        idempotency_key: str | None = None,
        model_calls: object | None = None,
        model_admission_status: str = "none",
    ) -> tuple[StoredExecution, bool]:
        db = self._db()
        payload = _json(input)
        model_calls_payload = _json({} if model_calls is None else model_calls)
        try:
            await db.execute(
                "INSERT INTO executions(execution_id,request_id,status,plan_id,input_json,"
                "binding_id,identity,idempotency_key,model_calls_json,model_admission_status) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    execution_id,
                    request_id,
                    status,
                    plan_id,
                    payload,
                    binding_id,
                    identity,
                    idempotency_key,
                    model_calls_payload,
                    model_admission_status,
                ),
            )
            await db.commit()
        except aiosqlite.IntegrityError as exc:
            await db.rollback()
            existing = (
                await self.get_execution_by_idempotency(
                    binding_id=binding_id,
                    identity=identity,
                    idempotency_key=idempotency_key,
                )
                if idempotency_key is not None
                else await self.get_execution(execution_id)
            )
            if existing is None or (
                existing.plan_id != plan_id
                or _json(existing.input) != payload
            ):
                raise HistoryConflictError(
                    "idempotency identity is committed with different input"
                ) from exc
            return existing, False
        result = await self.get_execution(execution_id)
        assert result is not None
        return result, True

    @_serialized_write
    async def commit_model_admission(
        self, execution_id: str, *, admission_id: str, manifest_digest: str
    ) -> None:
        cursor = await self._db().execute(
            "UPDATE executions SET model_admission_id=?,model_admission_digest=?,"
            "model_admission_status='committed',updated_at=CURRENT_TIMESTAMP "
            "WHERE execution_id=? AND model_admission_status IN ('preparing','committed')",
            (admission_id, manifest_digest, execution_id),
        )
        if cursor.rowcount != 1:
            await self._db().rollback()
            raise HistoryConflictError("model admission intent is not preparing")
        await self._db().execute(
            "INSERT OR IGNORE INTO execution_model_admissions(execution_id,admission_id) "
            "VALUES(?,?)",
            (execution_id, admission_id),
        )
        await self._db().commit()

    @_serialized_write
    async def add_model_admission_ref(
        self, execution_id: str, admission_id: str
    ) -> None:
        await self._db().execute(
            "INSERT OR IGNORE INTO execution_model_admissions(execution_id,admission_id) "
            "VALUES(?,?)",
            (execution_id, admission_id),
        )
        await self._db().commit()

    async def list_model_admission_refs(self, execution_id: str) -> tuple[str, ...]:
        rows = await (
            await self._db().execute(
                "SELECT admission_id FROM execution_model_admissions "
                "WHERE execution_id=? ORDER BY admission_id",
                (execution_id,),
            )
        ).fetchall()
        return tuple(str(row[0]) for row in rows)

    @_serialized_write
    async def abort_model_admission(self, execution_id: str) -> None:
        await self._db().execute(
            "UPDATE executions SET model_admission_status='aborted',"
            "updated_at=CURRENT_TIMESTAMP WHERE execution_id=? "
            "AND model_admission_status='preparing'",
            (execution_id,),
        )
        await self._db().commit()

    @_serialized_write
    async def delete_execution(self, execution_id: str) -> None:
        db = self._db()
        for table in ("events", "checkpoints", "effects"):
            await db.execute(f"DELETE FROM {table} WHERE execution_id=?", (execution_id,))
        await db.execute(
            "DELETE FROM execution_model_admissions WHERE execution_id=?",
            (execution_id,),
        )
        await db.execute("DELETE FROM executions WHERE execution_id=?", (execution_id,))
        await db.commit()

    @_serialized_write
    async def update_execution(
        self,
        execution_id: str,
        *,
        status: str,
        output: object | None = None,
        error: object | None = None,
        attempt: int | None = None,
    ) -> None:
        db = self._db()
        cursor = await db.execute(
            "UPDATE executions SET status=?, output_json=?, error_json=?, "
            "attempt=COALESCE(?,attempt), updated_at=CURRENT_TIMESTAMP WHERE execution_id=?",
            (
                status,
                None if output is None else _json(output),
                None if error is None else _json(error),
                attempt,
                execution_id,
            ),
        )
        if cursor.rowcount != 1:
            await db.rollback()
            raise KeyError(f"unknown execution {execution_id!r}")
        await db.commit()

    async def get_execution(self, execution_id: str) -> StoredExecution | None:
        return await self._select_run("execution_id", execution_id)

    async def get_execution_by_request(self, request_id: str) -> StoredExecution | None:
        return await self._select_run("request_id", request_id)

    async def get_execution_by_idempotency(
        self, *, binding_id: str, identity: str, idempotency_key: str
    ) -> StoredExecution | None:
        db = self._db()
        cursor = await db.execute(
            "SELECT execution_id,request_id,status,plan_id,input_json,output_json,"
            "error_json,attempt,binding_id,identity,idempotency_key,model_calls_json,"
            "model_admission_id,model_admission_digest,model_admission_status FROM executions "
            "WHERE binding_id=? AND identity=? AND idempotency_key=?",
            (binding_id, identity, idempotency_key),
        )
        row = await cursor.fetchone()
        return None if row is None else self._stored_execution(row)

    async def _select_run(self, column: str, value: str) -> StoredExecution | None:
        db = self._db()
        cursor = await db.execute(
            "SELECT execution_id,request_id,status,plan_id,input_json,output_json,"
            f"error_json,attempt,binding_id,identity,idempotency_key,model_calls_json,"
            f"model_admission_id,model_admission_digest,model_admission_status FROM executions "
            f"WHERE {column}=? ORDER BY updated_at DESC LIMIT 1",
            (value,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._stored_execution(row)

    @staticmethod
    def _stored_execution(row: Any) -> StoredExecution:
        return StoredExecution(
            execution_id=row[0],
            request_id=row[1],
            status=row[2],
            plan_id=row[3],
            input=_load(row[4]),
            output=_load(row[5]),
            error=_load(row[6]),
            attempt=row[7],
            binding_id=row[8],
            identity=row[9],
            idempotency_key=row[10],
            model_calls=_load(row[11]),
            model_admission_id=row[12],
            model_admission_digest=row[13],
            model_admission_status=row[14],
        )

    @_serialized_write
    async def put_task(
        self,
        *,
        task_id: str,
        kind: Literal["job", "tool_task"],
        status: str,
        request: object,
        result: object | None = None,
        error: object | None = None,
    ) -> None:
        request_json = _json(request)
        db = self._db()
        cursor = await db.execute(
            "INSERT INTO tasks(task_id,kind,status,request_json,result_json,error_json) "
            "VALUES(?,?,?,?,?,?) ON CONFLICT(task_id) DO UPDATE SET "
            "status=excluded.status,result_json=excluded.result_json,"
            "error_json=excluded.error_json,updated_at=CURRENT_TIMESTAMP "
            "WHERE tasks.kind=excluded.kind "
            "AND tasks.request_json=excluded.request_json",
            (
                task_id,
                kind,
                status,
                request_json,
                None if result is None else _json(result),
                None if error is None else _json(error),
            ),
        )
        if cursor.rowcount != 1:
            await db.rollback()
            raise HistoryConflictError(
                "task identity is already committed with a different kind or request"
            )
        await db.commit()

    @_serialized_write
    async def create_tool_job(
        self,
        *,
        job_id: str,
        task_id: str,
        logical_key: str,
        binding_id: str,
        plan_id: str,
        resource_key: str | None,
        required_capabilities: tuple[str, ...],
        request: object,
        status: str = "pending",
    ) -> StoredJob:
        """Atomically admit one Job carrying one durable ToolTask request."""

        db = self._db()
        request_json = _json(request)
        capabilities_json = _json(list(required_capabilities))
        try:
            await db.execute(
                "INSERT INTO jobs(job_id,task_id,logical_key,status,binding_id,plan_id,"
                "resource_key,required_capabilities_json,request_json) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    job_id,
                    task_id,
                    logical_key,
                    status,
                    binding_id,
                    plan_id,
                    resource_key,
                    capabilities_json,
                    request_json,
                ),
            )
            await db.commit()
        except aiosqlite.IntegrityError as exc:
            await db.rollback()
            existing = await self.get_job_by_logical_key(logical_key)
            if existing is None or (
                existing.binding_id != binding_id
                or existing.plan_id != plan_id
                or existing.resource_key != resource_key
                or existing.required_capabilities != tuple(required_capabilities)
                or _json(existing.request) != request_json
            ):
                raise HistoryConflictError(
                    "Job identity is already committed with different admission data"
                ) from exc
            return existing
        stored = await self.get_job(job_id)
        assert stored is not None
        return stored

    @_serialized_write
    async def update_tool_job(
        self,
        job_id: str,
        *,
        status: str,
        result: object | None = None,
        error: object | None = None,
        attempt: int | None = None,
    ) -> None:
        cursor = await self._db().execute(
            "UPDATE jobs SET status=?,result_json=?,error_json=?,"
            "attempt=COALESCE(?,attempt),updated_at=CURRENT_TIMESTAMP "
            "WHERE job_id=?",
            (
                status,
                None if result is None else _json(result),
                None if error is None else _json(error),
                attempt,
                job_id,
            ),
        )
        if cursor.rowcount != 1:
            await self._db().rollback()
            raise KeyError(f"unknown Job {job_id!r}")
        await self._db().commit()

    async def get_job(self, job_id: str) -> StoredJob | None:
        return await self._select_job("job_id", job_id)

    async def get_job_by_task(self, task_id: str) -> StoredJob | None:
        return await self._select_job("task_id", task_id)

    async def get_job_by_logical_key(self, logical_key: str) -> StoredJob | None:
        return await self._select_job("logical_key", logical_key)

    async def _select_job(self, column: str, value: str) -> StoredJob | None:
        cursor = await self._db().execute(
            "SELECT job_id,task_id,logical_key,status,binding_id,plan_id,resource_key,"
            "required_capabilities_json,request_json,result_json,error_json,attempt "
            f"FROM jobs WHERE {column}=?",
            (value,),
        )
        row = await cursor.fetchone()
        return None if row is None else self._stored_job(row)

    async def list_jobs(
        self,
        *,
        statuses: tuple[str, ...] = (),
        binding_id: str | None = None,
    ) -> tuple[StoredJob, ...]:
        clauses: list[str] = []
        parameters: list[object] = []
        if statuses:
            clauses.append("status IN (" + ",".join("?" for _ in statuses) + ")")
            parameters.extend(statuses)
        if binding_id is not None:
            clauses.append("binding_id=?")
            parameters.append(binding_id)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        cursor = await self._db().execute(
            "SELECT job_id,task_id,logical_key,status,binding_id,plan_id,resource_key,"
            "required_capabilities_json,request_json,result_json,error_json,attempt "
            f"FROM jobs{where} ORDER BY job_id",
            tuple(parameters),
        )
        return tuple(self._stored_job(row) for row in await cursor.fetchall())

    @staticmethod
    def _stored_job(row: Any) -> StoredJob:
        capabilities = _load(row[7])
        request = _load(row[8])
        if not isinstance(capabilities, tuple) or any(
            not isinstance(value, str) for value in capabilities
        ):
            raise HistoryStoreError("stored Job capabilities are invalid")
        if request is None:
            raise HistoryStoreError("stored Job request is missing")
        return StoredJob(
            job_id=row[0],
            task_id=row[1],
            logical_key=row[2],
            status=row[3],
            binding_id=row[4],
            plan_id=row[5],
            resource_key=row[6],
            required_capabilities=tuple(cast(str, value) for value in capabilities),
            request=request,
            result=_load(row[9]),
            error=_load(row[10]),
            attempt=row[11],
        )

    async def get_task(self, task_id: str) -> StoredTask | None:
        db = self._db()
        cursor = await db.execute(
            "SELECT task_id,kind,status,request_json,result_json,error_json "
            "FROM tasks WHERE task_id=?",
            (task_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return StoredTask(
            task_id=row[0],
            kind=row[1],
            status=row[2],
            request=_load(row[3]),
            result=_load(row[4]),
            error=_load(row[5]),
        )

    async def list_tasks(
        self, *, statuses: tuple[str, ...] = (), kind: str | None = None
    ) -> tuple[StoredTask, ...]:
        clauses: list[str] = []
        parameters: list[object] = []
        if statuses:
            clauses.append("status IN (" + ",".join("?" for _ in statuses) + ")")
            parameters.extend(statuses)
        if kind is not None:
            clauses.append("kind=?")
            parameters.append(kind)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        cursor = await self._db().execute(
            "SELECT task_id,kind,status,request_json,result_json,error_json "
            f"FROM tasks{where} ORDER BY task_id",
            tuple(parameters),
        )
        rows = await cursor.fetchall()
        return tuple(
            StoredTask(
                task_id=row[0],
                kind=row[1],
                status=row[2],
                request=_load(row[3]),
                result=_load(row[4]),
                error=_load(row[5]),
            )
            for row in rows
        )

    @_serialized_write
    async def record_effect(
        self,
        *,
        execution_id: str,
        module_path: str,
        call_index: int,
        effect_type: str,
        request: object,
        result: object,
        spec: object | None = None,
    ) -> StoredEffect:
        db = self._db()
        digest = effect_digest(request)
        result_json = _json(result)
        spec_json = None if spec is None else _json(spec)
        try:
            await db.execute(
                "INSERT INTO effects(execution_id,module_path,call_index,effect_type,"
                "request_digest,spec_json,status,result_json) VALUES(?,?,?,?,?,?,?,?)",
                (
                    execution_id,
                    module_path,
                    call_index,
                    effect_type,
                    digest,
                    spec_json,
                    "completed",
                    result_json,
                ),
            )
            await db.commit()
        except aiosqlite.IntegrityError as exc:
            await db.rollback()
            existing = await self.replay_effect(
                execution_id=execution_id,
                module_path=module_path,
                call_index=call_index,
                effect_type=effect_type,
                request=request,
                spec=spec,
            )
            if existing.status != "completed" or _json(existing.result) != result_json:
                raise HistoryConflictError(
                    "effect identity is already committed with another result"
                ) from exc
            return existing
        return StoredEffect(
            execution_id=execution_id,
            module_path=module_path,
            call_index=call_index,
            effect_type=effect_type,
            request_digest=digest,
            status="completed",
            spec=None if spec is None else freeze_json(spec),
            result=freeze_json(result),
        )

    @_serialized_write
    async def begin_effect(
        self,
        *,
        execution_id: str,
        module_path: str,
        call_index: int,
        effect_type: str,
        request: object,
        spec: object,
    ) -> tuple[StoredEffect, bool]:
        """Persist the started boundary before an operation can escape."""

        created = False
        try:
            await self._db().execute(
                "INSERT INTO effects(execution_id,module_path,call_index,effect_type,"
                "request_digest,spec_json,status,result_json) VALUES(?,?,?,?,?,?,?,NULL)",
                (
                    execution_id,
                    module_path,
                    call_index,
                    effect_type,
                    effect_digest(request),
                    _json(spec),
                    "started",
                ),
            )
            await self._db().commit()
            created = True
        except aiosqlite.IntegrityError:
            await self._db().rollback()
        return (
            await self.replay_effect(
                execution_id=execution_id,
                module_path=module_path,
                call_index=call_index,
                effect_type=effect_type,
                request=request,
                spec=spec,
            ),
            created,
        )

    @_serialized_write
    async def complete_effect(
        self,
        *,
        execution_id: str,
        module_path: str,
        call_index: int,
        result: object,
    ) -> None:
        cursor = await self._db().execute(
            "UPDATE effects SET status='completed',result_json=? WHERE execution_id=? "
            "AND module_path=? AND call_index=? AND status='started'",
            (_json(result), execution_id, module_path, call_index),
        )
        if cursor.rowcount != 1:
            await self._db().rollback()
            raise HistoryConflictError("effect is not in a completable started state")
        await self._db().commit()

    @_serialized_write
    async def mark_effect_unknown(
        self, *, execution_id: str, module_path: str, call_index: int
    ) -> None:
        cursor = await self._db().execute(
            "UPDATE effects SET status='unknown' WHERE execution_id=? AND module_path=? "
            "AND call_index=? AND status='started'",
            (execution_id, module_path, call_index),
        )
        if cursor.rowcount not in (0, 1):  # pragma: no cover - SQLite invariant
            raise HistoryStoreError("invalid effect update cardinality")
        await self._db().commit()

    async def replay_effect(
        self,
        *,
        execution_id: str,
        module_path: str,
        call_index: int,
        effect_type: str,
        request: object,
        spec: object | None = None,
    ) -> StoredEffect:
        db = self._db()
        cursor = await db.execute(
            "SELECT effect_type,request_digest,spec_json,status,result_json FROM effects WHERE "
            "execution_id=? AND module_path=? AND call_index=?",
            (execution_id, module_path, call_index),
        )
        row = await cursor.fetchone()
        if row is None:
            raise KeyError("effect history is not committed")
        if row[0] != effect_type:
            raise NonDeterministicReplayError(
                "effect type differs from committed deterministic history"
            )
        digest = effect_digest(request)
        if row[1] != digest:
            raise NonDeterministicReplayError(
                "effect request differs from committed deterministic history"
            )
        stored_spec = _load(row[2])
        if spec is not None and stored_spec is not None and _json(stored_spec) != _json(spec):
            raise NonDeterministicReplayError(
                "effect recovery policy differs from committed history"
            )
        status = cast(Literal["started", "completed", "unknown"], row[3])
        result = _load(row[4])
        return StoredEffect(
            execution_id=execution_id,
            module_path=module_path,
            call_index=call_index,
            effect_type=effect_type,
            request_digest=digest,
            status=status,
            spec=stored_spec,
            result=result,
        )

    @_serialized_write
    async def save_checkpoint(
        self,
        *,
        execution_id: str,
        checkpoint_index: int,
        graph_hash: str,
        state: object,
    ) -> None:
        await self._db().execute(
            "INSERT INTO checkpoints VALUES(?,?,?,?) ON CONFLICT DO UPDATE SET "
            "graph_hash=excluded.graph_hash,state_json=excluded.state_json",
            (execution_id, checkpoint_index, graph_hash, _json(state)),
        )
        await self._db().commit()

    async def load_checkpoint(
        self, *, execution_id: str, graph_hash: str
    ) -> tuple[int, JsonValue] | None:
        cursor = await self._db().execute(
            "SELECT checkpoint_index,graph_hash,state_json FROM checkpoints "
            "WHERE execution_id=? ORDER BY checkpoint_index DESC LIMIT 1",
            (execution_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        if row[1] != graph_hash:
            raise NonDeterministicReplayError(
                "checkpoint graph hash is incompatible with this ExecutionPlan"
            )
        state = _load(row[2])
        assert state is not None
        return row[0], state

    @_serialized_write
    async def append_event(
        self, *, execution_id: str, index: int, event: Mapping[str, Any]
    ) -> None:
        try:
            await self._db().execute(
                "INSERT INTO events VALUES(?,?,?)", (execution_id, index, _json(event))
            )
            await self._db().commit()
        except aiosqlite.IntegrityError as exc:
            await self._db().rollback()
            cursor = await self._db().execute(
                "SELECT event_json FROM events WHERE execution_id=? AND event_index=?",
                (execution_id, index),
            )
            row = await cursor.fetchone()
            if row is None or row[0] != _json(event):
                raise HistoryConflictError("event cursor has conflicting content") from exc

    async def events_after(
        self, *, execution_id: str, after: int = -1, limit: int = 256
    ) -> tuple[JsonValue, ...]:
        if limit <= 0 or limit > 4096:
            raise ValueError("event limit must be between 1 and 4096")
        cursor = await self._db().execute(
            "SELECT event_json FROM events WHERE execution_id=? AND event_index>? "
            "ORDER BY event_index LIMIT ?",
            (execution_id, after, limit),
        )
        rows = await cursor.fetchall()
        return tuple(value for row in rows if (value := _load(row[0])) is not None)

    async def events_tail(
        self, *, execution_id: str, limit: int = 256
    ) -> tuple[JsonValue, ...]:
        """Return the newest durable events in ascending cursor order."""

        if limit <= 0 or limit > 4096:
            raise ValueError("event limit must be between 1 and 4096")
        cursor = await self._db().execute(
            "SELECT event_json FROM events WHERE execution_id=? "
            "ORDER BY event_index DESC LIMIT ?",
            (execution_id, limit),
        )
        rows = list(await cursor.fetchall())
        rows.reverse()
        return tuple(value for row in rows if (value := _load(row[0])) is not None)


__all__ = [
    "HistoryConflictError",
    "HistoryStoreError",
    "NonDeterministicReplayError",
    "SQLiteHistoryStore",
    "StoredEffect",
    "StoredExecution",
    "StoredJob",
    "StoredTask",
    "effect_digest",
]
