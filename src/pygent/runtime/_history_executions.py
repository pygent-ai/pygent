"""SQLite execution identity, claims, and model admission persistence."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import aiosqlite

from ._history_types import (
    HistoryConflictError,
    StoredExecution,
    _json,
    _load,
    _prepare_json,
    _serialized_write,
)


@dataclass(frozen=True, slots=True)
class _BeginExecutionBatchItem:
    values: tuple[object, ...]
    stored: StoredExecution


@dataclass(frozen=True, slots=True)
class _UpdateExecutionBatchItem:
    values: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _FinalizeExecutionBatchItem:
    event_rows: tuple[tuple[str, int, str], ...]
    update_values: tuple[object, ...]
    execution_id: str


class ExecutionHistoryMixin:
    if TYPE_CHECKING:
        _write_lock: asyncio.Lock

        def _db(self) -> aiosqlite.Connection: ...
        async def _queue_transaction(
            self,
            operation: Callable[[aiosqlite.Connection], Awaitable[Any]],
            *,
            batch_key: str | None = None,
            batch_payload: object | None = None,
            batch_operation: (
                Callable[
                    [aiosqlite.Connection, list[object]], Awaitable[list[object]]
                ]
                | None
            ) = None,
        ) -> Any: ...

    async def claim_execution(
        self, *, execution_id: str, owner_id: str, lease_ttl: float
    ) -> int | None:
        """Atomically claim one durable recovery attempt across processes."""

        async def operation(db: aiosqlite.Connection) -> int | None:
            await db.execute(
                "DELETE FROM execution_claims "
                "WHERE execution_id=? AND expires_at<=unixepoch('subsec')",
                (execution_id,),
            )
            row = await (
                await db.execute(
                    "SELECT owner_id FROM execution_claims WHERE execution_id=?",
                    (execution_id,),
                )
            ).fetchone()
            if row is not None:
                return None
            cursor = await db.execute("INSERT INTO execution_fences DEFAULT VALUES")
            token = cursor.lastrowid
            assert token is not None
            await db.execute(
                "INSERT INTO execution_claims VALUES(?,?,?,unixepoch('subsec')+?)",
                (execution_id, owner_id, token, lease_ttl),
            )
            return int(token)

        return cast(int | None, await self._queue_transaction(operation))

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
        status: str = "pending",
        binding_id: str = "",
        identity: str = "",
        idempotency_key: str | None = None,
        model_calls: object | None = None,
        model_admission_status: str = "none",
        trace_id: str | None = None,
        attempt_id: str | None = None,
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
            trace_id=trace_id or str(uuid.uuid4()),
            attempt_id=attempt_id,
        )
        return stored

    async def begin_execution(
        self,
        *,
        execution_id: str,
        request_id: str,
        plan_id: str,
        input: object,
        status: str = "pending",
        binding_id: str = "",
        identity: str = "",
        idempotency_key: str | None = None,
        model_calls: object | None = None,
        model_admission_status: str = "none",
        trace_id: str,
        phase: str = "submitting",
        attempt_id: str | None = None,
    ) -> tuple[StoredExecution, bool]:
        prepared_input = _prepare_json(input)
        prepared_model_calls = _prepare_json({} if model_calls is None else model_calls)
        payload = prepared_input.payload
        model_calls_payload = prepared_model_calls.payload
        submitted_at_unix_ns = time.time_ns()
        frozen_input = prepared_input.frozen
        frozen_model_calls = prepared_model_calls.frozen
        values = (
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
            trace_id,
            phase,
            attempt_id,
            submitted_at_unix_ns,
            submitted_at_unix_ns,
        )
        created_record = StoredExecution(
            execution_id=execution_id,
            request_id=request_id,
            status=status,
            plan_id=plan_id,
            input=frozen_input,
            output=None,
            error=None,
            attempt=1,
            binding_id=binding_id,
            identity=identity,
            idempotency_key=idempotency_key,
            model_calls=frozen_model_calls,
            model_admission_status=model_admission_status,
            trace_id=trace_id,
            phase=phase,
            attempt_id=attempt_id,
            submitted_at_unix_ns=submitted_at_unix_ns,
            updated_at_unix_ns=submitted_at_unix_ns,
        )
        batch_item = _BeginExecutionBatchItem(values, created_record)

        async def operation(
            connection: aiosqlite.Connection,
        ) -> tuple[StoredExecution | None, bool]:
            try:
                await connection.execute(
                    "INSERT INTO executions(execution_id,request_id,status,plan_id,input_json,"
                    "binding_id,identity,idempotency_key,model_calls_json,"
                    "model_admission_status,trace_id,phase,attempt_id,"
                    "submitted_at_unix_ns,updated_at_unix_ns) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    values,
                )
            except aiosqlite.IntegrityError as exc:
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
                    existing.plan_id != plan_id or _json(existing.input) != payload
                ):
                    raise HistoryConflictError(
                        "idempotency identity is committed with different input"
                    ) from exc
                return existing, False
            return created_record, True

        existing, created = cast(
            tuple[StoredExecution | None, bool],
            await self._queue_transaction(
                operation,
                batch_key="begin_execution",
                batch_payload=batch_item,
                batch_operation=self._batch_begin_executions,
            ),
        )
        if not created:
            assert existing is not None
            return existing, False
        assert existing is not None
        return existing, True

    @_serialized_write
    async def commit_model_admission(
        self, execution_id: str, *, admission_id: str, manifest_digest: str
    ) -> None:
        db = self._db()
        try:
            cursor = await db.execute(
                "UPDATE executions SET model_admission_id=?,model_admission_digest=?,"
                "model_admission_status='committed',updated_at=CURRENT_TIMESTAMP "
                "WHERE execution_id=? AND model_admission_status IN ('preparing','committed')",
                (admission_id, manifest_digest, execution_id),
            )
            if cursor.rowcount != 1:
                raise HistoryConflictError("model admission intent is not preparing")
            await db.execute(
                "INSERT OR IGNORE INTO execution_model_admissions(execution_id,admission_id) "
                "VALUES(?,?)",
                (execution_id, admission_id),
            )
            await db.commit()
        except BaseException:
            await db.rollback()
            raise

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

    async def update_execution(
        self,
        execution_id: str,
        *,
        status: str,
        output: object | None = None,
        error: object | None = None,
        attempt: int | None = None,
        phase: str | None = None,
        attempt_id: str | None = None,
    ) -> None:
        values = (
            status,
            None if output is None else _json(output),
            None if error is None else _json(error),
            attempt,
            phase,
            attempt_id,
            time.time_ns(),
            execution_id,
        )
        batch_item = _UpdateExecutionBatchItem(values)

        async def operation(db: aiosqlite.Connection) -> None:
            cursor = await db.execute(
                "UPDATE executions SET status=?, output_json=?, error_json=?, "
                "attempt=COALESCE(?,attempt), phase=COALESCE(?,phase), "
                "attempt_id=COALESCE(?,attempt_id), "
                "updated_at_unix_ns=?, updated_at=CURRENT_TIMESTAMP "
                "WHERE execution_id=?",
                values,
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown execution {execution_id!r}")

        await self._queue_transaction(
            operation,
            batch_key="update_execution",
            batch_payload=batch_item,
            batch_operation=self._batch_update_executions,
        )

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
            "model_admission_id,model_admission_digest,model_admission_status,trace_id,phase,"
            "attempt_id,terminal_sequence,submitted_at_unix_ns,updated_at_unix_ns "
            "FROM executions "
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
            f"model_admission_id,model_admission_digest,model_admission_status,trace_id,phase,"
            f"attempt_id,terminal_sequence,submitted_at_unix_ns,updated_at_unix_ns "
            f"FROM executions "
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
            trace_id=row[15],
            phase=row[16],
            attempt_id=row[17],
            terminal_sequence=row[18],
            submitted_at_unix_ns=row[19],
            updated_at_unix_ns=row[20],
        )

    async def finalize_execution(
        self,
        execution_id: str,
        *,
        status: str,
        output: object | None,
        error: object | None,
        terminal_events: tuple[tuple[int, object], ...],
        terminal_sequence: int,
        _terminal_event_payloads: tuple[tuple[int, str], ...] | None = None,
    ) -> None:
        """Commit terminal journal entries and the materialized outcome atomically."""

        payloads = (
            tuple((index, _json(event)) for index, event in terminal_events)
            if _terminal_event_payloads is None
            else _terminal_event_payloads
        )
        output_payload = None if output is None else _json(output)
        error_payload = None if error is None else _json(error)
        updated_at_unix_ns = time.time_ns()
        event_rows = tuple(
            (execution_id, index, payload) for index, payload in payloads
        )
        update_values = (
            status,
            output_payload,
            error_payload,
            terminal_sequence,
            updated_at_unix_ns,
            execution_id,
        )
        batch_item = _FinalizeExecutionBatchItem(
            event_rows, update_values, execution_id
        )

        async def operation(db: aiosqlite.Connection) -> None:
            for event_execution_id, index, payload in event_rows:
                await db.execute(
                    "INSERT INTO events(execution_id,event_index,event_json) VALUES(?,?,?)",
                    (event_execution_id, index, payload),
                )
            cursor = await db.execute(
                "UPDATE executions SET status=?,phase='terminal',output_json=?,error_json=?,"
                "terminal_sequence=?,updated_at_unix_ns=?,updated_at=CURRENT_TIMESTAMP "
                "WHERE execution_id=? "
                "AND terminal_sequence IS NULL",
                update_values,
            )
            if cursor.rowcount != 1:
                raise HistoryConflictError("execution is already finalized or unknown")
            await db.execute("DELETE FROM execution_claims WHERE execution_id=?", (execution_id,))

        await self._queue_transaction(
            operation,
            batch_key="finalize_execution",
            batch_payload=batch_item,
            batch_operation=self._batch_finalize_executions,
        )

    async def _batch_begin_executions(
        self, db: aiosqlite.Connection, payloads: list[object]
    ) -> list[object]:
        items = [cast(_BeginExecutionBatchItem, item) for item in payloads]
        identities = {item.stored.execution_id for item in items}
        if len(identities) != len(items):
            raise HistoryConflictError("execution batch contains duplicate identities")
        await db.executemany(
            "INSERT INTO executions(execution_id,request_id,status,plan_id,input_json,"
            "binding_id,identity,idempotency_key,model_calls_json,"
            "model_admission_status,trace_id,phase,attempt_id,"
            "submitted_at_unix_ns,updated_at_unix_ns) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [item.values for item in items],
        )
        return [(item.stored, True) for item in items]

    async def _batch_update_executions(
        self, db: aiosqlite.Connection, payloads: list[object]
    ) -> list[object]:
        items = [cast(_UpdateExecutionBatchItem, item) for item in payloads]
        cursor = await db.executemany(
            "UPDATE executions SET status=?, output_json=?, error_json=?, "
            "attempt=COALESCE(?,attempt), phase=COALESCE(?,phase), "
            "attempt_id=COALESCE(?,attempt_id), "
            "updated_at_unix_ns=?, updated_at=CURRENT_TIMESTAMP "
            "WHERE execution_id=?",
            [item.values for item in items],
        )
        if cursor.rowcount != len(items):
            raise KeyError("execution batch contains an unknown execution")
        return [None] * len(items)

    async def _batch_finalize_executions(
        self, db: aiosqlite.Connection, payloads: list[object]
    ) -> list[object]:
        items = [cast(_FinalizeExecutionBatchItem, item) for item in payloads]
        event_rows = [row for item in items for row in item.event_rows]
        if event_rows:
            await db.executemany(
                "INSERT INTO events(execution_id,event_index,event_json) VALUES(?,?,?)",
                event_rows,
            )
        cursor = await db.executemany(
            "UPDATE executions SET status=?,phase='terminal',output_json=?,error_json=?,"
            "terminal_sequence=?,updated_at_unix_ns=?,updated_at=CURRENT_TIMESTAMP "
            "WHERE execution_id=? AND terminal_sequence IS NULL",
            [item.update_values for item in items],
        )
        if cursor.rowcount != len(items):
            raise HistoryConflictError(
                "execution batch contains an already finalized or unknown execution"
            )
        await db.executemany(
            "DELETE FROM execution_claims WHERE execution_id=?",
            [(item.execution_id,) for item in items],
        )
        return [None] * len(items)
