"""SQLite execution identity, claims, and model admission persistence."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import aiosqlite

from ._history_types import (
    HistoryConflictError,
    StoredExecution,
    _json,
    _load,
    _serialized_write,
)


class ExecutionHistoryMixin:
    if TYPE_CHECKING:
        _write_lock: asyncio.Lock

        def _db(self) -> aiosqlite.Connection: ...

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
