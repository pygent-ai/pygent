"""SQLite durable Tool task and Job persistence."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Literal, cast

import aiosqlite

from ._history_types import (
    HistoryConflictError,
    HistoryStoreError,
    StoredJob,
    StoredTask,
    _json,
    _load,
    _serialized_write,
)


class JobHistoryMixin:
    if TYPE_CHECKING:
        _write_lock: asyncio.Lock

        def _db(self) -> aiosqlite.Connection: ...

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
