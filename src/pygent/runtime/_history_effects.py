"""SQLite managed effects, checkpoints, and event journal."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, cast

import aiosqlite

from pygent.core import JsonValue, freeze_json

from ._history_types import (
    HistoryConflictError,
    HistoryStoreError,
    NonDeterministicReplayError,
    StoredEffect,
    _json,
    _load,
    _serialized_write,
    effect_digest,
)


class EffectHistoryMixin:
    if TYPE_CHECKING:
        _write_lock: asyncio.Lock

        def _db(self) -> aiosqlite.Connection: ...
        async def _queue_event(
            self, execution_id: str, index: int, payload: str
        ) -> None: ...

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

    async def append_event(
        self, *, execution_id: str, index: int, event: Mapping[str, Any]
    ) -> None:
        await self._queue_event(execution_id, index, _json(event))

    async def last_event_index(self, *, execution_id: str) -> int:
        cursor = await self._db().execute(
            "SELECT COALESCE(MAX(event_index), -1) FROM events WHERE execution_id=?",
            (execution_id,),
        )
        row = await cursor.fetchone()
        return -1 if row is None else int(row[0])

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
