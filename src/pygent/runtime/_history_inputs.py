"""SQLite persistence for ordered, replayable execution inputs."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, TypeVar, cast

import aiosqlite

from pygent.core import ExecutionInput, ExecutionInputDelivery, JsonValue, thaw_json

from ._execution_inputs import (
    MAX_PENDING_EXECUTION_INPUTS,
    ExecutionInputConsumerError,
    prepare_execution_input,
    validate_receive,
)
from ._history_types import _json, _load

_T = TypeVar("_T")


class ExecutionInputHistoryMixin:
    if TYPE_CHECKING:
        def _db(self) -> aiosqlite.Connection: ...
        async def _queue_transaction(
            self,
            operation: Callable[[aiosqlite.Connection], Awaitable[_T]],
            *,
            batch_key: str | None = None,
            batch_payload: object | None = None,
            batch_operation: (
                Callable[[aiosqlite.Connection, list[object]], Awaitable[list[object]]]
                | None
            ) = None,
        ) -> _T: ...

    async def send_execution_input(
        self,
        execution_id: str,
        *,
        input_id: str,
        kind: str,
        value: JsonValue,
    ) -> ExecutionInputDelivery:
        frozen = prepare_execution_input(input_id, kind, value)

        async def operation(db: aiosqlite.Connection) -> ExecutionInputDelivery:
            duplicate = await (
                await db.execute(
                    "SELECT sequence FROM execution_inputs WHERE execution_id=? AND input_id=?",
                    (execution_id, input_id),
                )
            ).fetchone()
            if duplicate is not None:
                return ExecutionInputDelivery(
                    "duplicate", execution_id, input_id, int(duplicate[0])
                )
            execution = await (
                await db.execute(
                    "SELECT terminal_sequence FROM executions WHERE execution_id=?",
                    (execution_id,),
                )
            ).fetchone()
            if execution is None:
                raise KeyError(f"unknown execution {execution_id!r}")
            await db.execute(
                "INSERT OR IGNORE INTO execution_inboxes(execution_id,next_sequence,sealed) VALUES(?,0,0)",
                (execution_id,),
            )
            inbox = await (
                await db.execute(
                    "SELECT next_sequence,sealed FROM execution_inboxes WHERE execution_id=?",
                    (execution_id,),
                )
            ).fetchone()
            assert inbox is not None
            if execution[0] is not None or bool(inbox[1]):
                return ExecutionInputDelivery("execution_finished", execution_id, input_id)
            pending = await (
                await db.execute(
                    "SELECT COUNT(*) FROM execution_inputs i "
                    "LEFT JOIN execution_input_consumers c "
                    "ON c.execution_id=i.execution_id AND c.kind=i.kind "
                    "WHERE i.execution_id=? AND i.sequence>COALESCE(c.last_sequence,-1)",
                    (execution_id,),
                )
            ).fetchone()
            assert pending is not None
            if int(pending[0]) >= MAX_PENDING_EXECUTION_INPUTS:
                raise OverflowError("execution input inbox is full")
            sequence = int(inbox[0])
            await db.execute(
                "INSERT INTO execution_inputs(execution_id,input_id,sequence,kind,value_json) VALUES(?,?,?,?,?)",
                (execution_id, input_id, sequence, kind, _json(thaw_json(frozen))),
            )
            await db.execute(
                "UPDATE execution_inboxes SET next_sequence=next_sequence+1 WHERE execution_id=?",
                (execution_id,),
            )
            return ExecutionInputDelivery("accepted", execution_id, input_id, sequence)

        return cast(ExecutionInputDelivery, await self._queue_transaction(operation))

    async def receive_execution_inputs(
        self,
        *,
        execution_id: str,
        module_path: str,
        receive_index: int,
        kinds: tuple[str, ...],
        limit: int,
        seal_if_empty: bool,
    ) -> tuple[ExecutionInput, ...]:
        validate_receive(kinds, limit, seal_if_empty)
        request_json = json.dumps(
            {"kinds": list(kinds), "limit": limit, "seal_if_empty": seal_if_empty},
            sort_keys=True,
            separators=(",", ":"),
        )

        async def operation(db: aiosqlite.Connection) -> tuple[ExecutionInput, ...]:
            receipt = await (
                await db.execute(
                    "SELECT request_json,batch_json FROM execution_input_receives "
                    "WHERE execution_id=? AND module_path=? AND receive_index=?",
                    (execution_id, module_path, receive_index),
                )
            ).fetchone()
            if receipt is not None:
                if receipt[0] != request_json:
                    raise RuntimeError("replayed execution input receive changed its request")
                values = json.loads(receipt[1])
                return tuple(ExecutionInput.from_dict(item) for item in values)
            await db.execute(
                "INSERT OR IGNORE INTO execution_inboxes(execution_id,next_sequence,sealed) VALUES(?,0,0)",
                (execution_id,),
            )
            for kind in kinds:
                owner = await (
                    await db.execute(
                        "SELECT module_path FROM execution_input_consumers WHERE execution_id=? AND kind=?",
                        (execution_id, kind),
                    )
                ).fetchone()
                if owner is not None and owner[0] != module_path:
                    raise ExecutionInputConsumerError(
                        f"execution input kind {kind!r} is owned by {owner[0]!r}"
                    )
                await db.execute(
                    "INSERT OR IGNORE INTO execution_input_consumers(execution_id,kind,module_path,last_sequence) VALUES(?,?,?,-1)",
                    (execution_id, kind, module_path),
                )
            placeholders = ",".join("?" for _ in kinds)
            rows = await (
                await db.execute(
                    "SELECT i.input_id,i.sequence,i.kind,i.value_json "
                    "FROM execution_inputs i JOIN execution_input_consumers c "
                    "ON c.execution_id=i.execution_id AND c.kind=i.kind "
                    f"WHERE i.execution_id=? AND i.kind IN ({placeholders}) "
                    "AND i.sequence>c.last_sequence ORDER BY i.sequence LIMIT ?",
                    (execution_id, *kinds, limit),
                )
            ).fetchall()
            selected = tuple(
                ExecutionInput(row[0], int(row[1]), row[2], cast(JsonValue, _load(row[3])))
                for row in rows
            )
            for item in selected:
                await db.execute(
                    "UPDATE execution_input_consumers SET last_sequence=MAX(last_sequence,?) "
                    "WHERE execution_id=? AND kind=?",
                    (item.sequence, execution_id, item.kind),
                )
            if not selected and seal_if_empty:
                await db.execute(
                    "UPDATE execution_inboxes SET sealed=1 WHERE execution_id=?",
                    (execution_id,),
                )
            batch_json = json.dumps(
                [item.to_dict() for item in selected],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            await db.execute(
                "INSERT INTO execution_input_receives VALUES(?,?,?,?,?)",
                (execution_id, module_path, receive_index, request_json, batch_json),
            )
            return selected

        return cast(tuple[ExecutionInput, ...], await self._queue_transaction(operation))
