"""Durable step-interrupt decisions.

A step interrupt is a Runtime Owner decision: an abortable Module Child step
was cancelled because a committed immediate steering input waited at the next
drain point. The decision is journaled per stable step identity so replay
reproduces ``interrupted`` without live signals, and the journaled pending
input snapshot ties the decision to its triggering input. Observing pending
inputs for wake-up purposes lives with the input domain
(``ExecutionInputHistoryMixin.peek_execution_inputs``) and never touches
consumer cursors or receipts either -- observation is not consumption.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, cast

import aiosqlite

from ._history_types import HistoryStoreError, StoredStepInterrupt, _json, _load


def _is_int_sequence(value: object) -> bool:
    """A JSON array of plain integers; booleans are not integers here."""

    return isinstance(value, (list, tuple)) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    )


def _decode_step_interrupt(
    module_path: str,
    step_index: int,
    detail_json: str,
) -> StoredStepInterrupt:
    detail = _load(detail_json)
    cancelled_value = detail.get("cancelled") if isinstance(detail, Mapping) else None
    pending_value = (
        detail.get("pending_input_sequences") if isinstance(detail, Mapping) else None
    )
    if not _is_int_sequence(cancelled_value):
        raise HistoryStoreError("journaled step interrupt cancelled set is not readable")
    if not _is_int_sequence(pending_value):
        raise HistoryStoreError(
            "journaled step interrupt input snapshot is not readable"
        )
    return StoredStepInterrupt(
        module_path=module_path,
        step_index=step_index,
        cancelled=tuple(item for item in cast("tuple[int, ...]", cancelled_value)),
        pending_input_sequences=tuple(
            item for item in cast("tuple[int, ...]", pending_value)
        ),
    )


class ExecutionStepHistoryMixin:
    if TYPE_CHECKING:

        def _db(self) -> aiosqlite.Connection: ...
        async def _queue_transaction(
            self,
            operation: Callable[[aiosqlite.Connection], Awaitable[object]],
            *,
            execution_id: str | None = None,
        ) -> object: ...

    async def mark_step_interrupted(
        self,
        *,
        execution_id: str,
        module_path: str,
        step_index: int,
        cancelled: tuple[int, ...],
    ) -> StoredStepInterrupt:
        """Journal one step-interrupt decision idempotently.

        The pending-input snapshot is captured inside the same transaction as
        the insert, so the decision and its triggering input association commit
        atomically. An already-journaled decision is returned unchanged: the
        first determination wins and replay never rewrites history.
        """

        async def operation(db: aiosqlite.Connection) -> StoredStepInterrupt:
            pending = await db.execute_fetchall(
                "SELECT i.sequence FROM execution_inputs i "
                "LEFT JOIN execution_input_consumers c "
                "ON c.execution_id=i.execution_id AND c.kind=i.kind "
                "WHERE i.execution_id=? AND i.sequence>COALESCE(c.last_sequence,-1) "
                "ORDER BY i.sequence",
                (execution_id,),
            )
            detail = {
                "cancelled": list(cancelled),
                "pending_input_sequences": [int(row[0]) for row in pending],
            }
            cursor = await db.execute(
                "INSERT INTO execution_step_interrupts"
                "(execution_id,module_path,step_index,detail_json) VALUES(?,?,?,?) "
                "ON CONFLICT(execution_id,module_path,step_index) DO NOTHING",
                (execution_id, module_path, step_index, _json(detail)),
            )
            if cursor.rowcount == 1:
                return StoredStepInterrupt(
                    module_path=module_path,
                    step_index=step_index,
                    cancelled=tuple(cancelled),
                    pending_input_sequences=tuple(
                        int(row[0]) for row in pending
                    ),
                )
            row = await (
                await db.execute(
                    "SELECT detail_json FROM execution_step_interrupts "
                    "WHERE execution_id=? AND module_path=? AND step_index=?",
                    (execution_id, module_path, step_index),
                )
            ).fetchone()
            assert row is not None
            return _decode_step_interrupt(module_path, step_index, row[0])

        return cast(
            StoredStepInterrupt,
            await self._queue_transaction(
                operation, execution_id=execution_id
            ),
        )

    async def get_step_interrupt(
        self,
        *,
        execution_id: str,
        module_path: str,
        step_index: int,
    ) -> StoredStepInterrupt | None:
        row = await (
            await self._db().execute(
                "SELECT detail_json FROM execution_step_interrupts "
                "WHERE execution_id=? AND module_path=? AND step_index=?",
                (execution_id, module_path, step_index),
            )
        ).fetchone()
        if row is None:
            return None
        return _decode_step_interrupt(module_path, step_index, row[0])

    async def has_step_interrupts(self, execution_id: str) -> bool:
        row = await (
            await self._db().execute(
                "SELECT 1 FROM execution_step_interrupts WHERE execution_id=? LIMIT 1",
                (execution_id,),
            )
        ).fetchone()
        return row is not None
