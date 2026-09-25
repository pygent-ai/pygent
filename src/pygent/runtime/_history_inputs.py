"""SQLite persistence for ordered, replayable execution inputs."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar, cast

import aiosqlite

from pygent.core import ExecutionInput, ExecutionInputDelivery, JsonValue, thaw_json

from ._execution_inputs import (
    MAX_PENDING_EXECUTION_INPUTS,
    ExecutionInputConsumerError,
    prepare_execution_input,
    validate_receive,
)
from ._history_types import HistoryStoreError, _json, _load

_T = TypeVar("_T")

# Reads that observed no input collapse into a single marker row per
# (execution, module). The marker records the highest receive index already
# observed as empty, so a replay of that index range returns the recorded empty
# batch without keeping one row per drain. Only a committed read extends the
# marker, so a crashed read stays unrecorded and is read again on replay.
#
# Invariant: the stored intervals are strictly ascending by ``through`` and each
# interval records the request shape that observed its index range. Only this
# module writes them (raise: append or extend the last interval), so any other
# order means a corrupted or foreign journal and is rejected by
# ``_decode_empty_ranges`` instead of being repaired.
_EMPTY_RECEIPT_INDEX = -1


def _decode_empty_ranges(
    marker_json: str, *, execution_id: str, module_path: str
) -> list[tuple[str, int]]:
    """Decode one empty-observation marker and enforce its ordering invariant.

    ``_empty_range_verdict`` derives each interval's start from the previous
    interval's ``through``, so the stored intervals must be strictly ascending.
    Sorting a malformed marker would have to invent which request shape observed
    which index, and a wrong answer can silently swallow inputs that were never
    observed; failing the replay loudly keeps "not observed" distinct from
    "observed empty".
    """

    try:
        payload = json.loads(marker_json)
        entries = payload["empty_ranges"]
    except (KeyError, TypeError, ValueError) as exc:
        raise HistoryStoreError(
            "recorded empty execution input receipt is not readable"
        ) from exc
    if not isinstance(entries, list) or not entries:
        raise HistoryStoreError(
            "recorded empty execution input receipt has no intervals"
        )
    ranges: list[tuple[str, int]] = []
    previous = -1
    for entry in entries:
        if not isinstance(entry, dict):
            raise HistoryStoreError(
                "recorded empty execution input receipt interval is not an object"
            )
        request = entry.get("request")
        through = entry.get("through")
        if not isinstance(request, str) or not request:
            raise HistoryStoreError(
                "recorded empty execution input receipt interval has no request"
            )
        if isinstance(through, bool) or not isinstance(through, int):
            raise HistoryStoreError(
                "recorded empty execution input receipt interval has no index"
            )
        if through <= previous:
            raise HistoryStoreError(
                "recorded empty execution input receipt intervals are not ordered"
            )
        ranges.append((request, through))
        previous = through
    return ranges


def _empty_range_verdict(
    ranges: list[tuple[str, int]], receive_index: int, request_json: str
) -> bool | None:
    """Classify one receive against the recorded empty observations.

    ``True`` means the index was already observed as empty under this request
    shape, ``False`` means it was observed under a different shape, and ``None``
    means it has not been observed yet. ``ranges`` must be strictly ascending by
    index, which ``_decode_empty_ranges`` enforces for stored markers.
    """

    previous = -1
    for recorded_request, through in ranges:
        if previous < receive_index <= through:
            return recorded_request == request_json
        previous = through
    return None


@dataclass(frozen=True, slots=True)
class _ReceiveInputs:
    execution_id: str
    module_path: str
    receive_index: int
    kinds: tuple[str, ...]
    limit: int
    seal_if_empty: bool
    request_json: str


class ExecutionInputHistoryMixin:
    if TYPE_CHECKING:

        def _db(self) -> aiosqlite.Connection: ...
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

        request = _ReceiveInputs(
            execution_id,
            module_path,
            receive_index,
            kinds,
            limit,
            seal_if_empty,
            request_json,
        )

        async def operation(db: aiosqlite.Connection) -> object:
            return (await self._batch_receive_inputs(db, [request]))[0]

        return cast(
            tuple[ExecutionInput, ...],
            await self._queue_transaction(
                operation,
                execution_id=execution_id,
                batch_key="receive_execution_inputs",
                batch_payload=request,
                batch_operation=self._batch_receive_inputs,
            ),
        )

    async def _empty_receipt_ranges(
        self,
        db: aiosqlite.Connection,
        requests: list[_ReceiveInputs],
        results: Mapping[str, tuple[ExecutionInput, ...]],
    ) -> dict[tuple[str, str], list[tuple[str, int]]]:
        """Return the recorded empty observations per execution/module.

        Each entry is ``(request_json, through)`` and covers the receive indexes
        after the previous entry's ``through`` up to its own, so a replay can
        tell which request shape observed a given index as empty.
        """

        outstanding = sorted({
            (r.execution_id, r.module_path)
            for r in requests
            if r.execution_id not in results
        })
        if not outstanding:
            return {}
        markers = await db.execute_fetchall(
            "SELECT m.execution_id,m.module_path,m.request_json "
            "FROM json_each(?) r JOIN execution_input_receives m "
            "ON m.execution_id=json_extract(r.value,'$[0]') "
            "AND m.module_path=json_extract(r.value,'$[1]') "
            "AND m.receive_index=?",
            (json.dumps(outstanding), _EMPTY_RECEIPT_INDEX),
        )
        return {
            (execution_id, module_path): _decode_empty_ranges(
                marker_json, execution_id=execution_id, module_path=module_path
            )
            for execution_id, module_path, marker_json in markers
        }

    async def _batch_receive_inputs(
        self, db: aiosqlite.Connection, payloads: list[object]
    ) -> list[object]:
        requests = [cast(_ReceiveInputs, item) for item in payloads]
        by_execution = {item.execution_id: item for item in requests}
        if len(by_execution) != len(requests):
            # Receives on one execution may advance the same cursor. Preserve
            # their original order using the transaction queue's isolated retry.
            raise RuntimeError(
                "overlapping execution input receives require ordered transactions"
            )
        receipts = await db.execute_fetchall(
            "SELECT h.execution_id,h.request_json,h.batch_json "
            "FROM json_each(?) r JOIN execution_input_receives h "
            "ON h.execution_id=json_extract(r.value,'$[0]') "
            "AND h.module_path=json_extract(r.value,'$[1]') "
            "AND h.receive_index=json_extract(r.value,'$[2]')",
            (
                json.dumps(
                    [(r.execution_id, r.module_path, r.receive_index) for r in requests]
                ),
            ),
        )
        results: dict[str, tuple[ExecutionInput, ...]] = {}
        for execution_id, original_request, batch_json in receipts:
            if original_request != by_execution[execution_id].request_json:
                raise RuntimeError(
                    "replayed execution input receive changed its request"
                )
            results[execution_id] = tuple(
                ExecutionInput.from_dict(item) for item in json.loads(batch_json)
            )
        empty_ranges = await self._empty_receipt_ranges(db, requests, results)
        pending: list[_ReceiveInputs] = []
        for request in requests:
            if request.execution_id in results:
                continue
            ranges = empty_ranges.get((request.execution_id, request.module_path), [])
            verdict = _empty_range_verdict(
                ranges, request.receive_index, request.request_json
            )
            if verdict is None:
                pending.append(request)
            elif verdict:
                results[request.execution_id] = ()
            else:
                raise RuntimeError(
                    "replayed execution input receive changed its request"
                )
        if not pending:
            return [results[r.execution_id] for r in requests]
        consumers = [
            (r.execution_id, kind, r.module_path, r.limit)
            for r in pending
            for kind in r.kinds
        ]
        wanted = json.dumps(consumers)
        owners = await db.execute_fetchall(
            "SELECT c.execution_id,c.kind,c.module_path "
            "FROM json_each(?) r JOIN execution_input_consumers c "
            "ON c.execution_id=json_extract(r.value,'$[0]') "
            "AND c.kind=json_extract(r.value,'$[1]')",
            (wanted,),
        )
        for execution_id, kind, owner in owners:
            if owner != by_execution[execution_id].module_path:
                raise ExecutionInputConsumerError(
                    f"execution input kind {kind!r} is owned by {owner!r}"
                )
        await db.executemany(
            "INSERT OR IGNORE INTO execution_inboxes(execution_id,next_sequence,sealed) VALUES(?,0,0)",
            [(r.execution_id,) for r in pending],
        )
        await db.executemany(
            "INSERT OR IGNORE INTO execution_input_consumers"
            "(execution_id,kind,module_path,last_sequence) VALUES(?,?,?,-1)",
            [
                (execution_id, kind, module_path)
                for execution_id, kind, module_path, _ in consumers
            ],
        )
        if len(pending) == 1:
            request = pending[0]
            # A single inbox can stop its index scan at LIMIT instead of
            # ranking every pending input for a cross-execution window.
            rows = await db.execute_fetchall(
                "SELECT i.execution_id,i.input_id,i.sequence,i.kind,i.value_json "
                "FROM execution_inputs i JOIN execution_input_consumers c "
                "ON c.execution_id=i.execution_id AND c.kind=i.kind "
                "WHERE i.execution_id=? AND i.kind IN (SELECT value FROM json_each(?)) "
                "AND i.sequence>c.last_sequence ORDER BY i.sequence LIMIT ?",
                (request.execution_id, json.dumps(request.kinds), request.limit),
            )
        else:
            rows = await db.execute_fetchall(
                "SELECT i.execution_id,i.input_id,i.sequence,i.kind,i.value_json FROM ("
                "SELECT i.execution_id,i.sequence,"
                "json_extract(r.value,'$[3]') AS batch_limit,"
                "ROW_NUMBER() OVER (PARTITION BY i.execution_id ORDER BY i.sequence) AS position "
                "FROM json_each(?) r JOIN execution_inputs i "
                "ON i.execution_id=json_extract(r.value,'$[0]') "
                "AND i.kind=json_extract(r.value,'$[1]') "
                "JOIN execution_input_consumers c ON c.execution_id=i.execution_id AND c.kind=i.kind "
                "WHERE i.sequence>c.last_sequence) selected "
                "JOIN execution_inputs i ON i.execution_id=selected.execution_id "
                "AND i.sequence=selected.sequence WHERE position<=batch_limit "
                "ORDER BY i.execution_id,i.sequence",
                (wanted,),
            )
        selected: dict[str, list[ExecutionInput]] = {
            r.execution_id: [] for r in pending
        }
        cursors: dict[tuple[str, str], int] = {}
        for execution_id, input_id, sequence, kind, value_json in rows:
            selected[execution_id].append(
                ExecutionInput(
                    input_id, int(sequence), kind, cast(JsonValue, _load(value_json))
                )
            )
            cursors[execution_id, kind] = int(sequence)
        if cursors:
            await db.executemany(
                "UPDATE execution_input_consumers SET last_sequence=MAX(last_sequence,?) "
                "WHERE execution_id=? AND kind=?",
                [
                    (sequence, execution_id, kind)
                    for (execution_id, kind), sequence in cursors.items()
                ],
            )
        sealed = [
            (r.execution_id,)
            for r in pending
            if r.seal_if_empty and not selected[r.execution_id]
        ]
        if sealed:
            await db.executemany(
                "UPDATE execution_inboxes SET sealed=1 WHERE execution_id=?", sealed
            )
        if any(selected[r.execution_id] or r.seal_if_empty for r in pending):
            await db.executemany(
                "INSERT INTO execution_input_receives VALUES(?,?,?,?,?)",
                [
                    (
                        r.execution_id,
                        r.module_path,
                        r.receive_index,
                        r.request_json,
                        json.dumps(
                            [item.to_dict() for item in selected[r.execution_id]],
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=False,
                        ),
                    )
                    for r in pending
                    if selected[r.execution_id] or r.seal_if_empty
                ],
            )
        observed_empty: dict[tuple[str, str], list[tuple[str, int]]] = {}
        for request in pending:
            if selected[request.execution_id] or request.seal_if_empty:
                continue
            identity = (request.execution_id, request.module_path)
            ranges = observed_empty.setdefault(
                identity, list(empty_ranges.get(identity, []))
            )
            if ranges and ranges[-1][0] == request.request_json:
                ranges[-1] = (ranges[-1][0], request.receive_index)
            else:
                ranges.append((request.request_json, request.receive_index))
        if observed_empty:
            await db.executemany(
                "INSERT OR REPLACE INTO execution_input_receives VALUES(?,?,?,?,?)",
                [
                    (
                        execution_id,
                        module_path,
                        _EMPTY_RECEIPT_INDEX,
                        json.dumps(
                            {
                                "empty_ranges": [
                                    {"request": request_json, "through": through}
                                    for request_json, through in ranges
                                ]
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        "[]",
                    )
                    for (execution_id, module_path), ranges in observed_empty.items()
                ],
            )
        results.update(
            (execution_id, tuple(items)) for execution_id, items in selected.items()
        )
        return [results[r.execution_id] for r in requests]
