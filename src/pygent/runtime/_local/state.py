"""Mutable state records and context-local execution identity."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, cast

from pygent.core import (
    EXECUTION_EVENT_SCHEMA_VERSION,
    Context,
    ExecutionFailure,
    FrozenJsonObject,
    JsonValue,
    Message,
    Module,
    freeze_json_object,
)
from pygent.core.execution import _trusted_execution_event
from pygent.core.json_values import (
    _patch_frozen_json_object,
)
from pygent.tool import ToolTaskManager

from .._history_store import SQLiteHistoryStore
from .._history_types import _json_frozen_object
from ..api import (
    ExecutionEvent,
    ExecutionOutcome,
    ExecutionOwnerState,
    ExecutionPhase,
    ExecutionSnapshot,
    ExecutionStatus,
)
from ..plan import ExecutionPlan
from .capacity import _BindingState


@dataclass(slots=True)
class _ExecutionRecord:
    execution_id: str
    trace_id: str
    root_span_id: str
    request_id: str
    binding_state: _BindingState
    plan: ExecutionPlan
    graph: dict[str, Module[Any, Any]]
    deadline: float | None
    parent_execution_id: str | None = None
    parent_span_id: str | None = None
    status: ExecutionStatus = ExecutionStatus.PENDING
    phase: ExecutionPhase = ExecutionPhase.SUBMITTING
    owner_state: ExecutionOwnerState = ExecutionOwnerState.ACTIVE
    attempt_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    submitted_at_unix_ns: int = field(default_factory=time.time_ns)
    updated_at_unix_ns: int = field(default_factory=time.time_ns)
    terminal_sequence: int | None = None
    outcome: ExecutionOutcome | None = None
    owner_id: str | None = None
    fencing_token: int | None = None
    events: list[ExecutionEvent] = field(default_factory=list)
    event_base_sequence: int = 0
    next_sequence: int = 0
    committed_sequence: int = -1
    event_stream_closed: bool = False
    event_condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    active_subscribers: int = 0
    event_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    journal_tail: asyncio.Future[None] | None = None
    journal_error: BaseException | None = None
    journal_notification: asyncio.Task[None] | None = None
    task: asyncio.Task[tuple[Message, Context]] | None = None
    child_calls: int = 0
    module_calls: dict[str, int] = field(default_factory=dict)
    effect_calls: dict[str, int] = field(default_factory=dict)
    runnable_held: bool = False
    deadline_fired: bool = False
    history: SQLiteHistoryStore | None = None
    history_started: bool = False
    attempt: int = 1
    idempotency_key: str | None = None
    model_calls: Any = None
    model_admission: Any = None
    deferred_tool_tasks: list[tuple[ToolTaskManager, str]] = field(default_factory=list)

    @property
    def terminal(self) -> bool:
        return self.status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
            ExecutionStatus.DEADLINE_EXCEEDED,
        }

    async def emit(
        self,
        *,
        execution_id: str | None = None,
        parent_execution_id: str | None,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        event_id: str | None = None,
        timestamp_unix_ns: int | None = None,
        module_path: str,
        kind: str,
        data: Mapping[str, JsonValue],
    ) -> ExecutionEvent:
        frame = _execution_frame.get()
        effective_span_id = span_id or (self.root_span_id if frame is None else frame.span_id)
        effective_parent_span_id = (
            parent_span_id
            if parent_span_id is not None
            else (self.parent_span_id if frame is None else frame.parent_span_id)
        )
        payload: Mapping[str, JsonValue] = data
        foreign_execution = execution_id is not None and execution_id != self.execution_id
        if foreign_execution:
            if isinstance(data, FrozenJsonObject):
                payload = (
                    data
                    if "origin_execution_id" in data
                    else _patch_frozen_json_object(
                        data,
                        {"origin_execution_id": execution_id},
                        overwrite=False,
                    )
                )
            else:
                copied_payload = dict(data)
                copied_payload.setdefault("origin_execution_id", execution_id)
                payload = copied_payload
        history = self.history if self.history_started else None
        reserved = False
        if history is not None:
            await history._reserve_event_slot()
            reserved = True
        try:
            async with self.event_lock:
                if self.phase is ExecutionPhase.FINALIZING or self.event_stream_closed:
                    raise RuntimeError("execution journal is finalizing")
                if self.journal_error is not None:
                    raise self.journal_error
                event = _trusted_execution_event(
                    schema_version=EXECUTION_EVENT_SCHEMA_VERSION,
                    event_id=event_id or str(uuid.uuid4()),
                    execution_id=self.execution_id,
                    attempt_id=self.attempt_id,
                    trace_id=self.trace_id,
                    span_id=effective_span_id,
                    parent_span_id=effective_parent_span_id,
                    module_path=module_path,
                    sequence=self.next_sequence,
                    timestamp_unix_ns=timestamp_unix_ns or time.time_ns(),
                    kind=kind,
                    data=(
                        payload
                        if isinstance(payload, FrozenJsonObject)
                        else freeze_json_object(payload)
                    ),
                )
                self.next_sequence += 1
                self.events.append(event)
                if history is not None:
                    self.journal_tail = history._enqueue_reserved_event_payload(
                        self.execution_id,
                        event.sequence,
                        self._event_payload(event),
                        on_commit=self._mark_journal_committed,
                        on_error=self._mark_journal_failed,
                    )
                    reserved = False
        except BaseException:
            if reserved:
                assert history is not None
                history._event_capacity.release()
            raise
        if history is None:
            await self._publish_committed(event.sequence)
        return event

    def _mark_journal_committed(self, sequence: int) -> None:
        self.committed_sequence = max(self.committed_sequence, sequence)
        if self.active_subscribers == 0:
            return
        task = self.journal_notification
        if task is None or task.done():
            self.journal_notification = asyncio.create_task(
                self._notify_journal_committed(),
                name=f"pygent-journal-notify-{self.execution_id}",
            )

    def _mark_journal_failed(self, exc: BaseException) -> None:
        if self.journal_error is None:
            self.journal_error = exc

    async def _notify_journal_committed(self) -> None:
        async with self.event_condition:
            self.event_condition.notify_all()

    async def _publish_committed(self, sequence: int) -> None:
        if self.active_subscribers == 0:
            self.committed_sequence = max(self.committed_sequence, sequence)
            return
        async with self.event_condition:
            self.committed_sequence = max(self.committed_sequence, sequence)
            self.event_condition.notify_all()

    @staticmethod
    def _event_value(event: ExecutionEvent) -> dict[str, JsonValue]:
        return {
            "schema_version": event.schema_version,
            "event_id": event.event_id,
            "execution_id": event.execution_id,
            "attempt_id": event.attempt_id,
            "trace_id": event.trace_id,
            "span_id": event.span_id,
            "parent_span_id": event.parent_span_id,
            "module_path": event.module_path,
            "sequence": event.sequence,
            "timestamp_unix_ns": event.timestamp_unix_ns,
            "kind": event.kind,
            "data": cast(JsonValue, event.data),
        }

    @classmethod
    def _event_payload(cls, event: ExecutionEvent) -> str:
        return _json_frozen_object(cls._event_value(event))

    async def finalize(
        self,
        *,
        status: ExecutionStatus,
        terminal_events: tuple[tuple[str, Mapping[str, JsonValue]], ...],
        output: object | None = None,
        error: object | None = None,
    ) -> None:
        """Publish the terminal journal and materialized status as one boundary."""

        if not status.terminal:
            raise ValueError("finalization requires a terminal status")
        failure: ExecutionFailure | None = None
        stored_error = error
        if status is not ExecutionStatus.SUCCEEDED:
            error_value = error if isinstance(error, Mapping) else {}
            kind = str(error_value.get("type", status.value))
            message = str(
                error_value.get("message", f"execution ended with {status.value}")
            )
            failure = ExecutionFailure(
                domain="runtime",
                kind=kind,
                message=message,
                details={
                    key: value
                    for key, value in error_value.items()
                    if key not in {"type", "message"}
                },
            )
            stored_error = failure.to_dict()
        self.phase = ExecutionPhase.FINALIZING
        async with self.event_lock:
            journal_tail = self.journal_tail
        if journal_tail is not None:
            await asyncio.shield(journal_tail)
        if self.journal_error is not None:
            raise self.journal_error
        async with self.event_lock:
            prepared: list[ExecutionEvent] = []
            for kind, data in terminal_events:
                prepared.append(
                    _trusted_execution_event(
                        schema_version=EXECUTION_EVENT_SCHEMA_VERSION,
                        event_id=str(uuid.uuid4()),
                        execution_id=self.execution_id,
                        attempt_id=self.attempt_id,
                        trace_id=self.trace_id,
                        span_id=self.root_span_id,
                        parent_span_id=self.parent_span_id,
                        module_path=self.plan.root,
                        sequence=self.next_sequence + len(prepared),
                        timestamp_unix_ns=time.time_ns(),
                        kind=kind,
                        data=(
                            data
                            if isinstance(data, FrozenJsonObject)
                            else freeze_json_object(data)
                        ),
                    )
                )
            terminal_sequence = prepared[-1].sequence
        if self.history is not None and self.history_started:
            await self.history.finalize_execution(
                self.execution_id,
                status=status.value,
                output=output,
                error=stored_error,
                terminal_events=(),
                _terminal_event_payloads=tuple(
                    (event.sequence, self._event_payload(event)) for event in prepared
                ),
                terminal_sequence=terminal_sequence,
            )
        async with self.event_lock:
            self.events.extend(prepared)
            self.next_sequence = terminal_sequence + 1
            self.committed_sequence = terminal_sequence
            self.status = status
            self.phase = ExecutionPhase.TERMINAL
            self.owner_state = ExecutionOwnerState.TERMINAL
            self.terminal_sequence = terminal_sequence
            self.outcome = ExecutionOutcome(
                execution_id=self.execution_id,
                status=status,
                attempt_id=self.attempt_id,
                terminal_sequence=terminal_sequence,
                error=failure,
            )
            self.updated_at_unix_ns = time.time_ns()
            self.event_stream_closed = True
        if self.active_subscribers:
            async with self.event_condition:
                self.event_condition.notify_all()

    async def notify_terminal(self) -> None:
        self.phase = ExecutionPhase.TERMINAL
        self.owner_state = ExecutionOwnerState.TERMINAL
        self.updated_at_unix_ns = time.time_ns()
        if self.terminal_sequence is None and self.committed_sequence >= 0:
            self.terminal_sequence = self.committed_sequence
        self.event_stream_closed = True
        if self.active_subscribers:
            async with self.event_condition:
                self.event_condition.notify_all()

    def snapshot(self) -> ExecutionSnapshot:
        return ExecutionSnapshot(
            execution_id=self.execution_id,
            trace_id=self.trace_id,
            status=self.status,
            phase=self.phase,
            owner_state=self.owner_state,
            attempt_id=self.attempt_id,
            last_sequence=self.committed_sequence,
            terminal_sequence=self.terminal_sequence,
            submitted_at_unix_ns=self.submitted_at_unix_ns,
            updated_at_unix_ns=self.updated_at_unix_ns,
        )


_module_stack: ContextVar[tuple[str, ...]] = ContextVar(
    "pygent_managed_module_stack", default=()
)


@dataclass(slots=True)
class _ExecutionFrame:
    execution_id: str
    parent_execution_id: str | None
    span_id: str
    parent_span_id: str | None
    module_path: str
    module_occurrence: int
    runtime: Any
    binding_state: _BindingState
    deadline: float | None
    runnable_held: bool = False
    model_calls: Any = None
    model_admission: Any = None


_execution_frame: ContextVar[_ExecutionFrame | None] = ContextVar(
    "pygent_managed_execution_frame", default=None
)

__all__ = ["_ExecutionFrame", "_ExecutionRecord", "_execution_frame", "_module_stack"]
