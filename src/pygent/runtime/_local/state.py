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
    JsonValue,
    Message,
    Module,
    freeze_json_object,
    thaw_json,
)
from pygent.tool import ToolTaskManager

from .._history_store import SQLiteHistoryStore
from ..api import ExecutionEvent, ExecutionStatus
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
    status: ExecutionStatus = ExecutionStatus.QUEUED
    events: list[ExecutionEvent] = field(default_factory=list)
    event_condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    event_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    task: asyncio.Task[tuple[Message, Context]] | None = None
    child_calls: int = 0
    module_calls: dict[str, int] = field(default_factory=dict)
    effect_calls: dict[str, int] = field(default_factory=dict)
    runnable_held: bool = False
    deadline_fired: bool = False
    history: SQLiteHistoryStore | None = None
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
        payload: dict[str, JsonValue] = dict(data)
        if execution_id is not None and execution_id != self.execution_id:
            payload.setdefault("origin_execution_id", execution_id)
        frozen = freeze_json_object(payload)
        async with self.event_lock:
            event = ExecutionEvent(
                schema_version=EXECUTION_EVENT_SCHEMA_VERSION,
                event_id=event_id or str(uuid.uuid4()),
                execution_id=self.execution_id,
                trace_id=self.trace_id,
                span_id=effective_span_id,
                parent_span_id=effective_parent_span_id,
                module_path=module_path,
                sequence=len(self.events),
                timestamp_unix_ns=timestamp_unix_ns or time.time_ns(),
                kind=kind,
                data=frozen,
            )
            self.events.append(event)
            if self.history is not None:
                await self.history.append_event(
                    execution_id=self.execution_id,
                    index=event.sequence,
                    event={
                        "schema_version": event.schema_version,
                        "event_id": event.event_id,
                        "execution_id": event.execution_id,
                        "trace_id": event.trace_id,
                        "span_id": event.span_id,
                        "parent_span_id": event.parent_span_id,
                        "module_path": event.module_path,
                        "sequence": event.sequence,
                        "timestamp_unix_ns": event.timestamp_unix_ns,
                        "kind": event.kind,
                        "data": thaw_json(cast(JsonValue, event.data)),
                    },
                )
        async with self.event_condition:
            self.event_condition.notify_all()
        return event

    async def notify_terminal(self) -> None:
        async with self.event_condition:
            self.event_condition.notify_all()


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
