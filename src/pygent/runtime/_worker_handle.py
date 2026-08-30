"""Client-side remote execution handle and event subscription."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self
from uuid import uuid4

from pygent.core import (
    ExecutionEvent,
    ExecutionInputDelivery,
    ExecutionOutcome,
    ExecutionSnapshot,
    FrozenJsonObject,
    JsonValue,
    PlacementMode,
)

from ._worker_protocol import WorkerTarget
from .api import _event_cursor

if TYPE_CHECKING:
    from .worker_client import HTTPWorkerClient


@dataclass(frozen=True, slots=True)
class RemoteExecutionHandle:
    """Client-side control plane for one remotely owned execution."""

    client: HTTPWorkerClient = field(repr=False, compare=False)
    execution_id: str
    target: WorkerTarget
    attempt_id: str = field(default_factory=lambda: str(uuid4()))
    binding_ref: str = ""
    input: FrozenJsonObject | None = None
    request_id: str = ""
    required_capabilities: tuple[str, ...] = ()
    idempotency_key: str | None = None
    trace_id: str | None = None
    parent_execution_id: str | None = None
    parent_span_id: str | None = None
    attempt: int = 1
    attempted_target_ids: tuple[str, ...] = ()
    plan_id: str = ""
    graph_hash: str = ""
    placement_mode: PlacementMode = PlacementMode.ADAPTIVE
    pinned_target_id: str | None = None
    model_calls: FrozenJsonObject = field(default_factory=FrozenJsonObject)
    model_admission_ref: str | None = None
    model_store_namespace: str | None = None

    async def snapshot(self) -> ExecutionSnapshot:
        return await self.client.snapshot(self)

    async def outcome(self) -> ExecutionOutcome:
        while True:
            snapshot = await self.snapshot()
            if snapshot.status.terminal:
                assert snapshot.attempt_id is not None
                assert snapshot.terminal_sequence is not None
                return ExecutionOutcome(
                    execution_id=self.execution_id,
                    status=snapshot.status,
                    attempt_id=snapshot.attempt_id,
                    terminal_sequence=snapshot.terminal_sequence,
                )

    async def result(self, *, deadline: float | None = None) -> JsonValue:
        return await self.client.result(self, deadline=deadline)

    async def cancel(self) -> bool:
        return await self.client.cancel(self)

    async def send_input(
        self, *, input_id: str, kind: str, value: JsonValue
    ) -> ExecutionInputDelivery:
        return await self.client.send_input(
            self, input_id=input_id, kind=kind, value=value
        )

    def subscribe(self, *, after: int | None = None) -> _RemoteExecutionSubscription:
        return _RemoteExecutionSubscription(self, _event_cursor(after))


class _RemoteExecutionSubscription:
    def __init__(self, handle: RemoteExecutionHandle, after: int) -> None:
        self._handle = handle
        self._after = after

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def __aiter__(self) -> AsyncIterator[ExecutionEvent]:
        return self._handle.client.events(
            self._handle.target,
            self._handle.execution_id,
            after=self._after,
        )
