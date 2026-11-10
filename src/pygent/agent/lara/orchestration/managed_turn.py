from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from pygent.agent.lara.schema import CaseRunRef

from .models import TurnCommand, TurnState


@dataclass(slots=True)
class ManagedSessionTurn:
    """Transport-independent state and controls for one submitted session turn."""

    lease: Any
    command: TurnCommand
    run_ref: CaseRunRef | None = None
    recovery_execution_id: str | None = None
    state: TurnState = TurnState.QUEUED
    task: asyncio.Task[None] | None = None
    execution_handle: Any | None = None
    startup_error: BaseException | None = None
    output: Any | None = None
    output_context: Any | None = None
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    finalized: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def execution_id(self) -> str | None:
        return (
            None
            if self.execution_handle is None
            else self.execution_handle.execution_id
        )

    @property
    def manager(self) -> Any:
        return self.lease.runtime.manager

    @property
    def runtime_service(self) -> Any:
        return self.lease.runtime.runtime_service

    @property
    def done(self) -> bool:
        return self.finalized.is_set()

    @property
    def status(self) -> str:
        return self.state.value

    async def wait_ready(self) -> None:
        await self.ready.wait()

    async def wait_finalized(self) -> None:
        await self.finalized.wait()

    async def result(self) -> tuple[Any, Any]:
        await self.wait_finalized()
        if self.startup_error is not None:
            raise self.startup_error
        return self.output, self.output_context

    async def cancel(self) -> bool:
        task = self.task
        if task is None or task.done():
            return False
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return True

    def subscribe(self, *, after: int | None = None) -> Any:
        if self.execution_handle is None:
            raise RuntimeError("session turn execution is not ready")
        return self.execution_handle.subscribe(after=after)


__all__ = ["ManagedSessionTurn"]
