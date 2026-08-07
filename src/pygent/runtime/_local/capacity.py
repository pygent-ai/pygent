"""Capacity primitives used by the process-local runtime."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from pygent.core import CapacityPermit

from ..api import (
    Binding,
    CapacityPolicy,
    ExecutionAdmissionError,
    ExecutionCapacityPolicy,
    ExecutionCapacityState,
    ResourceCapacityGate,
    RunnableCapacityGate,
)


@dataclass(slots=True)
class _ExecutionCapacityState:
    name: str
    capacity: ExecutionCapacityPolicy
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    live_executions: int = 0
    queued_executions: int = 0
    waiters: int = 0
    runnable: _RunnableGate = field(init=False)

    def __post_init__(self) -> None:
        self.runnable = _RunnableGate(self.capacity.max_runnable_executions)

    async def admit(self) -> None:
        capacity = self.capacity
        async with self.condition:
            if self.live_executions >= capacity.max_live_executions:
                if self.queued_executions >= capacity.max_queue_size:
                    raise ExecutionAdmissionError(
                        f"Binding {self.name!r} START queue is full"
                    )
                self.queued_executions += 1
                try:
                    await self.condition.wait_for(
                        lambda: self.live_executions < capacity.max_live_executions
                    )
                finally:
                    self.queued_executions -= 1
            self.live_executions += 1

    async def release_live(self) -> None:
        async with self.condition:
            self.live_executions -= 1
            self.condition.notify(1)

    @asynccontextmanager
    async def waiter_slot(self) -> AsyncIterator[CapacityPermit]:
        async with self.condition:
            if self.waiters >= self.capacity.max_waiters:
                raise ExecutionAdmissionError(
                    f"Binding {self.name!r} external waiter capacity is full"
                )
            self.waiters += 1
        try:
            yield CapacityPermit(owner_key=f"execution:{self.name}:waiters")
        finally:
            async with self.condition:
                self.waiters -= 1


@dataclass(slots=True)
class _BindingState:
    policy: Binding
    deployment_scope_id: str
    execution: ExecutionCapacityState
    model: ResourceCapacityGate
    tool: ResourceCapacityGate

    @property
    def runnable(self) -> RunnableCapacityGate:
        return self.execution.runnable

    async def admit(self) -> None:
        await self.execution.admit()

    async def release_live(self) -> None:
        await self.execution.release_live()


class _ResourceGate:
    def __init__(self, policy: CapacityPolicy, name: str) -> None:
        self._policy = policy
        self._name = name
        self._semaphore = (
            asyncio.Semaphore(policy.max_concurrency)
            if policy.max_concurrency is not None
            else None
        )
        self._waiting = 0
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def permit(self) -> AsyncIterator[CapacityPermit]:
        semaphore = self._semaphore
        if semaphore is None:
            yield CapacityPermit(owner_key=self._name)
            return
        acquired = False
        async with self._lock:
            if semaphore.locked():
                max_queue = self._policy.max_queue_size or 0
                if self._waiting >= max_queue:
                    raise ExecutionAdmissionError(f"{self._name} capacity queue is full")
                self._waiting += 1
        try:
            await semaphore.acquire()
            acquired = True
            async with self._lock:
                if self._waiting:
                    self._waiting -= 1
            yield CapacityPermit(owner_key=self._name)
        finally:
            if not acquired:
                async with self._lock:
                    if self._waiting:
                        self._waiting -= 1
            else:
                semaphore.release()


class _RunnableGate:
    """FIFO runnable gate with bounded RESUME priority and START fairness."""

    _MAX_CONSECUTIVE_RESUMES = 8

    def __init__(self, capacity: int) -> None:
        self._available = capacity
        self._resume_waiters: deque[asyncio.Future[None]] = deque()
        self._start_waiters: deque[asyncio.Future[None]] = deque()
        self._consecutive_resumes = 0

    async def acquire(self, *, resume: bool = False) -> None:
        future = asyncio.get_running_loop().create_future()
        queue = self._resume_waiters if resume else self._start_waiters
        queue.append(future)
        self._drain()
        try:
            await future
        except asyncio.CancelledError:
            if future.done() and not future.cancelled():
                self.release()
            else:
                future.cancel()
                self._discard(future)
                self._drain()
            raise

    def release(self) -> None:
        self._available += 1
        self._drain()

    def _discard(self, future: asyncio.Future[None]) -> None:
        for queue in (self._resume_waiters, self._start_waiters):
            try:
                queue.remove(future)
            except ValueError:
                pass

    def _drain(self) -> None:
        while self._available > 0:
            for waiters in (self._resume_waiters, self._start_waiters):
                while waiters and waiters[0].cancelled():
                    waiters.popleft()
            if self._resume_waiters and (
                not self._start_waiters
                or self._consecutive_resumes < self._MAX_CONSECUTIVE_RESUMES
            ):
                queue = self._resume_waiters
                self._consecutive_resumes = min(
                    self._consecutive_resumes + 1,
                    self._MAX_CONSECUTIVE_RESUMES,
                )
            else:
                queue = self._start_waiters
                if queue:
                    self._consecutive_resumes = 0
            if not queue:
                return
            future = queue.popleft()
            if future.cancelled():
                continue
            self._available -= 1
            future.set_result(None)


class InMemoryCapacityCoordinator:
    """Share deployment-scoped capacity across LocalRuntime instances."""

    def __init__(self) -> None:
        self._execution_states: dict[str, tuple[ExecutionCapacityPolicy, _ExecutionCapacityState]] = {}
        self._resource_gates: dict[
            tuple[str, str], tuple[CapacityPolicy, _ResourceGate]
        ] = {}

    def execution_state(self, name: str, policy: ExecutionCapacityPolicy) -> _ExecutionCapacityState:
        current = self._execution_states.get(name)
        if current is None:
            state = _ExecutionCapacityState(name, policy)
            self._execution_states[name] = (policy, state)
            return state
        current_policy, state = current
        if current_policy != policy:
            raise ValueError(
                f"deployment Binding {name!r} has conflicting execution capacity"
            )
        return state

    def resource_gate(
        self,
        kind: str,
        capacity_key: str,
        policy: CapacityPolicy,
    ) -> _ResourceGate:
        key = (kind, capacity_key)
        current = self._resource_gates.get(key)
        if current is None:
            gate = _ResourceGate(policy, f"{kind}:{capacity_key}")
            self._resource_gates[key] = (policy, gate)
            return gate
        current_policy, gate = current
        if current_policy != policy:
            raise ValueError(
                f"deployment {kind} capacity {capacity_key!r} has conflicting policies"
            )
        return gate


__all__ = [
    "InMemoryCapacityCoordinator",
    "_BindingState",
    "_ExecutionCapacityState",
    "_ResourceGate",
    "_RunnableGate",
]
