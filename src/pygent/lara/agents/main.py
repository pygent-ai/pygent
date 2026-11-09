"""The main agent: the single-agent appearance of the LARA architecture.

The user talks to one Lara (G1). This scaffold keeps that contract while the
main agent submits any number of background jobs — forks and independent
children (G2) — whose outcomes are collected as results, never surfaced as
separate conversations.

Lifecycle note: handles in this scaffold are in-memory. Durable job records,
capacity leases, deadlines, and event streams belong to the runtime
orchestrator (lara-architecture §6.8); MainAgent is the submission and
collection point, not the lifecycle owner.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Any

from .child import ChildAgent, ChildSpec
from .context import ContextWindow
from .fork import ForkAgent, ForkSpec
from .runner import AgentLimits, AgentRunResult, AgentRunner, ToolProfile


class BackgroundJobHandle:
    """Handle for one submitted background job.

    Exposes observability (``done``) and control (``cancel``) without
    exposing the underlying task; the job registry remains the single place
    that drains outcomes.
    """

    def __init__(
        self,
        *,
        job_id: str,
        label: str,
        task: asyncio.Task[AgentRunResult],
    ) -> None:
        self.job_id = job_id
        self.label = label
        self._task = task

    @property
    def done(self) -> bool:
        return self._task.done()

    @property
    def cancelled(self) -> bool:
        return self._task.cancelled()

    def cancel(self) -> bool:
        return self._task.cancel()

    async def wait(self) -> AgentRunResult:
        """Await the job's completion.

        Raises ``asyncio.CancelledError`` when the job was cancelled; callers
        that collect outcomes should catch it and record the cancellation.
        """

        return await self._task


@dataclass(frozen=True)
class BackgroundOutcome:
    """One collected background job outcome.

    ``cancelled`` distinguishes a cooperative cancellation from a failure;
    both are values, never raises, so one bad job cannot break collection or
    leave ghost registry entries.
    """

    label: str
    job_id: str
    result: AgentRunResult | None = None
    error: Exception | None = None
    cancelled: bool = False


class MainAgent:
    """Foreground representative able to submit any number of background jobs."""

    def __init__(
        self,
        runner: AgentRunner,
        *,
        system_prompt: str,
        tools: ToolProfile | None = None,
        label: str = "lara-main",
    ) -> None:
        if not system_prompt.strip():
            raise ValueError("MainAgent.system_prompt must be non-empty")
        self._runner = runner
        self._system_prompt = system_prompt
        self._tools = tools
        self._label = label
        # Registered background jobs, keyed by unique job id. Labels live in
        # ``_in_flight_labels`` and are released only when a job's outcome
        # has been delivered by collect() — never while the job runs.
        self._background: dict[str, BackgroundJobHandle] = {}
        self._in_flight_labels: set[str] = set()
        self._background_seq = 0
        self._collect_lock = asyncio.Lock()
        # Outcomes already collected but not yet delivered: populated when a
        # collect() is cancelled mid-drain so no result is lost.
        self._completed: list[BackgroundOutcome] = []

    async def run_turn(
        self,
        message: str,
        *,
        prefix: ContextWindow | None = None,
        limits: AgentLimits | None = None,
    ) -> AgentRunResult:
        """Run one foreground turn (optionally continuing a forked window).

        With ``prefix`` the fork window's system prompt and messages are
        continued verbatim; the main agent's own system prompt applies only
        to fresh turns.
        """

        return await self._runner.run(
            message=message,
            system_prompt=None if prefix is not None else self._system_prompt,
            prefix=prefix,
            tools=self._tools,
            limits=limits,
            label=self._label,
        )

    def spawn_fork(
        self,
        spec: ForkSpec,
        *,
        directive: str,
        label: str | None = None,
        limits: AgentLimits | None = None,
    ) -> BackgroundJobHandle:
        """Submit a fork agent run as a background job.

        ``limits`` apply to this job only, overriding the runner defaults.
        """

        resolved = self._reserve_label(label)
        fork = ForkAgent(self._runner, spec, label=resolved)
        return self._register_job(
            resolved=resolved, coro=fork.run(directive, limits=limits)
        )

    def spawn_child(
        self,
        spec: ChildSpec,
        *,
        directive: str,
        label: str | None = None,
    ) -> BackgroundJobHandle:
        """Submit an independent child agent run as a background job."""

        resolved = self._reserve_label(label)
        child = ChildAgent(self._runner, spec, label=resolved)
        return self._register_job(resolved=resolved, coro=child.run(directive))

    def _register_job(
        self,
        *,
        resolved: str,
        coro: Coroutine[Any, Any, AgentRunResult],
    ) -> BackgroundJobHandle:
        """Create the job task and register it under a fresh unique id.

        If task creation fails (e.g. no running event loop), the reserved
        label is released so it does not leak, and the not-yet-started
        coroutine is closed to avoid an un-awaited warning.
        """

        job_id = uuid.uuid4().hex
        try:
            task = asyncio.create_task(coro, name=f"lara-agent:{resolved}")
        except BaseException:
            self._in_flight_labels.discard(resolved)
            coro.close()
            raise
        handle = BackgroundJobHandle(job_id=job_id, label=resolved, task=task)
        self._background[job_id] = handle
        return handle

    async def collect(self) -> tuple[BackgroundOutcome, ...]:
        """Await every submitted background job and drain the registry.

        Cancellations and failures become outcomes, so collection always
        completes and the registry is empty afterwards — a cancelled or
        failed job can never block the others or linger as a ghost entry.

        Ownership and cancellation semantics:

        - Single collector: concurrent ``collect()`` calls serialize on a
          lock instead of racing over the registry.
        - Cancelling ``collect()`` only stops waiting. The in-flight job is
          shielded and keeps running, its handle stays registered, and its
          label stays reserved until its outcome is delivered by a later
          ``collect()``. Already-collected outcomes are buffered and
          returned by the next call; the cancellation itself is re-raised.
        - Cancelling the *job* itself (via its handle) remains a recorded
          cancelled outcome that releases the label.
        """

        async with self._collect_lock:
            outcomes: list[BackgroundOutcome] = list(self._completed)
            self._completed.clear()
            for job_id in list(self._background):
                handle = self._background[job_id]
                outcome = await self._await_outcome(handle)
                if outcome is None:
                    # collect() itself was cancelled: the handle stays
                    # registered, the label stays reserved, and partial
                    # outcomes are buffered for the next collect().
                    self._completed = outcomes
                    raise asyncio.CancelledError
                outcomes.append(outcome)
                del self._background[job_id]
                self._in_flight_labels.discard(handle.label)
            return tuple(outcomes)

    async def _await_outcome(
        self, handle: BackgroundJobHandle
    ) -> BackgroundOutcome | None:
        """Await one job and build its outcome.

        Returns ``None`` only when collect() itself was cancelled: the
        in-flight wait is shielded so the cancellation never reaches the
        job, and the caller must keep the handle registered.
        """

        try:
            # Shield: cancelling collect() must not cancel the job.
            result = await asyncio.shield(handle.wait())
        except asyncio.CancelledError:
            if handle.cancelled:
                return BackgroundOutcome(
                    label=handle.label, job_id=handle.job_id, cancelled=True
                )
            return None
        except Exception as exc:  # noqa: BLE001 - background isolation is the point
            return BackgroundOutcome(
                label=handle.label, job_id=handle.job_id, error=exc
            )
        return BackgroundOutcome(
            label=handle.label, job_id=handle.job_id, result=result
        )

    def _reserve_label(self, label: str | None) -> str:
        self._background_seq += 1
        resolved = label or f"{self._label}.bg{self._background_seq}"
        if resolved in self._in_flight_labels:
            raise ValueError(f"background label {resolved!r} is already in flight")
        self._in_flight_labels.add(resolved)
        return resolved
