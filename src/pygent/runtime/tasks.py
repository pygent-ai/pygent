"""SQLite-backed durable ToolTask admission and recovery."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from uuid import uuid4

from pygent.core import FrozenJsonObject, JsonValue, freeze_json
from pygent.tool import (
    ExecutorRegistry,
    IdempotencyPolicy,
    ToolCall,
    ToolExecutionContext,
    ToolResult,
    ToolSideEffect,
    ToolSpec,
    ToolTask,
    ToolTaskState,
)
from pygent.tool.executors import (
    ToolTaskExecution,
    _cancelled_task_result,
    _execute_with_timeout,
    result_from_exception,
)

from ._history_store import SQLiteHistoryStore
from ._history_types import StoredJob
from .api import JobSnapshot, JobState
from .codec import (
    WireCodecError,
    _tool_call_from_dict,
    _tool_call_to_dict,
    _tool_spec_from_dict,
    _tool_spec_to_dict,
    tool_result_from_dict,
    tool_result_to_dict,
)


def _request_to_dict(spec: ToolSpec, call: ToolCall) -> dict[str, object]:
    return {"spec": _tool_spec_to_dict(spec), "call": _tool_call_to_dict(call)}


def _request_from_dict(value: object) -> tuple[ToolSpec, ToolCall]:
    if not isinstance(value, FrozenJsonObject):
        raise TypeError("durable ToolTask request is invalid")
    try:
        return _tool_spec_from_dict(value["spec"]), _tool_call_from_dict(value["call"])
    except (KeyError, WireCodecError) as exc:
        raise TypeError("durable ToolTask request is invalid") from exc


class DurableToolTaskManager:
    """Durable ToolTask manager with explicit crash recovery semantics."""

    def __init__(
        self, history: SQLiteHistoryStore, registry: ExecutorRegistry
    ) -> None:
        self.history = history
        self.registry = registry
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._executions: dict[str, ToolTaskExecution] = {}
        self._lock = asyncio.Lock()
        self._prepared_ids: set[str] = set()
        self._owner_id = f"tool-owner-{uuid4()}"
        self._lease_ttl = 30.0
        self._heartbeat: asyncio.Task[None] | None = None

    async def _observe(self, task_id: str) -> None:
        await self.history.admit_tool_observation(task_id, self._owner_id, self._lease_ttl)
        if self._heartbeat is None:
            self._heartbeat = asyncio.create_task(self._renew_observations())

    async def _renew_observations(self) -> None:
        while True:
            await asyncio.sleep(self._lease_ttl / 3)
            await self.history.renew_tool_observations(self._owner_id, self._lease_ttl)

    async def _publish_output(self, task_id: str, value: JsonValue) -> None:
        await self.history.put_tool_output(task_id, self._owner_id, freeze_json(value))

    async def get_output(self, task_id: str) -> JsonValue:
        return await self.history.get_tool_output(task_id)

    @staticmethod
    def _retry_safe(spec: ToolSpec, call: ToolCall | None = None) -> bool:
        return (
            spec.side_effect in (ToolSideEffect.PURE, ToolSideEffect.READ)
            or spec.idempotency is IdempotencyPolicy.INHERENT
            or (
                spec.idempotency is IdempotencyPolicy.REQUIRES_KEY
                and call is not None
                and call.idempotency_key is not None
            )
        )

    async def prepare_job(
        self,
        spec: ToolSpec,
        call: ToolCall,
        *,
        logical_key: str,
        binding_id: str,
        plan_id: str,
        required_capabilities: tuple[str, ...],
        execution: ToolTaskExecution,
    ) -> ToolTask:
        """Reliably admit a durable Job carrying one ToolTask."""

        job_id = f"job-{uuid4()}"
        task_id = f"tool-{uuid4()}"
        stored = await self.history.create_tool_job(
            job_id=job_id,
            task_id=task_id,
            logical_key=logical_key,
            binding_id=binding_id,
            plan_id=plan_id,
            resource_key=spec.resource_key,
            required_capabilities=required_capabilities,
            request=_request_to_dict(spec, call),
            observation_owner=self._owner_id,
            observation_lease_ttl=self._lease_ttl,
        )
        if stored.task_id == task_id:
            self._prepared_ids.add(task_id)
        await self._observe(stored.task_id)
        self._executions[stored.task_id] = execution
        return self._job_task(stored, spec, call)

    async def recover_job(
        self, stored: StoredJob, *, execution: ToolTaskExecution
    ) -> JobSnapshot:
        """Recover a validated Job only through a Runtime-provided execution path."""

        spec, call = _request_from_dict(stored.request)
        if spec.resource_key != stored.resource_key:
            raise RuntimeError("durable Job resource identity is incompatible")
        if spec.wait_timeout is not None:
            await self._recover_observation(stored.task_id)
            return await self.get_job(stored.job_id) or self._job_snapshot(stored)
        acquired = await self.history.admit_tool_observation(
            stored.task_id, self._owner_id, self._lease_ttl, replace_owner=True,
        )
        if not acquired:
            return await self.get_job(stored.job_id) or self._job_snapshot(stored)
        # The supplied snapshot may predate a concurrent owner's terminal write.
        current = await self.history.get_job(stored.job_id)
        if current is not None:
            stored = current
        await self._observe(stored.task_id)
        if stored.task_id in self._tasks:
            return await self.get_job(stored.job_id) or self._job_snapshot(stored)
        if stored.status == JobState.RUNNING.value and not self._retry_safe(spec, call):
            snapshot = self._job_task(
                stored, spec, call, state=ToolTaskState.UNKNOWN
            )
            result = ToolResult(
                call_id=call.call_id,
                name=call.name,
                status="unknown",
                task=snapshot,
                error="worker exited after a side effect may have started",
                error_kind="recovery_uncertain",
                retryable=False,
                side_effect_committed=None,
                tool_id=spec.tool_id,
                tool_version=spec.version,
            )
            await self._store_job_terminal(stored, result)
        elif stored.status in (JobState.PENDING.value, JobState.RUNNING.value):
            self._executions[stored.task_id] = execution
            await self._launch_job(
                stored,
                spec,
                call,
                execution,
                attempt=stored.attempt + (stored.status == JobState.RUNNING.value),
                recovery=True,
            )
        return await self.get_job(stored.job_id) or self._job_snapshot(stored)

    async def submit(
        self,
        spec: ToolSpec,
        call: ToolCall,
        *,
        execution: ToolTaskExecution | None = None,
    ) -> ToolTask:
        snapshot = await self.prepare(spec, call, execution=execution)
        await self.start(snapshot.task_id)
        return snapshot

    async def prepare(
        self,
        spec: ToolSpec,
        call: ToolCall,
        *,
        execution: ToolTaskExecution | None = None,
    ) -> ToolTask:
        task_id = f"tool-{uuid4()}"
        snapshot = self._snapshot(task_id, spec, call, ToolTaskState.PENDING)
        # Persist the owner first so admitted tasks never lack crash detection.
        await self._observe(task_id)
        # Reliable admission point: persistence precedes background execution.
        await self.history.put_task(
            task_id=task_id,
            kind="tool_task",
            status=snapshot.state.value,
            request=_request_to_dict(spec, call),
        )
        self._prepared_ids.add(task_id)
        if execution is not None:
            self._executions[task_id] = execution
        return snapshot

    async def start(self, task_id: str) -> None:
        job = await self.history.get_job_by_task(task_id)
        if job is not None:
            if job.status != JobState.PENDING.value:
                # Re-admission through a recovered Parent may rediscover a
                # running or terminal independent Job.  Only explicit Job
                # recovery may decide whether RUNNING is replay-safe.
                return
            execution = self._executions.get(task_id)
            if execution is None:
                raise RuntimeError(
                    "durable Job requires a validated Runtime execution path"
                )
            spec, call = _request_from_dict(job.request)
            await self._launch_job(
                job,
                spec,
                call,
                execution,
                attempt=job.attempt,
                recovery=False,
            )
            return
        stored = await self.history.get_task(task_id)
        if stored is None or stored.kind != "tool_task":
            raise KeyError(f"unknown prepared ToolTask {task_id!r}")
        if stored.status != ToolTaskState.PENDING.value:
            return
        spec, call = _request_from_dict(stored.request)
        await self._launch(task_id, spec, call, self._executions.get(task_id))

    async def _launch_job(
        self,
        stored: StoredJob,
        spec: ToolSpec,
        call: ToolCall,
        execution: ToolTaskExecution,
        *,
        attempt: int,
        recovery: bool,
    ) -> None:
        async with self._lock:
            existing = self._tasks.get(stored.task_id)
            if existing is not None:
                return
            current = await self.history.get_job(stored.job_id)
            allowed = {JobState.PENDING.value}
            if recovery:
                allowed.add(JobState.RUNNING.value)
            if current is None or current.status not in allowed:
                return
            self._tasks[stored.task_id] = asyncio.create_task(
                self._run_job(
                    stored,
                    spec,
                    call,
                    execution,
                    attempt=attempt,
                    recovery=recovery,
                ),
                name=f"pygent-{stored.job_id}",
            )

    async def _run_job(
        self,
        stored: StoredJob,
        spec: ToolSpec,
        call: ToolCall,
        execution: ToolTaskExecution,
        *,
        attempt: int,
        recovery: bool,
    ) -> None:
        await self.history.update_tool_job(
            stored.job_id, status=JobState.RUNNING.value, attempt=attempt,
            observation_owner=self._owner_id
        )
        try:
            completed = await _execute_with_timeout(
                self.registry, spec, call, execution=execution,
                context=ToolExecutionContext(
                    task_id=stored.task_id, recovery=recovery,
                    publish_output=lambda value: self._publish_output(stored.task_id, value),
                ),
            )
            result = replace(
                completed,
                task=self._job_task(
                    stored, spec, call, state=ToolTaskState.SUCCEEDED
                ),
            )
        except asyncio.CancelledError:
            result = _cancelled_task_result(
                spec,
                call,
                self._job_task(stored, spec, call),
                started=True,
            )
        except Exception as exc:  # noqa: BLE001 - executor result boundary
            task = self._job_task(stored, spec, call, state=ToolTaskState.FAILED)
            result = result_from_exception(spec, call, exc, task=task)
            if result.status == "unknown":
                result = replace(
                    result, task=replace(task, state=ToolTaskState.UNKNOWN)
                )
        await self._store_job_terminal(stored, result)

    async def _store_job_terminal(
        self, stored: StoredJob, result: ToolResult
    ) -> None:
        state = result.task.state if result.task is not None else ToolTaskState.FAILED
        if result.output is not None:
            await self._publish_output(stored.task_id, result.output)
        await self.history.update_tool_job(
            stored.job_id,
            status=state.value,
            result=tool_result_to_dict(result),
            observation_owner=self._owner_id,
        )

    async def _launch(
        self,
        task_id: str,
        spec: ToolSpec,
        call: ToolCall,
        execution: ToolTaskExecution | None = None,
    ) -> None:
        async with self._lock:
            existing = self._tasks.get(task_id)
            if existing is not None:
                return
            current = await self.history.get_task(task_id)
            if current is None or current.status != ToolTaskState.PENDING.value:
                return
            self._tasks[task_id] = asyncio.create_task(
                self._run(task_id, spec, call, execution),
                name=f"pygent-durable-{task_id}",
            )

    async def _run(
        self,
        task_id: str,
        spec: ToolSpec,
        call: ToolCall,
        execution: ToolTaskExecution | None,
    ) -> None:
        await self.history.put_task(
            task_id=task_id,
            kind="tool_task",
            status=ToolTaskState.RUNNING.value,
            observation_owner=self._owner_id,
            request=_request_to_dict(spec, call),
        )
        try:
            completed = await _execute_with_timeout(
                self.registry,
                spec,
                call,
                execution=execution,
                context=ToolExecutionContext(
                    task_id=task_id,
                    publish_output=lambda value: self._publish_output(task_id, value),
                ),
            )
            snapshot = self._snapshot(
                task_id, spec, call, ToolTaskState.SUCCEEDED
            )
            result = replace(
                completed,
                task=snapshot,
            )
        except asyncio.CancelledError:
            result = _cancelled_task_result(
                spec,
                call,
                self._snapshot(task_id, spec, call, ToolTaskState.RUNNING),
                started=True,
            )
        except Exception as exc:  # noqa: BLE001 - executor result boundary
            snapshot = self._snapshot(task_id, spec, call, ToolTaskState.FAILED)
            result = result_from_exception(spec, call, exc, task=snapshot)
            if result.status == "unknown":
                result = replace(
                    result,
                    task=replace(snapshot, state=ToolTaskState.UNKNOWN),
                )
        await self._store_terminal(task_id, spec, call, result)

    async def _store_terminal(
        self,
        task_id: str,
        spec: ToolSpec,
        call: ToolCall,
        result: ToolResult,
    ) -> None:
        state = result.task.state if result.task is not None else ToolTaskState.FAILED
        if result.output is not None:
            await self._publish_output(task_id, result.output)
        await self.history.put_task(
            task_id=task_id,
            kind="tool_task",
            status=state.value,
            observation_owner=self._owner_id,
            request=_request_to_dict(spec, call),
            result=tool_result_to_dict(result),
        )

    @staticmethod
    def _snapshot(
        task_id: str,
        spec: ToolSpec,
        call: ToolCall,
        state: ToolTaskState,
    ) -> ToolTask:
        return ToolTask(
            task_id=task_id,
            call_id=call.call_id,
            tool_id=spec.tool_id,
            version=spec.version,
            state=state,
        )

    @staticmethod
    def _job_task(
        stored: StoredJob,
        spec: ToolSpec,
        call: ToolCall,
        *,
        state: ToolTaskState | None = None,
    ) -> ToolTask:
        return ToolTask(
            task_id=stored.task_id,
            call_id=call.call_id,
            tool_id=spec.tool_id,
            version=spec.version,
            state=state or ToolTaskState(stored.status),
            job_id=stored.job_id,
        )

    @staticmethod
    def _job_snapshot(stored: StoredJob) -> JobSnapshot:
        return JobSnapshot(
            job_id=stored.job_id,
            task_id=stored.task_id,
            logical_key=stored.logical_key,
            state=JobState(stored.status),
            binding_id=stored.binding_id,
            plan_id=stored.plan_id,
            resource_key=stored.resource_key,
            required_capabilities=stored.required_capabilities,
            attempt=stored.attempt,
        )

    async def _recover_observation(self, task_id: str) -> None:
        job = await self.history.get_job_by_task(task_id)
        item = job if job is not None else await self.history.get_task(task_id)
        if item is None or item.status not in {"pending", "running"}:
            return
        spec, call = _request_from_dict(item.request)
        # Replay-capable Jobs retain their existing explicit recovery contract.
        if job is not None and spec.wait_timeout is None:
            return
        snapshot = (
            self._job_task(job, spec, call, state=ToolTaskState.UNKNOWN)
            if job is not None else self._snapshot(task_id, spec, call, ToolTaskState.UNKNOWN)
        )
        result = ToolResult(
            call_id=call.call_id, name=call.name, status="unknown", task=snapshot,
            output=await self.get_output(task_id),
            error="tool owner lease expired; process completion is unconfirmed",
            error_kind="recovery_uncertain", retryable=False,
            side_effect_committed=None, tool_id=spec.tool_id, tool_version=spec.version,
        )
        await self.history.finalize_lost_tool_owner(
            task_id, tool_result_to_dict(result), job=job is not None,
        )

    async def get_job(self, job_id: str) -> JobSnapshot | None:
        stored = await self.history.get_job(job_id)
        return None if stored is None else self._job_snapshot(stored)

    async def get_task(self, task_id: str) -> ToolTask | None:
        await self._recover_observation(task_id)
        job = await self.history.get_job_by_task(task_id)
        if job is not None:
            spec, call = _request_from_dict(job.request)
            return self._job_task(job, spec, call)
        item = await self.history.get_task(task_id)
        if item is None:
            return None
        spec, call = _request_from_dict(item.request)
        return self._snapshot(task_id, spec, call, ToolTaskState(item.status))

    async def cancel(self, task_id: str) -> bool:
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is not None and (not task.done() or task.cancelled()):
                task.cancel()
            else:
                task = None
        if task is None:
            job = await self.history.get_job_by_task(task_id)
            if job is None:
                item = await self.history.get_task(task_id)
                if item is None or item.status != ToolTaskState.PENDING.value:
                    return False
                spec, call = _request_from_dict(item.request)
                result = _cancelled_task_result(
                    spec, call, self._snapshot(task_id, spec, call, ToolTaskState.PENDING),
                    started=False,
                )
                await self._store_terminal(task_id, spec, call, result)
                return True
            if job.status != JobState.PENDING.value:
                return False
            spec, call = _request_from_dict(job.request)
            result = _cancelled_task_result(
                spec,
                call,
                self._job_task(job, spec, call),
                started=False,
            )
            await self._store_job_terminal(job, result)
            return True
        await asyncio.shield(asyncio.gather(task, return_exceptions=True))
        job = await self.history.get_job_by_task(task_id)
        if job is not None and job.result is None:
            # A Task can be cancelled before its coroutine enters _run_job().
            # Persist a terminal result here so cancellation never leaves an
            # admitted Job permanently RUNNING.
            spec, call = _request_from_dict(job.request)
            result = _cancelled_task_result(
                spec,
                call,
                self._job_task(job, spec, call),
                started=job.status == JobState.RUNNING.value,
            )
            await self._store_job_terminal(job, result)
        elif job is None:
            item = await self.history.get_task(task_id)
            if item is not None and item.result is None:
                spec, call = _request_from_dict(item.request)
                result = _cancelled_task_result(
                    spec, call, self._snapshot(task_id, spec, call, ToolTaskState(item.status)),
                    started=item.status == ToolTaskState.RUNNING.value,
                )
                await self._store_terminal(task_id, spec, call, result)
        return True

    async def get_result(
        self, task_id: str, *, wait: bool = False
    ) -> ToolResult | None:
        await self._recover_observation(task_id)
        async with self._lock:
            task = self._tasks.get(task_id)
        if wait and task is not None:
            await asyncio.shield(asyncio.gather(task, return_exceptions=True))
        job = await self.history.get_job_by_task(task_id)
        if job is not None:
            return (
                None
                if job.result is None
                else tool_result_from_dict(job.result)
            )
        item = await self.history.get_task(task_id)
        if item is None or item.result is None:
            return None
        return tool_result_from_dict(item.result)

    async def close(self, *, cancel: bool = False) -> None:
        async with self._lock:
            tasks = tuple(self._tasks.values())
        if cancel:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*(
                self.cancel(task_id) for task_id in set(self._tasks) | self._prepared_ids
            ))
        await asyncio.gather(*tasks, return_exceptions=True)
        if self._heartbeat is not None:
            self._heartbeat.cancel()
            await asyncio.gather(self._heartbeat, return_exceptions=True)
            self._heartbeat = None


__all__ = ["DurableToolTaskManager"]
