"""SQLite-backed durable ToolTask admission and recovery."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any, cast
from uuid import uuid4

from pygent.core import FrozenJsonObject, freeze_json, freeze_json_object, thaw_json
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
    _execute_with_timeout,
    result_from_exception,
)

from ._history_store import SQLiteHistoryStore
from ._history_types import StoredJob
from .api import JobSnapshot, JobState
from .codec import (
    tool_definition_from_dict,
    tool_definition_to_dict,
    tool_result_from_dict,
    tool_result_to_dict,
)


def _request_to_dict(spec: ToolSpec, call: ToolCall) -> dict[str, object]:
    return {
        "spec": {
            "tool_id": spec.tool_id,
            "version": spec.version,
            "definition": tool_definition_to_dict(spec.definition),
            "side_effect": spec.side_effect.value,
            "idempotency": spec.idempotency.value,
            "timeout": spec.timeout,
            "resource_key": spec.resource_key,
            "sandbox_profile": spec.sandbox_profile,
            "required_permissions": list(spec.required_permissions),
        },
        "call": {
            "call_id": call.call_id,
            "name": call.name,
            "arguments": thaw_json(cast(FrozenJsonObject, call.arguments)),
            "tool_id": call.tool_id,
            "tool_version": call.tool_version,
            "idempotency_key": call.idempotency_key,
        },
    }


def _request_from_dict(value: object) -> tuple[ToolSpec, ToolCall]:
    if not isinstance(value, FrozenJsonObject):
        value = freeze_json_object(cast(dict[str, Any], value))
    spec_data = value["spec"]
    call_data = value["call"]
    if not isinstance(spec_data, FrozenJsonObject) or not isinstance(
        call_data, FrozenJsonObject
    ):
        raise TypeError("durable ToolTask request is invalid")
    permissions = spec_data.get("required_permissions", ())
    if not isinstance(permissions, tuple):
        raise TypeError("durable ToolTask permissions are invalid")
    spec = ToolSpec(
        tool_id=cast(str, spec_data["tool_id"]),
        version=cast(str, spec_data["version"]),
        definition=tool_definition_from_dict(spec_data["definition"]),
        side_effect=ToolSideEffect(cast(str, spec_data["side_effect"])),
        idempotency=IdempotencyPolicy(cast(str, spec_data["idempotency"])),
        timeout=cast(float | None, spec_data.get("timeout")),
        resource_key=cast(str | None, spec_data.get("resource_key")),
        sandbox_profile=cast(str | None, spec_data.get("sandbox_profile")),
        required_permissions=cast(tuple[str, ...], permissions),
    )
    call = ToolCall(
        call_id=cast(str, call_data["call_id"]),
        name=cast(str, call_data["name"]),
        arguments=cast(FrozenJsonObject, call_data["arguments"]),
        tool_id=cast(str | None, call_data.get("tool_id")),
        tool_version=cast(str | None, call_data.get("tool_version")),
        idempotency_key=cast(str | None, call_data.get("idempotency_key")),
    )
    return spec, call


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
        )
        self._executions[stored.task_id] = execution
        return self._job_task(stored, spec, call)

    async def recover_job(
        self, stored: StoredJob, *, execution: ToolTaskExecution
    ) -> JobSnapshot:
        """Recover a validated Job only through a Runtime-provided execution path."""

        spec, call = _request_from_dict(stored.request)
        if spec.resource_key != stored.resource_key:
            raise RuntimeError("durable Job resource identity is incompatible")
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
        # Reliable admission point: persistence precedes background execution.
        await self.history.put_task(
            task_id=task_id,
            kind="tool_task",
            status=snapshot.state.value,
            request=_request_to_dict(spec, call),
        )
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
            if existing is not None and not existing.done():
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
            stored.job_id, status=JobState.RUNNING.value, attempt=attempt
        )
        try:
            if spec.timeout is None:
                output = await execution(
                    spec,
                    call,
                    ToolExecutionContext(task_id=stored.task_id, recovery=recovery),
                )
            else:
                async with asyncio.timeout(spec.timeout):
                    output = await execution(
                        spec,
                        call,
                        ToolExecutionContext(task_id=stored.task_id, recovery=recovery),
                    )
            result = ToolResult(
                call_id=call.call_id,
                name=call.name,
                status="succeeded",
                task=self._job_task(
                    stored, spec, call, state=ToolTaskState.SUCCEEDED
                ),
                output=freeze_json(output),
                side_effect_committed=True,
                tool_id=spec.tool_id,
                tool_version=spec.version,
            )
        except asyncio.CancelledError:
            uncertain = spec.side_effect in (
                ToolSideEffect.WRITE,
                ToolSideEffect.EXTERNAL,
            )
            state = ToolTaskState.UNKNOWN if uncertain else ToolTaskState.CANCELLED
            result = ToolResult(
                call_id=call.call_id,
                name=call.name,
                status="unknown" if uncertain else "cancelled",
                task=self._job_task(stored, spec, call, state=state),
                error=(
                    "tool cancellation could not confirm whether the side effect committed"
                    if uncertain
                    else None
                ),
                error_kind="cancellation_uncertain" if uncertain else "cancelled",
                side_effect_committed=None,
                tool_id=spec.tool_id,
                tool_version=spec.version,
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
        await self.history.update_tool_job(
            stored.job_id,
            status=state.value,
            result=tool_result_to_dict(result),
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
            if existing is not None and not existing.done():
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
            request=_request_to_dict(spec, call),
        )
        try:
            output = await _execute_with_timeout(
                self.registry,
                spec,
                call,
                execution=execution,
                context=ToolExecutionContext(task_id=task_id),
            )
            snapshot = self._snapshot(
                task_id, spec, call, ToolTaskState.SUCCEEDED
            )
            result = ToolResult(
                call_id=call.call_id,
                name=call.name,
                status="succeeded",
                task=snapshot,
                output=freeze_json(output),
                side_effect_committed=True,
                tool_id=spec.tool_id,
                tool_version=spec.version,
            )
        except asyncio.CancelledError:
            uncertain = spec.side_effect in (
                ToolSideEffect.WRITE,
                ToolSideEffect.EXTERNAL,
            )
            state = ToolTaskState.UNKNOWN if uncertain else ToolTaskState.CANCELLED
            snapshot = self._snapshot(task_id, spec, call, state)
            result = ToolResult(
                call_id=call.call_id,
                name=call.name,
                status="unknown" if uncertain else "cancelled",
                task=snapshot,
                error=(
                    "tool cancellation could not confirm whether the side effect committed"
                    if uncertain
                    else None
                ),
                error_kind="cancellation_uncertain" if uncertain else "cancelled",
                side_effect_committed=None,
                tool_id=spec.tool_id,
                tool_version=spec.version,
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
        await self.history.put_task(
            task_id=task_id,
            kind="tool_task",
            status=state.value,
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

    async def get_job(self, job_id: str) -> JobSnapshot | None:
        stored = await self.history.get_job(job_id)
        return None if stored is None else self._job_snapshot(stored)

    async def get_task(self, task_id: str) -> ToolTask | None:
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
            if task is not None and not task.done():
                task.cancel()
            else:
                task = None
        if task is None:
            job = await self.history.get_job_by_task(task_id)
            if job is None or job.status != JobState.PENDING.value:
                return False
            spec, call = _request_from_dict(job.request)
            result = ToolResult(
                call_id=call.call_id,
                name=call.name,
                status="cancelled",
                task=self._job_task(
                    job, spec, call, state=ToolTaskState.CANCELLED
                ),
                error_kind="cancelled",
                side_effect_committed=False,
                tool_id=spec.tool_id,
                tool_version=spec.version,
            )
            await self._store_job_terminal(job, result)
            return True
        await asyncio.gather(task, return_exceptions=True)
        job = await self.history.get_job_by_task(task_id)
        if job is not None and job.result is None:
            # A Task can be cancelled before its coroutine enters _run_job().
            # Persist a terminal result here so cancellation never leaves an
            # admitted Job permanently RUNNING.
            spec, call = _request_from_dict(job.request)
            uncertain = job.status == JobState.RUNNING.value and spec.side_effect in (
                ToolSideEffect.WRITE,
                ToolSideEffect.EXTERNAL,
            )
            state = ToolTaskState.UNKNOWN if uncertain else ToolTaskState.CANCELLED
            result = ToolResult(
                call_id=call.call_id,
                name=call.name,
                status="unknown" if uncertain else "cancelled",
                task=self._job_task(job, spec, call, state=state),
                error_kind="cancellation_uncertain" if uncertain else "cancelled",
                side_effect_committed=None if uncertain else False,
                tool_id=spec.tool_id,
                tool_version=spec.version,
            )
            await self._store_job_terminal(job, result)
        return True

    async def get_result(
        self, task_id: str, *, wait: bool = False
    ) -> ToolResult | None:
        async with self._lock:
            task = self._tasks.get(task_id)
        if wait and task is not None:
            await asyncio.gather(task, return_exceptions=True)
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
        await asyncio.gather(*tasks, return_exceptions=True)


__all__ = ["DurableToolTaskManager"]
