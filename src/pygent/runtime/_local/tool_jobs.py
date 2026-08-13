# mypy: disable-error-code="attr-defined"
"""Runtime-facing ToolTask and durable Tool Job operations."""

from __future__ import annotations

from typing import Any, cast

from pygent.tool import ToolExecutionError, ToolResult, ToolTask, ToolTaskManager
from pygent.tool.executors import validate_executor_sandbox

from .._history_types import StoredJob
from ..api import ExecutionAdmissionError, JobSnapshot
from .handles import _LocalBoundModule
from .policies import _apply_binding_policy


class _ToolJobsMixin:
    def _task_manager(self) -> ToolTaskManager:
        if self._tool_tasks is None:
            raise RuntimeError("this Runtime has no ToolTaskManager capability")
        return self._tool_tasks

    async def get_tool_task(self, task_id: str) -> ToolTask | None:
        return await self._task_manager().get_task(task_id)

    async def get_job(self, job_id: str) -> JobSnapshot | None:
        manager = self._task_manager()
        get_job = getattr(manager, "get_job", None)
        if not callable(get_job):
            raise ExecutionAdmissionError("this Runtime has no durable Job capability")
        return cast(JobSnapshot | None, await get_job(job_id))

    async def recover_tool_jobs(
        self, bound: _LocalBoundModule[Any, Any]
    ) -> tuple[JobSnapshot, ...]:
        """Recover durable Tool Jobs through a validated Binding and Tool plane."""

        history = self.history
        manager = self._task_manager()
        recover_job = getattr(manager, "recover_job", None)
        registry = self._tool_registry
        if (
            history is None
            or "durability.sqlite" not in bound.durability.effective_capabilities
            or not callable(recover_job)
            or registry is None
            or getattr(manager, "history", None) is not history
        ):
            raise RuntimeError(
                "durable Tool Job recovery requires matching SQLite history, "
                "Job manager, Binding, and Runtime executor registry"
            )
        assert history is not None
        current_plan = _apply_binding_policy(
            self._compile_plan(
                bound.module,
                context_codec_identities=bound.plan.context_codecs,
            ),
            bound.binding,
        )
        if current_plan.graph_hash != bound.plan.graph_hash:
            raise ExecutionAdmissionError("durable Job ExecutionPlan changed after binding")
        stored_jobs = await history.list_jobs(
            statuses=("pending", "running"), binding_id=bound.binding.name
        )
        binding_state = self._state_for(bound.binding)
        recovered: list[JobSnapshot] = []
        for stored in stored_jobs:
            self._validate_job_recovery(stored, bound)
            execute = self._detached_tool_execution(binding_state, registry)
            recovered.append(await recover_job(stored, execution=execute))
        return tuple(recovered)

    def _validate_job_recovery(
        self, stored: StoredJob, bound: _LocalBoundModule[Any, Any]
    ) -> None:
        if stored.binding_id != bound.binding.name:
            raise ExecutionAdmissionError("durable Job Binding identity is incompatible")
        if stored.plan_id != bound.plan.plan_id:
            raise ExecutionAdmissionError("durable Job ExecutionPlan is incompatible")
        missing = set(stored.required_capabilities) - self.capabilities
        if missing:
            raise ExecutionAdmissionError(
                "durable Job is missing required capabilities: "
                + ", ".join(sorted(missing))
            )
        from ..tasks import _request_from_dict

        spec, _ = _request_from_dict(stored.request)
        if spec.resource_key != stored.resource_key:
            raise ExecutionAdmissionError("durable Job resource identity is incompatible")
        registry = self._tool_registry
        resolve = getattr(registry, "resolve", None)
        if callable(resolve):
            try:
                executor = resolve(spec.tool_id, spec.version)
                validate_executor_sandbox(
                    spec,
                    executor,
                    durable=spec.sandbox_profile is not None,
                    required_capabilities=stored.required_capabilities,
                )
            except LookupError as exc:
                raise ExecutionAdmissionError(
                    "durable Job tool version is not registered"
                ) from exc
            except ToolExecutionError as exc:
                raise ExecutionAdmissionError(str(exc)) from exc

    async def cancel_tool_task(self, task_id: str) -> bool:
        return await self._task_manager().cancel(task_id)

    async def get_tool_result(
        self, task_id: str, *, wait: bool = False
    ) -> ToolResult | None:
        return await self._task_manager().get_result(task_id, wait=wait)


__all__ = ["_ToolJobsMixin"]
