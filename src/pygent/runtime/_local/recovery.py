# mypy: disable-error-code="attr-defined"
"""Durable execution reconstruction and recovery claim handling."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Mapping
from typing import Any, TypeVar, cast

from pygent.core import (
    Context,
    Message,
    freeze_json_object,
    thaw_json,
)

from .._history_types import StoredExecution
from ..api import (
    ExecutionAdmissionError,
    ExecutionOptions,
    ExecutionPhase,
    ExecutionStatus,
)
from ..codec import invocation_from_dict
from .handles import _LocalBoundModule, _LocalExecutionHandle
from .policies import _finite_deadline_requirement
from .state import _ExecutionRecord

InputMessageT = TypeVar("InputMessageT", bound=Message)
OutputMessageT = TypeVar("OutputMessageT", bound=Message)


class _RecoveryMixin:
    async def recover(
        self,
        bound: _LocalBoundModule[InputMessageT, OutputMessageT],
        execution_id: str,
        *,
        deadline: float | None = None,
    ) -> _LocalExecutionHandle[OutputMessageT]:
        if bound.durability.recovery_level != "module_boundary_retry":
            raise RuntimeError("this Binding has no effective durable recovery")
        if self.history is None:
            raise RuntimeError("this LocalRuntime has no SQLiteHistoryStore")
        stored = await self.history.get_execution(execution_id)
        if stored is None:
            raise KeyError(f"unknown durable execution {execution_id!r}")
        return await self._recover_stored(bound, stored, deadline=deadline)

    async def _recover_stored(
        self,
        bound: _LocalBoundModule[InputMessageT, OutputMessageT],
        stored: StoredExecution,
        *,
        deadline: float | None,
    ) -> _LocalExecutionHandle[OutputMessageT]:
        history = self.history
        if history is None:
            raise RuntimeError("this LocalRuntime has no SQLiteHistoryStore")
        active = self._executions.get(stored.execution_id)
        if active is not None:
            return cast(_LocalExecutionHandle[OutputMessageT], _LocalExecutionHandle(active))
        if stored.plan_id != bound.plan.plan_id:
            raise RuntimeError("durable execution ExecutionPlan is incompatible")
        raw_model_calls = thaw_json(stored.model_calls) if stored.model_calls is not None else {}
        if not isinstance(raw_model_calls, Mapping):
            raise TypeError("durable model call options are invalid")
        model_calls = freeze_json_object(raw_model_calls)
        has_deferred_models = any(
            module.model_requirements for module in bound.plan.modules
        )
        model_admission = None
        if stored.model_admission_status == "preparing":
            await self._ensure_model_store_open()
            model_admission = await self.model_deployment_store.get_admission(
                stored.model_admission_id or stored.execution_id
            )
            if model_admission is None:
                await history.abort_model_admission(stored.execution_id)
                raise ExecutionAdmissionError(
                    "durable model admission intent has no exact manifest"
                )
            await history.commit_model_admission(
                stored.execution_id,
                admission_id=model_admission.admission_id,
                manifest_digest=model_admission.digest,
            )
        elif stored.model_admission_status == "committed":
            if stored.model_admission_id is None or stored.model_admission_digest is None:
                raise RuntimeError("durable model admission metadata is incomplete")
            await self._ensure_model_store_open()
            model_admission = await self.model_deployment_store.get_admission(
                stored.model_admission_id
            )
            if (
                model_admission is None
                or model_admission.digest != stored.model_admission_digest
            ):
                raise ExecutionAdmissionError(
                    "durable model admission exact manifest is unavailable"
                )
        elif stored.model_admission_status == "aborted":
            raise ExecutionAdmissionError("durable model admission was aborted")
        elif has_deferred_models:
            raise ExecutionAdmissionError(
                "durable execution has no exact model admission manifest"
            )
        if ExecutionStatus(stored.status).terminal:
            raise ExecutionAdmissionError(
                "terminal execution cannot be recovered; attach to its existing outcome"
            )
        deadline_requirement = _finite_deadline_requirement(bound.module)
        if deadline_requirement is not None and deadline is None:
            raise ExecutionAdmissionError(
                f"bound Module graph contains {type(deadline_requirement).__name__}, "
                "which requires a finite recovery deadline"
            )
        message, context = invocation_from_dict(
            stored.input, registry=self.context_codec_registry
        )
        last_event_index = await history.last_event_index(
            execution_id=stored.execution_id
        )
        record = _ExecutionRecord(
            execution_id=stored.execution_id,
            trace_id=stored.trace_id,
            root_span_id=str(uuid.uuid4()),
            request_id=stored.request_id,
            binding_state=self._state_for(bound.binding),
            plan=bound.plan,
            graph=bound.graph,
            deadline=deadline,
            history=history,
            history_started=True,
            attempt=stored.attempt + 1,
            idempotency_key=stored.idempotency_key,
            model_calls=model_calls,
            model_admission=model_admission,
            event_base_sequence=last_event_index + 1,
            next_sequence=last_event_index + 1,
            committed_sequence=last_event_index,
        )
        record.owner_id = f"{self._recovery_owner_id}:{record.attempt_id}"
        fencing_token = await history.claim_execution(
            execution_id=stored.execution_id,
            owner_id=record.owner_id,
            lease_ttl=self._recovery_lease_ttl,
        )
        if fencing_token is None:
            raise ExecutionAdmissionError("durable execution is owned by another recovery attempt")
        record.fencing_token = fencing_token
        try:
            await history.update_execution(
                stored.execution_id,
                status=ExecutionStatus.PENDING.value,
                attempt=stored.attempt + 1,
                attempt_id=record.attempt_id,
                phase=ExecutionPhase.PREPARING.value,
            )
        except BaseException:
            await history.release_execution_claim(
                execution_id=stored.execution_id,
                owner_id=record.owner_id,
                fencing_token=fencing_token,
            )
            raise
        record.task = asyncio.create_task(
            self._run_recovered_with_claim(
                record, bound, message, context, fencing_token
            ),
            name=f"pygent-recovered-{stored.execution_id}",
        )
        record.task.add_done_callback(self._execution_finished)
        self._executions[stored.execution_id] = record
        return cast(_LocalExecutionHandle[OutputMessageT], _LocalExecutionHandle(record))

    async def _run_recovered_with_claim(
        self,
        record: _ExecutionRecord,
        bound: _LocalBoundModule[Any, Any],
        message: Message,
        context: Context,
        fencing_token: int,
    ) -> tuple[Message, Context]:
        history = record.history
        assert history is not None
        return await self._run_root(
            record,
            bound,
            ExecutionOptions(deadline=record.deadline, model_calls=record.model_calls),
            message,
            context,
            {},
            bool(record.model_admission),
            prepared=True,
        )

__all__ = ["_RecoveryMixin"]
