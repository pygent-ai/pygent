"""Transactional ownership of resources acquired before root execution starts."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from .state import _ExecutionRecord


@dataclass(slots=True)
class AdmissionCoordinator:
    """Release partially or fully admitted resources in strict reverse order."""

    runtime: Any
    record: _ExecutionRecord
    has_deferred_models: bool
    live_acquired: bool = False
    runnable_acquired: bool = False
    model_manifest_committed: bool = False

    def mark_live(self) -> None:
        self.live_acquired = True

    def mark_runnable(self) -> None:
        self.runnable_acquired = True

    def mark_model_manifest_committed(self) -> None:
        self.model_manifest_committed = True

    async def release(self) -> None:
        if self.runnable_acquired or self.record.runnable_held:
            self.runnable_acquired = False
            self.record.runnable_held = False
            self.record.binding_state.runnable.release()
        if self.live_acquired:
            self.live_acquired = False
            await self.record.binding_state.release_live()
        if self.record.model_admission is not None:
            admission, self.record.model_admission = self.record.model_admission, None
            recoverable = self.model_manifest_committed
            if self.record.history is not None and not recoverable:
                stored = await self.record.history.get_execution(self.record.execution_id)
                recoverable = bool(
                    stored is not None
                    and stored.model_admission_status == "committed"
                    and stored.model_admission_id == admission.admission_id
                    and stored.model_admission_digest == admission.digest
                )
            await self.runtime.model_deployment_store.release_admission(
                admission.admission_id,
                recoverable=recoverable,
            )
        elif self.has_deferred_models and self.runtime._model_store_opened:
            # A store may commit immediately before cancellation is delivered to
            # its caller.  The pre-reserved identity makes that edge reversible.
            await asyncio.shield(
                self.runtime.model_deployment_store.release_admission(
                    self.record.execution_id,
                    recoverable=False,
                )
            )


__all__ = ["AdmissionCoordinator"]
