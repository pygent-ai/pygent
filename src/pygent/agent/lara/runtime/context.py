from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

from pygent import (
    ContextCodec,
    FrozenJsonObject,
    PygentAgentContext,
    freeze_json_object,
)
from pygent.agent.lara.core.io import plain_object
from pygent.agent.lara.schema import CaseRunRef
from pygent.runtime.codec import message_to_dict
from pygent.runtime.context_codec import ContextCodecRegistry

from .file_effect_models import DeferredFileEffectJob


@dataclass(frozen=True, slots=True)
class LaraContext(PygentAgentContext):
    """Portable Lara agent state carried through one Pygent execution."""

    context_schema: ClassVar[str] = "lara.agent-context"
    context_schema_version: ClassVar[int] = 6

    session_id: str = ""
    case_id: str = ""
    case_run_id: str = ""
    run_dir: str = ""
    turn_id: str | None = None
    eternal_memory_enabled: bool = False
    memory_covered_through: int = 0
    memory_projection: FrozenJsonObject = field(
        default_factory=lambda: freeze_json_object({})
    )
    raw_history_location: str = ""
    pending_file_effects: tuple[FrozenJsonObject, ...] = ()

    @property
    def history(self) -> list[dict[str, Any]]:
        """Return this invocation's native committed messages in storage shape."""

        return [message_to_dict(message) for message in self.committed_messages]

    @property
    def case_run_ref(self) -> CaseRunRef:
        """Rebuild the domain reference from portable execution facts."""

        return CaseRunRef(
            session_id=self.session_id,
            case_id=self.case_id,
            case_run_id=self.case_run_id,
            run_dir=self.run_dir,
        )

    def append_file_effects(self, *jobs: DeferredFileEffectJob) -> LaraContext:
        """Append deferred effects as strict portable JSON values."""

        encoded = tuple(freeze_json_object(job.to_dict()) for job in jobs)
        return replace(
            self, pending_file_effects=(*self.pending_file_effects, *encoded)
        )

    def drain_file_effects(
        self,
    ) -> tuple[tuple[DeferredFileEffectJob, ...], LaraContext]:
        """Return pending effects and a context with the queue cleared."""

        jobs = tuple(
            DeferredFileEffectJob.from_dict(plain_object(value))
            for value in self.pending_file_effects
        )
        return jobs, replace(self, pending_file_effects=())


LARA_CONTEXT_CODEC = ContextCodec.dataclass(LaraContext)
LARA_CONTEXT_CODECS = ContextCodecRegistry((LARA_CONTEXT_CODEC,))
