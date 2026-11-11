from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

from pygent.agent.pygent_v1.core.io import plain_object
from pygent.agent.pygent_v1.schema import CaseRunRef
from pygent.agent.react_agent import (
    ReActAgentContext,
)
from pygent.core import FrozenJsonObject, freeze_json_object
from pygent.runtime.codec import message_to_dict
from pygent.runtime.context_codec import ContextCodec, ContextCodecRegistry

from .file_effect_models import DeferredFileEffectJob


@dataclass(frozen=True, slots=True)
class PygentContext(ReActAgentContext):
    """Portable pygent_v1 agent state carried through one Pygent execution."""

    context_schema: ClassVar[str] = "pygent.v1.agent-context"
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

    def append_file_effects(self, *jobs: DeferredFileEffectJob) -> PygentContext:
        """Append deferred effects as strict portable JSON values."""

        encoded = tuple(freeze_json_object(job.to_dict()) for job in jobs)
        return replace(
            self, pending_file_effects=(*self.pending_file_effects, *encoded)
        )

    def drain_file_effects(
        self,
    ) -> tuple[tuple[DeferredFileEffectJob, ...], PygentContext]:
        """Return pending effects and a context with the queue cleared."""

        jobs = tuple(
            DeferredFileEffectJob.from_dict(plain_object(value))
            for value in self.pending_file_effects
        )
        return jobs, replace(self, pending_file_effects=())


PYGENT_CONTEXT_CODEC = ContextCodec.dataclass(PygentContext)
PYGENT_CONTEXT_CODECS = ContextCodecRegistry((PYGENT_CONTEXT_CODEC,))
