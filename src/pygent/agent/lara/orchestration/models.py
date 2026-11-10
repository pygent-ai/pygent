from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from pygent.agent.lara.schema import CaseRunStatus


@dataclass(frozen=True, slots=True)
class TurnCommand:
    session_id: str
    message: str
    case_id: str = "chat"
    turn_id: str | None = None
    submission_id: str | None = None
    interactive_approvals: bool = True
    message_kind: str = "lara.chat.turn"
    message_data: dict[str, object] = field(default_factory=dict)
    session_title: str | None = None
    attachments: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must be non-empty")
        if not self.case_id:
            raise ValueError("case_id must be non-empty")
        if not self.message:
            raise ValueError("message must be non-empty")


class TurnState(StrEnum):
    QUEUED = "queued"
    STARTING = "starting"
    RUNNING = "running"
    FINALIZING = "finalizing"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"

    @property
    def terminal(self) -> bool:
        # Terminal turn states are exactly the durable case-run verdicts; the
        # lifecycle-only states must never read as terminal.
        verdict = CaseRunStatus.from_value(self.value)
        return verdict.terminal if verdict is not None else False


__all__ = ["TurnCommand", "TurnState"]
