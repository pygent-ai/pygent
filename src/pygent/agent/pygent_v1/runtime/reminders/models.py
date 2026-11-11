from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pygent.agent.pygent_v1.schema import BashCliPreset


class BootstrapStatus(StrEnum):
    PENDING = "pending"
    PREPARING = "preparing"
    READY = "ready"
    CLAIMED = "claimed"
    CONSUMED = "consumed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ReminderScope:
    session_id: str
    session_dir: Path
    workspace_root: Path
    cli_bash_presets: tuple[BashCliPreset, ...]
    user_skills_dir: Path
    project_skills_dir: Path


@dataclass(frozen=True, slots=True)
class ReminderSection:
    source: str
    order: int
    lines: tuple[str, ...]
    include_time: bool = False


@dataclass(frozen=True, slots=True)
class InitialSnapshot:
    snapshot_id: str
    session_id: str
    prepared_at: str
    content: str
