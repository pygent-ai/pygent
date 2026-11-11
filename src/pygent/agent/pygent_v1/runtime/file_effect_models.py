from __future__ import annotations

import hashlib
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from pygent.agent.pygent_v1.schema import CaseRunRef


@dataclass(slots=True)
class FileSnapshot:
    path: str
    exists: bool
    kind: Literal["file", "dir", "other", "missing"]
    size: int | None = None
    mtime_ns: int | None = None
    content_hash: str | None = None
    content: str | None = None
    content_available: bool = False
    content_unavailable_reason: str | None = None


@dataclass(slots=True)
class FileEffect:
    type: Literal["file.read", "file.write", "file.edit", "file.delete"]
    path: str
    tool_call_id: str
    tool_name: str
    detected_by: list[Literal["tool_args", "snapshot_diff", "bash_command_parse"]]
    confidence: Literal["declared", "observed", "inferred"]
    before_hash: str | None = None
    after_hash: str | None = None
    before_exists: bool | None = None
    after_exists: bool | None = None
    before_content: str | None = None
    after_content: str | None = None
    before_content_available: bool = False
    after_content_available: bool = False
    before_content_unavailable_reason: str | None = None
    after_content_unavailable_reason: str | None = None

    def key(self) -> tuple[Any, ...]:
        return (
            self.tool_call_id, self.path, self.type, self.before_hash,
            self.after_hash, self.before_exists, self.after_exists,
        )


@dataclass(slots=True)
class DeferredFileEffectJob:
    tool_call_id: str
    tool_name: str
    args: dict[str, Any]
    turn_id: str | None
    declared: list[FileEffect]
    include_declared: bool = True
    requires_snapshot: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "args": self.args,
            "turn_id": self.turn_id,
            "declared": [asdict(effect) for effect in self.declared],
            "include_declared": self.include_declared,
            "requires_snapshot": self.requires_snapshot,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeferredFileEffectJob:
        return cls(
            tool_call_id=str(data["tool_call_id"]),
            tool_name=str(data["tool_name"]),
            args=dict(data.get("args") or {}),
            turn_id=data.get("turn_id"),
            declared=[FileEffect(**effect) for effect in data.get("declared", [])],
            include_declared=bool(data.get("include_declared", True)),
            requires_snapshot=bool(data.get("requires_snapshot", True)),
        )


@dataclass(slots=True)
class DeferredFileEffectBatch:
    batch_id: str
    case_run_ref: CaseRunRef
    workspace_root: str
    jobs: list[DeferredFileEffectJob]
    turn_id: str | None

    @classmethod
    def create(
        cls, *, case_run_ref: CaseRunRef, workspace_root: str | Path,
        jobs: list[DeferredFileEffectJob],
    ) -> DeferredFileEffectBatch:
        turn_id = jobs[0].turn_id if jobs else None
        identity = "\0".join(job.tool_call_id for job in jobs) or uuid.uuid4().hex
        return cls(
            batch_id=f"filefx-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:32]}",
            case_run_ref=case_run_ref,
            workspace_root=str(Path(workspace_root).expanduser().resolve()),
            jobs=list(jobs),
            turn_id=turn_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "case_run_ref": self.case_run_ref.to_dict(),
            "workspace_root": self.workspace_root,
            "jobs": [job.to_dict() for job in self.jobs],
            "turn_id": self.turn_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeferredFileEffectBatch:
        return cls(
            batch_id=str(data["batch_id"]),
            case_run_ref=CaseRunRef.from_dict(dict(data["case_run_ref"])),
            workspace_root=str(data["workspace_root"]),
            jobs=[DeferredFileEffectJob.from_dict(job) for job in data.get("jobs", [])],
            turn_id=data.get("turn_id"),
        )


__all__ = ["DeferredFileEffectBatch", "DeferredFileEffectJob", "FileEffect", "FileSnapshot"]
