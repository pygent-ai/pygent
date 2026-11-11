from __future__ import annotations

import asyncio
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pygent.agent.pygent_v1.core.io import (
    jsonl_path_lock,
    plain_object,
    read_json,
    utc_now,
    write_json_atomic,
)
from pygent.agent.pygent_v1.tracing.events import EventStore
from pygent.tool import IdempotencyPolicy, ToolDefinition, ToolSideEffect, ToolSpec
from pygent.tool.executors import SandboxExecutorSupport, ToolExecutionContext

from .file_effect_models import (
    DeferredFileEffectBatch,
    DeferredFileEffectJob,
    FileSnapshot,
)
from .tools import FileEffectTracker, SnapshotBudgetExceeded


class FileEffectBaselineStore:
    def __init__(self, session_dir: str | Path | None):
        self.session_dir = Path(session_dir).expanduser().resolve() if session_dir is not None else None
        self.path = self.session_dir / "state" / "file_effects_baseline.json" if self.session_dir is not None else None

    def load(self) -> dict[str, Any] | None:
        if self.path is None or not self.path.exists():
            return None
        data = read_json(self.path, default={"snapshots": {}})
        raw_snapshots = data.get("snapshots")
        if not isinstance(raw_snapshots, dict):
            return None
        return {
            str(path): FileSnapshot(**snapshot)
            for path, snapshot in raw_snapshots.items()
            if isinstance(snapshot, dict)
        }

    def is_complete(self) -> bool:
        if self.path is None:
            return False
        return bool(read_json(self.path, default={}).get("complete", True))

    def save(self, snapshots: dict[str, Any], *, complete: bool = True) -> None:
        if self.path is None:
            return
        write_json_atomic(
            self.path,
            {
                "updated_at": utc_now(),
                "complete": complete,
                "snapshots": {path: asdict(snapshot) for path, snapshot in snapshots.items()},
            },
        )


FILE_EFFECT_TOOL_SPEC = ToolSpec(
    tool_id="pygent.v1.internal.persist_file_effects",
    version="1",
    definition=ToolDefinition(
        name="pygent_persist_file_effects",
        description="Persist an internal batch of observed workspace file effects.",
        parameters={
            "type": "object",
            "properties": {"batch": {"type": "object"}},
            "required": ["batch"],
            "additionalProperties": False,
        },
    ),
    side_effect=ToolSideEffect.WRITE,
    idempotency=IdempotencyPolicy.REQUIRES_KEY,
    resource_key="workspace",
    sandbox_profile="workspace-write",
)


class FileEffectToolExecutor:
    sandbox_support = SandboxExecutorSupport(
        profiles=("workspace-write",),
        durable_reconnect=True,
        deployment_fingerprint="pygent.v1:file-effects:v1",
    )

    async def execute(
        self, spec: ToolSpec, call: Any, context: ToolExecutionContext
    ) -> object:
        del spec, context
        arguments = plain_object(call.arguments)
        batch = DeferredFileEffectBatch.from_dict(plain_object(arguments.get("batch")))
        await asyncio.to_thread(process_file_effect_batch, batch)
        return {"batch_id": batch.batch_id, "status": "completed"}


def process_file_effect_batch(batch: DeferredFileEffectBatch) -> None:
    """Execute one idempotently admitted diff batch; scheduling belongs to Pygent."""

    store = EventStore(batch.case_run_ref)
    path = FileEffectBaselineStore(store.session_dir).path or store.run_dir / "file-effects.lock"
    with jsonl_path_lock(path):
        _process_file_effect_batch(batch)


def _process_file_effect_batch(batch: DeferredFileEffectBatch) -> None:

    session_dir = EventStore(batch.case_run_ref).session_dir
    baseline_store = FileEffectBaselineStore(session_dir)
    tracker = FileEffectTracker(workspace_root=batch.workspace_root, store=EventStore(batch.case_run_ref))
    declared = [effect for job in batch.jobs if job.include_declared for effect in job.declared]
    if not any(job.requires_snapshot for job in batch.jobs):
        tracker.append_effects(declared, turn_id=batch.turn_id)
        return
    baseline = baseline_store.load()
    # Known write/edit targets do not require a whole-workspace scan.
    targeted = all(job.tool_name in {"write", "edit"} for job in batch.jobs if job.requires_snapshot)
    targets = {effect.path for job in batch.jobs for effect in job.declared} if targeted else None
    if targeted and not targets:
        targeted = False
        targets = None
    try:
        current = tracker.snapshot_workspace(
            paths=[Path(path) for path in targets] if targets is not None else None
        )
    except SnapshotBudgetExceeded as exc:
        tracker.append_effects(declared, turn_id=batch.turn_id)
        tracker.store.append(
            "runtime.file_scan.incomplete", actor="system",
            payload={"batch_id": batch.batch_id, "reason": str(exc)},
            turn_id=batch.turn_id,
        )
        return  # Never persist a partial scan or infer deletions from it.
    stored_baseline = baseline or {}
    was_complete = baseline is not None and baseline_store.is_complete()
    if not targeted and not was_complete:
        baseline = None  # A target-only baseline cannot describe the whole workspace.
    if baseline is not None:
        baseline = {
            path: snapshot for path, snapshot in baseline.items()
            if (path in targets if targets is not None else not tracker._is_ignored(Path(path)))
        }
    if baseline is None:
        tracker.append_effects(declared, turn_id=batch.turn_id)
    else:
        snapshot_jobs = [job for job in batch.jobs if job.requires_snapshot]
        owner = snapshot_jobs[0] if len(snapshot_jobs) == 1 else None
        observed = tracker.observed_effects(
            baseline,
            current,
            tool_name=owner.tool_name if owner is not None else "tool_batch",
            tool_call_id=owner.tool_call_id if owner is not None else batch.batch_id,
        )
        tracker.append_effects(tracker.merge_effects(declared, observed), turn_id=batch.turn_id)
    if targeted:
        # Preserve unrelated entries for later full scans.
        merged = {path: snapshot for path, snapshot in stored_baseline.items() if targets is not None and path not in targets}
        merged.update(current)
        baseline_store.save(merged, complete=was_complete)
    else:
        baseline_store.save(current)
__all__ = [
    "FILE_EFFECT_TOOL_SPEC",
    "DeferredFileEffectBatch",
    "DeferredFileEffectJob",
    "FileEffectBaselineStore",
    "FileEffectToolExecutor",
    "FileSnapshot",
    "process_file_effect_batch",
]
