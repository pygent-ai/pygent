from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from pygent.agent.pygent_v1.core.io import jsonl_path_lock, read_json, write_json_atomic
from pygent.agent.pygent_v1.schema import CaseRunRef, RunConfig
from pygent.agent.pygent_v1.tracing import EventStore
from pygent.core import independent_execution
from pygent.core.json_values import JsonValue
from pygent.runtime import LocalRuntime
from pygent.tool import ToolResult

from .context import PygentContext
from .file_effect_models import DeferredFileEffectBatch, DeferredFileEffectJob
from .file_effects import FileEffectBaselineStore, process_file_effect_batch
from .tools import FileEffectTracker, SnapshotBudgetExceeded, ToolObserver

_TOOL_TASK_EVENT_TYPES = {
    "tool.task.started": "runtime.tool_task.started",
    "tool.task.completed": "runtime.tool_task.completed",
    "tool.task.failed": "runtime.tool_task.failed",
    "tool.task.cancelled": "runtime.tool_task.cancelled",
    "tool.task.unknown": "runtime.tool_task.unknown",
}


class BashTaskObservations:
    """Persist Lara audit work; Pygent alone owns command execution and results."""

    def __init__(self, config: RunConfig, runtime: LocalRuntime, directory: Path) -> None:
        self.config = config
        self.runtime = runtime
        self.directory = directory
        self._observers: dict[Path, asyncio.Task[None]] = {}
        self._closed = False

    async def prepare(self, context: PygentContext) -> None:
        await self._finish_thread(self._prepare, context)

    def _prepare(self, context: PygentContext) -> None:
        store = EventStore(context.case_run_ref)
        baseline = FileEffectBaselineStore(store.session_dir)
        if baseline.path is None:
            return
        with jsonl_path_lock(baseline.path):
            if baseline.path.exists() and baseline.is_complete():
                return
            tracker = FileEffectTracker(workspace_root=self.config.workspace_root, store=store)
            try:
                baseline.save(tracker.snapshot_workspace())
            except SnapshotBudgetExceeded as exc:
                store.append("runtime.file_scan.incomplete", actor="system",
                             payload={"reason": str(exc)}, turn_id=context.turn_id)

    def track(self, context: PygentContext, result: ToolResult,
              payload: dict[str, Any], arguments: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("Bash task observations are closed")
        if result.task is None:
            raise ValueError("detached Bash result has no task reference")
        identity = hashlib.sha256(result.task.task_id.encode()).hexdigest()
        path = self.directory / f"{identity}.json"
        if not path.exists():
            write_json_atomic(path, {
                "task_id": result.task.task_id,
                "case_run_ref": context.case_run_ref.to_dict(),
                "turn_id": context.turn_id,
                "audit_call_id": payload["tool_call_id"],
                "arguments": arguments,
            })
        self._start(path)

    async def restore(self) -> None:
        if self.directory.exists():
            for path in self.directory.glob("*.json"):
                self._start(path)

    async def on_tool_task_event(self, kind: str, data: Mapping[str, Any]) -> None:
        """Trace pygent ToolTask lifecycle events into the owning case run.

        Serves as the DurableToolTaskManager emit sink. A fresh start emits
        ``started`` before the observation record exists, so only recovery
        relaunches (flagged by pygent) record a started event; terminal events
        always attribute through the persisted task mapping. Unattributable
        tasks are skipped — the guarded sink in pygent ignores sink failures,
        but tracing must not depend on that.
        """
        if self._closed:
            return
        event_type = _TOOL_TASK_EVENT_TYPES.get(kind)
        task_id = str(data.get("task_id", ""))
        if event_type is None or not task_id:
            return
        if kind == "tool.task.started" and not data.get("recovery"):
            return
        record = self._observation_record(task_id)
        if record is None:
            return
        store = EventStore(CaseRunRef.from_dict(record["case_run_ref"]))
        store.append(
            event_type,
            actor="system",
            payload=dict(data),
            turn_id=record.get("turn_id"),
        )

    def _observation_record(self, task_id: str) -> dict[str, Any] | None:
        identity = hashlib.sha256(task_id.encode()).hexdigest()
        path = self.directory / f"{identity}.json"
        if not path.exists():
            return None
        return read_json(path)

    def _start(self, path: Path) -> None:
        existing = self._observers.get(path)
        if existing is not None and not existing.done():
            return
        with independent_execution():
            observer = asyncio.create_task(
                self._observe(path), name=f"pygent.v1-bash-observation:{path.stem}",
            )
        self._observers[path] = observer
        observer.add_done_callback(lambda done: self._discard(path, done))

    def _discard(self, path: Path, observer: asyncio.Task[None]) -> None:
        if self._observers.get(path) is observer:
            self._observers.pop(path, None)

    async def _observe(self, path: Path) -> None:
        record = read_json(path)
        try:
            while not self._closed:
                final = await self._result(record["task_id"], wait=True)
                if final is not None:
                    await self._finish_thread(self._finalize, path, record, final)
                    return
                await asyncio.sleep(0.1)
        except Exception as exc:  # noqa: BLE001 - the failure is journaled as an event
            EventStore(CaseRunRef.from_dict(record["case_run_ref"])).append_error(
                exc, event_type="runtime.error", turn_id=record["turn_id"],
                payload={"task_id": record["task_id"], "phase": "bash-finalization"},
            )
            # Keep the durable observation for retry on the next runtime startup.

    def _finalize(self, path: Path, record: dict[str, Any], result: ToolResult) -> None:
        run = CaseRunRef.from_dict(record["case_run_ref"])
        store = EventStore(run)
        observer = ToolObserver(
            store, workspace_root=self.config.workspace_root, track_file_effects=True,
            defer_file_effects=True,
            allow_read_outside_workspace=self.config.allow_read_outside_workspace,
            bash_full_output_allowlist=self.config.bash_full_output_allowlist,
        )
        # Reuse the persisted audit call on restart. The command is only observed,
        # never run; retain the observation until file-effect processing completes.
        existing = next((event for event in store.list_events()
                         if event.type == "tool.result"
                         and event.payload.get("tool_call_id") == record["audit_call_id"]
                         and event.payload.get("status") != "running"), None)
        if existing is None:
            _, job = observer.record_framework_result(
                "bash", record["arguments"], record["turn_id"], result,
                audit_call_id=record["audit_call_id"],
            )
        else:
            job = DeferredFileEffectJob(
                tool_call_id=record["audit_call_id"], tool_name="bash",
                args=record["arguments"], turn_id=record["turn_id"], declared=[],
                include_declared=False,
            )
        if job is not None:
            process_file_effect_batch(DeferredFileEffectBatch.create(
                case_run_ref=run, workspace_root=self.config.workspace_root, jobs=[job],
            ))
        path.unlink(missing_ok=True)

    async def _result(self, task_id: str, *, wait: bool = False) -> ToolResult | None:
        result = await self.runtime.get_tool_result(task_id, wait=wait)
        if result is not None and result.output is None:
            retrieved = await self.runtime.get_tool_output(task_id)
            result = replace(result, output=cast("JsonValue", retrieved))
        return result

    @staticmethod
    async def _finish_thread(function, *args):
        """A cancelled observer must join a file writer before history can close."""
        worker = asyncio.create_task(asyncio.to_thread(function, *args))
        cancelled = False
        while True:
            try:
                result = await asyncio.shield(worker)
                break
            except asyncio.CancelledError:
                if worker.cancelled():
                    raise
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError
        return result

    async def close(self) -> None:
        self._closed = True
        observers = tuple(self._observers.values())
        for observer in observers:
            observer.cancel()
        await asyncio.gather(*observers, return_exceptions=True)
        # Runtime has already cancelled/drained commands; consume their saved finals.
        if self.directory.exists():
            for path in self.directory.glob("*.json"):
                record = read_json(path)
                final = await self._result(record["task_id"])
                if final is not None:
                    await self._finish_thread(self._finalize, path, record, final)
        self._observers.clear()
