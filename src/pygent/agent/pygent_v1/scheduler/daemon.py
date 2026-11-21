"""Scheduler daemon: verify submissions, promote tasks, and execute runs.

The daemon is the only process allowed to write ``tasks`` and ``runs``. It
polls every ``scheduler.poll_seconds``, claims due tasks with an expiring
lease, and invokes the Lara CLI in a child process with the inherited
environment (no scheduler-specific variables are injected).

Promoted ``tasks`` are the unified trigger bus. The bus carries timer
triggers, event triggers, and agent-registered intents; a user's direct chat
message never enters it and keeps using the unchanged foreground path. User
and agent submissions are HMAC-verified, while ``source=timer``/``source=event``
submissions are generated inside the daemon and are accepted unsigned. Agent
intents always deliver fire-and-forget, and an agent request for ``high``
priority is normalized to ``normal`` with an audit note on the submission.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pygent.agent.pygent_v1.core.io import append_jsonl
from pygent.agent.pygent_v1.schema import RunConfig

from .keys import load_secret, verify
from .message import build_task_message
from .schedule import advance, next_run, parse_schedule
from .store import (
    DEFAULT_DELIVERY,
    DEFAULT_PRIORITY,
    INTERNAL_SOURCES,
    TASK_ORIGINS,
    SchedulerStore,
    Submission,
    Task,
    iso,
    prompt_hash,
    scheduler_db_path,
    submission_payload,
    utc_now,
)


@dataclass(frozen=True, slots=True)
class TaskResult:
    exit_code: int | None
    output: str = ""
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class TaskRunner(Protocol):
    def run(self, task: Task) -> TaskResult: ...


def lara_command() -> tuple[str, ...]:
    """Launch Lara through the module entry point.

    A console script may still point at the pre-split ``lara.cli`` module after
    the kernel/shell split; ``python -m lara`` always resolves the package
    layout that owns the scheduler's session command.
    """

    return (sys.executable, "-m", "pygent_v1")


class SubprocessTaskRunner:
    """Run one task through the Lara CLI in a child process."""

    def __init__(
        self,
        *,
        command: tuple[str, ...] | None = None,
        timeout: float = 3600.0,
        user_identity: str = "default",
    ) -> None:
        self.command = command or lara_command()
        self.timeout = timeout
        self.user_identity = user_identity

    def run(self, task: Task) -> TaskResult:
        started = utc_now()
        message = build_task_message(
            task_id=task.id,
            name=task.name,
            source=task.source,
            instructions=task.instructions,
            user_message=task.prompt,
            user_identity=self.user_identity,
        )
        command = (*self.command, "session", "run", task.session, "-m", message)
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                check=False,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                env=None,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return TaskResult(
                exit_code=None,
                output="",
                error=str(exc),
                started_at=started,
                finished_at=utc_now(),
            )
        output = (completed.stdout or "") + (completed.stderr or "")
        return TaskResult(
            exit_code=completed.returncode,
            output=output[-20_000:],
            error=None if completed.returncode == 0 else output[-20_000:],
            started_at=started,
            finished_at=utc_now(),
        )


class SchedulerDaemon:
    """Single-agent scheduler process with durable submissions and runs."""

    def __init__(
        self,
        config: RunConfig,
        *,
        store: SchedulerStore | None = None,
        runner: TaskRunner | None = None,
        clock: Callable[[], datetime] | None = None,
        sleeper: Callable[[float], None] | None = None,
        alert: Callable[[str], None] | None = None,
        session_id: str | None = None,
        lease_seconds: float | None = None,
    ) -> None:
        self.config = config
        self.store = store or SchedulerStore(scheduler_db_path(config.pygent_root))
        self.user_pygent_root = config.user_pygent_root
        self.runner = runner or SubprocessTaskRunner(
            user_identity=config.user_identity
        )
        self.clock = clock or (lambda: datetime.now(UTC))
        self.alert_callback = alert
        self.owner = f"scheduler-{uuid.uuid4().hex}"
        self.lease_seconds = lease_seconds or max(
            60.0, float(config.scheduler.poll_seconds) * 3
        )
        self.session_id = session_id or config.scheduler.session
        self.alerts_path = Path(config.pygent_root) / "scheduler" / "alerts.jsonl"
        self.alerts: list[str] = []
        self.alert_sink_error: str | None = None
        self.running = False
        self.consecutive_failures = 0
        self.last_error: str | None = None
        self.last_error_at: datetime | None = None
        self.last_success_at: datetime | None = None
        self._started = False
        self._start_lock = threading.Lock()
        self._active_tasks: set[str] = set()
        self._active_lock = threading.Lock()
        self._stop = threading.Event()
        self.sleeper = sleeper or self._stop.wait

    def stop(self) -> None:
        self._stop.set()

    def health(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "started": self._started,
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error,
            "last_error_at": (
                None if self.last_error_at is None else iso(self.last_error_at)
            ),
            "last_success_at": (
                None if self.last_success_at is None else iso(self.last_success_at)
            ),
        }

    def run_forever(self) -> None:
        startup = True
        failures = 0
        self.running = True
        try:
            while not self._stop.is_set():
                try:
                    self.run_once(startup=startup)
                except Exception as exc:
                    self.consecutive_failures = failures + 1
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    self.last_error_at = self.clock()
                    if not self._is_recoverable(exc):
                        self._alert(f"scheduler fatal: {self.last_error}")
                        raise
                    failures += 1
                    self.consecutive_failures = failures
                    if failures == 1 or failures & (failures - 1) == 0:
                        self._alert(
                            f"scheduler loop failed ({failures}x): {self.last_error}"
                        )
                    delay = self._retry_delay(failures)
                else:
                    if failures:
                        self._alert(
                            f"scheduler recovered after {failures} failure(s); "
                            f"last error: {self.last_error}"
                        )
                    failures = 0
                    self.consecutive_failures = 0
                    self.last_success_at = self.clock()
                    startup = False
                    delay = float(self.config.scheduler.poll_seconds)
                if self._stop.is_set():
                    break
                self.sleeper(delay)
        finally:
            self.running = False

    @staticmethod
    def _is_recoverable(exc: BaseException) -> bool:
        if isinstance(exc, sqlite3.OperationalError):
            message = str(exc).lower()
            return any(
                marker in message
                for marker in (
                    "locked",
                    "busy",
                    "disk i/o",
                    "disk is full",
                    "unable to open",
                )
            )
        return isinstance(exc, OSError)

    def _retry_delay(self, failures: int) -> float:
        base = max(1.0, float(self.config.scheduler.poll_seconds))
        return min(max(60.0, base), base * (2 ** min(failures - 1, 6)))

    def run_once(self, *, now: datetime | None = None, startup: bool = False) -> dict[str, int]:
        self._ensure_started()
        current = now or self.clock()
        self._recover_stale_runs(current)
        submissions = self._process_submissions(current)
        tasks = self._run_due_tasks(current, startup=startup)
        return {"submissions": submissions, "tasks": tasks}

    def _ensure_started(self) -> None:
        if self._started:
            return
        with self._start_lock:
            if self._started:
                return
            self._startup()
            self._started = True

    def _startup(self) -> None:
        self.store.migrate()
        for source in ("user", "agent"):
            load_secret(self.user_pygent_root, source)
        self.alerts_path.parent.mkdir(parents=True, exist_ok=True)

    def _recover_stale_runs(self, now: datetime) -> int:
        recovered = 0
        for run in self.store.list_running_runs():
            task = self.store.task(run.task_id)
            if task is None:
                self.store.finalize_run(
                    run_id=run.id,
                    task_id=run.task_id,
                    lease_id=None,
                    status="interrupted",
                    started_at=run.started_at,
                    finished_at=utc_now(),
                    exit_code=None,
                    output=run.output,
                    error="previous execution outcome unknown",
                    missed=run.missed,
                    next_run_at=None,
                )
                self._alert(
                    f"task {run.task_id} interrupted: task no longer exists"
                )
                recovered += 1
                continue
            with self._active_lock:
                if task.id in self._active_tasks:
                    continue
            if task.lease_expires_at is not None and task.lease_expires_at > now:
                continue
            try:
                spec = parse_schedule(task.schedule)
            except ValueError:
                spec = None
            next_at = (
                advance(spec, fired_at=now)
                if spec is not None and spec.kind not in {"once", "event"}
                else None
            )
            self.store.finalize_run(
                run_id=run.id,
                task_id=task.id,
                lease_id=task.lease_id,
                status="interrupted",
                started_at=run.started_at,
                finished_at=utc_now(),
                exit_code=None,
                output=run.output,
                error="previous execution outcome unknown",
                missed=run.missed,
                next_run_at=next_at,
                task_status=(
                    "paused"
                    if spec is not None
                    and spec.kind == "once"
                    and task.status == "approved"
                    else None
                ),
            )
            self._alert(
                f"task {task.id} interrupted: previous execution outcome unknown"
            )
            recovered += 1
        return recovered

    def _process_submissions(self, now: datetime) -> int:
        submissions = self.store.pending_submissions(limit=1000)
        events = []
        processed = 0
        for submission in submissions:
            if submission.source in INTERNAL_SOURCES:
                valid = True
            else:
                try:
                    valid = verify(
                        submission_payload(
                            submission.kind, submission.source, submission.payload
                        ),
                        submission.source,
                        submission.signature,
                        self.user_pygent_root,
                    )
                except (OSError, ValueError):
                    valid = False
            if not valid:
                self._quarantine(
                    submission.id, "missing or invalid signature"
                )
                continue
            try:
                if submission.kind == "task":
                    self._promote_task(submission, now)
                elif submission.kind == "control":
                    self._apply_control(submission.id, submission.payload, now)
                elif submission.kind == "event":
                    events.append(submission)
                else:
                    raise ValueError(f"unsupported submission kind {submission.kind!r}")
                processed += 1
            except (ValueError, KeyError, TypeError) as exc:
                self.store.mark_submission(
                    submission.id, "rejected", error=f"{type(exc).__name__}: {exc}"
                )
                self._alert(f"rejected submission {submission.id}: {exc}")
        # Events run after task/control submissions so a task created in the
        # same poll can receive an event emitted with it.
        for submission in events:
            try:
                self._consume_event(submission, now)
                self.store.mark_submission(submission.id, "verified")
                processed += 1
            except (ValueError, KeyError, TypeError) as exc:
                self.store.mark_submission(
                    submission.id, "rejected", error=f"{type(exc).__name__}: {exc}"
                )
                self._alert(f"rejected event {submission.id}: {exc}")
        return processed

    def _quarantine(self, submission_id: str, reason: str) -> None:
        self.store.mark_submission(submission_id, "quarantined", error=reason)
        self._alert(f"quarantined submission {submission_id}: {reason}")

    def _promote_task(self, submission: Submission, now: datetime) -> Task:
        source = submission.source
        payload = submission.payload
        name = str(payload.get("name") or "").strip()
        schedule_text = str(payload.get("schedule") or "").strip()
        prompt = str(payload.get("prompt") or "")
        if not name:
            raise ValueError("task name must be non-empty")
        if not prompt.strip():
            raise ValueError("task prompt must be non-empty")
        spec = parse_schedule(schedule_text)
        next_at = (
            spec.at
            if spec.kind == "once"
            else next_run(spec, after=now, now=now)
        )
        status = (
            "approved"
            if source == "user" or self.config.scheduler.agent_create_enabled
            else "needs_review"
        )
        priority = str(payload.get("priority") or DEFAULT_PRIORITY).strip().lower()
        delivery = str(payload.get("delivery") or DEFAULT_DELIVERY).strip().lower()
        note: str | None = None
        if source == "agent":
            delivery = DEFAULT_DELIVERY
            if priority == "high":
                priority = DEFAULT_PRIORITY
                note = "agent-requested-high normalized"
        origin = source if source in TASK_ORIGINS else "user"
        return self.store.promote_task_submission(
            submission_id=submission.id,
            name=name,
            schedule=schedule_text,
            prompt=prompt,
            session=str(payload.get("session") or self.session_id),
            source="self" if source in INTERNAL_SOURCES else source,
            status=status,
            next_run_at=next_at,
            instructions=str(payload.get("instructions") or ""),
            origin=origin,
            delivery=delivery,
            priority=priority,
            note=note,
            created_at=now,
        )

    def _apply_control(
        self, submission_id: str, payload: dict[str, Any], now: datetime
    ) -> str:
        action = str(payload.get("action") or "").strip()
        task_id = str(payload.get("task_id") or "").strip()
        if not task_id:
            raise ValueError("control submission requires task_id")
        task = self.store.task(task_id)
        if task is None:
            raise ValueError(f"unknown task {task_id!r}")
        next_at = (
            self._next_for_resume(task, now)
            if action in {"approve", "resume"}
            else None
        )
        return self.store.apply_control_and_mark(
            submission_id=submission_id,
            action=action,
            task_id=task_id,
            next_run_at=next_at,
            now=now,
        )

    @staticmethod
    def _next_for_resume(task: Task, now: datetime) -> datetime | None:
        spec = parse_schedule(task.schedule)
        if spec.kind == "once":
            return spec.at
        return next_run(spec, after=now, now=now)

    def _consume_event(self, submission: Submission, now: datetime) -> None:
        event = str(submission.payload.get("event") or "").strip()
        if not event:
            raise ValueError("event submission requires event")
        for task in self.store.list_tasks(status="approved"):
            try:
                spec = parse_schedule(task.schedule)
            except ValueError:
                continue
            if spec.kind != "event" or spec.event != event:
                continue
            if not self.store.claim_event_delivery(
                submission.id, task.id, now=now
            ):
                continue
            try:
                claimed = self.store.claim_task(
                    task.id,
                    now=now,
                    lease_id=self.owner,
                    lease_seconds=self.lease_seconds,
                )
            except Exception as exc:
                self.store.finish_event_delivery(
                    submission.id,
                    task.id,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                    now=now,
                )
                raise
            if claimed is None:
                self.store.finish_event_delivery(
                    submission.id, task.id, status="skipped", now=now
                )
                continue
            self._execute_task(claimed, now, missed=False)
            self.store.finish_event_delivery(
                submission.id, task.id, status="done", now=now
            )

    def _run_due_tasks(self, now: datetime, *, startup: bool) -> int:
        tasks = self.store.claim_due_tasks(
            now=now,
            lease_id=self.owner,
            lease_seconds=self.lease_seconds,
            limit=self.config.scheduler.max_concurrent,
        )
        if not tasks:
            return 0
        missed_by_id = {
            task.id: bool(
                startup
                and task.next_run_at is not None
                and task.next_run_at < now
            )
            for task in tasks
        }
        with ThreadPoolExecutor(
            max_workers=self.config.scheduler.max_concurrent
        ) as pool:
            futures = [
                pool.submit(self._execute_task, task, now, missed_by_id[task.id])
                for task in tasks
            ]
            for future in futures:
                future.result()
        return len(tasks)

    def _execute_task(self, task: Task, now: datetime, missed: bool) -> None:
        started = utc_now()
        spec = parse_schedule(task.schedule)
        if task.prompt_hash != prompt_hash(task.prompt):
            message = f"prompt hash mismatch for task {task.id}"
            self.store.finalize_run(
                run_id=f"run-{uuid.uuid4().hex}",
                task_id=task.id,
                lease_id=self.owner,
                status="rejected",
                started_at=started,
                finished_at=utc_now(),
                exit_code=None,
                output="",
                error=message,
                missed=missed,
                next_run_at=None,
                task_status="rejected",
            )
            self._alert(message)
            return
        stale = self.store.running_run(task.id)
        if stale is not None:
            next_at = None if spec.kind == "once" else advance(spec, fired_at=now)
            self.store.finalize_run(
                run_id=stale.id,
                task_id=task.id,
                lease_id=self.owner,
                status="interrupted",
                started_at=stale.started_at,
                finished_at=utc_now(),
                exit_code=None,
                output=stale.output,
                error="previous execution outcome unknown",
                missed=stale.missed,
                next_run_at=next_at,
                task_status=(
                    "paused"
                    if spec.kind == "once" and task.status == "approved"
                    else None
                ),
            )
            self._alert(
                f"task {task.id} interrupted: previous execution outcome unknown"
            )
            return
        run = self.store.begin_run(
            task_id=task.id,
            lease_id=self.owner,
            started_at=started,
            missed=missed,
        )
        with self._active_lock:
            self._active_tasks.add(task.id)
        try:
            try:
                result = self.runner.run(task)
            except Exception as exc:  # noqa: BLE001 - a failed task must become an audit run
                result = TaskResult(
                    exit_code=None,
                    error=f"{type(exc).__name__}: {exc}",
                    started_at=started,
                    finished_at=utc_now(),
                )
            status = "success" if result.exit_code == 0 else "failed"
            finished = result.finished_at or utc_now()
            self.store.finalize_run(
                run_id=run.id,
                task_id=task.id,
                lease_id=self.owner,
                status=status,
                started_at=result.started_at or started,
                finished_at=finished,
                exit_code=result.exit_code,
                output=result.output,
                error=result.error,
                missed=missed,
                next_run_at=advance(spec, fired_at=finished),
            )
        finally:
            with self._active_lock:
                self._active_tasks.discard(task.id)
        if status == "failed":
            self._alert(f"task {task.id} failed: {result.error or result.output}")

    def _alert(self, message: str) -> None:
        self.alerts.append(message)
        try:
            append_jsonl(
                self.alerts_path,
                {"at": iso(utc_now()), "message": message},
            )
        except Exception as exc:  # noqa: BLE001 - alerting must not stop the loop
            self.alert_sink_error = f"{type(exc).__name__}: {exc}"
            self._write_alert_fallback(message)
        if self.alert_callback is not None:
            try:
                self.alert_callback(message)
            except Exception as exc:  # noqa: BLE001 - callbacks must not stop scheduling
                self.alert_sink_error = f"{type(exc).__name__}: {exc}"
                self._write_alert_fallback(message)

    def _write_alert_fallback(self, message: str) -> None:
        try:
            print(
                f"scheduler alert sink failed ({self.alert_sink_error}): {message}",
                file=sys.stderr,
            )
        except OSError:
            pass


__all__ = [
    "SchedulerDaemon",
    "SubprocessTaskRunner",
    "TaskResult",
    "TaskRunner",
    "lara_command",
]
