"""Scheduler daemon promotion, security, and execution tests."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pygent.agent.lara.scheduler.daemon import SchedulerDaemon, TaskResult
from pygent.agent.lara.scheduler.keys import init_keys, load_secret, sign
from pygent.agent.lara.scheduler.store import (
    SchedulerStore,
    prompt_hash,
    scheduler_db_path,
    submission_payload,
)
from pygent.agent.lara.schema import RunConfig


class _Runner:
    def __init__(self, exit_code: int = 0) -> None:
        self.exit_code = exit_code
        self.tasks: list[object] = []

    def run(self, task: object) -> TaskResult:
        self.tasks.append(task)
        return TaskResult(exit_code=self.exit_code, output="ok")


def _config(tmp_path: Path, *, agent_create_enabled: bool = False) -> RunConfig:
    return RunConfig(
        workspace_root=str(tmp_path / "workspace"),
        user_lara_root=str(tmp_path / "user"),
        scheduler={  # type: ignore[arg-type]
            "agent_create_enabled": agent_create_enabled,
            "session": "scheduler",
            "max_concurrent": 2,
            "poll_seconds": 10,
        },
    )


def _daemon(
    tmp_path: Path, *, agent_create_enabled: bool = False, runner: _Runner | None = None
) -> tuple[SchedulerDaemon, SchedulerStore, Path]:
    config = _config(tmp_path, agent_create_enabled=agent_create_enabled)
    store = SchedulerStore(scheduler_db_path(config.lara_root))
    store.migrate()
    init_keys(config.user_lara_root)
    daemon = SchedulerDaemon(
        config,
        store=store,
        runner=runner or _Runner(),
        session_id="scheduler-session",
    )
    return daemon, store, Path(config.user_lara_root)


def _sign(
    user_root: Path, *, kind: str, source: str, payload: Mapping[str, object]
) -> str:
    return sign(
        submission_payload(kind, source, payload),
        load_secret(user_root, source),
    )


def _submit(
    store: SchedulerStore,
    user_root: Path,
    *,
    kind: str,
    source: str,
    payload: Mapping[str, object],
    signature: str | None = None,
) -> str:
    signed = (
        signature
        if signature is not None
        else _sign(user_root, kind=kind, source=source, payload=payload)
    )
    return store.submit(
        kind=kind,
        source=source,
        payload=dict(payload),
        signature=signed,
        submission_id=None,
    ).id


def test_user_task_promotes_to_approved(tmp_path: Path) -> None:
    daemon, store, user_root = _daemon(tmp_path)
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    submission_id = _submit(
        store,
        user_root,
        kind="task",
        source="user",
        payload={
            "name": "daily",
            "schedule": "every 1h",
            "prompt": "do it",
            "session": "scheduler-session",
        },
    )
    daemon.run_once(now=now)
    task = store.list_tasks()[0]
    assert task.status == "approved"
    assert task.next_run_at == now + timedelta(hours=1)
    assert store.submission(submission_id).status == "verified"  # type: ignore[union-attr]


def test_agent_task_defaults_to_needs_review(tmp_path: Path) -> None:
    daemon, store, user_root = _daemon(tmp_path)
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    _submit(
        store,
        user_root,
        kind="task",
        source="agent",
        payload={"name": "agent", "schedule": "daily 09:00", "prompt": "review"},
    )
    daemon.run_once(now=now)
    assert store.list_tasks()[0].status == "needs_review"


def test_agent_task_auto_approves_when_enabled(tmp_path: Path) -> None:
    daemon, store, user_root = _daemon(tmp_path, agent_create_enabled=True)
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    _submit(
        store,
        user_root,
        kind="task",
        source="agent",
        payload={"name": "agent", "schedule": "daily 09:00", "prompt": "auto"},
    )
    daemon.run_once(now=now)
    assert store.list_tasks()[0].status == "approved"


def test_unsigned_tampered_and_forged_submissions_are_quarantined(
    tmp_path: Path,
) -> None:
    daemon, store, user_root = _daemon(tmp_path)
    payload = {"name": "bad", "schedule": "every 1h", "prompt": "bad"}
    unsigned = _submit(
        store, user_root, kind="task", source="user", payload=payload, signature=""
    )
    tampered = _submit(
        store,
        user_root,
        kind="task",
        source="user",
        payload={**payload, "prompt": "changed"},
        signature=_sign(user_root, kind="task", source="user", payload=payload),
    )
    forged = _submit(
        store,
        user_root,
        kind="task",
        source="user",
        payload=payload,
        signature=_sign(user_root, kind="task", source="agent", payload=payload),
    )
    daemon.run_once(now=datetime(2026, 1, 1, tzinfo=UTC))
    assert store.submission(unsigned).status == "quarantined"  # type: ignore[union-attr]
    assert store.submission(tampered).status == "quarantined"  # type: ignore[union-attr]
    assert store.submission(forged).status == "quarantined"  # type: ignore[union-attr]
    assert store.list_tasks() == []


def test_missed_due_task_runs_once_and_advances(tmp_path: Path) -> None:
    daemon, store, _user_root = _daemon(tmp_path)
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    task = store.create_task(
        name="missed",
        schedule="every 1h",
        prompt="run me",
        session="scheduler-session",
        source="user",
        status="approved",
        next_run_at=now - timedelta(hours=2),
    )
    daemon.run_once(now=now, startup=True)
    runs = store.list_runs(task_id=task.id)
    assert len(runs) == 1
    assert runs[0].missed is True
    assert store.task(task.id).next_run_at is not None  # type: ignore[union-attr]


def test_single_at_task_terminates_after_run(tmp_path: Path) -> None:
    daemon, store, _user_root = _daemon(tmp_path)
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    task = store.create_task(
        name="once",
        schedule="at 2026-01-01T11:00:00+00:00",
        prompt="run once",
        session="scheduler-session",
        source="user",
        status="approved",
        next_run_at=now - timedelta(hours=1),
    )
    daemon.run_once(now=now, startup=True)
    assert store.task(task.id).next_run_at is None  # type: ignore[union-attr]
    assert len(store.list_runs(task_id=task.id)) == 1


def test_changed_prompt_hash_rejects_and_alerts(tmp_path: Path) -> None:
    daemon, store, _user_root = _daemon(tmp_path)
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    task = store.create_task(
        name="tampered",
        schedule="every 1h",
        prompt="original",
        session="scheduler-session",
        source="user",
        status="approved",
        next_run_at=now - timedelta(minutes=1),
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE tasks SET prompt='tampered' WHERE id=?", (task.id,)
        )
    daemon.run_once(now=now)
    assert store.task(task.id).status == "rejected"  # type: ignore[union-attr]
    assert store.list_runs(task_id=task.id)[0].status == "rejected"
    assert any("prompt hash mismatch" in alert for alert in daemon.alerts)
    updated = store.task(task.id)
    assert updated is not None
    assert prompt_hash(updated.prompt) != updated.prompt_hash


def test_control_submission_updates_task_status(tmp_path: Path) -> None:
    daemon, store, user_root = _daemon(tmp_path)
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    task = store.create_task(
        name="review",
        schedule="daily 09:00",
        prompt="review",
        session="scheduler-session",
        source="agent",
        status="needs_review",
        next_run_at=None,
    )
    _submit(
        store,
        user_root,
        kind="control",
        source="user",
        payload={"action": "approve", "task_id": task.id},
    )
    daemon.run_once(now=now)
    assert store.task(task.id).status == "approved"  # type: ignore[union-attr]


def test_event_submission_triggers_on_task(tmp_path: Path) -> None:
    runner = _Runner()
    daemon, store, user_root = _daemon(tmp_path, runner=runner)
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    task = store.create_task(
        name="event",
        schedule="on build",
        prompt="event run",
        session="scheduler-session",
        source="user",
        status="approved",
        next_run_at=None,
    )
    submission_id = _submit(
        store,
        user_root,
        kind="event",
        source="user",
        payload={"event": "build", "data": {"ok": True}},
    )
    daemon.run_once(now=now)
    assert len(runner.tasks) == 1
    assert len(store.list_runs(task_id=task.id)) == 1
    assert store.submission(submission_id).status == "verified"  # type: ignore[union-attr]


def test_run_forever_processes_one_loop_and_stops(tmp_path: Path) -> None:
    runner = _Runner()
    daemon, store, _user_root = _daemon(tmp_path, runner=runner)
    now = datetime.now(UTC)
    task = store.create_task(
        name="loop",
        schedule="every 1h",
        prompt="run loop",
        session="scheduler-session",
        source="user",
        status="approved",
        next_run_at=now - timedelta(seconds=1),
    )
    daemon.sleeper = lambda _seconds: daemon.stop()
    daemon.run_forever()
    assert len(store.list_runs(task_id=task.id)) == 1


def test_default_session_is_scheduler_name(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = SchedulerStore(scheduler_db_path(config.lara_root))
    store.migrate()
    init_keys(config.user_lara_root)
    daemon = SchedulerDaemon(config, store=store, runner=_Runner())
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    _submit(
        store,
        Path(config.user_lara_root),
        kind="task",
        source="user",
        payload={"name": "default", "schedule": "every 1h", "prompt": "p"},
    )
    daemon.run_once(now=now)
    assert store.list_tasks()[0].session == "scheduler"


def test_agent_high_priority_is_normalized_with_audit_note(
    tmp_path: Path,
) -> None:
    daemon, store, user_root = _daemon(tmp_path, agent_create_enabled=True)
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    submission_id = _submit(
        store,
        user_root,
        kind="task",
        source="agent",
        payload={
            "name": "agent-high",
            "schedule": "every 1h",
            "prompt": "urgent",
            "priority": "high",
            "delivery": "sync-reply",
        },
    )
    daemon.run_once(now=now)
    task = store.list_tasks()[0]
    assert task.source == "agent"
    assert task.origin == "user"
    assert task.priority == "normal"
    assert task.delivery == "fire-and-forget"
    submission = store.submission(submission_id)
    assert submission is not None
    assert submission.status == "verified"
    assert submission.error == "agent-requested-high normalized"


def test_user_priority_and_delivery_are_honored(tmp_path: Path) -> None:
    daemon, store, user_root = _daemon(tmp_path)
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    _submit(
        store,
        user_root,
        kind="task",
        source="user",
        payload={
            "name": "loud",
            "schedule": "every 1h",
            "prompt": "speak",
            "priority": "high",
            "delivery": "speak",
        },
    )
    _submit(
        store,
        user_root,
        kind="task",
        source="user",
        payload={"name": "quiet", "schedule": "daily 09:00", "prompt": "quiet"},
    )
    daemon.run_once(now=now)
    tasks = {task.name: task for task in store.list_tasks()}
    assert (tasks["loud"].origin, tasks["loud"].delivery, tasks["loud"].priority) == (
        "user",
        "speak",
        "high",
    )
    assert (
        tasks["quiet"].origin,
        tasks["quiet"].delivery,
        tasks["quiet"].priority,
    ) == ("user", "fire-and-forget", "normal")


def test_internal_timer_and_event_submissions_are_unsigned(
    tmp_path: Path,
) -> None:
    daemon, store, _user_root = _daemon(tmp_path, agent_create_enabled=True)
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    timer = store.submit_internal(
        kind="task",
        origin="timer",
        payload={
            "name": "timer-intent",
            "schedule": "every 1h",
            "prompt": "tick",
            "priority": "high",
            "delivery": "speak",
        },
    )
    event = store.submit_internal(
        kind="task",
        origin="event",
        payload={
            "name": "event-intent",
            "schedule": "on build",
            "prompt": "build",
            "priority": "low",
        },
    )
    assert timer.signature == ""
    assert event.signature == ""
    daemon.run_once(now=now)
    tasks = {task.name: task for task in store.list_tasks()}
    assert (tasks["timer-intent"].source, tasks["timer-intent"].origin) == (
        "self",
        "timer",
    )
    assert (tasks["timer-intent"].delivery, tasks["timer-intent"].priority) == (
        "speak",
        "high",
    )
    assert (tasks["event-intent"].source, tasks["event-intent"].origin) == (
        "self",
        "event",
    )
    assert tasks["event-intent"].next_run_at is None
    assert (tasks["event-intent"].delivery, tasks["event-intent"].priority) == (
        "fire-and-forget",
        "low",
    )
    timer_submission = store.submission(timer.id)
    event_submission = store.submission(event.id)
    assert timer_submission is not None and timer_submission.status == "verified"
    assert event_submission is not None and event_submission.status == "verified"
