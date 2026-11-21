"""Scheduler daemon promotion, security, and execution tests."""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from pygent.agent.pygent_v1.scheduler.daemon import SchedulerDaemon, TaskResult
from pygent.agent.pygent_v1.scheduler.keys import (
    SchedulerKeysMissing,
    init_keys,
    load_secret,
    sign,
)
from pygent.agent.pygent_v1.scheduler.store import (
    SchedulerStore,
    prompt_hash,
    scheduler_db_path,
    submission_payload,
)
from pygent.agent.pygent_v1.schema import RunConfig


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
        user_pygent_root=str(tmp_path / "user"),
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
    store = SchedulerStore(scheduler_db_path(config.pygent_root))
    store.migrate()
    init_keys(config.user_pygent_root)
    daemon = SchedulerDaemon(
        config,
        store=store,
        runner=runner or _Runner(),
        session_id="scheduler-session",
    )
    return daemon, store, Path(config.user_pygent_root)


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
    store = SchedulerStore(scheduler_db_path(config.pygent_root))
    store.migrate()
    init_keys(config.user_pygent_root)
    daemon = SchedulerDaemon(config, store=store, runner=_Runner())
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    _submit(
        store,
        Path(config.user_pygent_root),
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


def test_initialization_runs_once_across_loops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    daemon, store, _user_root = _daemon(tmp_path)
    calls: list[int] = []
    original = store.migrate

    def migrate() -> None:
        calls.append(1)
        original()

    monkeypatch.setattr(store, "migrate", migrate)
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    daemon.run_once(now=now)
    daemon.run_once(now=now)
    assert calls == [1]
    assert daemon.health()["started"] is True


def test_transient_startup_failure_retries_and_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    daemon, store, _user_root = _daemon(tmp_path)
    calls: list[int] = []
    original = store.migrate

    def migrate() -> None:
        calls.append(1)
        if len(calls) == 1:
            raise sqlite3.OperationalError("database is locked")
        original()

    monkeypatch.setattr(store, "migrate", migrate)
    delays: list[float] = []

    def sleeper(seconds: float) -> None:
        delays.append(seconds)
        if len(delays) >= 2:
            daemon.stop()

    daemon.sleeper = sleeper
    daemon.run_forever()
    assert calls == [1, 1]
    assert delays == [10, 10]
    assert daemon.health()["started"] is True
    assert daemon.consecutive_failures == 0
    assert daemon.last_error is not None
    assert "OperationalError" in daemon.last_error
    assert daemon.last_success_at is not None
    assert any("scheduler loop failed" in alert for alert in daemon.alerts)
    assert any("recovered after 1 failure" in alert for alert in daemon.alerts)


def test_fatal_startup_error_alerts_and_raises(tmp_path: Path) -> None:
    daemon, _store, user_root = _daemon(tmp_path)
    (user_root / "scheduler" / "keys" / "agent.key").unlink()
    with pytest.raises(SchedulerKeysMissing):
        daemon.run_forever()
    assert daemon.running is False
    assert daemon.consecutive_failures == 1
    assert daemon.last_error is not None
    assert "SchedulerKeysMissing" in daemon.last_error
    assert any("scheduler fatal" in alert for alert in daemon.alerts)


def test_alert_failure_does_not_abort_task_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _Runner(exit_code=1)
    daemon, store, _user_root = _daemon(tmp_path, runner=runner)
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    task = store.create_task(
        name="fails",
        schedule="every 1h",
        prompt="fail",
        session="scheduler-session",
        source="user",
        status="approved",
        next_run_at=now - timedelta(seconds=1),
    )

    def broken_append(*args: object, **kwargs: object) -> None:
        raise OSError("alert disk full")

    monkeypatch.setattr(
        "pygent.agent.pygent_v1.scheduler.daemon.append_jsonl", broken_append
    )

    def broken_callback(message: str) -> None:
        raise RuntimeError("callback down")

    daemon.alert_callback = broken_callback
    daemon.run_once(now=now)
    assert [run.status for run in store.list_runs(task_id=task.id)] == ["failed"]
    assert any("failed" in alert for alert in daemon.alerts)
    assert daemon.alert_sink_error is not None
    assert "scheduler alert sink failed" in capsys.readouterr().err


def test_stop_interrupts_backoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    daemon, _store, _user_root = _daemon(tmp_path)

    def broken_run_once(*, now: datetime | None = None, startup: bool = False) -> dict[str, int]:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(daemon, "run_once", broken_run_once)
    thread = threading.Thread(target=daemon.run_forever, daemon=True)
    thread.start()
    deadline = time.time() + 2
    while daemon.consecutive_failures == 0 and time.time() < deadline:
        time.sleep(0.01)
    assert daemon.consecutive_failures == 1
    daemon.stop()
    thread.join(timeout=2)
    assert not thread.is_alive()


def test_transient_poll_error_recovers_and_runs_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _Runner()
    daemon, store, _user_root = _daemon(tmp_path, runner=runner)
    now = datetime.now(UTC)
    task = store.create_task(
        name="recover",
        schedule="every 1h",
        prompt="run after recovery",
        session="scheduler-session",
        source="user",
        status="approved",
        next_run_at=now - timedelta(seconds=1),
    )
    original = store.pending_submissions
    calls: list[int] = []

    def flaky(*, limit: int = 100):  # type: ignore[no-untyped-def]
        calls.append(1)
        if len(calls) == 1:
            raise sqlite3.OperationalError("disk I/O error")
        return original(limit=limit)

    monkeypatch.setattr(store, "pending_submissions", flaky)
    delays: list[float] = []

    def sleeper(seconds: float) -> None:
        delays.append(seconds)
        if len(delays) >= 2:
            daemon.stop()

    daemon.sleeper = sleeper
    daemon.run_forever()
    assert len(runner.tasks) == 1
    assert delays == [10, 10]
    assert daemon.consecutive_failures == 0
    assert daemon.last_error is not None
    assert "OperationalError" in daemon.last_error
    assert daemon.last_success_at is not None
    assert len(store.list_runs(task_id=task.id)) == 1
    assert any("scheduler loop failed" in alert for alert in daemon.alerts)
    assert any("recovered after 1 failure" in alert for alert in daemon.alerts)


def test_run_finalize_failure_is_interrupted_without_reexecution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _Runner()
    daemon, store, _user_root = _daemon(tmp_path, runner=runner)
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    task = store.create_task(
        name="interrupt",
        schedule="every 1h",
        prompt="run once",
        session="scheduler-session",
        source="user",
        status="approved",
        next_run_at=now - timedelta(seconds=1),
    )
    original = store.finalize_run
    calls: list[int] = []

    def flaky(**kwargs: object) -> None:
        calls.append(1)
        if len(calls) == 1:
            raise sqlite3.OperationalError("disk I/O error")
        original(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(store, "finalize_run", flaky)
    with pytest.raises(sqlite3.OperationalError):
        daemon.run_once(now=now)
    assert len(runner.tasks) == 1
    assert store.running_run(task.id) is not None

    daemon.run_once(now=now + timedelta(seconds=61))
    assert len(runner.tasks) == 1
    assert [run.status for run in store.list_runs(task_id=task.id)] == [
        "interrupted"
    ]
    assert store.task(task.id).next_run_at is not None  # type: ignore[union-attr]


def test_stale_event_run_is_interrupted_on_next_poll(
    tmp_path: Path,
) -> None:
    runner = _Runner()
    daemon, store, _user_root = _daemon(tmp_path, runner=runner)
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
    store.claim_task(
        task.id,
        now=now - timedelta(seconds=120),
        lease_id="old-daemon",
        lease_seconds=60,
    )
    store.begin_run(
        task_id=task.id,
        lease_id="old-daemon",
        started_at=now - timedelta(seconds=120),
        missed=False,
    )
    daemon.run_once(now=now)
    assert runner.tasks == []
    assert [run.status for run in store.list_runs(task_id=task.id)] == [
        "interrupted"
    ]
    updated = store.task(task.id)
    assert updated is not None
    assert updated.lease_id is None


def test_event_mark_failure_does_not_replay_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _Runner()
    daemon, store, user_root = _daemon(tmp_path, runner=runner)
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    store.create_task(
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
        payload={"event": "build", "data": {}},
    )
    original = store.mark_submission
    calls: list[int] = []

    def flaky(*args: object, **kwargs: object) -> None:
        calls.append(1)
        if len(calls) == 1:
            raise sqlite3.OperationalError("disk I/O error")
        original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(store, "mark_submission", flaky)
    with pytest.raises(sqlite3.OperationalError):
        daemon.run_once(now=now)
    daemon.run_once(now=now)
    assert len(runner.tasks) == 1
    assert store.submission(submission_id).status == "verified"  # type: ignore[union-attr]
