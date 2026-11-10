"""Unit tests for the scheduler SQLite store."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from pygent.agent.lara.scheduler.store import SchedulerStore, prompt_hash


def _store(tmp_path: Path) -> SchedulerStore:
    store = SchedulerStore(tmp_path / "scheduler" / "lara-scheduler.sqlite3")
    store.migrate()
    return store


def test_submission_queue_and_idempotent_insert(tmp_path: Path) -> None:
    store = _store(tmp_path)
    payload = {"name": "task", "schedule": "every 1h", "prompt": "hello"}
    first = store.submit(
        kind="task",
        source="user",
        payload=payload,
        signature="sig",
        submission_id="sub-1",
    )
    second = store.submit(
        kind="task",
        source="user",
        payload=payload,
        signature="sig",
        submission_id="sub-1",
    )
    assert first.id == second.id == "sub-1"
    assert [item.id for item in store.pending_submissions()] == ["sub-1"]
    store.mark_submission("sub-1", "verified", task_id="task-1")
    assert store.submission("sub-1").status == "verified"  # type: ignore[union-attr]
    assert store.pending_submissions() == []


def test_task_and_run_audit(tmp_path: Path) -> None:
    store = _store(tmp_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    task = store.create_task(
        name="daily",
        schedule="daily 09:00",
        prompt="do it",
        session="scheduler-session",
        source="user",
        status="approved",
        next_run_at=now,
    )
    assert task.prompt_hash == prompt_hash("do it")
    assert store.list_tasks()[0].id == task.id
    run = store.append_run(
        task_id=task.id,
        status="success",
        started_at=now,
        finished_at=now + timedelta(seconds=1),
        exit_code=0,
        output="ok",
        error=None,
        missed=True,
    )
    assert run.missed is True
    assert store.list_runs(task_id=task.id)[0].exit_code == 0
    store.set_task_status(task.id, "paused")
    assert store.task(task.id).status == "paused"  # type: ignore[union-attr]


def test_atomic_task_claim_and_release(tmp_path: Path) -> None:
    store = _store(tmp_path)
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    task = store.create_task(
        name="due",
        schedule="every 1h",
        prompt="run",
        session="scheduler-session",
        source="user",
        status="approved",
        next_run_at=now - timedelta(seconds=1),
    )
    first = store.claim_due_tasks(
        now=now, lease_id="daemon-1", lease_seconds=60, limit=10
    )
    assert [item.id for item in first] == [task.id]
    second = store.claim_due_tasks(
        now=now, lease_id="daemon-2", lease_seconds=60, limit=10
    )
    assert second == []
    store.release_task(task.id, lease_id="daemon-1")
    third = store.claim_due_tasks(
        now=now, lease_id="daemon-2", lease_seconds=60, limit=10
    )
    assert [item.id for item in third] == [task.id]


def test_claim_event_task_and_delete(tmp_path: Path) -> None:
    store = _store(tmp_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    task = store.create_task(
        name="event",
        schedule="on build",
        prompt="run",
        session="scheduler-session",
        source="user",
        status="approved",
        next_run_at=None,
    )
    claimed = store.claim_task(
        task.id, now=now, lease_id="daemon", lease_seconds=60
    )
    assert claimed is not None
    store.release_task(task.id, lease_id="daemon")
    store.delete_task(task.id)
    assert store.task(task.id) is None


def test_reads_before_daemon_migration_are_empty(tmp_path: Path) -> None:
    store = SchedulerStore(tmp_path / "scheduler" / "lara-scheduler.sqlite3")
    assert store.list_tasks() == []
    assert store.task("missing") is None
    assert store.list_runs() == []
    assert store.submission("missing") is None


def test_trigger_fields_default_and_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    task = store.create_task(
        name="defaults",
        schedule="every 1h",
        prompt="run",
        session="scheduler-session",
        source="user",
        status="approved",
        next_run_at=now,
    )
    assert (task.origin, task.delivery, task.priority) == (
        "user",
        "fire-and-forget",
        "normal",
    )
    explicit = store.create_task(
        name="explicit",
        schedule="on build",
        prompt="run",
        session="scheduler-session",
        source="user",
        status="approved",
        next_run_at=None,
        origin="event",
        delivery="speak",
        priority="high",
    )
    loaded = store.task(explicit.id)
    assert loaded is not None
    assert (loaded.origin, loaded.delivery, loaded.priority) == (
        "event",
        "speak",
        "high",
    )
    listed = {item.id: item for item in store.list_tasks()}
    assert listed[task.id].to_dict()["origin"] == "user"
    assert listed[explicit.id].to_dict()["priority"] == "high"


def test_invalid_trigger_fields_are_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)

    def create(**extra: Any) -> None:
        store.create_task(
            name="bad",
            schedule="every 1h",
            prompt="run",
            session="scheduler-session",
            source="user",
            status="approved",
            next_run_at=now,
            **extra,
        )

    with pytest.raises(ValueError, match="priority"):
        create(priority="urgent")
    with pytest.raises(ValueError, match="delivery"):
        create(delivery="email")
    with pytest.raises(ValueError, match="origin"):
        create(origin="agent")


def test_trigger_columns_migrate_idempotently(tmp_path: Path) -> None:
    path = tmp_path / "scheduler" / "lara-scheduler.sqlite3"
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE tasks (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                schedule TEXT NOT NULL,
                prompt TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                session TEXT NOT NULL,
                source TEXT NOT NULL,
                status TEXT NOT NULL,
                next_run_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                instructions TEXT NOT NULL DEFAULT '',
                lease_id TEXT,
                lease_expires_at TEXT
            )
            """
        )
        connection.execute(
            "INSERT INTO tasks VALUES "
            "('legacy', 'n', 'every 1h', 'p', 'h', 's', 'user', 'approved', "
            "NULL, '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00', "
            "'', NULL, NULL)"
        )
    store = SchedulerStore(path)
    store.migrate()
    store.migrate()
    task = store.task("legacy")
    assert task is not None
    assert (task.origin, task.delivery, task.priority) == (
        "user",
        "fire-and-forget",
        "normal",
    )
    with sqlite3.connect(path) as connection:
        columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(tasks)")
        }
    assert {"origin", "delivery", "priority"} <= columns


def test_claim_orders_by_priority_then_next_run(tmp_path: Path) -> None:
    store = _store(tmp_path)
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    due = now - timedelta(seconds=5)

    def add(name: str, priority: str, next_run_at: datetime) -> str:
        return store.create_task(
            name=name,
            schedule="every 1h",
            prompt=name,
            session="scheduler-session",
            source="user",
            status="approved",
            next_run_at=next_run_at,
            priority=priority,
        ).id

    low_late = add("low-late", "low", due)
    normal_early = add("normal-early", "normal", due - timedelta(minutes=5))
    high_late = add("high-late", "high", due)
    high_early = add("high-early", "high", due - timedelta(minutes=1))
    claimed = store.claim_due_tasks(
        now=now, lease_id="daemon", lease_seconds=60, limit=10
    )
    assert [item.id for item in claimed] == [
        high_early,
        high_late,
        normal_early,
        low_late,
    ]
