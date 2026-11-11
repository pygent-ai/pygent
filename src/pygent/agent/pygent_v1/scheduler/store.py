"""Two-layer SQLite store for scheduler submissions and daemon-owned state.

The ``tasks`` table is the unified trigger bus. Each promoted row records an
``origin`` (``user``/``timer``/``event``), a ``delivery`` mode, and a
``priority``. ``origin`` classifies how an intent entered the bus: signed
user/agent registrations are ``user`` (authorship stays in ``source``), while
daemon timer and event triggers are ``timer`` / ``event``. The bus only
carries timer triggers, event triggers, and agent-registered intents; a user's
direct chat message is served by the foreground runtime and never enters this
store.

Writers only append signed rows to ``submissions``. The daemon verifies and
promotes them into ``tasks``/``runs``; those tables are never updated by tool
or CLI writers. SQLite is opened with WAL and a busy timeout so the daemon,
tools, and CLI can share one file across processes.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TASK_STATUSES = (
    "needs_review",
    "approved",
    "quarantined",
    "rejected",
    "paused",
)
SUBMISSION_STATUSES = ("unverified", "verified", "quarantined", "rejected")
RUN_STATUSES = ("success", "failed", "rejected")
SUBMISSION_SOURCES = ("user", "agent", "timer", "event")
INTERNAL_SOURCES = ("timer", "event")
TASK_ORIGINS = ("user", "timer", "event")
TASK_DELIVERIES = ("sync-reply", "speak", "fire-and-forget")
TASK_PRIORITIES = ("high", "normal", "low")
DEFAULT_DELIVERY = "fire-and-forget"
DEFAULT_PRIORITY = "normal"


def utc_now() -> datetime:
    return datetime.now(UTC)


def iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def parse_iso(value: str | None) -> datetime | None:
    return None if value is None else datetime.fromisoformat(value)


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def scheduler_db_path(pygent_root: str | Path) -> Path:
    return Path(pygent_root) / "scheduler" / "pygent.v1-scheduler.sqlite3"


def scheduler_keys_dir(user_pygent_root: str | Path) -> Path:
    return Path(user_pygent_root) / "scheduler" / "keys"


def submission_payload(
    kind: str, source: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    """The exact object signed by ``lara.scheduler.keys``."""

    return {"kind": kind, "source": source, "payload": dict(payload)}


def _json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _dict(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("scheduler payload must be a JSON object")  # noqa: TRY004
    return parsed


@dataclass(frozen=True, slots=True)
class Submission:
    id: str
    kind: str
    source: str
    payload: dict[str, Any]
    signature: str
    status: str
    created_at: datetime
    processed_at: datetime | None = None
    task_id: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "source": self.source,
            "payload": self.payload,
            "signature": self.signature,
            "status": self.status,
            "created_at": iso(self.created_at),
            "processed_at": None if self.processed_at is None else iso(self.processed_at),
            "task_id": self.task_id,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class Task:
    id: str
    name: str
    schedule: str
    prompt: str
    prompt_hash: str
    session: str
    source: str
    status: str
    next_run_at: datetime | None
    created_at: datetime
    updated_at: datetime
    instructions: str = ""
    lease_id: str | None = None
    lease_expires_at: datetime | None = None
    origin: str = "user"
    delivery: str = DEFAULT_DELIVERY
    priority: str = DEFAULT_PRIORITY

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "schedule": self.schedule,
            "prompt": self.prompt,
            "prompt_hash": self.prompt_hash,
            "session": self.session,
            "source": self.source,
            "status": self.status,
            "next_run_at": None if self.next_run_at is None else iso(self.next_run_at),
            "created_at": iso(self.created_at),
            "updated_at": iso(self.updated_at),
            "instructions": self.instructions,
            "lease_id": self.lease_id,
            "lease_expires_at": (
                None if self.lease_expires_at is None else iso(self.lease_expires_at)
            ),
            "origin": self.origin,
            "delivery": self.delivery,
            "priority": self.priority,
        }


@dataclass(frozen=True, slots=True)
class Run:
    id: str
    task_id: str
    started_at: datetime
    finished_at: datetime | None
    status: str
    exit_code: int | None
    output: str
    error: str | None
    missed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "started_at": iso(self.started_at),
            "finished_at": None if self.finished_at is None else iso(self.finished_at),
            "status": self.status,
            "exit_code": self.exit_code,
            "output": self.output,
            "error": self.error,
            "missed": self.missed,
        }


def _submission(row: sqlite3.Row) -> Submission:
    return Submission(
        id=str(row["id"]),
        kind=str(row["kind"]),
        source=str(row["source"]),
        payload=_dict(str(row["payload"])),
        signature=str(row["signature"]),
        status=str(row["status"]),
        created_at=parse_iso(str(row["created_at"])) or utc_now(),
        processed_at=parse_iso(row["processed_at"]),
        task_id=None if row["task_id"] is None else str(row["task_id"]),
        error=None if row["error"] is None else str(row["error"]),
    )


def _task(row: sqlite3.Row) -> Task:
    keys = row.keys()
    origin = str(row["origin"] or "user") if "origin" in keys else "user"
    delivery = (
        str(row["delivery"] or DEFAULT_DELIVERY)
        if "delivery" in keys
        else DEFAULT_DELIVERY
    )
    priority = (
        str(row["priority"] or DEFAULT_PRIORITY)
        if "priority" in keys
        else DEFAULT_PRIORITY
    )
    return Task(
        id=str(row["id"]),
        name=str(row["name"]),
        schedule=str(row["schedule"]),
        prompt=str(row["prompt"]),
        prompt_hash=str(row["prompt_hash"]),
        session=str(row["session"]),
        source=str(row["source"]),
        status=str(row["status"]),
        next_run_at=parse_iso(row["next_run_at"]),
        created_at=parse_iso(str(row["created_at"])) or utc_now(),
        updated_at=parse_iso(str(row["updated_at"])) or utc_now(),
        instructions=str(row["instructions"] or ""),
        lease_id=None if row["lease_id"] is None else str(row["lease_id"]),
        lease_expires_at=parse_iso(row["lease_expires_at"]),
        origin=origin,
        delivery=delivery,
        priority=priority,
    )


def _run(row: sqlite3.Row) -> Run:
    return Run(
        id=str(row["id"]),
        task_id=str(row["task_id"]),
        started_at=parse_iso(str(row["started_at"])) or utc_now(),
        finished_at=parse_iso(row["finished_at"]),
        status=str(row["status"]),
        exit_code=None if row["exit_code"] is None else int(row["exit_code"]),
        output=str(row["output"] or ""),
        error=None if row["error"] is None else str(row["error"]),
        missed=bool(row["missed"]),
    )


class SchedulerStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA busy_timeout=5000")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _ensure_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS submissions (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                source TEXT NOT NULL,
                payload TEXT NOT NULL,
                signature TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'unverified',
                created_at TEXT NOT NULL,
                processed_at TEXT,
                task_id TEXT,
                error TEXT
            );
            CREATE INDEX IF NOT EXISTS submissions_status_created
                ON submissions(status, created_at);
            CREATE TABLE IF NOT EXISTS tasks (
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
                lease_expires_at TEXT,
                origin TEXT NOT NULL DEFAULT 'user',
                delivery TEXT NOT NULL DEFAULT 'fire-and-forget',
                priority TEXT NOT NULL DEFAULT 'normal'
            );
            CREATE INDEX IF NOT EXISTS tasks_due
                ON tasks(status, next_run_at, lease_expires_at);
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT NOT NULL,
                exit_code INTEGER,
                output TEXT NOT NULL DEFAULT '',
                error TEXT,
                missed INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS runs_task_started
                ON runs(task_id, started_at);
            """
        )
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(tasks)").fetchall()
        }
        if "instructions" not in columns:
            connection.execute(
                "ALTER TABLE tasks ADD COLUMN instructions TEXT NOT NULL DEFAULT ''"
            )
        trigger_columns = {
            "origin": "TEXT NOT NULL DEFAULT 'user'",
            "delivery": "TEXT NOT NULL DEFAULT 'fire-and-forget'",
            "priority": "TEXT NOT NULL DEFAULT 'normal'",
        }
        for name, definition in trigger_columns.items():
            if name not in columns:
                connection.execute(
                    f"ALTER TABLE tasks ADD COLUMN {name} {definition}"
                )

    def migrate(self) -> None:
        """Create or upgrade the schema. Only the daemon calls this."""

        with self._connect() as connection:
            self._ensure_schema(connection)
            connection.execute("PRAGMA user_version=1")

    def submit(
        self,
        *,
        kind: str,
        source: str,
        payload: Mapping[str, Any],
        signature: str,
        submission_id: str | None = None,
        created_at: datetime | None = None,
    ) -> Submission:
        if kind not in {"task", "event", "control"}:
            raise ValueError(f"unsupported scheduler submission kind: {kind!r}")
        if source not in SUBMISSION_SOURCES:
            raise ValueError(f"unsupported scheduler submission source: {source!r}")
        identifier = submission_id or f"sub-{uuid.uuid4().hex}"
        created = iso(created_at or utc_now())
        with self._connect() as connection:
            self._ensure_schema(connection)
            connection.execute(
                "INSERT OR IGNORE INTO submissions "
                "(id, kind, source, payload, signature, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, 'unverified', ?)",
                (identifier, kind, source, _json(payload), signature, created),
            )
            row = connection.execute(
                "SELECT * FROM submissions WHERE id=?", (identifier,)
            ).fetchone()
        assert row is not None
        return _submission(row)

    def submit_internal(
        self,
        *,
        kind: str,
        origin: str,
        payload: Mapping[str, Any],
        submission_id: str | None = None,
        created_at: datetime | None = None,
    ) -> Submission:
        """Append an unsigned submission generated inside the daemon.

        Timer and event triggers are daemon-owned, so they carry no HMAC
        signature; promotion marks their task ``source`` as ``self`` while
        ``origin`` keeps the trigger channel.
        """

        if origin not in ("timer", "event"):
            raise ValueError(
                f"internal submission origin must be timer or event: {origin!r}"
            )
        return self.submit(
            kind=kind,
            source=origin,
            payload=payload,
            signature="",
            submission_id=submission_id,
            created_at=created_at,
        )

    def submission(self, submission_id: str) -> Submission | None:
        with self._connect() as connection:
            if not self._table_exists(connection, "submissions"):
                return None
            row = connection.execute(
                "SELECT * FROM submissions WHERE id=?", (submission_id,)
            ).fetchone()
        return None if row is None else _submission(row)

    @staticmethod
    def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
        row = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        return row is not None

    def pending_submissions(self, *, limit: int = 100) -> list[Submission]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM submissions WHERE status='unverified' "
                "ORDER BY created_at, id LIMIT ?",
                (limit,),
            ).fetchall()
        return [_submission(row) for row in rows]

    def mark_submission(
        self,
        submission_id: str,
        status: str,
        *,
        task_id: str | None = None,
        error: str | None = None,
        processed_at: datetime | None = None,
    ) -> None:
        if status not in SUBMISSION_STATUSES:
            raise ValueError(f"unsupported scheduler submission status: {status!r}")
        with self._connect() as connection:
            connection.execute(
                "UPDATE submissions SET status=?, processed_at=?, task_id=?, error=? "
                "WHERE id=?",
                (
                    status,
                    iso(processed_at or utc_now()),
                    task_id,
                    error,
                    submission_id,
                ),
            )

    def create_task(
        self,
        *,
        name: str,
        schedule: str,
        prompt: str,
        session: str,
        source: str,
        status: str,
        next_run_at: datetime | None,
        instructions: str = "",
        origin: str = "user",
        delivery: str = DEFAULT_DELIVERY,
        priority: str = DEFAULT_PRIORITY,
        created_at: datetime | None = None,
        task_id: str | None = None,
    ) -> Task:
        if status not in TASK_STATUSES:
            raise ValueError(f"unsupported scheduler task status: {status!r}")
        if origin not in TASK_ORIGINS:
            raise ValueError(f"unsupported scheduler task origin: {origin!r}")
        if delivery not in TASK_DELIVERIES:
            raise ValueError(f"unsupported scheduler task delivery: {delivery!r}")
        if priority not in TASK_PRIORITIES:
            raise ValueError(f"unsupported scheduler task priority: {priority!r}")
        now = created_at or utc_now()
        identifier = task_id or f"task-{uuid.uuid4().hex}"
        with self._connect() as connection:
            self._ensure_schema(connection)
            connection.execute(
                "INSERT INTO tasks "
                "(id, name, schedule, prompt, prompt_hash, session, source, status, "
                "next_run_at, created_at, updated_at, instructions, origin, delivery, "
                "priority) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    identifier,
                    name,
                    schedule,
                    prompt,
                    prompt_hash(prompt),
                    session,
                    source,
                    status,
                    None if next_run_at is None else iso(next_run_at),
                    iso(now),
                    iso(now),
                    instructions,
                    origin,
                    delivery,
                    priority,
                ),
            )
            row = connection.execute(
                "SELECT * FROM tasks WHERE id=?", (identifier,)
            ).fetchone()
        assert row is not None
        return _task(row)

    def task(self, task_id: str) -> Task | None:
        with self._connect() as connection:
            if not self._table_exists(connection, "tasks"):
                return None
            row = connection.execute(
                "SELECT * FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
        return None if row is None else _task(row)

    def list_tasks(self, *, status: str | None = None) -> list[Task]:
        query = "SELECT * FROM tasks"
        parameters: tuple[Any, ...] = ()
        if status is not None:
            query += " WHERE status=?"
            parameters = (status,)
        query += " ORDER BY created_at, id"
        with self._connect() as connection:
            if not self._table_exists(connection, "tasks"):
                return []
            rows = connection.execute(query, parameters).fetchall()
        return [_task(row) for row in rows]

    def set_task_status(
        self, task_id: str, status: str, *, updated_at: datetime | None = None
    ) -> None:
        if status not in TASK_STATUSES:
            raise ValueError(f"unsupported scheduler task status: {status!r}")
        with self._connect() as connection:
            connection.execute(
                "UPDATE tasks SET status=?, lease_id=NULL, lease_expires_at=NULL, "
                "updated_at=? WHERE id=?",
                (status, iso(updated_at or utc_now()), task_id),
            )

    def update_task_next_run(
        self,
        task_id: str,
        next_run_at: datetime | None,
        *,
        status: str | None = None,
        updated_at: datetime | None = None,
    ) -> None:
        if status is not None and status not in TASK_STATUSES:
            raise ValueError(f"unsupported scheduler task status: {status!r}")
        with self._connect() as connection:
            if status is None:
                connection.execute(
                    "UPDATE tasks SET next_run_at=?, updated_at=? WHERE id=?",
                    (
                        None if next_run_at is None else iso(next_run_at),
                        iso(updated_at or utc_now()),
                        task_id,
                    ),
                )
            else:
                connection.execute(
                    "UPDATE tasks SET next_run_at=?, status=?, updated_at=? WHERE id=?",
                    (
                        None if next_run_at is None else iso(next_run_at),
                        status,
                        iso(updated_at or utc_now()),
                        task_id,
                    ),
                )

    def claim_due_tasks(
        self,
        *,
        now: datetime,
        lease_id: str,
        lease_seconds: float,
        limit: int,
    ) -> list[Task]:
        if limit < 1:
            return []
        now_text = iso(now)
        lease_text = iso(datetime.fromtimestamp(now.timestamp() + lease_seconds, UTC))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT * FROM tasks WHERE status='approved' AND next_run_at IS NOT NULL "
                "AND next_run_at <= ? AND (lease_expires_at IS NULL OR lease_expires_at <= ?) "
                "ORDER BY CASE priority "
                "WHEN 'high' THEN 0 WHEN 'normal' THEN 1 WHEN 'low' THEN 2 ELSE 1 END, "
                "next_run_at, id LIMIT ?",
                (now_text, now_text, limit),
            ).fetchall()
            claimed: list[Task] = []
            for row in rows:
                updated = connection.execute(
                    "UPDATE tasks SET lease_id=?, lease_expires_at=?, updated_at=? "
                    "WHERE id=? AND (lease_expires_at IS NULL OR lease_expires_at <= ?)",
                    (lease_id, lease_text, now_text, row["id"], now_text),
                )
                if updated.rowcount:
                    refreshed = connection.execute(
                        "SELECT * FROM tasks WHERE id=?", (row["id"],)
                    ).fetchone()
                    if refreshed is not None:
                        claimed.append(_task(refreshed))
            connection.commit()
        return claimed

    def claim_task(
        self,
        task_id: str,
        *,
        now: datetime,
        lease_id: str,
        lease_seconds: float,
    ) -> Task | None:
        now_text = iso(now)
        lease_text = iso(datetime.fromtimestamp(now.timestamp() + lease_seconds, UTC))
        with self._connect() as connection:
            updated = connection.execute(
                "UPDATE tasks SET lease_id=?, lease_expires_at=?, updated_at=? "
                "WHERE id=? AND status='approved' "
                "AND (lease_expires_at IS NULL OR lease_expires_at <= ?)",
                (lease_id, lease_text, now_text, task_id, now_text),
            )
            if not updated.rowcount:
                return None
            row = connection.execute(
                "SELECT * FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
        return None if row is None else _task(row)

    def release_task(self, task_id: str, *, lease_id: str | None = None) -> None:
        with self._connect() as connection:
            if lease_id is None:
                connection.execute(
                    "UPDATE tasks SET lease_id=NULL, lease_expires_at=NULL, updated_at=? "
                    "WHERE id=?",
                    (iso(utc_now()), task_id),
                )
            else:
                connection.execute(
                    "UPDATE tasks SET lease_id=NULL, lease_expires_at=NULL, updated_at=? "
                    "WHERE id=? AND lease_id=?",
                    (iso(utc_now()), task_id, lease_id),
                )

    def delete_task(self, task_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM tasks WHERE id=?", (task_id,))

    def append_run(
        self,
        *,
        task_id: str,
        status: str,
        started_at: datetime,
        finished_at: datetime | None,
        exit_code: int | None,
        output: str,
        error: str | None,
        missed: bool,
        run_id: str | None = None,
    ) -> Run:
        if status not in RUN_STATUSES:
            raise ValueError(f"unsupported scheduler run status: {status!r}")
        identifier = run_id or f"run-{uuid.uuid4().hex}"
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO runs "
                "(id, task_id, started_at, finished_at, status, exit_code, output, error, missed) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    identifier,
                    task_id,
                    iso(started_at),
                    None if finished_at is None else iso(finished_at),
                    status,
                    exit_code,
                    output,
                    error,
                    1 if missed else 0,
                ),
            )
            row = connection.execute(
                "SELECT * FROM runs WHERE id=?", (identifier,)
            ).fetchone()
        assert row is not None
        return _run(row)

    def list_runs(self, *, task_id: str | None = None) -> list[Run]:
        query = "SELECT * FROM runs"
        parameters: tuple[Any, ...] = ()
        if task_id is not None:
            query += " WHERE task_id=?"
            parameters = (task_id,)
        query += " ORDER BY started_at, id"
        with self._connect() as connection:
            if not self._table_exists(connection, "runs"):
                return []
            rows = connection.execute(query, parameters).fetchall()
        return [_run(row) for row in rows]


__all__ = [
    "DEFAULT_DELIVERY",
    "DEFAULT_PRIORITY",
    "INTERNAL_SOURCES",
    "RUN_STATUSES",
    "SUBMISSION_SOURCES",
    "SUBMISSION_STATUSES",
    "TASK_DELIVERIES",
    "TASK_ORIGINS",
    "TASK_PRIORITIES",
    "TASK_STATUSES",
    "Run",
    "SchedulerStore",
    "Submission",
    "Task",
    "iso",
    "parse_iso",
    "prompt_hash",
    "scheduler_db_path",
    "scheduler_keys_dir",
    "submission_payload",
    "utc_now",
]
