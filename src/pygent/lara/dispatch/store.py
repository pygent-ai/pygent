from __future__ import annotations

import sqlite3
import uuid
from contextlib import closing
from pathlib import Path

from pygent.lara.core.hashing import fingerprint as _fingerprint
from pygent.lara.core.io import validate_path_id

from .models import DispatchItem, DispatchItemKind, DispatchItemState, utc_now

LEGACY_MIGRATION_MARKER = "legacy_automations_migrated_at"


class DispatchStore:
    """User-level store for dispatch items and migrated task rows.

    Background start/fork submissions are recorded as items with task_id NULL;
    legacy automation tasks and their runs are imported on construction.
    """

    def __init__(self, path: str | Path, *, legacy_path: str | Path | None = None) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.legacy_path = (
            Path(legacy_path).expanduser().resolve()
            if legacy_path is not None
            else self.path.parent / "automations-v1.sqlite3"
        )
        self._initialize()
        self.migrate_legacy_automations()

    # ------------------------------------------------------------------ items

    def submit_item(
        self,
        *,
        kind: str,
        workspace_root: str,
        message: str,
        agent_alias: str = "",
        case_id: str = "automation",
        task_id: str | None = None,
        target_session_id: str | None = None,
        source_session_id: str | None = None,
        model_group: str | None = None,
        submission_id: str | None = None,
        fingerprint: str | None = None,
    ) -> DispatchItem:
        kind_value = DispatchItemKind(kind).value
        clean_message = _required(message, "message")
        workspace = str(Path(_required(workspace_root, "workspace_root")).resolve())
        clean_case_id = _required(case_id, "case_id")
        if model_group is not None and not model_group.strip():
            raise ValueError("model_group must be non-empty when provided")
        if submission_id is not None and not submission_id.strip():
            raise ValueError("submission_id must be non-empty")
        if target_session_id is not None:
            validate_path_id(target_session_id, "target_session_id")
        if source_session_id is not None:
            validate_path_id(source_session_id, "source_session_id")
        if submission_id is not None and fingerprint is None:
            fingerprint = _fingerprint(
                kind=kind_value,
                workspace_root=workspace,
                message=clean_message,
                agent_alias=agent_alias,
                case_id=clean_case_id,
                task_id=task_id,
                target_session_id=target_session_id,
                source_session_id=source_session_id,
                model_group=model_group,
            )
        item_id = f"op-{uuid.uuid4().hex}"
        now = utc_now()
        with closing(self._connect()) as connection:
            try:
                connection.execute(
                    """INSERT INTO dispatch_items(
                           item_id, task_id, kind, workspace_root, target_session_id,
                           source_session_id, agent_alias, case_id, model_group,
                           message, submission_id, fingerprint, state, created_at, updated_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)""",
                    (
                        item_id,
                        task_id,
                        kind_value,
                        workspace,
                        target_session_id,
                        source_session_id,
                        agent_alias,
                        clean_case_id,
                        model_group,
                        clean_message,
                        submission_id,
                        fingerprint,
                        now,
                        now,
                    ),
                )
                connection.commit()
            except sqlite3.IntegrityError:
                pass
            if submission_id is not None:
                row = connection.execute(
                    "SELECT * FROM dispatch_items WHERE submission_id = ?",
                    (submission_id,),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT * FROM dispatch_items WHERE item_id = ?", (item_id,)
                ).fetchone()
        if row is None:
            raise RuntimeError("dispatch item submission was not persisted")
        if (
            fingerprint is not None
            and row["fingerprint"] is not None
            and str(row["fingerprint"]) != fingerprint
        ):
                raise ValueError("submission_id was already used for a different request")
        return _item(row)

    def get_item(self, item_id: str) -> DispatchItem:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM dispatch_items WHERE item_id = ?", (item_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"dispatch item not found: {item_id}")
        return _item(row)

    def list_items_for_session(
        self, session_id: str, *, limit: int = 100
    ) -> list[DispatchItem]:
        """Items submitted by or targeting one session, newest first."""
        validate_path_id(session_id, "session_id")
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """SELECT * FROM dispatch_items
                   WHERE source_session_id=? OR target_session_id=?
                   ORDER BY rowid DESC LIMIT ?""",
                (session_id, session_id, max(1, min(limit, 200))),
            ).fetchall()
        return [_item(row) for row in rows]

    # -------------------------------------------------------------- migration

    def migrate_legacy_automations(self) -> bool:
        """Import the legacy user-level automation store exactly once.

        Task and run IDs are preserved; runs become kind='turn' items. A legacy
        run still marked running belongs to the old scheduler's lease, so it is
        requeued with no owner instead of freezing its task forever.
        """
        if not self.legacy_path.exists():
            return False
        with closing(self._connect()) as connection:
            marker = connection.execute(
                "SELECT value FROM dispatch_meta WHERE key=?", (LEGACY_MIGRATION_MARKER,)
            ).fetchone()
        if marker is not None:
            return False
        tasks, runs = _read_legacy_automations(self.legacy_path)
        last_run_ids = {
            run["automation_id"]: run["run_id"] for run in runs
        }
        task_by_id = {task["automation_id"]: task for task in tasks}
        with closing(self._connect()) as connection, connection:
            for task in tasks:
                connection.execute(
                    """INSERT OR IGNORE INTO dispatch_tasks(
                           task_id, name, prompt, workspace_root, destination,
                           target_session_id, timezone_name, at_time, rrule,
                           status, created_at, updated_at, last_run_id
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        task["automation_id"],
                        task["name"],
                        task["prompt"],
                        task["workspace_root"],
                        task["destination"],
                        task["target_session_id"],
                        task["timezone"],
                        task["at_time"],
                        task["rrule"],
                        task["status"],
                        task["created_at"],
                        task["updated_at"],
                        last_run_ids.get(task["automation_id"]),
                    ),
                )
            for run in runs:
                task = task_by_id[run["automation_id"]]
                state = run["status"]
                if state == DispatchItemState.RUNNING.value:
                    state = DispatchItemState.QUEUED.value
                connection.execute(
                    """INSERT OR IGNORE INTO dispatch_items(
                           item_id, task_id, kind, workspace_root, target_session_id,
                           source_session_id, agent_alias, case_id, model_group,
                           message, submission_id, fingerprint, state, owner_id,
                           lease_until_epoch, execution_id, case_run_id,
                           final_answer, error, created_at, updated_at
                       ) VALUES (?, ?, 'turn', ?, ?, NULL, '', 'automation', NULL,
                                 ?, NULL, NULL, ?, NULL, NULL, ?, ?, ?, ?, ?, ?)""",
                    (
                        run["run_id"],
                        run["automation_id"],
                        task["workspace_root"],
                        task["target_session_id"],
                        task["prompt"],
                        state,
                        run["execution_id"],
                        run["case_run_id"],
                        run["final_answer"] or "",
                        run["error"],
                        run["created_at"],
                        run["finished_at"] or run["created_at"],
                    ),
                )
            connection.execute(
                "INSERT INTO dispatch_meta(key, value) VALUES (?, ?)",
                (LEGACY_MIGRATION_MARKER, utc_now()),
            )
        return True

    # ---------------------------------------------------------------- internal

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS dispatch_tasks(
                    task_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    workspace_root TEXT NOT NULL,
                    destination TEXT NOT NULL,
                    target_session_id TEXT,
                    timezone_name TEXT,
                    at_time TEXT,
                    rrule TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_run_id TEXT
                );
                CREATE TABLE IF NOT EXISTS dispatch_items(
                    item_id TEXT PRIMARY KEY,
                    task_id TEXT REFERENCES dispatch_tasks(task_id) ON DELETE CASCADE,
                    kind TEXT NOT NULL,
                    workspace_root TEXT NOT NULL,
                    target_session_id TEXT,
                    source_session_id TEXT,
                    agent_alias TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    model_group TEXT,
                    message TEXT NOT NULL,
                    submission_id TEXT UNIQUE,
                    fingerprint TEXT,
                    state TEXT NOT NULL,
                    owner_id TEXT,
                    lease_until_epoch REAL,
                    execution_id TEXT,
                    case_run_id TEXT,
                    final_answer TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS dispatch_meta(
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_dispatch_items_task
                    ON dispatch_items(task_id, created_at);
                CREATE INDEX IF NOT EXISTS ix_dispatch_items_state
                    ON dispatch_items(state);
                """
            )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection


def _read_legacy_automations(
    legacy_path: Path,
) -> tuple[list[sqlite3.Row], list[sqlite3.Row]]:
    with closing(sqlite3.connect(legacy_path, timeout=10)) as connection:
        connection.row_factory = sqlite3.Row
        tasks = connection.execute("SELECT * FROM automations ORDER BY rowid").fetchall()
        runs = connection.execute("SELECT * FROM automation_runs ORDER BY rowid").fetchall()
    return tasks, runs


def _item(row: sqlite3.Row) -> DispatchItem:
    values = {field: row[field] for field in DispatchItem.__dataclass_fields__}
    values["state"] = DispatchItemState(str(values["state"]))
    values["lease_until_epoch"] = (
        None if values["lease_until_epoch"] is None else float(values["lease_until_epoch"])
    )
    values["final_answer"] = str(values["final_answer"] or "")
    return DispatchItem(**values)


def _required(value: str, name: str) -> str:
    clean = value.strip()
    if not clean:
        raise ValueError(f"{name} must be non-empty")
    return clean


__all__ = ["LEGACY_MIGRATION_MARKER", "DispatchStore"]
