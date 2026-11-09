from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pygent.lara.core.io import read_json, utc_now, validate_path_id


@dataclass(frozen=True, slots=True)
class SessionLocation:
    session_id: str
    lara_root: str
    workspace_root: str
    case_id: str
    mode: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class ItemLocation:
    item_id: str
    kind: str
    lara_root: str


def user_lara_root_for(lara_root: str | Path) -> Path:
    """Return the user-level Lara root that owns the given project root.

    Project roots follow ``<user root>/projects/<hash>`` (see
    :func:`lara.core.paths.project_lara_root`). Any other layout keeps the
    directory beside the project root so tests and custom homes stay
    self-contained.
    """
    project_root = Path(lara_root).resolve()
    if project_root.parent.name == "projects":
        return project_root.parent.parent
    return project_root.parent


class SessionDirectory:
    """Cross-project index of registered sessions and collaboration items.

    The directory is the addressing layer for collaboration: a session or
    collaboration item recorded here resolves to the project store that owns
    it, regardless of which workspace the caller runs in. Sessions register
    themselves on create, fork, and load; collaboration items register when
    they are enqueued.
    """

    def __init__(self, user_lara_root: str | Path) -> None:
        self.path = (
            Path(user_lara_root).resolve() / "index" / "session-directory-v1.sqlite3"
        )

    def register_project(self, *, lara_root: str, workspace_root: str) -> None:
        now = utc_now()
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO projects (lara_root, workspace_root, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(lara_root) DO UPDATE SET
                    workspace_root = excluded.workspace_root,
                    updated_at = excluded.updated_at
                """,
                (
                    str(Path(lara_root).resolve()),
                    str(Path(workspace_root).resolve()),
                    now,
                    now,
                ),
            )

    def register_session(
        self,
        *,
        session_id: str,
        lara_root: str,
        workspace_root: str,
        case_id: str,
        mode: str,
    ) -> None:
        validate_path_id(session_id, "session_id")
        resolved_root = str(Path(lara_root).resolve())
        self.register_project(
            lara_root=resolved_root,
            workspace_root=str(Path(workspace_root).resolve()),
        )
        now = utc_now()
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO session_locations (
                    session_id, lara_root, workspace_root,
                    case_id, mode, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    lara_root = excluded.lara_root,
                    workspace_root = excluded.workspace_root,
                    case_id = excluded.case_id,
                    mode = excluded.mode,
                    updated_at = excluded.updated_at
                """,
                (
                    session_id,
                    resolved_root,
                    str(Path(workspace_root).resolve()),
                    case_id,
                    mode,
                    now,
                ),
            )

    def locate_session(self, session_id: str) -> SessionLocation | None:
        validate_path_id(session_id, "session_id")
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM session_locations WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is not None:
            return self._session_location(row)
        # Sessions created before the directory existed are adopted on first
        # lookup by probing the on-disk layout.
        return self._adopt_session(session_id)

    def register_item(
        self,
        *,
        item_id: str,
        kind: str,
        lara_root: str,
        source_session_id: str | None,
        target_session_id: str | None,
    ) -> None:
        validate_path_id(item_id, "item_id")
        if kind not in {"operation", "message"}:
            raise ValueError(f"unknown collaboration item kind {kind!r}")
        now = utc_now()
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO collaboration_items (
                    item_id, kind, lara_root,
                    source_session_id, target_session_id, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(item_id) DO UPDATE SET
                    kind = excluded.kind,
                    lara_root = excluded.lara_root,
                    source_session_id = excluded.source_session_id,
                    target_session_id = excluded.target_session_id,
                    updated_at = excluded.updated_at
                """,
                (
                    item_id,
                    kind,
                    str(Path(lara_root).resolve()),
                    source_session_id,
                    target_session_id,
                    now,
                ),
            )

    def locate_item(self, item_id: str) -> ItemLocation | None:
        validate_path_id(item_id, "item_id")
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM collaboration_items WHERE item_id = ?",
                (item_id,),
            ).fetchone()
        if row is None:
            return None
        return ItemLocation(
            item_id=str(row["item_id"]),
            kind=str(row["kind"]),
            lara_root=str(row["lara_root"]),
        )

    def list_projects(self) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT p.lara_root, p.workspace_root, p.created_at, p.updated_at,
                       COUNT(s.session_id) AS session_count
                FROM projects p
                LEFT JOIN session_locations s ON s.lara_root = p.lara_root
                GROUP BY p.lara_root
                ORDER BY p.updated_at DESC
                """
            ).fetchall()
        return [
            {
                "lara_root": str(row["lara_root"]),
                "workspace_root": str(row["workspace_root"]),
                "session_count": int(row["session_count"]),
                "created_at": str(row["created_at"]),
                "updated_at": str(row["updated_at"]),
            }
            for row in rows
        ]

    def _adopt_session(self, session_id: str) -> SessionLocation | None:
        # Adoption follows the same on-disk contract as SessionManager:
        # a session is real when its metadata.json sits next to session.json.
        for metadata_path in self.path.parent.parent.glob(
            f"projects/*/sessions/{session_id}/metadata.json"
        ):
            if not (metadata_path.parent / "session.json").is_file():
                continue
            metadata = read_json(metadata_path)
            location = SessionLocation(
                session_id=session_id,
                lara_root=str(metadata_path.parents[2]),
                workspace_root=str(metadata.get("workspace_root") or ""),
                case_id=str(metadata.get("case_id") or ""),
                mode=str(metadata.get("mode") or ""),
                updated_at=utc_now(),
            )
            self.register_session(
                session_id=location.session_id,
                lara_root=location.lara_root,
                workspace_root=location.workspace_root,
                case_id=location.case_id,
                mode=location.mode,
            )
            return location
        return None

    @staticmethod
    def _session_location(row: sqlite3.Row) -> SessionLocation:
        return SessionLocation(
            session_id=str(row["session_id"]),
            lara_root=str(row["lara_root"]),
            workspace_root=str(row["workspace_root"]),
            case_id=str(row["case_id"]),
            mode=str(row["mode"]),
            updated_at=str(row["updated_at"]),
        )

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
                lara_root TEXT PRIMARY KEY,
                workspace_root TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS session_locations (
                session_id TEXT PRIMARY KEY,
                lara_root TEXT NOT NULL,
                workspace_root TEXT NOT NULL,
                case_id TEXT NOT NULL,
                mode TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS session_locations_project_idx"
            " ON session_locations(lara_root)"
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS collaboration_items (
                item_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                lara_root TEXT NOT NULL,
                source_session_id TEXT,
                target_session_id TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS collaboration_items_target_idx"
            " ON collaboration_items(target_session_id)"
        )
        return connection


__all__ = [
    "ItemLocation",
    "SessionDirectory",
    "SessionLocation",
    "user_lara_root_for",
]
