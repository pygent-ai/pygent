from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any

from pygent.agent.pygent_v1.core.io import (
    append_jsonl,
    read_json,
    utc_now,
    validate_path_id,
    write_json,
    write_json_atomic,
)
from pygent.agent.pygent_v1.core.process import current_owner
from pygent.agent.pygent_v1.core.redaction import redact_secrets
from pygent.agent.pygent_v1.schema import (
    AgentSession,
    CaseRunRef,
    CaseRunStatus,
    RunConfig,
    SessionRef,
)

# The dedicated session where realtime silent-work interaction lives: the
# agent works silently and ``speak`` is the only user-visible channel.
PYGENT_SAYS_SESSION_MODE = "pygent.v1-says"


def realtime_enabled_for(session: AgentSession) -> bool:
    """Realtime interaction is exclusive to the ``pygent.v1-says`` session.

    Every other session keeps the live-work behavior regardless of the
    realtime runtime config, so the dedicated sidebar entry is the only
    place speak bubbles, fork replies and nudges exist.
    """

    return str(session.metadata.get("mode") or "") == PYGENT_SAYS_SESSION_MODE


class SessionManager:
    def __init__(self, config: RunConfig):
        self.config = config
        self.workspace_root = Path(config.workspace_root)
        self.pygent_root = Path(config.pygent_root)
        self.sessions_root = self.pygent_root / "sessions"

    def create(
        self,
        case_id: str,
        mode: str = "e2e",
        *,
        model_group_name: str | None = None,
        selected_model_key: str | None = None,
        eternal_memory: bool = False,
    ) -> SessionRef:
        validate_path_id(case_id, "case_id")
        selected_group, selected_model = self._selection_for_group(model_group_name)
        # Validate before any directory is created so a bad model key cannot
        # leave an orphan session behind.
        if selected_model_key:
            self._require_group_model(selected_group, selected_model_key)
            selected_model = selected_model_key
        session_id = (
            PYGENT_SAYS_SESSION_MODE
            if mode == PYGENT_SAYS_SESSION_MODE
            else self._new_session_id(case_id, mode)
        )
        session_dir = self.sessions_root / session_id
        session_dir.mkdir(parents=True, exist_ok=False)
        for child in ("context", "state", "logs", "cases"):
            (session_dir / child).mkdir(parents=True, exist_ok=True)

        now = utc_now()
        session = AgentSession(
            session_id=session_id,
            workspace_root=str(self.workspace_root),
            session_dir=str(session_dir),
            created_at=now,
            updated_at=now,
            model_group_name=selected_group,
            selected_model_key=selected_model,
            metadata={
                "active_case_id": case_id,
                "mode": mode,
                "eternal_memory": eternal_memory,
            },
        )
        write_json_atomic(session_dir / "session.json", session.to_dict())
        write_json(
            session_dir / "metadata.json",
            {
                "session_id": session_id,
                "case_id": case_id,
                "mode": mode,
                "created_at": now,
                "updated_at": now,
                "workspace_root": str(self.workspace_root),
                "model_group_name": selected_group,
                "selected_model_key": selected_model,
            },
        )
        return SessionRef(
            session_id=session_id,
            session_dir=str(session_dir),
            workspace_root=str(self.workspace_root),
        )

    def load(self, session_id: str) -> AgentSession:
        session_dir = self._session_dir(session_id)
        path = session_dir / "session.json"
        if not path.exists():
            raise FileNotFoundError(f"Session {session_id!r} does not exist")
        data = read_json(path)
        data["session_dir"] = str(session_dir)
        session = AgentSession.from_dict(data)
        self._resolve_missing_selection(session)
        session = self._apply_history_checkpoints(session)
        return session

    def model_selection(self, session_id: str) -> tuple[str, str]:
        session = self.load(session_id)
        if not session.model_group_name or not session.selected_model_key:
            raise RuntimeError("model_configuration_required")
        return session.model_group_name, session.selected_model_key

    def _require_group_model(self, group_name: str, model_key: str) -> None:
        native = self.config.model_config
        if native is None or not group_name:
            raise RuntimeError("model_configuration_required")
        group = native.model_groups.get(group_name)
        if group is None:
            raise ValueError(
                f"unknown fixed model group {group_name!r}"
            )
        if model_key not in {entry.key for entry in group.models}:
            raise ValueError(
                f"model {model_key!r} is not in fixed model group "
                f"{group_name!r}"
            )

    def set_eternal_memory(self, session_id: str, *, enabled: bool) -> AgentSession:
        """Opt one session into (or out of) the eternal conversation."""
        session = self.load(session_id)
        session.metadata["eternal_memory"] = enabled
        self.save(session)
        return session

    def fork(
        self,
        source_session_id: str,
        *,
        mode: str = "fork",
        at_sequence: int | None = None,
    ) -> SessionRef:
        source_dir = self._session_dir(source_session_id)
        if not source_dir.exists():
            raise FileNotFoundError(
                f"Source session {source_session_id!r} does not exist"
            )
        # Validate the boundary before creating anything: a failed fork must
        # not leave an orphan target session behind.
        if at_sequence is not None:
            self._require_checkpoint_sequence(source_session_id, at_sequence)
        source_meta = read_json(source_dir / "metadata.json")
        case_id = source_meta.get("case_id", "fork")
        target = self.create(case_id=case_id, mode=mode)
        target_dir = Path(target.session_dir)
        target_session = self.load(target.session_id)
        # A fork is a semantic continuation at the source cursor. Copy every
        # durable conversation layer, not only the foreground context mirror.
        for name in (
            "session.json",
            "context",
            "state",
            "memory",
            "raw-history",
            "agent-history",
        ):
            src = source_dir / name
            dst = target_dir / name
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            elif src.exists():
                shutil.copy2(src, dst)
        # A source with an active turn keeps appending checkpoints while the
        # files are copied, so the durable conversation must come from a
        # consistent backup snapshot rather than a possibly torn file copy.
        self._snapshot_checkpoint_database(source_session_id, target.session_id)
        data = read_json(target_dir / "session.json")
        data.update(
            {
                "session_id": target.session_id,
                "session_dir": str(target_dir),
                "created_at": target_session.created_at,
                "updated_at": utc_now(),
            }
        )
        session = AgentSession.from_dict(data)
        session.metadata.update({"forked_from": source_session_id, "mode": mode})
        reminder_state = target_dir / "state" / "reminders"
        if reminder_state.exists():
            shutil.rmtree(reminder_state)
        self._save_session(session)
        if at_sequence is not None:
            self._truncate_fork_history(target.session_id, at_sequence)
        return target

    def _require_checkpoint_sequence(
        self, session_id: str, at_sequence: int
    ) -> None:
        database_path = self._checkpoint_database_path(session_id)
        if not database_path.exists():
            raise ValueError(
                f"Checkpoint sequence {at_sequence!r} does not exist "
                f"in session {session_id!r}"
            )
        with closing(self._checkpoint_connection(database_path)) as connection:
            row = connection.execute(
                "SELECT 1 FROM conversation_checkpoints WHERE sequence = ?",
                (at_sequence,),
            ).fetchone()
        if row is None:
            raise ValueError(
                f"Checkpoint sequence {at_sequence!r} does not exist "
                f"in session {session_id!r}"
            )

    def _truncate_fork_history(self, session_id: str, at_sequence: int) -> None:
        """Rewind a freshly forked session to one checkpoint boundary.

        The copied checkpoint database must lose the rows after the boundary:
        otherwise the next load() would resurrect them past the rewound
        history_checkpoint_seq cursor.
        """
        session = self.load(session_id)
        database_path = self._checkpoint_database_path(session_id)
        with closing(self._checkpoint_connection(database_path)) as connection, connection:
            rows = connection.execute(
                """
                SELECT message_json FROM conversation_checkpoints
                WHERE sequence <= ? AND include_in_session = 1
                ORDER BY sequence
                """,
                (at_sequence,),
            ).fetchall()
            connection.execute(
                "DELETE FROM conversation_checkpoints WHERE sequence > ?",
                (at_sequence,),
            )
        session.history = [json.loads(row[0]) for row in rows]
        session.metadata["history_checkpoint_seq"] = at_sequence
        session.updated_at = utc_now()
        self._save_session(session)

    def _snapshot_checkpoint_database(
        self, source_session_id: str, target_session_id: str
    ) -> None:
        """Replace the copied checkpoint database with a consistent snapshot.

        A raw file copy can tear against a running source: its WAL keeps
        accepting committed frames while ``context`` is copied, and copying
        the main file and the WAL at different moments can drop rows or
        resurrect stale pages. The backup API reads one committed state even
        while the source keeps writing.
        """
        source_path = self._checkpoint_database_path(source_session_id)
        if not source_path.exists():
            return
        target_path = self._checkpoint_database_path(target_session_id)
        # Drop the raw copy first: a torn header would fail backup
        # initialization instead of being replaced.
        target_path.unlink(missing_ok=True)
        for suffix in ("-wal", "-shm"):
            target_path.with_name(target_path.name + suffix).unlink(missing_ok=True)
        with closing(sqlite3.connect(source_path, timeout=30)) as source, closing(
            sqlite3.connect(target_path, timeout=30)
        ) as target:
            source.backup(target)

    def start_case_run(
        self, session_id: str, case_id: str, run_config: RunConfig | None = None
    ) -> CaseRunRef:
        validate_path_id(case_id, "case_id")
        session = self.load(session_id)
        case_run_id = self._new_case_run_id(session_id, case_id)
        run_dir = (
            self._session_dir(session_id) / "cases" / case_id / "runs" / case_run_id
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        ref = CaseRunRef(
            session_id=session_id,
            case_id=case_id,
            case_run_id=case_run_id,
            run_dir=str(run_dir),
        )
        config = run_config or self.config
        write_json(run_dir / "run_config.json", config.to_dict())
        write_json_atomic(
            run_dir / "run_metadata.json",
            {
                "status": "running",
                "started_at": utc_now(),
                "history_start_index": len(session.history),
                "owner": current_owner(),
                **ref.to_dict(),
            },
        )
        metadata_path = self._session_dir(session_id) / "metadata.json"
        metadata = read_json(metadata_path)
        metadata.update(
            last_case_run_id=case_run_id,
            last_case_run_status="running",
            updated_at=utc_now(),
        )
        write_json_atomic(metadata_path, metadata)
        self._append_session_event(
            session_id, "case.started", {"case_id": case_id, "case_run_id": case_run_id}
        )
        return ref

    def finish_case_run(
        self, case_run_ref: CaseRunRef, status: str, *, error: str | None = None
    ) -> bool:
        verdict = CaseRunStatus.from_value(status)
        if verdict is None or not verdict.terminal:
            raise ValueError("status must be one of passed, failed, error, skipped")
        run_dir = Path(case_run_ref.run_dir)
        metadata_path = run_dir / "run_metadata.json"
        metadata = read_json(metadata_path)
        if metadata.get("status") != "running":
            # A case run has exactly one terminal. A second finish call means a
            # lifecycle bug elsewhere (for example a peer host reconciling a run
            # it does not own); keep the first verdict and the single terminal
            # session event the UI consumes instead of regressing the outcome.
            return False
        metadata.update({"status": status, "finished_at": utc_now()})
        if error is not None:
            metadata["error"] = error
        write_json_atomic(metadata_path, metadata)
        metadata_path = self._session_dir(case_run_ref.session_id) / "metadata.json"
        session_metadata = read_json(metadata_path)
        if session_metadata.get("last_case_run_id") in (None, case_run_ref.case_run_id):
            # Finishing the latest run moves the session pointers; finishing an
            # older run (stale-run reconcile after a crashed host) must not
            # regress them onto the already-superseded run. metadata.json is
            # the authoritative pointer: start_case_run updates it directly.
            session = self.load(case_run_ref.session_id)
            session.updated_at = utc_now()
            session.metadata.update(
                {
                    "active_case_id": case_run_ref.case_id,
                    "last_case_run_id": case_run_ref.case_run_id,
                    "last_case_run_status": status,
                }
            )
            self._save_session(session)
            session_metadata.update(
                {
                    "updated_at": session.updated_at,
                    "last_case_run_id": case_run_ref.case_run_id,
                    "last_case_run_status": status,
                }
            )
            write_json(metadata_path, session_metadata)
        self._append_session_event(
            case_run_ref.session_id,
            "case.finished",
            {
                "case_id": case_run_ref.case_id,
                "case_run_id": case_run_ref.case_run_id,
                "status": status,
                "error": error,
            },
        )
        return True

    def show(self, session_id: str) -> dict[str, Any]:
        session = self.load(session_id)
        metadata = read_json(self._session_dir(session_id) / "metadata.json")
        return {"session": session.to_dict(), "metadata": metadata}

    def list_sessions(self, *, mode: str | None = None) -> list[dict[str, Any]]:
        if not self.sessions_root.exists():
            return []
        records: list[dict[str, Any]] = []
        for metadata_path in self.sessions_root.glob("*/metadata.json"):
            if not (metadata_path.parent / "session.json").is_file():
                continue
            metadata = read_json(metadata_path)
            session_mode = str(metadata.get("mode") or "")
            if mode is not None and session_mode != mode:
                continue
            session_id = str(metadata.get("session_id") or metadata_path.parent.name)
            records.append(
                {
                    "session_id": session_id,
                    "case_id": str(metadata.get("case_id") or ""),
                    "mode": session_mode,
                    "title": str(metadata.get("title") or session_id),
                    "created_at": str(metadata.get("created_at") or ""),
                    "updated_at": str(metadata.get("updated_at") or ""),
                    "last_case_run_id": metadata.get("last_case_run_id"),
                    "last_case_run_status": metadata.get("last_case_run_status"),
                }
            )
        return sorted(
            records, key=lambda record: str(record["updated_at"]), reverse=True
        )

    def save_title_from_user_input(self, session_id: str, user_input: str) -> None:
        """Persist the first meaningful user message as the session title."""

        session = self.load(session_id)
        metadata_path = Path(session.session_dir) / "metadata.json"
        metadata = read_json(metadata_path)
        if _clean_title(str(metadata.get("title") or "")):
            return
        title = _clean_title(user_input)
        if not title:
            return
        metadata["title"] = title[:120]
        write_json(metadata_path, metadata)

    def save_title(self, session_id: str, title: str) -> None:
        """Persist an explicit application-owned session title."""

        session = self.load(session_id)
        clean = _clean_title(title)
        if not clean:
            raise ValueError("title must be non-empty")
        metadata_path = Path(session.session_dir) / "metadata.json"
        metadata = read_json(metadata_path)
        metadata["title"] = clean[:120]
        metadata["updated_at"] = utc_now()
        write_json_atomic(metadata_path, metadata)

    def save(self, session: AgentSession) -> None:
        session.updated_at = utc_now()
        self._apply_history_checkpoints(session)
        self._save_session(session)

    def append_history_checkpoint(
        self,
        case_run_ref: CaseRunRef,
        *,
        turn_id: str | None,
        checkpoint_id: str,
        message: dict[str, Any],
        include_in_session: bool = True,
    ) -> bool:
        """Durably append one raw conversation boundary exactly once."""

        validate_path_id(case_run_ref.session_id, "session_id")
        database_path = self._checkpoint_database_path(case_run_ref.session_id)
        with closing(self._checkpoint_connection(database_path)) as connection, connection:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO conversation_checkpoints (
                        checkpoint_id,
                        session_id,
                        case_id,
                        case_run_id,
                        turn_id,
                        role,
                        message_json,
                        include_in_session,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        checkpoint_id,
                        case_run_ref.session_id,
                        case_run_ref.case_id,
                        case_run_ref.case_run_id,
                        turn_id,
                        str(message.get("role") or "user"),
                        json.dumps(
                            redact_secrets(message), ensure_ascii=False, sort_keys=True
                        ),
                        int(include_in_session),
                        utc_now(),
                    ),
                )
                return cursor.rowcount == 1

    def _save_session(self, session: AgentSession) -> None:
        validate_path_id(session.session_id, "session_id")
        session_dir = self._session_dir(session.session_id)
        write_json_atomic(
            session_dir / "session.json", redact_secrets(session.to_dict())
        )
        metadata_path = session_dir / "metadata.json"
        if metadata_path.exists():
            metadata = read_json(metadata_path)
            metadata.update(
                {
                    "updated_at": session.updated_at,
                    "model_group_name": session.model_group_name,
                    "selected_model_key": session.selected_model_key,
                }
            )
            write_json_atomic(metadata_path, metadata)

    def _selection_for_group(
        self, group_name: str | None
    ) -> tuple[str, str]:
        resolved = self.config.resolved_agent
        native = self.config.model_config
        if resolved is None or native is None:
            if group_name:
                raise RuntimeError("model_configuration_required")
            return "", ""
        selected_group = group_name or resolved.default_model_group
        group = native.model_groups.get(selected_group)
        if group is None:
            raise ValueError(f"unknown model group {selected_group!r}")
        return selected_group, group.models[0].key

    def _resolve_missing_selection(self, session: AgentSession) -> None:
        if session.model_group_name and session.selected_model_key:
            return
        group_name, model_key = self._selection_for_group(None)
        session.model_group_name = group_name
        session.selected_model_key = model_key

    def _apply_history_checkpoints(self, session: AgentSession) -> AgentSession:
        database_path = self._checkpoint_database_path(session.session_id)
        if not database_path.exists():
            return session
        cursor_sequence = int(session.metadata.get("history_checkpoint_seq") or 0)
        with closing(self._checkpoint_connection(database_path)) as connection:
            rows = connection.execute(
                """
                SELECT sequence, message_json, include_in_session
                FROM conversation_checkpoints
                WHERE sequence > ?
                ORDER BY sequence
                """,
                (cursor_sequence,),
            ).fetchall()
        for sequence, message_json, include_in_session in rows:
            cursor_sequence = int(sequence)
            if include_in_session:
                message = json.loads(message_json)
                session.history.append(message)
        if rows:
            session.metadata["history_checkpoint_seq"] = cursor_sequence
        return session

    def _checkpoint_database_path(self, session_id: str) -> Path:
        return (
            self._session_dir(session_id)
            / "context"
            / "conversation-checkpoints.sqlite3"
        )

    @staticmethod
    def _checkpoint_connection(path: Path) -> sqlite3.Connection:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_checkpoints (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                checkpoint_id TEXT NOT NULL UNIQUE,
                session_id TEXT NOT NULL,
                case_id TEXT NOT NULL,
                case_run_id TEXT NOT NULL,
                turn_id TEXT,
                role TEXT NOT NULL,
                message_json TEXT NOT NULL,
                include_in_session INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            )
            """
        )
        return connection

    def _session_dir(self, session_id: str) -> Path:
        validate_path_id(session_id, "session_id")
        session_dir = (self.sessions_root / session_id).resolve()
        if not session_dir.is_relative_to(self.sessions_root.resolve()):
            raise ValueError("session_id escapes sessions root")
        return session_dir

    def _new_session_id(self, case_id: str, mode: str) -> str:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # noqa: DTZ005 - ids use local-time stamps
        digest = hashlib.sha1(
            f"{case_id}:{mode}:{stamp}:{utc_now()}".encode()
        ).hexdigest()[:6]
        return f"{mode}-{_slug(case_id)}-{stamp}-{digest}"

    def _new_case_run_id(self, session_id: str, case_id: str) -> str:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")  # noqa: DTZ005 - ids use local-time stamps
        digest = hashlib.sha1(
            f"{session_id}:{case_id}:{stamp}".encode()
        ).hexdigest()[:6]
        return f"run-{stamp}-{digest}"

    def _append_session_event(
        self, session_id: str, event_type: str, payload: dict[str, Any]
    ) -> None:
        path = self._session_dir(session_id) / "logs" / "session-events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "type": event_type,
            "timestamp": utc_now(),
            "session_id": session_id,
            "payload": payload,
        }
        append_jsonl(path, event)


def _slug(value: str) -> str:
    clean = "".join(ch if ch.isalnum() else "-" for ch in value.lower()).strip("-")
    return clean or "case"


def _clean_title(value: str) -> str:
    return " ".join(value.split())
