from __future__ import annotations

import sqlite3
import time
import uuid
from contextlib import closing
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from pygent.lara.core.io import utc_now, validate_path_id


class AgentMessageState(StrEnum):
    QUEUED = "queued"
    CLAIMED = "claimed"
    DELIVERED = "delivered"


@dataclass(frozen=True, slots=True)
class AgentMessage:
    message_id: str
    submission_id: str
    source_session_id: str | None
    source_agent_alias: str | None
    target_session_id: str
    content: str
    state: AgentMessageState
    created_at: str
    updated_at: str
    delivered_at: str | None = None
    delivered_execution_id: str | None = None

    @property
    def done(self) -> bool:
        return self.state is AgentMessageState.DELIVERED

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AgentMessage:
        return cls(
            message_id=str(value["message_id"]),
            submission_id=str(value["submission_id"]),
            source_session_id=(
                None
                if value.get("source_session_id") is None
                else str(value["source_session_id"])
            ),
            source_agent_alias=(
                None
                if value.get("source_agent_alias") is None
                else str(value["source_agent_alias"])
            ),
            target_session_id=str(value["target_session_id"]),
            content=str(value.get("content") or ""),
            state=AgentMessageState(str(value["status"])),
            created_at=str(value["created_at"]),
            updated_at=str(value["updated_at"]),
            delivered_at=(
                None
                if value.get("delivered_at") is None
                else str(value["delivered_at"])
            ),
            delivered_execution_id=(
                None
                if value.get("delivered_execution_id") is None
                else str(value["delivered_execution_id"])
            ),
        )


class SessionCollaborationStore:
    """SQLite agent-message inbox with idempotency and target-session claims."""

    def __init__(self, lara_root: str | Path) -> None:
        self.path = (
            Path(lara_root).resolve() / "runtime" / "session-collaboration-v1.sqlite3"
        )

    def enqueue_message(
        self,
        *,
        submission_id: str,
        fingerprint: str,
        source_session_id: str | None,
        source_agent_alias: str | None,
        target_session_id: str,
        content: str,
    ) -> AgentMessage:
        if not submission_id.strip():
            raise ValueError("submission_id must be non-empty")
        if not content.strip():
            raise ValueError("agent message content must be non-empty")
        validate_path_id(target_session_id, "target_session_id")
        if source_session_id is not None:
            validate_path_id(source_session_id, "source_session_id")
        message_id = f"msg-{uuid.uuid4().hex}"
        now = utc_now()
        with closing(self._connect()) as connection, connection:
            row = self._enqueue_message(
                connection,
                message_id=message_id,
                submission_id=submission_id,
                fingerprint=fingerprint,
                source_session_id=source_session_id,
                source_agent_alias=source_agent_alias,
                target_session_id=target_session_id,
                content=content,
                now=now,
            )
        return self._agent_message(row)

    def get_message(self, message_id: str) -> AgentMessage:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM agent_messages WHERE message_id = ?",
                (message_id,),
            ).fetchone()
        if row is None:
            raise FileNotFoundError(f"Agent message {message_id!r} does not exist")
        return self._agent_message(row)

    def get_claimed_message(
        self, message_id: str, *, claim_id: str
    ) -> AgentMessage | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT * FROM agent_messages
                WHERE message_id = ? AND state = 'claimed' AND claim_id = ?
                """,
                (message_id, claim_id),
            ).fetchone()
        return None if row is None else self._agent_message(row)

    def claim_messages(
        self,
        target_session_id: str,
        *,
        claim_id: str,
        lease_seconds: float,
        limit: int = 16,
    ) -> list[AgentMessage]:
        validate_path_id(target_session_id, "target_session_id")
        if limit <= 0:
            return []
        now_epoch = time.time()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT * FROM agent_messages
                WHERE target_session_id = ? AND (
                    state = 'queued'
                    OR (state = 'claimed' AND claim_until_epoch <= ?)
                    OR (state = 'claimed' AND claim_id = ?)
                )
                ORDER BY rowid
                LIMIT ?
                """,
                (target_session_id, now_epoch, claim_id, limit),
            ).fetchall()
            message_ids = [str(row["message_id"]) for row in rows]
            if message_ids:
                placeholders = ", ".join("?" for _ in message_ids)
                connection.execute(
                    f"""
                    UPDATE agent_messages
                    SET state = 'claimed', claim_id = ?, claim_until_epoch = ?, updated_at = ?
                    WHERE message_id IN ({placeholders})
                    """,
                    (claim_id, now_epoch + lease_seconds, utc_now(), *message_ids),
                )
            connection.commit()
        return [self.get_message(message_id) for message_id in message_ids]

    def pending_messages(
        self,
        target_session_id: str,
        *,
        limit: int = 16,
    ) -> list[AgentMessage]:
        """Queued or lease-expired agent messages, without claiming them.

        A read-only peek for delivery decisions: a message under a live lease
        belongs to another claim, and delivered ones are done.
        """

        now_epoch = time.time()
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT * FROM agent_messages
                WHERE target_session_id = ? AND (
                    state = 'queued'
                    OR (state = 'claimed' AND claim_until_epoch <= ?)
                )
                ORDER BY rowid
                LIMIT ?
                """,
                (target_session_id, now_epoch, limit),
            ).fetchall()
        return [self._agent_message(row) for row in rows]

    def acknowledge_message(
        self,
        message_id: str,
        *,
        claim_id: str,
        execution_id: str,
    ) -> None:
        now = utc_now()
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """
                UPDATE agent_messages
                SET state = 'delivered', delivered_at = ?,
                    delivered_execution_id = ?, claim_id = NULL,
                    claim_until_epoch = NULL, updated_at = ?
                WHERE message_id = ? AND (
                    (state = 'claimed' AND claim_id = ?)
                    OR (state = 'delivered' AND delivered_execution_id = ?)
                )
                """,
                (now, execution_id, now, message_id, claim_id, execution_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("agent message claim was lost")

    def release_message(self, message_id: str, *, claim_id: str) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                UPDATE agent_messages
                SET state = 'queued', claim_id = NULL,
                    claim_until_epoch = NULL, updated_at = ?
                WHERE message_id = ? AND state = 'claimed' AND claim_id = ?
                """,
                (utc_now(), message_id, claim_id),
            )

    @staticmethod
    def _enqueue_message(
        connection: sqlite3.Connection,
        *,
        message_id: str,
        submission_id: str,
        fingerprint: str,
        source_session_id: str | None,
        source_agent_alias: str | None,
        target_session_id: str,
        content: str,
        now: str,
    ) -> sqlite3.Row:
        try:
            connection.execute(
                """
                INSERT INTO agent_messages (
                    message_id, submission_id, fingerprint,
                    source_session_id, source_agent_alias,
                    target_session_id, content, state, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)
                """,
                (
                    message_id,
                    submission_id,
                    fingerprint,
                    source_session_id,
                    source_agent_alias,
                    target_session_id,
                    content,
                    now,
                    now,
                ),
            )
        except sqlite3.IntegrityError:
            pass
        row = connection.execute(
            "SELECT * FROM agent_messages WHERE submission_id = ?",
            (submission_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("agent message reservation was not persisted")
        if str(row["fingerprint"]) != fingerprint:
            raise ValueError("submission_id was already used for a different message")
        return row

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            "PRAGMA synchronous=FULL"
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_messages (
                message_id TEXT PRIMARY KEY,
                submission_id TEXT NOT NULL UNIQUE,
                fingerprint TEXT NOT NULL,
                source_session_id TEXT,
                source_agent_alias TEXT,
                target_session_id TEXT NOT NULL,
                content TEXT NOT NULL,
                state TEXT NOT NULL,
                claim_id TEXT,
                claim_until_epoch REAL,
                delivered_at TEXT,
                delivered_execution_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS agent_messages_target_state_idx ON agent_messages(target_session_id, state, claim_until_epoch)"
        )
        return connection

    @staticmethod
    def _agent_message(row: sqlite3.Row) -> AgentMessage:
        value = dict(row)
        value["status"] = value.pop("state")
        return AgentMessage.from_dict(value)


__all__ = [
    "AgentMessage",
    "AgentMessageState",
    "SessionCollaborationStore",
]
