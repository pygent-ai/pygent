from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pygent.agent.lara.graph.common import _write_json_atomic

from .models import BootstrapStatus, InitialSnapshot, ReminderScope


class ReminderStateStore:
    """Own the durable state of one session's reminder lifecycle."""

    @staticmethod
    def _root(scope: ReminderScope) -> Path:
        return scope.session_dir / "state" / "reminders"

    def load_bootstrap(self, scope: ReminderScope) -> dict[str, Any]:
        path = self._root(scope) / "bootstrap.json"
        if not path.exists():
            return {
                "schema_version": 1,
                "status": BootstrapStatus.PENDING.value,
                "snapshot_id": "",
                "prepared_at": "",
                "claimed_by_turn_id": "",
                "consumed_by_execution_id": "",
                "last_error": "",
            }
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            value = {}
        return {
            "schema_version": 1,
            "status": str(value.get("status") or BootstrapStatus.PENDING.value),
            "snapshot_id": str(value.get("snapshot_id") or ""),
            "prepared_at": str(value.get("prepared_at") or ""),
            "claimed_by_turn_id": str(value.get("claimed_by_turn_id") or ""),
            "consumed_by_execution_id": str(
                value.get("consumed_by_execution_id") or ""
            ),
            "last_error": str(value.get("last_error") or ""),
        }

    def save_bootstrap(self, scope: ReminderScope, state: dict[str, Any]) -> None:
        path = self._root(scope) / "bootstrap.json"
        _write_json_atomic(path, {"schema_version": 1, **state})

    def save_snapshot(self, scope: ReminderScope, snapshot: InitialSnapshot) -> None:
        path = self._root(scope) / "initial-snapshot.json"
        _write_json_atomic(
            path,
            {
                "schema_version": 1,
                "snapshot_id": snapshot.snapshot_id,
                "session_id": snapshot.session_id,
                "prepared_at": snapshot.prepared_at,
                "content": snapshot.content,
            },
        )

    def load_snapshot(self, scope: ReminderScope) -> InitialSnapshot:
        path = self._root(scope) / "initial-snapshot.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        return InitialSnapshot(
            snapshot_id=str(value["snapshot_id"]),
            session_id=str(value["session_id"]),
            prepared_at=str(value["prepared_at"]),
            content=str(value["content"]),
        )

    def load_observations(self, scope: ReminderScope) -> dict[str, Any]:
        path = self._root(scope) / "observations.json"
        if not path.exists():
            return {"schema_version": 1, "git": {}, "cli": {}, "skills": {}}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            value = {}
        return {
            "schema_version": 1,
            "git": dict(value.get("git") or {}),
            "cli": dict(value.get("cli") or {}),
            "skills": dict(value.get("skills") or {}),
        }

    def save_observations(self, scope: ReminderScope, state: dict[str, Any]) -> None:
        path = self._root(scope) / "observations.json"
        _write_json_atomic(path, {"schema_version": 1, **state})
