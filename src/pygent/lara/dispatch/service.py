from __future__ import annotations

from pathlib import Path

from pygent.lara.schema import RunConfig
from pygent.lara.sessions import SessionDirectory, SessionManager, user_lara_root_for

from .models import DispatchItem, DispatchItemKind
from .store import DispatchStore


class DispatchService:
    """Validated application operations shared by CLI, HTTP, and agent tools."""

    def __init__(self, store: DispatchStore) -> None:
        self.store = store

    def submit_start(
        self,
        *,
        config: RunConfig,
        message: str,
        source_session_id: str | None = None,
        submission_id: str | None = None,
        case_id: str = "collaboration",
        model_group: str | None = None,
    ) -> DispatchItem:
        """Submit one background agent start; the dispatch scheduler executes it."""
        if model_group is not None:
            self._require_model_group(config, model_group)
        if source_session_id is not None:
            self._resolve_session(config, source_session_id)
        return self.store.submit_item(
            kind=DispatchItemKind.START.value,
            workspace_root=str(Path(config.workspace_root).resolve()),
            message=message,
            agent_alias=config.agent_alias,
            case_id=case_id,
            source_session_id=source_session_id,
            model_group=model_group,
            submission_id=submission_id,
        )

    def submit_fork(
        self,
        *,
        config: RunConfig,
        fork_from: str,
        message: str,
        submission_id: str | None = None,
        case_id: str = "collaboration",
        model_group: str | None = None,
    ) -> DispatchItem:
        """Submit one background fork; the fork source must be project-local."""
        if model_group is not None:
            self._require_model_group(config, model_group)
        # The fork copies session files, so the source has to be in this project.
        SessionManager(config).load(fork_from)
        return self.store.submit_item(
            kind=DispatchItemKind.FORK.value,
            workspace_root=str(Path(config.workspace_root).resolve()),
            message=message,
            agent_alias=config.agent_alias,
            case_id=case_id,
            source_session_id=fork_from,
            model_group=model_group,
            submission_id=submission_id,
        )

    @staticmethod
    def _require_model_group(config: RunConfig, model_group: str) -> None:
        clean = model_group.strip()
        if not clean:
            raise ValueError("model_group must be non-empty when provided")
        native = config.model_config
        if native is None or clean not in native.model_groups:
            raise ValueError(f"unknown model group {clean!r}")

    @staticmethod
    def _resolve_session(config: RunConfig, session_id: str) -> None:
        """Resolve the project owning a session; unknown sessions fail closed."""
        try:
            SessionManager(config).load(session_id)
        except FileNotFoundError:
            directory = SessionDirectory(user_lara_root_for(config.lara_root))
            if directory.locate_session(session_id) is None:
                raise


__all__ = ["DispatchService"]
