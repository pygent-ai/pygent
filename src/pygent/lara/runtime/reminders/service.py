from __future__ import annotations

import asyncio
from pathlib import Path

from pygent.lara.core.io import read_json
from pygent.lara.schema import RunConfig
from pygent.lara.sessions import AgentMessage, SessionCollaborationStore

from .bootstrap import build_initial_snapshot
from .cli_context import detect_cli_changes
from .git_context import detect_git_change
from .models import BootstrapStatus, ReminderScope, ReminderSection
from .rendering import render_context_body
from .skills_context import detect_skill_changes
from .store import ReminderStateStore


class ReminderService:
    """Session bootstrap, runtime observations, and the Agent-message inbox."""

    def __init__(
        self,
        config: RunConfig,
        *,
        store: ReminderStateStore | None = None,
        message_store: SessionCollaborationStore | None = None,
    ) -> None:
        self.config = config
        self.store = store or ReminderStateStore()
        self.message_store = message_store or SessionCollaborationStore(
            config.lara_root
        )
        self._preparations: dict[str, asyncio.Task[None]] = {}
        self._observations: dict[str, asyncio.Task[ReminderSection | None]] = {}
        self._queued_observations: set[str] = set()
        self._locks: dict[str, asyncio.Lock] = {}
        self._closed = False

    def scope(self, session_id: str) -> ReminderScope:
        session_dir = Path(self.config.lara_root) / "sessions" / session_id
        user_lara_root = Path(self.config.user_lara_root or Path.home() / ".lara")
        project_lara_root = Path(self.config.lara_root)
        return ReminderScope(
            session_id=session_id,
            session_dir=session_dir,
            workspace_root=Path(self.config.workspace_root),
            cli_bash_presets=tuple(self.config.cli_bash_presets),
            user_skills_dir=(user_lara_root / "skills").expanduser().resolve(),
            project_skills_dir=(project_lara_root / "skills").expanduser().resolve(),
        )

    def prewarm_session(self, session_id: str) -> asyncio.Task[None] | None:
        if self._closed:
            raise RuntimeError("ReminderService is closed")
        scope = self.scope(session_id)
        state = self.store.load_bootstrap(scope)
        status = state["status"]
        if status in {
            BootstrapStatus.READY.value,
            BootstrapStatus.CLAIMED.value,
            BootstrapStatus.CONSUMED.value,
        }:
            return None
        if status == BootstrapStatus.PENDING.value and self._has_existing_user_history(
            scope
        ):
            state["status"] = BootstrapStatus.CONSUMED.value
            self.store.save_bootstrap(scope, state)
            return None
        existing = self._preparations.get(session_id)
        if existing is not None and not existing.done():
            return existing
        task = asyncio.create_task(
            self._prepare(scope), name=f"lara-reminder-prewarm:{session_id}"
        )
        self._preparations[session_id] = task
        task.add_done_callback(
            lambda completed, key=session_id: self._discard_preparation(key, completed)
        )
        return task

    async def claim_initial(self, session_id: str, turn_id: str) -> str | None:
        task = self.prewarm_session(session_id)
        if task is not None:
            await asyncio.shield(task)
        scope = self.scope(session_id)
        async with self._lock(session_id):
            state = self.store.load_bootstrap(scope)
            status = state["status"]
            if status == BootstrapStatus.CONSUMED.value:
                return None
            if status == BootstrapStatus.FAILED.value:
                raise RuntimeError(
                    f"initial runtime context preparation failed: {state['last_error']}"
                )
            if status == BootstrapStatus.CLAIMED.value:
                if state["claimed_by_turn_id"] != turn_id:
                    raise RuntimeError(
                        "initial runtime context is already claimed by another turn"
                    )
            elif status == BootstrapStatus.READY.value:
                state["status"] = BootstrapStatus.CLAIMED.value
                state["claimed_by_turn_id"] = turn_id
                self.store.save_bootstrap(scope, state)
            else:
                raise RuntimeError(
                    f"initial runtime context reached unexpected state {status!r}"
                )
            return self.store.load_snapshot(scope).content or None

    async def acknowledge_initial(
        self, session_id: str, turn_id: str, execution_id: str
    ) -> None:
        scope = self.scope(session_id)
        async with self._lock(session_id):
            state = self.store.load_bootstrap(scope)
            if state["status"] != BootstrapStatus.CLAIMED.value:
                return
            if state["claimed_by_turn_id"] != turn_id:
                raise RuntimeError(
                    "cannot consume an initial reminder claimed by another turn"
                )
            state.update(
                {
                    "status": BootstrapStatus.CONSUMED.value,
                    "consumed_by_execution_id": execution_id,
                }
            )
            self.store.save_bootstrap(scope, state)

    async def release_initial(self, session_id: str, turn_id: str) -> None:
        scope = self.scope(session_id)
        async with self._lock(session_id):
            state = self.store.load_bootstrap(scope)
            if (
                state["status"] == BootstrapStatus.CLAIMED.value
                and state["claimed_by_turn_id"] == turn_id
            ):
                state.update(
                    {"status": BootstrapStatus.READY.value, "claimed_by_turn_id": ""}
                )
                self.store.save_bootstrap(scope, state)

    async def observe_after_tools(
        self,
        session_id: str,
        *,
        bash_commands: tuple[str, ...],
        file_mutation: bool,
        has_results: bool,
    ) -> str | None:
        if not has_results:
            return None
        completed = self._take_completed_observation(session_id)
        scope = self.scope(session_id)
        async with self._lock(session_id):
            immediate = await asyncio.to_thread(
                self._observe_fast_sync,
                scope,
                bash_commands,
                file_mutation,
            )
        active = self._observations.get(session_id)
        if active is None:
            self._start_git_observation(session_id)
        elif not active.done():
            self._queued_observations.add(session_id)
        sections = [*immediate, *([completed] if completed is not None else [])]
        return render_context_body(sections)

    async def collect_pending(self, session_id: str) -> str | None:
        result = self._take_completed_observation(session_id)
        return render_context_body([result]) if result is not None else None

    async def claim_agent_messages(
        self,
        session_id: str,
        *,
        claim_id: str,
        lease_seconds: float = 30.0,
    ) -> tuple[AgentMessage, ...]:
        messages = await asyncio.to_thread(
            self.message_store.claim_messages,
            session_id,
            claim_id=claim_id,
            lease_seconds=lease_seconds,
        )
        return tuple(messages)

    async def claimed_agent_message(
        self,
        message_id: str,
        *,
        claim_id: str,
    ) -> AgentMessage | None:
        return await asyncio.to_thread(
            self.message_store.get_claimed_message,
            message_id,
            claim_id=claim_id,
        )

    async def agent_message_status(self, message_id: str) -> AgentMessage:
        return await asyncio.to_thread(self.message_store.get_message, message_id)

    async def acknowledge_agent_message(
        self,
        message_id: str,
        *,
        claim_id: str,
        execution_id: str,
    ) -> None:
        await asyncio.to_thread(
            self.message_store.acknowledge_message,
            message_id,
            claim_id=claim_id,
            execution_id=execution_id,
        )

    async def release_agent_message(self, message_id: str, *, claim_id: str) -> None:
        await asyncio.to_thread(
            self.message_store.release_message,
            message_id,
            claim_id=claim_id,
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        tasks = tuple(task for task in self._preparations.values() if not task.done())
        tasks += tuple(task for task in self._observations.values() if not task.done())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._preparations.clear()
        self._observations.clear()
        self._queued_observations.clear()

    async def _prepare(self, scope: ReminderScope) -> None:
        async with self._lock(scope.session_id):
            state = self.store.load_bootstrap(scope)
            if state["status"] in {
                BootstrapStatus.READY.value,
                BootstrapStatus.CLAIMED.value,
                BootstrapStatus.CONSUMED.value,
            }:
                return
            state.update({"status": BootstrapStatus.PREPARING.value, "last_error": ""})
            self.store.save_bootstrap(scope, state)
        try:
            snapshot, observations = await asyncio.to_thread(
                build_initial_snapshot, scope
            )
            async with self._lock(scope.session_id):
                self.store.save_snapshot(scope, snapshot)
                self.store.save_observations(scope, observations)
                state = self.store.load_bootstrap(scope)
                state.update(
                    {
                        "status": BootstrapStatus.READY.value,
                        "snapshot_id": snapshot.snapshot_id,
                        "prepared_at": snapshot.prepared_at,
                        "last_error": "",
                    }
                )
                self.store.save_bootstrap(scope, state)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            async with self._lock(scope.session_id):
                state = self.store.load_bootstrap(scope)
                state.update(
                    {"status": BootstrapStatus.FAILED.value, "last_error": str(exc)}
                )
                self.store.save_bootstrap(scope, state)
            raise

    def _observe_fast_sync(
        self,
        scope: ReminderScope,
        bash_commands: tuple[str, ...],
        file_mutation: bool,
    ) -> list[ReminderSection]:
        observations = self.store.load_observations(scope)
        sections = []
        cli_state = dict(observations.get("cli") or {})
        for command in bash_commands:
            section = detect_cli_changes(
                cli_state, scope.cli_bash_presets, command=command
            )
            if section is not None:
                sections.append(section)
        observations["cli"] = cli_state
        if file_mutation:
            skills_state = dict(observations.get("skills") or {})
            section = detect_skill_changes(scope, skills_state)
            observations["skills"] = skills_state
            if section is not None:
                sections.append(section)
        self.store.save_observations(scope, observations)
        return sections

    def _start_git_observation(self, session_id: str) -> None:
        scope = self.scope(session_id)

        async def observe() -> ReminderSection | None:
            async with self._lock(session_id):
                return await asyncio.to_thread(
                    self._observe_git_sync,
                    scope,
                )

        self._observations[session_id] = asyncio.create_task(
            observe(),
            name=f"lara-reminder-observe:{session_id}",
        )

    def _observe_git_sync(self, scope: ReminderScope) -> ReminderSection | None:
        observations = self.store.load_observations(scope)
        git_state = dict(observations.get("git") or {})
        section = detect_git_change(scope.workspace_root, git_state)
        observations["git"] = git_state
        self.store.save_observations(scope, observations)
        return section

    def _take_completed_observation(self, session_id: str) -> ReminderSection | None:
        task = self._observations.get(session_id)
        if task is None or not task.done():
            return None
        self._observations.pop(session_id, None)
        try:
            result = task.result()
        except (asyncio.CancelledError, Exception):
            result = None
        if session_id in self._queued_observations and not self._closed:
            self._queued_observations.discard(session_id)
            self._start_git_observation(session_id)
        return result

    def _lock(self, session_id: str) -> asyncio.Lock:
        return self._locks.setdefault(session_id, asyncio.Lock())

    def _discard_preparation(self, session_id: str, task: asyncio.Task[None]) -> None:
        if self._preparations.get(session_id) is task:
            self._preparations.pop(session_id, None)
        if not task.cancelled():
            task.exception()

    @staticmethod
    def _has_existing_user_history(scope: ReminderScope) -> bool:
        session_path = scope.session_dir / "session.json"
        if not session_path.exists():
            return False
        session = read_json(session_path, default={})
        metadata = session.get("metadata") or {}
        # Every fork follows the fork reminder policy regardless of its display
        # mode: the inherited history must not consume the bootstrap shortcut.
        if isinstance(metadata, dict) and (
            metadata.get("mode") == "fork" or metadata.get("forked_from")
        ):
            return False
        return any(
            isinstance(message, dict) and message.get("role") == "user"
            for message in session.get("history") or []
        )
