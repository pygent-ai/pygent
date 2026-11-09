from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TypeVar

from pygent.lara.core.hashing import fingerprint as _fingerprint
from pygent.lara.core.io import validate_path_id
from pygent.lara.dispatch import DispatchOperationView, DispatchStore
from pygent.lara.schema import RunConfig
from pygent.lara.sessions import (
    AgentMessage,
    CollaborationOperation,
    SessionCollaborationStore,
    SessionDirectory,
    SessionManager,
    user_lara_root_for,
)

Item = AgentMessage | CollaborationOperation | DispatchOperationView
_WaitItem = TypeVar("_WaitItem", bound=Item)


async def _wait_until_done(
    fetch: Callable[[], Awaitable[_WaitItem]],
    *,
    poll_interval: float,
    timeout: float | None,
) -> _WaitItem:
    async def _wait() -> _WaitItem:
        while True:
            item = await fetch()
            if item.done:
                return item
            await asyncio.sleep(poll_interval)

    return await _wait() if timeout is None else await asyncio.wait_for(_wait(), timeout)


class SessionCollaborationService:
    """Tool-boundary Agent-message delivery over project-level message stores.

    Background starts and forks are dispatched through the user-level
    DispatchStore (see lara.dispatch); this service only routes messages and
    reports item status, falling back to the legacy project operation table
    for operations recorded before the unified dispatch channel.
    """

    def __init__(
        self,
        *,
        lease_seconds: float = 30.0,
        poll_interval: float = 0.1,
    ) -> None:
        if lease_seconds <= 0 or poll_interval <= 0:
            raise ValueError("collaboration timing values must be positive")
        self.lease_seconds = float(lease_seconds)
        self.poll_interval = float(poll_interval)
        self._directories: dict[str, SessionDirectory] = {}
        self._dispatch: DispatchStore | None = None
        self._accepting = True
        self._lock = asyncio.Lock()

    async def send(
        self,
        *,
        config: RunConfig,
        manager: SessionManager,
        target_session_id: str,
        message: str,
        source_session_id: str | None = None,
        source_agent_alias: str | None = None,
        submission_id: str | None = None,
    ) -> AgentMessage:
        if not message.strip():
            raise ValueError("message must be non-empty")
        if source_session_id is not None and source_session_id == target_session_id:
            raise ValueError("source and target sessions must be different")
        async with self._lock:
            if not self._accepting:
                raise RuntimeError("session collaboration service is closing")
        if source_session_id is not None:
            await self._resolve_session_root(config, manager, source_session_id)
        target_root = await self._resolve_session_root(
            config, manager, target_session_id
        )
        effective_submission_id = submission_id or f"submission-{uuid.uuid4().hex}"
        fingerprint = _fingerprint(
            intent="agent-message",
            source_session_id=source_session_id,
            target_session_id=target_session_id,
            message=message,
            source_agent_alias=source_agent_alias,
        )
        # Messages are always enqueued into the project store that owns the
        # target session: the target runtime claims from its own store, so
        # cross-project delivery needs no changes on the receiving side.
        queued = await asyncio.to_thread(
            self._store_for_root(target_root).enqueue_message,
            submission_id=effective_submission_id,
            fingerprint=fingerprint,
            source_session_id=source_session_id,
            source_agent_alias=source_agent_alias,
            target_session_id=target_session_id,
            content=message,
        )
        await asyncio.to_thread(
            self._directory_for_root(target_root).register_item,
            item_id=queued.message_id,
            kind="message",
            lara_root=target_root,
            source_session_id=source_session_id,
            target_session_id=target_session_id,
        )
        return queued

    async def message_status(
        self,
        *,
        config: RunConfig,
        message_id: str,
    ) -> AgentMessage:
        try:
            return await asyncio.to_thread(
                self._store(config).get_message, message_id
            )
        except FileNotFoundError:
            store = await self._routed_store(config, message_id)
            if store is None:
                raise
            return await asyncio.to_thread(store.get_message, message_id)

    async def _routed_store(
        self, config: RunConfig, collaboration_id: str
    ) -> SessionCollaborationStore | None:
        """Resolve the project store holding a collaboration item.

        Items register in the user-level directory when they are enqueued;
        the directory routes lookups from any project.
        """
        location = await asyncio.to_thread(
            self._directory_for_root(config.lara_root).locate_item, collaboration_id
        )
        if location is None:
            return None
        return self._store_for_root(location.lara_root)

    async def _resolve_session_root(
        self,
        config: RunConfig,
        manager: SessionManager,
        session_id: str,
    ) -> str:
        """Resolve the project root owning a session.

        Project-local sessions resolve through the manager; any session
        registered in the user-level directory is addressable across
        projects. Unknown sessions fail closed.
        """
        try:
            manager.load(session_id)
            return str(manager.lara_root)
        except FileNotFoundError:
            location = await asyncio.to_thread(
                self._directory_for_root(config.lara_root).locate_session,
                session_id,
            )
            if location is None:
                raise
            return location.lara_root

    async def list_operations(
        self,
        *,
        config: RunConfig,
        session_id: str,
        limit: int = 100,
    ) -> list[Item]:
        validate_path_id(session_id, "session_id")
        if not 1 <= limit <= 200:
            raise ValueError("limit must be between 1 and 200")
        dispatched = await asyncio.to_thread(
            self._dispatch_store(config).list_items_for_session,
            session_id,
            limit=limit,
        )
        legacy = await asyncio.to_thread(
            self._store(config).list_operations, session_id, limit=limit
        )
        return [
            *[DispatchOperationView.from_item(item) for item in dispatched],
            *legacy,
        ]

    async def wait_message(
        self,
        *,
        config: RunConfig,
        message_id: str,
        timeout: float | None = None,
    ) -> AgentMessage:
        async def fetch() -> AgentMessage:
            return await self.message_status(config=config, message_id=message_id)

        return await _wait_until_done(
            fetch, poll_interval=self.poll_interval, timeout=timeout
        )

    async def item_status(
        self,
        *,
        config: RunConfig,
        collaboration_id: str,
    ) -> Item:
        if collaboration_id.startswith("msg-"):
            return await self.message_status(
                config=config,
                message_id=collaboration_id,
            )
        return await self.status(config=config, operation_id=collaboration_id)

    async def wait_any(
        self,
        *,
        config: RunConfig,
        collaboration_ids: tuple[str, ...],
        timeout: float,
    ) -> tuple[tuple[Item, ...], bool]:
        if not collaboration_ids:
            raise ValueError("collaboration_ids must be non-empty")
        if len(collaboration_ids) > 32:
            raise ValueError("collaboration_ids cannot contain more than 32 items")
        if len(set(collaboration_ids)) != len(collaboration_ids):
            raise ValueError("collaboration_ids must be unique")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        deadline = time.monotonic() + timeout
        while True:
            items = tuple(
                await asyncio.gather(
                    *(
                        self.item_status(config=config, collaboration_id=item_id)
                        for item_id in collaboration_ids
                    )
                )
            )
            if any(item.done for item in items):
                return items, False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return items, True
            await asyncio.sleep(min(self.poll_interval, remaining))

    async def status(
        self, *, config: RunConfig, operation_id: str
    ) -> CollaborationOperation | DispatchOperationView:
        try:
            item = await asyncio.to_thread(
                self._dispatch_store(config).get_item, operation_id
            )
            return DispatchOperationView.from_item(item)
        except KeyError:
            pass
        try:
            return await asyncio.to_thread(self._store(config).get, operation_id)
        except FileNotFoundError:
            store = await self._routed_store(config, operation_id)
            if store is None:
                raise
            return await asyncio.to_thread(store.get, operation_id)

    async def wait(
        self,
        *,
        config: RunConfig,
        operation_id: str,
        timeout: float | None = None,
    ) -> CollaborationOperation | DispatchOperationView:
        async def fetch() -> CollaborationOperation | DispatchOperationView:
            return await self.status(config=config, operation_id=operation_id)

        return await _wait_until_done(
            fetch, poll_interval=self.poll_interval, timeout=timeout
        )

    async def aclose(self) -> None:
        async with self._lock:
            self._accepting = False

    def _dispatch_store(self, config: RunConfig) -> DispatchStore:
        if self._dispatch is None:
            root = user_lara_root_for(config.lara_root)
            self._dispatch = DispatchStore(Path(root) / "dispatch-v1.sqlite3")
        return self._dispatch

    def _store_for_root(self, lara_root: str | Path) -> SessionCollaborationStore:
        return SessionCollaborationStore(lara_root)

    def _store(self, config: RunConfig) -> SessionCollaborationStore:
        return self._store_for_root(config.lara_root)

    def _directory_for_root(self, lara_root: str | Path) -> SessionDirectory:
        key = str(user_lara_root_for(lara_root))
        return self._directories.setdefault(key, SessionDirectory(key))


__all__ = ["SessionCollaborationService"]
