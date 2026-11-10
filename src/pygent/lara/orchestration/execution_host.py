from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, Self

from pygent.lara.runtime.agent.prompts import PromptRegistry
from pygent.lara.runtime.service import LaraRuntimeService
from pygent.lara.schema import RunConfig
from pygent.lara.sessions import SessionManager

from .runtime_pool import WorkspaceRuntimePool
from .session_execution import SessionExecutionCoordinator
from .session_turns import SessionTurnService


class LocalExecutionHost:
    """Short-lived composition root for non-server execution adapters."""

    def __init__(
        self,
        *,
        runtime_pool: WorkspaceRuntimePool | None = None,
        coordinator: SessionExecutionCoordinator | None = None,
        credential_environ: Mapping[str, str] | None = None,
        run_config_loader: Callable[..., RunConfig] | None = None,
        prompt_registry: PromptRegistry | None = None,
    ) -> None:
        self.coordinator = coordinator or SessionExecutionCoordinator()
        self.runtime_pool = runtime_pool or WorkspaceRuntimePool(
            runtime_factory=self._build_runtime,
            credential_environ=credential_environ,
            run_config_loader=run_config_loader,
            prompt_registry=prompt_registry,
        )
        self.turns = SessionTurnService(
            coordinator=self.coordinator,
            acquire_runtime=self.acquire_runtime,
        )
        self._drain_tasks: set[asyncio.Task[None]] = set()
        self._closed = False

    def _build_runtime(self, config: RunConfig, **kwargs: Any) -> LaraRuntimeService:
        # Pooled runtimes learn about the coordinator through this hook: an
        # exchange envelope landing in an idle session must drain immediately
        # (see SessionExecutionCoordinator.submit_pending_drain). The drain
        # runs detached — send_interrupt must not block on it, and the turn
        # needs a task generation of its own.
        return LaraRuntimeService(
            config,
            on_exchange_queued=partial(self._spawn_exchange_drain, config),
            **kwargs,
        )

    def _spawn_exchange_drain(self, config: RunConfig, session_id: str) -> None:
        task = asyncio.create_task(self._drain_after_exchange(config, session_id))
        self._drain_tasks.add(task)
        task.add_done_callback(self._drain_tasks.discard)

    async def _drain_after_exchange(self, config: RunConfig, session_id: str) -> None:
        lease = await self.acquire_runtime(config=config)
        turn: Any = None
        try:
            turn = await self.coordinator.submit_pending_drain(
                lease=lease, session_id=session_id
            )
        finally:
            # A submitted turn carries the lease and releases it when its drive
            # finishes; an unused lease is returned right away.
            if turn is None:
                await lease.release()

    async def acquire_runtime(
        self,
        *,
        config: RunConfig,
        manager: SessionManager | None = None,
    ) -> Any:
        return await self.runtime_pool.acquire(config=config, manager=manager)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self.runtime_pool.stop_accepting()
        if self._drain_tasks:
            await asyncio.gather(*self._drain_tasks, return_exceptions=True)
        await self.coordinator.close()
        await self.runtime_pool.close(cancel=True)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.aclose()


__all__ = ["LocalExecutionHost"]
