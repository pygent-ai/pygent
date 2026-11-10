from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Self

from pygent.lara.runtime.agent.prompts import PromptRegistry
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
        self.runtime_pool = runtime_pool or WorkspaceRuntimePool(
            credential_environ=credential_environ,
            run_config_loader=run_config_loader,
            prompt_registry=prompt_registry,
        )
        self.coordinator = coordinator or SessionExecutionCoordinator()
        self.turns = SessionTurnService(
            coordinator=self.coordinator,
            acquire_runtime=self.acquire_runtime,
        )
        self._closed = False

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
        await self.coordinator.close()
        await self.runtime_pool.close(cancel=True)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.aclose()


__all__ = ["LocalExecutionHost"]
