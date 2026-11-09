from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from pygent.tool import LocalToolExecutor, SandboxExecutorSupport


class LaraModelResourceResolver:
    """Reconstructable deployment identity with process-local live invokers."""

    resolver_id = "lara-model"

    def __init__(self) -> None:
        self._invokers: dict[str, Any] = {}

    def register(self, revision: str, invoker: Any) -> None:
        self._invokers[revision] = invoker

    async def validate(self, model_group: Any, resources: Any) -> None:
        del model_group
        for _, resource in resources.model_resources:
            if resource.revision not in self._invokers:
                raise ValueError(
                    f"model resource revision {resource.revision!r} is unavailable"
                )

    @asynccontextmanager
    async def acquire(self, model_group: Any, resources: Any) -> AsyncIterator[Any]:
        del model_group
        revision = resources.model_resources[0][1].revision
        invoker = self._invokers.get(revision)
        if invoker is None:
            raise RuntimeError(f"model resource revision {revision!r} is unavailable")
        yield invoker


class WorkspaceToolExecutor(LocalToolExecutor):
    """Deployment adapter for Lara's workspace-based standard tools."""

    sandbox_support = SandboxExecutorSupport(profiles=("workspace",))


__all__ = [
    "LaraModelResourceResolver",
    "WorkspaceToolExecutor",
]
