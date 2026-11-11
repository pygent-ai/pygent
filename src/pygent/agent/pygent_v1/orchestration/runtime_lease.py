from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pygent.agent.pygent_v1.schema import RunConfig
from pygent.agent.pygent_v1.sessions import SessionManager

from .runtime_keys import RuntimeGenerationKey


@dataclass(slots=True)
class SessionRuntime:
    generation_key: RuntimeGenerationKey
    config: RunConfig
    manager: SessionManager
    runtime_service: Any

    @property
    def reminders(self) -> Any:
        return self.runtime_service.reminders


@dataclass(slots=True)
class RuntimeLease:
    runtime: SessionRuntime
    _release: Callable[[], Awaitable[None]] = field(repr=False)
    _released: bool = field(default=False, init=False, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    async def release(self) -> None:
        async with self._lock:
            if self._released:
                return
            self._released = True
        await self._release()


__all__ = ["RuntimeLease", "SessionRuntime"]
