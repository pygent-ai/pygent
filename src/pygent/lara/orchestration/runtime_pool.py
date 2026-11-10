from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pygent.lara.graph.prompts import PromptRegistry
from pygent.lara.runtime.service import LaraRuntimeService
from pygent.lara.schema import RunConfig
from pygent.lara.sessions import SessionManager

from .runtime_keys import RuntimeGenerationKey, RuntimeScopeKey
from .runtime_lease import RuntimeLease, SessionRuntime


@dataclass(slots=True)
class _RuntimeEntry:
    entry_id: str
    runtime: SessionRuntime
    initialization: asyncio.Task[None]
    references: int = 0
    retiring: bool = False


class WorkspaceRuntimePool:
    """Own and lease configuration generations of workspace runtimes."""

    def __init__(
        self,
        *,
        runtime_factory: Callable[..., Any] = LaraRuntimeService,
        credential_environ: Mapping[str, str] | None = None,
        run_config_loader: Callable[..., RunConfig] | None = None,
        prompt_registry: PromptRegistry | None = None,
    ) -> None:
        self._runtime_factory = runtime_factory
        self._credential_environ = credential_environ
        self._run_config_loader = run_config_loader
        self._prompt_registry = prompt_registry
        self._entries: dict[str, _RuntimeEntry] = {}
        self._active: dict[RuntimeScopeKey, str] = {}
        self._lock = asyncio.Lock()
        self._accepting = True

    async def acquire(
        self,
        *,
        config: RunConfig,
        manager: SessionManager | None = None,
        reminders: Any | None = None,
    ) -> RuntimeLease:
        generation = RuntimeGenerationKey.from_config(config)
        runtime_to_retire: Any | None = None
        async with self._lock:
            if not self._accepting:
                raise RuntimeError("workspace runtime pool is closing")
            entry = self._active_entry_locked(generation)
            if entry is None:
                previous_id = self._active.get(generation.scope)
                if previous_id is not None:
                    previous = self._entries.get(previous_id)
                    if previous is not None:
                        previous.retiring = True
                        if previous.references == 0:
                            self._entries.pop(previous.entry_id, None)
                            runtime_to_retire = previous.runtime.runtime_service
                effective_manager = manager or SessionManager(config)
                runtime_service = self._runtime_factory(
                    config,
                    reminders=reminders,
                    own_reminders=reminders is None,
                    credential_environ=self._credential_environ,
                    run_config_loader=self._run_config_loader,
                    prompt_registry=self._prompt_registry,
                )
                runtime = SessionRuntime(
                    generation_key=generation,
                    config=config,
                    manager=effective_manager,
                    runtime_service=runtime_service,
                )
                entry = _RuntimeEntry(
                    entry_id=uuid.uuid4().hex,
                    runtime=runtime,
                    initialization=asyncio.create_task(runtime_service.initialize()),
                )
                self._entries[entry.entry_id] = entry
                self._active[generation.scope] = entry.entry_id
            entry.references += 1
        if runtime_to_retire is not None:
            await runtime_to_retire.close(cancel=False)
        try:
            await asyncio.shield(entry.initialization)
        except BaseException:
            await self._discard_failed_acquisition(entry)
            raise
        return RuntimeLease(
            runtime=entry.runtime,
            _release=lambda: self._release(entry.entry_id),
        )

    def _active_entry_locked(
        self, generation: RuntimeGenerationKey
    ) -> _RuntimeEntry | None:
        entry_id = self._active.get(generation.scope)
        entry = None if entry_id is None else self._entries.get(entry_id)
        if (
            entry is None
            or entry.retiring
            or entry.runtime.generation_key != generation
        ):
            return None
        return entry

    async def _discard_failed_acquisition(self, entry: _RuntimeEntry) -> None:
        runtime_to_close: Any | None = None
        async with self._lock:
            entry.retiring = True
            if self._active.get(entry.runtime.generation_key.scope) == entry.entry_id:
                del self._active[entry.runtime.generation_key.scope]
            entry.references -= 1
            if (
                entry.references == 0
                and self._entries.pop(entry.entry_id, None) is not None
            ):
                runtime_to_close = entry.runtime.runtime_service
        if runtime_to_close is not None:
            await runtime_to_close.close(cancel=True)

    async def _release(self, entry_id: str) -> None:
        runtime_to_close: Any | None = None
        async with self._lock:
            entry = self._entries.get(entry_id)
            if entry is None:
                return
            if entry.references <= 0:
                raise RuntimeError("runtime lease reference count underflow")
            entry.references -= 1
            if entry.references == 0 and entry.retiring:
                del self._entries[entry_id]
                runtime_to_close = entry.runtime.runtime_service
        if runtime_to_close is not None:
            await runtime_to_close.close(cancel=False)

    async def stop_accepting(self) -> None:
        async with self._lock:
            self._accepting = False

    async def close(self, *, cancel: bool) -> None:
        async with self._lock:
            self._accepting = False
            entries = tuple(self._entries.values())
            self._entries.clear()
            self._active.clear()
        if entries:
            for entry in entries:
                if not entry.initialization.done():
                    entry.initialization.cancel()
            await asyncio.gather(
                *(entry.initialization for entry in entries),
                return_exceptions=True,
            )
            await asyncio.gather(
                *(
                    entry.runtime.runtime_service.close(cancel=cancel)
                    for entry in entries
                ),
                return_exceptions=True,
            )


__all__ = ["WorkspaceRuntimePool"]
