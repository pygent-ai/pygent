"""Observable model execution and provider stream ownership."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Self

from pygent.core import (
    FrozenJsonObject,
)

from ._adapter_contracts import (
    EventSink,
    ModelProviderClient,
    ModelProviderResponse,
    ModelStreamEvent,
)
from .types import (
    ModelRoute,
)


class _ProviderStreamOwner:
    """Sole task owner of one provider iterator from open through close."""

    def __init__(
        self,
        client: ModelProviderClient,
        route: ModelRoute,
        payload: FrozenJsonObject,
    ) -> None:
        self._client = client
        self._route = route
        self._payload = payload
        self._items: asyncio.Queue[FrozenJsonObject] = asyncio.Queue(maxsize=1)
        self._stopping = False
        self._task = asyncio.create_task(self._run(), name="pygent-model-stream-owner")

    @property
    def task(self) -> asyncio.Task[None]:
        return self._task

    @property
    def done(self) -> bool:
        return self._task.done()

    def cancel(self) -> bool:
        self._stopping = True
        if self._task.done():
            return False
        self._task.cancel()
        return True

    async def next(self) -> FrozenJsonObject:
        if not self._items.empty():
            return self._items.get_nowait()
        if self._task.done():
            await self._task
            raise StopAsyncIteration
        item_task = asyncio.create_task(
            self._items.get(), name="pygent-model-stream-item"
        )
        try:
            done, _ = await asyncio.wait(
                {item_task, self._task}, return_when=asyncio.FIRST_COMPLETED
            )
            if item_task in done:
                return item_task.result()
            item_task.cancel()
            await asyncio.gather(item_task, return_exceptions=True)
            if not self._items.empty():
                return self._items.get_nowait()
            await self._task
            raise StopAsyncIteration
        finally:
            if not item_task.done():
                item_task.cancel()
                await asyncio.gather(item_task, return_exceptions=True)

    async def _run(self) -> None:
        iterator = self._client.stream(self._route, self._payload).__aiter__()
        try:
            async for item in iterator:
                if self._stopping:
                    return
                await self._items.put(item)
                if self._stopping:
                    return
        finally:
            close = getattr(iterator, "aclose", None)
            if callable(close):
                await close()


class _ModelExecutionSubscription:
    def __init__(self, execution: ModelExecution, after: int | None) -> None:
        self._execution = execution
        self._next = 0 if after is None else after + 1

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def __aiter__(self) -> AsyncIterator[ModelStreamEvent]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[ModelStreamEvent]:
        while True:
            async with self._execution._condition:
                while (
                    self._next >= len(self._execution._events)
                    and not self._execution._task.done()
                ):
                    await self._execution._condition.wait()
                available = tuple(self._execution._events[self._next :])
                terminal = self._execution._task.done()
            for event in available:
                self._next += 1
                yield event
            if terminal and self._next >= len(self._execution._events):
                return


class ModelExecution:
    """Single owner of one model operation, its events, result, and cancellation."""

    def __init__(
        self,
        operation: Callable[[EventSink], Awaitable[ModelProviderResponse]],
    ) -> None:
        self._events: list[ModelStreamEvent] = []
        self._condition = asyncio.Condition()

        async def run() -> ModelProviderResponse:
            return await operation(self._publish)

        self._task: asyncio.Task[ModelProviderResponse] = asyncio.create_task(
            run(), name="pygent-model-execution"
        )
        self._task.add_done_callback(lambda _: asyncio.create_task(self._notify()))

    async def _publish(self, kind: str, data: FrozenJsonObject) -> None:
        async with self._condition:
            self._events.append(ModelStreamEvent(kind, data))
            self._condition.notify_all()

    async def _notify(self) -> None:
        async with self._condition:
            self._condition.notify_all()

    async def result(self) -> ModelProviderResponse:
        return await self._task

    async def cancel(self) -> bool:
        if self._task.done():
            return False
        self._task.cancel()
        return True

    def subscribe(self, *, after: int | None = None) -> _ModelExecutionSubscription:
        if after is not None and (isinstance(after, bool) or after < -1):
            raise ValueError("after must be a non-negative sequence or -1")
        return _ModelExecutionSubscription(self, after)
