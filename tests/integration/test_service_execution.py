from __future__ import annotations

import asyncio

import pytest

from examples.service.app import create_service
from examples.service.domain import (
    ChatRequest,
    ConversationConflict,
    InMemoryConversationStore,
)
from examples.service.main import DemoModelInvoker
from pygent.runtime import LocalRuntime


@pytest.mark.asyncio
async def test_service_example_real_invoke_stream_and_cas_commit():
    runtime = LocalRuntime()
    store = InMemoryConversationStore()
    service = create_service(runtime, store, model_invoker=DemoModelInvoker())

    response = await service.invoke(
        ChatRequest("invoke", "user", "hello", ("weather:read",))
    )
    assert response.revision == 1
    assert response.text.startswith("reviewed:")
    invoke_snapshot = await store.read("invoke")
    assert [message.content for message in invoke_snapshot.messages] == [
        "hello",
        "draft: hello",
        "reviewed: 请审核并改写：draft: hello",
    ]

    events = [
        event
        async for event in service.stream(
            ChatRequest("stream", "user", "hello", ("weather:read",))
        )
    ]
    assert [event.sequence for event in events] == list(range(len(events)))
    assert (await store.read("stream")).revision == 1

    with pytest.raises(ConversationConflict):
        await store.commit("invoke", 0, ())
    await runtime.close()


@pytest.mark.asyncio
async def test_service_cas_allows_only_one_concurrent_commit():
    runtime = LocalRuntime()
    store = InMemoryConversationStore()
    invoker = _BarrierInvoker()
    service = create_service(runtime, store, model_invoker=invoker)

    results = await asyncio.gather(
        service.invoke(ChatRequest("shared", "user-1", "first")),
        service.invoke(ChatRequest("shared", "user-2", "second")),
        return_exceptions=True,
    )

    assert sum(not isinstance(item, BaseException) for item in results) == 1
    assert sum(isinstance(item, ConversationConflict) for item in results) == 1
    snapshot = await store.read("shared")
    assert snapshot.revision == 1
    await runtime.close()


class _BarrierInvoker(DemoModelInvoker):
    def __init__(self) -> None:
        self._draft_calls = 0
        self._lock = asyncio.Lock()
        self._both_started = asyncio.Event()

    async def _invoke(self, emit, **kwargs):
        if kwargs["model_group"].name == "assistant":
            async with self._lock:
                self._draft_calls += 1
                if self._draft_calls == 2:
                    self._both_started.set()
            await self._both_started.wait()
        return await super()._invoke(emit, **kwargs)
