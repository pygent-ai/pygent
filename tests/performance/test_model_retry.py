from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from benchmarks.config import ModelSettings
from benchmarks.models import TrackedClient, build_resources
from pygent import (
    Context,
    ExponentialBackoff,
    FallbackPolicy,
    GenerationConfig,
    ModelErrorKind,
    ModelGroupConfig,
    ModelRoute,
    RetryPolicy,
    UserMessage,
)
from pygent.core import FrozenJsonObject, freeze_json_object
from pygent.llm import (
    DefaultModelInvoker,
    ModelCallError,
    ModelProviderCapabilities,
    OpenAICompatibleAdapter,
)


class SlowOnceClient:
    def __init__(self) -> None:
        self.calls = 0
        self.closed = False

    async def invoke(
        self, route: ModelRoute, payload: FrozenJsonObject
    ) -> FrozenJsonObject:
        del route, payload
        self.calls += 1
        if self.calls == 1:
            await asyncio.sleep(1)
        return freeze_json_object(
            {
                "choices": [{"message": {"content": "retried"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            }
        )

    async def stream(
        self, route: ModelRoute, payload: FrozenJsonObject
    ) -> AsyncIterator[FrozenJsonObject]:
        del route, payload
        if False:
            yield freeze_json_object({})

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
@pytest.mark.performance
async def test_benchmark_resources_project_retry_configuration() -> None:
    resources = build_resources(
        backend="synthetic",
        settings=ModelSettings(
            retry_max_attempts=2,
            retry_on=("timeout", "unavailable"),
            retry_backoff_seconds=1.0,
            attempt_timeout_seconds=90.0,
        ),
        seed=1729,
        streaming=False,
    )
    try:
        policy = resources.model.retry_policy
        assert policy.max_attempts_per_route == 2
        assert policy.retry_on == (
            ModelErrorKind.TIMEOUT,
            ModelErrorKind.UNAVAILABLE,
        )
        assert policy.attempt_timeout_seconds == 90.0
        assert policy.backoff.delay(0) == 1.0
    finally:
        await resources.aclose()


@pytest.mark.asyncio
@pytest.mark.performance
async def test_benchmark_attempt_timeout_retries_through_pygent_model_policy() -> None:
    provider = SlowOnceClient()
    tracked = TrackedClient(provider)
    invoker = DefaultModelInvoker(
        adapters={"openai": OpenAICompatibleAdapter()},
        clients={"live": tracked},
        capabilities={"live": ModelProviderCapabilities(streaming=False)},
    )
    execution = invoker.execute(
        model_group=ModelGroupConfig(
            name="retry-check",
            routes=(ModelRoute("live", "openai", "private"),),
            fallback=FallbackPolicy(("live",)),
        ),
        retry_policy=RetryPolicy(
            max_attempts_per_route=2,
            retry_on=(ModelErrorKind.TIMEOUT,),
            backoff=ExponentialBackoff(0, 0),
            attempt_timeout_seconds=0.01,
        ),
        generation=GenerationConfig(),
        message=UserMessage(content="Return done for request 1."),
        context=Context(),
        deadline=asyncio.get_running_loop().time() + 1,
    )
    response = await execution.result()
    async with execution.subscribe() as subscription:
        model_events = [event async for event in subscription]
    events = [(event.kind, event.data) for event in model_events]
    await invoker.aclose()

    assert response.message.content == "retried"
    assert provider.calls == 2
    assert tracked.tracker.calls == 2
    assert tracked.tracker.active == 0
    assert provider.closed
    assert [kind for kind, _ in events] == [
        "model.started",
        "model.attempt.started",
        "model.usage",
        "model.attempt.failed",
        "model.attempt.started",
        "model.text.delta",
        "model.usage",
        "model.attempt.succeeded",
        "model.completed",
    ]
    assert freeze_json_object(events[3][1])["error_kind"] == "timeout"
    assert freeze_json_object(events[-2][1])["attempt"] == 2


@pytest.mark.asyncio
@pytest.mark.performance
@pytest.mark.parametrize("streaming", [False, True])
async def test_benchmark_attempt_timeout_fails_closed_when_cleanup_is_unknown(
    monkeypatch: pytest.MonkeyPatch, streaming: bool
) -> None:
    class StuckClient(SlowOnceClient):
        def __init__(self) -> None:
            super().__init__()
            self.release = asyncio.Event()

        async def invoke(self, route, payload):
            del route, payload
            self.calls += 1
            while not self.release.is_set():
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    continue
            return freeze_json_object(
                {"choices": [{"message": {"content": "released"}}]}
            )

        async def stream(self, route, payload):
            del route, payload
            self.calls += 1
            while not self.release.is_set():
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    continue
            yield freeze_json_object({"choices": [{"delta": {"content": "released"}}]})
            yield freeze_json_object({"done": True})

    monkeypatch.setattr(
        "pygent.llm.adapter._CANCELLATION_CLEANUP_GRACE_SECONDS", 0.02
    )
    provider = StuckClient()
    tracked = TrackedClient(provider)
    invoker = DefaultModelInvoker(
        adapters={"openai": OpenAICompatibleAdapter()},
        clients={"live": tracked},
        capabilities={"live": ModelProviderCapabilities(streaming=streaming)},
    )
    execution = invoker.execute(
        model_group=ModelGroupConfig(
            name="unknown-check",
            routes=(ModelRoute("live", "openai", "private"),),
            fallback=FallbackPolicy(("live",)),
        ),
        retry_policy=RetryPolicy(
            max_attempts_per_route=3,
            retry_on=(ModelErrorKind.TIMEOUT,),
            backoff=ExponentialBackoff(0, 0),
            attempt_timeout_seconds=0.01,
        ),
        generation=GenerationConfig(),
        message=UserMessage(content="request 1"),
        context=Context(),
        deadline=asyncio.get_running_loop().time() + 1,
    )
    with pytest.raises(ModelCallError) as raised:
        await execution.result()
    assert raised.value.kind is ModelErrorKind.OUTCOME_UNKNOWN
    assert provider.calls == 1
    provider.release.set()
    await asyncio.sleep(0)
    await invoker.aclose()
