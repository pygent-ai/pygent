from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from pygent import (
    Context,
    UserMessage,
)
from pygent.core import FrozenJsonObject, freeze_json_object
from pygent.llm import (
    DefaultModelInvoker,
    ExponentialBackoff,
    FallbackPolicy,
    GenerationConfig,
    ModelCallError,
    ModelErrorKind,
    ModelFailureReason,
    ModelGroupConfig,
    ModelProviderCapabilities,
    ModelProviderError,
    ModelRoute,
    OpenAICompatibleAdapter,
    RetryPolicy,
)
from pygent.llm import invoker as invoker_module


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    async def invoke(self, route, payload):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def stream(self, route, payload):
        for outcome in self.outcomes:
            if isinstance(outcome, BaseException):
                raise outcome
            yield outcome

    async def aclose(self):
        return None


class CancellationSwallowingClient:
    def __init__(self) -> None:
        self.calls = 0
        self.cancel_seen = asyncio.Event()
        self.release = asyncio.Event()
        self.stream_closed = asyncio.Event()

    async def _wait(self) -> None:
        while not self.release.is_set():
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancel_seen.set()

    async def invoke(self, route, payload):
        del route, payload
        self.calls += 1
        await self._wait()
        return completion()

    async def stream(self, route, payload):
        del route, payload
        self.calls += 1
        try:
            await self._wait()
            yield freeze_json_object({"choices": [{"delta": {"content": "ok"}}]})
            yield freeze_json_object({"done": True})
        finally:
            self.stream_closed.set()

    async def aclose(self):
        return None


class CloseSensitiveStreamingClient:
    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.started = asyncio.Event()
        self.closed = asyncio.Event()
        self.close_error: RuntimeError | None = None
        self.iterator = None

    async def invoke(self, route, payload):
        del route, payload
        return completion()

    def stream(self, route, payload):
        del route, payload

        async def events():
            self.started.set()
            while not self.release.is_set():
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    continue
            yield freeze_json_object({"choices": [{"delta": {"content": "ok"}}]})
            yield freeze_json_object({"done": True})

        self.iterator = events()
        return self.iterator

    async def aclose(self):
        try:
            if self.iterator is not None:
                await self.iterator.aclose()
        except RuntimeError as exc:
            self.close_error = exc
            raise
        finally:
            self.closed.set()


def completion(content="ok", usage=None, finish_reason=None) -> FrozenJsonObject:
    choice: dict[str, object] = {"message": {"content": content}}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    return freeze_json_object({"choices": [choice], "usage": usage or {}})


def group() -> ModelGroupConfig:
    return ModelGroupConfig(
        name="assistant",
        routes=(
            ModelRoute("primary", "openai", "first"),
            ModelRoute("fallback", "openai", "second"),
        ),
        fallback=FallbackPolicy(("primary", "fallback")),
    )


@pytest.mark.asyncio
async def test_retry_then_fallback_and_usage_events():
    primary = FakeClient([httpx.ConnectError("offline"), httpx.ConnectError("offline")])
    fallback = FakeClient([completion(usage={"total_tokens": 7})])
    invoker = DefaultModelInvoker(
        adapters={"openai": OpenAICompatibleAdapter()},
        clients={"primary": primary, "fallback": fallback},
        capabilities={"openai": ModelProviderCapabilities(streaming=False)},
    )
    execution = invoker.execute(
        model_group=group(),
        retry_policy=RetryPolicy(
            2,
            (ModelErrorKind.UNAVAILABLE,),
            ExponentialBackoff(0, 0),
        ),
        generation=GenerationConfig(),
        message=UserMessage(content="hello"),
        context=Context(),
    )
    result = await execution.result()
    async with execution.subscribe() as subscription:
        events = [event async for event in subscription]
    assert result.message.content == "ok"
    assert primary.calls == 2
    assert fallback.calls == 1
    assert [event.kind for event in events].count("model.attempt.failed") == 2
    assert "model.usage" in [event.kind for event in events]


@pytest.mark.asyncio
async def test_default_retry_policy_allows_two_total_attempts():
    primary = FakeClient([httpx.ConnectError("offline"), completion("recovered")])
    invoker = DefaultModelInvoker(
        adapters={"openai": OpenAICompatibleAdapter()},
        clients={"primary": primary, "fallback": FakeClient([completion("unused")])},
        capabilities={"openai": ModelProviderCapabilities(streaming=False)},
    )

    result = await invoker.execute(
        model_group=group(),
        retry_policy=RetryPolicy(backoff=ExponentialBackoff(0, 0)),
        generation=GenerationConfig(),
        message=UserMessage(content="hello"),
        context=Context(),
    ).result()

    assert result.message.content == "recovered"
    assert primary.calls == 2


@pytest.mark.parametrize("value", (0, -1, float("inf"), float("nan"), True))
def test_retry_policy_requires_a_positive_finite_idle_timeout(value: object) -> None:
    with pytest.raises(ValueError, match="attempt_idle_timeout_seconds"):
        RetryPolicy(attempt_idle_timeout_seconds=value)  # type: ignore[arg-type]


def test_retry_policy_does_not_accept_the_removed_total_attempt_timeout() -> None:
    with pytest.raises(TypeError, match="attempt_timeout_seconds"):
        RetryPolicy(attempt_timeout_seconds=1)  # type: ignore[call-arg]


@pytest.mark.asyncio
async def test_non_streaming_output_limit_retries_before_exposing_partial_answer():
    primary = FakeClient(
        [
            completion(
                "truncated",
                usage={"completion_tokens": 4},
                finish_reason="length",
            ),
            completion("complete", finish_reason="stop"),
        ]
    )
    invoker = DefaultModelInvoker(
        adapters={"openai": OpenAICompatibleAdapter()},
        clients={"primary": primary, "fallback": FakeClient([])},
        capabilities={"openai": ModelProviderCapabilities(streaming=False)},
    )
    execution = invoker.execute(
        model_group=group(),
        retry_policy=RetryPolicy(backoff=ExponentialBackoff(0, 0)),
        generation=GenerationConfig(),
        message=UserMessage(content="hello"),
        context=Context(),
    )

    result = await execution.result()
    async with execution.subscribe() as subscription:
        events = [event async for event in subscription]

    assert result.message.content == "complete"
    assert result.finish_reason == "stop"
    assert primary.calls == 2
    assert [event.kind for event in events].count("model.text.delta") == 1
    assert [event.kind for event in events].count("model.attempt.failed") == 1
    assert [event.kind for event in events].count("model.attempt.succeeded") == 1
    failed_usage = next(
        event.data
        for event in events
        if event.kind == "model.usage" and event.data["attempt"] == 1
    )
    assert failed_usage["output_tokens"] == 4
    assert events[-1].data["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_non_streaming_content_filter_fails_without_successful_output():
    client = FakeClient([completion("blocked", finish_reason="content_filter")])
    invoker = DefaultModelInvoker(
        adapters={"openai": OpenAICompatibleAdapter()},
        clients={"primary": client},
        capabilities={"openai": ModelProviderCapabilities(streaming=False)},
    )
    model_group = ModelGroupConfig(
        name="assistant",
        routes=(ModelRoute("primary", "openai", "first"),),
        fallback=FallbackPolicy(("primary",)),
    )
    execution = invoker.execute(
        model_group=model_group,
        retry_policy=RetryPolicy(max_attempts_per_route=1),
        generation=GenerationConfig(),
        message=UserMessage(content="hello"),
        context=Context(),
    )

    with pytest.raises(ModelCallError) as raised:
        await execution.result()
    async with execution.subscribe() as subscription:
        events = [event async for event in subscription]

    assert raised.value.kind is ModelErrorKind.INVALID_RESPONSE
    assert (
        raised.value.attempts[-1].reason_code
        is ModelFailureReason.CONTENT_POLICY_REJECTED
    )
    assert "model.text.delta" not in [event.kind for event in events]
    assert "model.attempt.succeeded" not in [event.kind for event in events]
    assert "model.completed" not in [event.kind for event in events]


@pytest.mark.asyncio
async def test_absolute_deadline_covers_provider_wait():
    class SlowClient(FakeClient):
        async def invoke(self, route, payload):
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                # Cancellation acknowledgement may require another loop turn;
                # an expired request deadline must not remove the cleanup grace.
                await asyncio.sleep(0.01)
                raise

    invoker = DefaultModelInvoker(
        adapters={"openai": OpenAICompatibleAdapter()},
        clients={"openai": SlowClient([])},
        capabilities={"openai": ModelProviderCapabilities(streaming=False)},
    )
    with pytest.raises(ModelCallError) as raised:
        await invoker.execute(
            model_group=group(),
            retry_policy=RetryPolicy(max_attempts_per_route=1),
            generation=GenerationConfig(),
            message=UserMessage(content="hello"),
            context=Context(),
            deadline=time.monotonic() + 0.01,
        ).result()
    assert raised.value.kind is ModelErrorKind.TIMEOUT


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_cancellation_cleanup_timeout_is_terminal_and_quarantines_client(
    monkeypatch: pytest.MonkeyPatch, streaming: bool
):
    monkeypatch.setattr(invoker_module, "_CANCELLATION_CLEANUP_GRACE_SECONDS", 0.02)
    primary = CancellationSwallowingClient()
    fallback = FakeClient([completion("fallback")])
    invoker = DefaultModelInvoker(
        adapters={"openai": OpenAICompatibleAdapter()},
        clients={"primary": primary, "fallback": fallback},
        capabilities={
            "primary": ModelProviderCapabilities(streaming=streaming),
            "fallback": ModelProviderCapabilities(streaming=False),
        },
    )

    started = time.monotonic()
    execution = invoker.execute(
        model_group=group(),
        retry_policy=RetryPolicy(
            max_attempts_per_route=3,
            attempt_idle_timeout_seconds=0.01,
        ),
        generation=GenerationConfig(),
        message=UserMessage(content="hello"),
        context=Context(),
        deadline=time.monotonic() + 1,
    )
    with pytest.raises(ModelCallError) as raised:
        await execution.result()
    assert time.monotonic() - started < 0.2
    assert raised.value.kind is ModelErrorKind.OUTCOME_UNKNOWN
    assert primary.calls == 1
    assert fallback.calls == 0
    async with execution.subscribe() as subscription:
        events = [event async for event in subscription]
    failure = next(event for event in events if event.kind == "model.attempt.failed")
    failure_data = freeze_json_object(failure.data)
    assert failure_data["error_kind"] == "outcome_unknown"
    assert failure_data["reason"] == "cancellation_cleanup_timeout"

    quarantined = invoker.execute(
        model_group=group(),
        retry_policy=RetryPolicy(max_attempts_per_route=3),
        generation=GenerationConfig(),
        message=UserMessage(content="hello again"),
        context=Context(),
        deadline=time.monotonic() + 1,
    )
    with pytest.raises(ModelCallError) as second:
        await quarantined.result()
    assert second.value.kind is ModelErrorKind.OUTCOME_UNKNOWN
    assert primary.calls == 1
    assert fallback.calls == 0

    primary.release.set()
    if streaming:
        await asyncio.wait_for(primary.stream_closed.wait(), timeout=0.2)
    else:
        await asyncio.sleep(0)
        await asyncio.sleep(0)
    recovered = await invoker.execute(
        model_group=group(),
        retry_policy=RetryPolicy(max_attempts_per_route=1),
        generation=GenerationConfig(),
        message=UserMessage(content="recovered"),
        context=Context(),
        deadline=time.monotonic() + 1,
    ).result()
    assert recovered.message.content == "ok"
    assert primary.calls == 2
    await invoker.aclose()


@pytest.mark.asyncio
async def test_close_waits_for_running_stream_anext_before_closing_client(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(invoker_module, "_CANCELLATION_CLEANUP_GRACE_SECONDS", 0.02)
    client = CloseSensitiveStreamingClient()
    invoker = DefaultModelInvoker(
        adapters={"openai": OpenAICompatibleAdapter()},
        clients={"primary": client},
        capabilities={"primary": ModelProviderCapabilities(streaming=True)},
    )
    execution = invoker.execute(
        model_group=ModelGroupConfig(
            name="close-race",
            routes=(ModelRoute("primary", "openai", "first"),),
            fallback=FallbackPolicy(("primary",)),
        ),
        retry_policy=RetryPolicy(
            max_attempts_per_route=1,
            attempt_idle_timeout_seconds=0.01,
        ),
        generation=GenerationConfig(),
        message=UserMessage(content="hello"),
        context=Context(),
        deadline=time.monotonic() + 1,
    )
    await asyncio.wait_for(client.started.wait(), timeout=0.2)
    with pytest.raises(ModelCallError) as raised:
        await execution.result()
    assert raised.value.kind is ModelErrorKind.OUTCOME_UNKNOWN

    close_task = asyncio.create_task(invoker.aclose())
    await asyncio.sleep(0.03)
    assert not close_task.done()
    assert not client.closed.is_set()
    assert client.close_error is None

    client.release.set()
    await asyncio.wait_for(close_task, timeout=0.2)
    assert client.closed.is_set()
    assert client.close_error is None
    await invoker.aclose()
    assert not any(
        task.get_name().startswith("pygent-model-")
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
    )


@pytest.mark.asyncio
async def test_close_cancels_active_execution_before_closing_stream_client(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(invoker_module, "_CANCELLATION_CLEANUP_GRACE_SECONDS", 0.02)
    client = CloseSensitiveStreamingClient()
    invoker = DefaultModelInvoker(
        adapters={"openai": OpenAICompatibleAdapter()},
        clients={"primary": client},
        capabilities={"primary": ModelProviderCapabilities(streaming=True)},
    )
    execution = invoker.execute(
        model_group=ModelGroupConfig(
            name="active-close-race",
            routes=(ModelRoute("primary", "openai", "first"),),
            fallback=FallbackPolicy(("primary",)),
        ),
        retry_policy=RetryPolicy(
            max_attempts_per_route=1,
            attempt_idle_timeout_seconds=1,
        ),
        generation=GenerationConfig(),
        message=UserMessage(content="hello"),
        context=Context(),
        deadline=time.monotonic() + 2,
    )
    await asyncio.wait_for(client.started.wait(), timeout=0.2)

    close_task = asyncio.create_task(invoker.aclose())
    await asyncio.sleep(0.03)
    assert not close_task.done()
    assert not client.closed.is_set()
    assert client.close_error is None

    client.release.set()
    await asyncio.wait_for(close_task, timeout=0.2)
    with pytest.raises(asyncio.CancelledError):
        await execution.result()
    assert client.closed.is_set()
    assert client.close_error is None


def test_asyncio_run_shutdown_has_no_concurrent_generator_close() -> None:
    probe = Path(__file__).parents[1] / "support" / "stream_shutdown_probe.py"
    completed = subprocess.run(
        [sys.executable, str(probe)],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "asynchronous generator is already running" not in completed.stderr
    assert "Task was destroyed but it is pending" not in completed.stderr
    assert "async generator" not in completed.stderr.lower()


@pytest.mark.asyncio
async def test_close_is_idempotent_and_rejects_new_executions() -> None:
    class CountingCloseClient(FakeClient):
        def __init__(self) -> None:
            super().__init__([completion()])
            self.close_calls = 0

        async def aclose(self) -> None:
            self.close_calls += 1

    client = CountingCloseClient()
    invoker = DefaultModelInvoker(
        adapters={"openai": OpenAICompatibleAdapter()},
        clients={"openai": client},
        capabilities={"openai": ModelProviderCapabilities(streaming=False)},
    )
    await asyncio.gather(invoker.aclose(), invoker.aclose())
    assert client.close_calls == 1

    execution = invoker.execute(
        model_group=group(),
        retry_policy=RetryPolicy(max_attempts_per_route=1),
        generation=GenerationConfig(),
        message=UserMessage(content="after close"),
        context=Context(),
    )
    with pytest.raises(ModelCallError) as raised:
        await execution.result()
    assert raised.value.kind is ModelErrorKind.OUTCOME_UNKNOWN


@pytest.mark.asyncio
async def test_explicit_task_cancellation_stays_cancelled_when_cleanup_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(invoker_module, "_CANCELLATION_CLEANUP_GRACE_SECONDS", 0.02)
    client = CancellationSwallowingClient()
    invoker = DefaultModelInvoker(
        adapters={"openai": OpenAICompatibleAdapter()},
        clients={"primary": client, "fallback": FakeClient([completion()])},
        capabilities={"primary": ModelProviderCapabilities(streaming=False)},
    )
    execution = invoker.execute(
        model_group=group(),
        retry_policy=RetryPolicy(max_attempts_per_route=3),
        generation=GenerationConfig(),
        message=UserMessage(content="hello"),
        context=Context(),
        deadline=time.monotonic() + 1,
    )
    await asyncio.sleep(0)
    assert await execution.cancel()
    with pytest.raises(asyncio.CancelledError):
        await execution.result()
    assert client.calls == 1
    client.release.set()
    await asyncio.sleep(0)
    await invoker.aclose()


@pytest.mark.asyncio
async def test_explicit_cancellation_is_not_retried():
    cancel = asyncio.Event()
    cancel.set()
    invoker = DefaultModelInvoker(
        adapters={"openai": OpenAICompatibleAdapter()},
        clients={"openai": FakeClient([completion()])},
        capabilities={"openai": ModelProviderCapabilities(streaming=False)},
    )
    with pytest.raises(asyncio.CancelledError):
        await invoker.execute(
            model_group=group(),
            retry_policy=RetryPolicy(max_attempts_per_route=3),
            generation=GenerationConfig(),
            message=UserMessage(content="hello"),
            context=Context(),
            cancel_event=cancel,
        ).result()


@pytest.mark.asyncio
async def test_stream_normalizes_text_usage_and_completion():
    client = FakeClient(
        [
            freeze_json_object({"choices": [{"delta": {"content": "hello"}}]}),
            freeze_json_object({"usage": {"completion_tokens": 1}}),
            freeze_json_object({"done": True}),
        ]
    )
    invoker = DefaultModelInvoker(
        adapters={"openai": OpenAICompatibleAdapter()},
        clients={"openai": client},
    )
    execution = invoker.execute(
        model_group=group(),
        retry_policy=RetryPolicy(max_attempts_per_route=1),
        generation=GenerationConfig(),
        message=UserMessage(content="hello"),
        context=Context(),
    )
    async with execution.subscribe() as subscription:
        events = [event async for event in subscription]
    await execution.result()
    assert [event.kind for event in events] == [
        "model.started",
        "model.attempt.started",
        "model.request.prepared",
        "model.text.delta",
        "model.usage",
        "model.attempt.succeeded",
        "model.completed",
    ]
    usage = next(event.data for event in events if event.kind == "model.usage")
    assert usage.to_dict() == {
        "route_id": "primary",
        "attempt": 1,
        "mode": "cumulative",
        "final": True,
        "available": True,
        "input_tokens": None,
        "output_tokens": 1,
        "total_tokens": None,
        "cached_input_tokens": None,
        "reasoning_tokens": None,
    }


@pytest.mark.asyncio
async def test_stream_idle_timeout_resets_after_every_provider_frame() -> None:
    class ActiveStreamClient(FakeClient):
        async def stream(self, route, payload):
            del route, payload
            for _ in range(8):
                await asyncio.sleep(0.02)
                yield freeze_json_object(
                    {"choices": [{"delta": {"content": "x"}}]}
                )
            await asyncio.sleep(0.02)
            yield freeze_json_object({"done": True})

    invoker = DefaultModelInvoker(
        adapters={"openai": OpenAICompatibleAdapter()},
        clients={"primary": ActiveStreamClient([])},
        capabilities={"primary": ModelProviderCapabilities(streaming=True)},
    )
    started = time.monotonic()
    result = await invoker.execute(
        model_group=group(),
        retry_policy=RetryPolicy(
            max_attempts_per_route=1,
            attempt_idle_timeout_seconds=0.1,
        ),
        generation=GenerationConfig(),
        message=UserMessage(content="hello"),
        context=Context(),
        deadline=time.monotonic() + 1,
    ).result()

    assert time.monotonic() - started > 0.1
    assert result.message.content == "x" * 8
    await invoker.aclose()


@pytest.mark.asyncio
async def test_first_stream_frame_idle_timeout_retries_before_public_output() -> None:
    class SlowFirstFrameClient(FakeClient):
        def __init__(self) -> None:
            super().__init__([])
            self.stream_calls = 0

        async def stream(self, route, payload):
            del route, payload
            self.stream_calls += 1
            if self.stream_calls == 1:
                await asyncio.sleep(1)
            yield freeze_json_object(
                {"choices": [{"delta": {"content": "retried"}}]}
            )
            yield freeze_json_object({"done": True})

    client = SlowFirstFrameClient()
    invoker = DefaultModelInvoker(
        adapters={"openai": OpenAICompatibleAdapter()},
        clients={"primary": client},
        capabilities={"primary": ModelProviderCapabilities(streaming=True)},
    )
    result = await invoker.execute(
        model_group=group(),
        retry_policy=RetryPolicy(
            max_attempts_per_route=2,
            attempt_idle_timeout_seconds=0.03,
            backoff=ExponentialBackoff(0, 0),
        ),
        generation=GenerationConfig(),
        message=UserMessage(content="hello"),
        context=Context(),
        deadline=time.monotonic() + 1,
    ).result()

    assert result.message.content == "retried"
    assert client.stream_calls == 2
    await invoker.aclose()


@pytest.mark.asyncio
async def test_stream_idle_timeout_resets_partial_output_and_retries() -> None:
    class StalledStreamClient(FakeClient):
        def __init__(self) -> None:
            super().__init__([])
            self.stream_calls = 0

        async def stream(self, route, payload):
            del route, payload
            self.stream_calls += 1
            if self.stream_calls == 1:
                yield freeze_json_object(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "reasoning_content": "old reasoning",
                                    "content": "partial",
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "old-call",
                                            "function": {
                                                "name": "old-tool",
                                                "arguments": '{"old":',
                                            },
                                        }
                                    ],
                                }
                            }
                        ]
                    }
                )
                await asyncio.sleep(1)
            yield freeze_json_object(
                {"choices": [{"delta": {"content": "recovered"}}]}
            )
            yield freeze_json_object({"done": True})

    client = StalledStreamClient()
    invoker = DefaultModelInvoker(
        adapters={"openai": OpenAICompatibleAdapter()},
        clients={"primary": client},
        capabilities={"primary": ModelProviderCapabilities(streaming=True)},
    )
    execution = invoker.execute(
        model_group=group(),
        retry_policy=RetryPolicy(
            max_attempts_per_route=2,
            attempt_idle_timeout_seconds=0.03,
            backoff=ExponentialBackoff(0, 0),
        ),
        generation=GenerationConfig(),
        message=UserMessage(content="hello"),
        context=Context(),
        deadline=time.monotonic() + 1,
    )

    result = await execution.result()
    async with execution.subscribe() as subscription:
        events = [event async for event in subscription]

    assert result.message.content == "recovered"
    assert result.message.tool_calls == ()
    assert client.stream_calls == 2
    kinds = [event.kind for event in events]
    assert kinds.count("model.attempt.failed") == 1
    assert kinds.count("model.output.reset") == 1
    assert kinds.index("model.attempt.failed") < kinds.index("model.output.reset")
    assert kinds.index("model.output.reset") < kinds.index(
        "model.attempt.started", kinds.index("model.output.reset")
    )
    reset = next(event.data for event in events if event.kind == "model.output.reset")
    assert reset.to_dict() == {"route_id": "primary", "attempt": 1}
    await invoker.aclose()


@pytest.mark.asyncio
async def test_stream_idle_timeout_exhaustion_keeps_last_partial_output() -> None:
    class AlwaysStalledStreamClient(FakeClient):
        def __init__(self) -> None:
            super().__init__([])
            self.stream_calls = 0

        async def stream(self, route, payload):
            del route, payload
            self.stream_calls += 1
            yield freeze_json_object(
                {"choices": [{"delta": {"content": str(self.stream_calls)}}]}
            )
            await asyncio.sleep(1)
            yield freeze_json_object({"done": True})

    client = AlwaysStalledStreamClient()
    invoker = DefaultModelInvoker(
        adapters={"openai": OpenAICompatibleAdapter()},
        clients={"primary": client},
        capabilities={"primary": ModelProviderCapabilities(streaming=True)},
    )
    execution = invoker.execute(
        model_group=ModelGroupConfig(
            name="idle-exhaustion",
            routes=(ModelRoute("primary", "openai", "first"),),
            fallback=FallbackPolicy(("primary",)),
        ),
        retry_policy=RetryPolicy(
            max_attempts_per_route=2,
            attempt_idle_timeout_seconds=0.03,
            backoff=ExponentialBackoff(0, 0),
        ),
        generation=GenerationConfig(),
        message=UserMessage(content="hello"),
        context=Context(),
        deadline=time.monotonic() + 1,
    )

    with pytest.raises(ModelCallError) as raised:
        await execution.result()
    async with execution.subscribe() as subscription:
        events = [event async for event in subscription]

    assert raised.value.kind is ModelErrorKind.TIMEOUT
    assert raised.value.partial_output is True
    assert all(
        attempt.reason_code is ModelFailureReason.PROVIDER_IDLE_TIMEOUT
        for attempt in raised.value.attempts
    )
    assert client.stream_calls == 2
    assert [event.kind for event in events].count("model.output.reset") == 1
    await invoker.aclose()


@pytest.mark.asyncio
async def test_stream_partial_idle_timeout_obeys_retry_on_policy() -> None:
    class StalledStreamClient(FakeClient):
        def __init__(self) -> None:
            super().__init__([])
            self.stream_calls = 0

        async def stream(self, route, payload):
            del route, payload
            self.stream_calls += 1
            yield freeze_json_object(
                {"choices": [{"delta": {"content": "partial"}}]}
            )
            await asyncio.sleep(1)
            yield freeze_json_object({"done": True})

    client = StalledStreamClient()
    invoker = DefaultModelInvoker(
        adapters={"openai": OpenAICompatibleAdapter()},
        clients={"primary": client},
        capabilities={"primary": ModelProviderCapabilities(streaming=True)},
    )
    execution = invoker.execute(
        model_group=ModelGroupConfig(
            name="idle-policy",
            routes=(ModelRoute("primary", "openai", "first"),),
            fallback=FallbackPolicy(("primary",)),
        ),
        retry_policy=RetryPolicy(
            max_attempts_per_route=2,
            retry_on=(),
            attempt_idle_timeout_seconds=0.03,
        ),
        generation=GenerationConfig(),
        message=UserMessage(content="hello"),
        context=Context(),
        deadline=time.monotonic() + 1,
    )

    with pytest.raises(ModelCallError) as raised:
        await execution.result()

    assert raised.value.partial_output is True
    assert client.stream_calls == 1
    await invoker.aclose()


@pytest.mark.asyncio
async def test_stream_partial_output_is_not_reset_when_backoff_exhausts_deadline() -> None:
    class StalledStreamClient(FakeClient):
        def __init__(self) -> None:
            super().__init__([])
            self.stream_calls = 0

        async def stream(self, route, payload):
            del route, payload
            self.stream_calls += 1
            yield freeze_json_object(
                {"choices": [{"delta": {"content": "still-visible"}}]}
            )
            await asyncio.sleep(1)
            yield freeze_json_object({"done": True})

    client = StalledStreamClient()
    invoker = DefaultModelInvoker(
        adapters={"openai": OpenAICompatibleAdapter()},
        clients={"primary": client},
        capabilities={"primary": ModelProviderCapabilities(streaming=True)},
    )
    execution = invoker.execute(
        model_group=ModelGroupConfig(
            name="idle-backoff-deadline",
            routes=(ModelRoute("primary", "openai", "first"),),
            fallback=FallbackPolicy(("primary",)),
        ),
        retry_policy=RetryPolicy(
            max_attempts_per_route=2,
            attempt_idle_timeout_seconds=0.1,
            backoff=ExponentialBackoff(1, 1),
        ),
        generation=GenerationConfig(),
        message=UserMessage(content="hello"),
        context=Context(),
        deadline=time.monotonic() + 0.25,
    )

    with pytest.raises(ModelCallError) as raised:
        await execution.result()
    async with execution.subscribe() as subscription:
        events = [event async for event in subscription]

    assert raised.value.kind is ModelErrorKind.TIMEOUT
    assert raised.value.partial_output is True
    assert client.stream_calls == 1
    assert "model.output.reset" not in [event.kind for event in events]
    await invoker.aclose()


@pytest.mark.asyncio
async def test_stream_partial_idle_timeout_resets_before_fallback_route() -> None:
    class PartialPrimary(FakeClient):
        async def stream(self, route, payload):
            del route, payload
            yield freeze_json_object(
                {"choices": [{"delta": {"content": "primary-partial"}}]}
            )
            await asyncio.sleep(1)
            yield freeze_json_object({"done": True})

    fallback = FakeClient(
        [
            freeze_json_object(
                {"choices": [{"delta": {"content": "fallback-complete"}}]}
            ),
            freeze_json_object({"done": True}),
        ]
    )
    invoker = DefaultModelInvoker(
        adapters={"openai": OpenAICompatibleAdapter()},
        clients={"primary": PartialPrimary([]), "fallback": fallback},
        capabilities={
            "primary": ModelProviderCapabilities(streaming=True),
            "fallback": ModelProviderCapabilities(streaming=True),
        },
    )
    execution = invoker.execute(
        model_group=group(),
        retry_policy=RetryPolicy(
            max_attempts_per_route=1,
            attempt_idle_timeout_seconds=0.03,
        ),
        generation=GenerationConfig(),
        message=UserMessage(content="hello"),
        context=Context(),
        deadline=time.monotonic() + 1,
    )

    result = await execution.result()
    async with execution.subscribe() as subscription:
        events = [event async for event in subscription]

    assert result.message.content == "fallback-complete"
    assert [event.kind for event in events].count("model.output.reset") == 1
    assert events[-1].data["route_id"] == "fallback"
    await invoker.aclose()


@pytest.mark.asyncio
async def test_execution_deadline_still_bounds_an_active_provider_stream() -> None:
    class EndlessActiveStreamClient(FakeClient):
        async def stream(self, route, payload):
            del route, payload
            while True:
                await asyncio.sleep(0.01)
                yield freeze_json_object(
                    {"choices": [{"delta": {"content": "x"}}]}
                )

    invoker = DefaultModelInvoker(
        adapters={"openai": OpenAICompatibleAdapter()},
        clients={"primary": EndlessActiveStreamClient([])},
        capabilities={"primary": ModelProviderCapabilities(streaming=True)},
    )
    execution = invoker.execute(
        model_group=group(),
        retry_policy=RetryPolicy(
            max_attempts_per_route=1,
            attempt_idle_timeout_seconds=0.1,
        ),
        generation=GenerationConfig(),
        message=UserMessage(content="hello"),
        context=Context(),
        deadline=time.monotonic() + 0.05,
    )

    with pytest.raises(ModelCallError) as raised:
        await execution.result()

    assert raised.value.kind is ModelErrorKind.TIMEOUT
    assert raised.value.partial_output is True
    await invoker.aclose()


@pytest.mark.asyncio
async def test_streaming_output_limit_fails_as_partial_without_retrying():
    client = FakeClient(
        [
            freeze_json_object({"choices": [{"delta": {"content": "partial"}}]}),
            freeze_json_object(
                {"choices": [{"delta": {}, "finish_reason": "length"}]}
            ),
        ]
    )
    model_group = ModelGroupConfig(
        name="assistant",
        routes=(ModelRoute("primary", "openai", "first"),),
        fallback=FallbackPolicy(("primary",)),
    )
    execution = DefaultModelInvoker(
        adapters={"openai": OpenAICompatibleAdapter()},
        clients={"primary": client},
    ).execute(
        model_group=model_group,
        retry_policy=RetryPolicy(backoff=ExponentialBackoff(0, 0)),
        generation=GenerationConfig(),
        message=UserMessage(content="hello"),
        context=Context(),
    )

    with pytest.raises(ModelCallError) as raised:
        await execution.result()
    async with execution.subscribe() as subscription:
        events = [event async for event in subscription]

    assert raised.value.kind is ModelErrorKind.INCOMPLETE_RESPONSE
    assert raised.value.partial_output is True
    assert raised.value.attempts[-1].reason_code is ModelFailureReason.OUTPUT_LIMIT_REACHED
    assert "model.text.delta" in [event.kind for event in events]
    assert "model.attempt.succeeded" not in [event.kind for event in events]
    assert "model.completed" not in [event.kind for event in events]


@pytest.mark.asyncio
async def test_stream_owner_does_not_create_a_task_for_each_provider_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient(
        [
            *(
                freeze_json_object({"choices": [{"delta": {"content": "x"}}]})
                for _ in range(100)
            ),
            freeze_json_object({"done": True}),
        ]
    )
    invoker = DefaultModelInvoker(
        adapters={"openai": OpenAICompatibleAdapter()},
        clients={"primary": client, "fallback": client},
        capabilities={"primary": ModelProviderCapabilities(streaming=True)},
    )
    task_names: list[str | None] = []
    create_task = asyncio.create_task

    def record_task(coro, *, name=None, context=None):
        task_names.append(name)
        return create_task(coro, name=name, context=context)

    monkeypatch.setattr(asyncio, "create_task", record_task)
    result = await invoker.execute(
        model_group=group(),
        retry_policy=RetryPolicy(max_attempts_per_route=1),
        generation=GenerationConfig(),
        message=UserMessage(content="hello"),
        context=Context(),
    ).result()

    assert result.message.content == "x" * 100
    assert task_names.count("pygent-model-stream-owner") == 1
    assert "pygent-model-stream-item" not in task_names
    assert "pygent-model-stream-next" not in task_names


@pytest.mark.asyncio
async def test_stream_rejects_transport_eof_without_completion_marker():
    client = FakeClient(
        [freeze_json_object({"choices": [{"delta": {"content": "partial"}}]})]
    )
    invoker = DefaultModelInvoker(
        adapters={"openai": OpenAICompatibleAdapter()},
        clients={"openai": client},
    )

    with pytest.raises(ModelCallError, match="after output was emitted") as raised:
        await invoker.execute(
            model_group=group(),
            retry_policy=RetryPolicy(max_attempts_per_route=1),
            generation=GenerationConfig(),
            message=UserMessage(content="hello"),
            context=Context(),
        ).result()

    assert raised.value.kind is ModelErrorKind.INVALID_RESPONSE
    assert raised.value.attempts[-1].reason_code is ModelFailureReason.STREAM_INCOMPLETE


@pytest.mark.asyncio
async def test_stream_retries_only_when_failure_precedes_public_output():
    class FlakyStreamClient(FakeClient):
        def __init__(self):
            super().__init__([])
            self.stream_calls = 0

        async def stream(self, route, payload):
            self.stream_calls += 1
            if self.stream_calls == 1:
                raise httpx.ConnectError("offline before first chunk")
            yield freeze_json_object({"choices": [{"delta": {"content": "ok"}}]})
            yield freeze_json_object({"done": True})

    client = FlakyStreamClient()
    invoker = DefaultModelInvoker(
        adapters={"openai": OpenAICompatibleAdapter()},
        clients={"openai": client},
    )
    execution = invoker.execute(
        model_group=group(),
        retry_policy=RetryPolicy(
            max_attempts_per_route=2,
            retry_on=(ModelErrorKind.UNAVAILABLE,),
            backoff=ExponentialBackoff(0, 0),
        ),
        generation=GenerationConfig(),
        message=UserMessage(content="hello"),
        context=Context(),
    )
    async with execution.subscribe() as subscription:
        events = [event async for event in subscription]
    await execution.result()

    assert client.stream_calls == 2
    assert [event.kind for event in events if ".attempt." not in event.kind] == [
        "model.started",
        "model.request.prepared",
        "model.usage",
        "model.request.prepared",
        "model.text.delta",
        "model.usage",
        "model.completed",
    ]


@pytest.mark.asyncio
async def test_stream_fallback_emits_attempt_lifecycle_before_public_output():
    primary = FakeClient(
        [ModelProviderError(ModelErrorKind.AUTHENTICATION, "unauthorized")]
    )
    fallback = FakeClient(
        [
            freeze_json_object({"choices": [{"delta": {"content": "ok"}}]}),
            freeze_json_object({"done": True}),
        ]
    )
    invoker = DefaultModelInvoker(
        adapters={"openai": OpenAICompatibleAdapter()},
        clients={"primary": primary, "fallback": fallback},
    )
    execution = invoker.execute(
        model_group=group(),
        retry_policy=RetryPolicy(max_attempts_per_route=1),
        generation=GenerationConfig(),
        message=UserMessage(content="hello"),
        context=Context(),
    )
    async with execution.subscribe() as subscription:
        events = [event async for event in subscription]
    await execution.result()
    lifecycle = [
        (event.kind, event.data["route_id"], event.data.get("error_kind"))
        for event in events
        if ".attempt." in event.kind
    ]

    assert [event.kind for event in events if ".attempt." not in event.kind] == [
        "model.started",
        "model.request.prepared",
        "model.usage",
        "model.request.prepared",
        "model.text.delta",
        "model.usage",
        "model.completed",
    ]
    assert lifecycle == [
        ("model.attempt.started", "primary", None),
        ("model.attempt.failed", "primary", "authentication"),
        ("model.attempt.started", "fallback", None),
        ("model.attempt.succeeded", "fallback", None),
    ]


@pytest.mark.asyncio
async def test_stream_emits_fixed_reasoning_and_multiple_tool_call_events():
    client = FakeClient(
        [
            freeze_json_object(
                {
                    "choices": [
                        {
                            "delta": {
                                "reasoning_content": "check tools",
                                "content": "working",
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call-",
                                        "function": {
                                            "name": "wea",
                                            "arguments": '{"city":"',
                                        },
                                    },
                                    {
                                        "index": 1,
                                        "id": "call-",
                                        "function": {
                                            "name": "clo",
                                            "arguments": '{"zone":"',
                                        },
                                    },
                                ],
                            }
                        }
                    ]
                }
            ),
            freeze_json_object(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "weather",
                                        "function": {
                                            "name": "ther",
                                            "arguments": 'Hangzhou"}',
                                        },
                                    },
                                    {
                                        "index": 1,
                                        "id": "clock",
                                        "function": {
                                            "name": "ck",
                                            "arguments": 'UTC+8"}',
                                        },
                                    },
                                ]
                            }
                        }
                    ]
                }
            ),
            freeze_json_object(
                {
                    "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 4,
                        "completion_tokens_details": {"reasoning_tokens": 2},
                    },
                }
            ),
            freeze_json_object({"done": True}),
        ]
    )
    execution = DefaultModelInvoker(
        adapters={"openai": OpenAICompatibleAdapter()},
        clients={"openai": client},
    ).execute(
        model_group=group(),
        retry_policy=RetryPolicy(max_attempts_per_route=1),
        generation=GenerationConfig(),
        message=UserMessage(content="hello"),
        context=Context(),
    )

    async with execution.subscribe() as subscription:
        events = [event async for event in subscription]
    result = await execution.result()

    assert result.message.content == "working"
    assert [(call.call_id, call.name) for call in result.message.tool_calls] == [
        ("call-weather", "weather"),
        ("call-clock", "clock"),
    ]
    assert result.message.tool_calls[0].arguments.to_dict() == {"city": "Hangzhou"}
    assert result.message.tool_calls[1].arguments.to_dict() == {"zone": "UTC+8"}
    assert [event.kind for event in events] == [
        "model.started",
        "model.attempt.started",
        "model.request.prepared",
        "model.reasoning.delta",
        "model.text.delta",
        "model.tool_call.started",
        "model.tool_call.delta",
        "model.tool_call.started",
        "model.tool_call.delta",
        "model.tool_call.delta",
        "model.tool_call.delta",
        "model.tool_call.completed",
        "model.tool_call.completed",
        "model.usage",
        "model.attempt.succeeded",
        "model.completed",
    ]
    completed = events[-1].data
    assert completed["finish_reason"] == "tool_calls"
    usage = next(event.data for event in events if event.kind == "model.usage")
    assert usage["input_tokens"] == 10
    assert usage["output_tokens"] == 4
    assert usage["reasoning_tokens"] == 2


@pytest.mark.asyncio
async def test_stream_synthesizes_a_missing_tool_call_id():
    client = FakeClient(
        [
            freeze_json_object(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": "0",
                                        "function": {
                                            "name": "double",
                                            "arguments": {"value": 2},
                                        },
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "id": "req-tool",
                }
            )
        ]
    )
    result = await DefaultModelInvoker(
        adapters={"openai": OpenAICompatibleAdapter()},
        clients={"openai": client},
    ).execute(
        model_group=group(),
        retry_policy=RetryPolicy(max_attempts_per_route=1),
        generation=GenerationConfig(),
        message=UserMessage(content="hello"),
        context=Context(),
    ).result()

    call = result.message.tool_calls[0]
    assert call.call_id.startswith("call_")
    assert call.name == "double"
    assert call.arguments.to_dict() == {"value": 2}


@pytest.mark.asyncio
async def test_invalid_tool_arguments_never_emit_model_completed():
    client = FakeClient(
        [
            freeze_json_object(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call-bad",
                                        "function": {
                                            "name": "broken",
                                            "arguments": "{",
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                }
            ),
            freeze_json_object({"done": True}),
        ]
    )
    execution = DefaultModelInvoker(
        adapters={"openai": OpenAICompatibleAdapter()},
        clients={"openai": client},
    ).execute(
        model_group=group(),
        retry_policy=RetryPolicy(max_attempts_per_route=1),
        generation=GenerationConfig(),
        message=UserMessage(content="hello"),
        context=Context(),
    )

    with pytest.raises(ModelCallError) as raised:
        await execution.result()
    async with execution.subscribe() as subscription:
        events = [event async for event in subscription]

    assert raised.value.kind is ModelErrorKind.INVALID_RESPONSE
    assert raised.value.attempts[-1].reason_code is ModelFailureReason.TOOL_CALL_INVALID
    assert "model.completed" not in [event.kind for event in events]
    assert [event.kind for event in events][-3:] == [
        "model.usage",
        "model.attempt.failed",
        "model.failed",
    ]
