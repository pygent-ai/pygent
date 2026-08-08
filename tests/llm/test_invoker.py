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


def completion(content="ok", usage=None) -> FrozenJsonObject:
    return freeze_json_object(
        {
            "choices": [{"message": {"content": content}}],
            "usage": usage or {},
        }
    )


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
        retry_policy=RetryPolicy(max_attempts_per_route=3),
        generation=GenerationConfig(),
        message=UserMessage(content="hello"),
        context=Context(),
        deadline=time.monotonic() + 0.02,
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
            attempt_timeout_seconds=0.01,
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
            attempt_timeout_seconds=1,
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
        "model.usage",
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
        "model.usage",
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
    assert "model.completed" not in [event.kind for event in events]
    assert [event.kind for event in events][-3:] == [
        "model.usage",
        "model.attempt.failed",
        "model.failed",
    ]
