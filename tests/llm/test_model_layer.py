from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager

import httpx
import pytest

from pygent import (
    Context,
    EffectDisposition,
    EffectOutcome,
    ExecutionAdmissionError,
    ExecutionDeadlineExceeded,
    ExecutionOptions,
    LocalRuntime,
    ToolMessage,
    UserMessage,
)
from pygent.core import freeze_json_object
from pygent.core.module import _execution_scope
from pygent.llm import (
    DefaultModelInvoker,
    FallbackPolicy,
    GenerationConfig,
    ModelCallError,
    ModelCallLayer,
    ModelExecution,
    ModelGroupConfig,
    ModelProviderCapabilities,
    ModelProviderResponse,
    ModelRoute,
    OpenAICompatibleAdapter,
    OpenAICompatibleClient,
    RetryPolicy,
)
from pygent.llm.layer import _model_effect_request
from pygent.tool import ToolDefinition, ToolResult


class RecordingInvoker:
    def __init__(self):
        self.tools = None

    def execute(self, **kwargs):
        async def operation(emit):
            self.tools = kwargs["tools"]
            from pygent import AIMessage

            return ModelProviderResponse(
                AIMessage(content="done"), {"total_tokens": 1}
            )

        return ModelExecution(operation)


def definition(name: str) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"{name} tool",
        parameters={"type": "object"},
    )


def layer(invoker=None) -> ModelCallLayer:
    return ModelCallLayer(
        model_group=ModelGroupConfig(
            "assistant",
            (ModelRoute("main", "openai", "test"),),
            FallbackPolicy(("main",)),
        ),
        retry_policy=RetryPolicy(),
        generation=GenerationConfig(),
        tools=(definition("search"), definition("weather")),
        invoker=invoker,
    )


@pytest.mark.asyncio
async def test_managed_model_call_requires_a_finite_deadline() -> None:
    runtime = LocalRuntime()
    bound = runtime.bind(layer(RecordingInvoker()))

    with pytest.raises(ExecutionAdmissionError, match="ModelCallLayer.*finite execution deadline"):
        await bound.start(UserMessage(content="missing"), Context())

    answer, _ = await bound.invoke(
        UserMessage(content="bounded"),
        Context(),
        execution=ExecutionOptions(deadline=time.monotonic() + 1),
    )
    assert answer.content == "done"
    await runtime.close()


@pytest.mark.asyncio
async def test_managed_deadline_remains_execution_deadline_when_provider_ignores_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pygent.llm import adapter as adapter_module

    monkeypatch.setattr(
        adapter_module, "_CANCELLATION_CLEANUP_GRACE_SECONDS", 0.02
    )

    class StuckClient:
        def __init__(self) -> None:
            self.calls = 0
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
            if False:
                yield freeze_json_object({})

        async def aclose(self):
            return None

    client = StuckClient()
    invoker = DefaultModelInvoker(
        adapters={"openai": OpenAICompatibleAdapter()},
        clients={"main": client},
        capabilities={"main": ModelProviderCapabilities(streaming=False)},
    )
    runtime = LocalRuntime()
    handle = await runtime.bind(layer(invoker)).start(
        UserMessage(content="deadline"),
        Context(),
        execution=ExecutionOptions(deadline=time.monotonic() + 0.03),
    )
    with pytest.raises(ExecutionDeadlineExceeded):
        await handle.result()
    assert client.calls == 1
    client.release.set()
    await asyncio.sleep(0)
    await runtime.close()
    await invoker.aclose()


@pytest.mark.parametrize("deadline", (0.0, float("inf"), float("nan")))
def test_model_run_options_reject_invalid_deadlines(deadline: float) -> None:
    with pytest.raises(ValueError, match="deadline must be finite"):
        ExecutionOptions(deadline=deadline)


@pytest.mark.asyncio
async def test_layer_projects_ordered_tool_intersection_and_keeps_context():
    invoker = RecordingInvoker()
    model = layer(invoker)
    context = Context(
        tools=(definition("weather"), definition("private"), definition("search"))
    )
    answer, returned_context = await model.invoke(
        UserMessage(content="question"), context
    )
    assert answer.content == "done"
    assert [tool.name for tool in invoker.tools] == ["search", "weather"]
    assert returned_context is context


@pytest.mark.asyncio
async def test_layer_without_local_invoker_rejects_direct_execution():
    with pytest.raises(RuntimeError, match="no local ModelInvoker"):
        await layer().invoke(UserMessage(content="question"), Context())


@pytest.mark.asyncio
async def test_streaming_invoker_drives_module_stream_and_final_result():
    class StreamingClient:
        async def invoke(self, route, payload):
            raise AssertionError("streaming invoker must use SSE")

        async def stream(self, route, payload):
            yield freeze_json_object(
                {"choices": [{"delta": {"content": "hel"}}]}
            )
            yield freeze_json_object(
                {"choices": [{"delta": {"content": "lo"}}]}
            )
            yield freeze_json_object({"usage": {"completion_tokens": 2}})
            yield freeze_json_object({"done": True})

        async def aclose(self):
            return None

    invoker = DefaultModelInvoker(
        adapters={"openai": OpenAICompatibleAdapter()},
        clients={"openai": StreamingClient()},
    )
    model = layer(invoker)
    async with model.stream(UserMessage(content="question"), Context()) as stream:
        events = [event async for event in stream]
        answer, _ = await stream.final_result()
    assert answer.content == "hello"
    assert [event.kind for event in events if event.kind.startswith("model.")] == [
        "model.started",
        "model.attempt.started",
        "model.text.delta",
        "model.text.delta",
        "model.usage",
        "model.attempt.succeeded",
        "model.completed",
    ]


def test_durable_effect_identity_includes_tool_results() -> None:
    model = layer()
    first = ToolMessage(
        results=(
            ToolResult(
                call_id="lookup-1",
                name="search",
                status="succeeded",
                output={"answer": 1},
                side_effect_committed=True,
            ),
        )
    )
    second = ToolMessage(
        results=(
            ToolResult(
                call_id="lookup-1",
                name="search",
                status="succeeded",
                output={"answer": 2},
                side_effect_committed=True,
            ),
        )
    )

    first_request = _model_effect_request(model, first, Context(), ())
    second_request = _model_effect_request(model, second, Context(), ())

    assert first_request != second_request


@pytest.mark.asyncio
async def test_managed_scope_can_supply_deployment_model_invoker() -> None:
    invoker = RecordingInvoker()
    model = layer()

    class ManagedScope:
        deadline = None

        def resolve_model_invoker(self, model_group):
            assert model_group == "assistant"
            return invoker

        async def emit_event(self, module, kind, data):
            return None

        @asynccontextmanager
        async def model_permit(self, resource_key=None, *, max_concurrency=None):
            yield

        async def execute_effect(self, *, spec, request, operation):
            return EffectOutcome(
                value=await operation(),
                disposition=EffectDisposition.EXECUTED,
                effect_id="test-effect",
            )

    token = _execution_scope.set(ManagedScope())
    try:
        answer, returned = await model.forward(
            UserMessage(content="question"), Context()
        )
    finally:
        _execution_scope.reset(token)

    assert answer.content == "done"
    assert returned == Context()


@pytest.mark.asyncio
async def test_raw_provider_fields_do_not_leak_to_message_context_or_events() -> None:
    canaries = (
        "synthetic-api-key-canary",
        "https://synthetic-endpoint.invalid/private",
        "provider-internal-stack-canary",
    )

    class RawClient:
        async def invoke(self, route, payload):
            return freeze_json_object(
                {
                    "id": "safe-request-id",
                    "choices": [{"message": {"content": "safe answer"}}],
                    "provider_debug": list(canaries),
                    "usage": {
                        "total_tokens": 2,
                        "debug": canaries[0],
                        "endpoint": canaries[1],
                        "internal": canaries[2],
                    },
                }
            )

        async def stream(self, route, payload):
            if False:
                yield

        async def aclose(self):
            return None

    invoker = DefaultModelInvoker(
        adapters={"openai": OpenAICompatibleAdapter()},
        clients={"main": RawClient()},
        capabilities={"openai": ModelProviderCapabilities(streaming=False)},
    )
    model = layer(invoker)
    original_context = Context(metadata={"request": "safe"})
    async with model.stream(UserMessage(content="question"), original_context) as stream:
        events = [event async for event in stream]
        answer, returned_context = await stream.final_result()

    assert answer.content == "safe answer"
    assert returned_context is original_context
    assert [event.kind for event in events if event.kind.startswith("model.")] == [
        "model.started",
        "model.attempt.started",
        "model.text.delta",
        "model.usage",
        "model.attempt.succeeded",
        "model.completed",
    ]
    public = json.dumps(
        {
            "message": repr(answer),
            "context": repr(returned_context),
            "events": [repr(event) for event in events],
        },
        sort_keys=True,
    )
    assert all(canary not in public for canary in canaries)


@pytest.mark.asyncio
async def test_provider_errors_are_sanitized_for_invoke_stream_and_run_events() -> None:
    secret = "synthetic-api-key-canary"
    endpoint = "https://synthetic-endpoint.invalid/private"
    internal = "provider-internal-stack-canary"
    canaries = (secret, endpoint, internal)
    calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls % 2:
            return httpx.Response(
                503,
                text=f"raw body {secret} {endpoint} {internal}",
            )
        raise RuntimeError(f"transport exploded: {secret} {endpoint} {internal}")

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider_client = OpenAICompatibleClient(
        base_url=endpoint,
        api_key=secret,
        client=http_client,
    )

    def failing_layer(*, streaming: bool) -> ModelCallLayer:
        invoker = DefaultModelInvoker(
            adapters={"openai": OpenAICompatibleAdapter()},
            clients={"primary": provider_client, "fallback": provider_client},
            capabilities={
                "openai": ModelProviderCapabilities(streaming=streaming)
            },
        )
        return ModelCallLayer(
            model_group=ModelGroupConfig(
                "sanitized-errors",
                (
                    ModelRoute("primary", "openai", "test"),
                    ModelRoute("fallback", "openai", "test"),
                ),
                FallbackPolicy(("primary", "fallback")),
            ),
            retry_policy=RetryPolicy(max_attempts_per_route=1),
            generation=GenerationConfig(),
            invoker=invoker,
        )

    runtime = LocalRuntime()
    handle = await runtime.bind(failing_layer(streaming=False)).start(
        UserMessage(content="question"),
        Context(metadata={"request": "safe"}),
        execution=ExecutionOptions(deadline=time.monotonic() + 1),
    )
    with pytest.raises(ModelCallError) as invoke_error:
        await handle.result()
    async with handle.subscribe() as subscription:
        managed_events = [event async for event in subscription]

    streamed_events = []
    with pytest.raises(ModelCallError) as stream_error:
        async with failing_layer(streaming=True).stream(
            UserMessage(content="question"), Context(metadata={"request": "safe"})
        ) as stream:
            async for event in stream:
                streamed_events.append(event)
            await stream.final_result()

    await runtime.close()
    await http_client.aclose()

    public = json.dumps(
        {
            "invoke_error_str": str(invoke_error.value),
            "invoke_error_repr": repr(invoke_error.value),
            "invoke_attempts": repr(invoke_error.value.attempts),
            "stream_error_str": str(stream_error.value),
            "stream_error_repr": repr(stream_error.value),
            "stream_attempts": repr(stream_error.value.attempts),
            "managed_events": [repr(event) for event in managed_events],
            "streamed_events": [repr(event) for event in streamed_events],
        },
        sort_keys=True,
    )
    assert all(canary not in public for canary in canaries)
    assert [
        event.kind for event in streamed_events if event.kind.startswith("model.")
    ] == [
        "model.started",
        "model.attempt.started",
        "model.usage",
        "model.attempt.failed",
        "model.attempt.started",
        "model.usage",
        "model.attempt.failed",
        "model.failed",
    ]
