from __future__ import annotations

import pytest

from pygent import Context, GenerationConfig, ToolDefinition, UserMessage
from pygent.llm import (
    DefaultModelInvoker,
    FallbackPolicy,
    ModelCallError,
    ModelErrorKind,
    ModelGroupConfig,
    ModelProviderError,
    ModelProviderRequest,
    ModelRoute,
    ModelStreamEvent,
    OpenAICompatibleAdapter,
    RetryPolicy,
)
from pygent.llm._request_snapshot import (
    MAX_MODEL_REQUEST_SNAPSHOT_BYTES,
    prepared_request_event,
)


def request(*, content: str = "question") -> ModelProviderRequest:
    return ModelProviderRequest(
        route=ModelRoute("primary", "openai", "model-1"),
        message=UserMessage(content=content),
        context=Context(
            system_prompt="fixed prompt",
            messages=(UserMessage(content="history"),),
            tools=(
                ToolDefinition(
                    name="read_file",
                    description="Read one file",
                    parameters={"type": "object"},
                ),
            ),
            metadata={"secret": "must-not-be-projected"},
            projection_revision=7,
        ),
        generation=GenerationConfig(temperature=0.2, max_output_tokens=100),
        tools=(
            ToolDefinition(
                name="read_file",
                description="Read one file",
                parameters={"type": "object"},
            ),
        ),
    )


def test_prepared_request_is_stable_allowlisted_and_validated() -> None:
    first = prepared_request_event(request(), attempt=1)
    second = prepared_request_event(request(), attempt=2)

    assert first["request_digest"] == second["request_digest"]
    assert first["request_id"] != second["request_id"]
    assert first["attempt"] == 1
    projection = first["request"]
    assert isinstance(projection, dict)
    assert projection["system_prompt"] == "fixed prompt"
    assert projection["projection_revision"] == 7
    assert "metadata" not in projection
    assert "must-not-be-projected" not in repr(projection)

    event = ModelStreamEvent("model.request.prepared", first)
    assert event.kind == "model.request.prepared"


def test_prepared_request_fails_closed_above_one_mib() -> None:
    oversized = "x" * (MAX_MODEL_REQUEST_SNAPSHOT_BYTES + 1)

    with pytest.raises(ModelProviderError) as raised:
        prepared_request_event(request(content=oversized), attempt=1)

    assert raised.value.kind is ModelErrorKind.INVALID_REQUEST


@pytest.mark.asyncio
async def test_oversized_snapshot_stops_before_provider_io() -> None:
    class NeverCalledClient:
        def __init__(self) -> None:
            self.calls = 0

        async def invoke(self, route, payload):
            self.calls += 1
            raise AssertionError("provider must not be called")

        async def stream(self, route, payload):
            self.calls += 1
            if False:
                yield

        async def aclose(self):
            return None

    client = NeverCalledClient()
    invoker = DefaultModelInvoker(
        adapters={"openai": OpenAICompatibleAdapter()},
        clients={"openai": client},
    )
    execution = invoker.execute(
        model_group=ModelGroupConfig(
            "snapshot-limit",
            (ModelRoute("primary", "openai", "model-1"),),
            FallbackPolicy(("primary",)),
        ),
        retry_policy=RetryPolicy(max_attempts_per_route=1),
        generation=GenerationConfig(),
        message=UserMessage(
            content="x" * (MAX_MODEL_REQUEST_SNAPSHOT_BYTES + 1)
        ),
        context=Context(),
    )
    async with execution.subscribe() as subscription:
        events = [event async for event in subscription]

    with pytest.raises(ModelCallError) as raised:
        await execution.result()

    assert raised.value.kind is ModelErrorKind.INVALID_REQUEST
    assert client.calls == 0
    assert [event.kind for event in events] == ["model.started", "model.failed"]
