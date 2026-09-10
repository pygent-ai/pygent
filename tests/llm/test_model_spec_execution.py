from __future__ import annotations

from dataclasses import replace

import httpx
import pytest

from pygent import Context, UserMessage
from pygent.core import FrozenJsonObject, freeze_json_object
from pygent.llm import (
    CapabilityPresetCatalog,
    DefaultModelInvoker,
    ExponentialBackoff,
    GenerationConfig,
    ModelCallLayer,
    ModelEntry,
    ModelErrorKind,
    ModelGroup,
    ModelSpec,
    ModelStreamingCapabilities,
    OpenAICompatibleAdapter,
    RetryPolicy,
)
from pygent.tool import ToolDefinition


class FakeClient:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    async def invoke(self, model, payload):
        del model, payload
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def stream(self, model, payload):
        del model, payload
        self.calls += 1
        for outcome in self.outcomes:
            if isinstance(outcome, BaseException):
                raise outcome
            yield outcome

    async def aclose(self) -> None:
        return None


def _completion(content: str = "ok") -> FrozenJsonObject:
    return freeze_json_object(
        {
            "choices": [
                {"message": {"content": content}, "finish_reason": "stop"}
            ],
            "usage": {},
        }
    )


def _entry(
    name: str,
    model_id: str,
    *,
    preset: str = "text",
    max_output_tokens: int = 4096,
) -> ModelEntry:
    capabilities = CapabilityPresetCatalog.builtin().presets[preset].materialize(
        context_tokens=32_768,
        max_output_tokens=max_output_tokens,
    )
    capabilities = replace(
        capabilities,
        streaming=ModelStreamingCapabilities(text=False),
    )
    return ModelEntry(
        name,
        ModelSpec(
            provider="custom",
            model_id=model_id,
            protocol="openai_compatible",
            capabilities=capabilities,
        ),
    )


def test_model_call_layer_requires_exactly_one_model_source() -> None:
    entry = _entry("primary", "first")
    group = ModelGroup("assistant", (entry,))

    layer = ModelCallLayer(
        model=entry,
        retry_policy=RetryPolicy(),
        generation=GenerationConfig(),
    )
    assert layer.model_group == ModelGroup("primary", (entry,))
    with pytest.raises(ValueError, match="exactly one"):
        ModelCallLayer(
            model=entry,
            model_group=group,
            retry_policy=RetryPolicy(),
            generation=GenerationConfig(),
        )
    with pytest.raises(ValueError, match="exactly one"):
        ModelCallLayer(
            retry_policy=RetryPolicy(),
            generation=GenerationConfig(),
        )


@pytest.mark.asyncio
async def test_invoker_dispatches_by_protocol_and_falls_back_in_model_order() -> None:
    primary = FakeClient([httpx.ConnectError("offline")])
    fallback = FakeClient([_completion("fallback")])
    group = ModelGroup(
        "assistant",
        (_entry("primary", "first"), _entry("fallback", "second")),
    )
    invoker = DefaultModelInvoker(
        adapters={"openai_compatible": OpenAICompatibleAdapter()},
        clients={"primary": primary, "fallback": fallback},
    )

    execution = invoker.execute(
        model_group=group,
        retry_policy=RetryPolicy(
            max_attempts_per_route=1,
            retry_on=(ModelErrorKind.UNAVAILABLE,),
            backoff=ExponentialBackoff(0, 0),
        ),
        generation=GenerationConfig(),
        message=UserMessage(content="hello"),
        context=Context(),
    )
    result = await execution.result()
    async with execution.subscribe() as events:
        captured = [event async for event in events]

    assert result.message.content == "fallback"
    assert primary.calls == 1
    assert fallback.calls == 1
    started = [event for event in captured if event.kind == "model.attempt.started"]
    assert [event.data["model_key"] for event in started] == ["primary", "fallback"]
    assert all("route_id" not in event.data for event in captured)


@pytest.mark.asyncio
async def test_capability_warning_is_once_per_reached_model_not_retry() -> None:
    primary = FakeClient(
        [httpx.ConnectError("offline"), httpx.ConnectError("offline")]
    )
    fallback = FakeClient([_completion("{}")])
    group = ModelGroup(
        "assistant",
        (
            _entry("primary", "first", max_output_tokens=16),
            _entry("fallback", "second", max_output_tokens=16),
        ),
    )
    invoker = DefaultModelInvoker(
        adapters={"openai_compatible": OpenAICompatibleAdapter()},
        clients={"primary": primary, "fallback": fallback},
    )
    tool = ToolDefinition(
        name="weather",
        description="Weather",
        parameters={"type": "object"},
    )

    execution = invoker.execute(
        model_group=group,
        retry_policy=RetryPolicy(
            max_attempts_per_route=2,
            retry_on=(ModelErrorKind.UNAVAILABLE,),
            backoff=ExponentialBackoff(0, 0),
        ),
        generation=GenerationConfig(
            max_output_tokens=32,
            tool_choice="required",
            response_schema={"type": "object"},
            response_schema_name="answer",
        ),
        message=UserMessage(content="hello"),
        context=Context(),
        tools=(tool,),
    )
    await execution.result()
    async with execution.subscribe() as events:
        captured = [event async for event in events]

    warnings = [
        event for event in captured if event.kind == "model.capability.warning"
    ]
    assert [event.data["model_key"] for event in warnings] == ["primary", "fallback"]
    assert all(
        event.data["missing_capabilities"]
        == (
            "tools.call",
            "tools.choice.required",
            "structured_output.json_schema",
            "limits.max_output_tokens",
        )
        for event in warnings
    )


@pytest.mark.asyncio
async def test_matching_capabilities_emit_no_warning() -> None:
    client = FakeClient([_completion()])
    invoker = DefaultModelInvoker(
        adapters={"openai_compatible": OpenAICompatibleAdapter()},
        clients={"primary": client},
    )
    execution = invoker.execute(
        model_group=ModelGroup(
            "assistant",
            (_entry("primary", "first", preset="text_tools_structured_reasoning"),),
        ),
        retry_policy=RetryPolicy(),
        generation=GenerationConfig(max_output_tokens=32),
        message=UserMessage(content="hello"),
        context=Context(),
    )
    await execution.result()
    async with execution.subscribe() as events:
        captured = [event async for event in events]
    assert "model.capability.warning" not in [event.kind for event in captured]
