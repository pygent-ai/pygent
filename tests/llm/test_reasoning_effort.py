"""Typed reasoning_effort knob: adapter mapping, conflicts, capability degradation."""

from __future__ import annotations

from dataclasses import replace

import pytest

from pygent import Context, GenerationConfig, RetryPolicy, UserMessage
from pygent.core import FrozenJsonObject, freeze_json_object
from pygent.llm import (
    AnthropicMessagesAdapter,
    GeminiGenerateContentAdapter,
    ModelCallError,
    ModelErrorKind,
    ModelProviderError,
    ModelProviderRequest,
    ModelReasoningCapabilities,
    OpenAICompatibleAdapter,
    OpenAIResponsesAdapter,
)
from pygent.llm.invoker import _missing_capabilities
from tests.support.model_specs import (
    configured_invoker,
    model_entry,
    model_group,
    transport_mode,
)


def _request(
    adapter_protocol: str,
    provider: str,
    *,
    generation: GenerationConfig,
    options: dict[str, object] | None = None,
) -> ModelProviderRequest:
    entry = model_entry("main", provider, "model")
    spec = replace(
        entry.spec,
        protocol=adapter_protocol,
        provider_options=options or {},
    )
    return ModelProviderRequest(
        model_key=entry.key,
        model=spec,
        message=UserMessage(content="hello"),
        context=Context(),
        generation=generation,
    )


def _responses_adapter() -> OpenAIResponsesAdapter:
    return OpenAIResponsesAdapter()


def _anthropic_adapter() -> AnthropicMessagesAdapter:
    return AnthropicMessagesAdapter()


class _CountingClient:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, route: object, payload: FrozenJsonObject) -> FrozenJsonObject:
        del route, payload
        self.calls += 1
        return freeze_json_object({})

    async def stream(self, route: object, payload: object):
        del route, payload
        self.calls += 1
        if False:
            yield freeze_json_object({})

    async def aclose(self) -> None:
        return None


class _PassthroughAdapter:
    """Minimal adapter: maps nothing, answers successfully."""

    protocol = "openai_chat_completions"

    def build_request(self, request: object) -> FrozenJsonObject:
        del request
        return freeze_json_object({})

    def parse_response(self, request: object, payload: object) -> object:
        del request, payload
        from pygent import AIMessage
        from pygent.llm import ModelProviderResponse

        return ModelProviderResponse(AIMessage(content="degraded"), {})

    def create_stream_decoder(self, request: object) -> object:
        raise AssertionError("provider I/O must not occur")

    def normalize_error(self, error: BaseException) -> ModelErrorKind:
        del error
        return ModelErrorKind.UNKNOWN


def test_responses_typed_effort_maps_to_reasoning_object() -> None:
    body = _responses_adapter().build_request(
        _request(
            "openai_responses",
            "openai",
            generation=GenerationConfig(reasoning_effort="high"),
        )
    )
    assert dict(body["reasoning"]) == {"effort": "high"}


def test_responses_typed_effort_rejects_unknown_value() -> None:
    with pytest.raises(ModelProviderError) as raised:
        _responses_adapter().build_request(
            _request(
                "openai_responses",
                "openai",
                generation=GenerationConfig(reasoning_effort="bogus"),
            )
        )
    assert raised.value.kind is ModelErrorKind.INVALID_REQUEST


def test_responses_typed_effort_conflicts_with_provider_option() -> None:
    with pytest.raises(ModelProviderError) as raised:
        _responses_adapter().build_request(
            _request(
                "openai_responses",
                "openai",
                generation=GenerationConfig(reasoning_effort="high"),
                options={"reasoning": {"effort": "low"}},
            )
        )
    assert raised.value.kind is ModelErrorKind.INVALID_REQUEST


def test_anthropic_typed_effort_maps_to_output_config() -> None:
    body = _anthropic_adapter().build_request(
        _request(
            "anthropic_messages",
            "anthropic",
            generation=GenerationConfig(
                max_output_tokens=64, reasoning_effort="max"
            ),
        )
    )
    assert body["output_config"]["effort"] == "max"


def test_anthropic_typed_effort_conflicts_with_provider_option() -> None:
    with pytest.raises(ModelProviderError) as raised:
        _anthropic_adapter().build_request(
            _request(
                "anthropic_messages",
                "anthropic",
                generation=GenerationConfig(
                    max_output_tokens=64, reasoning_effort="high"
                ),
                options={"output_config": {"effort": "low"}},
            )
        )
    assert raised.value.kind is ModelErrorKind.INVALID_REQUEST


def test_anthropic_typed_effort_rejects_value_outside_native_set() -> None:
    with pytest.raises(ModelProviderError) as raised:
        _anthropic_adapter().build_request(
            _request(
                "anthropic_messages",
                "anthropic",
                generation=GenerationConfig(
                    max_output_tokens=64, reasoning_effort="minimal"
                ),
            )
        )
    assert raised.value.kind is ModelErrorKind.INVALID_REQUEST


def test_openai_compatible_typed_effort_conflicts_with_provider_option() -> None:
    with pytest.raises(ModelProviderError) as raised:
        OpenAICompatibleAdapter().build_request(
            _request(
                "openai_chat_completions",
                "openai",
                generation=GenerationConfig(reasoning_effort="high"),
                options={"reasoning_effort": "low"},
            )
        )
    assert raised.value.kind is ModelErrorKind.INVALID_REQUEST


def test_gemini_ignores_typed_effort_without_failing() -> None:
    body = GeminiGenerateContentAdapter().build_request(
        _request(
            "gemini_generate_content",
            "google",
            generation=GenerationConfig(
                max_output_tokens=64, reasoning_effort="high"
            ),
        )
    )
    assert "thinkingConfig" not in body
    assert "reasoning_effort" not in body


def _entry_with_reasoning_capabilities(
    *, supported: bool, controllable: bool
):
    entry = model_entry("main", "openai", "model")
    capabilities = replace(
        entry.spec.capabilities,
        reasoning=ModelReasoningCapabilities(
            supported=supported, controllable=controllable
        ),
    )
    return replace(entry, spec=replace(entry.spec, capabilities=capabilities))


@pytest.mark.parametrize(
    ("supported", "controllable", "expected"),
    [
        (True, True, ()),
        (True, False, ("reasoning.controllable",)),
        (False, False, ("reasoning.controllable",)),
    ],
)
def test_missing_capabilities_report_reasoning_effort_gaps(
    supported: bool,
    controllable: bool,
    expected: tuple[str, ...],
) -> None:
    entry = _entry_with_reasoning_capabilities(
        supported=supported, controllable=controllable
    )
    gaps = _missing_capabilities(
        entry.spec,
        generation=GenerationConfig(reasoning_effort="high"),
        tools=(),
    )
    assert gaps == expected


def test_missing_capabilities_stay_quiet_without_effort() -> None:
    entry = _entry_with_reasoning_capabilities(supported=False, controllable=False)
    gaps = _missing_capabilities(
        entry.spec,
        generation=GenerationConfig(),
        tools=(),
    )
    assert "reasoning.controllable" not in gaps


@pytest.mark.asyncio
async def test_capabilityless_model_degrades_with_warning_and_proceeds() -> None:
    client = _CountingClient()
    invoker = configured_invoker(
        adapters={"openai_chat_completions": _PassthroughAdapter()},
        clients={"main": client},
        capabilities={"openai": transport_mode(streaming=False)},
    )
    entry = _entry_with_reasoning_capabilities(supported=False, controllable=False)
    execution = invoker.execute(
        model_group=model_group("g", (entry,), ("main",)),
        retry_policy=RetryPolicy(max_attempts_per_model=1),
        generation=GenerationConfig(reasoning_effort="high"),
        message=UserMessage(content="hello"),
        context=Context(),
    )
    response = await execution.result()
    assert client.calls == 1
    assert response.message is not None
