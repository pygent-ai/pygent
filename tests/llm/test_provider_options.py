from __future__ import annotations

import math
from typing import Any, cast

import httpx
import pytest

from pygent import (
    Context,
    GenerationConfig,
    RetryPolicy,
    UserMessage,
)
from pygent.core import FrozenJsonObject, JsonValueError, freeze_json_object
from pygent.llm import (
    ModelCallError,
    ModelEntry,
    ModelErrorKind,
    ModelGroupConfigurationError,
    ModelProviderError,
    ModelProviderRequest,
    OpenAICompatibleAdapter,
    openai_compatible_adapters,
)
from tests.support.model_specs import (
    configured_invoker,
    model_entry,
    model_group,
    transport_mode,
)


def _request(
    entry: ModelEntry, generation: GenerationConfig | None = None
) -> ModelProviderRequest:
    return ModelProviderRequest(
        model_key=entry.name,
        model=entry.spec,
        message=UserMessage(content="hello"),
        context=Context(),
        generation=generation or GenerationConfig(),
    )


def test_model_route_provider_options_are_keyword_only_frozen_and_hidden() -> None:
    raw: dict[str, Any] = {
        "thinking": {"type": "disabled"},
        "items": [1, {"ok": True}],
    }
    route = model_entry("main", "deepseek", "deepseek-chat", provider_options=raw)
    cast(dict[str, str], raw["thinking"])["type"] = "enabled"
    cast(list[object], raw["items"]).append(2)

    assert isinstance(route.spec.provider_options, FrozenJsonObject)
    thinking = cast(FrozenJsonObject, route.spec.provider_options["thinking"])
    assert thinking["type"] == "disabled"
    assert route.spec.provider_options["items"] == (1, freeze_json_object({"ok": True}))
    assert "disabled" not in repr(route)
    with pytest.raises(TypeError):
        model_entry("main", "deepseek", "deepseek-chat", {})  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "value",
    [
        {"bad": math.nan},
        {"bad": math.inf},
        {"bad": b"bytes"},
        {"bad": lambda: None},
    ],
)
def test_model_route_rejects_non_json_provider_options(value: object) -> None:
    with pytest.raises(JsonValueError):
        model_entry("main", "custom", "model", provider_options=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [[], "options", 1, None])
def test_model_route_rejects_non_object_provider_options(value: object) -> None:
    with pytest.raises(TypeError):
        model_entry("main", "custom", "model", provider_options=value)  # type: ignore[arg-type]


def test_openai_compatible_projects_deepseek_and_generic_options() -> None:
    deepseek = model_entry(
        "main",
        "deepseek",
        "deepseek-chat",
        provider_options={"thinking": {"type": "disabled"}},
    )
    payload = OpenAICompatibleAdapter().build_request(_request(deepseek))
    assert payload["thinking"] == freeze_json_object({"type": "disabled"})

    custom = model_entry(
        "main",
        "custom",
        "custom-model",
        provider_options={"vendor_feature": {"mode": "fast"}},
    )
    payload = OpenAICompatibleAdapter().build_request(_request(custom))
    assert payload["vendor_feature"] == freeze_json_object({"mode": "fast"})
    assert set(openai_compatible_adapters()) == {"openai_compatible"}


@pytest.mark.parametrize("field", ["max_tokens", "max_completion_tokens"])
def test_openai_compatible_accepts_one_route_token_limit(field: str) -> None:
    route = model_entry(
        "main", "custom", "model", provider_options={field: 4096}
    )

    payload = OpenAICompatibleAdapter().build_request(_request(route))

    assert payload[field] == 4096
    assert set(payload) & {"max_tokens", "max_completion_tokens"} == {field}


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "4096"])
def test_openai_compatible_rejects_invalid_route_token_limit(value: object) -> None:
    route = model_entry(
        "main", "custom", "model", provider_options={"max_tokens": value}
    )

    with pytest.raises(ModelProviderError) as raised:
        OpenAICompatibleAdapter().build_request(_request(route))

    assert raised.value.kind is ModelErrorKind.INVALID_REQUEST


def test_openai_compatible_rejects_ambiguous_or_conflicting_token_limits() -> None:
    adapter = OpenAICompatibleAdapter()
    both = model_entry(
        "main",
        "custom",
        "model",
        provider_options={"max_tokens": 1024, "max_completion_tokens": 1024},
    )
    with pytest.raises(ModelProviderError):
        adapter.build_request(_request(both))

    configured = model_entry(
        "main", "custom", "model", provider_options={"max_completion_tokens": 1024}
    )
    with pytest.raises(ModelProviderError) as raised:
        adapter.build_request(
            _request(configured, GenerationConfig(max_output_tokens=2048))
        )
    assert raised.value.kind is ModelErrorKind.INVALID_REQUEST


@pytest.mark.parametrize(
    "options",
    [
        {"model": "other"},
        {"stream": True},
        {"nested": {"api_key": "not-allowed"}},
        {"secret": "not-allowed"},
        {"retry": 3},
        {"endpoint": "https://example.invalid"},
        {"verify_ssl": False},
    ],
)
def test_openai_compatible_rejects_reserved_or_nonportable_options(
    options: dict[str, object],
) -> None:
    route = model_entry("main", "custom", "model", provider_options=options)
    with pytest.raises(ModelProviderError) as raised:
        OpenAICompatibleAdapter().build_request(_request(route))
    assert raised.value.kind is ModelErrorKind.INVALID_REQUEST
    assert "not-allowed" not in str(raised.value)


@pytest.mark.parametrize(
    "thinking",
    [{"type": "automatic"}, {"type": "disabled", "extra": True}, "disabled"],
)
def test_deepseek_thinking_schema_is_strict(thinking: object) -> None:
    route = model_entry(
        "main",
        "deepseek",
        "deepseek-chat",
        provider_options={"thinking": thinking},
    )
    with pytest.raises(ModelProviderError) as raised:
        OpenAICompatibleAdapter().build_request(_request(route))
    assert raised.value.kind is ModelErrorKind.INVALID_REQUEST


class _NoValidatorAdapter:
    protocol = "openai_compatible"

    def build_request(self, request: object) -> FrozenJsonObject:
        del request
        return freeze_json_object({})

    def parse_response(self, request: object, payload: object) -> object:
        raise AssertionError("provider I/O must not occur")

    def parse_stream_events(self, request: object, payload: object) -> tuple[()]:
        raise AssertionError("provider I/O must not occur")

    def normalize_error(self, error: BaseException) -> ModelErrorKind:
        del error
        return ModelErrorKind.UNKNOWN


class _CountingClient:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(
        self, route: object, payload: FrozenJsonObject
    ) -> FrozenJsonObject:
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


class _OutcomeClient(_CountingClient):
    def __init__(self, outcome: FrozenJsonObject | BaseException) -> None:
        super().__init__()
        self.outcome = outcome
        self.payloads: list[FrozenJsonObject] = []

    async def invoke(
        self, route: object, payload: FrozenJsonObject
    ) -> FrozenJsonObject:
        del route
        self.calls += 1
        self.payloads.append(payload)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


@pytest.mark.asyncio
async def test_third_party_adapter_without_validator_fails_closed_before_io() -> None:
    client = _CountingClient()
    route = model_entry(
        "main", "custom", "model", provider_options={"vendor_feature": True}
    )
    invoker = configured_invoker(
        adapters={"openai_compatible": _NoValidatorAdapter()},
        clients={"main": client},
    )
    with pytest.raises(ModelGroupConfigurationError):
        invoker.validate_model(route)

    execution = invoker.execute(
        model_group=model_group("custom", (route,), ("main",)),
        retry_policy=RetryPolicy(),
        generation=GenerationConfig(),
        message=UserMessage(content="hello"),
        context=Context(),
    )
    with pytest.raises(ModelCallError) as raised:
        await execution.result()
    assert raised.value.kind is ModelErrorKind.INVALID_REQUEST
    assert client.calls == 0


@pytest.mark.asyncio
async def test_fallback_models_receive_only_their_own_provider_options() -> None:
    primary = _OutcomeClient(httpx.ConnectError("offline"))
    fallback = _OutcomeClient(
        freeze_json_object({"choices": [{"message": {"content": "fallback"}}]})
    )
    models = (
        model_entry(
            "primary",
            "openai",
            "primary-model",
            provider_options={"primary_feature": True},
        ),
        model_entry(
            "fallback",
            "openai",
            "fallback-model",
            provider_options={"fallback_feature": True},
        ),
    )
    invoker = configured_invoker(
        adapters={"openai": OpenAICompatibleAdapter()},
        clients={"primary": primary, "fallback": fallback},
        capabilities={"openai": transport_mode(streaming=False)},
    )
    response = await invoker.execute(
        model_group=model_group("fallback", models, ("primary", "fallback")),
        retry_policy=RetryPolicy(max_attempts_per_route=1),
        generation=GenerationConfig(),
        message=UserMessage(content="hello"),
        context=Context(),
    ).result()

    assert response.message.content == "fallback"
    assert primary.payloads[0]["primary_feature"] is True
    assert "fallback_feature" not in primary.payloads[0]
    assert fallback.payloads[0]["fallback_feature"] is True
    assert "primary_feature" not in fallback.payloads[0]
