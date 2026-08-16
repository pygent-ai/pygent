from __future__ import annotations

import math
from typing import Any, cast

import httpx
import pytest

from pygent import (
    Context,
    FallbackPolicy,
    GenerationConfig,
    ModelGroupConfig,
    ModelRoute,
    RetryPolicy,
    UserMessage,
)
from pygent.core import FrozenJsonObject, JsonValueError, freeze_json_object
from pygent.llm import (
    DefaultModelInvoker,
    ModelCallError,
    ModelErrorKind,
    ModelGroupConfigurationError,
    ModelProviderCapabilities,
    ModelProviderError,
    ModelProviderRequest,
    OpenAICompatibleAdapter,
    openai_compatible_adapters,
)


def _request(route: ModelRoute) -> ModelProviderRequest:
    return ModelProviderRequest(
        route=route,
        message=UserMessage(content="hello"),
        context=Context(),
        generation=GenerationConfig(),
    )


def test_model_route_provider_options_are_keyword_only_frozen_and_hidden() -> None:
    raw: dict[str, Any] = {
        "thinking": {"type": "disabled"},
        "items": [1, {"ok": True}],
    }
    route = ModelRoute(
        "main", "deepseek", "deepseek-chat", provider_options=raw
    )
    cast(dict[str, str], raw["thinking"])["type"] = "enabled"
    cast(list[object], raw["items"]).append(2)

    assert isinstance(route.provider_options, FrozenJsonObject)
    thinking = cast(FrozenJsonObject, route.provider_options["thinking"])
    assert thinking["type"] == "disabled"
    assert route.provider_options["items"] == (1, freeze_json_object({"ok": True}))
    assert "disabled" not in repr(route)
    with pytest.raises(TypeError):
        ModelRoute("main", "deepseek", "deepseek-chat", {})  # type: ignore[call-arg]


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
        ModelRoute("main", "custom", "model", provider_options=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [[], "options", 1, None])
def test_model_route_rejects_non_object_provider_options(value: object) -> None:
    with pytest.raises(JsonValueError):
        ModelRoute("main", "custom", "model", provider_options=value)  # type: ignore[arg-type]


def test_openai_compatible_projects_deepseek_and_generic_options() -> None:
    deepseek = ModelRoute(
        "main",
        "deepseek",
        "deepseek-chat",
        provider_options={"thinking": {"type": "disabled"}},
    )
    payload = OpenAICompatibleAdapter("deepseek").build_request(_request(deepseek))
    assert payload["thinking"] == freeze_json_object({"type": "disabled"})

    custom = ModelRoute(
        "main",
        "custom",
        "custom-model",
        provider_options={"vendor_feature": {"mode": "fast"}},
    )
    payload = OpenAICompatibleAdapter("custom").build_request(_request(custom))
    assert payload["vendor_feature"] == freeze_json_object({"mode": "fast"})
    assert "deepseek" in openai_compatible_adapters()


@pytest.mark.parametrize(
    "options",
    [
        {"model": "other"},
        {"stream": True},
        {"nested": {"api_key": "not-allowed"}},
        {"secret": "not-allowed"},
        {"retry": 3},
        {"endpoint": "https://example.invalid"},
    ],
)
def test_openai_compatible_rejects_reserved_or_nonportable_options(
    options: dict[str, object],
) -> None:
    route = ModelRoute("main", "custom", "model", provider_options=options)
    with pytest.raises(ModelProviderError) as raised:
        OpenAICompatibleAdapter("custom").build_request(_request(route))
    assert raised.value.kind is ModelErrorKind.INVALID_REQUEST
    assert "not-allowed" not in str(raised.value)


@pytest.mark.parametrize(
    "thinking",
    [{"type": "automatic"}, {"type": "disabled", "extra": True}, "disabled"],
)
def test_deepseek_thinking_schema_is_strict(thinking: object) -> None:
    route = ModelRoute(
        "main",
        "deepseek",
        "deepseek-chat",
        provider_options={"thinking": thinking},
    )
    with pytest.raises(ModelProviderError) as raised:
        OpenAICompatibleAdapter("deepseek").build_request(_request(route))
    assert raised.value.kind is ModelErrorKind.INVALID_REQUEST


class _NoValidatorAdapter:
    provider = "custom"

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
    route = ModelRoute(
        "main", "custom", "model", provider_options={"vendor_feature": True}
    )
    invoker = DefaultModelInvoker(
        adapters={"custom": _NoValidatorAdapter()},  # type: ignore[dict-item]
        clients={"custom": client},
    )
    with pytest.raises(ModelGroupConfigurationError):
        invoker.validate_route(route)

    execution = invoker.execute(
        model_group=ModelGroupConfig(
            "custom", (route,), FallbackPolicy(("main",))
        ),
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
async def test_fallback_routes_receive_only_their_own_provider_options() -> None:
    primary = _OutcomeClient(httpx.ConnectError("offline"))
    fallback = _OutcomeClient(
        freeze_json_object(
            {"choices": [{"message": {"content": "fallback"}}]}
        )
    )
    routes = (
        ModelRoute(
            "primary",
            "openai",
            "primary-model",
            provider_options={"primary_feature": True},
        ),
        ModelRoute(
            "fallback",
            "openai",
            "fallback-model",
            provider_options={"fallback_feature": True},
        ),
    )
    invoker = DefaultModelInvoker(
        adapters={"openai": OpenAICompatibleAdapter()},
        clients={"primary": primary, "fallback": fallback},
        capabilities={"openai": ModelProviderCapabilities(streaming=False)},
    )
    response = await invoker.execute(
        model_group=ModelGroupConfig(
            "fallback", routes, FallbackPolicy(("primary", "fallback"))
        ),
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
