from __future__ import annotations

import pytest

from pygent.llm import (
    CapabilityPresetCatalog,
    ModelCapabilityCatalog,
    ProviderCatalog,
)
from tests.live.az_conformance.schemas import RouteKind, load_manifest

ALIYUN_TEXT_MODELS = {
    "qwen3.8-max",
    "qwen3.8-flash",
    "qwen3.7-max",
    "qwen3.7-plus",
    "qwen3.6-flash",
    "deepseek-v4-pro",
    "deepseek-v4-pro-0813",
    "deepseek-v4-flash-0731",
    "glm-5.2",
}

ALIYUN_SPECIALIZED_MODELS = {
    "qwen-image-3.0-pro",
    "wan2.7-image",
    "wan2.7-image-pro",
    "happyhorse-1.1-i2v",
    "happyhorse-1.1-t2v",
    "happyhorse-1.1-r2v",
    "qwen-audio-3.0-tts-plus",
    "qwen-audio-3.0-realtime-plus",
    "qwen-audio-3.0-asr-flash",
}

OFFICIAL_ROUTE_PROTOCOLS = {
    "openai": {"openai_chat_completions"},
    "anthropic": {"anthropic_messages"},
    "google": {"gemini_generate_content"},
    "alibaba_cloud": {"openai_chat_completions"},
    "deepseek": {"openai_chat_completions", "anthropic_messages"},
    "zhipu": {"openai_chat_completions"},
    "moonshot": {"openai_chat_completions", "anthropic_messages"},
    "minimax": {"openai_chat_completions", "anthropic_messages"},
    "volcengine": {"openai_chat_completions"},
    "xai": {"openai_chat_completions"},
}


def _complete_capability_mapping(
    *,
    input_modalities: tuple[str, ...],
    output_modalities: tuple[str, ...],
    streaming_output: tuple[str, ...],
    tools: bool,
    json_object: bool,
    json_schema: bool,
    reasoning: bool,
    context_tokens: int | None,
    max_output_tokens: int | None,
) -> dict[str, object]:
    return {
        "modalities": {
            "input": list(input_modalities),
            "output": list(output_modalities),
        },
        "streaming": {"output": list(streaming_output)},
        "tools": {
            "call": tools,
            "choice": ["none", "auto", "required", "named"] if tools else [],
            "parallel": tools,
        },
        "structured_output": {
            "json_object": json_object,
            "json_schema": json_schema,
        },
        "reasoning": {"supported": reasoning, "controllable": reasoning},
        "limits": {
            "context_tokens": context_tokens,
            "max_output_tokens": max_output_tokens,
        },
    }


def test_builtin_provider_catalog_projects_protocol_specific_connections() -> None:
    catalog = ProviderCatalog.builtin()

    assert tuple(catalog.providers) == (
        "deepseek",
        "anthropic",
        "openai",
        "google",
        "alibaba_cloud",
        "zhipu",
        "moonshot",
        "minimax",
        "volcengine",
        "xai",
        "aliyun_token_plan",
    )
    deepseek = catalog.providers["deepseek"]
    assert deepseek.display_name == "DeepSeek"
    assert deepseek.default_protocol == "openai_chat_completions"
    openai = deepseek.protocols["openai_chat_completions"]
    assert openai.protocol == "openai_chat_completions"
    assert openai.base_url == "https://api.deepseek.com"
    assert openai.authentication == "bearer"
    assert openai.api_key_env == "DEEPSEEK_API_KEY"
    assert "thinking" in openai.provider_options_schema["properties"]
    anthropic_wire = deepseek.protocols["anthropic_messages"]
    assert anthropic_wire.base_url == "https://api.deepseek.com/anthropic"
    assert anthropic_wire.api_key_env == "DEEPSEEK_API_KEY"
    assert set(anthropic_wire.provider_options_schema["properties"]) == {
        "thinking",
        "output_config",
        "service_tier",
        "stop_sequences",
    }
    anthropic = catalog.providers["anthropic"]
    assert anthropic.default_protocol == "anthropic_messages"
    assert (
        anthropic.protocols["anthropic_messages"].base_url
        == "https://api.anthropic.com"
    )
    assert anthropic.protocols["anthropic_messages"].api_key_env == "ANTHROPIC_API_KEY"
    assert (
        anthropic.protocols["anthropic_messages"].provider_options_schema
        == anthropic_wire.provider_options_schema
    )


def test_builtin_provider_catalog_has_official_multi_provider_connections() -> None:
    catalog = ProviderCatalog.builtin()

    expected = {
        "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
        "google": (
            "https://generativelanguage.googleapis.com/v1beta",
            "GEMINI_API_KEY",
        ),
        "alibaba_cloud": (
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "DASHSCOPE_API_KEY",
        ),
        "zhipu": ("https://open.bigmodel.cn/api/paas/v4", "ZHIPU_API_KEY"),
        "moonshot": ("https://api.moonshot.cn/v1", "MOONSHOT_API_KEY"),
        "minimax": ("https://api.minimax.io/v1", "MINIMAX_API_KEY"),
        "volcengine": (
            "https://ark.cn-beijing.volces.com/api/v3",
            "ARK_API_KEY",
        ),
        "xai": ("https://api.x.ai/v1", "XAI_API_KEY"),
    }
    for provider, (base_url, api_key_env) in expected.items():
        preset = catalog.providers[provider]
        protocol = preset.protocols[preset.default_protocol]
        assert protocol.base_url == base_url
        assert protocol.api_key_env == api_key_env

    assert catalog.providers["openai"].protocols["openai_responses"].base_url == (
        "https://api.openai.com/v1"
    )
    assert catalog.providers["moonshot"].protocols["anthropic_messages"].base_url == (
        "https://api.moonshot.cn/anthropic"
    )
    assert catalog.providers["minimax"].protocols["anthropic_messages"].base_url == (
        "https://api.minimax.io/anthropic"
    )


def test_builtin_catalog_projects_every_official_manifest_model() -> None:
    manifest = load_manifest()
    expected = {
        (route.canonical_provider, route.canonical_model_id, protocol)
        for route in manifest.routes
        if route.catalog_eligible
        and route.canonical_provider in OFFICIAL_ROUTE_PROTOCOLS
        for protocol in route.protocols
        if protocol in OFFICIAL_ROUTE_PROTOCOLS[route.canonical_provider]
    }
    actual = {
        key
        for key in ModelCapabilityCatalog.builtin().models
        if key[0] in OFFICIAL_ROUTE_PROTOCOLS
        and key[2] in OFFICIAL_ROUTE_PROTOCOLS[key[0]]
    }

    assert actual == expected


def test_az_gateway_aliases_never_enter_builtin_catalog() -> None:
    manifest = load_manifest()
    forbidden = {
        route.route_id
        for route in manifest.routes
        if route.kind is not RouteKind.OFFICIAL_MODEL
    }

    assert forbidden.isdisjoint(
        model_id for _, model_id, _ in ModelCapabilityCatalog.builtin().models
    )


def test_builtin_token_plan_provider_has_two_executable_protocol_presets() -> None:
    preset = ProviderCatalog.builtin().providers["aliyun_token_plan"]

    assert preset.display_name == "Alibaba Cloud Token Plan"
    assert preset.default_protocol == "openai_chat_completions"
    assert set(preset.protocols) == {
        "openai_chat_completions",
        "anthropic_messages",
    }
    assert (
        preset.protocols["openai_chat_completions"].base_url
        == "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    )
    assert preset.protocols["openai_chat_completions"].api_key_env == (
        "ALIYUN_TOKEN_PLAN_OPENAI_API_KEY"
    )
    openai_schema = preset.protocols[
        "openai_chat_completions"
    ].provider_options_schema
    assert set(openai_schema["properties"]) == {
        "enable_thinking",
        "preserve_thinking",
        "reasoning_effort",
        "thinking",
        "thinking_budget",
        "tool_stream",
    }
    assert openai_schema["additionalProperties"] is False
    assert (
        preset.protocols["anthropic_messages"].base_url
        == "https://token-plan.cn-beijing.maas.aliyuncs.com/apps/anthropic"
    )
    assert preset.protocols["anthropic_messages"].api_key_env == (
        "ALIYUN_TOKEN_PLAN_ANTHROPIC_API_KEY"
    )


def test_builtin_token_plan_catalog_has_all_18_models_and_27_records() -> None:
    records = {
        key: value
        for key, value in ModelCapabilityCatalog.builtin().models.items()
        if key[0] == "aliyun_token_plan"
    }

    assert {key[1] for key in records} == (
        ALIYUN_TEXT_MODELS | ALIYUN_SPECIALIZED_MODELS
    )
    assert len(records) == 27
    for model_id in ALIYUN_TEXT_MODELS:
        assert (
            "aliyun_token_plan",
            model_id,
            "openai_chat_completions",
        ) in records
        assert ("aliyun_token_plan", model_id, "anthropic_messages") in records


@pytest.mark.parametrize(
    "model_id,input_modalities",
    [
        ("qwen3.8-max", ("text", "image")),
        ("qwen3.8-flash", ("text", "image")),
        ("qwen3.7-max", ("text",)),
        ("qwen3.7-plus", ("text", "image")),
        ("qwen3.6-flash", ("text", "image")),
        ("deepseek-v4-pro", ("text",)),
        ("deepseek-v4-pro-0813", ("text",)),
        ("deepseek-v4-flash-0731", ("text",)),
        ("glm-5.2", ("text",)),
    ],
)
@pytest.mark.parametrize(
    "protocol", ["openai_chat_completions", "anthropic_messages"]
)
def test_builtin_token_plan_text_capability_record_is_complete(
    model_id: str,
    input_modalities: tuple[str, ...],
    protocol: str,
) -> None:
    modern_qwen = model_id.startswith(("qwen3.7", "qwen3.8"))
    json_schema = (
        modern_qwen
        if protocol == "openai_chat_completions"
        else model_id != "qwen3.6-flash"
    )

    capabilities = ModelCapabilityCatalog.builtin().models[
        ("aliyun_token_plan", model_id, protocol)
    ]

    assert capabilities.to_mapping() == _complete_capability_mapping(
        input_modalities=input_modalities,
        output_modalities=("text",),
        streaming_output=("text",),
        tools=True,
        json_object=True,
        json_schema=json_schema,
        reasoning=True,
        context_tokens=1_000_000,
        max_output_tokens=None,
    )


@pytest.mark.parametrize(
    (
        "model_id",
        "protocol",
        "input_modalities",
        "output_modalities",
        "streaming_output",
        "tools",
        "context_tokens",
        "max_output_tokens",
    ),
    [
        (
            "qwen-image-3.0-pro",
            "dashscope_multimodal_generation",
            ("text", "image"),
            ("image",),
            (),
            False,
            None,
            None,
        ),
        (
            "wan2.7-image",
            "dashscope_multimodal_generation",
            ("text", "image"),
            ("image",),
            (),
            False,
            None,
            None,
        ),
        (
            "wan2.7-image-pro",
            "dashscope_multimodal_generation",
            ("text", "image"),
            ("image",),
            (),
            False,
            None,
            None,
        ),
        (
            "happyhorse-1.1-i2v",
            "dashscope_video_generation",
            ("text", "image"),
            ("video",),
            (),
            False,
            None,
            None,
        ),
        (
            "happyhorse-1.1-t2v",
            "dashscope_video_generation",
            ("text",),
            ("video",),
            (),
            False,
            None,
            None,
        ),
        (
            "happyhorse-1.1-r2v",
            "dashscope_video_generation",
            ("text", "image"),
            ("video",),
            (),
            False,
            None,
            None,
        ),
        (
            "qwen-audio-3.0-tts-plus",
            "dashscope_speech_synthesis",
            ("text",),
            ("audio",),
            ("audio",),
            False,
            None,
            None,
        ),
        (
            "qwen-audio-3.0-realtime-plus",
            "dashscope_realtime",
            ("text", "audio"),
            ("text", "audio"),
            ("text", "audio"),
            True,
            40_960,
            8_192,
        ),
        (
            "qwen-audio-3.0-asr-flash",
            "dashscope_speech_recognition",
            ("audio",),
            ("text",),
            (),
            False,
            None,
            None,
        ),
    ],
)
def test_builtin_token_plan_specialized_capability_record_is_complete(
    model_id: str,
    protocol: str,
    input_modalities: tuple[str, ...],
    output_modalities: tuple[str, ...],
    streaming_output: tuple[str, ...],
    tools: bool,
    context_tokens: int | None,
    max_output_tokens: int | None,
) -> None:
    capabilities = ModelCapabilityCatalog.builtin().models[
        ("aliyun_token_plan", model_id, protocol)
    ]

    assert capabilities.to_mapping() == _complete_capability_mapping(
        input_modalities=input_modalities,
        output_modalities=output_modalities,
        streaming_output=streaming_output,
        tools=tools,
        json_object=False,
        json_schema=False,
        reasoning=False,
        context_tokens=context_tokens,
        max_output_tokens=max_output_tokens,
    )


def test_builtin_model_capabilities_use_provider_model_protocol_key() -> None:
    catalog = ModelCapabilityCatalog.builtin()

    expected = {
        ("deepseek", "deepseek-v4-flash", "openai_chat_completions"),
        ("deepseek", "deepseek-v4-pro", "openai_chat_completions"),
        ("deepseek", "deepseek-v4-flash", "anthropic_messages"),
        ("deepseek", "deepseek-v4-pro", "anthropic_messages"),
        ("anthropic", "claude-fable-5", "anthropic_messages"),
        ("anthropic", "claude-opus-5", "anthropic_messages"),
        ("anthropic", "claude-sonnet-5", "anthropic_messages"),
        ("anthropic", "claude-haiku-4-5-20251001", "anthropic_messages"),
    }
    expected.update(
        ("aliyun_token_plan", model_id, protocol)
        for model_id in ALIYUN_TEXT_MODELS
        for protocol in ("openai_chat_completions", "anthropic_messages")
    )
    specialized_protocols = {
        "qwen-image-3.0-pro": "dashscope_multimodal_generation",
        "wan2.7-image": "dashscope_multimodal_generation",
        "wan2.7-image-pro": "dashscope_multimodal_generation",
        "happyhorse-1.1-i2v": "dashscope_video_generation",
        "happyhorse-1.1-t2v": "dashscope_video_generation",
        "happyhorse-1.1-r2v": "dashscope_video_generation",
        "qwen-audio-3.0-tts-plus": "dashscope_speech_synthesis",
        "qwen-audio-3.0-realtime-plus": "dashscope_realtime",
        "qwen-audio-3.0-asr-flash": "dashscope_speech_recognition",
    }
    expected.update(
        ("aliyun_token_plan", model_id, protocol)
        for model_id, protocol in specialized_protocols.items()
    )
    assert expected <= set(catalog.models)
    capabilities = catalog.models[
        ("deepseek", "deepseek-v4-flash", "openai_chat_completions")
    ]
    assert capabilities.modalities.input == ("text",)
    assert capabilities.streaming.output == ("text",)
    assert capabilities.tools.call
    assert capabilities.structured_output.json_object
    assert not capabilities.structured_output.json_schema
    assert capabilities.reasoning.controllable
    assert capabilities.limits.context_tokens == 1_000_000
    assert capabilities.limits.max_output_tokens == 384_000


@pytest.mark.parametrize(
    "model_id, context_tokens, max_output_tokens",
    [
        ("claude-fable-5", 1_000_000, 128_000),
        ("claude-opus-5", 1_000_000, 128_000),
        ("claude-sonnet-5", 1_000_000, 128_000),
        ("claude-haiku-4-5-20251001", 200_000, 64_000),
    ],
)
def test_builtin_anthropic_capabilities_are_complete(
    model_id: str, context_tokens: int, max_output_tokens: int
) -> None:
    capabilities = ModelCapabilityCatalog.builtin().models[
        ("anthropic", model_id, "anthropic_messages")
    ]

    assert capabilities.modalities.input == ("text", "image")
    assert capabilities.modalities.output == ("text",)
    assert capabilities.streaming.output == ("text",)
    assert capabilities.tools.call
    assert capabilities.tools.choice == ("none", "auto", "required", "named")
    assert capabilities.tools.parallel
    assert capabilities.structured_output.json_object
    assert capabilities.structured_output.json_schema
    assert capabilities.reasoning.supported
    assert capabilities.reasoning.controllable
    assert capabilities.limits.context_tokens == context_tokens
    assert capabilities.limits.max_output_tokens == max_output_tokens


@pytest.mark.parametrize("model_id", ["deepseek-v4-flash", "deepseek-v4-pro"])
def test_builtin_deepseek_capabilities_are_identical_across_protocols(
    model_id: str,
) -> None:
    catalog = ModelCapabilityCatalog.builtin()
    assert catalog.models[
        ("deepseek", model_id, "anthropic_messages")
    ] == catalog.models[("deepseek", model_id, "openai_chat_completions")]


@pytest.mark.parametrize(
    "name, structured, tools, reasoning",
    [
        ("text", False, False, False),
        ("text_structured", True, False, False),
        ("text_tools", False, True, False),
        ("text_tools_structured_reasoning", True, True, True),
    ],
)
def test_capability_presets_expand_to_complete_capabilities(
    name: str, structured: bool, tools: bool, reasoning: bool
) -> None:
    preset = CapabilityPresetCatalog.builtin().presets[name]
    capabilities = preset.materialize(
        context_tokens=32_768, max_output_tokens=4_096
    )

    assert capabilities.modalities.input == ("text",)
    assert capabilities.modalities.output == ("text",)
    assert capabilities.streaming.output == ("text",)
    assert capabilities.structured_output.json_object is structured
    assert capabilities.structured_output.json_schema is structured
    assert capabilities.tools.call is tools
    assert capabilities.reasoning.supported is reasoning
    assert capabilities.reasoning.controllable is reasoning
    assert capabilities.limits.context_tokens == 32_768
    assert capabilities.limits.max_output_tokens == 4_096


def test_catalog_from_mapping_is_strict_and_immutable() -> None:
    catalog = ProviderCatalog.from_mapping(
        {
            "schema_version": 1,
            "providers": {
                "custom": {
                    "display_name": "Custom",
                    "default_protocol": "openai_chat_completions",
                    "protocols": {
                        "openai_chat_completions": {
                            "protocol": "openai_chat_completions",
                            "base_url": "https://models.example.com/v1",
                            "authentication": "none",
                            "api_key_env": None,
                            "provider_options_schema": {
                                "type": "object",
                                "properties": {},
                                "additionalProperties": True,
                            },
                        }
                    },
                }
            },
        }
    )
    assert (
        catalog.providers["custom"]
        .protocols["openai_chat_completions"]
        .authentication
        == "none"
    )
    with pytest.raises(TypeError):
        catalog.providers["other"] = catalog.providers["custom"]  # type: ignore[index]
    with pytest.raises(TypeError):
        catalog.providers["custom"].protocols["other"] = catalog.providers[  # type: ignore[index]
            "custom"
        ].protocols["openai_chat_completions"]
    with pytest.raises(ValueError, match="unknown provider catalog fields"):
        ProviderCatalog.from_mapping(
            {"schema_version": 1, "providers": {}, "unknown": True}
        )


@pytest.mark.parametrize(
    "mutate, match",
    [
        (
            lambda provider: provider.update(
                {"protocols": ["openai_chat_completions"]}
            ),
            "protocols must be an object",
        ),
        (
            lambda provider: provider["protocols"][  # type: ignore[index]
                "openai_chat_completions"
            ].update({"unknown": True}),
            "unknown provider protocol preset fields",
        ),
        (
            lambda provider: provider["protocols"][  # type: ignore[index]
                "openai_chat_completions"
            ].update({"protocol": "anthropic_messages"}),
            "protocol must match",
        ),
        (
            lambda provider: provider.update({"default_protocol": "missing"}),
            "default_protocol",
        ),
    ],
)
def test_provider_catalog_rejects_invalid_protocol_projection(mutate, match) -> None:
    provider = {
        "display_name": "Custom",
        "default_protocol": "openai_chat_completions",
        "protocols": {
            "openai_chat_completions": {
                "protocol": "openai_chat_completions",
                "base_url": "https://models.example.com/v1",
                "authentication": "none",
                "api_key_env": None,
                "provider_options_schema": {},
            }
        },
    }
    mutate(provider)
    with pytest.raises((TypeError, ValueError), match=match):
        ProviderCatalog.from_mapping(
            {"schema_version": 1, "providers": {"custom": provider}}
        )
