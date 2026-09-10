from __future__ import annotations

import pytest

from pygent.llm import (
    CapabilityPresetCatalog,
    ModelCapabilityCatalog,
    ProviderCatalog,
)


def test_builtin_provider_catalog_projects_protocol_specific_connections() -> None:
    catalog = ProviderCatalog.builtin()

    assert tuple(catalog.providers) == ("deepseek", "anthropic")
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


def test_builtin_model_capabilities_use_provider_model_protocol_key() -> None:
    catalog = ModelCapabilityCatalog.builtin()

    assert set(catalog.models) == {
        ("deepseek", "deepseek-v4-flash", "openai_chat_completions"),
        ("deepseek", "deepseek-v4-pro", "openai_chat_completions"),
        ("deepseek", "deepseek-v4-flash", "anthropic_messages"),
        ("deepseek", "deepseek-v4-pro", "anthropic_messages"),
        ("anthropic", "claude-fable-5-1", "anthropic_messages"),
        ("anthropic", "claude-opus-5", "anthropic_messages"),
        ("anthropic", "claude-sonnet-5", "anthropic_messages"),
        ("anthropic", "claude-haiku-4-5-20251001", "anthropic_messages"),
    }
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
        ("claude-fable-5-1", 1_000_000, 128_000),
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
