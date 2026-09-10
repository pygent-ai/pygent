from __future__ import annotations

import pytest

from pygent.llm import (
    CapabilityPresetCatalog,
    ModelCapabilityCatalog,
    ProviderCatalog,
)


def test_builtin_provider_catalog_contains_only_deepseek_official() -> None:
    catalog = ProviderCatalog.builtin()

    assert tuple(catalog.providers) == ("deepseek",)
    preset = catalog.providers["deepseek"]
    assert preset.display_name == "DeepSeek"
    assert preset.protocols == ("openai_compatible",)
    assert preset.default_protocol == "openai_compatible"
    assert preset.base_url == "https://api.deepseek.com"
    assert preset.authentication == "bearer"
    assert preset.api_key_env == "DEEPSEEK_API_KEY"
    assert "thinking" in preset.provider_options_schema["properties"]
    assert "anthropic_compatible" not in preset.protocols


def test_builtin_model_capabilities_use_provider_model_protocol_key() -> None:
    catalog = ModelCapabilityCatalog.builtin()

    assert set(catalog.models) == {
        ("deepseek", "deepseek-v4-flash", "openai_compatible"),
        ("deepseek", "deepseek-v4-pro", "openai_compatible"),
    }
    capabilities = catalog.models[
        ("deepseek", "deepseek-v4-flash", "openai_compatible")
    ]
    assert capabilities.modalities.input == ("text",)
    assert capabilities.streaming.text
    assert capabilities.tools.call
    assert capabilities.structured_output.json_object
    assert not capabilities.structured_output.json_schema
    assert capabilities.reasoning.controllable
    assert capabilities.limits.context_tokens == 1_000_000
    assert capabilities.limits.max_output_tokens == 384_000


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
    assert capabilities.streaming.text
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
                    "protocols": ["openai_compatible"],
                    "default_protocol": "openai_compatible",
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
    )
    assert catalog.providers["custom"].authentication == "none"
    with pytest.raises(TypeError):
        catalog.providers["other"] = catalog.providers["custom"]  # type: ignore[index]
    with pytest.raises(ValueError, match="unknown provider catalog fields"):
        ProviderCatalog.from_mapping(
            {"schema_version": 1, "providers": {}, "unknown": True}
        )

