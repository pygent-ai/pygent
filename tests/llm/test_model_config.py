from __future__ import annotations

from collections.abc import Mapping

import pytest

from pygent.llm import (
    BuiltinModelProtocol,
    CredentialRef,
    ModelCapabilities,
    ModelConfig,
    ModelEntry,
    ModelGroup,
    ModelGroupResolution,
    ModelLimits,
    ModelModalities,
    ModelReasoningCapabilities,
    ModelSpec,
    ModelStreamingCapabilities,
    ModelStructuredOutputCapabilities,
    ModelToolCapabilities,
)


def _capabilities(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "modalities": {"input": ["text"], "output": ["text"]},
        "streaming": {"output": ["text"]},
        "tools": {
            "call": True,
            "choice": ["none", "auto", "required", "named"],
            "parallel": True,
        },
        "structured_output": {"json_object": True, "json_schema": False},
        "reasoning": {"supported": True, "controllable": True},
        "limits": {"context_tokens": 1_000_000, "max_output_tokens": 384_000},
    }
    value.update(overrides)
    return value


def _mapping() -> dict[str, object]:
    return {
        "models": {
            "deepseek_primary": {
                "provider": "deepseek",
                "model_id": "deepseek-v4-flash",
                "protocol": "openai_chat_completions",
                "connection": {
                    "base_url": "https://api.deepseek.com",
                    "credential": {"env": "DEEPSEEK_API_KEY"},
                    "verify_ssl": True,
                },
                "provider_options": {"thinking": {"type": "disabled"}},
                "capabilities": _capabilities(),
            }
        },
        "model_groups": {"assistant": {"models": ["deepseek_primary"]}},
    }


def test_builtin_protocols_are_precise_and_model_spec_stores_plain_string() -> None:
    assert (
        BuiltinModelProtocol.OPENAI_CHAT_COMPLETIONS.value
        == "openai_chat_completions"
    )
    assert BuiltinModelProtocol.ANTHROPIC_MESSAGES.value == "anthropic_messages"
    spec = ModelSpec(
        provider="deepseek",
        model_id="deepseek-v4-pro",
        protocol=BuiltinModelProtocol.OPENAI_CHAT_COMPLETIONS,
        capabilities=ModelCapabilities.from_mapping(_capabilities()),
    )
    assert spec.protocol == "openai_chat_completions"
    assert type(spec.protocol) is str


def test_model_config_parses_named_semantics_and_connection_projection() -> None:
    raw = _mapping()
    config = ModelConfig.from_mapping(raw)

    entry = config.models["deepseek_primary"]
    assert isinstance(entry, ModelEntry)
    assert entry.name == "deepseek_primary"
    assert entry.spec == ModelSpec(
        provider="deepseek",
        model_id="deepseek-v4-flash",
        protocol="openai_chat_completions",
        provider_options={"thinking": {"type": "disabled"}},
        capabilities=ModelCapabilities.from_mapping(_capabilities()),
    )
    assert config.model_groups["assistant"] == ModelGroup(
        name="assistant", models=(entry,)
    )
    connection = config.connections["deepseek_primary"]
    assert connection.base_url == "https://api.deepseek.com"
    assert connection.credential == CredentialRef.environment("DEEPSEEK_API_KEY")
    assert connection.verify_ssl is True
    assert connection.proxy is None

    model_values = raw["models"]
    assert isinstance(model_values, Mapping)
    model = model_values["deepseek_primary"]
    assert isinstance(model, Mapping)
    options = model["provider_options"]
    assert isinstance(options, dict)
    thinking = options["thinking"]
    assert isinstance(thinking, dict)
    thinking["type"] = "enabled"
    assert entry.spec.provider_options["thinking"]["type"] == "disabled"
    with pytest.raises(TypeError):
        config.models["other"] = entry  # type: ignore[index]


def test_credential_resolution_is_explicit_and_secret_free() -> None:
    credential = CredentialRef.environment("DEEPSEEK_API_KEY")
    assert credential.resolve({"DEEPSEEK_API_KEY": "secret-value"}) == "secret-value"
    assert "secret-value" not in repr(credential)
    with pytest.raises(LookupError, match="DEEPSEEK_API_KEY"):
        credential.resolve({})
    assert CredentialRef.none().resolve({}) is None


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda value: value.update({"unknown": True}), "unknown model config fields"),
        (
            lambda value: value["models"]["deepseek_primary"].update(  # type: ignore[index,union-attr]
                {"unknown": True}
            ),
            "unknown model fields",
        ),
        (
            lambda value: value["models"]["deepseek_primary"]["connection"].update(  # type: ignore[index,union-attr]
                {"unknown": True}
            ),
            "unknown connection fields",
        ),
        (
            lambda value: value["models"]["deepseek_primary"].update(  # type: ignore[index,union-attr]
                {"capabilities": {"modalities": {"input": ["text"], "output": ["text"]}}}
            ),
            "capabilities fields",
        ),
        (
            lambda value: value["model_groups"]["assistant"].update(  # type: ignore[index,union-attr]
                {"models": ["missing"]}
            ),
            "unknown model",
        ),
        (
            lambda value: value["model_groups"]["assistant"].update(  # type: ignore[index,union-attr]
                {"models": ["deepseek_primary", "deepseek_primary"]}
            ),
            "duplicate models",
        ),
        (
            lambda value: value["models"]["deepseek_primary"]["connection"].update(  # type: ignore[index,union-attr]
                {"base_url": "https://user:password@example.com"}
            ),
            "embedded credentials",
        ),
        (
            lambda value: value["models"]["deepseek_primary"]["connection"].update(  # type: ignore[index,union-attr]
                {"credential": {"env": "DEEPSEEK_API_KEY", "none": True}}
            ),
            "exactly one",
        ),
    ],
)
def test_model_config_rejects_invalid_or_ambiguous_input(mutate, match: str) -> None:
    value = _mapping()
    mutate(value)
    with pytest.raises((TypeError, ValueError), match=match):
        ModelConfig.from_mapping(value)


def test_model_group_deferred_has_no_concrete_models_or_capacity_fields() -> None:
    group = ModelGroup.deferred(name="assistant")
    assert group.resolution is ModelGroupResolution.DEFERRED
    assert group.models == ()
    assert group.is_deferred
    assert not hasattr(group, "max_concurrency")
    assert not hasattr(group, "capacity_key")


def test_direct_capability_values_normalize_sequences_and_validate_types() -> None:
    modalities = ModelModalities(input=["text"], output=["text"])  # type: ignore[arg-type]
    tools = ModelToolCapabilities(
        call=True,
        choice=["none", "auto"],  # type: ignore[arg-type]
        parallel=False,
    )
    capabilities = ModelCapabilities(
        modalities=modalities,
        streaming=ModelStreamingCapabilities(output=("text",)),
        tools=tools,
        structured_output=ModelStructuredOutputCapabilities(
            json_object=True,
            json_schema=False,
        ),
        reasoning=ModelReasoningCapabilities(
            supported=True,
            controllable=False,
        ),
        limits=ModelLimits(context_tokens=32_768, max_output_tokens=4_096),
    )

    assert capabilities.modalities.input == ("text",)
    assert capabilities.tools.choice == ("none", "auto")
    with pytest.raises(TypeError, match="streaming.output"):
        ModelStreamingCapabilities(output=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reasoning.controllable"):
        ModelReasoningCapabilities(supported=False, controllable=True)
    with pytest.raises(ValueError, match="positive integer"):
        ModelLimits(context_tokens=0, max_output_tokens=4_096)


def test_streaming_output_and_nullable_limits_round_trip() -> None:
    capabilities = ModelCapabilities.from_mapping(
        _capabilities(
            modalities={
                "input": ["text", "image"],
                "output": ["text", "audio"],
            },
            streaming={"output": ["text", "audio"]},
            limits={"context_tokens": None, "max_output_tokens": None},
        )
    )

    assert capabilities.streaming.output == ("text", "audio")
    assert capabilities.limits == ModelLimits(
        context_tokens=None,
        max_output_tokens=None,
    )
    assert capabilities.to_mapping()["streaming"] == {
        "output": ["text", "audio"]
    }
    assert capabilities.to_mapping()["limits"] == {
        "context_tokens": None,
        "max_output_tokens": None,
    }


@pytest.mark.parametrize("field", ["input", "output"])
def test_modalities_reject_unknown_values(field: str) -> None:
    modalities = {"input": ["text"], "output": ["text"]}
    modalities[field] = ["text", "embedding"]

    with pytest.raises(ValueError, match="unsupported modalities"):
        ModelCapabilities.from_mapping(
            _capabilities(
                modalities=modalities,
                streaming={"output": ["text"]},
            )
        )


def test_streaming_output_must_be_output_modality_subset() -> None:
    with pytest.raises(ValueError, match="streaming.output"):
        ModelCapabilities.from_mapping(
            _capabilities(
                modalities={"input": ["text"], "output": ["text"]},
                streaming={"output": ["audio"]},
            )
        )


def test_old_streaming_text_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="streaming"):
        ModelCapabilities.from_mapping(_capabilities(streaming={"text": True}))
