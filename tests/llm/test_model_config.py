from __future__ import annotations

from collections.abc import Mapping

import pytest

from pygent.llm import (
    BuiltinModelProtocol,
    ConnectionConfig,
    CredentialRef,
    EnabledModelConfig,
    ModelCapabilities,
    ModelConfig,
    ModelEntry,
    ModelGroup,
    ModelGroupConfig,
    ModelGroupResolution,
    ModelLimits,
    ModelModalities,
    ModelReasoningCapabilities,
    ModelSpec,
    ModelStreamingCapabilities,
    ModelStructuredOutputCapabilities,
    ModelToolCapabilities,
    ResolvedModelConnection,
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
        "connections": {
            "deepseek_official": {
                "provider": "deepseek",
                "credential": {"env": "DEEPSEEK_API_KEY"},
                "verify_ssl": True,
                "protocols": {
                    "openai_chat_completions": {
                        "base_url": "https://api.deepseek.com"
                    },
                    "anthropic_messages": {
                        "base_url": "https://api.deepseek.com/anthropic"
                    },
                },
            }
        },
        "models": {
            "deepseek_primary": {
                "connection": "deepseek_official",
                "model_id": "deepseek-v4-flash",
                "protocol": "openai_chat_completions",
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
    assert entry.key == "deepseek_primary"
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
    configured = config.connections["deepseek_official"]
    assert configured == ConnectionConfig(
        provider="deepseek",
        credential=CredentialRef.environment("DEEPSEEK_API_KEY"),
        protocols={
            "openai_chat_completions": "https://api.deepseek.com",
            "anthropic_messages": "https://api.deepseek.com/anthropic",
        },
    )
    connection = config.connection_for("deepseek_primary")
    assert connection == ResolvedModelConnection(
        connection_key="deepseek_official",
        provider="deepseek",
        protocol="openai_chat_completions",
        base_url="https://api.deepseek.com",
        credential=CredentialRef.environment("DEEPSEEK_API_KEY"),
        verify_ssl=True,
    )
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


def test_models_share_connection_and_resolve_protocol_independently() -> None:
    value = _mapping()
    models = value["models"]
    assert isinstance(models, dict)
    models["deepseek_anthropic"] = {
        "connection": "deepseek_official",
        "model_id": "deepseek-v4-pro",
        "protocol": "anthropic_messages",
        "provider_options": {},
        "capabilities": _capabilities(),
    }

    config = ModelConfig.from_mapping(value)

    primary = config.connection_for("deepseek_primary")
    anthropic = config.connection_for("deepseek_anthropic")
    assert (
        primary.connection_key
        == anthropic.connection_key
        == "deepseek_official"
    )
    assert primary.provider == anthropic.provider == "deepseek"
    assert primary.protocol == "openai_chat_completions"
    assert anthropic.protocol == "anthropic_messages"
    assert anthropic.base_url == "https://api.deepseek.com/anthropic"


def test_models_on_same_connection_and_protocol_have_equal_resolution() -> None:
    value = _mapping()
    models = value["models"]
    assert isinstance(models, dict)
    models["deepseek_reasoner"] = {
        **models["deepseek_primary"],  # type: ignore[dict-item]
        "model_id": "deepseek-v4-pro",
    }

    config = ModelConfig.from_mapping(value)

    assert config.connection_for("deepseek_primary") == config.connection_for(
        "deepseek_reasoner"
    )
    with pytest.raises(KeyError):
        config.connection_for("missing")


def test_each_configuration_layer_can_be_parsed_before_final_assembly() -> None:
    raw = _mapping()
    raw_connections = raw["connections"]
    raw_models = raw["models"]
    raw_groups = raw["model_groups"]
    assert isinstance(raw_connections, Mapping)
    assert isinstance(raw_models, Mapping)
    assert isinstance(raw_groups, Mapping)

    connection = ConnectionConfig.from_mapping(
        raw_connections["deepseek_official"]  # type: ignore[arg-type]
    )
    model = EnabledModelConfig.from_mapping(
        raw_models["deepseek_primary"]  # type: ignore[arg-type]
    )
    group = ModelGroupConfig.from_mapping(
        raw_groups["assistant"]  # type: ignore[arg-type]
    )

    assert connection.provider == "deepseek"
    assert model.connection_key == "deepseek_official"
    assert model.model_id == "deepseek-v4-flash"
    assert model.protocol == "openai_chat_completions"
    assert model.capabilities == ModelCapabilities.from_mapping(_capabilities())
    assert group.model_keys == ("deepseek_primary",)


def test_configuration_layers_round_trip_and_compose_as_typed_values() -> None:
    raw = _mapping()
    raw_connections = raw["connections"]
    raw_models = raw["models"]
    raw_groups = raw["model_groups"]
    assert isinstance(raw_connections, Mapping)
    assert isinstance(raw_models, Mapping)
    assert isinstance(raw_groups, Mapping)

    connection = ConnectionConfig.from_mapping(
        raw_connections["deepseek_official"]  # type: ignore[arg-type]
    )
    model = EnabledModelConfig.from_mapping(
        raw_models["deepseek_primary"]  # type: ignore[arg-type]
    )
    group = ModelGroupConfig.from_mapping(
        raw_groups["assistant"]  # type: ignore[arg-type]
    )

    assert ConnectionConfig.from_mapping(connection.to_mapping()) == connection
    assert EnabledModelConfig.from_mapping(model.to_mapping()) == model
    assert ModelGroupConfig.from_mapping(group.to_mapping()) == group

    composed = ModelConfig.from_components(
        connections={"deepseek_official": connection},
        models={"deepseek_primary": model},
        model_groups={"assistant": group},
    )
    parsed = ModelConfig.from_mapping(raw)

    assert composed == parsed
    assert composed.models["deepseek_primary"].key == "deepseek_primary"
    assert composed.model_groups["assistant"].models == (
        composed.models["deepseek_primary"],
    )


def test_typed_composition_copies_inputs_and_checks_cross_layer_references() -> None:
    raw = _mapping()
    raw_connections = raw["connections"]
    raw_models = raw["models"]
    assert isinstance(raw_connections, Mapping)
    assert isinstance(raw_models, Mapping)
    connection = ConnectionConfig.from_mapping(
        raw_connections["deepseek_official"]  # type: ignore[arg-type]
    )
    model = EnabledModelConfig.from_mapping(
        raw_models["deepseek_primary"]  # type: ignore[arg-type]
    )
    connections = {"deepseek_official": connection}
    models = {"deepseek_primary": model}

    config = ModelConfig.from_components(connections=connections, models=models)
    connections.clear()
    models.clear()

    assert tuple(config.connections) == ("deepseek_official",)
    assert tuple(config.models) == ("deepseek_primary",)
    assert config.model_groups == {}

    with pytest.raises(ValueError, match="unknown connection"):
        ModelConfig.from_components(
            connections={"deepseek_official": connection},
            models={
                "deepseek_primary": EnabledModelConfig(
                    connection_key="missing",
                    model_id=model.model_id,
                    protocol=model.protocol,
                    provider_options=model.provider_options,
                    capabilities=model.capabilities,
                )
            },
        )


def test_standalone_layer_parsers_do_not_require_referenced_records() -> None:
    model = EnabledModelConfig.from_mapping(
        {
            "connection": "saved_later",
            "model_id": "provider-model-id",
            "protocol": "custom_protocol",
            "provider_options": {},
            "capabilities": _capabilities(),
        }
    )
    group = ModelGroupConfig.from_mapping({"models": ["saved_later"]})

    assert model.connection_key == "saved_later"
    assert group.model_keys == ("saved_later",)


def test_model_groups_are_optional_for_direct_single_model_use() -> None:
    value = _mapping()
    del value["model_groups"]

    config = ModelConfig.from_mapping(value)

    assert config.model_groups == {}
    with pytest.raises(TypeError, match="from_mapping"):
        ModelConfig()


def test_connection_projection_is_immutable_and_detached_from_input() -> None:
    value = _mapping()
    config = ModelConfig.from_mapping(value)
    configured = config.connections["deepseek_official"]

    connections = value["connections"]
    assert isinstance(connections, dict)
    raw_connection = connections["deepseek_official"]
    assert isinstance(raw_connection, dict)
    protocols = raw_connection["protocols"]
    assert isinstance(protocols, dict)
    endpoint = protocols["openai_chat_completions"]
    assert isinstance(endpoint, dict)
    endpoint["base_url"] = "https://changed.example"

    assert configured.protocols["openai_chat_completions"] == "https://api.deepseek.com"
    with pytest.raises(TypeError):
        configured.protocols["openai_chat_completions"] = "https://changed.example"  # type: ignore[index]


def test_credential_resolution_is_explicit_and_secret_free() -> None:
    credential = CredentialRef.environment("DEEPSEEK_API_KEY")
    assert credential.resolve({"DEEPSEEK_API_KEY": "secret-value"}) == "secret-value"
    assert "secret-value" not in repr(credential)
    with pytest.raises(LookupError, match="DEEPSEEK_API_KEY"):
        credential.resolve({})
    assert CredentialRef.none().resolve({}) is None
    assert credential.to_mapping() == {"env": "DEEPSEEK_API_KEY"}
    assert CredentialRef.none().to_mapping() == {"none": True}


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda value: value.update({"unknown": True}), "unknown model config fields"),
        (
            lambda value: value["connections"]["deepseek_official"].update(  # type: ignore[index,union-attr]
                {"unknown": True}
            ),
            "unknown connection fields",
        ),
        (
            lambda value: value["models"]["deepseek_primary"].update(  # type: ignore[index,union-attr]
                {"unknown": True}
            ),
            "unknown model fields",
        ),
        (
            lambda value: value["connections"]["deepseek_official"]["protocols"][  # type: ignore[index]
                "openai_chat_completions"
            ].update(  # type: ignore[union-attr]
                {"unknown": True}
            ),
            "unknown protocol endpoint fields",
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
            lambda value: value["connections"]["deepseek_official"]["protocols"][  # type: ignore[index]
                "openai_chat_completions"
            ].update({"base_url": "https://user:password@example.com"}),  # type: ignore[union-attr]
            "embedded credentials",
        ),
        (
            lambda value: value["connections"]["deepseek_official"].update(  # type: ignore[index,union-attr]
                {"credential": {"env": "DEEPSEEK_API_KEY", "none": True}}
            ),
            "exactly one",
        ),
        (
            lambda value: value["models"]["deepseek_primary"].update(  # type: ignore[index,union-attr]
                {"connection": "missing"}
            ),
            "unknown connection",
        ),
        (
            lambda value: value["models"]["deepseek_primary"].update(  # type: ignore[index,union-attr]
                {"protocol": "openai_responses"}
            ),
            "does not provide protocol",
        ),
        (
            lambda value: value["models"]["deepseek_primary"].update(  # type: ignore[index,union-attr]
                {"provider": "deepseek"}
            ),
            "unknown model fields",
        ),
        (
            lambda value: value["models"]["deepseek_primary"].update(  # type: ignore[index,union-attr]
                {"connection": {"base_url": "https://api.deepseek.com"}}
            ),
            "connection must be a non-empty string",
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
    modalities[field] = ["text", "files"]

    with pytest.raises(ValueError, match="unsupported modalities"):
        ModelCapabilities.from_mapping(
            _capabilities(
                modalities=modalities,
                streaming={"output": ["text"]},
            )
        )


def test_embedding_is_an_output_modality_only() -> None:
    capabilities = ModelCapabilities.from_mapping(
        _capabilities(
            modalities={"input": ["text"], "output": ["embedding"]},
            streaming={"output": []},
        )
    )

    assert capabilities.modalities.output == ("embedding",)
    with pytest.raises(ValueError, match="unsupported modalities.input"):
        ModelCapabilities.from_mapping(
            _capabilities(modalities={"input": ["embedding"], "output": ["text"]})
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
