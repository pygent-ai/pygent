from __future__ import annotations

from collections.abc import Mapping

import pytest

from pygent.llm import (
    CredentialRef,
    ModelCapabilities,
    ModelConfig,
    ModelEntry,
    ModelGroup,
    ModelGroupResolution,
    ModelSpec,
)


def _capabilities(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "modalities": {"input": ["text"], "output": ["text"]},
        "streaming": {"text": True},
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
                "protocol": "openai_compatible",
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


def test_model_config_parses_named_semantics_and_connection_projection() -> None:
    raw = _mapping()
    config = ModelConfig.from_mapping(raw)

    entry = config.models["deepseek_primary"]
    assert isinstance(entry, ModelEntry)
    assert entry.name == "deepseek_primary"
    assert entry.spec == ModelSpec(
        provider="deepseek",
        model_id="deepseek-v4-flash",
        protocol="openai_compatible",
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

