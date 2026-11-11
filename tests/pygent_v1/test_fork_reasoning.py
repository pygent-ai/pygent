"""Fork-specific reasoning-effort profiles (fork-reply latency)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import pytest

from pygent.agent.pygent_v1.runtime.realtime.fork_reasoning import (
    fork_model_profiles,
    fork_profile_name,
)
from pygent.agent.pygent_v1.schema.models import RealtimeInteractionConfig
from pygent.llm import ModelConfig


def _capabilities(
    *, supported: bool, controllable: bool
) -> dict[str, object]:
    return {
        "modalities": {"input": ["text"], "output": ["text"]},
        "streaming": {"output": []},
        "tools": {"call": True, "choice": ["auto"], "parallel": False},
        "structured_output": {"json_object": False, "json_schema": False},
        "reasoning": {
            "supported": supported,
            "controllable": controllable,
        },
        "limits": {"context_tokens": 100_000, "max_output_tokens": 4_096},
    }


def _model(
    connection: str,
    model_id: str,
    protocol: str,
    *,
    supported: bool,
    controllable: bool,
    provider_options: dict[str, Any] | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "connection": connection,
        "model_id": model_id,
        "protocol": protocol,
        "capabilities": _capabilities(
            supported=supported, controllable=controllable
        ),
    }
    if provider_options is not None:
        value["provider_options"] = provider_options
    return value


def _config() -> ModelConfig:
    return ModelConfig.from_mapping(
        {
            "connections": {
                "openai": {
                    "provider": "openai",
                    "credential": {"none": True},
                    "protocols": {
                        "openai_chat_completions": {
                            "base_url": "http://127.0.0.1:1"
                        }
                    },
                },
                "anthropic": {
                    "provider": "anthropic",
                    "credential": {"none": True},
                    "protocols": {
                        "anthropic_messages": {"base_url": "http://127.0.0.1:1"}
                    },
                },
            },
            "models": {
                "controllable": _model(
                    "openai",
                    "gpt-reasoning",
                    "openai_chat_completions",
                    supported=True,
                    controllable=True,
                    provider_options={"max_completion_tokens": 128},
                ),
                "fixed": _model(
                    "openai",
                    "gpt-fixed",
                    "openai_chat_completions",
                    supported=True,
                    controllable=False,
                ),
                "plain": _model(
                    "openai",
                    "gpt-plain",
                    "openai_chat_completions",
                    supported=False,
                    controllable=False,
                ),
                "anthropic": _model(
                    "anthropic",
                    "claude-reasoning",
                    "anthropic_messages",
                    supported=True,
                    controllable=True,
                ),
            },
            "model_groups": {
                "mixed": {
                    "models": ["controllable", "fixed", "plain", "anthropic"]
                }
            },
        }
    )


def test_fork_profile_injects_effort_only_for_controllable_openai_models() -> None:
    config = _config()
    original = config.models["controllable"]

    profiles = fork_model_profiles(config, "mixed", "low")

    assert set(profiles) == {"controllable", "fixed", "plain", "anthropic"}
    profile = profiles["controllable"]
    assert profile.profile == fork_profile_name("controllable")
    assert [model.key for model in profile.models] == [
        "controllable",
        "fixed",
        "plain",
        "anthropic",
    ]
    forked = profile.models[0]
    forked_options = dict(
        cast(Mapping[str, object], forked.spec.provider_options)
    )
    original_options = dict(
        cast(Mapping[str, object], original.spec.provider_options)
    )
    assert forked_options["reasoning_effort"] == "low"
    assert forked_options["max_completion_tokens"] == 128
    assert "reasoning_effort" not in original_options

    reasons = {decision.model_key: decision.reason for decision in profile.decisions}
    assert reasons == {
        "controllable": "reasoning_effort_applied",
        "fixed": "reasoning_not_controllable",
        "plain": "reasoning_not_supported",
        "anthropic": "protocol_not_supported",
    }
    assert profile.models[1] is config.models["fixed"]
    assert profile.models[2] is config.models["plain"]
    assert profile.models[3] is config.models["anthropic"]


def test_none_effort_keeps_the_main_profile() -> None:
    assert fork_model_profiles(_config(), "mixed", None) == {}


def test_unknown_group_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown model group"):
        fork_model_profiles(_config(), "missing", "low")


def test_realtime_config_defaults_fork_reasoning_effort_to_low() -> None:
    assert RealtimeInteractionConfig().fork_reasoning_effort == "low"
    assert (
        RealtimeInteractionConfig(fork_reasoning_effort=None).fork_reasoning_effort
        is None
    )


def test_realtime_config_rejects_blank_fork_reasoning_effort() -> None:
    with pytest.raises(ValueError, match="fork_reasoning_effort"):
        RealtimeInteractionConfig(fork_reasoning_effort="  ")
