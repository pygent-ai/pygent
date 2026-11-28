"""Fork-specific reasoning-effort profiles (fork-reply latency)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import pytest

from pygent import Context, GenerationConfig, UserMessage
from pygent.agent.pygent_v1.runtime.realtime.fork_reasoning import (
    _wire_reasoning_mode,
    fork_model_profiles,
    fork_profile_name,
)
from pygent.agent.pygent_v1.schema.models import RealtimeInteractionConfig
from pygent.core import freeze_json_object
from pygent.llm import (
    AnthropicMessagesAdapter,
    ModelConfig,
    ModelEntry,
    ModelProviderRequest,
)


def _capabilities(
    *,
    supported: bool,
    controllable: bool,
    disable_supported: bool = False,
) -> dict[str, object]:
    return {
        "modalities": {"input": ["text"], "output": ["text"]},
        "streaming": {"output": []},
        "tools": {"call": True, "choice": ["auto"], "parallel": False},
        "structured_output": {"json_object": False, "json_schema": False},
        "reasoning": {
            "supported": supported,
            "controllable": controllable,
            **(
                {}
                if not disable_supported
                else {"disable_supported": True}
            ),
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
    disable_supported: bool = False,
    provider_options: dict[str, Any] | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "connection": connection,
        "model_id": model_id,
        "protocol": protocol,
        "capabilities": _capabilities(
            supported=supported,
            controllable=controllable,
            disable_supported=disable_supported,
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
                    provider_options={
                        "thinking": {"type": "adaptive", "display": "summarized"},
                        "output_config": {"effort": "high"},
                    },
                ),
                "anthropic_off": _model(
                    "anthropic",
                    "claude-off",
                    "anthropic_messages",
                    supported=True,
                    controllable=True,
                    disable_supported=True,
                    provider_options={
                        "thinking": {"type": "adaptive", "display": "summarized"},
                        "output_config": {"effort": "high"},
                    },
                ),
            },
            "model_groups": {
                "mixed": {
                    "models": ["controllable", "fixed", "plain", "anthropic"]
                },
                "cutoff": {"models": ["anthropic_off"]},
            },
        }
    )


def test_fork_profile_injects_effort_for_openai_and_anthropic_models() -> None:
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
        "anthropic": "reasoning_effort_applied",
    }
    assert profile.models[1] is config.models["fixed"]
    assert profile.models[2] is config.models["plain"]

    anthropic_original = config.models["anthropic"]
    anthropic_forked = profile.models[3]
    assert anthropic_forked is not anthropic_original
    anthropic_options = dict(
        cast(Mapping[str, object], anthropic_forked.spec.provider_options)
    )
    anthropic_original_options = dict(
        cast(Mapping[str, object], anthropic_original.spec.provider_options)
    )
    assert dict(
        cast(Mapping[str, object], anthropic_options["output_config"])
    ) == {"effort": "low"}
    assert dict(
        cast(Mapping[str, object], anthropic_options["thinking"])
    ) == {"type": "adaptive", "display": "summarized"}
    assert dict(
        cast(Mapping[str, object], anthropic_original_options["output_config"])
    ) == {"effort": "high"}


def test_forked_anthropic_entry_requests_the_fork_effort() -> None:
    config = _config()
    profile = fork_model_profiles(config, "mixed", "low")["anthropic"]
    forked = profile.models[0]

    def body(entry: ModelEntry) -> dict[str, object]:
        return AnthropicMessagesAdapter().build_request(
            ModelProviderRequest(
                model_key=entry.key,
                model=entry.spec,
                message=UserMessage(content="question"),
                context=Context(),
                generation=GenerationConfig(max_output_tokens=4096),
            )
        ).to_dict()

    assert body(forked)["output_config"] == {"effort": "low"}
    assert body(forked)["thinking"] == {
        "type": "adaptive",
        "display": "summarized",
    }
    assert body(config.models["anthropic"])["output_config"] == {"effort": "high"}


def test_fork_none_effort_disables_thinking_for_declared_anthropic_models() -> None:
    config = _config()
    profile = fork_model_profiles(config, "cutoff", "none")["anthropic_off"]
    forked = profile.models[0]

    options = dict(cast(Mapping[str, object], forked.spec.provider_options))
    assert dict(cast(Mapping[str, object], options["thinking"])) == {
        "type": "disabled"
    }
    assert "output_config" not in options
    reasons = {decision.model_key: decision.reason for decision in profile.decisions}
    assert reasons == {"anthropic_off": "reasoning_effort_none_applied"}


def test_forked_anthropic_none_entry_requests_thinking_disabled() -> None:
    config = _config()
    forked = fork_model_profiles(config, "cutoff", "none")["anthropic_off"].models[0]

    def body(entry: ModelEntry) -> dict[str, object]:
        return AnthropicMessagesAdapter().build_request(
            ModelProviderRequest(
                model_key=entry.key,
                model=entry.spec,
                message=UserMessage(content="question"),
                context=Context(),
                generation=GenerationConfig(max_output_tokens=4096),
            )
        ).to_dict()

    assert body(forked)["thinking"] == {"type": "disabled"}
    assert "output_config" not in body(forked)


def test_fork_none_effort_degrades_to_low_without_disable_support() -> None:
    config = _config()
    profile = fork_model_profiles(config, "mixed", "none")["controllable"]

    reasons = {decision.model_key: decision.reason for decision in profile.decisions}
    assert reasons == {
        "controllable": "reasoning_effort_none_degraded",
        "fixed": "reasoning_not_controllable",
        "plain": "reasoning_not_supported",
        "anthropic": "reasoning_effort_none_degraded",
    }

    by_key = {model.key: model for model in profile.models}
    anthropic_options = dict(
        cast(Mapping[str, object], by_key["anthropic"].spec.provider_options)
    )
    assert dict(cast(Mapping[str, object], anthropic_options["output_config"])) == {
        "effort": "low"
    }
    assert dict(cast(Mapping[str, object], anthropic_options["thinking"])) == {
        "type": "adaptive",
        "display": "summarized",
    }
    openai_options = dict(
        cast(Mapping[str, object], by_key["controllable"].spec.provider_options)
    )
    assert openai_options["reasoning_effort"] == "low"


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


@pytest.mark.parametrize(
    ("protocol", "body", "mode"),
    [
        ("openai_chat_completions", {"enable_thinking": False}, "off"),
        ("anthropic_messages", {"thinking": {"type": "disabled"}}, "off"),
        ("anthropic_messages", {"output_config": {"effort": "low"}}, "low"),
        ("openai_responses", {"reasoning": {"effort": "low"}}, "low"),
        ("gemini_generate_content", {"generationConfig": {"thinkingConfig": {"thinkingBudget": 0}}}, "off"),
        ("openai_chat_completions", {}, "other"),
    ],
)
def test_fork_budget_classifies_prepared_wire_reasoning(protocol, body, mode) -> None:
    assert _wire_reasoning_mode(protocol, freeze_json_object(body)) == mode
