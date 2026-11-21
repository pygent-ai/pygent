"""Fork-specific model profiles for the interrupt reply (D-RT2/3).

The fork reply is a single, latency-sensitive model call. When the configured
``fork_reasoning_effort`` is set, each reasoning-capable model in the session's
group gets a fork-only ``ModelEntry`` copy whose ``provider_options`` carry
that effort. ``PygentRuntimeService.bind`` publishes these entries under a
distinct profile (``fork:<model-key>``) so the main turn's profiles stay
untouched.

The effort is expressed in each protocol's native dial: the top-level
``reasoning_effort`` field for OpenAI Chat Completions and
``output_config.effort`` for Anthropic Messages. The Responses protocol
expects a nested ``reasoning: {effort: ...}`` object and is left unchanged
for now, with the decision recorded on
:class:`ForkModelProfile.decisions` for auditing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal

from pygent.agent.pygent_v1.runtime.model_configuration import preferred_models
from pygent.llm import ModelConfig, ModelEntry

# Adapters that forward ``provider_options`` as top-level request fields.
_REASONING_EFFORT_PROTOCOLS = frozenset({"openai_chat_completions"})
_ANTHROPIC_MESSAGES = "anthropic_messages"

ForkReasoningDecisionReason = Literal[
    "reasoning_effort_applied",
    "reasoning_not_supported",
    "reasoning_not_controllable",
    "protocol_not_supported",
]


@dataclass(frozen=True, slots=True)
class ForkReasoningDecision:
    """Why one fork model entry did (or did not) receive the effort override."""

    model_key: str
    protocol: str
    applied: bool
    reason: ForkReasoningDecisionReason


@dataclass(frozen=True, slots=True)
class ForkModelProfile:
    """One fork-only profile: preferred model first, then group fallbacks."""

    profile: str
    models: tuple[ModelEntry, ...]
    decisions: tuple[ForkReasoningDecision, ...]


def fork_profile_name(model_key: str) -> str:
    if not isinstance(model_key, str) or not model_key:
        raise ValueError("model_key must be a non-empty string")
    return f"fork:{model_key}"


def fork_model_profiles(
    config: ModelConfig,
    group_name: str,
    reasoning_effort: str | None,
) -> dict[str, ForkModelProfile]:
    """Build one fork profile per model key in ``group_name``.

    ``None`` means "follow the model's own default" and creates no fork
    profiles, so the fork shares the main turn's preferred profile.
    """

    if reasoning_effort is None:
        return {}
    try:
        group = config.model_groups[group_name]
    except KeyError:
        raise ValueError(f"unknown model group {group_name!r}") from None
    profiles: dict[str, ForkModelProfile] = {}
    for entry in group.models:
        ordered = preferred_models(config, group_name, entry.key)
        models: list[ModelEntry] = []
        decisions: list[ForkReasoningDecision] = []
        for model in ordered:
            forked, reason = _apply_reasoning_effort(
                model, reasoning_effort
            )
            models.append(forked)
            decisions.append(
                ForkReasoningDecision(
                    model_key=model.key,
                    protocol=model.spec.protocol,
                    applied=forked is not model,
                    reason=reason,
                )
            )
        profiles[entry.key] = ForkModelProfile(
            profile=fork_profile_name(entry.key),
            models=tuple(models),
            decisions=tuple(decisions),
        )
    return profiles


def _apply_reasoning_effort(
    entry: ModelEntry, reasoning_effort: str
) -> tuple[ModelEntry, ForkReasoningDecisionReason]:
    reasoning = entry.spec.capabilities.reasoning
    if not reasoning.supported:
        return entry, "reasoning_not_supported"
    if not reasoning.controllable:
        return entry, "reasoning_not_controllable"
    protocol = entry.spec.protocol
    if protocol in _REASONING_EFFORT_PROTOCOLS:
        options = dict(entry.spec.provider_options)
        options["reasoning_effort"] = reasoning_effort
    elif protocol == _ANTHROPIC_MESSAGES:
        options = dict(entry.spec.provider_options)
        raw_output = options.get("output_config")
        output_config = dict(raw_output) if isinstance(raw_output, Mapping) else {}
        output_config["effort"] = reasoning_effort
        options["output_config"] = output_config
    else:
        return entry, "protocol_not_supported"
    return (
        replace(
            entry,
            spec=replace(entry.spec, provider_options=options),
        ),
        "reasoning_effort_applied",
    )


__all__ = [
    "ForkModelProfile",
    "ForkReasoningDecision",
    "ForkReasoningDecisionReason",
    "fork_model_profiles",
    "fork_profile_name",
]
