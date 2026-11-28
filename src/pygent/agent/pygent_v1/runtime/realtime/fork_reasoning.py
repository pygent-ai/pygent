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

``"none"`` asks for the model's lowest thinking posture. The Anthropic
protocol expresses it as ``thinking: {"type": "disabled"}`` — a separate
field from the effort dial — and only models whose capabilities declare
``reasoning.disable_supported`` receive it (official Anthropic rejects
``disabled`` on its adaptive-only models, Kimi's Anthropic endpoint has no
switch at all). Everything else degrades to the low effort dial instead of
sending a field the endpoint rejects; the degrade is recorded as
``reasoning_effort_none_degraded``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Literal

from pygent.agent.pygent_v1.runtime.model_configuration import preferred_models
from pygent.agent.pygent_v1.runtime.realtime.speak import SPEAK_TOOL_NAME
from pygent.core import FrozenJsonObject
from pygent.llm import (
    ModelConfig,
    ModelEntry,
    ModelProviderStreamKind,
    ModelProviderStreamPart,
)

# Adapters that forward ``provider_options`` as top-level request fields.
_REASONING_EFFORT_PROTOCOLS = frozenset({"openai_chat_completions"})
_ANTHROPIC_MESSAGES = "anthropic_messages"

#: The fork's lowest controllable effort, used when "none" cannot be
#: expressed on the model's endpoint.
_NONE_FALLBACK_EFFORT = "low"

ForkReasoningDecisionReason = Literal[
    "reasoning_effort_applied",
    "reasoning_effort_none_applied",
    "reasoning_effort_none_degraded",
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
    if reasoning_effort == "none":
        # The off dial only exists on Anthropic Messages as thinking.type,
        # and only for endpoints that accept "disabled" — the forked options
        # drop the effort dial entirely so the fork fully owns the posture.
        if protocol == _ANTHROPIC_MESSAGES and reasoning.disable_supported:
            options = dict(entry.spec.provider_options)
            options["thinking"] = {"type": "disabled"}
            options.pop("output_config", None)
            return (
                replace(
                    entry,
                    spec=replace(entry.spec, provider_options=options),
                ),
                "reasoning_effort_none_applied",
            )
        forked, reason = _apply_reasoning_effort(entry, _NONE_FALLBACK_EFFORT)
        return (
            forked,
            (
                reason
                if reason != "reasoning_effort_applied"
                else "reasoning_effort_none_degraded"
            ),
        )
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


def _wire_reasoning_mode(
    protocol: str, payload: FrozenJsonObject
) -> Literal["off", "low", "other"]:
    thinking = payload.get("thinking")
    effort = None
    if protocol in (_ANTHROPIC_MESSAGES, "openai_chat_completions"):
        if isinstance(thinking, Mapping) and thinking.get("type") == "disabled":
            return "off"
        if protocol == _ANTHROPIC_MESSAGES:
            output = payload.get("output_config")
            effort = output.get("effort") if isinstance(output, Mapping) else None
        else:
            if payload.get("enable_thinking") is False:
                return "off"
            effort = payload.get("reasoning_effort")
    elif protocol == "openai_responses":
        reasoning = payload.get("reasoning")
        effort = reasoning.get("effort") if isinstance(reasoning, Mapping) else None
    elif protocol == "gemini_generate_content":
        generation = payload.get("generationConfig")
        if isinstance(generation, Mapping):
            thinking_config = generation.get("thinkingConfig")
            if isinstance(thinking_config, Mapping):
                budget = thinking_config.get("thinkingBudget")
                if type(budget) is int and budget == 0:
                    return "off"
    if effort == "none":
        return "off"
    return "low" if effort == "low" else "other"


class ForkReasoningBudget:
    """Adjust the fork's owner-task deadline using the current wire attempt."""

    def __init__(self, timeout: asyncio.Timeout, seconds: float) -> None:
        self._timeout = timeout
        self._seconds = seconds
        hard_deadline = timeout.when()
        assert hard_deadline is not None
        self._hard_deadline = hard_deadline
        self.expired = False

    def observe_attempt(
        self, protocol: str, payload: FrozenJsonObject
    ) -> Callable[[ModelProviderStreamPart], None]:
        mode = _wire_reasoning_mode(protocol, payload)
        if not self.expired and not self._timeout.expired():
            self._timeout.reschedule(self._hard_deadline)
        short_deadline: float | None = None
        started_reply = False
        tool_names: dict[int, str] = {}

        def observe(part: ModelProviderStreamPart) -> None:
            nonlocal short_deadline, started_reply
            if mode == "other" or self.expired or self._timeout.expired():
                return
            now = asyncio.get_running_loop().time()
            if short_deadline is not None and now >= short_deadline:
                self.expired = True
                return
            data = part.data
            if not isinstance(data, Mapping):
                return
            text = data.get("text")
            non_empty = isinstance(text, str) and bool(text.strip())
            if part.kind == ModelProviderStreamKind.REASONING and non_empty:
                if mode == "off":
                    self.expired = True
                    self._timeout.reschedule(now)
                elif not started_reply and short_deadline is None:
                    short_deadline = min(now + self._seconds, self._hard_deadline)
                    self._timeout.reschedule(short_deadline)
            elif mode == "low" and not started_reply:
                if part.kind == ModelProviderStreamKind.TEXT and non_empty:
                    started_reply = True
                elif part.kind == ModelProviderStreamKind.TOOL_CALL:
                    index, delta = data.get("index", 0), data.get("name_delta")
                    if isinstance(index, int) and isinstance(delta, str):
                        tool_names[index] = tool_names.get(index, "") + delta
                        started_reply = tool_names[index] == SPEAK_TOOL_NAME
                if started_reply and short_deadline is not None:
                    self._timeout.reschedule(self._hard_deadline)
                    short_deadline = None

        return observe


__all__ = [
    "ForkModelProfile",
    "ForkReasoningDecision",
    "ForkReasoningDecisionReason",
    "fork_model_profiles",
    "fork_profile_name",
]
