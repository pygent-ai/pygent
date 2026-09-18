"""Provider-specific behavior composed with protocol adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pygent.core import FrozenJsonObject


@dataclass(frozen=True, slots=True)
class ProviderProtocolPolicy:
    validate_options: Callable[[FrozenJsonObject], None] | None = None
    requires_signed_thinking: bool = False


_ALIYUN_TOKEN_PLAN_FIELDS = frozenset(
    {
        "enable_thinking",
        "preserve_thinking",
        "reasoning_effort",
        "thinking",
        "thinking_budget",
        "tool_stream",
    }
)
_ALIYUN_REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
)


def _validate_aliyun_token_plan(options: FrozenJsonObject) -> None:
    unknown = set(options) - _ALIYUN_TOKEN_PLAN_FIELDS
    if unknown:
        raise ValueError(
            "unknown Alibaba Token Plan provider options: " + ", ".join(sorted(unknown))
        )
    for key in ("enable_thinking", "preserve_thinking", "tool_stream"):
        if key in options and not isinstance(options[key], bool):
            raise TypeError(f"provider option {key!r} must be a bool")
    if (
        "reasoning_effort" in options
        and options["reasoning_effort"] not in _ALIYUN_REASONING_EFFORTS
    ):
        raise ValueError("provider option 'reasoning_effort' is not supported")
    if "thinking" in options:
        thinking = options["thinking"]
        if not isinstance(thinking, FrozenJsonObject):
            raise TypeError("provider option 'thinking' must be an object")
        if set(thinking) != {"type"}:
            raise ValueError("provider option 'thinking' accepts only the 'type' field")
        if thinking["type"] not in ("adaptive", "disabled"):
            raise ValueError(
                "provider option 'thinking.type' must be 'adaptive' or 'disabled'"
            )
    if "thinking_budget" in options:
        budget = options["thinking_budget"]
        if not isinstance(budget, int) or isinstance(budget, bool) or budget < 0:
            raise ValueError(
                "provider option 'thinking_budget' must be a non-negative integer"
            )


def _validate_deepseek_chat_completions(options: FrozenJsonObject) -> None:
    if "thinking" not in options:
        return
    thinking = options["thinking"]
    if not isinstance(thinking, FrozenJsonObject):
        raise TypeError("provider option 'thinking' must be an object")
    if set(thinking) != {"type"}:
        raise ValueError("provider option 'thinking' accepts only the 'type' field")
    if thinking["type"] not in ("enabled", "disabled"):
        raise ValueError(
            "provider option 'thinking.type' must be 'enabled' or 'disabled'"
        )


_DEFAULT_POLICY = ProviderProtocolPolicy()
_POLICIES = {
    ("aliyun_token_plan", "openai_chat_completions"): ProviderProtocolPolicy(
        validate_options=_validate_aliyun_token_plan
    ),
    ("deepseek", "openai_chat_completions"): ProviderProtocolPolicy(
        validate_options=_validate_deepseek_chat_completions
    ),
    ("anthropic", "anthropic_messages"): ProviderProtocolPolicy(
        requires_signed_thinking=True
    ),
}


def provider_protocol_policy(provider: str, protocol: str) -> ProviderProtocolPolicy:
    return _POLICIES.get((provider, protocol), _DEFAULT_POLICY)


__all__ = ["ProviderProtocolPolicy", "provider_protocol_policy"]
