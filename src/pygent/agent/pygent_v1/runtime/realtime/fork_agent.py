"""Fork-reply managed execution (D-RT2/3, pygent runtime form).

The interrupt path runs as a real pygent execution: a minimal single-module
agent graph — one managed model call with ``tool_choice`` forced to speak,
then argument extraction. No ReAct loop, no tool execution, no conversation
checkpoints, so the session transcript stays untouched while the runtime
provides admission, capacity coordination, a durable journal and crash
recovery (idempotent per interrupt via the execution idempotency key).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pygent.agent.pygent_v1.runtime.realtime.fork_reply import (
    ForkReplyOutcome,
    extract_speak_args,
)
from pygent.agent.pygent_v1.runtime.realtime.speak import (
    SPEAK_TOOL_NAME,
    parse_speak_args,
)
from pygent.core import (
    AIMessage,
    EffectSafety,
    ExecutionRequirements,
    Module,
    RecoverySafety,
    UserMessage,
)
from pygent.llm import (
    GenerationConfig,
    ModelCallLayer,
    ModelCallPolicy,
    ModelGroup,
    RetryPolicy,
)

FORK_RESULT_KIND = "pygent.v1.realtime.fork-result"


class ForkReplyModule(Module[UserMessage, AIMessage]):
    """One managed model call; the graph of the fork-reply execution."""

    execution_requirements = ExecutionRequirements(
        recovery_safety=RecoverySafety.MODULE_BOUNDARY_RETRY,
        effect_safety=EffectSafety.MANAGED_EFFECTS,
    )
    trusted_live_resource_attributes = ("config", "model_group_name", "llm", "model")

    def __init__(
        self,
        *,
        config: Any,
        model_group_name: str,
        llm: Any,
        tools: tuple,
        retry_policy: RetryPolicy,
    ) -> None:
        super().__init__()
        self.config = config
        self.model_group_name = model_group_name
        self.llm = llm
        self.model = ModelCallLayer(
            model_group=ModelGroup.deferred(name=f"pygent.v1:{model_group_name}"),
            policy=ModelCallPolicy(allow_profile_override=True),
            retry_policy=retry_policy,
            generation=GenerationConfig(tool_choice=SPEAK_TOOL_NAME),
            tools=tools,
            invoker=None,  # managed: the runtime invokes with admission
        )

    async def forward(
        self, message: UserMessage, context: Any
    ) -> tuple[AIMessage, Any]:
        answer, next_context = await self.model(message, context)
        outcome = _outcome_from_answer(answer)
        return (
            AIMessage(
                content="",
                kind=FORK_RESULT_KIND,
                data={
                    "via": outcome.via,
                    "speak_args": (
                        None
                        if outcome.speak_args is None
                        else {
                            "text": outcome.speak_args.text,
                            "attachments": list(outcome.speak_args.attachments),
                            "say_nothing": outcome.speak_args.say_nothing,
                        }
                    ),
                    "text": outcome.text,
                },
            ),
            next_context,
        )


def _outcome_from_answer(answer: AIMessage) -> ForkReplyOutcome:
    speak_args = extract_speak_args(answer)
    if speak_args is not None:
        return ForkReplyOutcome(
            via="speak", speak_args=speak_args, text=None, message=answer
        )
    return ForkReplyOutcome(
        via="text", speak_args=None, text=answer.content or "", message=answer
    )


def outcome_from_result(message: AIMessage) -> ForkReplyOutcome:
    """Decode the fork execution's result message back into an outcome."""

    if message.kind != FORK_RESULT_KIND:
        raise ValueError(
            f"execution result is not a fork reply ({message.kind!r})"
        )
    data = message.data
    if not isinstance(data, Mapping):
        raise ValueError(f"execution result data must be a mapping (got {type(data).__name__})")  # noqa: TRY004
    raw_speak = data.get("speak_args")
    if raw_speak is not None and not isinstance(raw_speak, Mapping):
        raise ValueError("fork reply speak_args must be a mapping")
    speak_args = None if raw_speak is None else parse_speak_args(dict(raw_speak))
    return ForkReplyOutcome(
        via=str(data.get("via") or ("speak" if speak_args else "text")),
        speak_args=speak_args,
        text=None if data.get("text") is None else str(data["text"]),
        message=message,
    )


__all__ = [
    "FORK_RESULT_KIND",
    "ForkReplyModule",
    "outcome_from_result",
]
