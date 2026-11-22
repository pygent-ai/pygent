"""Fork-reply fallback: one wrapped-user-message retry when the first
attempt yields no deliverable reply (no ``speak`` call and empty content —
the observed signature of an endpoint that silently ignores the forced
``tool_choice``)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from pygent.agent.pygent_v1.runtime.realtime.fork_agent import (
    FORK_FALLBACK_INSTRUCTIONS,
    FORK_MAX_MODEL_ATTEMPTS,
    FORK_RESULT_KIND,
    ForkReplyModule,
    outcome_from_result,
)
from pygent.core import AIMessage, Context, UserMessage
from pygent.llm import RetryPolicy
from pygent.tool import ToolCall


class _ScriptedForkModel:
    """Stands in for the module's ``ModelCallLayer``; scripts FIFO answers."""

    def __init__(self, *answers: AIMessage) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[UserMessage, Any]] = []

    async def __call__(
        self, message: UserMessage, context: Any
    ) -> tuple[AIMessage, Any]:
        self.calls.append((message, context))
        return self.answers.pop(0), context


def _module(
    *answers: AIMessage,
    model_call_timeout: float = 60.0,
    model_factory: type[_ScriptedForkModel] = _ScriptedForkModel,
) -> tuple[ForkReplyModule, _ScriptedForkModel]:
    module = ForkReplyModule(
        config=None,
        model_group_name="test",
        llm=None,
        tools=(),
        retry_policy=RetryPolicy(),
        model_call_timeout=model_call_timeout,
    )
    model = model_factory(*answers)
    module.model = model
    return module, model


def _interrupt(content: str = "1. 为啥要环境变量？") -> UserMessage:
    return UserMessage(
        content=content,
        kind="pygent.v1.user.interrupt",
    )


def _speak_answer(
    text: str | None = None, *, say_nothing: bool = False
) -> AIMessage:
    arguments: dict[str, object] = {"say_nothing": say_nothing}
    if text is not None:
        arguments["text"] = text
    return AIMessage(
        content="",
        tool_calls=(ToolCall(call_id="call-1", name="speak", arguments=arguments),),
    )


def _phantom_answer() -> AIMessage:
    """The observed failure: phantom tool calls, reasoning, empty content."""

    return AIMessage(
        content="",
        tool_calls=(
            ToolCall(call_id="call-1", name="bash", arguments={"command": "pwd"}),
        ),
    )


def _context() -> Context:
    return Context(system_prompt="foreground prompt", messages=(), tools=())


@pytest.mark.asyncio
async def test_undeliverable_first_answer_retries_with_wrapped_message() -> None:
    original = _interrupt()
    module, model = _module(_phantom_answer(), _speak_answer("收到，马上说明。"))
    context = _context()

    result, next_context = await module.forward(original, context)

    assert len(model.calls) == FORK_MAX_MODEL_ATTEMPTS
    first_message, first_context = model.calls[0]
    assert first_message is original
    assert first_context is context
    retry_message, retry_context = model.calls[1]
    # the retry reuses the original context: the failed attempt stays out
    assert retry_context is context
    assert retry_message.kind == original.kind
    assert retry_message.content.startswith(
        "<user-interrupt>\n1. 为啥要环境变量？\n</user-interrupt>"
    )
    assert "call `speak`" in retry_message.content
    assert "say_nothing=true" in retry_message.content

    assert result.kind == FORK_RESULT_KIND
    assert result.data["via"] == "speak"
    assert result.data["attempts"] == FORK_MAX_MODEL_ATTEMPTS
    outcome = outcome_from_result(result)
    assert outcome.speak_args is not None
    assert outcome.speak_args.text == "收到，马上说明。"
    assert next_context is context


@pytest.mark.asyncio
async def test_deliverable_speak_answer_skips_the_retry() -> None:
    module, model = _module(_speak_answer("done"))
    context = _context()

    result, _ = await module.forward(_interrupt(), context)

    assert len(model.calls) == 1
    assert result.data["attempts"] == 1
    assert result.data["via"] == "speak"


@pytest.mark.asyncio
async def test_deliverable_plain_text_skips_the_retry() -> None:
    module, model = _module(AIMessage(content="plain answer"))
    context = _context()

    result, _ = await module.forward(_interrupt(), context)

    assert len(model.calls) == 1
    assert result.data["attempts"] == 1
    assert result.data["via"] == "text"
    assert result.data["text"] == "plain answer"


@pytest.mark.asyncio
async def test_say_nothing_speak_call_skips_the_retry() -> None:
    module, model = _module(_speak_answer(say_nothing=True))
    context = _context()

    result, _ = await module.forward(_interrupt(), context)

    assert len(model.calls) == 1
    assert result.data["attempts"] == 1
    outcome = outcome_from_result(result)
    assert outcome.speak_args is not None
    assert outcome.speak_args.say_nothing is True


@pytest.mark.asyncio
async def test_retry_exhaustion_still_reports_the_outcome() -> None:
    module, model = _module(_phantom_answer(), AIMessage(content=""))
    context = _context()

    result, _ = await module.forward(_interrupt(), context)

    assert len(model.calls) == FORK_MAX_MODEL_ATTEMPTS
    assert result.data["attempts"] == FORK_MAX_MODEL_ATTEMPTS
    outcome = outcome_from_result(result)
    assert outcome.speak_args is None
    assert outcome.text == ""


@pytest.mark.asyncio
async def test_invalid_speak_args_with_empty_content_retries() -> None:
    """``speak({})`` counts as "no speak call": empty content -> one retry."""

    module, model = _module(_speak_answer(), _speak_answer("补试成功"))
    context = _context()

    result, _ = await module.forward(_interrupt(), context)

    assert len(model.calls) == FORK_MAX_MODEL_ATTEMPTS
    retry_message, retry_context = model.calls[1]
    assert retry_context is context
    assert retry_message.content.startswith("<user-interrupt>")
    assert result.data["via"] == "speak"
    assert result.data["attempts"] == FORK_MAX_MODEL_ATTEMPTS
    outcome = outcome_from_result(result)
    assert outcome.speak_args is not None
    assert outcome.speak_args.text == "补试成功"


@pytest.mark.asyncio
async def test_invalid_speak_args_with_body_text_delivered_as_text() -> None:
    """``speak({})`` alongside body text delivers the text without a retry."""

    answer = AIMessage(
        content="正文在 content 里",
        tool_calls=(ToolCall(call_id="call-1", name="speak", arguments={}),),
    )
    module, model = _module(answer)
    context = _context()

    result, _ = await module.forward(_interrupt(), context)

    assert len(model.calls) == 1
    assert result.data["attempts"] == 1
    assert result.data["via"] == "text"
    assert result.data["text"] == "正文在 content 里"


@pytest.mark.asyncio
async def test_model_call_timeout_caps_each_attempt() -> None:
    """Each attempt is capped at ``model_call_timeout`` — a slow first call
    fails on its own slot instead of eating into the retry's."""

    class _SlowModel(_ScriptedForkModel):
        async def __call__(self, message: UserMessage, context: Any):
            await asyncio.sleep(0.3)
            return await super().__call__(message, context)

    module, model = _module(
        _speak_answer("迟到"),
        model_factory=_SlowModel,
        model_call_timeout=0.05,
    )
    context = _context()

    with pytest.raises(TimeoutError):
        await module.forward(_interrupt(), context)

    # the timeout fired inside the first attempt, on its own slot: the
    # retry's scripted answer was never consumed
    assert len(model.answers) == 1


def test_fallback_wrapper_keeps_the_interrupt_verbatim() -> None:
    from pygent.agent.pygent_v1.runtime.realtime.fork_agent import (
        _wrap_fallback_message,
    )

    original = _interrupt("line one\nline two")
    wrapped = _wrap_fallback_message(original)

    assert wrapped.content.startswith(
        "<user-interrupt>\nline one\nline two\n</user-interrupt>"
    )
    assert wrapped.content.endswith(FORK_FALLBACK_INSTRUCTIONS.strip())
    assert wrapped.kind == original.kind
    assert wrapped.media == original.media
