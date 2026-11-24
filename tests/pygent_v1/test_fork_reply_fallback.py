"""Fork-reply fallback: one wrapped-user-message retry when the first attempt
fails transiently (retryable model error or the per-attempt deadline) or
yields no deliverable reply (no ``speak`` call and empty content — the
observed signature of an endpoint that silently ignores the forced
``tool_choice``)."""

from __future__ import annotations

import asyncio
from functools import partial
from typing import Any

import pytest

from pygent.agent.pygent_v1.runtime.realtime.fork_agent import (
    FORK_FALLBACK_INSTRUCTIONS,
    FORK_MAX_MODEL_ATTEMPTS,
    FORK_RESULT_KIND,
    ForkReplyModule,
    outcome_from_result,
)
from pygent.agent.pygent_v1.runtime.realtime.fork_reply import (
    build_fork_context,
)
from pygent.core import AIMessage, Context, ToolMessage, UserMessage
from pygent.llm import (
    AnthropicMessagesAdapter,
    GenerationConfig,
    ModelCallError,
    ModelConfig,
    ModelErrorKind,
    ModelProviderRequest,
    RetryPolicy,
)
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


class _FailingForkModel:
    """Raises scripted failures FIFO, then answers FIFO."""

    def __init__(
        self,
        *failures: BaseException,
        answers: tuple[AIMessage, ...] = (),
    ) -> None:
        self.failures = list(failures)
        self.answers = list(answers)
        self.calls: list[UserMessage] = []

    async def __call__(
        self, message: UserMessage, context: Any
    ) -> tuple[AIMessage, Any]:
        self.calls.append(message)
        if self.failures:
            raise self.failures.pop(0)
        return self.answers.pop(0), context


def _module(
    *answers: AIMessage,
    model: Any | None = None,
    model_call_timeout: float = 60.0,
    model_factory: Any = _ScriptedForkModel,
) -> tuple[ForkReplyModule, Any]:
    module = ForkReplyModule(
        config=None,
        model_group_name="test",
        llm=None,
        tools=(),
        retry_policy=RetryPolicy(),
        model_call_timeout=model_call_timeout,
    )
    if model is None:
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
    fails on its own slot and the wrapped retry runs on the second reserved
    one instead of ending the fork."""

    class _SlowOnceModel(_ScriptedForkModel):
        def __init__(self, *answers: AIMessage, first_delay: float) -> None:
            super().__init__(*answers)
            self.first_delay = first_delay
            self.slowed = False

        async def __call__(self, message: UserMessage, context: Any):
            if not self.slowed:
                # The first attempt times out mid-sleep, before the scripted
                # call is ever recorded — guard with a flag, not ``calls``.
                self.slowed = True
                await asyncio.sleep(self.first_delay)
            return await super().__call__(message, context)

    module, model = _module(
        _speak_answer("迟到后兜底"),
        model_factory=partial(_SlowOnceModel, first_delay=0.3),
        model_call_timeout=0.05,
    )
    context = _context()

    result, _ = await module.forward(_interrupt(), context)

    # the timeout fired inside the first attempt, on its own slot — before
    # the scripted model recorded anything; the wrapped retry consumed the
    # scripted answer on the second reserved slot
    assert len(model.calls) == 1
    assert model.calls[0][0].content.startswith("<user-interrupt>")
    assert result.data["attempts"] == FORK_MAX_MODEL_ATTEMPTS
    outcome = outcome_from_result(result)
    assert outcome.speak_args is not None
    assert outcome.speak_args.text == "迟到后兜底"


@pytest.mark.asyncio
async def test_retryable_model_error_retries_with_wrapped_message() -> None:
    model = _FailingForkModel(
        ModelCallError("provider hiccup", kind=ModelErrorKind.RATE_LIMIT),
        answers=(_speak_answer("重试成功"),),
    )
    module, _ = _module(model=model)
    context = _context()

    result, next_context = await module.forward(_interrupt(), context)

    assert len(model.calls) == FORK_MAX_MODEL_ATTEMPTS
    assert model.calls[1].content.startswith("<user-interrupt>")
    # the retry reuses the original context: the failed attempt stays out
    assert next_context is context
    assert result.data["attempts"] == FORK_MAX_MODEL_ATTEMPTS
    assert result.data["via"] == "speak"
    outcome = outcome_from_result(result)
    assert outcome.speak_args is not None
    assert outcome.speak_args.text == "重试成功"


@pytest.mark.asyncio
async def test_first_attempt_timeout_retries_with_wrapped_message() -> None:
    model = _FailingForkModel(TimeoutError(), answers=(_speak_answer("超时兜底"),))
    module, _ = _module(model=model)
    context = _context()

    result, _ = await module.forward(_interrupt(), context)

    assert len(model.calls) == FORK_MAX_MODEL_ATTEMPTS
    assert result.data["attempts"] == FORK_MAX_MODEL_ATTEMPTS
    outcome = outcome_from_result(result)
    assert outcome.speak_args is not None
    assert outcome.speak_args.text == "超时兜底"


@pytest.mark.asyncio
async def test_persistent_timeout_propagates_after_both_attempts() -> None:
    model = _FailingForkModel(TimeoutError(), TimeoutError())
    module, _ = _module(model=model)

    with pytest.raises(TimeoutError):
        await module.forward(_interrupt(), _context())

    assert len(model.calls) == FORK_MAX_MODEL_ATTEMPTS


@pytest.mark.asyncio
async def test_non_retryable_model_error_propagates_without_retry() -> None:
    model = _FailingForkModel(
        ModelCallError("bad credential", kind=ModelErrorKind.AUTHENTICATION)
    )
    module, _ = _module(model=model)

    with pytest.raises(ModelCallError):
        await module.forward(_interrupt(), _context())

    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_unexpected_exception_propagates_without_retry() -> None:
    model = _FailingForkModel(ValueError("unexpected"))
    module, _ = _module(model=model)

    with pytest.raises(ValueError):
        await module.forward(_interrupt(), _context())

    assert len(model.calls) == 1


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


def _dangling_answer() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=(
            ToolCall(call_id="call-1", name="bash", arguments={"command": "pytest"}),
            ToolCall(call_id="call-2", name="bash", arguments={"command": "lint"}),
        ),
    )


def test_fork_context_pairs_dangling_tool_calls() -> None:
    """A snapshot taken mid-tool-batch ends with unanswered tool_calls: the
    fork request pairs them with truthful in-flight results — the calls are
    still running in the foreground, not cancelled."""

    messages = (UserMessage(content="先跑测试"), _dangling_answer())

    context = build_fork_context(system_prompt="p", messages=messages, tools=())

    assert context.messages[:-1] == messages
    paired = context.messages[-1]
    assert isinstance(paired, ToolMessage)
    assert [result.call_id for result in paired.results] == ["call-1", "call-2"]
    for result in paired.results:
        assert result.status == "unknown"
        assert result.side_effect_committed is None
        assert "still executing in the foreground session" in str(result.output)


def test_fork_context_keeps_clean_tails_verbatim() -> None:
    for tail in (UserMessage(content="hello"), AIMessage(content="final answer")):
        messages = (UserMessage(content="q"), tail)
        context = build_fork_context(system_prompt="p", messages=messages, tools=())
        assert context.messages == messages


def test_normalized_fork_context_encodes_paired_tool_results() -> None:
    """The wire shape: the dangling tool_use is followed by matching
    tool_result ids in the next user turn — what strict providers require."""

    config = ModelConfig.from_mapping(
        {
            "connections": {
                "anthropic": {
                    "provider": "anthropic",
                    "credential": {"none": True},
                    "protocols": {
                        "anthropic_messages": {"base_url": "http://127.0.0.1:1"}
                    },
                }
            },
            "models": {
                "anthropic": {
                    "connection": "anthropic",
                    "model_id": "claude-sonnet-5",
                    "protocol": "anthropic_messages",
                    "capabilities": {
                        "modalities": {"input": ["text"], "output": ["text"]},
                        "streaming": {"output": ["text"]},
                        "tools": {
                            "call": True,
                            "choice": ["auto"],
                            "parallel": False,
                        },
                        "structured_output": {
                            "json_object": False,
                            "json_schema": False,
                        },
                        "reasoning": {"supported": False, "controllable": False},
                        "limits": {
                            "context_tokens": 100000,
                            "max_output_tokens": 4096,
                        },
                    },
                }
            },
        }
    )
    entry = config.models["anthropic"]
    context = build_fork_context(
        system_prompt="p",
        messages=(UserMessage(content="跑一下"), _dangling_answer()),
        tools=(),
    )

    body = AnthropicMessagesAdapter().build_request(
        ModelProviderRequest(
            model_key=entry.key,
            model=entry.spec,
            message=_interrupt(),
            context=context,
            generation=GenerationConfig(max_output_tokens=4096),
        )
    ).to_dict()

    encoded = body["messages"]
    assistant = encoded[-3]
    results = encoded[-2]
    assert assistant["role"] == "assistant"
    tool_use_ids = [
        block["id"]
        for block in assistant["content"]
        if block["type"] == "tool_use"
    ]
    assert tool_use_ids == ["call-1", "call-2"]
    assert results["role"] == "user"
    assert [
        block["tool_use_id"]
        for block in results["content"]
        if block["type"] == "tool_result"
    ] == ["call-1", "call-2"]
    assert all(block["is_error"] for block in results["content"])
