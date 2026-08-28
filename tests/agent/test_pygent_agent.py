from __future__ import annotations

import time
from dataclasses import replace

import pytest

from pygent import (
    AIMessage,
    Context,
    Message,
    Module,
    PygentAgent,
    PygentAgentContext,
    ToolDefinition,
    ToolMessage,
    UserMessage,
)
from pygent.agent import (
    ContextCompressionLimitExceeded,
    ContextCompressionUnavailable,
)
from pygent.runtime import ContextCodec, ExecutionOptions, LocalRuntime


class CallRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[Message, Context]] = []


class RecordingModel(Module[Message, AIMessage]):
    trusted_live_resource_attributes = ("recorder",)

    def __init__(self, answer: AIMessage) -> None:
        super().__init__()
        self.answer = answer
        self.recorder = CallRecorder()

    @property
    def calls(self) -> list[tuple[Message, Context]]:
        return self.recorder.calls

    async def forward(self, message: Message, context: Context):
        self.calls.append((message, context))
        return self.answer, context


class ContextChangingCompressor(RecordingModel):
    async def forward(self, message: Message, context: Context):
        self.calls.append((message, context))
        return self.answer, replace(context, metadata={"changed": True})


class EmptyTools(Module[AIMessage, ToolMessage]):
    async def forward(self, message: AIMessage, context: Context):
        raise AssertionError("tools should not be called")


class ToolFilteringModel(RecordingModel):
    def effective_tools(self, context: Context) -> tuple[ToolDefinition, ...]:
        return ()


def build_agent(
    *,
    model: Module[Message, AIMessage],
    compressor: Module[Message, AIMessage],
    context_window_tokens: int = 4096,
    compression_context_window_tokens: int | None = None,
    compression_prompt: str = "Summarize goals, constraints, and unfinished work.",
    max_compressions: int = 4,
) -> PygentAgent:
    return PygentAgent(
        system_prompt="You are a careful coding agent.",
        compression_prompt=compression_prompt,
        model=model,
        compressor=compressor,
        tools=EmptyTools(),
        context_window_tokens=context_window_tokens,
        compression_context_window_tokens=compression_context_window_tokens,
        max_compressions=max_compressions,
    )


def test_pygent_agent_context_has_a_portable_round_trip() -> None:
    context = PygentAgentContext(
        system_prompt="fixed",
        messages=(UserMessage(content="visible"),),
        full_history=(UserMessage(content="original"),),
        compression_count=2,
        input_token_scale_ppm=1_250_000,
        last_input_tokens=321,
        projection_revision=3,
    )

    codec = ContextCodec.dataclass(PygentAgentContext)
    decoded = codec.decode(codec.encode(context))

    assert decoded == context
    assert type(decoded) is PygentAgentContext
    assert codec.schema == "pygent.agent-context"
    assert codec.version == 2


def test_new_context_uses_the_agent_configuration() -> None:
    model = RecordingModel(AIMessage(content="done"))
    compressor = RecordingModel(AIMessage(content="summary"))
    agent = build_agent(model=model, compressor=compressor)

    context = agent.new_context(metadata={"workspace": "write"})

    assert context.system_prompt == "You are a careful coding agent."
    assert context.metadata["workspace"] == "write"
    assert context.full_history == ()
    assert context.input_token_scale_ppm == 1_100_000
    assert agent.react.model.model is model
    assert agent.react.model.compressor is compressor
    assert agent.react.model.compression_context_window_tokens == 4096


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"context_window_tokens": 0}, "context_window_tokens"),
        ({"compression_context_window_tokens": 0}, "compression_context_window_tokens"),
        ({"compression_trigger_ratio": 0}, "compression_trigger_ratio"),
        ({"compression_trigger_ratio": 1}, "compression_trigger_ratio"),
    ],
)
def test_agent_rejects_invalid_window_configuration(kwargs, message) -> None:
    base = {
        "system_prompt": "system",
        "compression_prompt": "compress",
        "model": RecordingModel(AIMessage(content="done")),
        "compressor": RecordingModel(AIMessage(content="summary")),
        "tools": EmptyTools(),
        "context_window_tokens": 4096,
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        PygentAgent(**base)


@pytest.mark.asyncio
async def test_managed_agent_auto_registers_context_schema_v2() -> None:
    agent = build_agent(
        model=RecordingModel(AIMessage(content="done")),
        compressor=RecordingModel(AIMessage(content="summary")),
    )
    runtime = LocalRuntime()
    handle = await runtime.bind(agent).start(
        UserMessage(content="hello"),
        agent.new_context(),
        execution=ExecutionOptions(deadline=time.monotonic() + 5),
    )

    answer, context = await handle.result()

    assert answer.content == "done"
    assert isinstance(context, PygentAgentContext)
    assert [message.content for message in context.full_history] == ["hello", "done"]
    await runtime.close()


@pytest.mark.asyncio
async def test_token_estimate_uses_foreground_effective_tools() -> None:
    model = ToolFilteringModel(AIMessage(content="done"))
    compressor = RecordingModel(AIMessage(content="summary"))
    agent = build_agent(
        model=model,
        compressor=compressor,
        context_window_tokens=1024,
    )
    hidden_tool = ToolDefinition(
        name="hidden",
        description="x" * 4096,
        parameters={"type": "object"},
    )

    await agent.invoke(
        UserMessage(content="hello"),
        agent.new_context(tools=(hidden_tool,)),
    )

    assert not compressor.calls
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_agent_forks_full_projection_and_preserves_current_and_full_history() -> None:
    answer = AIMessage(content="done", usage={"input_tokens": 120})
    model = RecordingModel(answer)
    compressor = RecordingModel(AIMessage(content="compact snapshot"))
    agent = build_agent(
        model=model,
        compressor=compressor,
        context_window_tokens=300,
        compression_context_window_tokens=4096,
    )
    original = (
        UserMessage(content="u1 " + "x" * 500),
        AIMessage(content="a1 " + "y" * 500),
        UserMessage(content="u2 " + "z" * 500),
    )
    context = PygentAgentContext(
        system_prompt=agent.system_prompt,
        messages=original,
        full_history=original,
        projection_revision=5,
    )
    current = UserMessage(content="continue")

    returned, final_context = await agent.invoke(current, context)

    assert returned == answer
    compression_request, compression_context = compressor.calls[0]
    assert compression_request.content == agent.compression_prompt
    assert compression_context.system_prompt == agent.system_prompt
    assert compression_context.messages == original
    assert compression_context.tools == ()

    foreground_current, foreground_context = model.calls[0]
    assert foreground_current == current
    assert foreground_context.messages[0].kind == "pygent.context.snapshot"
    assert foreground_context.messages[0].content == "compact snapshot"
    assert foreground_context.compression_count == 1
    assert foreground_context.projection_revision == 7

    assert final_context.full_history == original + (current, answer)
    assert [message.content for message in final_context.messages] == [
        "compact snapshot",
        "continue",
        "done",
    ]
    assert final_context.projection_revision == 8
    assert final_context.last_input_tokens == 120


@pytest.mark.asyncio
async def test_compression_prompt_can_trigger_early_compression() -> None:
    model = RecordingModel(AIMessage(content="done"))
    compressor = RecordingModel(AIMessage(content="summary"))
    agent = build_agent(
        model=model,
        compressor=compressor,
        context_window_tokens=10_000,
        compression_context_window_tokens=500,
        compression_prompt="p" * 1000,
    )
    old = UserMessage(content="old")
    context = replace(agent.new_context(), messages=(old,), full_history=(old,))

    await agent.invoke(UserMessage(content="current"), context)

    assert len(compressor.calls) == 1


@pytest.mark.asyncio
async def test_foreground_usage_increases_calibration_but_never_lowers_it() -> None:
    model = RecordingModel(
        AIMessage(content="done", usage={"input_tokens": 10_000})
    )
    agent = build_agent(
        model=model,
        compressor=RecordingModel(AIMessage(content="summary")),
        context_window_tokens=1_000_000,
    )

    _, first = await agent.invoke(UserMessage(content="hello"), agent.new_context())
    assert first.input_token_scale_ppm > 1_100_000
    assert first.last_input_tokens == 10_000

    second_agent = build_agent(
        model=RecordingModel(
            AIMessage(content="again", usage={"input_tokens": 1})
        ),
        compressor=RecordingModel(AIMessage(content="summary")),
        context_window_tokens=1_000_000,
    )
    _, second = await second_agent.invoke(UserMessage(content="next"), first)
    assert second.input_token_scale_ppm == first.input_token_scale_ppm
    assert second.last_input_tokens == 1


@pytest.mark.asyncio
async def test_missing_usage_preserves_calibration_state() -> None:
    model = RecordingModel(AIMessage(content="done"))
    agent = build_agent(
        model=model,
        compressor=RecordingModel(AIMessage(content="summary")),
    )
    context = replace(
        agent.new_context(),
        input_token_scale_ppm=1_300_000,
        last_input_tokens=99,
    )

    _, returned = await agent.invoke(UserMessage(content="hello"), context)

    assert returned.input_token_scale_ppm == 1_300_000
    assert returned.last_input_tokens == 99


@pytest.mark.asyncio
async def test_compressor_must_preserve_fork_context() -> None:
    agent = build_agent(
        model=RecordingModel(AIMessage(content="done")),
        compressor=ContextChangingCompressor(AIMessage(content="summary")),
        context_window_tokens=300,
        compression_context_window_tokens=4096,
    )
    old = UserMessage(content="x" * 2000)
    context = replace(agent.new_context(), messages=(old,), full_history=(old,))

    with pytest.raises(ContextCompressionUnavailable, match="preserve"):
        await agent.invoke(UserMessage(content="current"), context)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "summary",
    [AIMessage(content=""), AIMessage(content="summary", tool_calls=())],
)
async def test_compressor_result_must_be_non_empty_without_tool_calls(summary) -> None:
    if summary.content:
        from pygent.tool import ToolCall

        summary = replace(
            summary,
            tool_calls=(ToolCall(call_id="call", name="tool", arguments={}),),
        )
    agent = build_agent(
        model=RecordingModel(AIMessage(content="done")),
        compressor=RecordingModel(summary),
        context_window_tokens=300,
        compression_context_window_tokens=4096,
    )
    old = UserMessage(content="x" * 2000)
    context = replace(agent.new_context(), messages=(old,), full_history=(old,))

    with pytest.raises(ContextCompressionUnavailable, match="non-empty"):
        await agent.invoke(UserMessage(content="current"), context)


@pytest.mark.asyncio
async def test_oversized_request_without_history_cannot_be_compressed() -> None:
    agent = build_agent(
        model=RecordingModel(AIMessage(content="done")),
        compressor=RecordingModel(AIMessage(content="summary")),
        context_window_tokens=300,
        compression_context_window_tokens=4096,
    )

    with pytest.raises(ContextCompressionUnavailable, match="no projected history"):
        await agent.invoke(UserMessage(content="x" * 2000), agent.new_context())


@pytest.mark.asyncio
async def test_compression_request_must_fit_its_window() -> None:
    agent = build_agent(
        model=RecordingModel(AIMessage(content="done")),
        compressor=RecordingModel(AIMessage(content="summary")),
        context_window_tokens=10_000,
        compression_context_window_tokens=300,
        compression_prompt="p" * 2000,
    )
    old = UserMessage(content="old")
    context = replace(agent.new_context(), messages=(old,), full_history=(old,))

    with pytest.raises(ContextCompressionUnavailable, match="compression context window"):
        await agent.invoke(UserMessage(content="current"), context)


@pytest.mark.asyncio
async def test_compressed_request_must_fall_below_foreground_trigger() -> None:
    agent = build_agent(
        model=RecordingModel(AIMessage(content="done")),
        compressor=RecordingModel(AIMessage(content="summary")),
        context_window_tokens=300,
        compression_context_window_tokens=4096,
    )
    old = UserMessage(content="old " + "x" * 2000)
    context = replace(agent.new_context(), messages=(old,), full_history=(old,))

    with pytest.raises(ContextCompressionUnavailable, match="remains oversized"):
        await agent.invoke(UserMessage(content="y" * 2000), context)


@pytest.mark.asyncio
async def test_compression_budget_is_explicit() -> None:
    agent = build_agent(
        model=RecordingModel(AIMessage(content="done")),
        compressor=RecordingModel(AIMessage(content="summary")),
        context_window_tokens=300,
        compression_context_window_tokens=4096,
        max_compressions=1,
    )
    old = UserMessage(content="x" * 2000)
    context = replace(
        agent.new_context(),
        messages=(old,),
        full_history=(old,),
        compression_count=1,
    )

    with pytest.raises(ContextCompressionLimitExceeded):
        await agent.invoke(UserMessage(content="current"), context)


class ReplacingModel(Module[Message, AIMessage]):
    def __init__(self, *, revision_delta: int) -> None:
        super().__init__()
        self.revision_delta = revision_delta

    async def forward(self, message: Message, context: Context):
        return AIMessage(content="done"), replace(
            context,
            messages=(UserMessage(content="replacement"),),
            projection_revision=context.projection_revision + self.revision_delta,
        )


@pytest.mark.asyncio
async def test_react_rejects_an_invalid_model_projection_revision() -> None:
    agent = PygentAgent(
        system_prompt="fixed",
        compression_prompt="compress",
        model=ReplacingModel(revision_delta=2),
        compressor=RecordingModel(AIMessage(content="summary")),
        tools=EmptyTools(),
        context_window_tokens=4096,
    )

    with pytest.raises(ValueError, match="projection_revision"):
        await agent.invoke(UserMessage(content="hello"), agent.new_context())
