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
from pygent.tool import ToolCall, ToolResult


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


class EmptyTools(Module[AIMessage, ToolMessage]):
    async def forward(self, message: AIMessage, context: Context):
        raise AssertionError("tools should not be called")


class ToolFilteringModel(RecordingModel):
    def effective_tools(self, context: Context) -> tuple[ToolDefinition, ...]:
        return ()


def build_agent(
    *,
    model: RecordingModel,
    compression_model: RecordingModel,
    threshold: int,
    keep_recent_units: int = 1,
    max_compressions: int = 4,
) -> PygentAgent:
    return PygentAgent(
        system_prompt="You are a careful coding agent.",
        compression_prompt="Summarize goals, constraints, and unfinished work.",
        model=model,
        compression_model=compression_model,
        tools=EmptyTools(),
        compression_threshold_bytes=threshold,
        keep_recent_units=keep_recent_units,
        max_compressions=max_compressions,
    )


def test_pygent_agent_context_has_a_portable_round_trip() -> None:
    context = PygentAgentContext(
        system_prompt="fixed",
        messages=(UserMessage(content="visible"),),
        full_history=(UserMessage(content="original"),),
        compression_count=2,
        projection_revision=3,
    )

    codec = ContextCodec.dataclass(PygentAgentContext)
    decoded = codec.decode(codec.encode(context))

    assert decoded == context
    assert type(decoded) is PygentAgentContext
    assert codec.schema == "pygent.agent-context"
    assert codec.version == 1


def test_new_context_uses_the_agent_system_prompt() -> None:
    model = RecordingModel(AIMessage(content="done"))
    compressor = RecordingModel(AIMessage(content="summary"))
    agent = build_agent(model=model, compression_model=compressor, threshold=4096)

    context = agent.new_context(metadata={"workspace": "write"})

    assert context.system_prompt == "You are a careful coding agent."
    assert context.metadata["workspace"] == "write"
    assert context.full_history == ()
    assert agent.react.model.model is model
    assert agent.react.model.compression_model is compressor


@pytest.mark.asyncio
async def test_managed_agent_auto_registers_its_context_codec() -> None:
    model = RecordingModel(AIMessage(content="done"))
    compressor = RecordingModel(AIMessage(content="summary"))
    agent = build_agent(model=model, compression_model=compressor, threshold=4096)
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
async def test_compression_size_uses_foreground_effective_tools() -> None:
    model = ToolFilteringModel(AIMessage(content="done"))
    compressor = RecordingModel(AIMessage(content="summary"))
    agent = build_agent(model=model, compression_model=compressor, threshold=1024)
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
async def test_agent_compresses_projection_and_preserves_full_history() -> None:
    model = RecordingModel(AIMessage(content="done"))
    compressor = RecordingModel(AIMessage(content="compact snapshot"))
    agent = build_agent(model=model, compression_model=compressor, threshold=1)
    original = (
        UserMessage(content="u1"),
        AIMessage(content="a1"),
        UserMessage(content="u2"),
        AIMessage(content="a2"),
    )
    context = PygentAgentContext(
        system_prompt=agent.system_prompt,
        messages=original,
        full_history=original,
        projection_revision=5,
    )
    current = UserMessage(content="continue")

    answer, final_context = await agent.invoke(current, context)

    assert answer.content == "done"
    assert len(compressor.calls) == 1
    compression_request, compression_context = compressor.calls[0]
    assert compression_request.content == agent.compression_prompt
    assert compression_context.system_prompt == ""
    assert compression_context.messages == original
    assert compression_context.tools == ()

    foreground_current, foreground_context = model.calls[0]
    assert foreground_current == current
    assert foreground_context.system_prompt == agent.system_prompt
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


@pytest.mark.asyncio
async def test_compression_keeps_tool_call_and_result_in_one_unit() -> None:
    model = RecordingModel(AIMessage(content="done"))
    compressor = RecordingModel(AIMessage(content="summary"))
    agent = build_agent(model=model, compression_model=compressor, threshold=1)
    call = ToolCall(call_id="call-1", name="read", arguments={})
    ai = AIMessage(tool_calls=(call,))
    tool = ToolMessage(
        results=(
            ToolResult(
                call_id="call-1",
                name="read",
                status="succeeded",
                output={"text": "ok"},
            ),
        )
    )
    context = PygentAgentContext(
        system_prompt=agent.system_prompt,
        messages=(UserMessage(content="old"), ai),
        full_history=(UserMessage(content="old"), ai),
        projection_revision=2,
    )

    await agent.react.model.invoke(tool, context)

    foreground_current, foreground_context = model.calls[0]
    assert foreground_current == tool
    assert foreground_context.messages[1] == ai
    assert foreground_context.messages[0].kind == "pygent.context.snapshot"
    assert compressor.calls[0][1].messages == (UserMessage(content="old"),)


@pytest.mark.asyncio
async def test_compression_fails_when_no_complete_prefix_is_available() -> None:
    model = RecordingModel(AIMessage(content="done"))
    compressor = RecordingModel(AIMessage(content="summary"))
    agent = build_agent(model=model, compression_model=compressor, threshold=1)

    with pytest.raises(ContextCompressionUnavailable):
        await agent.invoke(UserMessage(content="oversized"), agent.new_context())
    assert not compressor.calls
    assert not model.calls


@pytest.mark.asyncio
async def test_compression_budget_is_explicit() -> None:
    model = RecordingModel(AIMessage(content="done"))
    compressor = RecordingModel(AIMessage(content="summary"))
    agent = build_agent(
        model=model,
        compression_model=compressor,
        threshold=1,
        max_compressions=1,
    )
    old = UserMessage(content="old")
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
        compression_model=RecordingModel(AIMessage(content="summary")),
        tools=EmptyTools(),
        compression_threshold_bytes=4096,
    )

    with pytest.raises(ValueError, match="projection_revision"):
        await agent.invoke(UserMessage(content="hello"), agent.new_context())
