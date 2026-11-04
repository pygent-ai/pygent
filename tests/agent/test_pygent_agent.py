from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, replace
from io import BytesIO
from typing import ClassVar

import pytest
from PIL import Image

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
    ContextCompressionLayer,
    ContextCompressionLimitExceeded,
    ContextCompressionUnavailable,
)
from pygent.runtime import (
    ContextCodec,
    ExecutionOptions,
    LocalRuntime,
    SQLiteHistoryStore,
)
from pygent.runtime.codec import invocation_to_dict
from pygent.runtime.context_codec import ContextCodecRegistry
from pygent.tool import MediaBlock, MediaSource, ToolCall, ToolResult


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


class CompressorDown(RuntimeError):
    pass


class FailureState:
    def __init__(self, failures: int) -> None:
        self.remaining = failures


class FlakyCompressor(RecordingModel):
    trusted_live_resource_attributes = ("recorder", "state")

    def __init__(self, answer: AIMessage, failures: int = 1) -> None:
        super().__init__(answer)
        self.state = FailureState(failures)

    async def forward(self, message: Message, context: Context):
        self.calls.append((message, context))
        if self.state.remaining > 0:
            self.state.remaining -= 1
            raise CompressorDown("compressor down")
        return self.answer, context


class GatedCompressor(RecordingModel):
    trusted_live_resource_attributes = ("recorder", "entered", "gate")

    def __init__(self, answer: AIMessage) -> None:
        super().__init__(answer)
        self.entered = asyncio.Event()
        self.gate = asyncio.Event()

    async def forward(self, message: Message, context: Context):
        self.entered.set()
        await self.gate.wait()
        return await super().forward(message, context)


class EchoCompressor(Module[Message, AIMessage]):
    trusted_live_resource_attributes = ()

    async def forward(self, message: Message, context: Context):
        marker = context.messages[0].content.split(" ")[0]
        return AIMessage(content=f"snapshot:{marker}"), context


class EmptyTools(Module[AIMessage, ToolMessage]):
    async def forward(self, message: AIMessage, context: Context):
        raise AssertionError("tools should not be called")


class LongSessionState:
    def __init__(self) -> None:
        self.call_index = 0


class LongSessionModel(Module[Message, AIMessage]):
    trusted_live_resource_attributes = ("state",)

    def __init__(self, *, tool_calls_per_invocation: int) -> None:
        super().__init__()
        self.tool_calls_per_invocation = tool_calls_per_invocation
        self.state = LongSessionState()

    async def forward(self, message: Message, context: Context):
        invocation_index = self.state.call_index % (
            self.tool_calls_per_invocation + 1
        )
        self.state.call_index += 1
        if invocation_index == self.tool_calls_per_invocation:
            return AIMessage(content="done"), context
        return (
            AIMessage(
                content="call",
                tool_calls=(
                    ToolCall(
                        call_id=f"call-{self.state.call_index}",
                        name="lookup",
                        arguments={},
                    ),
                ),
            ),
            context,
        )


class LongSessionTools(Module[AIMessage, ToolMessage]):
    async def forward(self, message: AIMessage, context: Context):
        call = message.tool_calls[0]
        return (
            ToolMessage(
                results=(
                    ToolResult(
                        call_id=call.call_id,
                        name=call.name,
                        status="succeeded",
                        output={"value": 1},
                    ),
                )
            ),
            context,
        )


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
        committed_messages=(UserMessage(content="original"),),
        compression_count=2,
        input_token_scale_ppm=1_250_000,
        last_input_tokens=321,
        projection_revision=3,
    )

    codec = ContextCodec.dataclass(PygentAgentContext)
    encoded = codec.encode(context)
    decoded = codec.decode(encoded)

    assert decoded == context
    assert type(decoded) is PygentAgentContext
    assert codec.schema == "pygent.agent-context"
    assert codec.version == 3
    assert "committed_messages" in encoded
    assert "full_history" not in encoded


def test_pygent_agent_context_rejects_the_removed_history_field() -> None:
    with pytest.raises(TypeError, match="full_history"):
        PygentAgentContext(full_history=())  # type: ignore[call-arg]


def test_new_context_uses_the_agent_configuration() -> None:
    model = RecordingModel(AIMessage(content="done"))
    compressor = RecordingModel(AIMessage(content="summary"))
    agent = build_agent(model=model, compressor=compressor)

    context = agent.new_context(metadata={"workspace": "write"})

    assert context.system_prompt == "You are a careful coding agent."
    assert context.metadata["workspace"] == "write"
    assert context.committed_messages == ()
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
async def test_managed_agent_auto_registers_context_schema_v3() -> None:
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
    assert [message.content for message in context.committed_messages] == [
        "hello",
        "done",
    ]
    await runtime.close()


@pytest.mark.asyncio
async def test_pygent_agent_context_v3_survives_durable_reattach(tmp_path) -> None:
    agent = build_agent(
        model=RecordingModel(AIMessage(content="done")),
        compressor=RecordingModel(AIMessage(content="summary")),
    )
    async with SQLiteHistoryStore(tmp_path / "pygent-agent-v3.sqlite3") as history:
        runtime = LocalRuntime(history=history)
        handle = await runtime.bind(agent).start(
            UserMessage(content="hello"),
            agent.new_context(),
            execution=ExecutionOptions(deadline=time.monotonic() + 5),
        )
        expected = await handle.result()
        attached = await runtime.get_execution_handle(handle.execution_id)

        assert await attached.result() == expected
        assert isinstance(expected[1], PygentAgentContext)
        assert [message.content for message in expected[1].committed_messages] == [
            "hello",
            "done",
        ]
        await runtime.close()


@pytest.mark.asyncio
async def test_agent_invocation_reset_preserves_a_concrete_context_subtype() -> None:
    @dataclass(frozen=True, slots=True)
    class DerivedContext(PygentAgentContext):
        context_schema: ClassVar[str] = "tests.pygent-derived"
        context_schema_version: ClassVar[int] = 1
        workspace: str = "repo"

    class DerivedAgent(PygentAgent):
        context_type = DerivedContext

    agent = DerivedAgent(
        system_prompt="system",
        compression_prompt="compress",
        model=RecordingModel(AIMessage(content="done")),
        compressor=RecordingModel(AIMessage(content="summary")),
        tools=EmptyTools(),
        context_window_tokens=4096,
    )
    context = DerivedContext(
        committed_messages=(UserMessage(content="previous"),),
        compression_count=3,
    )

    _, returned = await agent.invoke(UserMessage(content="current"), context)

    assert type(returned) is DerivedContext
    assert returned.workspace == "repo"
    assert [message.content for message in returned.committed_messages] == [
        "current",
        "done",
    ]
    assert returned.compression_count == 0


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
async def test_image_transport_base64_is_not_estimated_as_text() -> None:
    stream = BytesIO()
    Image.new("RGB", (1024, 1024), color=(12, 34, 56)).save(stream, format="BMP")
    media = MediaBlock(
        media_type="image",
        mime_type="image/bmp",
        source=MediaSource.inline(stream.getvalue()),
        detail="high",
    )
    model = RecordingModel(AIMessage(content="done"))
    compressor = RecordingModel(AIMessage(content="summary"))
    agent = build_agent(
        model=model,
        compressor=compressor,
        context_window_tokens=10_000,
    )
    history = ToolMessage(
        results=(
            ToolResult(
                call_id="image-call",
                name="read_image",
                status="succeeded",
                output={"size_bytes": media.source.size_bytes},
                content=(media,),
            ),
        )
    )
    context = replace(agent.new_context(), messages=(history,))

    await agent.invoke(UserMessage(content="describe it"), context)

    assert not compressor.calls
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_video_transport_base64_is_not_estimated_as_text() -> None:
    media = MediaBlock(
        media_type="video",
        mime_type="video/mp4",
        source=MediaSource.inline(
            b"\x00\x00\x00\x18ftypmp42" + b"x" * 2_000_000
        ),
        width=1280,
        height=720,
        duration_seconds=2,
        fps=24,
        has_audio=True,
    )
    model = RecordingModel(AIMessage(content="done"))
    compressor = RecordingModel(AIMessage(content="summary"))
    agent = build_agent(
        model=model,
        compressor=compressor,
        context_window_tokens=10_000,
    )
    history = ToolMessage(
        results=(
            ToolResult(
                call_id="video-call",
                name="read_video",
                status="succeeded",
                content=(media,),
            ),
        )
    )

    await agent.invoke(
        UserMessage(content="describe it"),
        replace(agent.new_context(), messages=(history,)),
    )

    assert not compressor.calls
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_image_usage_does_not_recalibrate_text_scale() -> None:
    stream = BytesIO()
    Image.new("RGB", (32, 32), color=(1, 2, 3)).save(stream, format="PNG")
    media = MediaBlock(
        media_type="image",
        mime_type="image/png",
        source=MediaSource.inline(stream.getvalue()),
    )
    model = RecordingModel(AIMessage(content="done", usage={"input_tokens": 10_000}))
    agent = build_agent(
        model=model,
        compressor=RecordingModel(AIMessage(content="summary")),
        context_window_tokens=100_000,
    )
    history = ToolMessage(
        results=(
            ToolResult(
                call_id="image-call",
                name="read_image",
                status="succeeded",
                output={},
                content=(media,),
            ),
        )
    )

    _, returned = await agent.invoke(
        UserMessage(content="describe it"),
        replace(agent.new_context(), messages=(history,)),
    )

    assert returned.input_token_scale_ppm == 1_100_000
    assert returned.last_input_tokens == 10_000


@pytest.mark.asyncio
async def test_agent_forks_projection_and_preserves_current_commits() -> None:
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
        committed_messages=(UserMessage(content="previous invocation"),),
        compression_count=3,
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

    assert final_context.committed_messages == (current, answer)
    assert [message.content for message in final_context.messages] == [
        "compact snapshot",
        "continue",
        "done",
    ]
    assert final_context.projection_revision == 8
    assert final_context.last_input_tokens == 120


@pytest.mark.asyncio
async def test_compressor_failure_keeps_context_and_recovers() -> None:
    model = RecordingModel(AIMessage(content="done"))
    compressor = FlakyCompressor(AIMessage(content="compact snapshot"))
    agent = build_agent(
        model=model,
        compressor=compressor,
        context_window_tokens=300,
        compression_context_window_tokens=4096,
    )
    original = replace(
        agent.new_context(),
        messages=(UserMessage(content="old " + "x" * 2000),),
    )

    with pytest.raises(CompressorDown):
        await agent.invoke(UserMessage(content="current"), original)

    assert original.compression_count == 0
    assert [message.content for message in original.messages] == [
        "old " + "x" * 2000
    ]
    assert not model.calls
    assert len(compressor.calls) == 1

    answer, final_context = await agent.invoke(
        UserMessage(content="current"), original
    )

    assert answer.content == "done"
    assert [message.content for message in final_context.messages] == [
        "compact snapshot",
        "current",
        "done",
    ]
    assert final_context.compression_count == 1
    assert len(model.calls) == 1
    assert len(compressor.calls) == 2


@pytest.mark.asyncio
async def test_manual_compress_propagates_failure_without_pollution() -> None:
    compressor = FlakyCompressor(AIMessage(content="compact snapshot"), failures=10)
    agent = build_agent(
        model=RecordingModel(AIMessage(content="done")),
        compressor=compressor,
    )
    original = replace(
        agent.new_context(),
        messages=(UserMessage(content="old " + "x" * 200),),
        compression_count=1,
        projection_revision=3,
    )

    with pytest.raises(CompressorDown):
        await agent.compress(original)

    assert original.compression_count == 1
    assert original.projection_revision == 3
    assert [message.content for message in original.messages] == [
        "old " + "x" * 200
    ]
    assert len(compressor.calls) == 1


@pytest.mark.asyncio
async def test_cancelled_manual_compress_leaves_no_state() -> None:
    compressor = GatedCompressor(AIMessage(content="compact snapshot"))
    agent = build_agent(
        model=RecordingModel(AIMessage(content="done")),
        compressor=compressor,
    )
    original = replace(
        agent.new_context(),
        messages=(UserMessage(content="old " + "x" * 200),),
        compression_count=2,
        projection_revision=4,
    )

    task = asyncio.create_task(agent.compress(original))
    await compressor.entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert original.compression_count == 2
    assert original.projection_revision == 4
    assert [message.content for message in original.messages] == [
        "old " + "x" * 200
    ]

    compressor.gate.set()
    compressed = await agent.compress(original)

    assert [message.content for message in compressed.messages] == [
        "compact snapshot"
    ]
    assert compressed.compression_count == 3
    assert compressed.projection_revision == 5
    await asyncio.sleep(0.1)  # drain the orphaned Root call behind the gate


@pytest.mark.asyncio
async def test_concurrent_manual_compresses_stay_isolated() -> None:
    agent = build_agent(
        model=RecordingModel(AIMessage(content="done")),
        compressor=EchoCompressor(),
    )
    alpha = replace(
        agent.new_context(),
        messages=(UserMessage(content="alpha " + "x" * 200),),
        compression_count=1,
        projection_revision=1,
    )
    beta = replace(
        agent.new_context(),
        messages=(UserMessage(content="beta " + "y" * 200),),
        compression_count=7,
        projection_revision=9,
    )

    alpha_out, beta_out = await asyncio.gather(
        agent.compress(alpha),
        agent.compress(beta),
    )

    assert [message.content for message in alpha_out.messages] == [
        "snapshot:alpha"
    ]
    assert alpha_out.compression_count == 2
    assert alpha_out.projection_revision == 2
    assert [message.content for message in beta_out.messages] == [
        "snapshot:beta"
    ]
    assert beta_out.compression_count == 8
    assert beta_out.projection_revision == 10
    assert [message.content for message in alpha.messages] == [
        "alpha " + "x" * 200
    ]
    assert [message.content for message in beta.messages] == [
        "beta " + "y" * 200
    ]


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
    context = replace(agent.new_context(), messages=(old,))

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
    context = replace(agent.new_context(), messages=(old,))

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
    context = replace(agent.new_context(), messages=(old,))

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
    context = replace(agent.new_context(), messages=(old,))

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
    context = replace(agent.new_context(), messages=(old,))

    with pytest.raises(ContextCompressionUnavailable, match="remains oversized"):
        await agent.invoke(UserMessage(content="y" * 2000), context)


@pytest.mark.asyncio
async def test_compression_budget_resets_at_invocation_boundary() -> None:
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
        compression_count=1,
    )

    _, returned = await agent.invoke(UserMessage(content="current"), context)

    assert returned.compression_count == 1


@pytest.mark.asyncio
async def test_compression_budget_is_explicit_within_an_invocation() -> None:
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
        compression_count=1,
    )

    with pytest.raises(ContextCompressionLimitExceeded):
        await agent.react.model.invoke(UserMessage(content="current"), context)


@pytest.mark.asyncio
async def test_manual_compress_replaces_projection_with_snapshot() -> None:
    model = RecordingModel(AIMessage(content="done"))
    compressor = RecordingModel(AIMessage(content="compact snapshot"))
    agent = build_agent(model=model, compressor=compressor)
    original = (
        UserMessage(content="u1 " + "x" * 500),
        AIMessage(content="a1 " + "y" * 500),
    )
    context = replace(
        agent.new_context(),
        messages=original,
        committed_messages=(UserMessage(content="previous invocation"),),
        compression_count=2,
        projection_revision=5,
    )

    compressed = await agent.compress(context)

    compression_request, compression_context = compressor.calls[0]
    assert compression_request.content == agent.compression_prompt
    assert compression_request.kind == "pygent.context.compression_request"
    assert compression_context.messages == original
    assert compression_context.tools == ()
    assert not model.calls
    assert len(compressed.messages) == 1
    snapshot = compressed.messages[0]
    assert snapshot.content == "compact snapshot"
    assert snapshot.kind == "pygent.context.snapshot"
    assert snapshot.slot == "pygent.context.snapshot"
    assert dict(snapshot.metadata) == {
        "compression_number": 3,
        "source_projection_revision": 5,
    }
    assert compressed.compression_count == 3
    assert compressed.projection_revision == 6
    assert compressed.committed_messages == (
        UserMessage(content="previous invocation"),
    )


@pytest.mark.asyncio
async def test_manual_compress_can_run_repeatedly_without_budget() -> None:
    compressor = RecordingModel(AIMessage(content="compact snapshot"))
    agent = build_agent(
        model=RecordingModel(AIMessage(content="done")),
        compressor=compressor,
        max_compressions=1,
    )
    context = replace(
        agent.new_context(),
        messages=(UserMessage(content="old " + "x" * 200),),
        compression_count=1,
        projection_revision=3,
    )

    first = await agent.compress(context)
    second = await agent.compress(first)

    assert len(compressor.calls) == 2
    assert second.compression_count == 3
    assert second.projection_revision == 5
    assert dict(second.messages[0].metadata) == {
        "compression_number": 3,
        "source_projection_revision": 4,
    }


@pytest.mark.asyncio
async def test_manual_compress_without_history_raises() -> None:
    compressor = RecordingModel(AIMessage(content="summary"))
    agent = build_agent(
        model=RecordingModel(AIMessage(content="done")),
        compressor=compressor,
    )

    with pytest.raises(ContextCompressionUnavailable, match="no projected history"):
        await agent.compress(agent.new_context())

    assert not compressor.calls


@pytest.mark.asyncio
async def test_manual_compress_request_must_fit_its_window() -> None:
    compressor = RecordingModel(AIMessage(content="summary"))
    agent = build_agent(
        model=RecordingModel(AIMessage(content="done")),
        compressor=compressor,
        compression_context_window_tokens=300,
        compression_prompt="p" * 2000,
    )
    context = replace(agent.new_context(), messages=(UserMessage(content="old"),))

    with pytest.raises(
        ContextCompressionUnavailable, match="compression context window"
    ):
        await agent.compress(context)


@pytest.mark.asyncio
async def test_compression_layer_compresses_standalone() -> None:
    compressor = RecordingModel(AIMessage(content="compact snapshot"))
    layer = ContextCompressionLayer(
        model=RecordingModel(AIMessage(content="done")),
        compressor=compressor,
        compression_prompt="Summarize goals, constraints, and unfinished work.",
        context_window_tokens=4096,
        compression_trigger_ratio=0.9,
        compression_context_window_tokens=4096,
        max_compressions=1,
    )
    context = PygentAgentContext(
        system_prompt="You are a careful coding agent.",
        messages=(UserMessage(content="old " + "x" * 200),),
        compression_count=1,
        projection_revision=3,
    )

    compressed = await layer.compress(context)

    assert [message.content for message in compressed.messages] == [
        "compact snapshot"
    ]
    assert compressed.compression_count == 2
    assert compressed.projection_revision == 4


@pytest.mark.asyncio
async def test_manual_compressed_context_feeds_the_next_invocation() -> None:
    answer = AIMessage(content="done")
    model = RecordingModel(answer)
    compressor = RecordingModel(AIMessage(content="compact snapshot"))
    agent = build_agent(model=model, compressor=compressor)
    original = (
        UserMessage(content="u1 " + "x" * 500),
        AIMessage(content="a1 " + "y" * 500),
    )
    context = replace(agent.new_context(), messages=original)

    compressed = await agent.compress(context)
    returned, final_context = await agent.invoke(
        UserMessage(content="continue"), compressed
    )

    assert returned == answer
    assert len(compressor.calls) == 1
    foreground_current, foreground_context = model.calls[0]
    assert foreground_current.content == "continue"
    assert [message.content for message in foreground_context.messages] == [
        "compact snapshot",
    ]
    assert [message.content for message in final_context.messages] == [
        "compact snapshot",
        "continue",
        "done",
    ]


@pytest.mark.asyncio
async def test_long_session_keeps_only_each_invocation_commit_delta() -> None:
    model = LongSessionModel(tool_calls_per_invocation=15)
    compressor = RecordingModel(AIMessage(content="compact snapshot"))
    agent = PygentAgent(
        system_prompt="system",
        compression_prompt="Return a compact continuation.",
        model=model,
        compressor=compressor,
        tools=LongSessionTools(),
        context_window_tokens=1024,
        compression_context_window_tokens=4096,
        max_steps=16,
        max_model_calls=16,
        max_tool_calls=16,
    )
    codec = ContextCodec.dataclass(PygentAgentContext)
    registry = ContextCodecRegistry((codec,))
    context = agent.new_context()
    external_history: list[Message] = []

    for round_index in range(113):
        _, context = await agent.invoke(
            UserMessage(content=f"round {round_index}"),
            context,
        )
        assert len(context.committed_messages) == 32
        external_history.extend(context.committed_messages)
        invocation_to_dict(
            UserMessage(content="next"),
            context,
            registry=registry,
        )

    assert len(external_history) == 3616
    assert len(context.committed_messages) == 32
    assert len(compressor.calls) > 4


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
