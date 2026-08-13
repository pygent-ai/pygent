from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from pygent import (
    AIMessage,
    Context,
    ContextCodec,
    Message,
    Module,
    UserMessage,
)
from pygent.runtime import LocalRuntime, SQLiteHistoryStore
from pygent.runtime.codec import (
    WireCodecError,
    context_from_dict,
    context_to_dict,
)
from pygent.runtime.context_codec import ContextCodecError, ContextCodecRegistry


@dataclass(frozen=True, slots=True)
class ToolState:
    summaries: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FileState:
    paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AgentContext(Context):
    context_schema = "tests.agent-context"
    context_schema_version = 1

    tool_state: ToolState = ToolState()
    file_state: FileState = FileState()
    full_history: tuple[Message, ...] = ()

    def __add__(self, value: object):
        updated = Context.__add__(self, value)
        if updated is NotImplemented:
            return NotImplemented
        assert isinstance(value, Message)
        return replace(updated, full_history=updated.full_history + (value,))


CODEC = ContextCodec.dataclass(AgentContext)
REGISTRY = ContextCodecRegistry((CODEC,))


class StateModule(Module[UserMessage, Message]):
    async def forward(self, message: UserMessage, context: AgentContext):
        prompt = ",".join(context.tool_state.summaries)
        return Message(kind="tool.prompt.computed", content=prompt), context


class MyAgent(Module[UserMessage, AIMessage]):
    def __init__(self) -> None:
        super().__init__()
        self.state_module = StateModule()

    async def forward(self, message: UserMessage, context: AgentContext):
        tool_info, context = await self.state_module(message, context)
        return AIMessage(content=f"{message.content}:{tool_info.content}"), context + message


def _context() -> AgentContext:
    return AgentContext(
        tool_state=ToolState(("weather",)),
        file_state=FileState(("README.md",)),
    )


def test_agent_context_add_and_codec_preserve_concrete_type_and_fields():
    original = _context()
    updated = original
    updated += UserMessage(content="hello", slot="current")

    assert type(updated) is AgentContext
    assert updated is not original
    assert original.messages == ()
    assert updated.tool_state is original.tool_state
    assert updated.full_history == updated.messages

    encoded = context_to_dict(updated, registry=REGISTRY)
    assert set(encoded) == {"schema", "version", "codec", "codec_digest", "data"}
    assert context_from_dict(encoded, registry=REGISTRY) == updated


def test_old_context_wire_shape_is_rejected():
    with pytest.raises(WireCodecError):
        context_from_dict(
            {"system_prompt": "", "messages": [], "tools": [], "metadata": {}},
            registry=REGISTRY,
        )


def test_codec_digest_is_stable_and_tampering_fails_closed():
    assert ContextCodec.dataclass(AgentContext).codec_digest == CODEC.codec_digest
    encoded = context_to_dict(_context(), registry=REGISTRY)
    encoded["codec_digest"] = "sha256:" + "0" * 64
    with pytest.raises(WireCodecError):
        context_from_dict(encoded, registry=REGISTRY)


def test_runtime_rejects_unregistered_agent_context_before_forward():
    with pytest.raises((ContextCodecError, WireCodecError)):
        LocalRuntime().context_codec_registry.for_value(_context())


def test_registry_rejects_duplicate_codec_registration():
    with pytest.raises(ContextCodecError, match="duplicate"):
        ContextCodecRegistry((CODEC, CODEC))


@pytest.mark.asyncio
async def test_documented_agent_context_pattern_runs_direct_and_managed():
    direct_message, direct_context = await MyAgent().invoke(UserMessage(content="q"), _context())
    assert direct_message.content == "q:weather"
    assert isinstance(direct_context, AgentContext)
    assert direct_context.file_state.paths == ("README.md",)

    runtime = LocalRuntime(context_codecs=(CODEC,))
    managed_message, managed_context = await runtime.bind(MyAgent()).invoke(
        UserMessage(content="q"), _context()
    )
    assert managed_message == direct_message
    assert managed_context == direct_context


@pytest.mark.asyncio
async def test_agent_context_survives_durable_result_reattach(tmp_path):
    async with SQLiteHistoryStore(tmp_path / "agent-context.sqlite3") as history:
        runtime = LocalRuntime(history=history, context_codecs=(CODEC,))
        handle = await runtime.bind(MyAgent()).start(UserMessage(content="q"), _context())
        expected = await handle.result()
        attached = await runtime.get_execution_handle(handle.execution_id)
        assert await attached.result() == expected


def test_agent_context_rejects_mutable_or_live_values():
    with pytest.raises(TypeError, match="unsupported portable Context"):

        @dataclass(frozen=True, slots=True)
        class InvalidContext(Context):
            context_schema = "tests.invalid-context"
            context_schema_version = 1
            state: object = object()

        InvalidContext()
