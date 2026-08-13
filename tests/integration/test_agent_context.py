from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from pygent import (
    Agent,
    AIMessage,
    Context,
    ContextCodec,
    Message,
    Module,
    UserMessage,
)
from pygent.runtime import ExecutionAdmissionError, LocalRuntime, SQLiteHistoryStore
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


class MyAgent(Agent[UserMessage, AIMessage]):
    context_type = AgentContext

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


def test_runtime_auto_registers_declared_agent_context_at_bind():
    runtime = LocalRuntime()

    bound = runtime.bind(MyAgent())

    assert runtime.context_codec_registry.for_value(_context()).identity == CODEC.identity
    assert CODEC.identity in bound.plan.context_codecs


def test_registry_rejects_duplicate_codec_registration():
    with pytest.raises(ContextCodecError, match="duplicate"):
        ContextCodecRegistry((CODEC, CODEC))


@pytest.mark.asyncio
async def test_documented_agent_context_pattern_runs_direct_and_managed():
    direct_message, direct_context = await MyAgent().invoke(UserMessage(content="q"), _context())
    assert direct_message.content == "q:weather"
    assert isinstance(direct_context, AgentContext)
    assert direct_context.file_state.paths == ("README.md",)

    runtime = LocalRuntime()
    managed_message, managed_context = await runtime.bind(MyAgent()).invoke(
        UserMessage(content="q"), _context()
    )
    assert managed_message == direct_message
    assert managed_context == direct_context


@pytest.mark.asyncio
async def test_direct_agent_rejects_context_outside_declared_type():
    with pytest.raises(TypeError, match="requires Context type AgentContext"):
        await MyAgent().invoke(UserMessage(content="q"), Context())


@pytest.mark.asyncio
async def test_agent_context_survives_durable_result_reattach(tmp_path):
    async with SQLiteHistoryStore(tmp_path / "agent-context.sqlite3") as history:
        runtime = LocalRuntime(history=history)
        handle = await runtime.bind(MyAgent()).start(UserMessage(content="q"), _context())
        expected = await handle.result()
        attached = await runtime.get_execution_handle(handle.execution_id)
        assert await attached.result() == expected


@dataclass(frozen=True, slots=True)
class OtherContext(Context):
    context_schema = "tests.other-context"
    context_schema_version = 1
    value: str = "other"


class OtherAgent(Agent[UserMessage, AIMessage]):
    context_type = OtherContext

    async def forward(self, message: UserMessage, context: OtherContext):
        return AIMessage(content=message.content), context


@pytest.mark.asyncio
async def test_binding_rejects_another_registered_agent_context_before_forward():
    runtime = LocalRuntime()
    first = runtime.bind(MyAgent())
    runtime.bind(OtherAgent())

    with pytest.raises(ExecutionAdmissionError, match="Context codec"):
        await first.invoke(UserMessage(content="q"), OtherContext())


def test_binding_order_does_not_change_agent_context_plan_identity():
    first_runtime = LocalRuntime()
    first_plan = first_runtime.bind(MyAgent()).plan
    first_runtime.bind(OtherAgent())

    second_runtime = LocalRuntime()
    second_runtime.bind(OtherAgent())
    second_plan = second_runtime.bind(MyAgent()).plan

    assert first_plan == second_plan


@pytest.mark.asyncio
async def test_later_agent_registration_does_not_drift_existing_module_plan():
    class PlainModule(Module[UserMessage, AIMessage]):
        async def forward(self, message, context):
            return AIMessage(content=message.content), context

    runtime = LocalRuntime(context_codecs=(CODEC,))
    bound = runtime.bind(PlainModule())
    runtime.bind(OtherAgent())

    output, context = await bound.invoke(UserMessage(content="q"), _context())

    assert output.content == "q"
    assert type(context) is AgentContext


def test_invalid_declared_context_type_fails_at_bind():
    class InvalidAgent(Agent[UserMessage, AIMessage]):
        context_type = str  # type: ignore[assignment]

        async def forward(self, message, context):
            return AIMessage(content=message.content), context

    with pytest.raises(ContextCodecError, match="Context subclass"):
        LocalRuntime().bind(InvalidAgent())


def test_auto_registration_rejects_schema_conflict():
    @dataclass(frozen=True, slots=True)
    class ConflictingContext(Context):
        context_schema = AgentContext.context_schema
        context_schema_version = AgentContext.context_schema_version
        different: str = "different"

    class ConflictingAgent(Agent[UserMessage, AIMessage]):
        context_type = ConflictingContext

        async def forward(self, message, context):
            return AIMessage(content=message.content), context

    runtime = LocalRuntime()
    runtime.bind(MyAgent())
    with pytest.raises(ContextCodecError, match="conflicting"):
        runtime.bind(ConflictingAgent())


def test_failed_bind_does_not_leave_auto_registered_codec():
    class UnsupportedAgent(MyAgent):
        execution_requirements = MyAgent.execution_requirements.__class__(
            required_capabilities=("tests.unsupported",)
        )

    runtime = LocalRuntime()
    with pytest.raises(ExecutionAdmissionError, match="required capabilities"):
        runtime.bind(UnsupportedAgent())
    with pytest.raises(ContextCodecError, match="unregistered"):
        runtime.context_codec_registry.for_value(_context())


def test_explicit_codec_registration_remains_compatible():
    runtime = LocalRuntime(context_codecs=(CODEC,))

    bound = runtime.bind(MyAgent())

    assert CODEC.identity in bound.plan.context_codecs


def test_agent_context_rejects_mutable_or_live_values():
    with pytest.raises(TypeError, match="unsupported portable Context"):

        @dataclass(frozen=True, slots=True)
        class InvalidContext(Context):
            context_schema = "tests.invalid-context"
            context_schema_version = 1
            state: object = object()

        InvalidContext()
