"""Executable checks for the foundational Module and Context contract."""

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from pygent import (
    AIMessage,
    Context,
    Message,
    Module,
    ToolCall,
    ToolDefinition,
    ToolMessage,
    ToolResult,
    UserMessage,
)
from pygent.runtime import LocalRuntime


def test_first_principles_freeze_the_module_state_transition():
    principles = (
        Path(__file__).resolve().parents[2] / "docs" / "FEATURES.md"
    ).read_text(encoding="utf-8")

    assert "(message, context) -> (message, context)" in principles
    assert "Context 是显式传递的当前有效上下文快照" in principles
    assert "**可恢复执行只属于托管 Runtime**" in principles
    assert "不要求普通业务代码显式声明恢复点" in principles
    assert "不等同于序列化任意 Python coroutine" in principles


def test_module_document_defines_custom_events_without_a_stream_forward():
    contract = (
        Path(__file__).resolve().parents[2] / "docs" / "module" / "SDK.md"
    ).read_text(encoding="utf-8")

    assert "await self.emit(" in contract
    assert "stream_forward" not in contract
    assert "同一个 `forward()`" in contract


def test_external_wait_documentation_preserves_the_short_wait_risk_boundary():
    runtime_contract = (
        Path(__file__).resolve().parents[2] / "docs" / "runtime" / "README.md"
    ).read_text(encoding="utf-8")
    agent_sdk = (
        Path(__file__).resolve().parents[2] / "docs" / "agent" / "SDK.md"
    ).read_text(encoding="utf-8")

    assert "`wait_external()` 是 Runtime 的通用能力" in runtime_contract
    assert "WAITING_EXTERNAL，释放 runnable lease" in runtime_contract
    assert "持续保留 live execution、Task、调用栈、局部变量和内存" in agent_sdk
    assert "秒级或分钟级短等待" in agent_sdk
    assert "`LocalRuntime` 已实现 `wait_external()`" in runtime_contract


def test_transparent_recovery_is_a_runtime_draft_not_a_first_principle():
    docs_root = Path(__file__).resolve().parents[2] / "docs"
    replay = (docs_root / "runtime" / "REPLAY.md").read_text(encoding="utf-8")
    runtime_principles = (docs_root / "runtime" / "FEATURES.md").read_text(
        encoding="utf-8"
    )

    assert "用户不声明 checkpoint、step 或源码恢复标签" in replay
    assert "不序列化 CPython coroutine" in replay
    assert "可调整策略" in replay
    assert "透明恢复与确定性重放" not in runtime_principles


def test_emit_is_inherited_runtime_capability_not_user_business_entrypoint():
    assert "emit" not in Echo.__dict__
    assert inspect.iscoroutinefunction(Module.emit)


def test_context_plus_message_returns_a_new_history_value():
    original = Context(system_prompt="system")
    message = UserMessage(content="hello")

    updated = original + message

    assert updated is not original
    assert original.messages == ()
    assert updated.messages == (message,)


def test_context_plus_equals_rebinds_without_mutating_the_old_value():
    original = Context()
    context = original
    message = UserMessage(content="hello")

    context += message

    assert context is not original
    assert original.messages == ()
    assert context.messages == (message,)


def test_context_slot_replaces_only_the_previous_effective_slot_value():
    first = UserMessage(content="first")
    old_retrieval = UserMessage(content="old retrieval", slot="retrieval/current")
    new_retrieval = UserMessage(content="new retrieval", slot="retrieval/current")

    context = Context() + first + old_retrieval + new_retrieval

    assert context.messages == (first, new_retrieval)


def test_context_rejects_non_message_addition_and_field_mutation():
    context = Context()

    with pytest.raises(TypeError):
        _ = context + "hello"
    with pytest.raises(FrozenInstanceError):
        context.system_prompt = "changed"  # type: ignore[misc]


def test_message_and_context_validate_typed_elements():
    call = ToolCall(call_id="call-1", name="search", arguments={"q": "x"})
    result = ToolResult(
        call_id="call-1", name="search", status="succeeded", output={"ok": True}
    )
    definition = ToolDefinition(
        name="search", description="Search", parameters={"type": "object"}
    )

    assert AIMessage(tool_calls=[call]).tool_calls == (call,)
    assert ToolMessage(results=[result]).results == (result,)
    assert Context(messages=[UserMessage(content="hi")], tools=[definition]).tools == (
        definition,
    )

    with pytest.raises(TypeError, match="tool_calls"):
        AIMessage(tool_calls=[object()])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="results"):
        ToolMessage(results=[object()])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="messages"):
        Context(messages=["not a message"])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="tools"):
        Context(tools=["search"])  # type: ignore[list-item]


def test_context_and_message_validate_scalar_field_types():
    with pytest.raises(TypeError, match="system_prompt"):
        Context(system_prompt=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="slot"):
        UserMessage(slot=1)  # type: ignore[arg-type]


def test_portable_domain_message_freezes_data_and_requires_stable_kind():
    source = {"operation": {"id": "op-1"}, "approvers": ["alice"]}
    message = Message(kind="approval.requested", data=source)

    source["approvers"].append("mallory")

    assert message.data["approvers"] == ("alice",)
    with pytest.raises(ValueError, match="requires a non-empty kind"):
        Message(data={"operation": "op-1"})
    with pytest.raises(TypeError):
        Message(kind="approval.requested", data={"handler": object()})


def test_message_and_context_reject_unsafe_public_subclasses():
    with pytest.raises(TypeError, match="portable domain messages"):

        class UnsafeMessage(Message):
            pass

    with pytest.raises(TypeError, match="Context cannot be subclassed"):

        class UnsafeContext(Context):
            pass

    with pytest.raises(TypeError, match="portable domain messages"):

        class ForgedFrameworkMessage(Message):
            __module__ = "pygent.user_domain"


def test_context_defensively_copies_message_and_tool_collections():
    messages = [UserMessage(content="first")]
    tools = [
        ToolDefinition(
            name="search", description="Search", parameters={"type": "object"}
        )
    ]
    context = Context(messages=messages, tools=tools)

    messages.append(UserMessage(content="second"))
    tools.clear()

    assert len(context.messages) == 1
    assert len(context.tools) == 1


class Echo(Module[UserMessage, AIMessage]):
    async def forward(
        self, message: UserMessage, context: Context
    ) -> tuple[AIMessage, Context]:
        return AIMessage(content=message.content), context + message


@pytest.mark.asyncio
async def test_user_module_forward_uses_message_context_order():
    message = UserMessage(content="hello")
    context = Context()

    output, next_context = await Echo().forward(message, context)

    assert output == AIMessage(content="hello")
    assert next_context.messages == (message,)


@pytest.mark.asyncio
async def test_domain_messages_compose_approval_handoff_and_termination() -> None:
    class Approval(Module[Message, Message]):
        async def forward(self, message: Message, context: Context):
            assert message.kind == "approval.requested"
            decision = message.data["decision"]
            kind = (
                "approval.approved"
                if decision == "approve"
                else "workflow.terminated"
            )
            output = Message(
                kind=kind,
                data={"operation_id": message.data["operation_id"]},
            )
            return output, context + message + output

    class Handoff(Module[Message, Message]):
        async def forward(self, message: Message, context: Context):
            assert message.kind == "handoff.requested"
            output = Message(kind="handoff.completed", data=message.data)
            return output, context + message + output

    class Workflow(Module[Message, Message]):
        def __init__(self) -> None:
            super().__init__()
            self.approval = Approval()
            self.handoff = Handoff()

        async def forward(self, message: Message, context: Context):
            decision, context = await self.approval(message, context)
            if decision.kind == "workflow.terminated":
                return decision, context
            return await self.handoff(
                Message(kind="handoff.requested", data=decision.data),
                context,
            )

    workflow = Workflow()
    approved = Message(
        kind="approval.requested",
        data={"operation_id": "op-1", "decision": "approve"},
    )
    terminated = Message(
        kind="approval.requested",
        data={"operation_id": "op-2", "decision": "terminate"},
    )

    direct, direct_context = await workflow.invoke(approved, Context())
    assert direct.kind == "handoff.completed"
    assert [item.kind for item in direct_context.messages] == [
        "approval.requested",
        "approval.approved",
        "handoff.requested",
        "handoff.completed",
    ]

    runtime = LocalRuntime()
    managed, managed_context = await runtime.bind(workflow).invoke(
        terminated, Context()
    )
    await runtime.close()

    assert managed.kind == "workflow.terminated"
    assert [item.kind for item in managed_context.messages] == [
        "approval.requested",
        "workflow.terminated",
    ]
