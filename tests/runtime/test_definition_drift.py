"""Bound execution must not diverge from its immutable ExecutionPlan."""

from __future__ import annotations

import pytest

from pygent import AIMessage, Context, Module, UserMessage, freeze_json_object
from pygent.runtime import ExecutionAdmissionError, LocalRuntime


class ConfiguredReply(Module[UserMessage, AIMessage]):
    def __init__(self, prefix: str) -> None:
        super().__init__()
        self.prefix = prefix

    async def forward(
        self, message: UserMessage, context: Context
    ) -> tuple[AIMessage, Context]:
        return AIMessage(content=f"{self.prefix}:{message.content}"), context


@pytest.mark.asyncio
async def test_bind_recursively_freezes_definition_before_plan_compilation() -> None:
    runtime = LocalRuntime()
    module = ConfiguredReply("before")
    bound = runtime.bind(module)
    original_hash = bound.plan.graph_hash

    with pytest.raises(RuntimeError, match="definition is frozen"):
        module.prefix = "after"

    output, _ = await bound.invoke(UserMessage(content="request"), Context())
    assert output.content == "before:request"
    assert bound.plan.graph_hash == original_hash
    await runtime.close()


@pytest.mark.asyncio
async def test_runtime_start_keeps_plan_drift_guard_for_unsafe_bypass() -> None:
    runtime = LocalRuntime()
    module = ConfiguredReply("before")
    bound = runtime.bind(module)

    # The public attribute boundary rejects this mutation.  Simulate hostile or
    # legacy code deliberately bypassing __setattr__ to verify the Runtime still
    # refuses to execute semantics that differ from the bound plan identity.
    object.__setattr__(module, "prefix", "after")

    with pytest.raises(ExecutionAdmissionError, match="changed after binding"):
        await bound.invoke(UserMessage(content="request"), Context())
    await runtime.close()


@pytest.mark.asyncio
async def test_hook_snapshot_drift_is_detected_even_through_private_storage() -> None:
    class HookedReply(Module[UserMessage, AIMessage]):
        def __init__(self) -> None:
            super().__init__()
            self._settings = freeze_json_object({"prefix": "before"})

        def execution_plan_config(self):
            return {"settings": dict(self._settings)}

        async def forward(self, message, context):
            return AIMessage(
                content=f"{self._settings['prefix']}:{message.content}"
            ), context

    runtime = LocalRuntime()
    module = HookedReply()
    bound = runtime.bind(module)

    object.__setattr__(
        module,
        "_settings",
        freeze_json_object({"prefix": "after"}),
    )

    with pytest.raises(ExecutionAdmissionError, match="changed after binding"):
        await bound.invoke(UserMessage(content="request"), Context())
    await runtime.close()


@pytest.mark.asyncio
async def test_runtime_rejects_user_class_config_drift_after_binding() -> None:
    class ClassConfiguredReply(Module[UserMessage, AIMessage]):
        prefix = "before"

        async def forward(self, message, context):
            return AIMessage(content=f"{self.prefix}:{message.content}"), context

    runtime = LocalRuntime()
    bound = runtime.bind(ClassConfiguredReply())
    original_hash = bound.plan.graph_hash

    ClassConfiguredReply.prefix = "after"
    with pytest.raises(ExecutionAdmissionError, match="changed after binding"):
        await bound.invoke(UserMessage(content="request"), Context())
    assert bound.plan.graph_hash == original_hash
    await runtime.close()
