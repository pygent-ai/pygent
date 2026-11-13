"""Regression: a queued agent message must survive the tool-reminder effect.

pygent-ai 0.6.0 dropped ``AgentMessage.to_dict`` while ``RuntimeReminderModule``
kept serializing claimed messages with it, so the first realtime exchange
envelope claimed at a tool-result boundary killed the main turn.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from pygent.agent.pygent_v1.graph.pipeline import RuntimeReminderModule
from pygent.agent.pygent_v1.runtime.context import PygentContext
from pygent.agent.pygent_v1.runtime.reminders.service import ReminderService
from pygent.agent.pygent_v1.schema import RunConfig
from pygent.agent.pygent_v1.sessions.collaboration import (
    AgentMessageState,
    SessionCollaborationStore,
)
from pygent import ToolMessage, ToolResult
from pygent.core._module_contracts import _execution_scope
from pygent.core import EffectDisposition, EffectOutcome


class _EffectScope:
    """Managed-execution stand-in: runs the effect once, then replays it."""

    deadline = None
    managed_execution_id = "exec-1"

    def __init__(self) -> None:
        self.effect = None

    async def invoke_module(self, module, message, current_context):
        return await module.forward(message, current_context)

    async def emit_event(self, module, kind, data):
        return None

    @asynccontextmanager
    async def model_permit(self):
        yield

    @asynccontextmanager
    async def tool_permit(self, resource_key=None):
        yield

    async def execute_effect(self, *, spec, request, operation):
        if self.effect is None:
            self.effect = await operation()
            disposition = EffectDisposition.EXECUTED
        else:
            disposition = EffectDisposition.REPLAYED
        return EffectOutcome(
            value=self.effect, disposition=disposition, effect_id="effect-1"
        )


@pytest.mark.asyncio
async def test_claimed_agent_message_is_rendered_and_replayed(
    tmp_path: Path,
) -> None:
    config = RunConfig(
        workspace_root=str(tmp_path / "workspace"),
        pygent_root=str(tmp_path / "pygent_v1"),
    )
    Path(config.workspace_root).mkdir()
    store = SessionCollaborationStore(config.pygent_root)
    reminders = ReminderService(config, message_store=store)
    queued = store.enqueue_message(
        submission_id="sub-1",
        fingerprint="fp-1",
        source_session_id=None,
        source_agent_alias=None,
        target_session_id="session-1",
        content="user asked mid-turn",
    )
    module = RuntimeReminderModule(reminders)
    message = ToolMessage(
        results=(
            ToolResult(call_id="c1", name="bash", status="succeeded", output="ok"),
        )
    )
    context = PygentContext(session_id="session-1", turn_id="turn-1")

    scope = _EffectScope()
    token = _execution_scope.set(scope)
    try:
        first, _ = await module.forward(message, context)
        replayed, _ = await module.forward(message, context)
    finally:
        _execution_scope.reset(token)
        await reminders.close()

    assert "user asked mid-turn" in first.content
    assert replayed.content == first.content
    assert store.get_message(queued.message_id).state is AgentMessageState.DELIVERED
    assert store.get_message(queued.message_id).delivered_execution_id == "exec-1"
