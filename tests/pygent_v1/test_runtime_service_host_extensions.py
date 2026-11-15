"""Host extension points on PygentRuntimeService.

Covers the constructor's ``speak_channels`` / ``external_tools`` slots, the
subclass-preserving child-config copy, and the public fork-profile export.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from pygent.agent.pygent_v1.runtime import (
    ForkModelProfile,
    SpeakChannel,
    fork_model_profiles,
    fork_profile_name,
)
from pygent.agent.pygent_v1.runtime.realtime.speak import SPEAK_TOOL_SPEC
from pygent.agent.pygent_v1.runtime.service import (
    PygentRuntimeService,
    _SpeakExecutor,
)
from pygent.agent.pygent_v1.schema import RunConfig
from pygent.tool import (
    IdempotencyPolicy,
    SandboxExecutorSupport,
    ToolDefinition,
    ToolSideEffect,
    ToolSpec,
)
from pygent.tool.executors import ToolExecutionContext


def _config(tmp_path: Path, **overrides: Any) -> RunConfig:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return RunConfig(
        workspace_root=str(workspace),
        user_pygent_root=str(tmp_path / "user-root"),
        pygent_root=str(tmp_path / "project-root"),
        **overrides,
    )


class _HostExecutor:
    def __init__(self) -> None:
        self.sandbox_support = SandboxExecutorSupport(
            profiles=("agent",),
            durable_reconnect=True,
            deployment_fingerprint="test:host:v1",
        )

    async def execute(
        self, spec: ToolSpec, call: Any, context: ToolExecutionContext
    ) -> object:
        return "ok"


def _host_tool(name: str) -> tuple[ToolSpec, _HostExecutor]:
    spec = ToolSpec(
        tool_id=f"host.{name}",
        version="1",
        definition=ToolDefinition(
            name=name,
            description=f"host tool {name}",
            parameters={"type": "object", "properties": {}},
        ),
        side_effect=ToolSideEffect.EXTERNAL,
        idempotency=IdempotencyPolicy.REQUIRES_KEY,
        resource_key="host",
        sandbox_profile="agent",
        timeout=5,
    )
    return spec, _HostExecutor()


async def _close(service: PygentRuntimeService) -> None:
    await service.close()


# --- item 7: constructor slots --------------------------------------------


def test_external_tools_join_model_visible_set(tmp_path: Path) -> None:
    spec, executor = _host_tool("agent_send")
    service = PygentRuntimeService(_config(tmp_path), external_tools=[(spec, executor)])
    try:
        names = [item.definition.name for item in service.external_tools]
        assert "agent_send" in names
        assert names.index("agent_send") > names.index("schedule_task")
        assert service.executor_registry.resolve(spec.tool_id, spec.version) is executor
    finally:
        asyncio.run(_close(service))


@pytest.mark.parametrize("name", ["bash", "schedule_task"])
def test_external_tools_cannot_shadow_builtin_names(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match=f"duplicate host model-visible tool name: {name}"):
        PygentRuntimeService(_config(tmp_path), external_tools=[_host_tool(name)])


def test_default_service_registers_no_host_tools(tmp_path: Path) -> None:
    service = PygentRuntimeService(_config(tmp_path))
    try:
        names = {item.definition.name for item in service.external_tools}
        assert "schedule_task" in names
        assert not any(item.tool_id.startswith("host.") for item in service.external_tools)
    finally:
        asyncio.run(_close(service))


class _RecordingChannel:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def deliver(self, text: str) -> bool:
        self.texts.append(text)
        return True


def test_speak_channels_receive_delivered_text(tmp_path: Path) -> None:
    channel = _RecordingChannel()
    # Structural check: the host type satisfies the exported protocol.
    typed_channel: SpeakChannel = channel
    service = PygentRuntimeService(_config(tmp_path), speak_channels=(typed_channel,))
    try:
        executor = service.executor_registry.resolve(
            SPEAK_TOOL_SPEC.tool_id, SPEAK_TOOL_SPEC.version
        )
        assert isinstance(executor, _SpeakExecutor)
        assert executor.channels == (channel,)
        events: list[tuple[str, Any]] = []

        async def emit(kind: str, payload: Any) -> None:
            events.append((kind, payload))

        async def run() -> str:
            result = await executor.execute(
                SPEAK_TOOL_SPEC,
                SimpleNamespace(arguments={"text": "hello host"}),
                ToolExecutionContext(emit=emit),
            )
            await asyncio.gather(*executor._delivery_tasks)
            return str(result)

        assert asyncio.run(run()) == "delivered"
        assert events == [("pygent.v1.realtime.speak", {"text": "hello host", "attachments": []})]
        assert channel.texts == ["hello host"]
    finally:
        asyncio.run(_close(service))


# --- item 2: child config keeps the host's RunConfig type --------------------


@dataclass(slots=True)
class HostRunConfig(RunConfig):
    allowed_agents: tuple[str, ...] = field(default_factory=tuple)


def test_child_agent_config_preserves_host_subclass(tmp_path: Path) -> None:
    config = HostRunConfig(
        workspace_root=str(tmp_path),
        user_pygent_root=str(tmp_path / "u"),
        allowed_agents=("reviewer",),
    )
    service = object.__new__(PygentRuntimeService)
    service.config = config
    service._run_config_loader = None

    child = service._child_agent_config(config.agent_alias)

    assert type(child) is HostRunConfig
    assert child is not config
    assert child.allowed_agents == ("reviewer",)
    assert child.workspace_root == config.workspace_root


# --- item 5: public fork-profile surface -----------------------------------


def test_fork_profile_helpers_are_exported_from_runtime_package() -> None:
    from pygent.agent.pygent_v1.runtime.realtime import fork_reasoning

    assert fork_model_profiles is fork_reasoning.fork_model_profiles
    assert ForkModelProfile is fork_reasoning.ForkModelProfile
    assert fork_profile_name("gpt") == "fork:gpt"
