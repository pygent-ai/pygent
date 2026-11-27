"""Host extension points on PygentRuntimeService.

Covers the constructor's ``speak_channels`` / ``external_tools`` slots, the
subclass-preserving child-config copy, and the public fork-profile export.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence
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
    RESERVED_TOOL_NAMES,
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


@pytest.mark.parametrize(
    "name", ["bash", "diff", "speak", "pygent_persist_file_effects", "schedule_task"]
)
def test_external_tools_cannot_shadow_reserved_names(tmp_path: Path, name: str) -> None:
    assert name in RESERVED_TOOL_NAMES
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
        self.messages: list[tuple[str, tuple[Path, ...]]] = []

    def deliver(self, text: str, attachments: Sequence[Path]) -> bool:
        self.messages.append((text, tuple(attachments)))
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
        assert channel.messages == [("hello host", ())]
    finally:
        asyncio.run(_close(service))


def test_speak_channels_receive_resolved_attachment_paths(tmp_path: Path) -> None:
    config = _config(tmp_path)
    report = Path(config.workspace_root) / "out" / "report.txt"
    report.parent.mkdir()
    report.write_text("done", encoding="utf-8")
    channel = _RecordingChannel()
    service = PygentRuntimeService(config, speak_channels=(channel,))
    try:
        executor = service._speak_executor
        events: list[tuple[str, Any]] = []

        async def emit(kind: str, payload: Any) -> None:
            events.append((kind, payload))

        async def run() -> None:
            await executor.execute(
                SPEAK_TOOL_SPEC,
                SimpleNamespace(
                    arguments={"text": "finished", "attachments": ["out/report.txt"]}
                ),
                ToolExecutionContext(emit=emit),
            )
            await asyncio.gather(*executor._delivery_tasks)

        asyncio.run(run())
        # The stream event keeps workspace-relative names; the channel gets
        # resolved paths it can open, and the text stays untouched.
        assert events == [
            (
                "pygent.v1.realtime.speak",
                {"text": "finished", "attachments": ["out/report.txt"]},
            )
        ]
        assert channel.messages == [("finished", (report.resolve(),))]
    finally:
        asyncio.run(_close(service))


class _FailingChannel:
    def deliver(self, text: str, attachments: Sequence[Path]) -> bool:
        raise ConnectionError("webhook down")


class _DecliningChannel:
    def deliver(self, text: str, attachments: Sequence[Path]) -> bool:
        return False


def _speak(service: PygentRuntimeService, text: str) -> None:
    executor = service.executor_registry.resolve(
        SPEAK_TOOL_SPEC.tool_id, SPEAK_TOOL_SPEC.version
    )
    assert isinstance(executor, _SpeakExecutor)

    async def emit(kind: str, payload: Any) -> None:
        del kind, payload

    async def run() -> None:
        result = await executor.execute(
            SPEAK_TOOL_SPEC,
            SimpleNamespace(arguments={"text": text}),
            ToolExecutionContext(emit=emit),
        )
        assert result == "delivered"
        await asyncio.gather(*executor._delivery_tasks, return_exceptions=True)
        # Let the done callbacks run.
        await asyncio.sleep(0)

    asyncio.run(run())


def test_speak_channel_failures_are_recorded_not_raised(tmp_path: Path) -> None:
    service = PygentRuntimeService(
        _config(tmp_path), speak_channels=(_FailingChannel(), _DecliningChannel())
    )
    try:
        with pytest.warns(RuntimeWarning) as captured:
            _speak(service, "hello")
        messages = [str(item.message) for item in captured]
        assert any("_FailingChannel failed" in m and "webhook down" in m for m in messages)
        assert any("_DecliningChannel declined" in m for m in messages)
        assert any("_FailingChannel failed" in m for m in service.warnings)
        assert any("_DecliningChannel declined" in m for m in service.warnings)
    finally:
        asyncio.run(_close(service))


def test_close_drains_pending_speak_deliveries(tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()

    class _SlowChannel:
        def deliver(self, text: str, attachments: Sequence[Path]) -> bool:
            started.set()
            release.wait(5)
            return True

    service = PygentRuntimeService(_config(tmp_path), speak_channels=(_SlowChannel(),))
    executor = service._speak_executor
    executor.DRAIN_TIMEOUT_SECONDS = 0.2

    async def run() -> None:
        async def emit(kind: str, payload: Any) -> None:
            del kind, payload

        await executor.execute(
            SPEAK_TOOL_SPEC,
            SimpleNamespace(arguments={"text": "slow"}),
            ToolExecutionContext(emit=emit),
        )
        # Yield so the delivery task starts its worker thread.
        assert await asyncio.to_thread(started.wait, 2)
        with pytest.warns(RuntimeWarning, match="still pending at runtime close"):
            await service.close()
        release.set()
        await asyncio.gather(*executor._delivery_tasks, return_exceptions=True)

    asyncio.run(run())
    assert any("still pending" in m for m in service.warnings)


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


async def test_speak_receipts_retry_in_a_new_runtime(tmp_path: Path) -> None:
    class Channel:
        recipient_id = "destination"

        def __init__(self, success: bool) -> None:
            self.success = success
            self.reply_ids: list[str] = []

        def deliver(self, text: str, attachments: Sequence[Path]) -> bool:
            raise AssertionError("the delivery id should reach the channel")

        def deliver_with_id(
            self, reply_id: str, text: str, attachments: Sequence[Path]
        ) -> bool:
            assert text == "reply"
            assert not attachments
            self.reply_ids.append(reply_id)
            return self.success

    config = _config(tmp_path)
    failing = Channel(False)
    first = PygentRuntimeService(config, speak_channels=(failing,))
    executor = first._speak_executor
    await executor.publish("reply-1", "reply", ())
    with pytest.warns(RuntimeWarning, match="declined"):
        await asyncio.gather(*executor._delivery_tasks)
    store = executor._store
    assert store is not None
    assert store.get("speak:destination", "reply-1")["receipt"] is None
    await first.close()

    with store.transaction() as db:
        db.execute("UPDATE deliveries SET due=0 WHERE lane='speak:destination'")

    succeeding = Channel(True)
    second = PygentRuntimeService(config, speak_channels=(succeeding,))
    try:
        # Initialization starts the replay worker without another speak call.
        await second.initialize()

        async def wait_for_receipt() -> None:
            while store.get("speak:destination", "reply-1")["receipt"] is None:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(wait_for_receipt(), 5)
        await second._speak_executor.publish("reply-1", "reply", ())
        await asyncio.gather(*second._speak_executor._delivery_tasks)
        assert failing.reply_ids == succeeding.reply_ids == ["reply-1"]
    finally:
        await second.close()
