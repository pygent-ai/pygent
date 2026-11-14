"""End-to-end tests for the realtime continuation turn (D-RT1/D-RT11).

Pins the delivery contract for user input arriving around the end of a turn
chain: lane-queued turns and fork exchange envelopes must reach the main
agent's context through the coordinator's follow-up rounds — never strand as
pending envelopes, and never be decided against a stale draft pointer.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pygent.agent.pygent_v1.orchestration.models import TurnCommand
from pygent.agent.pygent_v1.orchestration.runtime_keys import RuntimeGenerationKey
from pygent.agent.pygent_v1.orchestration.runtime_lease import (
    RuntimeLease,
    SessionRuntime,
)
from pygent.agent.pygent_v1.orchestration.session_execution import (
    SessionExecutionCoordinator,
)
from pygent.agent.pygent_v1.runtime.realtime.continuation import (
    CONTINUATION_MESSAGE_KIND,
    render_pending_input_message,
)
from pygent.agent.pygent_v1.runtime.realtime.envelope import ExchangeEnvelope
from pygent.agent.pygent_v1.runtime.realtime.turn_finalizer import (
    FINALIZER_MESSAGE_KIND,
)
from pygent.agent.pygent_v1.schema import AgentSession, CaseRunRef, RunConfig
from pygent.agent.react_projection_operations import (
    REACT_PROJECTION_OPERATION_KIND,
    StandaloneUserMessage,
    decode_react_projection_operation,
)
from pygent.core import ExecutionInputDelivery


def _run_config(tmp_path: Path) -> RunConfig:
    return RunConfig(
        workspace_root=str(tmp_path / "workspace"),
        pygent_root=str(tmp_path / "pygent_v1"),
        realtime_interaction={"enabled": True},  # type: ignore[arg-type]
    )


def _session(mode: str = "pygent.v1-says") -> AgentSession:
    now = datetime(2026, 1, 1, tzinfo=UTC).isoformat()
    return AgentSession(
        session_id="session-1",
        workspace_root=".",
        session_dir=".",
        created_at=now,
        updated_at=now,
        metadata={"mode": mode},
    )


class _Script:
    """One fake turn execution: a history tail and an optional late envelope.

    ``deliver_after`` models a fork reply that lands while the turn is
    finishing — after the execution sealed its inbox, before the drive's tail
    checks run. ``deliver_during`` models one landing mid-turn, while the
    execution still accepts inputs.
    """

    def __init__(
        self,
        *,
        history: list[dict[str, Any]] | None = None,
        gate: asyncio.Event | None = None,
        deliver_after: list[str] | None = None,
        deliver_during: list[str] | None = None,
    ) -> None:
        self.history = history or []
        self.gate = gate
        self.deliver_after = deliver_after or []
        self.deliver_during = deliver_during or []
        self.deliveries: list[str] = []

    async def run(
        self,
        coordinator: SessionExecutionCoordinator,
        manager: Any,
        handle: _FakeHandle,
        session: AgentSession,
        session_id: str,
    ) -> None:
        for content in self.deliver_during:
            self.deliveries.append(
                await coordinator.deliver_exchange(manager, _envelope(session_id, content))
            )
        if self.gate is not None:
            await self.gate.wait()
        session.history.extend(self.history)
        handle.sealed = True
        for content in self.deliver_after:
            self.deliveries.append(
                await coordinator.deliver_exchange(manager, _envelope(session_id, content))
            )


def _envelope(session_id: str, content: str) -> ExchangeEnvelope:
    return ExchangeEnvelope(
        exchange_id=f"realtime-exchange:{content[:12]}",
        session_id=session_id,
        content=content,
    )


def _spoke_ending(text: str) -> list[dict[str, Any]]:
    """A tool round with a successful speak, then the final assistant reply."""

    return [
        {"role": "tool", "results": [{"name": "speak", "status": "succeeded"}]},
        {"role": "assistant", "content": text},
    ]


def _draft_ending(text: str) -> list[dict[str, Any]]:
    """A pure-text ending: the draft is invisible in speak mode."""

    return [{"role": "assistant", "content": text}]


class _FakeHandle:
    """A turn execution with a react-style input inbox.

    ``send_input`` accepts wait-mode steering operations until the execution
    seals its inbox at the final answer; afterwards the delivery reports
    ``execution_finished`` exactly like a durable execution.
    """

    def __init__(self, execution_id: str) -> None:
        self.execution_id = execution_id
        self.sealed = False
        self.inputs: list[str] = []
        self.finished = asyncio.Event()

    async def send_input(
        self, *, input_id: str, kind: str, value: Any
    ) -> ExecutionInputDelivery:
        assert kind == REACT_PROJECTION_OPERATION_KIND
        operation = decode_react_projection_operation(value)
        assert isinstance(operation, StandaloneUserMessage)
        if self.sealed:
            return ExecutionInputDelivery(
                status="execution_finished",
                execution_id=self.execution_id,
                input_id=input_id,
            )
        self.inputs.append(operation.message.content)
        return ExecutionInputDelivery(
            status="accepted",
            execution_id=self.execution_id,
            input_id=input_id,
            sequence=len(self.inputs) - 1,
        )

    async def result(self) -> tuple[Any, None]:
        await self.finished.wait()
        return SimpleNamespace(data={"result": {"status": "passed"}}), None


class _FakeRuntimeService:
    """Drives the coordinator with scripted turn executions.

    ``start_turn`` mirrors the opening-context contract of
    ``PygentRuntimeService.start_turn``: the pending envelopes the coordinator
    hands over render into the recorded opening. The scripted execution then
    runs detached, as a real execution would, so exchange deliveries can
    land while the turn is RUNNING.
    """

    def __init__(self, coordinator: SessionExecutionCoordinator) -> None:
        self.coordinator = coordinator
        self.scripts: dict[str, _Script] = {}
        self.started: asyncio.Queue[str] = asyncio.Queue()
        self.completed: asyncio.Queue[str] = asyncio.Queue()
        self.messages: dict[str, str] = {}
        self.kinds: dict[str, str] = {}
        self.openings: dict[str, str] = {}
        self.handles: dict[str, _FakeHandle] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self._counter = 0

    async def start_turn(
        self,
        *,
        manager: Any,
        message: str,
        run_ref: CaseRunRef,
        turn_id: str,
        interactive_approvals: bool,
        message_kind: str,
        message_data: dict[str, object],
        attachments: Any,
        opening_exchanges: tuple[ExchangeEnvelope, ...] = (),
        deadline: float | None = None,
    ) -> _FakeHandle:
        del interactive_approvals, message_data, attachments, deadline
        self._counter += 1
        session = manager.load(run_ref.session_id)
        self.started.put_nowait(turn_id)
        self.messages[turn_id] = message
        self.kinds[turn_id] = message_kind
        self.openings[turn_id] = "\n".join(item.content for item in opening_exchanges)
        session.history.append({"role": "user", "content": message})
        handle = _FakeHandle(f"exec-{self._counter}")
        self.handles[turn_id] = handle
        script = self.scripts.get(
            turn_id, _Script(history=_spoke_ending(f"reply: {message}"))
        )

        async def execute() -> None:
            # Yield once so the coordinator marks the turn RUNNING before the
            # script delivers anything into it.
            await asyncio.sleep(0)
            await script.run(
                self.coordinator, manager, handle, session, run_ref.session_id
            )
            self.completed.put_nowait(turn_id)
            handle.finished.set()

        task = asyncio.create_task(execute())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return handle


class _FakeManager:
    def __init__(self, session: AgentSession, tmp_path: Path) -> None:
        self.session = session
        self.sessions_root = str(tmp_path / "sessions")
        self.config = _run_config(tmp_path)
        self._tmp_path = tmp_path
        self._run_count = 0

    def load(self, _session_id: str) -> AgentSession:
        return self.session

    def start_case_run(
        self, session_id: str, case_id: str, run_config: Any = None
    ) -> CaseRunRef:
        del run_config
        self._run_count += 1
        return CaseRunRef(
            session_id=session_id,
            case_id=case_id,
            case_run_id=f"run-{self._run_count:04d}",
            run_dir=str(self._tmp_path / f"run-{self._run_count:04d}"),
        )

    def finish_case_run(
        self, run_ref: CaseRunRef, status: str, error: str | None = None
    ) -> None:
        del run_ref, status, error


def _lease(manager: _FakeManager, service: _FakeRuntimeService) -> RuntimeLease:
    runtime = SessionRuntime(
        generation_key=RuntimeGenerationKey.from_config(manager.config),
        config=manager.config,
        manager=manager,  # type: ignore[arg-type]
        runtime_service=service,
    )

    async def _release() -> None:
        return None

    return RuntimeLease(runtime=runtime, _release=_release)


async def _completed(service: _FakeRuntimeService, count: int) -> list[str]:
    return [
        await asyncio.wait_for(service.completed.get(), timeout=5.0)
        for _ in range(count)
    ]


async def _harness(
    tmp_path: Path, *, mode: str = "pygent.v1-says"
) -> tuple[SessionExecutionCoordinator, _FakeManager, _FakeRuntimeService]:
    manager = _FakeManager(_session(mode), tmp_path)
    coordinator = SessionExecutionCoordinator()
    service = _FakeRuntimeService(coordinator)
    return coordinator, manager, service


async def _submit(
    coordinator: SessionExecutionCoordinator,
    lease: RuntimeLease,
    turn_id: str,
    message: str,
) -> Any:
    return await coordinator.submit_turn(
        lease=lease,
        command=TurnCommand(session_id="session-1", message=message, turn_id=turn_id),
    )


async def test_envelope_during_turn_is_injected_into_its_execution(
    tmp_path: Path,
) -> None:
    coordinator, manager, service = await _harness(tmp_path)
    script = _Script(
        history=_spoke_ending("work done"), deliver_during=["user asked: status?"]
    )
    service.scripts["turn-1"] = script
    lease = _lease(manager, service)
    await _submit(coordinator, lease, "turn-1", "do work")

    assert await _completed(service, 1) == ["turn-1"]
    assert script.deliveries == ["injected"]
    assert service.handles["turn-1"].inputs == ["user asked: status?"]
    assert set(service.kinds) == {"turn-1"}


async def test_envelope_pending_after_chain_end_gets_drained(tmp_path: Path) -> None:
    coordinator, manager, service = await _harness(tmp_path)
    script = _Script(
        history=_spoke_ending("work done"), deliver_after=["user asked: status?"]
    )
    service.scripts["turn-1"] = script
    lease = _lease(manager, service)
    await _submit(coordinator, lease, "turn-1", "do work")

    assert await _completed(service, 2) == ["turn-1", "turn-1:cont"]
    assert script.deliveries == ["pending"]
    assert service.handles["turn-1"].inputs == []
    assert service.kinds["turn-1:cont"] == CONTINUATION_MESSAGE_KIND
    assert service.messages["turn-1:cont"] == render_pending_input_message()
    assert "user asked: status?" in service.openings["turn-1:cont"]
    assert FINALIZER_MESSAGE_KIND not in service.kinds.values()
    assert not await coordinator._has_pending_exchanges(manager, "session-1")


async def test_both_drain_entry_points_deliver_a_late_envelope_once(
    tmp_path: Path,
) -> None:
    """A sealed-inbox rejection reaches both drain paths: the host's detached
    ``submit_pending_drain`` and the finishing turn's own tail. Whichever
    admits first wins; the other yields, so the envelope renders into exactly
    one continuation opening and is never duplicated or stranded."""

    coordinator, manager, service = await _harness(tmp_path)
    lease = _lease(manager, service)
    host_drains: list[Any] = []

    class _HostRacingScript(_Script):
        async def run(self, *args: Any) -> None:
            await super().run(*args)
            host_drains.append(
                await coordinator.submit_pending_drain(
                    lease=lease, session_id="session-1"
                )
            )

    script = _HostRacingScript(
        history=_spoke_ending("work done"), deliver_after=["user asked: status?"]
    )
    service.scripts["turn-1"] = script
    await _submit(coordinator, lease, "turn-1", "do work")

    completed = await _completed(service, 2)
    assert script.deliveries == ["pending"]
    continuations = [
        turn_id
        for turn_id, kind in service.kinds.items()
        if kind == CONTINUATION_MESSAGE_KIND
    ]
    assert len(continuations) == 1
    assert completed == ["turn-1", continuations[0]]
    assert service.openings[continuations[0]] == "user asked: status?"
    assert len(host_drains) == 1
    if host_drains[0] is None:
        assert continuations == ["turn-1:cont"]
    else:
        assert host_drains[0].command.turn_id == continuations[0]
    assert not await coordinator._has_pending_exchanges(manager, "session-1")
    await asyncio.sleep(0)
    assert service.completed.empty()


async def test_envelope_at_pure_text_end_defers_finalizer(tmp_path: Path) -> None:
    coordinator, manager, service = await _harness(tmp_path)
    script = _Script(
        history=_draft_ending("重构完成,改动 A/B/C"),
        deliver_after=["user asked: 影响性能吗?"],
    )
    service.scripts["turn-1"] = script
    lease = _lease(manager, service)
    await _submit(coordinator, lease, "turn-1", "帮我重构这个模块")

    assert await _completed(service, 2) == ["turn-1", "turn-1:cont"]
    assert service.kinds["turn-1:cont"] == CONTINUATION_MESSAGE_KIND
    assert "重构完成,改动 A/B/C" in service.messages["turn-1:cont"]
    assert "user asked: 影响性能吗?" in service.openings["turn-1:cont"]
    assert FINALIZER_MESSAGE_KIND not in service.kinds.values()


async def test_queued_user_turn_defers_finalizer_into_continuation(
    tmp_path: Path,
) -> None:
    coordinator, manager, service = await _harness(tmp_path)
    gate = asyncio.Event()
    service.scripts["turn-1"] = _Script(
        history=_draft_ending("重构完成,改动 A/B/C"), gate=gate
    )
    service.scripts["turn-2"] = _Script(
        history=_spoke_ending("好的,B 文件已恢复")
    )
    lease = _lease(manager, service)
    await _submit(coordinator, lease, "turn-1", "帮我重构这个模块")
    assert await service.started.get() == "turn-1"
    await _submit(coordinator, lease, "turn-2", "等等,先别动 B 文件")
    for _ in range(10):
        await asyncio.sleep(0)
    gate.set()

    assert await _completed(service, 3) == ["turn-1", "turn-2", "turn-1:cont"]
    assert service.kinds["turn-2"] == "pygent.v1.chat.turn"
    assert service.kinds["turn-1:cont"] == CONTINUATION_MESSAGE_KIND
    assert "重构完成,改动 A/B/C" in service.messages["turn-1:cont"]
    assert FINALIZER_MESSAGE_KIND not in service.kinds.values()


async def test_quiet_pure_text_end_keeps_plain_finalizer(tmp_path: Path) -> None:
    coordinator, manager, service = await _harness(tmp_path)
    service.scripts["turn-1"] = _Script(history=_draft_ending("draft text"))
    lease = _lease(manager, service)
    await _submit(coordinator, lease, "turn-1", "do work")

    assert await _completed(service, 2) == ["turn-1", "turn-1:finalize"]
    assert service.kinds["turn-1:finalize"] == FINALIZER_MESSAGE_KIND
    assert "NOT visible" in service.messages["turn-1:finalize"]
    assert CONTINUATION_MESSAGE_KIND not in service.kinds.values()


async def test_envelope_during_finalizer_chain_gets_drained(tmp_path: Path) -> None:
    coordinator, manager, service = await _harness(tmp_path)
    service.scripts["turn-1"] = _Script(history=_draft_ending("draft text"))
    service.scripts["turn-1:finalize"] = _Script(
        history=_spoke_ending("delivered"), deliver_after=["user asked: late?"]
    )
    lease = _lease(manager, service)
    await _submit(coordinator, lease, "turn-1", "do work")

    assert await _completed(service, 3) == [
        "turn-1",
        "turn-1:finalize",
        "turn-1:finalize:cont",
    ]
    assert service.kinds["turn-1:finalize"] == FINALIZER_MESSAGE_KIND
    assert service.kinds["turn-1:finalize:cont"] == CONTINUATION_MESSAGE_KIND
    assert "user asked: late?" in service.openings["turn-1:finalize:cont"]


async def test_plain_session_neither_finalizes_nor_drains(tmp_path: Path) -> None:
    coordinator, manager, service = await _harness(tmp_path, mode="e2e")
    service.scripts["turn-1"] = _Script(history=_draft_ending("draft text"))
    lease = _lease(manager, service)
    await _submit(coordinator, lease, "turn-1", "do work")

    assert await _completed(service, 1) == ["turn-1"]
    assert set(service.kinds.values()) == {"pygent.v1.chat.turn"}


async def test_envelope_in_idle_session_waits_for_next_turn(tmp_path: Path) -> None:
    """With no turn on the lane the coordinator only holds the envelope; the
    host decides whether to start a drain turn (``submit_pending_drain``)."""

    coordinator, manager, service = await _harness(tmp_path)
    assert (
        await coordinator.deliver_exchange(manager, _envelope("session-1", "early"))
        == "pending"
    )
    assert await coordinator._has_pending_exchanges(manager, "session-1")

    lease = _lease(manager, service)
    drain = await coordinator.submit_pending_drain(lease=lease, session_id="session-1")
    assert drain is not None
    drain_id = drain.command.turn_id
    assert drain_id is not None
    assert await _completed(service, 1) == [drain_id]
    assert service.kinds[drain_id] == CONTINUATION_MESSAGE_KIND
    assert service.openings[drain_id] == "early"
    assert not await coordinator._has_pending_exchanges(manager, "session-1")
    assert (
        await coordinator.submit_pending_drain(lease=lease, session_id="session-1")
        is None
    )


async def test_concurrent_idle_deliveries_share_one_drain_turn(tmp_path: Path) -> None:
    """Two envelopes landing in an idle session each ask the host for a drain;
    only the first admission goes through and its opening carries both."""

    coordinator, manager, service = await _harness(tmp_path)
    lease = _lease(manager, service)
    for content in ("first", "second"):
        assert (
            await coordinator.deliver_exchange(manager, _envelope("session-1", content))
            == "pending"
        )

    drains = await asyncio.gather(
        coordinator.submit_pending_drain(lease=lease, session_id="session-1"),
        coordinator.submit_pending_drain(lease=lease, session_id="session-1"),
    )
    admitted = [turn for turn in drains if turn is not None]
    assert len(admitted) == 1
    drain_id = admitted[0].command.turn_id
    assert drain_id is not None
    assert await _completed(service, 1) == [drain_id]
    assert service.openings[drain_id] == "first\nsecond"
    assert not await coordinator._has_pending_exchanges(manager, "session-1")
    await asyncio.sleep(0)
    assert service.completed.empty()


async def test_finalizer_turn_leaves_pending_envelopes_to_continuation(
    tmp_path: Path,
) -> None:
    """The finalizer decides only its own bottom line: an envelope that lands
    between the turn's end and the finalizer's start is not consumed by the
    finalizer's opening context."""

    coordinator, manager, service = await _harness(tmp_path)
    gate = asyncio.Event()
    service.scripts["turn-1"] = _Script(history=_draft_ending("draft text"))
    service.scripts["turn-1:finalize"] = _Script(
        history=_spoke_ending("delivered"), gate=gate
    )
    lease = _lease(manager, service)
    await _submit(coordinator, lease, "turn-1", "do work")
    assert await _completed(service, 1) == ["turn-1"]
    assert await service.started.get() == "turn-1"
    assert await service.started.get() == "turn-1:finalize"
    assert (
        await coordinator.deliver_exchange(manager, _envelope("session-1", "late?"))
        == "pending"
    )
    gate.set()

    assert await _completed(service, 2) == ["turn-1:finalize", "turn-1:finalize:cont"]
    assert service.openings["turn-1:finalize"] == ""
    assert service.openings["turn-1:finalize:cont"] == "late?"
