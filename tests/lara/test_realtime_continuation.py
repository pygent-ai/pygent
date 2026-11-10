"""End-to-end tests for the realtime continuation turn (D-RT1/D-RT11).

Pins the delivery contract for user input arriving around the end of a turn
chain: lane-queued turns and fork exchange envelopes must reach the main
agent's context through the coordinator's follow-up rounds — never strand in
the collaboration store, and never be decided against a stale draft pointer.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pygent.agent.lara.orchestration.models import TurnCommand
from pygent.agent.lara.orchestration.runtime_keys import RuntimeGenerationKey
from pygent.agent.lara.orchestration.runtime_lease import RuntimeLease, SessionRuntime
from pygent.agent.lara.orchestration.session_execution import (
    SessionExecutionCoordinator,
)
from pygent.agent.lara.runtime.realtime.continuation import (
    CONTINUATION_MESSAGE_KIND,
    render_pending_input_message,
)
from pygent.agent.lara.runtime.realtime.turn_finalizer import (
    FINALIZER_MESSAGE_KIND,
)
from pygent.agent.lara.runtime.reminders.agent_messages import (
    render_agent_message,
)
from pygent.agent.lara.runtime.reminders.service import ReminderService
from pygent.agent.lara.schema import AgentSession, CaseRunRef, RunConfig
from pygent.agent.lara.sessions.collaboration import (
    AgentMessageState,
    SessionCollaborationStore,
)
from pygent.agent.lara.sessions.manager import realtime_enabled_for


def _run_config(tmp_path: Path) -> RunConfig:
    return RunConfig(
        workspace_root=str(tmp_path / "workspace"),
        lara_root=str(tmp_path / "lara"),
        realtime_interaction={"enabled": True},  # type: ignore[arg-type]
    )


def _session(mode: str = "lara-says") -> AgentSession:
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

    ``enqueue_after`` models a fork reply that lands while the turn is
    finishing — after the turn's last claim opportunity, before the drive's
    tail checks run.
    """

    def __init__(
        self,
        *,
        history: list[dict[str, Any]] | None = None,
        gate: asyncio.Event | None = None,
        enqueue_after: list[str] | None = None,
    ) -> None:
        self.history = history or []
        self.gate = gate
        self.enqueue_after = enqueue_after or []
        self.enqueued: list[Any] = []

    async def run(
        self,
        store: SessionCollaborationStore,
        session: AgentSession,
        session_id: str,
    ) -> None:
        if self.gate is not None:
            await self.gate.wait()
        session.history.extend(self.history)
        for content in self.enqueue_after:
            self.enqueued.append(
                store.enqueue_message(
                    submission_id=f"realtime-exchange:{content[:12]}",
                    fingerprint=content,
                    source_session_id=None,
                    source_agent_alias=None,
                    target_session_id=session_id,
                    content=content,
                )
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
    def __init__(self, execution_id: str) -> None:
        self.execution_id = execution_id

    async def result(self) -> tuple[Any, None]:
        return SimpleNamespace(data={"result": {"status": "passed"}}), None


class _FakeRuntimeService:
    """Drives the coordinator against the real reminder/message machinery.

    ``start_turn`` mirrors the real opening-context slice of
    ``LaraRuntimeService.start_turn``: the turn-start claim runs against the
    real ``ReminderService``/store, the claimed envelopes render into the
    recorded opening, and the acknowledged state is what production leaves
    behind.
    """

    def __init__(self, reminders: ReminderService) -> None:
        self.reminders = reminders
        self.scripts: dict[str, _Script] = {}
        self.started: asyncio.Queue[str] = asyncio.Queue()
        self.completed: asyncio.Queue[str] = asyncio.Queue()
        self.messages: dict[str, str] = {}
        self.kinds: dict[str, str] = {}
        self.openings: dict[str, str] = {}
        self._counter = 0

    async def pending_agent_messages(self, session_id: str) -> tuple[Any, ...]:
        return await self.reminders.pending_agent_messages(session_id)

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
        deadline: float | None = None,
    ) -> _FakeHandle:
        del interactive_approvals, message_data, attachments, deadline
        self._counter += 1
        session = manager.load(run_ref.session_id)
        initial = await self.reminders.claim_turn_opening(
            run_ref.session_id,
            turn_id=turn_id,
            message_kind=message_kind,
            realtime_enabled=realtime_enabled_for(session),
        )
        self.started.put_nowait(turn_id)
        self.messages[turn_id] = message
        self.kinds[turn_id] = message_kind
        self.openings[turn_id] = "\n".join(
            render_agent_message(item) for item in initial
        )
        session.history.append({"role": "user", "content": message})
        script = self.scripts.get(
            turn_id, _Script(history=_spoke_ending(f"reply: {message}"))
        )
        await script.run(self.reminders.message_store, session, run_ref.session_id)
        claim_id = self.reminders.turn_opening_claim_id(run_ref.session_id, turn_id)
        for item in initial:
            await self.reminders.acknowledge_agent_message(
                item.message_id,
                claim_id=claim_id,
                execution_id=f"exec-{self._counter}",
            )
        self.completed.put_nowait(turn_id)
        return _FakeHandle(f"exec-{self._counter}")


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
    tmp_path: Path, *, mode: str = "lara-says"
) -> tuple[
    SessionExecutionCoordinator, _FakeManager, _FakeRuntimeService,
    SessionCollaborationStore,
]:
    store = SessionCollaborationStore(tmp_path / "lara")
    reminders = ReminderService(_run_config(tmp_path), message_store=store)
    manager = _FakeManager(_session(mode), tmp_path)
    service = _FakeRuntimeService(reminders)
    coordinator = SessionExecutionCoordinator()
    return coordinator, manager, service, store  # type: ignore[return-value]


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


async def test_envelope_queued_after_chain_end_gets_drained(tmp_path: Path) -> None:
    coordinator, manager, service, store = await _harness(tmp_path)
    script = _Script(
        history=_spoke_ending("work done"), enqueue_after=["user asked: status?"]
    )
    service.scripts["turn-1"] = script
    lease = _lease(manager, service)
    await _submit(coordinator, lease, "turn-1", "do work")

    assert await _completed(service, 2) == ["turn-1", "turn-1:cont"]
    assert service.kinds["turn-1:cont"] == CONTINUATION_MESSAGE_KIND
    assert service.messages["turn-1:cont"] == render_pending_input_message()
    assert "user asked: status?" in service.openings["turn-1:cont"]
    assert (
        store.get_message(script.enqueued[0].message_id).state
        is AgentMessageState.DELIVERED
    )
    assert FINALIZER_MESSAGE_KIND not in service.kinds.values()


async def test_envelope_at_pure_text_end_defers_finalizer(tmp_path: Path) -> None:
    coordinator, manager, service, _store = await _harness(tmp_path)
    script = _Script(
        history=_draft_ending("重构完成,改动 A/B/C"),
        enqueue_after=["user asked: 影响性能吗?"],
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
    coordinator, manager, service, _store = await _harness(tmp_path)
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
    assert service.kinds["turn-2"] == "lara.chat.turn"
    assert service.kinds["turn-1:cont"] == CONTINUATION_MESSAGE_KIND
    assert "重构完成,改动 A/B/C" in service.messages["turn-1:cont"]
    assert FINALIZER_MESSAGE_KIND not in service.kinds.values()


async def test_quiet_pure_text_end_keeps_plain_finalizer(tmp_path: Path) -> None:
    coordinator, manager, service, _store = await _harness(tmp_path)
    service.scripts["turn-1"] = _Script(history=_draft_ending("draft text"))
    lease = _lease(manager, service)
    await _submit(coordinator, lease, "turn-1", "do work")

    assert await _completed(service, 2) == ["turn-1", "turn-1:finalize"]
    assert service.kinds["turn-1:finalize"] == FINALIZER_MESSAGE_KIND
    assert "NOT visible" in service.messages["turn-1:finalize"]
    assert CONTINUATION_MESSAGE_KIND not in service.kinds.values()


async def test_envelope_during_finalizer_chain_gets_drained(tmp_path: Path) -> None:
    coordinator, manager, service, _store = await _harness(tmp_path)
    service.scripts["turn-1"] = _Script(history=_draft_ending("draft text"))
    service.scripts["turn-1:finalize"] = _Script(
        history=_spoke_ending("delivered"), enqueue_after=["user asked: late?"]
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
    coordinator, manager, service, _store = await _harness(tmp_path, mode="e2e")
    service.scripts["turn-1"] = _Script(history=_draft_ending("draft text"))
    lease = _lease(manager, service)
    await _submit(coordinator, lease, "turn-1", "do work")

    assert await _completed(service, 1) == ["turn-1"]
    assert set(service.kinds.values()) == {"lara.chat.turn"}


def test_pending_messages_peeks_without_claiming(tmp_path: Path) -> None:
    store = SessionCollaborationStore(tmp_path / "lara")
    first = store.enqueue_message(
        submission_id="s1",
        fingerprint="a",
        source_session_id=None,
        source_agent_alias=None,
        target_session_id="session-1",
        content="a",
    )
    second = store.enqueue_message(
        submission_id="s2",
        fingerprint="b",
        source_session_id=None,
        source_agent_alias=None,
        target_session_id="session-1",
        content="b",
    )
    other = store.enqueue_message(
        submission_id="s3",
        fingerprint="c",
        source_session_id=None,
        source_agent_alias=None,
        target_session_id="session-2",
        content="c",
    )

    store.claim_messages("session-1", claim_id="claim-1", lease_seconds=3600.0)
    assert store.get_message(second.message_id).state is AgentMessageState.CLAIMED
    assert store.pending_messages("session-1") == []

    store.release_message(first.message_id, claim_id="claim-1")
    assert [
        message.message_id for message in store.pending_messages("session-1")
    ] == [first.message_id]

    store.claim_messages("session-2", claim_id="claim-2", lease_seconds=3600.0)
    store.acknowledge_message(
        other.message_id, claim_id="claim-2", execution_id="exec-1"
    )
    assert store.pending_messages("session-2") == []
