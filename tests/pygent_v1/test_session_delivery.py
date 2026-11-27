"""Routing tests for the unified user-message delivery entry.

Real session manager, real coordinator, real runtime pool; the runtime
service is a scripted fake so the tests observe which path
``deliver_user_message`` picks without a model wire. The fork path's own
behavior (supersede, merge, envelope) is covered by test_realtime_supersede
and the e2e delivery tests.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from pygent import AIMessage
from pygent.agent.pygent_v1.orchestration.execution_host import LocalExecutionHost
from pygent.agent.pygent_v1.orchestration.runtime_pool import WorkspaceRuntimePool
from pygent.agent.pygent_v1.runtime.realtime.fork_reply import ForkReplyOutcome
from pygent.agent.pygent_v1.schema import RunConfig
from pygent.agent.pygent_v1.sessions import SessionManager


class _FakeHandle:
    execution_id = "exec-fake"

    async def result(self) -> tuple[AIMessage, None]:
        return (
            AIMessage(
                kind="pygent.v1.chat.result",
                data={"result": {"status": "passed"}},
            ),
            None,
        )

    async def cancel(self) -> bool:
        return True

    def subscribe(self, *, after: int | None = None) -> Any:
        raise RuntimeError("routing tests never subscribe")


class _FakeRuntimeService:
    def __init__(self, config: RunConfig, **kwargs: Any) -> None:
        del kwargs
        self.config = config
        self.reminders = SimpleNamespace(prewarm_session=lambda _sid: None)
        self.started_turns: list[str] = []
        self.interrupts: list[tuple[str, str, tuple[str, ...]]] = []
        self.turn_gate: asyncio.Event | None = None

    async def initialize(self) -> None:
        return None

    async def close(self, *, cancel: bool) -> None:
        return None

    async def start_turn(
        self,
        *,
        manager: Any,
        message: str,
        run_ref: Any,
        turn_id: str | None,
        interactive_approvals: bool,
        message_kind: str = "pygent.v1.chat.turn",
        message_data: dict[str, object] | None = None,
        attachments: Sequence[str] = (),
        opening_exchanges: Sequence[Any] = (),
    ) -> _FakeHandle:
        del manager, run_ref, turn_id, interactive_approvals
        del message_kind, message_data, attachments, opening_exchanges
        self.started_turns.append(message)
        if self.turn_gate is not None:
            await self.turn_gate.wait()
        return _FakeHandle()

    async def send_interrupt(
        self,
        *,
        manager: Any,
        session_id: str,
        input_id: str,
        message: str,
        case_id: str = "chat",
        attachments: Sequence[str] = (),
    ) -> ForkReplyOutcome:
        del manager, session_id, case_id
        self.interrupts.append((input_id, message, tuple(attachments)))
        return ForkReplyOutcome(
            via="text",
            speak_args=None,
            text="ack",
            message=AIMessage(content="ack"),
        )


class _Harness:
    def __init__(self, tmp_path: Path, *, create_session: bool, mode: str) -> None:
        self.config = RunConfig(
            workspace_root=str(tmp_path / "ws"),
            user_pygent_root=str(tmp_path / "home"),
        )
        self.manager = SessionManager(self.config)
        self.session_id: str | None = None
        if create_session:
            self.session_id = self.manager.create("chat", mode=mode).session_id
        self.services: list[_FakeRuntimeService] = []

        def factory(config: RunConfig, **kwargs: Any) -> _FakeRuntimeService:
            service = _FakeRuntimeService(config, **kwargs)
            self.services.append(service)
            return service

        self.host = LocalExecutionHost(
            runtime_pool=WorkspaceRuntimePool(runtime_factory=factory)
        )

    @property
    def service(self) -> _FakeRuntimeService:
        assert self.services, "runtime service not created yet"
        return self.services[0]

    async def service_eagerly(self) -> _FakeRuntimeService:
        lease = await self.host.acquire_runtime(
            config=self.config, manager=self.manager
        )
        service = lease.runtime.runtime_service
        assert isinstance(service, _FakeRuntimeService)
        await lease.release()
        return service

    async def deliver(
        self,
        message: str,
        *,
        input_id: str | None = None,
        session_id: str | None = None,
        attachments: Sequence[str] = (),
    ) -> Any:
        return await self.host.deliver_user_message(
            config=self.config,
            manager=self.manager,
            message=message,
            session_id=self.session_id if session_id is None else session_id,
            input_id=input_id,
            attachments=attachments,
        )

    async def wait_turn_started(self) -> None:
        while not self.service.started_turns:
            await asyncio.sleep(0)


async def test_idle_realtime_session_starts_turn(tmp_path: Path) -> None:
    harness = _Harness(tmp_path, create_session=True, mode="pygent.v1-says")
    delivery = await harness.deliver("帮我重构这个模块", input_id="t1")

    assert delivery.mode == "turn"
    assert delivery.turn is not None
    assert harness.service.interrupts == []
    await delivery.turn.result()
    assert delivery.turn.status == "passed"
    assert harness.service.started_turns == ["帮我重构这个模块"]


async def test_busy_realtime_session_interrupts(tmp_path: Path) -> None:
    harness = _Harness(tmp_path, create_session=True, mode="pygent.v1-says")
    service = await harness.service_eagerly()
    gate = asyncio.Event()
    service.turn_gate = gate
    first = await harness.deliver("第一条", input_id="t1")
    assert first.mode == "turn"
    await harness.wait_turn_started()

    second = await harness.deliver("第二条", input_id="i1")

    assert second.mode == "interrupt"
    assert second.turn is None
    assert second.outcome is not None
    assert second.outcome.text == "ack"
    assert service.interrupts == [("i1", "第二条", ())]
    gate.set()
    await first.turn.result()


async def test_plain_busy_session_queues_turns_never_interrupts(
    tmp_path: Path,
) -> None:
    harness = _Harness(tmp_path, create_session=True, mode="chat")
    service = await harness.service_eagerly()
    gate = asyncio.Event()
    service.turn_gate = gate
    first = await harness.deliver("第一条", input_id="t1")
    await harness.wait_turn_started()

    second = await harness.deliver("第二条", input_id="t2")

    assert second.mode == "turn"
    assert service.interrupts == []
    gate.set()
    await first.turn.result()
    await second.turn.result()
    assert service.started_turns == ["第一条", "第二条"]


async def test_attachments_on_interrupt_reach_the_fork(tmp_path: Path) -> None:
    harness = _Harness(tmp_path, create_session=True, mode="pygent.v1-says")
    service = await harness.service_eagerly()
    gate = asyncio.Event()
    service.turn_gate = gate
    first = await harness.deliver("第一条", input_id="t1")
    await harness.wait_turn_started()

    second = await harness.deliver("带图的消息", input_id="i1", attachments=["a.png"])

    assert second.mode == "interrupt"
    assert service.interrupts == [("i1", "带图的消息", ("a.png",))]
    gate.set()
    await first.turn.result()


async def test_delivery_without_session_creates_chat_turn(tmp_path: Path) -> None:
    harness = _Harness(tmp_path, create_session=False, mode="chat")
    delivery = await harness.deliver("你好")

    assert delivery.mode == "turn"
    assert delivery.turn is not None
    session_id = delivery.turn.command.session_id
    session = harness.manager.load(session_id)
    assert session.metadata.get("mode") == "chat"
    await delivery.turn.result()
    assert delivery.turn.status == "passed"


async def test_queued_message_is_replayed_after_host_restart(tmp_path: Path) -> None:
    original = _Harness(tmp_path, create_session=True, mode="chat")
    service = await original.service_eagerly()
    service.turn_gate = asyncio.Event()
    first = await original.deliver("first", input_id="first")
    await original.wait_turn_started()
    second = await original.deliver("second", input_id="second")
    assert first.turn is not None and second.turn is not None
    await original.host.aclose()

    restarted = _Harness(tmp_path, create_session=False, mode="chat")
    try:
        restarted.session_id = original.session_id
        await restarted.host.recover_pending_messages(
            config=restarted.config, manager=restarted.manager
        )
        result = await asyncio.wait_for(
            restarted.deliver("second", input_id="second"), timeout=10
        )
        assert result.turn is not None
        await asyncio.wait_for(result.turn.result(), timeout=10)
        duplicate = await restarted.deliver("second", input_id="second")
        assert duplicate.turn is not None
        assert (await duplicate.turn.result())[0].kind == "pygent.v1.chat.result"
        assert restarted.service.started_turns == ["first", "second"]
    finally:
        await restarted.host.aclose()


async def test_two_hosts_share_foreground_order(tmp_path: Path) -> None:
    first_host = _Harness(tmp_path, create_session=True, mode="chat")
    second_host = _Harness(tmp_path, create_session=False, mode="chat")
    second_host.session_id = first_host.session_id
    first_runtime = await first_host.service_eagerly()
    second_runtime = await second_host.service_eagerly()
    gate = asyncio.Event()
    first_runtime.turn_gate = gate
    try:
        first = await first_host.deliver("first", input_id="first")
        await first_host.wait_turn_started()
        second = await second_host.deliver("second", input_id="second")
        await asyncio.sleep(0.1)
        assert first_runtime.started_turns == ["first"]
        assert second_runtime.started_turns == []

        gate.set()
        assert first.turn is not None and second.turn is not None
        await asyncio.wait_for(asyncio.gather(first.turn.result(), second.turn.result()), 10)
        assert first_runtime.started_turns == ["first"]
        assert second_runtime.started_turns == ["second"]
    finally:
        gate.set()
        await first_host.host.aclose()
        await second_host.host.aclose()


async def test_delivery_lock_wait_keeps_loop_responsive_and_joins_cancelled_write(
    tmp_path: Path,
) -> None:
    harness = _Harness(tmp_path, create_session=True, mode="chat")
    store = harness.host.coordinator.delivery_store(harness.manager)
    with store.transaction():
        pass

    locked = threading.Event()
    release = threading.Event()
    writer_done = threading.Event()

    def hold_database_lock() -> None:
        with closing(sqlite3.connect(store.path)) as db, db:
            db.execute("BEGIN EXCLUSIVE")
            locked.set()
            release.wait(3)
        writer_done.set()

    writer = asyncio.create_task(asyncio.to_thread(hold_database_lock))
    try:
        assert await asyncio.to_thread(locked.wait, 2)
        delivery = asyncio.create_task(harness.deliver("message", input_id="lock-test"))
        await asyncio.sleep(0.05)
        assert not writer_done.is_set()
        assert not delivery.done()

        delivery.cancel()
        await asyncio.sleep(0)
        delivery.cancel()
        await asyncio.sleep(0)
        assert not delivery.done()

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(delivery, 5)
        lane = f"user:{harness.session_id}"
        assert (await store.run(store.get, lane, "lock-test"))["receipt"] is None

        retry = await asyncio.wait_for(harness.deliver("message", input_id="lock-test"), 5)
        assert retry.turn is not None
        await asyncio.wait_for(retry.turn.result(), 5)
        assert harness.service.started_turns == ["message"]
    finally:
        release.set()
        await writer
        await harness.host.aclose()
