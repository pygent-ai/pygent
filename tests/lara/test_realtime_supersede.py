"""Realtime supersede-and-merge tests for fork replies."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from pygent import AIMessage
from pygent.lara.runtime.realtime.envelope import (
    format_user_asked,
    render_exchange_envelope,
)
from pygent.lara.runtime.realtime.fork_agent import FORK_RESULT_KIND
from pygent.lara.runtime.service import LaraRuntimeService
from pygent.lara.schema import AgentSession, CaseRunRef, RunConfig


class _Store:
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []

    def enqueue_message(self, **kwargs: object) -> object:
        self.messages.append(kwargs)
        return SimpleNamespace(**kwargs)


class _Handle:
    def __init__(self, message: str, *, final: bool, late: bool = False) -> None:
        self.message = message
        self.final = final
        self.late = late
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.cancel_count = 0

    async def cancel(self) -> bool:
        self.cancel_count += 1
        self.cancelled.set()
        if not self.late:
            self.release.set()
        return True

    async def result(self) -> tuple[AIMessage, None]:
        await self.release.wait()
        if self.cancelled.is_set() and not self.late:
            raise asyncio.CancelledError
        return (
            AIMessage(
                kind=FORK_RESULT_KIND,
                data={"via": "text", "text": f"reply:{self.message}", "speak_args": None},
            ),
            None,
        )


class _FakeService:
    def __init__(self, *, final_index: int, late_first: bool = False) -> None:
        self.config = RunConfig(
            workspace_root="C:/workspace",
            user_lara_root="C:/user",
            realtime_interaction={"enabled": True},  # type: ignore[arg-type]
        )
        self._fork_states: dict[str, object] = {}
        self.reminders = SimpleNamespace(message_store=_Store())
        self.handles: list[_Handle] = []
        self.started: asyncio.Queue[str] = asyncio.Queue()
        self.final_index = final_index
        self.late_first = late_first

    async def initialize(self) -> None:
        return None

    async def _start_fork_reply(
        self,
        *,
        session: AgentSession,
        run_ref: CaseRunRef,
        config: RunConfig,
        input_id: str,
        message: str,
    ) -> _Handle:
        del session, run_ref, config, input_id
        index = len(self.handles)
        handle = _Handle(
            message,
            final=index == self.final_index,
            late=self.late_first and index == 0,
        )
        self.handles.append(handle)
        await self.started.put(message)
        return handle


def _manager(tmp_path: Path) -> SimpleNamespace:
    session = AgentSession(
        session_id="session-1",
        workspace_root=str(tmp_path),
        session_dir=str(tmp_path / "session"),
        created_at=datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
        metadata={"mode": "lara-says"},
    )
    return SimpleNamespace(load=lambda _sid: session)


def _run_ref(tmp_path: Path) -> CaseRunRef:
    return CaseRunRef(
        session_id="session-1",
        case_id="case-1",
        case_run_id="run-1",
        run_dir=str(tmp_path / "run"),
    )


async def _send(service: _FakeService, tmp_path: Path, message: str, input_id: str):
    return await LaraRuntimeService.send_interrupt(
        service,  # type: ignore[arg-type]
        "execution-1",
        manager=_manager(tmp_path),  # type: ignore[arg-type]
        run_ref=_run_ref(tmp_path),
        session_id="session-1",
        input_id=input_id,
        message=message,
    )


async def test_three_interrupts_supersede_and_merge(tmp_path: Path) -> None:
    service = _FakeService(final_index=2)
    first = asyncio.create_task(_send(service, tmp_path, "first", "i1"))
    assert await service.started.get() == "first"
    second = asyncio.create_task(_send(service, tmp_path, "second", "i2"))
    assert await service.started.get() == "1. first\n2. second"
    third = asyncio.create_task(_send(service, tmp_path, "third", "i3"))
    assert await service.started.get() == "1. first\n2. second\n3. third"
    service.handles[2].release.set()
    await asyncio.gather(first, second, third)
    assert service.handles[0].cancel_count == 1
    assert service.handles[1].cancel_count == 1
    content = str(service.reminders.message_store.messages[-1]["content"])
    assert "1. first\n2. second\n3. third" in content


async def test_stale_generation_drops_late_envelope(tmp_path: Path) -> None:
    service = _FakeService(final_index=1, late_first=True)
    first = asyncio.create_task(_send(service, tmp_path, "first", "i1"))
    assert await service.started.get() == "first"
    second = asyncio.create_task(_send(service, tmp_path, "second", "i2"))
    assert await service.started.get() == "1. first\n2. second"
    service.handles[0].release.set()
    await first
    service.handles[1].release.set()
    await second
    assert len(service.reminders.message_store.messages) == 1
    assert "1. first\n2. second" in str(
        service.reminders.message_store.messages[0]["content"]
    )


def test_merged_user_asked_format_is_numbered() -> None:
    envelope = render_exchange_envelope(
        user_asked=["first", "second"], response="answer"
    )
    assert "<user-asked>\n1. first\n2. second\n</user-asked>" in envelope
    assert "<your-response-in-another-session>\nanswer\n</your-response-in-another-session>" in envelope
    assert format_user_asked("single") == "single"
