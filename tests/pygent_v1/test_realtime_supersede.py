"""Realtime supersede-and-merge tests for fork replies."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from pygent import AIMessage
from pygent.agent.pygent_v1.runtime.attachments import ResolvedAttachment
from pygent.agent.pygent_v1.runtime.realtime.envelope import (
    ExchangeEnvelope,
    format_user_asked,
    render_exchange_envelope,
)
from pygent.agent.pygent_v1.runtime.realtime.fork_agent import FORK_RESULT_KIND
from pygent.agent.pygent_v1.runtime.realtime.fork_state import SessionForkState
from pygent.agent.pygent_v1.runtime.service import PygentRuntimeService
from pygent.agent.pygent_v1.schema import AgentSession, CaseRunRef, RunConfig


class _Handle:
    def __init__(
        self,
        message: str,
        *,
        final: bool,
        late: bool = False,
        attachments: tuple[ResolvedAttachment, ...] = (),
        failure: BaseException | None = None,
    ) -> None:
        self.message = message
        self.attachments = attachments
        self.final = final
        self.late = late
        self.failure = failure
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
        if self.failure is not None:
            raise self.failure
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
    def __init__(
        self,
        *,
        final_index: int,
        late_first: bool = False,
        workspace_root: str = "C:/workspace",
        start_failure: BaseException | None = None,
        failure_index: int | None = None,
    ) -> None:
        self.config = RunConfig(
            workspace_root=workspace_root,
            user_pygent_root="C:/user",
            realtime_interaction={"enabled": True},  # type: ignore[arg-type]
        )
        self._fork_states: dict[str, SessionForkState] = {}
        self._speak_executor = SimpleNamespace(publish=self._publish)
        self.envelopes: list[ExchangeEnvelope] = []
        self.handles: list[_Handle] = []
        self.started: asyncio.Queue[str] = asyncio.Queue()
        self.final_index = final_index
        self.late_first = late_first
        self.start_failure = start_failure
        self.failure_index = failure_index

    async def _publish(self, *_args: object, **_kwargs: object) -> None:
        pass

    async def initialize(self) -> None:
        return None

    async def _on_exchange_ready(self, envelope: ExchangeEnvelope) -> None:
        self.envelopes.append(envelope)

    async def _deliver_unanswered_exchange(
        self,
        *,
        session_id: str,
        input_id: str,
        user_asked: Sequence[str],
        source_input_ids: tuple[str, ...] = (),
        attachments: Sequence[ResolvedAttachment] = (),
    ) -> None:
        await PygentRuntimeService._deliver_unanswered_exchange(
            self,
            session_id=session_id,
            input_id=input_id,
            user_asked=user_asked,
            source_input_ids=source_input_ids,
            attachments=attachments,
        )

    async def _start_fork_reply(
        self,
        *,
        session: AgentSession,
        run_ref: CaseRunRef,
        config: RunConfig,
        input_id: str,
        message: str,
        attachments: tuple[ResolvedAttachment, ...] = (),
    ) -> _Handle:
        del session, run_ref, config, input_id
        if self.start_failure is not None:
            raise self.start_failure
        index = len(self.handles)
        handle = _Handle(
            message,
            final=index == self.final_index,
            late=self.late_first and index == 0,
            attachments=attachments,
            failure=(
                RuntimeError("fork execution collapsed")
                if self.failure_index == index
                else None
            ),
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
        metadata={"mode": "pygent.v1-says"},
    )
    return SimpleNamespace(
        load=lambda _sid: session,
        start_case_run=lambda _sid, _cid: _run_ref(tmp_path),
    )


def _run_ref(tmp_path: Path) -> CaseRunRef:
    return CaseRunRef(
        session_id="session-1",
        case_id="case-1",
        case_run_id="run-1",
        run_dir=str(tmp_path / "run"),
    )


async def _send(
    service: _FakeService,
    tmp_path: Path,
    message: str,
    input_id: str,
    attachments: tuple[str, ...] = (),
):
    return await PygentRuntimeService.send_interrupt(
        service,  # type: ignore[arg-type]
        manager=_manager(tmp_path),  # type: ignore[arg-type]
        session_id="session-1",
        input_id=input_id,
        message=message,
        attachments=attachments,
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
    assert [item.exchange_id for item in service.envelopes] == [
        "realtime-exchange:i3:g3"
    ]
    assert "1. first\n2. second\n3. third" in service.envelopes[-1].content


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
    assert len(service.envelopes) == 1
    assert "1. first\n2. second" in service.envelopes[0].content


async def test_superseded_attachments_carry_into_merged_fork(tmp_path: Path) -> None:
    """A cancelled generation's files stay visible to the fork that replaces
    it, and the final envelope lists the whole batch's attachments."""

    (tmp_path / "shot.png").write_bytes(b"PNG")
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
    service = _FakeService(final_index=1, workspace_root=str(tmp_path))
    first = asyncio.create_task(
        _send(service, tmp_path, "look at this", "i1", ("shot.png",))
    )
    assert await service.started.get() == "look at this"
    assert [item.relative for item in service.handles[0].attachments] == ["shot.png"]
    second = asyncio.create_task(
        _send(service, tmp_path, "and this", "i2", ("notes.txt", "shot.png"))
    )
    assert await service.started.get() == "1. look at this\n2. and this"
    assert [item.relative for item in service.handles[1].attachments] == [
        "shot.png",
        "notes.txt",
    ]
    service.handles[1].release.set()
    await asyncio.gather(first, second)
    envelope = service.envelopes[-1]
    assert envelope.attachments == ("shot.png", "notes.txt")
    assert '<attachment delivery="media" name="shot.png" path="shot.png"' in envelope.content
    assert '<attachment delivery="inline" name="notes.txt" path="notes.txt"' in envelope.content
    message = envelope.as_user_message(case_run_id="run-1")
    assert tuple(message.data["attachments"]) == ("shot.png", "notes.txt")
    # The next interrupt after the batch settled starts clean.
    third = asyncio.create_task(_send(service, tmp_path, "plain", "i3"))
    assert await service.started.get() == "plain"
    assert service.handles[2].attachments == ()
    service.handles[2].release.set()
    await third
    assert service.envelopes[-1].attachments == ()


async def test_failed_fork_delivers_the_unanswered_interrupt(tmp_path: Path) -> None:
    """A fork that fails outright still injects the bare question: the main
    agent must learn the interrupt happened (D-RT2/3)."""

    service = _FakeService(final_index=0, failure_index=0)
    task = asyncio.create_task(_send(service, tmp_path, "帮我看下日志", "i1"))
    assert await service.started.get() == "帮我看下日志"
    service.handles[0].release.set()
    with pytest.raises(RuntimeError):
        await task
    assert len(service.envelopes) == 1
    envelope = service.envelopes[0]
    assert envelope.exchange_id == "realtime-exchange:i1"
    assert "<user-asked>\n帮我看下日志\n</user-asked>" in envelope.content
    assert "your-response-in-another-session" not in envelope.content
    state = service._fork_states["session-1"]
    assert state.handle is None
    assert state.pending == []


async def test_failed_fork_start_delivers_the_unanswered_interrupt(
    tmp_path: Path,
) -> None:
    """A fork that never starts (start/admission failure) loses nothing."""

    service = _FakeService(
        final_index=0, start_failure=RuntimeError("admission refused")
    )
    with pytest.raises(RuntimeError):
        await _send(service, tmp_path, "帮我看下日志", "i2")
    assert len(service.envelopes) == 1
    envelope = service.envelopes[0]
    assert envelope.exchange_id == "realtime-exchange:i2"
    assert "<user-asked>\n帮我看下日志\n</user-asked>" in envelope.content
    assert "your-response-in-another-session" not in envelope.content
    state = service._fork_states["session-1"]
    assert state.handle is None
    assert state.pending == []


async def test_failed_attachment_resolution_still_delivers_the_question(
    tmp_path: Path,
) -> None:
    service = _FakeService(final_index=0)
    with pytest.raises(ValueError):
        await _send(service, tmp_path, "看下这张图", "i3", ("missing.png",))
    assert len(service.envelopes) == 1
    envelope = service.envelopes[0]
    assert "<user-asked>\n看下这张图\n</user-asked>" in envelope.content
    assert "your-response-in-another-session" not in envelope.content


async def test_superseded_fork_failure_delivers_nothing(tmp_path: Path) -> None:
    """A failing generation that was already superseded stays silent: the
    merged envelope of the newer generation carries the information."""

    service = _FakeService(final_index=1, failure_index=0)
    first = asyncio.create_task(_send(service, tmp_path, "first", "i1"))
    assert await service.started.get() == "first"
    second = asyncio.create_task(_send(service, tmp_path, "second", "i2"))
    assert await service.started.get() == "1. first\n2. second"
    service.handles[1].release.set()
    results = await asyncio.gather(first, second, return_exceptions=True)
    assert isinstance(results[0], RuntimeError)
    assert not isinstance(results[1], BaseException)
    assert [item.exchange_id for item in service.envelopes] == [
        "realtime-exchange:i2:g2"
    ]
    assert "1. first\n2. second" in service.envelopes[0].content


def test_merged_user_asked_format_is_numbered() -> None:
    envelope = render_exchange_envelope(
        user_asked=["first", "second"], response="answer"
    )
    assert "<user-asked>\n1. first\n2. second\n</user-asked>" in envelope
    assert "<your-response-in-another-session>\nanswer\n</your-response-in-another-session>" in envelope
    assert format_user_asked("single") == "single"
