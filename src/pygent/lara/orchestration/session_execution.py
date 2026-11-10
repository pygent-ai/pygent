from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pygent.lara.core.io import plain_object
from pygent.lara.core.prompt import system_reminder
from pygent.lara.runtime.realtime.continuation import (
    CONTINUATION_MESSAGE_KIND,
    render_draft_delivery_message,
    render_pending_input_message,
)
from pygent.lara.runtime.realtime.speak import SPEAK_TOOL_NAME
from pygent.lara.runtime.realtime.turn_finalizer import (
    FINALIZER_MESSAGE_KIND,
    decide_turn_finalization,
)
from pygent.lara.sessions import realtime_enabled_for

from .managed_turn import ManagedSessionTurn
from .models import TurnCommand, TurnState


@dataclass(frozen=True, slots=True)
class SessionExecutionKey:
    """Unambiguous identity for one session execution lane."""

    sessions_root: str
    session_id: str

    @classmethod
    def from_manager(cls, manager: Any, session_id: str) -> SessionExecutionKey:
        return cls(
            sessions_root=str(Path(manager.sessions_root).resolve()),
            session_id=session_id,
        )


@dataclass(slots=True)
class _SessionLane:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


class SessionExecutionCoordinator:
    """Own submission, lifecycle, lookup, and shutdown for session turns."""

    def __init__(self) -> None:
        self._managed_runs: dict[int, ManagedSessionTurn] = {}
        self._session_lanes: dict[SessionExecutionKey, _SessionLane] = {}
        self._lock = asyncio.Lock()
        self._closing = False
        self._finalized_turns: set[str] = set()

    async def submit_turn(
        self,
        *,
        lease: Any,
        command: TurnCommand,
    ) -> ManagedSessionTurn:
        """Admit a turn and return immediately, even when its session is busy."""

        turn = ManagedSessionTurn(
            lease=lease,
            command=command,
        )
        await self._admit(turn)
        return turn

    async def _admit(self, turn: ManagedSessionTurn) -> None:
        async with self._lock:
            if self._closing:
                raise RuntimeError("session execution coordinator is closing")
            self._managed_runs[id(turn)] = turn
            identity = turn.command.submission_id or turn.command.session_id
            turn.task = asyncio.create_task(
                self._drive_turn(turn),
                name=f"session-turn:{identity}",
            )

    async def _drive_turn(self, turn: ManagedSessionTurn) -> None:
        try:
            async with self.session_execution(
                turn.manager,
                turn.command.session_id,
                admitted=True,
            ):
                await self._drive_in_session_lane(turn)
        except asyncio.CancelledError:
            turn.state = TurnState.SKIPPED
            turn.ready.set()
            turn.finalized.set()
        except BaseException as exc:  # noqa: BLE001 - startup errors are captured and surfaced
            turn.startup_error = exc
            turn.state = TurnState.ERROR
            turn.ready.set()
            turn.finalized.set()
        finally:
            await self._remove(turn)

    async def _drive_in_session_lane(self, turn: ManagedSessionTurn) -> None:
        status = "error"
        error_detail: str | None = None
        try:
            turn.state = TurnState.STARTING
            if turn.run_ref is None:
                turn.run_ref = turn.manager.start_case_run(
                    turn.command.session_id,
                    turn.command.case_id,
                    run_config=turn.manager.config,
                )
            run_ref = turn.run_ref
            assert run_ref is not None
            if turn.recovery_execution_id is None:
                turn.execution_handle = await turn.runtime_service.start_turn(
                    manager=turn.manager,
                    message=turn.command.message,
                    run_ref=run_ref,
                    turn_id=(
                        turn.command.turn_id or f"turn-{run_ref.case_run_id[-8:]}"
                    ),
                    interactive_approvals=turn.command.interactive_approvals,
                    message_kind=turn.command.message_kind,
                    message_data=turn.command.message_data,
                    attachments=turn.command.attachments,
                )
            else:
                turn.execution_handle = await turn.runtime_service.recover_turn(
                    turn.recovery_execution_id,
                )
            execution_handle = turn.execution_handle
            assert execution_handle is not None
            turn.state = TurnState.RUNNING
            turn.ready.set()
            turn.output, turn.output_context = await execution_handle.result()
            output = turn.output
            assert output is not None
            result = plain_object(plain_object(output.data).get("result"))
            status = str(result.get("status") or "passed")
            if status == "error":
                error_detail = str(result.get("error") or "") or None
            else:
                # D-RT11: a pure-text ending is invisible in speak mode, so the
                # lane queues one bottom-line round before releasing the turn.
                # The follow-up enters the same lane after this drive exits.
                if not await self._maybe_start_finalizer(turn):
                    await self._maybe_drain_pending(turn)
        except asyncio.CancelledError:
            status = "skipped"
            if turn.execution_handle is not None:
                await turn.execution_handle.cancel()
        except BaseException as exc:  # noqa: BLE001 - startup errors are captured and surfaced
            turn.startup_error = exc
            # The startup error otherwise lives only in memory; persisting it
            # with the terminal is what makes a failed start diagnosable.
            error_detail = f"{type(exc).__name__}: {exc}"
        finally:
            turn.state = TurnState.FINALIZING
            terminal = _terminal_state(status)
            if turn.run_ref is not None:
                # The next turn cannot enter until final metadata is durable.
                turn.manager.finish_case_run(
                    turn.run_ref, terminal.value, error=error_detail
                )
            turn.state = terminal
            turn.ready.set()
            turn.finalized.set()

    async def _maybe_start_finalizer(self, turn: ManagedSessionTurn) -> bool:
        """Queue the D-RT11 bottom-line round for a pure-text turn ending.

        The final assistant message is invisible in speak mode, so the turn
        gets exactly one follow-up round whose reminder demands an explicit
        decision (speak the draft, or say_nothing). A second pure-text ending
        — including any ending of a finalizer turn itself — is accepted as a
        deliberate silent finish. The follow-up submits through the same
        session lane, so it serializes behind this drive.

        User input already pending when the turn ends — a lane-queued turn or
        a queued exchange envelope — defers the decision into a continuation
        turn instead: the continuation serializes behind the queued input,
        quotes the undelivered draft explicitly, and, being a normal turn,
        its own ending receives the standard bottom-line check. A finalizer
        submitted ahead of queued input would decide against a stale draft
        pointer, and its ``:finalize`` key would pin the inbox shut
        afterwards.

        Returns whether a follow-up round was submitted.
        """

        turn_key = _turn_key(turn)
        if turn_key in self._finalized_turns or turn_key.endswith(":finalize"):
            return False
        session = turn.manager.load(turn.command.session_id)
        if not realtime_enabled_for(session):
            return False
        last = session.history[-1] if session.history else None
        if (
            not isinstance(last, dict)
            or last.get("role") != "assistant"
            or last.get("tool_calls")
        ):
            return False
        decision = decide_turn_finalization(
            final_text=str(last.get("content") or ""),
            nudge_used=False,
            spoke=_preceding_round_spoke(session.history),
        )
        if decision.action != "nudge" or decision.reminder is None:
            return False
        draft = str(last.get("content") or "")
        self._finalized_turns.add(turn_key)
        if await self._has_pending_input(turn):
            await self.submit_turn(
                lease=turn.lease,
                command=TurnCommand(
                    session_id=turn.command.session_id,
                    case_id=turn.command.case_id,
                    message=render_draft_delivery_message(draft=draft),
                    turn_id=f"{turn_key}:cont",
                    interactive_approvals=False,
                    message_kind=CONTINUATION_MESSAGE_KIND,
                ),
            )
            return True
        await self.submit_turn(
            lease=turn.lease,
            command=TurnCommand(
                session_id=turn.command.session_id,
                case_id=turn.command.case_id,
                message=system_reminder(decision.reminder),
                turn_id=f"{turn_key}:finalize",
                interactive_approvals=False,
                message_kind=FINALIZER_MESSAGE_KIND,
            ),
        )
        return True

    def _lane_has_waiters(self, manager: Any, session_id: str) -> bool:
        """Whether turns are queued behind the lane lock.

        The driving turn itself is counted while it holds the lock, so more
        than one user means waiters.
        """

        key = SessionExecutionKey.from_manager(manager, session_id)
        lane = self._session_lanes.get(key)
        return lane is not None and lane.users > 1

    async def _has_pending_input(self, turn: ManagedSessionTurn) -> bool:
        """Whether user input is waiting at this turn's ending: lane-queued
        turns or queued exchange envelopes. Both defer the bottom-line
        decision into a continuation turn.
        """

        if self._lane_has_waiters(turn.manager, turn.command.session_id):
            return True
        return bool(
            await turn.runtime_service.pending_agent_messages(
                turn.command.session_id
            )
        )

    async def _maybe_drain_pending(self, turn: ManagedSessionTurn) -> None:
        """Submit a continuation turn for exchange envelopes queued after the
        whole turn chain ended (an interrupt that lost the race with turn
        completion). Without the drain the envelope would wait for the next
        user turn's opening context — or strand forever when none comes.
        """

        if await self._pending_drain_gate(
            turn.manager, turn.runtime_service, turn.command.session_id
        ):
            await self.submit_turn(
                lease=turn.lease,
                command=self._drain_command(
                    turn.command.session_id,
                    turn.command.case_id,
                    f"{_turn_key(turn)}:cont",
                ),
            )

    async def submit_pending_drain(
        self, *, lease: Any, session_id: str
    ) -> ManagedSessionTurn | None:
        """Submit a continuation turn for envelopes queued in an idle session.

        Called by the interrupt delivery after an exchange envelope lands: if
        no turn chain is active, no drive tail will ever drain the queue, so
        the delivery itself must. Returns the submitted turn, or ``None`` when
        the session is busy (its own tail will drain) or has nothing pending.
        """

        runtime = lease.runtime
        if not await self._pending_drain_gate(
            runtime.manager, runtime.runtime_service, session_id
        ):
            return None
        return await self.submit_turn(
            lease=lease,
            command=self._drain_command(
                session_id, "chat", f"drain-{uuid.uuid4().hex[:12]}"
            ),
        )

    async def _pending_drain_gate(
        self, manager: Any, runtime_service: Any, session_id: str
    ) -> bool:
        """Whether an idle realtime session holds undelivered envelopes.

        Both drain entry points share the gate: the chain's own tail (a turn
        just ended) and the interrupt delivery (an envelope just landed with
        no turn chain left that would run a tail).
        """

        session = manager.load(session_id)
        if not realtime_enabled_for(session):
            return False
        if self._lane_has_waiters(manager, session_id):
            return False
        return bool(await runtime_service.pending_agent_messages(session_id))

    def _drain_command(
        self, session_id: str, case_id: str, turn_id: str
    ) -> TurnCommand:
        return TurnCommand(
            session_id=session_id,
            case_id=case_id,
            message=render_pending_input_message(),
            turn_id=turn_id,
            interactive_approvals=False,
            message_kind=CONTINUATION_MESSAGE_KIND,
        )

    @contextlib.asynccontextmanager
    async def session_execution(
        self,
        manager: Any,
        session_id: str,
        *,
        admitted: bool = False,
    ) -> AsyncIterator[None]:
        key = SessionExecutionKey.from_manager(manager, session_id)
        async with self._lock:
            if self._closing and not admitted:
                raise RuntimeError("session execution coordinator is closing")
            lane = self._session_lanes.setdefault(key, _SessionLane())
            lane.users += 1
        try:
            async with lane.lock:
                yield
        finally:
            async with self._lock:
                lane.users -= 1
                if lane.users == 0 and self._session_lanes.get(key) is lane:
                    del self._session_lanes[key]

    async def _remove(self, turn: ManagedSessionTurn) -> None:
        async with self._lock:
            self._managed_runs.pop(id(turn), None)
        await turn.lease.release()

    async def close(self) -> None:
        async with self._lock:
            self._closing = True
            turns = tuple(self._managed_runs.values())
        if turns:
            await asyncio.gather(
                *(turn.cancel() for turn in turns),
                return_exceptions=True,
            )
            # A task cancelled before its coroutine receives its first timeslice
            # cannot execute _drive_turn's finally block. Seal those handles and
            # clear the index explicitly so shutdown remains observable.
            for turn in turns:
                if not turn.finalized.is_set():
                    turn.state = TurnState.SKIPPED
                    turn.ready.set()
                    turn.finalized.set()
            async with self._lock:
                for turn in turns:
                    self._managed_runs.pop(id(turn), None)
            await asyncio.gather(
                *(turn.lease.release() for turn in turns),
                return_exceptions=True,
            )


def _turn_key(turn: ManagedSessionTurn) -> str:
    return (
        turn.command.turn_id
        or f"turn-{(turn.run_ref.case_run_id if turn.run_ref else '')[-8:]}"
    )


def _preceding_round_spoke(history: list[dict[str, Any]]) -> bool:
    """Whether the tool round immediately before the final assistant message
    delivered a successful speak call.

    User-approved semantics: only the round right before the final reply
    counts. If the agent spoke earlier in the turn but the last tool round
    did not, the trailing draft still needs the bottom-line decision —
    the final answer text is otherwise invisible to the user.
    """

    if len(history) < 2:
        return False
    previous = history[-2]
    if previous.get("role") != "tool":
        return False
    return any(
        isinstance(result, dict)
        and result.get("name") == SPEAK_TOOL_NAME
        and result.get("status") == "succeeded"
        for result in previous.get("results") or ()
    )


def _terminal_state(status: str) -> TurnState:
    try:
        state = TurnState(status)
    except ValueError:
        return TurnState.ERROR
    return state if state.terminal else TurnState.ERROR


__all__ = ["SessionExecutionCoordinator", "SessionExecutionKey"]
