from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pygent.agent.pygent_v1.core.io import plain_object
from pygent.agent.pygent_v1.core.prompt import system_reminder
from pygent.agent.pygent_v1.runtime.delivery_store import (
    DeliveryStore,
    session_delivery_path,
)
from pygent.agent.pygent_v1.runtime.realtime.continuation import (
    CONTINUATION_MESSAGE_KIND,
    render_draft_delivery_message,
    render_pending_input_message,
)
from pygent.agent.pygent_v1.runtime.realtime.envelope import ExchangeEnvelope
from pygent.agent.pygent_v1.runtime.realtime.speak import SPEAK_TOOL_NAME
from pygent.agent.pygent_v1.runtime.realtime.turn_finalizer import (
    FINALIZER_MESSAGE_KIND,
    decide_turn_finalization,
)
from pygent.agent.pygent_v1.schema import CaseRunRef
from pygent.agent.pygent_v1.sessions import realtime_enabled_for
from pygent.agent.react_projection_operations import REACT_PROJECTION_OPERATION_KIND

from .managed_turn import ManagedSessionTurn
from .models import TurnCommand, TurnState


def transcript_contains_input(manager: Any, session_id: str, input_id: str) -> bool:
    for item in reversed(manager.load(session_id).history):
        data = plain_object(item.get("data"))
        if data.get("input_id") == input_id or input_id in (
            data.get("opening_exchange_ids") or ()
        ):
            return True
    return False


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
    # The turn currently driving inside the lane lock; None while no turn
    # holds the lock or between a turn's exit and the next one's entry.
    active: ManagedSessionTurn | None = None


class SessionExecutionCoordinator:
    """Own submission, lifecycle, lookup, and shutdown for session turns."""

    def __init__(self) -> None:
        self._managed_runs: dict[int, ManagedSessionTurn] = {}
        self._session_lanes: dict[SessionExecutionKey, _SessionLane] = {}
        self._delivery_stores: dict[str, DeliveryStore] = {}
        self._delivery_owner = uuid.uuid4().hex
        # Continuation turns admitted for pending envelopes but not yet opened.
        # Admission increments the lane's user count only once the turn's task
        # runs, so this reservation is what keeps the chain's tail and the
        # host's detached delivery from both admitting a drain for the same
        # envelopes (the second would open empty).
        self._admitted_drains: dict[SessionExecutionKey, ManagedSessionTurn] = {}
        self._lock = asyncio.Lock()
        self._closing = False
        self._finalized_turns: set[str] = set()

    def delivery_store(self, manager: Any) -> DeliveryStore:
        root = str(Path(manager.sessions_root).resolve())
        store = self._delivery_stores.get(root)
        if store is None:
            store = DeliveryStore(session_delivery_path(root))
            self._delivery_stores[root] = store
        return store

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

    async def find_delivery_turn(self, execution_id: str) -> ManagedSessionTurn | None:
        async with self._lock:
            return next(
                (
                    turn
                    for turn in self._managed_runs.values()
                    if execution_id in (turn.execution_id, turn.recovery_execution_id)
                ),
                None,
            )

    async def recover_turn(
        self, *, lease: Any, run_ref: CaseRunRef, execution_id: str
    ) -> ManagedSessionTurn:
        turn = ManagedSessionTurn(
            lease=lease,
            command=TurnCommand(
                session_id=run_ref.session_id,
                case_id=run_ref.case_id,
                message="recovered execution",
            ),
            run_ref=run_ref,
            recovery_execution_id=execution_id,
        )
        async with self._lock:
            existing = next(
                (
                    active
                    for active in self._managed_runs.values()
                    if execution_id
                    in (active.execution_id, active.recovery_execution_id)
                ),
                None,
            )
            if existing is None:
                self._admit_locked(turn)
                return turn
        await lease.release()
        return existing

    async def _admit(self, turn: ManagedSessionTurn) -> None:
        async with self._lock:
            self._admit_locked(turn)

    def _admit_locked(self, turn: ManagedSessionTurn) -> None:
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
            user_lane = f"user:{turn.command.session_id}"
            if (
                turn.command.message_data.get("delivery_execution_id")
                and turn.command.submission_id
            ):
                store = self.delivery_store(turn.manager)
                while await store.run(
                    store.prior_turn_pending, user_lane, turn.command.submission_id
                ):
                    await asyncio.sleep(0.05)
            after_deliveries = turn.command.message_data.get("after_deliveries")
            if isinstance(after_deliveries, list) and after_deliveries:
                store = self.delivery_store(turn.manager)
                while any(
                    row["key"] in after_deliveries
                    for row in await store.run(store.pending, user_lane)
                ):
                    await asyncio.sleep(0.05)
            async with self.session_execution(
                turn.manager,
                turn.command.session_id,
                admitted=True,
                active=turn,
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
        opening_exchanges: tuple[ExchangeEnvelope, ...] = ()
        opening_is_durable = False
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
                opening_exchanges = await self._opening_exchanges(turn)
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
                    opening_exchanges=opening_exchanges,
                )
                if opening_exchanges:
                    history = getattr(turn.runtime_service, "history", None)
                    opening_is_durable = (
                        history is None
                        or await history.get_execution(
                            turn.execution_handle.execution_id
                        )
                        is not None
                    )
                    if opening_is_durable:
                        await self._acknowledge_opening_exchanges(
                            turn, opening_exchanges
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
            if opening_exchanges and not opening_is_durable:
                await self._acknowledge_opening_exchanges(turn, opening_exchanges)
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
            if opening_exchanges:
                store = self.delivery_store(turn.manager)
                lane = f"exchange:{turn.command.session_id}"
                for exchange in opening_exchanges:
                    await store.run(
                        store.retry, lane, exchange.exchange_id, self._delivery_owner
                    )
            turn.state = TurnState.FINALIZING
            terminal = _terminal_state(status)
            if turn.run_ref is not None:
                # The next turn cannot enter until final metadata is durable.
                turn.manager.finish_case_run(
                    turn.run_ref, terminal.value, error=error_detail
                )
                if opening_exchanges or turn.execution_id is not None:
                    store = self.delivery_store(turn.manager)
                    recovery_lane = f"recover:{turn.command.session_id}"
                    case_run_id = turn.run_ref.case_run_id

                    def record_recovery_receipts() -> None:
                        with store.transaction() as db:
                            if opening_exchanges:
                                store.set_receipt(
                                    db,
                                    recovery_lane,
                                    f"request:{case_run_id}",
                                    {
                                        "status": terminal.value,
                                    },
                                )
                            if turn.execution_id is not None:
                                store.set_receipt(
                                    db,
                                    recovery_lane,
                                    f"input:{turn.execution_id}",
                                    {
                                        "status": terminal.value,
                                    },
                                )

                    await store.run(record_recovery_receipts)
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
            store = self.delivery_store(turn.manager)
            queued_inputs = [
                row["key"]
                for row in await store.run(
                    store.pending, f"user:{turn.command.session_id}"
                )
                if row["target"] is None or row["target"].startswith("turn:")
            ]
            await self.submit_turn(
                lease=turn.lease,
                command=TurnCommand(
                    session_id=turn.command.session_id,
                    case_id=turn.command.case_id,
                    message=render_draft_delivery_message(draft=draft),
                    turn_id=f"{turn_key}:cont",
                    interactive_approvals=False,
                    message_kind=CONTINUATION_MESSAGE_KIND,
                    message_data={"after_deliveries": queued_inputs},
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

    def has_active_execution(self, manager: Any, session_id: str) -> bool:
        """Whether a turn chain currently holds or queues on the session lane.

        Called from outside the lane (message delivery routing), so any
        counted user means an active or pending turn — unlike
        :meth:`_lane_has_waiters`, which runs from inside the driving turn.
        """

        key = SessionExecutionKey.from_manager(manager, session_id)
        lane = self._session_lanes.get(key)
        return (lane is not None and lane.users > 0) or any(
            SessionExecutionKey.from_manager(turn.manager, turn.command.session_id)
            == key
            for turn in self._managed_runs.values()
            if not turn.state.terminal
        )

    def _lane_has_waiters(self, manager: Any, session_id: str) -> bool:
        """Whether turns are queued behind the lane lock.

        The driving turn itself is counted while it holds the lock, so more
        than one user means waiters.
        """

        key = SessionExecutionKey.from_manager(manager, session_id)
        lane = self._session_lanes.get(key)
        return lane is not None and (
            lane.users > 1
            or any(
                turn is not lane.active
                and not turn.state.terminal
                and SessionExecutionKey.from_manager(
                    turn.manager, turn.command.session_id
                )
                == key
                for turn in self._managed_runs.values()
            )
        )

    async def _has_pending_input(self, turn: ManagedSessionTurn) -> bool:
        """Whether user input is waiting at this turn's ending: lane-queued
        turns or pending exchange envelopes. Both defer the bottom-line
        decision into a continuation turn.
        """

        if self._lane_has_waiters(turn.manager, turn.command.session_id):
            return True
        return await self._has_pending_exchanges(turn.manager, turn.command.session_id)

    async def _maybe_drain_pending(self, turn: ManagedSessionTurn) -> None:
        """Submit a continuation turn for exchange envelopes left pending after
        the whole turn chain ended (a fork reply that lost the race with turn
        completion). Without the drain the envelope would wait for the next
        user turn's opening context — or strand forever when none comes.
        """

        await self._admit_pending_drain(
            lease=turn.lease,
            manager=turn.manager,
            session_id=turn.command.session_id,
            case_id=turn.command.case_id,
            turn_id=f"{_turn_key(turn)}:cont",
        )

    async def submit_pending_drain(
        self, *, lease: Any, session_id: str
    ) -> ManagedSessionTurn | None:
        """Submit a continuation turn for envelopes pending in an idle session.

        Called by the exchange delivery when :meth:`deliver_exchange` could not
        inject: if no turn chain is active, no drive tail will ever pick the
        envelope up, so the delivery itself must. Returns the submitted turn,
        or ``None`` when the session is busy (its own tail will drain) or has
        nothing pending.
        """

        return await self._admit_pending_drain(
            lease=lease,
            manager=lease.runtime.manager,
            session_id=session_id,
            case_id="chat",
            turn_id=f"drain-{uuid.uuid4().hex[:12]}",
        )

    async def _admit_pending_drain(
        self,
        *,
        lease: Any,
        manager: Any,
        session_id: str,
        case_id: str,
        turn_id: str,
    ) -> ManagedSessionTurn | None:
        """Admit one continuation turn when a realtime session holds undelivered
        envelopes and no admitted turn will open with them.

        Both drain entry points share this: the chain's own tail (a turn just
        ended) and the interrupt delivery (an envelope just landed with no turn
        chain left that would run a tail). The gate and the admission hold one
        lock so the two cannot both admit a drain for the same envelopes; a
        lane-queued user turn or an already admitted drain opens with them.
        """

        session = manager.load(session_id)
        if not realtime_enabled_for(session):
            return None
        key = SessionExecutionKey.from_manager(manager, session_id)
        async with self._lock:
            lane = self._session_lanes.get(key)
            if lane is not None:
                if self._lane_has_waiters(manager, session_id):
                    return None
                if lane.active is not None and lane.active.state is TurnState.STARTING:
                    return None
            if key in self._admitted_drains:
                return None
            store = self.delivery_store(manager)
            if not await store.run(
                store.pending, f"exchange:{session_id}", available=True
            ):
                return None
            turn = ManagedSessionTurn(
                lease=lease,
                command=self._drain_command(session_id, case_id, turn_id),
            )
            self._admit_locked(turn)
            self._admitted_drains[key] = turn
        return turn

    async def deliver_exchange(self, manager: Any, envelope: ExchangeEnvelope) -> str:
        """Hand one fork-reply exchange to the session's main agent.

        A turn running on the lane receives it as a wait-mode steering input,
        which the react layer applies after its pending tool results and
        before the next model call. Anything else — no turn, a turn still
        starting, a turn whose execution already sealed its inbox, or the
        finalizer round (which decides only its own turn's bottom line) —
        holds the envelope for the next turn's opening context. Returns
        ``"injected"`` or ``"pending"``; the caller decides whether a pending
        envelope needs a continuation turn (:meth:`submit_pending_drain`).
        """

        key = SessionExecutionKey.from_manager(manager, envelope.session_id)
        store = self.delivery_store(manager)
        lane_name = f"exchange:{envelope.session_id}"

        def exchange_already_delivered() -> bool:
            with store.transaction() as db:
                # The reply producer records the exchange with the outbox.
                row = (
                    store.find(db, lane_name, envelope.exchange_id)
                    if envelope.reply is not None
                    else None
                )
                if row is None:
                    row = store.record_exchange(db, envelope)
                return row["receipt"] is not None

        if await store.run(exchange_already_delivered):
            return "injected"
        async with self._lock:
            lane = self._session_lanes.get(key)
            active = lane.active if lane is not None else None
        if (
            active is not None
            and active.state is TurnState.RUNNING
            and active.command.message_kind != FINALIZER_MESSAGE_KIND
            and active.execution_handle is not None
            and active.run_ref is not None
        ):
            row = await store.run(
                store.claim, lane_name, envelope.exchange_id, self._delivery_owner, 30
            )
            if row is None:
                return "pending"
            try:
                if await self._exchange_committed(
                    active, row["target"], envelope.exchange_id
                ):
                    await store.run(
                        store.finish,
                        lane_name,
                        envelope.exchange_id,
                        {"delivered": True},
                    )
                    return "injected"
                destination = f"input:{active.execution_id}"
                await store.run(
                    store.set_target, lane_name, envelope.exchange_id, destination
                )

                def record_recovery_target() -> None:
                    with store.transaction() as db:
                        store.put(db, f"recover:{envelope.session_id}", destination, {})

                await store.run(record_recovery_target)
                delivery = await active.execution_handle.send_input(
                    input_id=envelope.exchange_id,
                    kind=REACT_PROJECTION_OPERATION_KIND,
                    value=envelope.projection_operation(
                        case_run_id=active.run_ref.case_run_id
                    ),
                )
                if delivery.status in {"accepted", "duplicate"}:
                    history = getattr(active.runtime_service, "history", None)
                    if history is None or await history.has_execution_input(
                        active.execution_handle.execution_id, envelope.exchange_id
                    ):
                        await store.run(
                            store.finish,
                            lane_name,
                            envelope.exchange_id,
                            {"delivered": True},
                        )
                    return "injected"
            finally:
                await store.run(
                    store.retry, lane_name, envelope.exchange_id, self._delivery_owner
                )
        return "pending"

    async def _exchange_committed(
        self, turn: ManagedSessionTurn, target: str | None, exchange_id: str
    ) -> bool:
        history = getattr(turn.runtime_service, "history", None)
        if target is None:
            return False
        if history is not None:
            kind, identity = target.split(":", 1)
            if kind == "request":
                if await history.get_execution_by_request(identity) is not None:
                    return True
            elif await history.has_execution_input(identity, exchange_id):
                return True
        return transcript_contains_input(
            turn.manager, turn.command.session_id, exchange_id
        )

    async def _has_pending_exchanges(self, manager: Any, session_id: str) -> bool:
        store = self.delivery_store(manager)
        return bool(await store.run(store.pending, f"exchange:{session_id}"))

    async def _opening_exchanges(
        self, turn: ManagedSessionTurn
    ) -> tuple[ExchangeEnvelope, ...]:
        if turn.command.message_kind == FINALIZER_MESSAGE_KIND:
            return ()
        assert turn.run_ref is not None
        session_id = turn.command.session_id
        key = SessionExecutionKey.from_manager(turn.manager, session_id)
        store = self.delivery_store(turn.manager)
        lane = f"exchange:{session_id}"
        async with self._lock:
            if self._admitted_drains.get(key) is turn:
                del self._admitted_drains[key]
        exchanges: list[ExchangeEnvelope] = []
        for candidate in await store.run(store.pending, lane):
            exchange_id = candidate["key"]
            row = await store.run(
                store.claim, lane, exchange_id, self._delivery_owner, 30
            )
            if row is None:
                continue
            if await self._exchange_committed(turn, row["target"], exchange_id):
                await store.run(store.finish, lane, exchange_id, {"delivered": True})
                continue
            destination = f"request:{turn.run_ref.case_run_id}"

            def record_opening_target(destination: str, exchange_id: str) -> None:
                with store.transaction() as db:
                    store.put(db, f"recover:{session_id}", destination, {})
                    db.execute(
                        "UPDATE deliveries SET target=? WHERE lane=? AND key=?",
                        (destination, lane, exchange_id),
                    )

            await store.run(record_opening_target, destination, exchange_id)
            exchanges.append(
                ExchangeEnvelope(
                    exchange_id=exchange_id,
                    session_id=session_id,
                    **json.loads(row["payload"]),
                )
            )
        return tuple(exchanges)

    async def _acknowledge_opening_exchanges(
        self, turn: ManagedSessionTurn, exchanges: tuple[ExchangeEnvelope, ...]
    ) -> None:
        store = self.delivery_store(turn.manager)
        lane = f"exchange:{turn.command.session_id}"

        def acknowledge_batch() -> None:
            with store.transaction() as db:
                for exchange in exchanges:
                    store.set_receipt(
                        db, lane, exchange.exchange_id, {"delivered": True}
                    )

        await store.run(acknowledge_batch)

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
        active: ManagedSessionTurn | None = None,
    ) -> AsyncIterator[None]:
        key = SessionExecutionKey.from_manager(manager, session_id)
        async with self._lock:
            if self._closing and not admitted:
                raise RuntimeError("session execution coordinator is closing")
            lane = self._session_lanes.setdefault(key, _SessionLane())
            lane.users += 1
        try:
            async with (
                lane.lock,
                self.delivery_store(manager).session_execution(session_id),
            ):
                lane.active = active
                try:
                    yield
                finally:
                    lane.active = None
        finally:
            async with self._lock:
                lane.users -= 1
                if lane.users == 0 and self._session_lanes.get(key) is lane:
                    del self._session_lanes[key]

    async def _remove(self, turn: ManagedSessionTurn) -> None:
        async with self._lock:
            self._managed_runs.pop(id(turn), None)
            # A drain that never opened (cancelled or failed at startup) must
            # not keep later drains out; its envelopes are still pending.
            key = SessionExecutionKey.from_manager(
                turn.manager, turn.command.session_id
            )
            if self._admitted_drains.get(key) is turn:
                del self._admitted_drains[key]
        await turn.lease.release()

    async def close(self) -> None:
        async with self._lock:
            self._closing = True
            turns = tuple(self._managed_runs.values())
        if turns:
            await asyncio.gather(
                *(turn.cancel(user_requested=False) for turn in turns),
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
