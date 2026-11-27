from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal, Self

from pygent.agent.pygent_v1.core.io import validate_path_id
from pygent.agent.pygent_v1.graph.prompts import PromptRegistry
from pygent.agent.pygent_v1.runtime.delivery_store import DeliveryStore
from pygent.agent.pygent_v1.runtime.realtime.envelope import ExchangeEnvelope
from pygent.agent.pygent_v1.runtime.realtime.fork_agent import outcome_from_result
from pygent.agent.pygent_v1.runtime.realtime.fork_reply import ForkReplyOutcome
from pygent.agent.pygent_v1.runtime.service import PygentRuntimeService
from pygent.agent.pygent_v1.schema import CaseRunRef, RunConfig
from pygent.agent.pygent_v1.sessions import SessionManager, realtime_enabled_for
from pygent.core import AIMessage
from pygent.runtime.codec import message_from_dict, message_to_dict

from .managed_turn import ManagedSessionTurn
from .models import TurnCommand, TurnState
from .runtime_pool import WorkspaceRuntimePool
from .session_execution import SessionExecutionCoordinator, transcript_contains_input

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UserDelivery:
    """Result of routing one user message through the unified entry.

    ``turn`` deliveries admit immediately; await ``turn.result()`` for the
    final answer or ``turn.subscribe()`` for the event stream. ``interrupt``
    deliveries resolve after the fork ack has been produced; ``outcome``
    carries the reply text the transport can show right away.
    """

    mode: Literal["turn", "interrupt", "steering"]
    turn: ManagedSessionTurn | None = None
    outcome: ForkReplyOutcome | None = None
    input_id: str | None = None


class LocalExecutionHost:
    """Short-lived composition root for non-server execution adapters."""

    def __init__(
        self,
        *,
        runtime_pool: WorkspaceRuntimePool | None = None,
        coordinator: SessionExecutionCoordinator | None = None,
        credential_environ: Mapping[str, str] | None = None,
        run_config_loader: Callable[..., RunConfig] | None = None,
        prompt_registry: PromptRegistry | None = None,
    ) -> None:
        self.coordinator = coordinator or SessionExecutionCoordinator()
        self.runtime_pool = runtime_pool or WorkspaceRuntimePool(
            runtime_factory=self._build_runtime,
            credential_environ=credential_environ,
            run_config_loader=run_config_loader,
            prompt_registry=prompt_registry,
        )
        self._drain_tasks: set[asyncio.Task[None]] = set()
        self._fork_receipt_tasks: set[asyncio.Task[None]] = set()
        self._routes: dict[tuple[Path, str, str], asyncio.Task[UserDelivery]] = {}
        self._recovery_tasks: dict[Path, asyncio.Task[None]] = {}
        self._recovery_configs: dict[Path, tuple[RunConfig, SessionManager]] = {}
        self._active_inputs: dict[tuple[Path, str, str], UserDelivery] = {}
        self._owner = uuid.uuid4().hex
        self._closed = False

    def _build_runtime(self, config: RunConfig, **kwargs: Any) -> PygentRuntimeService:
        # Pooled runtimes reach the coordinator through this hook: the fork
        # reply's exchange goes to the turn running on the session lane, or
        # waits for the next turn — which an idle session must start itself
        # (see SessionExecutionCoordinator.submit_pending_drain).
        return PygentRuntimeService(
            config,
            on_exchange_ready=partial(self._deliver_exchange, config),
            **kwargs,
        )

    async def _deliver_exchange(
        self, config: RunConfig, envelope: ExchangeEnvelope
    ) -> None:
        manager = SessionManager(config)
        if await self.coordinator.deliver_exchange(manager, envelope) == "injected":
            return
        # The drain runs detached: send_interrupt must not block on a lease
        # or on the continuation turn, which needs a task generation of its own.
        task = asyncio.create_task(self._drain_pending(config, envelope.session_id))
        self._drain_tasks.add(task)
        task.add_done_callback(self._drain_tasks.discard)

    async def _drain_pending(self, config: RunConfig, session_id: str) -> None:
        lease = await self.acquire_runtime(config=config)
        turn: Any = None
        try:
            turn = await self.coordinator.submit_pending_drain(
                lease=lease, session_id=session_id
            )
        finally:
            # A submitted turn carries the lease and releases it when its drive
            # finishes; an unused lease is returned right away.
            if turn is None:
                await lease.release()

    async def deliver_user_message(
        self,
        *,
        config: RunConfig,
        manager: SessionManager,
        message: str,
        session_id: str | None = None,
        input_id: str | None = None,
        case_id: str = "chat",
        interactive_approvals: bool = True,
        attachments: Sequence[str] = (),
        session_title: str | None = None,
        model_group_name: str | None = None,
        selected_model_key: str | None = None,
    ) -> UserDelivery:
        """Persist the input before resolving its live delivery route."""
        if self._closed:
            raise RuntimeError("execution host is closing")
        if not case_id or (not message and not attachments):
            raise ValueError(
                "delivery requires a case and either a message or attachments"
            )
        validate_path_id(case_id, "case_id")
        input_id = input_id or uuid.uuid4().hex
        store = self.coordinator.delivery_store(manager)
        with store.transaction() as db:
            if session_id is None:
                # A retried initial request must return to the session it created.
                db.execute("BEGIN IMMEDIATE")
                initial = db.execute(
                    "SELECT payload FROM deliveries WHERE lane='new' AND key=?",
                    (input_id,),
                ).fetchone()
                if initial is None:
                    session_id = manager.create(
                        case_id,
                        mode="chat",
                        model_group_name=model_group_name,
                        selected_model_key=selected_model_key,
                    ).session_id
                    store.put(db, "new", input_id, {"session_id": session_id})
                    store.set_receipt(db, "new", input_id, {"created": True})
                else:
                    session_id = json.loads(initial["payload"])["session_id"]
            else:
                manager.load(session_id)
            assert session_id is not None
            lane = f"user:{session_id}"
            payload = {
                "message": message,
                "case_id": case_id,
                "interactive_approvals": interactive_approvals,
                "attachments": list(attachments),
                "session_title": session_title,
            }
            store.put(db, lane, input_id, payload)
            if json.loads(store.read(db, lane, input_id)["payload"]) != payload:
                raise ValueError(
                    "input_id is already associated with a different message"
                )

        await self.recover_pending_messages(config=config, manager=manager)
        active = self._active_inputs.get((store.path, lane, input_id))
        if active is not None:
            return active
        return await asyncio.shield(
            self._route_task(config, manager, store, lane, input_id)
        )

    async def recover_pending_messages(
        self, *, config: RunConfig, manager: SessionManager
    ) -> None:
        """Resume a workspace's delivery journal when its owning host starts."""
        if self._closed:
            return
        store = self.coordinator.delivery_store(manager)
        first_recovery = store.path not in self._recovery_tasks
        self._recovery_configs[store.path] = (config, manager)
        if first_recovery:
            try:
                await self._recover_exchange_turns(config, manager, store)
            except Exception:
                _logger.exception("Could not reattach foreground delivery targets")
            if store.path not in self._recovery_tasks and not self._closed:
                self._recovery_tasks[store.path] = asyncio.create_task(
                    self._recover_deliveries(store),
                    name="pygent-message-recovery",
                )
        for row in store.pending("user:", prefix=True, available=True):
            self._route_task(config, manager, store, row["lane"], row["key"])

    async def _recover_deliveries(self, store: DeliveryStore) -> None:
        while True:
            try:
                config, manager = self._recovery_configs[store.path]
                await self._recover_exchange_turns(config, manager, store)
                if store.pending("speak:", prefix=True, available=True):
                    lease = await self.acquire_runtime(config=config, manager=manager)
                    await lease.release()
                for row in store.pending("user:", prefix=True, available=True):
                    self._route_task(config, manager, store, row["lane"], row["key"])
                sessions = {
                    row["lane"].removeprefix("exchange:")
                    for row in store.pending("exchange:", prefix=True, available=True)
                }
                for session_id in sessions:
                    if not self.coordinator.has_active_execution(manager, session_id):
                        await self._drain_pending(config, session_id)
            except Exception:
                _logger.exception("Could not drain session deliveries")
            await asyncio.sleep(1)

    async def _recover_exchange_turns(
        self, config: RunConfig, manager: SessionManager, store: DeliveryStore
    ) -> None:
        pending = store.pending("recover:", prefix=True)
        if not pending:
            return
        lease = await self.acquire_runtime(config=config, manager=manager)
        try:
            history = getattr(lease.runtime.runtime_service, "history", None)
            if history is None:
                return
            for row in pending:
                session_id = row["lane"].removeprefix("recover:")
                if self.coordinator.has_active_execution(manager, session_id):
                    continue
                kind, identity = row["key"].split(":", 1)
                if kind == "request":
                    stored = await history.get_execution_by_request(identity)
                else:
                    stored = await history.get_execution(identity)
                if stored is None:
                    targets = store.pending(f"exchange:{session_id}")
                    if not any(target["target"] == row["key"] for target in targets):
                        store.finish(row["lane"], row["key"], {"committed": False})
                    continue
                turn = await self._existing_turn(config, manager, stored.execution_id)
                if turn is not None and turn.done:
                    store.finish(row["lane"], row["key"], {"status": turn.status})
        finally:
            await lease.release()

    def _route_task(
        self,
        config: RunConfig,
        manager: SessionManager,
        store: DeliveryStore,
        lane: str,
        input_id: str,
    ) -> asyncio.Task[UserDelivery]:
        key = (store.path, lane, input_id)
        job = self._routes.get(key)
        if job is None:
            job = asyncio.create_task(
                self._route_input(config, manager, store, lane, input_id),
                name=f"pygent-delivery:{input_id}",
            )
            self._routes[key] = job

            def done(task: asyncio.Task[UserDelivery]) -> None:
                if self._routes.get(key) is task:
                    self._routes.pop(key)
                if not task.cancelled():
                    error = task.exception()
                    if error is not None:
                        _logger.warning(
                            "Session delivery failed for %s: %s", input_id, error
                        )

            job.add_done_callback(done)
        return job

    async def _route_input(
        self,
        config: RunConfig,
        manager: SessionManager,
        store: DeliveryStore,
        lane: str,
        input_id: str,
    ) -> UserDelivery:
        session_id = lane.removeprefix("user:")
        while True:
            current = store.get(lane, input_id)
            if current["receipt"] is not None:
                return await self._read_receipt(
                    config,
                    manager,
                    session_id,
                    input_id,
                    json.loads(current["receipt"]),
                )
            row = store.claim(lane, input_id, self._owner, 30)
            if row is not None:
                break
            await asyncio.sleep(0.1)
        renew = asyncio.create_task(self._renew_route(store, lane, input_id))
        try:
            payload = json.loads(row["payload"])
            target = row["target"]
            turn = None
            if target and target.startswith(("turn:", "steering:")):
                mode, execution_id = target.split(":", 1)
                if mode == "turn":
                    turn = await self._existing_turn(config, manager, execution_id)
                else:
                    lease = await self.acquire_runtime(config=config, manager=manager)
                    try:
                        history = getattr(
                            lease.runtime.runtime_service, "history", None
                        )
                        if history is not None and await history.has_execution_input(
                            execution_id, input_id
                        ):
                            turn = await self._existing_turn(
                                config, manager, execution_id
                            )
                    finally:
                        await lease.release()
                    if turn is None and transcript_contains_input(
                        manager, session_id, input_id
                    ):
                        receipt = {"mode": "steering", "execution_id": execution_id}
                        store.finish(lane, input_id, receipt)
                        return await self._read_receipt(
                            config, manager, session_id, input_id, receipt
                        )
                if turn is not None:
                    if mode == "turn":
                        self._watch_turn(store, lane, input_id, turn)
                    else:
                        store.finish(
                            lane, input_id, {"mode": mode, "execution_id": execution_id}
                        )
                    return UserDelivery(mode=mode, turn=turn, input_id=input_id)

            session = manager.load(session_id)
            if target is None and not realtime_enabled_for(session):

                async def select_steering(execution_id: str) -> None:
                    store.set_target(lane, input_id, f"steering:{execution_id}")

                try:
                    steered = await self._try_steering(
                        manager, session_id, input_id, payload, select_steering
                    )
                except ValueError as exc:
                    store.finish(
                        lane, input_id, {"mode": "rejected", "error": str(exc)}
                    )
                    raise
                if steered is not None:
                    history = getattr(steered.runtime_service, "history", None)
                    if history is None or await history.has_execution_input(
                        steered.execution_id, input_id
                    ):
                        store.finish(
                            lane,
                            input_id,
                            {
                                "mode": "steering",
                                "execution_id": steered.execution_id,
                            },
                        )
                    else:
                        self._watch_turn(
                            store, lane, input_id, steered, mode="steering"
                        )
                    return UserDelivery("steering", turn=steered, input_id=input_id)

            if target == "interrupt" or (
                target is None
                and realtime_enabled_for(session)
                and self.coordinator.has_active_execution(manager, session_id)
            ):
                store.set_target(lane, input_id, "interrupt")
                lease = await self.acquire_runtime(config=config, manager=manager)
                service = lease.runtime.runtime_service
                try:
                    outcome = await service.send_interrupt(
                        manager=manager,
                        session_id=session_id,
                        input_id=input_id,
                        message=payload["message"],
                        case_id=payload["case_id"],
                        attachments=payload["attachments"],
                    )
                finally:
                    await lease.release()
                if outcome is not None:
                    store.finish(
                        lane,
                        input_id,
                        {
                            "mode": "interrupt",
                            "reply": message_to_dict(outcome.message),
                            "reply_owner": input_id,
                        },
                    )
                elif getattr(service, "owns_interrupt", lambda *_: False)(
                    session_id, input_id
                ):
                    watcher = asyncio.create_task(
                        self._await_fork_receipt(store, lane, input_id, service),
                        name=f"pygent-fork-receipt:{input_id}",
                    )
                    self._fork_receipt_tasks.add(watcher)
                    watcher.add_done_callback(self._fork_receipt_tasks.discard)
                else:
                    store.retry(lane, input_id, self._owner, delay=2)
                return UserDelivery("interrupt", outcome=outcome, input_id=input_id)

            execution_id = (
                target.removeprefix("turn:")
                if target and target.startswith("turn:")
                else uuid.uuid5(
                    uuid.NAMESPACE_URL, json.dumps([str(store.path), lane, input_id])
                ).hex
            )
            store.set_target(lane, input_id, f"turn:{execution_id}")
            turn = await self._submit_user_turn(
                config=config,
                manager=manager,
                session_id=session_id,
                input_id=input_id,
                execution_id=execution_id,
                **payload,
            )
            self._watch_turn(store, lane, input_id, turn)
            return UserDelivery("turn", turn=turn, input_id=input_id)
        except BaseException:
            store.retry(lane, input_id, self._owner, delay=2)
            raise
        finally:
            renew.cancel()
            with suppress(asyncio.CancelledError):
                await renew

    async def _await_fork_receipt(
        self,
        store: DeliveryStore,
        lane: str,
        input_id: str,
        service: PygentRuntimeService,
    ) -> None:
        # A superseded request returned to its caller, but its raw input still
        # belongs to the successor fork until the shared exchange commits.
        session_id = lane.removeprefix("user:")
        try:
            while service.owns_interrupt(session_id, input_id):
                if not store.renew(lane, input_id, self._owner, 30):
                    return
                await asyncio.sleep(5)
        finally:
            store.retry(lane, input_id, self._owner)

    async def _renew_route(self, store: DeliveryStore, lane: str, key: str) -> None:
        while True:
            await asyncio.sleep(10)
            if not store.renew(lane, key, self._owner, 30):
                return

    def _watch_turn(
        self,
        store: DeliveryStore,
        lane: str,
        input_id: str,
        turn: ManagedSessionTurn,
        *,
        mode: Literal["turn", "steering"] = "turn",
    ) -> None:
        key = (store.path, lane, input_id)
        delivery = UserDelivery(mode, turn=turn, input_id=input_id)
        self._active_inputs[key] = delivery
        watcher = asyncio.create_task(
            self._settle_turn(store, lane, input_id, turn, mode),
            name=f"pygent-delivery-handoff:{input_id}",
        )
        self._drain_tasks.add(watcher)

        def done(task: asyncio.Task[None]) -> None:
            self._drain_tasks.discard(task)
            if self._active_inputs.get(key) is delivery:
                self._active_inputs.pop(key)
            if not task.cancelled() and task.exception() is not None:
                _logger.warning("Could not record turn receipt: %s", task.exception())

        watcher.add_done_callback(done)

    async def _settle_turn(
        self,
        store: DeliveryStore,
        lane: str,
        input_id: str,
        turn: ManagedSessionTurn,
        mode: Literal["turn", "steering"],
    ) -> None:
        try:
            while not turn.done:
                try:
                    await asyncio.wait_for(turn.wait_finalized(), 10)
                except TimeoutError:
                    store.renew(lane, input_id, self._owner, 30)
            if turn.execution_id is None and not turn.user_cancelled:
                return
            if (
                mode == "steering"
                and not turn.user_cancelled
                and not transcript_contains_input(
                    turn.manager, turn.command.session_id, input_id
                )
            ):
                return
            receipt: dict[str, Any] = {
                "mode": mode,
                "execution_id": turn.execution_id,
                "status": turn.status,
                "run_ref": turn.run_ref.to_dict() if turn.run_ref else None,
            }
            if isinstance(turn.output, AIMessage):
                receipt["output"] = message_to_dict(turn.output)
            if turn.startup_error is not None:
                receipt["error"] = str(turn.startup_error)
            store.finish(lane, input_id, receipt)
        finally:
            store.retry(lane, input_id, self._owner, delay=2 if not self._closed else 0)

    async def _try_steering(
        self,
        manager: SessionManager,
        session_id: str,
        input_id: str,
        payload: dict[str, Any],
        before_send: Callable[[str], Awaitable[None]],
    ) -> ManagedSessionTurn | None:
        del manager, session_id, input_id, payload, before_send
        return None

    async def _read_receipt(
        self,
        config: RunConfig,
        manager: SessionManager,
        session_id: str,
        input_id: str,
        receipt: dict[str, Any],
    ) -> UserDelivery:
        mode = receipt["mode"]
        if mode == "rejected":
            raise ValueError(receipt["error"])
        if mode == "interrupt":
            outcome = None
            if receipt["reply"] is not None and receipt["reply_owner"] == input_id:
                message = message_from_dict(receipt["reply"])
                if isinstance(message, AIMessage):
                    outcome = outcome_from_result(message)
            return UserDelivery("interrupt", outcome=outcome, input_id=input_id)
        execution_id = receipt["execution_id"]
        turn = (
            await self._existing_turn(config, manager, execution_id)
            if execution_id is not None
            else None
        )
        if turn is None:
            payload = json.loads(
                self.coordinator.delivery_store(manager).get(
                    f"user:{session_id}", input_id
                )["payload"]
            )
            turn = ManagedSessionTurn(
                lease=None,
                command=TurnCommand(
                    session_id=session_id,
                    message=payload["message"],
                    attachments=tuple(payload["attachments"]),
                ),
                run_ref=CaseRunRef.from_dict(receipt["run_ref"])
                if receipt.get("run_ref")
                else None,
            )
            turn.state = TurnState(receipt.get("status", "passed"))
            if "output" in receipt:
                turn.output = message_from_dict(receipt["output"])
            if "error" in receipt:
                turn.startup_error = RuntimeError(receipt["error"])
            turn.ready.set()
            turn.finalized.set()
        return UserDelivery(mode, turn=turn, input_id=input_id)

    async def _existing_turn(
        self, config: RunConfig, manager: SessionManager, execution_id: str
    ) -> ManagedSessionTurn | None:
        active = await self.coordinator.find_delivery_turn(execution_id)
        if active is not None:
            return active
        lease = await self.acquire_runtime(config=config, manager=manager)
        transfer = False
        try:
            service = lease.runtime.runtime_service
            if not hasattr(service, "history"):
                return None
            stored = await service.history.get_execution(execution_id)
            if stored is None:
                return None
            run_ref = await service.recovery_case_run(execution_id)
            handle = await service.runtime.get_execution_handle(execution_id)
            if not handle.status.terminal:
                turn = await self.coordinator.recover_turn(
                    lease=lease, run_ref=run_ref, execution_id=execution_id
                )
                transfer = True
                return turn
            turn = ManagedSessionTurn(
                lease=lease,
                command=TurnCommand(
                    session_id=run_ref.session_id, message="previous delivery"
                ),
                run_ref=run_ref,
                execution_handle=handle,
            )
            try:
                turn.output, turn.output_context = await handle.result()
                turn.state = TurnState.PASSED
            except Exception as exc:  # noqa: BLE001 - expose the original stored error
                turn.startup_error = exc
                turn.state = TurnState.ERROR
            turn.ready.set()
            turn.finalized.set()
            return turn
        finally:
            if not transfer:
                await lease.release()

    async def _submit_user_turn(
        self,
        *,
        config: RunConfig,
        manager: SessionManager,
        message: str,
        session_id: str,
        input_id: str,
        execution_id: str,
        case_id: str,
        interactive_approvals: bool,
        attachments: Sequence[str],
        session_title: str | None,
    ) -> ManagedSessionTurn:
        Path(config.workspace_root).mkdir(parents=True, exist_ok=True)
        lease = await self.acquire_runtime(config=config, manager=manager)
        try:
            lease.runtime.reminders.prewarm_session(session_id)
            if session_title:
                manager.save_title(session_id, session_title)
            else:
                manager.save_title_from_user_input(session_id, message)
            return await self.coordinator.submit_turn(
                lease=lease,
                command=TurnCommand(
                    session_id=session_id,
                    message=message,
                    case_id=case_id,
                    submission_id=input_id,
                    interactive_approvals=interactive_approvals,
                    attachments=tuple(attachments),
                    message_data={
                        "delivery_execution_id": execution_id,
                        "input_id": input_id,
                        "interactive_approvals": interactive_approvals,
                    },
                ),
            )
        except BaseException:
            await lease.release()
            raise

    async def acquire_runtime(
        self,
        *,
        config: RunConfig,
        manager: SessionManager | None = None,
    ) -> Any:
        return await self.runtime_pool.acquire(config=config, manager=manager)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        recovery = tuple(self._recovery_tasks.values())
        routes = tuple(self._routes.values())
        forks = tuple(self._fork_receipt_tasks)
        for task in (*recovery, *routes, *forks):
            task.cancel()
        await asyncio.gather(*recovery, *routes, *forks, return_exceptions=True)
        await self.runtime_pool.stop_accepting()
        await self.coordinator.close()
        if self._drain_tasks:
            await asyncio.gather(*self._drain_tasks, return_exceptions=True)
        await self.runtime_pool.close(cancel=True)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.aclose()


__all__ = ["LocalExecutionHost", "UserDelivery"]
