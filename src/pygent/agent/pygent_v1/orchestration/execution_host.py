from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal, Self

from pygent.agent.pygent_v1.graph.prompts import PromptRegistry
from pygent.agent.pygent_v1.runtime.realtime.envelope import ExchangeEnvelope
from pygent.agent.pygent_v1.runtime.realtime.fork_reply import ForkReplyOutcome
from pygent.agent.pygent_v1.runtime.service import PygentRuntimeService
from pygent.agent.pygent_v1.schema import RunConfig
from pygent.agent.pygent_v1.sessions import SessionManager, realtime_enabled_for

from .managed_turn import ManagedSessionTurn
from .models import TurnCommand
from .runtime_pool import WorkspaceRuntimePool
from .session_execution import SessionExecutionCoordinator


@dataclass(frozen=True)
class UserDelivery:
    """Result of routing one user message through the unified entry.

    ``turn`` deliveries admit immediately; await ``turn.result()`` for the
    final answer or ``turn.subscribe()`` for the event stream. ``interrupt``
    deliveries resolve after the fork ack has been produced; ``outcome``
    carries the reply text the transport can show right away.
    """

    mode: Literal["turn", "interrupt"]
    turn: ManagedSessionTurn | None = None
    outcome: ForkReplyOutcome | None = None


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
        """Single user-message entry across both delivery paths.

        A realtime session with a turn chain on its lane delivers through the
        fork interrupt: the user gets a fast speak ack and an exchange
        envelope, which the running turn applies at its next step boundary
        (or a continuation turn drains afterwards). Everything else —
        plain sessions at any time, idle realtime sessions, and brand-new
        sessions — starts or queues a real turn. The lane state is the only
        routing signal: an in-flight fork needs no routing, its own
        supersede-and-merge state absorbs the message.
        """

        if session_id is not None:
            session = manager.load(session_id)
            if (
                realtime_enabled_for(session)
                and self.coordinator.has_active_execution(manager, session_id)
            ):
                if attachments:
                    raise ValueError(
                        "attachments are only supported on turns; the fork "
                        "interrupt path is text-only"
                    )
                lease = await self.acquire_runtime(config=config, manager=manager)
                try:
                    outcome = await lease.runtime.runtime_service.send_interrupt(
                        manager=manager,
                        session_id=session_id,
                        input_id=input_id or uuid.uuid4().hex,
                        message=message,
                        case_id=case_id,
                    )
                finally:
                    await lease.release()
                return UserDelivery(mode="interrupt", outcome=outcome)
        return UserDelivery(
            mode="turn",
            turn=await self._submit_user_turn(
                config=config,
                manager=manager,
                message=message,
                session_id=session_id,
                input_id=input_id,
                case_id=case_id,
                interactive_approvals=interactive_approvals,
                attachments=attachments,
                session_title=session_title,
                model_group_name=model_group_name,
                selected_model_key=selected_model_key,
            ),
        )

    async def _submit_user_turn(
        self,
        *,
        config: RunConfig,
        manager: SessionManager,
        message: str,
        session_id: str | None,
        input_id: str | None,
        case_id: str,
        interactive_approvals: bool,
        attachments: Sequence[str],
        session_title: str | None,
        model_group_name: str | None,
        selected_model_key: str | None,
    ) -> ManagedSessionTurn:
        Path(config.workspace_root).mkdir(parents=True, exist_ok=True)
        lease = await self.acquire_runtime(config=config, manager=manager)
        try:
            active_session_id = session_id
            if active_session_id is None:
                active_session_id = manager.create(
                    case_id,
                    mode="chat",
                    model_group_name=model_group_name,
                    selected_model_key=selected_model_key,
                ).session_id
            lease.runtime.reminders.prewarm_session(active_session_id)
            if session_title:
                manager.save_title(active_session_id, session_title)
            else:
                manager.save_title_from_user_input(active_session_id, message)
            return await self.coordinator.submit_turn(
                lease=lease,
                command=TurnCommand(
                    session_id=active_session_id,
                    message=message,
                    case_id=case_id,
                    submission_id=input_id,
                    interactive_approvals=interactive_approvals,
                    attachments=tuple(attachments),
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
        await self.runtime_pool.stop_accepting()
        if self._drain_tasks:
            await asyncio.gather(*self._drain_tasks, return_exceptions=True)
        await self.coordinator.close()
        await self.runtime_pool.close(cancel=True)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.aclose()


__all__ = ["LocalExecutionHost", "UserDelivery"]
