from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path

from pygent.lara.schema import RunConfig
from pygent.lara.sessions import SessionManager

from .managed_turn import ManagedSessionTurn
from .models import TurnCommand
from .runtime_lease import RuntimeLease
from .session_execution import SessionExecutionCoordinator

RuntimeAcquirer = Callable[..., Awaitable[RuntimeLease]]


class SessionTurnService:
    """Submit ordinary session turns without transport-specific behavior."""

    def __init__(
        self,
        *,
        coordinator: SessionExecutionCoordinator,
        acquire_runtime: RuntimeAcquirer,
    ) -> None:
        self.coordinator = coordinator
        self._acquire_runtime = acquire_runtime

    async def submit(
        self,
        *,
        config: RunConfig,
        manager: SessionManager,
        message: str,
        session_id: str | None = None,
        case_id: str = "chat",
        turn_id: str | None = None,
        submission_id: str | None = None,
        interactive_approvals: bool = True,
        message_kind: str = "lara.chat.turn",
        message_data: dict[str, object] | None = None,
        session_title: str | None = None,
        model_group_name: str | None = None,
        selected_model_key: str | None = None,
        attachments: Sequence[str] = (),
    ) -> ManagedSessionTurn:
        Path(config.workspace_root).mkdir(parents=True, exist_ok=True)
        lease = await self._acquire_runtime(config=config, manager=manager)
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
            elif message_kind == "lara.chat.turn":
                manager.save_title_from_user_input(active_session_id, message)
            return await self.coordinator.submit_turn(
                lease=lease,
                command=TurnCommand(
                    session_id=active_session_id,
                    message=message,
                    case_id=case_id,
                    turn_id=turn_id,
                    submission_id=submission_id,
                    interactive_approvals=interactive_approvals,
                    message_kind=message_kind,
                    message_data=dict(message_data or {}),
                    session_title=session_title,
                    attachments=tuple(attachments),
                ),
            )
        except BaseException:
            await lease.release()
            raise

    async def prewarm_session(
        self,
        *,
        config: RunConfig,
        manager: SessionManager,
        session_id: str,
    ) -> None:
        lease = await self._acquire_runtime(config=config, manager=manager)
        try:
            lease.runtime.reminders.prewarm_session(session_id)
        finally:
            await lease.release()


__all__ = ["RuntimeAcquirer", "SessionTurnService"]
