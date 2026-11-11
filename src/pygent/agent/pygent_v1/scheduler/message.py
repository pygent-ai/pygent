"""Scheduled-task execution message built with the core Prompt API."""

from __future__ import annotations

from datetime import UTC, datetime

from pygent.agent.pygent_v1.core.prompt import Prompt
from pygent.agent.pygent_v1.core.timefmt import format_current_time_line

SCHEDULER_GUIDANCE = (
    "You are executing a scheduled pygent_v1 task in an isolated session. "
    "Use the available tools to complete the task, then report the concrete "
    "result. Do not modify scheduler task definitions, signing keys, or the "
    "scheduler database."
)


def _indent(text: str) -> str:
    return "\n".join(f"  {line}" for line in str(text).splitlines())


def build_task_message(
    *,
    task_id: str,
    name: str,
    source: str,
    instructions: str,
    user_message: str,
    user_identity: str = "default",
    now: datetime | None = None,
) -> str:
    """Build the ``<task-context>`` message for one scheduler execution."""

    timestamp = (now or datetime.now(UTC)).astimezone(UTC)
    instruction_body = SCHEDULER_GUIDANCE
    if instructions.strip():
        instruction_body = f"{instruction_body}\n\n{instructions}"
    time_block = Prompt(f"  {format_current_time_line(timestamp)}").wrap("time")
    instructions_block = Prompt(instruction_body).wrap("instructions", indent="  ")
    body = "\n".join(
        (
            f"  {Prompt.self_closing('task', id=task_id, name=name, source=source)}",
            f"  <user-identity>{Prompt(user_identity).escape()}</user-identity>",
            _indent(str(time_block)),
            _indent(str(instructions_block)),
            f"  <user-message>{Prompt(user_message).escape()}</user-message>",
        )
    )
    return str(Prompt(body).wrap("task-context"))


__all__ = ["SCHEDULER_GUIDANCE", "build_task_message"]
