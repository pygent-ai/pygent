"""Pygent tools that submit signed scheduler tasks and events.

Tools never write ``tasks`` or ``runs`` directly. They only insert a signed
``submissions`` row; the independent daemon promotes or rejects it. Agent
authorship is represented by the agent HMAC key, so a compromised CLI writer
cannot forge an agent submission or approve a task.
"""

from __future__ import annotations

import asyncio
from typing import Any

from pygent import IdempotencyPolicy, ToolDefinition, ToolSideEffect
from pygent.lara.core.io import plain_object
from pygent.lara.schema import RunConfig
from pygent.tool import SandboxExecutorSupport, ToolSpec
from pygent.tool.executors import ToolExecutionContext

from .keys import load_secret, sign
from .schedule import parse_schedule
from .store import SchedulerStore, scheduler_db_path, submission_payload


def _spec(
    name: str,
    description: str,
    parameters: dict[str, object],
) -> ToolSpec:
    return ToolSpec(
        tool_id=f"lara.scheduler.{name}",
        version="1",
        definition=ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
        ),
        side_effect=ToolSideEffect.EXTERNAL,
        idempotency=IdempotencyPolicy.REQUIRES_KEY,
        resource_key="lara-scheduler",
        sandbox_profile="agent",
        timeout=30,
    )


SCHEDULE_TASK_TOOL_SPEC = _spec(
    "schedule_task",
    "Submit a task to the Lara scheduler for approval or execution.",
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "schedule": {
                "type": "string",
                "minLength": 1,
                "description": "every <n>[smhd], at <ISO>, daily HH:MM, or on <event>.",
            },
            "prompt": {"type": "string", "minLength": 1},
            "instructions": {
                "type": "string",
                "description": "Optional task-specific instructions.",
            },
            "session": {
                "type": "string",
                "description": "Optional scheduler session name override.",
            },
            "priority": {
                "type": "string",
                "enum": ["high", "normal", "low"],
                "description": (
                    "Bus priority; agent requests for high are normalized to normal. "
                    "Delivery is always fire-and-forget for agent submissions."
                ),
            },
        },
        "required": ["name", "schedule", "prompt"],
        "additionalProperties": False,
    },
)

EMIT_EVENT_TOOL_SPEC = _spec(
    "emit_event",
    "Emit a signed scheduler event that can trigger `on <event>` tasks.",
    {
        "type": "object",
        "properties": {
            "event": {"type": "string", "minLength": 1},
            "data": {"type": "object"},
        },
        "required": ["event"],
        "additionalProperties": False,
    },
)


SCHEDULER_TOOL_SPECS = (SCHEDULE_TASK_TOOL_SPEC, EMIT_EVENT_TOOL_SPEC)


class SchedulerToolExecutor:
    def __init__(
        self,
        config: RunConfig,
        *,
        store: SchedulerStore | None = None,
    ) -> None:
        self.config = config
        self.store = store or SchedulerStore(scheduler_db_path(config.lara_root))
        self.user_lara_root = config.user_lara_root
        self.sandbox_support = SandboxExecutorSupport(
            profiles=("agent",),
            durable_reconnect=True,
            deployment_fingerprint="lara:scheduler:v1",
        )

    async def execute(
        self, spec: ToolSpec, call: Any, context: ToolExecutionContext
    ) -> object:
        del context
        if not self.config.scheduler.agent_create_enabled:
            raise PermissionError(
                "agent scheduler tools are disabled; set scheduler.agent_create_enabled"
            )
        secret = load_secret(self.user_lara_root, "agent")
        arguments = plain_object(call.arguments)
        name = spec.definition.name
        submission_id = str(
            getattr(call, "idempotency_key", None) or getattr(call, "call_id", "")
        )
        if name == "schedule_task":
            payload = {
                "name": str(arguments.get("name") or ""),
                "schedule": str(arguments.get("schedule") or ""),
                "prompt": str(arguments.get("prompt") or ""),
                "instructions": str(arguments.get("instructions") or ""),
                "session": str(
                    arguments.get("session") or self.config.scheduler.session
                ),
                "priority": str(arguments.get("priority") or "normal"),
            }
            if not payload["name"].strip():
                raise ValueError("schedule_task.name must be non-empty")
            if not payload["prompt"].strip():
                raise ValueError("schedule_task.prompt must be non-empty")
            parse_schedule(payload["schedule"])
            signature = sign(
                submission_payload("task", "agent", payload), secret
            )
            submission = await asyncio.to_thread(
                self.store.submit,
                kind="task",
                source="agent",
                payload=payload,
                signature=signature,
                submission_id=submission_id or None,
            )
            return submission.to_dict()
        if name == "emit_event":
            event_payload: dict[str, Any] = {
                "event": str(arguments.get("event") or ""),
                "data": arguments.get("data") or {},
            }
            if not event_payload["event"].strip():
                raise ValueError("emit_event.event must be non-empty")
            signature = sign(
                submission_payload("event", "agent", event_payload), secret
            )
            submission = await asyncio.to_thread(
                self.store.submit,
                kind="event",
                source="agent",
                payload=event_payload,
                signature=signature,
                submission_id=submission_id or None,
            )
            return submission.to_dict()
        raise ValueError(f"unsupported scheduler tool {name!r}")


__all__ = [
    "EMIT_EVENT_TOOL_SPEC",
    "SCHEDULER_TOOL_SPECS",
    "SCHEDULE_TASK_TOOL_SPEC",
    "SchedulerToolExecutor",
]
