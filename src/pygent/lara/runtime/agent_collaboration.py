from __future__ import annotations

from pygent import IdempotencyPolicy, ToolDefinition, ToolSideEffect
from pygent.lara.schema import RunConfig
from pygent.tool import ToolSpec


def _spec(
    name: str,
    description: str,
    parameters: dict[str, object],
    *,
    side_effect: ToolSideEffect,
    idempotency: IdempotencyPolicy,
) -> ToolSpec:
    return ToolSpec(
        tool_id=f"lara.agent.{name}",
        version="1",
        definition=ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
        ),
        side_effect=side_effect,
        idempotency=idempotency,
        resource_key="lara-agent-collaboration",
        sandbox_profile="agent",
    )


AGENT_START_TOOL_SPEC = _spec(
    "agent_start",
    "Start an allowed Lara agent in a new session and return immediately with its operation and session IDs.",
    {
        "type": "object",
        "properties": {
            "agent": {
                "type": "string",
                "description": "Configured Lara agent alias allowed for collaboration.",
            },
            "task": {
                "type": "string",
                "minLength": 1,
                "description": "Complete task for the new agent session.",
            },
            "model_group": {
                "type": "string",
                "description": (
                    "Optional model group from the target agent's configuration "
                    "to run the new session with."
                ),
            },
        },
        "required": ["agent", "task"],
        "additionalProperties": False,
    },
    side_effect=ToolSideEffect.EXTERNAL,
    idempotency=IdempotencyPolicy.REQUIRES_KEY,
)

AGENT_SEND_TOOL_SPEC = _spec(
    "agent_send",
    "Queue a message for any registered parent or child agent session, including sessions in other projects, without starting a new turn.",
    {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": "Target session ID; may belong to another project.",
            },
            "message": {
                "type": "string",
                "minLength": 1,
                "description": "Message to deliver at the target agent's next tool boundary.",
            },
        },
        "required": ["session_id", "message"],
        "additionalProperties": False,
    },
    side_effect=ToolSideEffect.EXTERNAL,
    idempotency=IdempotencyPolicy.REQUIRES_KEY,
)

_COLLABORATION_ID_PARAMETER = {
    "type": "string",
    "description": "Operation ID or Agent-message ID returned by an Agent collaboration tool.",
}

AGENT_STATUS_TOOL_SPEC = _spec(
    "agent_status",
    "Read the current state of a related Agent operation or message.",
    {
        "type": "object",
        "properties": {"collaboration_id": _COLLABORATION_ID_PARAMETER},
        "required": ["collaboration_id"],
        "additionalProperties": False,
    },
    side_effect=ToolSideEffect.READ,
    idempotency=IdempotencyPolicy.INHERENT,
)

AGENT_LIST_TOOL_SPEC = _spec(
    "agent_list",
    "List the most recent Agent collaboration operations related to this session.",
    {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
    side_effect=ToolSideEffect.READ,
    idempotency=IdempotencyPolicy.INHERENT,
)

AGENT_WAIT_TOOL_SPEC = _spec(
    "agent_wait",
    "Wait until related Agent work is ready. Provide exactly one of collaboration_id or collaboration_ids.",
    {
        "type": "object",
        "properties": {
            "collaboration_id": _COLLABORATION_ID_PARAMETER,
            "collaboration_ids": {
                "type": "array",
                "items": _COLLABORATION_ID_PARAMETER,
                "minItems": 1,
                "maxItems": 32,
                "uniqueItems": True,
                "description": "Operation or Agent-message IDs; returns when any item is ready.",
            },
            "timeout_seconds": {
                "type": "number",
                "exclusiveMinimum": 0,
                "maximum": 120,
                "default": 30,
                "description": "Maximum wait; a timeout returns the latest state instead of failing.",
            },
        },
        "additionalProperties": False,
    },
    side_effect=ToolSideEffect.READ,
    idempotency=IdempotencyPolicy.INHERENT,
)

AGENT_COLLABORATION_TOOL_SPECS = (
    AGENT_START_TOOL_SPEC,
    AGENT_SEND_TOOL_SPEC,
    AGENT_STATUS_TOOL_SPEC,
    AGENT_LIST_TOOL_SPEC,
    AGENT_WAIT_TOOL_SPEC,
)


def visible_agent_collaboration_specs(
    config: RunConfig, *, collaboration_available: bool
) -> tuple[ToolSpec, ...]:
    if not collaboration_available:
        return ()
    if not config.delegation.allowed_agents:
        return tuple(
            spec
            for spec in AGENT_COLLABORATION_TOOL_SPECS
            if spec is not AGENT_START_TOOL_SPEC
        )
    return AGENT_COLLABORATION_TOOL_SPECS


__all__ = [
    "AGENT_COLLABORATION_TOOL_SPECS",
    "AGENT_LIST_TOOL_SPEC",
    "AGENT_SEND_TOOL_SPEC",
    "AGENT_START_TOOL_SPEC",
    "AGENT_STATUS_TOOL_SPEC",
    "AGENT_WAIT_TOOL_SPEC",
    "visible_agent_collaboration_specs",
]
