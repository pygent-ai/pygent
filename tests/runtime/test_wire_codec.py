from __future__ import annotations

import pytest

from pygent import AIMessage, Context, Message, ToolMessage, UserMessage
from pygent.runtime.codec import (
    WireCodecError,
    context_from_dict,
    context_to_dict,
    invocation_from_dict,
    invocation_to_dict,
    message_from_dict,
    message_to_dict,
)
from pygent.tool import (
    IdempotencyPolicy,
    ToolAuthorizationDecision,
    ToolAuthorizationRequest,
    ToolCall,
    ToolDefinition,
    ToolResult,
    ToolSideEffect,
    ToolSpec,
    ToolTask,
    ToolTaskState,
)


def test_message_and_context_wire_round_trip_all_public_variants():
    definition = ToolDefinition(
        name="lookup",
        description="Lookup a value",
        parameters={"type": "object"},
    )
    call = ToolCall(call_id="call-1", name="lookup", arguments={"id": 1})
    task = ToolTask(
        task_id="task-1",
        call_id="call-1",
        tool_id="lookup",
        version="1",
        state=ToolTaskState.SUCCEEDED,
    )
    values = (
        Message(
            kind="approval.requested",
            content="Review this operation",
            data={"operation_id": "op-1", "scopes": ["publish"]},
            slot="approval/current",
        ),
        UserMessage(content="hello", metadata={"request": 1}),
        AIMessage(content="calling", tool_calls=(call,)),
        ToolMessage(
            results=(
                ToolResult(
                    call_id="call-1",
                    name="lookup",
                    status="succeeded",
                    task=task,
                    output={"value": 2},
                ),
            )
        ),
    )
    context = Context(
        system_prompt="system",
        messages=values,
        tools=(definition,),
        metadata={"tenant": "one"},
    )

    assert tuple(message_from_dict(message_to_dict(item)) for item in values) == values
    assert context_from_dict(context_to_dict(context)) == context
    assert invocation_from_dict(invocation_to_dict(values[0], context)) == (
        values[0],
        context,
    )


def test_authorization_messages_and_context_round_trip_without_field_loss():
    spec = ToolSpec(
        tool_id="weather.lookup",
        version="2",
        definition=ToolDefinition(
            name="weather.lookup",
            description="Weather",
            parameters={"type": "object"},
        ),
        side_effect=ToolSideEffect.READ,
        idempotency=IdempotencyPolicy.REQUIRES_KEY,
        timeout=3.0,
        resource_key="weather-api",
        sandbox_profile="network-read",
        required_permissions=("weather:read",),
    )
    call = ToolCall(
        call_id="call-1",
        name="weather.lookup",
        arguments={"city": "Beijing"},
        tool_id="weather.lookup",
        tool_version="2",
        idempotency_key="stable-key",
    )
    request = ToolAuthorizationRequest(
        call=call,
        spec=spec,
        permissions=("weather:read",),
        metadata={"tenant": "one"},
    )
    decision = ToolAuthorizationDecision(
        call_id="call-1",
        allowed=True,
        reason_code="allowed",
        lifecycle="detach",
        metadata={"policy": "v2"},
    )
    context = Context(messages=(request, decision))

    assert message_from_dict(message_to_dict(request)) == request
    assert message_from_dict(message_to_dict(decision)) == decision
    assert context_from_dict(context_to_dict(context)) == context


def test_wire_codec_accepts_legacy_message_without_domain_fields():
    restored = message_from_dict(
        {"role": "message", "content": "legacy", "slot": None, "metadata": {}}
    )

    assert restored == Message(content="legacy")


@pytest.mark.parametrize(
    "payload",
    [
        {"role": "admin", "content": "x", "slot": None, "metadata": {}},
        {
            "role": "user",
            "content": "x",
            "slot": None,
            "metadata": {},
            "unknown": True,
        },
    ],
)
def test_wire_codec_rejects_unknown_roles_and_fields(payload):
    with pytest.raises(WireCodecError):
        message_from_dict(payload)
