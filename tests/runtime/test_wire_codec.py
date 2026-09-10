from __future__ import annotations

import pytest

from pygent import (
    AIMessage,
    Context,
    Message,
    ModelContinuation,
    ToolMessage,
    UserMessage,
)
from pygent.core import FrozenJsonObject
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
        AIMessage(
            content="calling",
            tool_calls=(call,),
            usage={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
            continuation=ModelContinuation(
                provider="deepseek",
                protocol="anthropic_messages",
                data={"thinking": [{"signature": "opaque"}]},
            ),
        ),
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


def test_assistant_wire_requires_usage() -> None:
    value = message_to_dict(AIMessage(content="answer"))
    value.pop("usage")

    with pytest.raises(WireCodecError, match="invalid Message"):
        message_from_dict(value)


def test_assistant_wire_requires_explicit_continuation() -> None:
    value = message_to_dict(AIMessage(content="answer"))
    assert value["continuation"] is None
    value.pop("continuation")

    with pytest.raises(WireCodecError, match="invalid Message"):
        message_from_dict(value)


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


@pytest.mark.parametrize("message", [
    UserMessage(metadata={"nested": [{"x": 1}]}),
    AIMessage(tool_calls=(ToolCall(call_id="c", name="t", arguments={"a": [1, 2]}),), usage={"input_tokens": 3}),
    ToolMessage(results=(ToolResult(call_id="c", name="t", status="succeeded", output={"a": [1, 2]}),)),
])
def test_invocation_retains_frozen_values_and_preserves_mutable_wire_projection(message):
    from pygent.core import freeze_json_object
    from pygent.runtime._history_types import _json_frozen

    context = Context(messages=(message,), metadata={"nested": [{"x": 2}]})
    expected = freeze_json_object({"message": message_to_dict(message), "context": context_to_dict(context)})
    actual = invocation_to_dict(message, context)
    assert actual == expected
    assert _json_frozen(actual) == _json_frozen(expected)
    assert actual["message"]["metadata"] is message.metadata
    assert actual["context"]["data"]["metadata"] is context.metadata
    wire = context_to_dict(context)
    wire["data"]["metadata"]["nested"][0]["x"] = 99
    assert context.metadata["nested"][0]["x"] == 2


def test_invocation_preserves_custom_context_codec_encode_override():
    from dataclasses import dataclass, fields

    from pygent.runtime.context_codec import ContextCodec, ContextCodecRegistry

    @dataclass(frozen=True, slots=True)
    class CustomContext(Context):
        context_schema = "test.custom-projection"
        context_schema_version = 1

    class CustomCodec(ContextCodec):
        def encode(self, value):
            encoded = super().encode(value)
            encoded["metadata"]["custom"] = True
            return encoded

    base = ContextCodec.dataclass(CustomContext)
    codec = CustomCodec(**{f.name: getattr(base, f.name) for f in fields(base)})
    registry = ContextCodecRegistry((codec,))
    result = invocation_to_dict(UserMessage(), CustomContext(), registry=registry)
    assert result["context"]["data"]["metadata"]["custom"] is True


def test_internal_context_projection_keeps_type_validation():
    from dataclasses import dataclass

    from pygent.runtime.context_codec import ContextCodec, ContextCodecRegistry

    @dataclass(frozen=True, slots=True)
    class CustomContext(Context):
        context_schema = "test.projection-validation"
        context_schema_version = 1
        count: int = 1

    context = CustomContext()
    object.__setattr__(context, "count", True)
    registry = ContextCodecRegistry((ContextCodec.dataclass(CustomContext),))
    with pytest.raises(WireCodecError):
        invocation_to_dict(UserMessage(), context, registry=registry)


def test_internal_projection_preserves_frozen_subclass_conversion():
    from dataclasses import dataclass, field

    from pygent.runtime.context_codec import ContextCodec, ContextCodecRegistry

    class CustomJson(FrozenJsonObject):
        def to_dict(self):
            return {"custom": True}

    @dataclass(frozen=True, slots=True)
    class CustomContext(Context):
        context_schema = "test.json-subclass"
        context_schema_version = 1
        payload: FrozenJsonObject = field(default_factory=CustomJson)

    registry = ContextCodecRegistry((ContextCodec.dataclass(CustomContext),))
    context = CustomContext()
    expected = context_to_dict(context, registry=registry)
    actual = invocation_to_dict(UserMessage(), context, registry=registry)
    assert actual["context"].to_dict() == expected
