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
    MediaBlock,
    MediaSource,
    ToolAuthorizationDecision,
    ToolAuthorizationRequest,
    ToolCall,
    ToolDefinition,
    ToolResult,
    ToolResultJson,
    ToolResultText,
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
        UserMessage(
            content="hello",
            media=(
                MediaBlock(
                    media_type="image",
                    mime_type="image/png",
                    source=MediaSource.resource(
                        "media://user-image",
                        sha256="1" * 64,
                        size_bytes=12,
                    ),
                    width=640,
                    height=480,
                ),
            ),
            metadata={"request": 1},
        ),
        AIMessage(
            content="calling",
            tool_calls=(call,),
            usage={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
            continuation=ModelContinuation(
                model_key="main",
                provider="deepseek",
                model_id="deepseek-reasoner",
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
                    content=(
                        ToolResultText("image"),
                        ToolResultJson({"value": 2}),
                        MediaBlock(
                            media_type="image",
                            mime_type="image/png",
                            source=MediaSource.resource(
                                "media://image-1",
                                sha256="0" * 64,
                                size_bytes=12,
                            ),
                            width=640,
                            height=480,
                        ),
                    ),
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


def test_media_wire_reader_accepts_legacy_blocks_without_dimensions() -> None:
    message = ToolMessage(
        results=(
            ToolResult(
                call_id="call-1",
                name="lookup",
                status="succeeded",
                content=(
                    MediaBlock(
                        media_type="image",
                        mime_type="image/png",
                        source=MediaSource.resource("media://legacy"),
                    ),
                ),
            ),
        )
    )
    value = message_to_dict(message)
    media = value["results"][0]["content"][0]
    assert media["width"] is None
    assert media["height"] is None
    for field in ("width", "height", "duration_seconds", "fps", "has_audio"):
        media.pop(field)

    assert message_from_dict(value) == message


def test_user_media_wire_round_trip_and_legacy_payload_without_media() -> None:
    message = UserMessage(
        content="what is this",
        media=(
            MediaBlock(
                media_type="image",
                mime_type="image/png",
                source=MediaSource.resource(
                    "media://user-image", sha256="a" * 64, size_bytes=12
                ),
                width=64,
                height=48,
            ),
        ),
    )
    value = message_to_dict(message)
    assert value["media"][0]["type"] == "media"
    assert message_from_dict(value) == message

    legacy = {key: item for key, item in value.items() if key != "media"}
    assert message_from_dict(legacy) == UserMessage(content="what is this")


def test_user_media_wire_rejects_non_media_blocks() -> None:
    value = message_to_dict(UserMessage(content="hello"))
    value["media"] = [{"type": "text", "text": "not media"}]
    with pytest.raises(WireCodecError):
        message_from_dict(value)


def test_video_media_metadata_round_trips_on_the_wire() -> None:
    video = MediaBlock(
        media_type="video",
        mime_type="video/mp4",
        source=MediaSource.resource("media://video"),
        width=1920,
        height=1080,
        duration_seconds=2.5,
        fps=24,
        has_audio=True,
    )
    message = ToolMessage(
        results=(
            ToolResult(
                call_id="video",
                name="read",
                status="succeeded",
                content=(video,),
            ),
        )
    )

    assert message_from_dict(message_to_dict(message)) == message


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


def test_assistant_wire_rejects_continuation_without_model_identity() -> None:
    value = message_to_dict(
        AIMessage(
            content="answer",
            continuation=ModelContinuation(
                model_key="main",
                provider="deepseek",
                model_id="deepseek-reasoner",
                protocol="openai_chat_completions",
            ),
        )
    )
    continuation = value["continuation"]
    assert isinstance(continuation, dict)
    continuation.pop("model_id")

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
