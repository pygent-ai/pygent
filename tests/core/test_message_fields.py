"""Contracts for the unified Message projection (spec §3, WP-core).

Covers the WP-core acceptance items: reflection completeness over
``dataclasses.fields`` for every preset, decode strict field sets equal to
the two pre-refactor literal sets, and explicit rejection of unknown decode
fields.  The preset-versus-golden tests additionally pin every preset to the
recorded pre-refactor consumer outputs byte for byte.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pytest

from pygent.core import (
    AIMessage,
    Message,
    MessageProfile,
    ModelContinuation,
    ToolMessage,
    UserMessage,
    message_fields_expected_keys,
    message_from_fields,
    project_message_fields,
)
from pygent.core._tool_values import (
    MediaBlock,
    MediaSource,
    ToolCall,
    ToolResult,
    ToolTask,
    ToolTaskState,
)
from pygent.tool import (
    IdempotencyPolicy,
    ToolAuthorizationDecision,
    ToolAuthorizationRequest,
    ToolDefinition,
    ToolSideEffect,
    ToolSpec,
)
from tests.support.message_projection_corpus import CORPUS

GOLDEN = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "support"
        / "message_projection_golden.json"
    ).read_text(encoding="utf-8")
)

_MEDIA_BLOCK = MediaBlock(
    media_type="image",
    mime_type="image/png",
    source=MediaSource.inline(b"\x00\x01\x02\x03"),
    width=1,
    height=1,
)

_TOOL_SPEC = ToolSpec(
    tool_id="tool-1",
    version="1.0.0",
    definition=ToolDefinition(
        name="lookup",
        description="Lookup a value.",
        parameters={"type": "object", "properties": {}},
    ),
    side_effect=ToolSideEffect.PURE,
    idempotency=IdempotencyPolicy.INHERENT,
)

_PROFILES: tuple[tuple[str, MessageProfile], ...] = (
    ("codec", MessageProfile.WIRE),
    ("layer_effect", MessageProfile.EFFECT),
    ("snapshot", MessageProfile.SNAPSHOT),
    ("agent_estimate", MessageProfile.ESTIMATE),
)

_MESSAGE_TYPES: tuple[type[Message], ...] = (
    AIMessage,
    Message,
    ToolAuthorizationDecision,
    ToolAuthorizationRequest,
    ToolMessage,
    UserMessage,
)

_PRESET_GOLDEN: tuple[tuple[MessageProfile, str], ...] = (
    (MessageProfile.WIRE, "codec_wire"),
    (MessageProfile.EFFECT, "layer_effect"),
    (MessageProfile.SNAPSHOT, "request_snapshot"),
    (MessageProfile.ESTIMATE, "agent_estimate"),
)


def _projection_samples() -> dict[type[Message], Message]:
    base = {
        "content": "sample",
        "slot": "slot-1",
        "kind": "domain.event",
        "data": {"epoch": 3},
        "metadata": {"trace": "t-1"},
    }
    return {
        Message: Message(**base),
        UserMessage: UserMessage(**base, media=(_MEDIA_BLOCK,)),
        AIMessage: AIMessage(
            **base,
            media=(_MEDIA_BLOCK,),
            tool_calls=(
                ToolCall(
                    call_id="call-1",
                    name="lookup",
                    arguments={"id": 7},
                    tool_id="tool-1",
                    tool_version="1.0.0",
                    idempotency_key="idem-1",
                ),
            ),
            usage={"input_tokens": 12},
            continuation=ModelContinuation(
                model_key="main",
                provider="google",
                model_id="gemini-3.7-flash",
                protocol="gemini_generate_content",
                data={"version": 1},
            ),
        ),
        ToolMessage: ToolMessage(
            **base,
            results=(
                ToolResult(
                    call_id="call-1",
                    name="lookup",
                    status="failed",
                    error="boom",
                    task=ToolTask(
                        task_id="task-1",
                        call_id="call-1",
                        tool_id="tool-1",
                        version="1.0.0",
                        state=ToolTaskState.FAILED,
                        metadata={"attempt": 1},
                    ),
                ),
            ),
        ),
        ToolAuthorizationRequest: ToolAuthorizationRequest(
            **base,
            call=ToolCall(call_id="call-9", name="deploy", arguments={"env": "prod"}),
            spec=_TOOL_SPEC,
            permissions=("env:prod",),
        ),
        ToolAuthorizationDecision: ToolAuthorizationDecision(
            **base,
            call_id="call-9",
            allowed=False,
            reason_code="policy_denied",
            lifecycle="sync",
        ),
    }


# Dataclass fields each profile intentionally does not project (WP-core
# acceptance 1: omissions are declared per field).  The snapshot profile
# renames continuation to continuation_digest instead of omitting it.
OMITTED_FIELDS: dict[tuple[str, type[Message]], frozenset[str]] = {
    ("codec", Message): frozenset(),
    ("codec", UserMessage): frozenset(),
    ("codec", AIMessage): frozenset(),
    ("codec", ToolMessage): frozenset(),
    ("codec", ToolAuthorizationRequest): frozenset(),
    ("codec", ToolAuthorizationDecision): frozenset(),
    ("layer_effect", Message): frozenset({"kind", "data"}),
    ("layer_effect", UserMessage): frozenset({"kind", "data", "media"}),
    ("layer_effect", AIMessage): frozenset({"kind", "data", "usage"}),
    ("layer_effect", ToolMessage): frozenset({"kind", "data"}),
    ("layer_effect", ToolAuthorizationRequest): frozenset(
        {"kind", "data", "call", "spec", "permissions"}
    ),
    ("layer_effect", ToolAuthorizationDecision): frozenset(
        {"kind", "data", "call_id", "allowed", "reason_code", "lifecycle"}
    ),
    ("snapshot", Message): frozenset({"data", "metadata"}),
    ("snapshot", UserMessage): frozenset({"data", "metadata"}),
    ("snapshot", AIMessage): frozenset({"data", "metadata", "usage"}),
    ("snapshot", ToolMessage): frozenset({"data", "metadata"}),
    ("snapshot", ToolAuthorizationRequest): frozenset(
        {"data", "metadata", "call", "spec", "permissions"}
    ),
    ("snapshot", ToolAuthorizationDecision): frozenset(
        {"data", "metadata", "call_id", "allowed", "reason_code", "lifecycle"}
    ),
    ("agent_estimate", Message): frozenset({"data", "metadata"}),
    ("agent_estimate", UserMessage): frozenset({"data", "metadata", "media"}),
    ("agent_estimate", AIMessage): frozenset(
        {"data", "metadata", "media", "usage", "continuation"}
    ),
    ("agent_estimate", ToolMessage): frozenset({"data", "metadata"}),
    ("agent_estimate", ToolAuthorizationRequest): frozenset(
        {"data", "metadata", "call", "spec", "permissions"}
    ),
    ("agent_estimate", ToolAuthorizationDecision): frozenset(
        {"data", "metadata", "call_id", "allowed", "reason_code", "lifecycle"}
    ),
}

RENAMED_FIELDS: dict[tuple[str, type[Message]], dict[str, str]] = {
    ("snapshot", AIMessage): {"continuation": "continuation_digest"},
}


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


@pytest.mark.parametrize("profile_name,profile", _PROFILES)
@pytest.mark.parametrize("message_type", _MESSAGE_TYPES)
def test_every_dataclass_field_is_projected_or_declared(
    profile_name: str, profile: MessageProfile, message_type: type[Message]
) -> None:
    message = _projection_samples()[message_type]
    projected = project_message_fields(message, profile)
    omitted = OMITTED_FIELDS[(profile_name, message_type)]
    renamed = RENAMED_FIELDS.get((profile_name, message_type), {})
    missing = {
        item.name
        for item in fields(message_type)
        if item.name not in projected
        and item.name not in omitted
        and item.name not in renamed
    }
    assert not missing, f"unexplained missing fields: {sorted(missing)}"
    for projected_key in renamed.values():
        assert projected_key in projected


@pytest.mark.parametrize("sample_name", sorted(CORPUS))
@pytest.mark.parametrize("profile,golden_key", _PRESET_GOLDEN)
def test_preset_projection_matches_recorded_golden(
    profile: MessageProfile, golden_key: str, sample_name: str
) -> None:
    projected = project_message_fields(CORPUS[sample_name], profile)
    assert _canonical(projected) == GOLDEN[golden_key][sample_name]


def test_codec_assistant_decode_field_set_matches_current_literal() -> None:
    assert message_fields_expected_keys(
        "assistant", MessageProfile.WIRE, None
    ) == frozenset(
        {
            "role",
            "content",
            "slot",
            "kind",
            "data",
            "metadata",
            "tool_calls",
            "usage",
            "continuation",
            "media",
        }
    )


def test_layer_effect_decode_field_set_matches_effect_literal() -> None:
    from pygent.llm.layer import _EFFECT_MESSAGE_FIELDS

    assert (
        message_fields_expected_keys("assistant", MessageProfile.EFFECT, None)
        == _EFFECT_MESSAGE_FIELDS
    )


def test_decode_rejects_unknown_fields_explicitly() -> None:
    projected = project_message_fields(AIMessage(content="answer"), MessageProfile.WIRE)
    assert isinstance(projected, dict)
    projected["unexpected"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        message_from_fields(projected, MessageProfile.WIRE)


def test_decode_rejects_unknown_fields_in_effect_profile() -> None:
    projected = project_message_fields(
        AIMessage(content="answer"), MessageProfile.EFFECT
    )
    assert isinstance(projected, dict)
    projected["unexpected"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        message_from_fields(projected, MessageProfile.EFFECT)


def test_decode_rejects_wire_message_missing_required_usage() -> None:
    projected = project_message_fields(AIMessage(content="answer"), MessageProfile.WIRE)
    assert isinstance(projected, dict)
    projected.pop("usage")
    with pytest.raises(ValueError, match="invalid Message"):
        message_from_fields(projected, MessageProfile.WIRE)


def test_authorization_messages_project_base_fields_outside_wire_profile() -> None:
    message = _projection_samples()[ToolAuthorizationRequest]
    projected = project_message_fields(message, MessageProfile.EFFECT)
    assert set(projected) == {"role", "content", "slot", "metadata"}
    assert message_fields_expected_keys(
        "message", MessageProfile.EFFECT, None
    ) == frozenset({"role", "content", "slot", "metadata"})


@pytest.mark.parametrize("sample_name", sorted(CORPUS))
def test_wire_decode_roundtrip(sample_name: str) -> None:
    value = GOLDEN["codec_wire"][sample_name]
    restored = message_from_fields(json.loads(value), MessageProfile.WIRE)
    assert _canonical(project_message_fields(restored, MessageProfile.WIRE)) == value


@pytest.mark.parametrize(
    "sample_name",
    # The effect decoder is the model-response boundary by design: it only
    # reconstructs AIMessage samples.
    ["ai_plain", "ai_tool_calls_full", "ai_media", "ai_continuation"],
)
def test_effect_decode_roundtrip(sample_name: str) -> None:
    value = GOLDEN["layer_effect"][sample_name]
    restored = message_from_fields(json.loads(value), MessageProfile.EFFECT)
    assert isinstance(restored, AIMessage)
    assert _canonical(project_message_fields(restored, MessageProfile.EFFECT)) == value
