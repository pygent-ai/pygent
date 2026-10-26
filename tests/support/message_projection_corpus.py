"""Golden corpus of Message samples for the projection unification refactor.

WP0 baseline: every sample is recorded against the four pre-refactor
consumers in ``tests/support/message_projection_golden.json``. The golden
test pins those outputs byte-for-byte; the refactor must keep them green.
"""

from __future__ import annotations

import base64
from typing import cast

from pygent import UserMessage
from pygent.core import (
    AIMessage,
    JsonValue,
    Message,
    ModelContinuation,
    ToolMessage,
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
    ToolSideEffect,
    ToolSpec,
    ToolTask,
    ToolTaskState,
)

PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAE"
    "hQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _image_inline() -> MediaBlock:
    return MediaBlock(
        media_type="image",
        mime_type="image/png",
        source=MediaSource.inline(base64.b64decode(PNG_BASE64)),
        width=1,
        height=1,
    )


def _audio_inline() -> MediaBlock:
    return MediaBlock(
        media_type="audio",
        mime_type="audio/l16",
        source=MediaSource.inline(b"\x00\x01\x02\x03"),
        mime_parameters="rate=24000; channels=1",
    )


def _video_resource() -> MediaBlock:
    return MediaBlock(
        media_type="video",
        mime_type="video/mp4",
        source=MediaSource.resource(
            "media://video-1", sha256="a" * 64, size_bytes=2048
        ),
        duration_seconds=2.5,
        fps=24,
        has_audio=True,
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

_MESSAGES: dict[str, Message] = {
    "message_base": Message(
        content="plain domain message",
        kind="domain.event",
        data={"epoch": 3, "labels": ("a", "b")},
        slot="inbox",
        metadata={"trace": "t-1"},
    ),
    "user_plain": UserMessage(content="hello there"),
    "user_media_image": UserMessage(
        content="what is this?",
        media=(_image_inline(),),
    ),
    "user_media_mixed": UserMessage(
        content="mixed media",
        media=(
            MediaBlock(
                media_type="image",
                mime_type="image/png",
                source=MediaSource.resource(
                    "media://img-1", sha256="b" * 64, size_bytes=99
                ),
                width=64,
                height=48,
                detail="high",
            ),
            _video_resource(),
            _audio_inline(),
        ),
    ),
    "ai_plain": AIMessage(content="just text"),
    "ai_tool_calls_full": AIMessage(
        content="calling tools",
        tool_calls=(
            ToolCall(
                call_id="call-1",
                name="lookup",
                arguments={"id": 7},
                tool_id="tool-1",
                tool_version="1.0.0",
                idempotency_key="idem-1",
            ),
            ToolCall(call_id="call-2", name="other", arguments={"x": [1, 2]}),
        ),
        usage={"input_tokens": 12, "output_tokens": 34, "reasoning_tokens": 2},
        metadata={"model_key": "main"},
    ),
    "ai_media": AIMessage(
        content="here is your image",
        media=(_image_inline(),),
        metadata={"model_key": "main"},
    ),
    "ai_continuation": AIMessage(
        content="with continuation",
        continuation=ModelContinuation(
            model_key="main",
            provider="google",
            model_id="gemini-3.7-flash",
            protocol="gemini_generate_content",
            data={"version": 1, "parts": [{"text": "thought"}]},
        ),
    ),
    "tool_results_full": ToolMessage(
        results=(
            ToolResult(
                call_id="call-1",
                name="lookup",
                status="succeeded",
                output=cast(JsonValue, {"value": "ok"}),
                content=(_image_inline(),),
                tool_id="tool-1",
                tool_version="1.0.0",
            ),
            ToolResult(
                call_id="call-2",
                name="flaky",
                status="failed",
                output=None,
                error="boom",
                error_kind="provider",
                error_code="E_TEST",
                retryable=True,
                side_effect_committed=True,
                tool_id="tool-2",
                tool_version="2.0.0",
                task=ToolTask(
                    task_id="task-1",
                    call_id="call-2",
                    tool_id="tool-2",
                    version="2.0.0",
                    state=ToolTaskState.FAILED,
                    metadata={"attempt": 1},
                ),
            ),
        )
    ),
    "tool_results_detached": ToolMessage(
        results=(
            ToolResult(
                call_id="call-3",
                name="batch",
                status="detached",
                task=ToolTask(
                    task_id="task-2",
                    call_id="call-3",
                    tool_id="tool-3",
                    version="1.0.0",
                    state=ToolTaskState.RUNNING,
                ),
            ),
        )
    ),
    "tool_auth_request": ToolAuthorizationRequest(
        call=ToolCall(call_id="call-9", name="deploy", arguments={"env": "prod"}),
        spec=_TOOL_SPEC,
        permissions=("env:prod",),
    ),
    "tool_auth_decision": ToolAuthorizationDecision(
        call_id="call-9",
        allowed=False,
        reason_code="policy_denied",
        lifecycle="sync",
    ),
}

CORPUS: dict[str, Message] = dict(_MESSAGES)

__all__ = ["CORPUS", "PNG_BASE64"]
