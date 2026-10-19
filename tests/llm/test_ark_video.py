"""Tests for the Ark task-based video wire codec and its invoker delivery."""

from __future__ import annotations

import base64
import time
from collections.abc import Mapping

import httpx
import pytest

from pygent.agent import ReActLayer
from pygent.core import AIMessage, Context, Module, ToolMessage, UserMessage
from pygent.llm import (
    ArkVideoAdapter,
    ArkVideoClient,
    DefaultModelInvoker,
    GenerationConfig,
    ModelCallLayer,
    ModelEntry,
    ModelGroup,
    ModelProviderError,
    ModelSpec,
    RetryPolicy,
    ark_video_adapters,
)
from pygent.llm._adapter_contracts import ModelProviderRequest
from pygent.llm.catalogs import ModelCapabilityCatalog
from pygent.llm.types import ModelErrorKind, ModelFailureReason
from pygent.runtime import ExecutionOptions, LocalRuntime
from pygent.tool import MediaBlock, MediaSource, ToolDefinition

_MP4 = (
    b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2avc1mp41"
    b"\x00\x00\x00\x08free"
)
_VIDEO_B64 = base64.b64encode(_MP4).decode("ascii")


def _spec(model_id: str = "doubao-seedance-2-0-260128", **options) -> ModelSpec:
    capabilities = ModelCapabilityCatalog.builtin().models[
        ("volcengine", model_id, "ark_video_generation")
    ]
    return ModelSpec(
        provider="volcengine",
        model_id=model_id,
        protocol="ark_video_generation",
        provider_options=options,
        capabilities=capabilities,
    )


def _request(spec=None, *, content: str = "A blue square moves.") -> ModelProviderRequest:
    return ModelProviderRequest(
        model_key="primary",
        model=spec or _spec(),
        message=UserMessage(content=content),
        context=Context(),
        generation=GenerationConfig(max_output_tokens=None),
    )


def _terminal_payload(**overrides) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": "cgt-1",
        "status": "succeeded",
        "content": {"video_url": "https://video.example/result.mp4"},
        "video_b64": _VIDEO_B64,
    }
    payload.update(overrides)
    return payload


class FakeTaskClient:
    """Invoker-level fake: returns a terminal task payload from invoke."""

    def __init__(self, payload: Mapping[str, object]) -> None:
        self.payload = freeze(payload)

    async def invoke(self, model, payload):
        return self.payload

    async def stream(self, model, payload):  # pragma: no cover - must not be called
        raise AssertionError("stream must not be called")
        yield freeze({})

    async def aclose(self) -> None:
        return None


def freeze(value: Mapping[str, object]):
    from pygent.core import freeze_json_object

    return freeze_json_object(value)


def test_ark_video_builds_the_task_request() -> None:
    spec = _spec(duration=5, ratio="16:9", poll_interval_seconds=0.5)
    payload = ArkVideoAdapter().build_request(_request(spec)).to_dict()

    assert payload["model"] == "doubao-seedance-2-0-260128"
    assert payload["content"] == [{"type": "text", "text": "A blue square moves."}]
    assert payload["duration"] == 5
    assert payload["ratio"] == "16:9"
    # Behavior options tune the client lifecycle, never the wire body.
    assert "poll_interval_seconds" not in payload


def test_ark_video_rejects_tools_schemas_media_and_non_user() -> None:
    adapter = ArkVideoAdapter()
    tool = ToolDefinition(name="lookup", description="d", parameters={"type": "object"})
    tools_request = ModelProviderRequest(
        model_key="primary",
        model=_spec(),
        message=UserMessage(content="prompt"),
        context=Context(),
        generation=GenerationConfig(max_output_tokens=None),
        tools=(tool,),
    )
    with pytest.raises(ModelProviderError, match="does not accept tools"):
        adapter.build_request(tools_request)

    schema_request = ModelProviderRequest(
        model_key="primary",
        model=_spec(),
        message=UserMessage(content="prompt"),
        context=Context(),
        generation=GenerationConfig(
            response_schema={"type": "object"}, max_output_tokens=None
        ),
    )
    with pytest.raises(ModelProviderError, match="response schemas"):
        adapter.build_request(schema_request)

    media_message = UserMessage(
        content="animate this",
        media=(
            MediaBlock(
                media_type="image",
                mime_type="image/png",
                source=MediaSource.inline(b"\x89PNG\r\n\x1a\n"),
            ),
        ),
    )
    media_request = ModelProviderRequest(
        model_key="primary",
        model=_spec(),
        message=media_message,
        context=Context(),
        generation=GenerationConfig(max_output_tokens=None),
    )
    with pytest.raises(ModelProviderError, match="image input"):
        adapter.build_request(media_request)

    assistant_request = ModelProviderRequest(
        model_key="primary",
        model=_spec(),
        message=AIMessage(content="not a prompt"),
        context=Context(),
        generation=GenerationConfig(max_output_tokens=None),
    )
    with pytest.raises(ModelProviderError, match="user message prompt"):
        adapter.build_request(assistant_request)


def test_ark_video_validates_provider_options() -> None:
    with pytest.raises(ValueError, match="reserved fields"):
        ArkVideoAdapter().validate_model(_spec(model="other"))
    with pytest.raises(ValueError, match="positive"):
        ArkVideoAdapter().validate_model(_spec(poll_interval_seconds=0))
    with pytest.raises(ValueError, match="positive"):
        ArkVideoAdapter().validate_model(_spec(max_output_bytes=-1))
    ArkVideoAdapter().validate_model(_spec(poll_timeout_seconds=120.0))


def test_ark_video_parse_response_builds_inline_video_media() -> None:
    response = ArkVideoAdapter().parse_response(
        _request(), freeze(_terminal_payload())
    )

    assert [block.media_type for block in response.message.media] == ["video"]
    block = response.message.media[0]
    assert block.mime_type == "video/mp4"
    assert base64.b64decode(block.source.base64_data) == _MP4
    assert response.finish_reason == "stop"
    assert response.provider_request_id == "cgt-1"


def test_ark_video_parse_response_rejects_bad_payloads() -> None:
    adapter = ArkVideoAdapter()
    with pytest.raises(ModelProviderError, match="did not succeed"):
        adapter.parse_response(
            _request(),
            freeze(_terminal_payload(status="failed", error={"code": "Quota"})),
        )
    with pytest.raises(ModelProviderError, match="no inline video"):
        adapter.parse_response(
            _request(), freeze(_terminal_payload(video_b64=None))
        )
    with pytest.raises(ModelProviderError, match="valid Base64"):
        adapter.parse_response(
            _request(), freeze(_terminal_payload(video_b64="not-base64!!"))
        )
    with pytest.raises(ModelProviderError, match="video container"):
        adapter.parse_response(
            _request(),
            freeze(_terminal_payload(video_b64=base64.b64encode(b"plain").decode())),
        )


def test_ark_video_streaming_is_rejected() -> None:
    adapter = ArkVideoAdapter()
    decoder = adapter.create_stream_decoder(_request())
    with pytest.raises(ModelProviderError, match="does not support streaming"):
        decoder.feed(freeze({}))
    assert decoder.finish() == ()


def test_ark_video_adapter_registry() -> None:
    assert set(ark_video_adapters()) == {"ark_video_generation"}


def _mock_client(handler) -> ArkVideoClient:
    return ArkVideoClient(
        base_url="https://ark.example/api/v3",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def _mp4_response() -> httpx.Response:
    return httpx.Response(200, content=_MP4)


@pytest.mark.asyncio
async def test_client_runs_the_full_task_lifecycle() -> None:
    polls = iter(
        [
            {"id": "cgt-1", "status": "queued"},
            {"id": "cgt-1", "status": "running"},
            {
                "id": "cgt-1",
                "status": "succeeded",
                "content": {"video_url": "https://video.example/result.mp4"},
            },
        ]
    )
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.method == "POST":
            return httpx.Response(200, json={"id": "cgt-1", "status": "queued"})
        if request.url.host == "video.example":
            return _mp4_response()
        return httpx.Response(200, json=next(polls))

    client = _mock_client(handler)
    try:
        payload = await client.invoke(
            _spec(), freeze({"model": "doubao-seedance-2-0-260128"})
        )
    finally:
        await client.aclose()

    assert calls[0] == ("POST", "/api/v3/contents/generations/tasks")
    assert all(
        method == "GET" and path.startswith("/api/v3/contents/generations/tasks/")
        for method, path in calls[1:-1]
    )
    assert calls[-1][1] == "/result.mp4"
    assert base64.b64decode(payload["video_b64"]) == _MP4
    assert payload["status"] == "succeeded"


@pytest.mark.asyncio
async def test_client_maps_http_errors_by_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"code": "AuthenticationError"}})

    client = _mock_client(handler)
    try:
        with pytest.raises(ModelProviderError) as raised:
            await client.invoke(_spec(), freeze({"model": "m"}))
    finally:
        await client.aclose()
    assert raised.value.kind is ModelErrorKind.AUTHENTICATION


@pytest.mark.asyncio
async def test_client_reports_failed_tasks_without_retryable_kind() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"id": "cgt-1", "status": "queued"})
        return httpx.Response(
            200,
            json={
                "id": "cgt-1",
                "status": "failed",
                "error": {"code": "OutputVideoSensitiveContentDetected"},
            },
        )

    client = _mock_client(handler)
    try:
        with pytest.raises(ModelProviderError) as raised:
            await client.invoke(_spec(), freeze({"model": "m"}))
    finally:
        await client.aclose()
    assert raised.value.kind is ModelErrorKind.INVALID_RESPONSE
    assert raised.value.reason_code is ModelFailureReason.INVALID_PROVIDER_RESPONSE
    assert "OutputVideoSensitiveContentDetected" in str(raised.value)


@pytest.mark.asyncio
async def test_client_poll_timeout_stays_non_retryable() -> None:
    # A retryable kind here would make the outer RetryPolicy re-run invoke and
    # create a second billed task for an existing prompt.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"id": "cgt-1", "status": "queued"})
        return httpx.Response(200, json={"id": "cgt-1", "status": "running"})

    client = ArkVideoClient(
        base_url="https://ark.example/api/v3",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        poll_interval_seconds=0.01,
        poll_timeout_seconds=0.05,
    )
    try:
        with pytest.raises(ModelProviderError) as raised:
            await client.invoke(_spec(), freeze({"model": "m"}))
    finally:
        await client.aclose()
    assert raised.value.kind is ModelErrorKind.INVALID_RESPONSE
    assert raised.value.reason_code is ModelFailureReason.PROVIDER_TIMEOUT


@pytest.mark.asyncio
async def test_client_poll_transport_errors_stay_inside_the_window() -> None:
    polls = iter(["error", {"id": "cgt-1", "status": "running"},
                  {"id": "cgt-1", "status": "succeeded",
                   "content": {"video_url": "https://video.example/result.mp4"}}])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"id": "cgt-1", "status": "queued"})
        if request.url.host == "video.example":
            return _mp4_response()
        outcome = next(polls)
        if outcome == "error":
            raise httpx.ConnectError("offline")
        return httpx.Response(200, json=outcome)

    client = ArkVideoClient(
        base_url="https://ark.example/api/v3",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        poll_interval_seconds=0.01,
        poll_timeout_seconds=5.0,
    )
    try:
        payload = await client.invoke(_spec(), freeze({"model": "m"}))
    finally:
        await client.aclose()
    assert base64.b64decode(payload["video_b64"]) == _MP4


@pytest.mark.asyncio
async def test_client_surfaces_unknown_task_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"id": "cgt-1", "status": "queued"})
        return httpx.Response(200, json={"id": "cgt-1", "status": "weird"})

    client = _mock_client(handler)
    try:
        with pytest.raises(ModelProviderError, match="unknown status"):
            await client.invoke(_spec(), freeze({"model": "m"}))
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_video_media_reaches_the_invoker_with_capability_warning() -> None:
    invoker = DefaultModelInvoker(
        adapters=ark_video_adapters(),
        clients={"primary": FakeTaskClient(_terminal_payload())},
    )
    execution = invoker.execute(
        model_group=ModelGroup("video", (ModelEntry("primary", _spec()),)),
        retry_policy=RetryPolicy(max_attempts_per_model=1),
        generation=GenerationConfig(),
        message=UserMessage(content="A blue square moves."),
        context=Context(),
    )
    result = await execution.result()
    async with execution.subscribe() as events:
        captured = [event async for event in events]

    assert [block.media_type for block in result.message.media] == ["video"]
    warnings = [e for e in captured if e.kind == "model.capability.warning"]
    assert warnings
    assert "modalities.output.text" in warnings[0].data["missing_capabilities"]

    declared = invoker.execute(
        model_group=ModelGroup("video", (ModelEntry("primary", _spec()),)),
        retry_policy=RetryPolicy(max_attempts_per_model=1),
        generation=GenerationConfig(output_modalities=("video",)),
        message=UserMessage(content="A blue square moves."),
        context=Context(),
    )
    declared_result = await declared.result()
    async with declared.subscribe() as events:
        declared_events = [event async for event in events]
    assert [block.media_type for block in declared_result.message.media] == ["video"]
    assert not [
        e for e in declared_events if e.kind == "model.capability.warning"
    ]
    await invoker.aclose()


class _NoTools(Module[AIMessage, ToolMessage]):
    async def forward(
        self, message: AIMessage, context: Context
    ) -> tuple[ToolMessage, Context]:
        raise AssertionError("a media-only answer must not enter the tool loop")


@pytest.mark.asyncio
async def test_video_model_serves_as_the_react_main_model() -> None:
    invoker = DefaultModelInvoker(
        adapters=ark_video_adapters(),
        clients={"primary": FakeTaskClient(_terminal_payload())},
    )
    model = ModelCallLayer(
        model=ModelEntry("primary", _spec()),
        retry_policy=RetryPolicy(max_attempts_per_model=1),
        generation=GenerationConfig(output_modalities=("video",)),
        invoker=invoker,
    )
    react = ReActLayer(model=model, tools=_NoTools())
    runtime = LocalRuntime()
    bound = runtime.bind(react)

    answer, _context = await bound.invoke(
        UserMessage(content="make a video"),
        Context(),
        execution=ExecutionOptions(deadline=time.monotonic() + 10),
    )
    assert [block.media_type for block in answer.media] == ["video"]
    await runtime.close()
    await invoker.aclose()
