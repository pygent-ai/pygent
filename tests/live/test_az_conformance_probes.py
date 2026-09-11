from __future__ import annotations

import base64
from collections.abc import AsyncIterator

import httpx
import pytest

from pygent.core import FrozenJsonObject, freeze_json_object
from pygent.llm import ModelSpec
from tests.live.az_conformance.anthropic_probes import (
    anthropic_image_input_probe,
    anthropic_json_object_probe,
    anthropic_json_schema_probe,
    anthropic_reasoning_probe,
    anthropic_stream_probe,
    anthropic_text_probe,
    anthropic_tool_choice_probe,
    anthropic_tool_probe,
)
from tests.live.az_conformance.gemini_probes import (
    gemini_image_input_probe,
    gemini_json_schema_probe,
    gemini_reasoning_probe,
    gemini_stream_probe,
    gemini_text_probe,
    gemini_tool_probe,
)
from tests.live.az_conformance.media_probes import (
    audio_input_probe,
    audio_output_probe,
    embedding_probe,
    image_edit_probe,
    image_output_probe,
    realtime_probe,
    video_output_probe,
)
from tests.live.az_conformance.openai_probes import (
    openai_image_input_probe,
    openai_json_object_probe,
    openai_json_schema_probe,
    openai_reasoning_probe,
    openai_responses_image_input_probe,
    openai_responses_json_schema_probe,
    openai_responses_reasoning_probe,
    openai_responses_stream_probe,
    openai_responses_text_probe,
    openai_responses_tool_probe,
    openai_stream_probe,
    openai_text_probe,
    openai_tool_choice_probe,
    openai_tool_probe,
)
from tests.live.az_conformance.probe_registry import builtin_probe_registry
from tests.live.az_conformance.runner import ProbeContext
from tests.live.az_conformance.schemas import (
    Scenario,
    load_manifest,
    route_from_mapping,
)
from tests.live.az_conformance.search_probes import serpapi_search_probe

_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1Pe"
    "AAAADElEQVR4nGNgYPgPAAEDAQAIicLsAAAAAElFTkSuQmCC"
)


class ScriptedClient:
    def __init__(
        self,
        *,
        responses: list[dict[str, object]] | None = None,
        stream: list[dict[str, object]] | None = None,
    ) -> None:
        self.responses = list(responses or [])
        self.stream_items = list(stream or [])
        self.requests: list[dict[str, object]] = []
        self.models: list[ModelSpec] = []

    async def invoke(
        self, model: ModelSpec, payload: FrozenJsonObject
    ) -> FrozenJsonObject:
        self.requests.append(payload.to_dict())
        self.models.append(model)
        return freeze_json_object(self.responses.pop(0))

    async def stream(
        self, model: ModelSpec, payload: FrozenJsonObject
    ) -> AsyncIterator[FrozenJsonObject]:
        self.requests.append(payload.to_dict())
        self.models.append(model)
        for item in self.stream_items:
            yield freeze_json_object(item)

    async def aclose(self) -> None:
        return None


def _route(protocol: str = "openai_chat_completions"):
    return route_from_mapping(
        {
            "route_id": "gpt-5.4-urg",
            "kind": "gateway_alias",
            "canonical_provider": "openai",
            "canonical_model_id": "gpt-5.4",
            "protocols": [protocol],
            "required_scenarios": ["text"],
            "catalog_eligible": False,
        }
    )


def _context(
    client: ScriptedClient,
    scenario: Scenario,
    protocol: str = "openai_chat_completions",
) -> ProbeContext:
    return ProbeContext(
        snapshot_sha256="7" * 64,
        source_revision="abc1234",
        protocol=protocol,
        scenario=scenario,
        attempt=1,
        client=client,
    )


def _openai_response(
    content: str = "ok", **message: object
) -> dict[str, object]:
    return {
        "id": "request-1",
        "choices": [
            {
                "message": {"content": content, **message},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def _anthropic_response(
    content: list[dict[str, object]], stop_reason: str = "end_turn"
) -> dict[str, object]:
    return {
        "id": "message-1",
        "type": "message",
        "role": "assistant",
        "content": content,
        "stop_reason": stop_reason,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


@pytest.mark.asyncio
async def test_openai_text_probe_uses_route_only_on_wire() -> None:
    client = ScriptedClient(responses=[_openai_response("answer")])
    result = await openai_text_probe(_context(client, Scenario.TEXT), _route())

    assert result.status == "passed"
    assert result.canonical_model_id == "gpt-5.4"
    assert client.requests[0]["model"] == "gpt-5.4-urg"
    assert client.requests[0]["max_tokens"] == 256
    assert "answer" not in str(result.to_public_mapping())


@pytest.mark.asyncio
async def test_openai_stream_probe_requires_content_and_termination() -> None:
    client = ScriptedClient(
        stream=[
            {"choices": [{"delta": {"content": "o"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"done": True},
        ]
    )
    result = await openai_stream_probe(
        _context(client, Scenario.TEXT_STREAM), _route()
    )
    assert result.status == "passed"


@pytest.mark.asyncio
async def test_openai_tool_probe_executes_tool_result_continuation() -> None:
    client = ScriptedClient(
        responses=[
            _openai_response(
                "",
                tool_calls=[
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"value":"probe"}',
                        },
                    }
                ],
            ),
            _openai_response("continued"),
        ]
    )
    result = await openai_tool_probe(_context(client, Scenario.TOOLS), _route())
    assert result.status == "passed"
    assert len(client.requests) == 2
    assert client.requests[0]["tools"]
    assert client.requests[1]["messages"][-1]["role"] == "tool"


@pytest.mark.asyncio
async def test_openai_choice_and_structured_probes_project_exact_controls() -> None:
    client = ScriptedClient(
        responses=[
            _openai_response(
                "",
                tool_calls=[
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            ),
            _openai_response('{"answer":"ok"}'),
            _openai_response('{"answer":"ok"}'),
        ]
    )
    route = _route()
    assert (
        await openai_tool_choice_probe(
            _context(client, Scenario.TOOL_CHOICE), route
        )
    ).status == "passed"
    assert (
        await openai_json_object_probe(
            _context(client, Scenario.JSON_OBJECT), route
        )
    ).status == "passed"
    assert (
        await openai_json_schema_probe(
            _context(client, Scenario.JSON_SCHEMA), route
        )
    ).status == "passed"
    assert client.requests[0]["tool_choice"]["function"]["name"] == "lookup"
    assert client.requests[1]["response_format"] == {"type": "json_object"}
    assert client.requests[2]["response_format"]["type"] == "json_schema"


@pytest.mark.asyncio
async def test_openai_reasoning_and_image_probes_validate_protocol_evidence() -> None:
    client = ScriptedClient(
        responses=[
            _openai_response("answer", reasoning_content="reasoning"),
            _openai_response("blue square"),
        ]
    )
    route = _route()
    reasoning = await openai_reasoning_probe(
        _context(client, Scenario.REASONING), route
    )
    vision = await openai_image_input_probe(
        _context(client, Scenario.IMAGE_INPUT), route
    )
    assert reasoning.status == vision.status == "passed"
    assert client.requests[0]["reasoning_effort"] == "low"
    parts = client.requests[1]["messages"][-1]["content"]
    assert parts[1]["type"] == "image_url"
    data_uri = parts[1]["image_url"]["url"]
    assert data_uri.startswith("data:image/png;base64,")
    assert base64.b64decode(data_uri.partition(",")[2]).endswith(b"IEND\xaeB`\x82")


@pytest.mark.asyncio
async def test_anthropic_stream_and_tool_round_trip_use_messages_contract() -> None:
    stream_client = ScriptedClient(
        stream=[
            {
                "type": "message_start",
                "message": {"id": "m1", "usage": {"input_tokens": 1}},
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "ok"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 1},
            },
            {"type": "message_stop"},
        ]
    )
    route = _route("anthropic_messages")
    assert (
        await anthropic_stream_probe(
            _context(stream_client, Scenario.TEXT_STREAM, "anthropic_messages"),
            route,
        )
    ).status == "passed"

    tool_client = ScriptedClient(
        responses=[
            _anthropic_response(
                [
                    {
                        "type": "tool_use",
                        "id": "call-1",
                        "name": "lookup",
                        "input": {"value": "probe"},
                    }
                ],
                "tool_use",
            ),
            _anthropic_response([{"type": "text", "text": "continued"}]),
        ]
    )
    assert (
        await anthropic_tool_probe(
            _context(tool_client, Scenario.TOOLS, "anthropic_messages"), route
        )
    ).status == "passed"
    assert tool_client.requests[1]["messages"][-1]["content"][0]["type"] == "tool_result"


@pytest.mark.asyncio
async def test_anthropic_text_choice_and_structured_probes_use_native_controls() -> None:
    client = ScriptedClient(
        responses=[
            _anthropic_response([{"type": "text", "text": "answer"}]),
            _anthropic_response(
                [
                    {
                        "type": "tool_use",
                        "id": "call-1",
                        "name": "lookup",
                        "input": {},
                    }
                ],
                "tool_use",
            ),
            _anthropic_response([{"type": "text", "text": '{"answer":"ok"}'}]),
            _anthropic_response([{"type": "text", "text": '{"answer":"ok"}'}]),
        ]
    )
    route = _route("anthropic_messages")
    probes = (
        (anthropic_text_probe, Scenario.TEXT),
        (anthropic_tool_choice_probe, Scenario.TOOL_CHOICE),
        (anthropic_json_object_probe, Scenario.JSON_OBJECT),
        (anthropic_json_schema_probe, Scenario.JSON_SCHEMA),
    )
    for probe, scenario in probes:
        result = await probe(_context(client, scenario, "anthropic_messages"), route)
        assert result.status == "passed"
    assert client.requests[1]["tool_choice"] == {"type": "tool", "name": "lookup"}
    assert "output_config" not in client.requests[2]
    assert client.requests[3]["output_config"]["format"]["type"] == "json_schema"


@pytest.mark.asyncio
async def test_anthropic_reasoning_and_image_probes_use_native_blocks() -> None:
    client = ScriptedClient(
        responses=[
            _anthropic_response(
                [
                    {"type": "thinking", "thinking": "reason", "signature": "sig"},
                    {"type": "text", "text": "answer"},
                ]
            ),
            _anthropic_response([{"type": "text", "text": "blue square"}]),
        ]
    )
    route = _route("anthropic_messages")
    assert (
        await anthropic_reasoning_probe(
            _context(client, Scenario.REASONING, "anthropic_messages"), route
        )
    ).status == "passed"
    assert (
        await anthropic_image_input_probe(
            _context(client, Scenario.IMAGE_INPUT, "anthropic_messages"), route
        )
    ).status == "passed"
    assert client.requests[0]["thinking"]["type"] == "enabled"
    blocks = client.requests[1]["messages"][-1]["content"]
    assert blocks[1]["type"] == "image"
    assert blocks[1]["source"]["type"] == "base64"
    assert base64.b64decode(blocks[1]["source"]["data"]).endswith(
        b"IEND\xaeB`\x82"
    )


def _gemini_response(
    parts: list[dict[str, object]], finish_reason: str = "STOP"
) -> dict[str, object]:
    return {
        "candidates": [
            {
                "content": {"role": "model", "parts": parts},
                "finishReason": finish_reason,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 1,
            "candidatesTokenCount": 1,
            "totalTokenCount": 2,
        },
    }


@pytest.mark.asyncio
async def test_gemini_text_and_stream_probes_use_route_only_for_transport() -> None:
    route = _route("gemini_generate_content")
    text_client = ScriptedClient(responses=[_gemini_response([{"text": "answer"}])])
    text = await gemini_text_probe(
        _context(text_client, Scenario.TEXT, "gemini_generate_content"), route
    )
    assert text.status == "passed"
    assert text_client.models[0].model_id == route.route_id
    assert text.canonical_model_id == route.canonical_model_id

    stream_client = ScriptedClient(
        stream=[
            {"candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}}]},
            _gemini_response([], "STOP"),
        ]
    )
    stream = await gemini_stream_probe(
        _context(stream_client, Scenario.TEXT_STREAM, "gemini_generate_content"),
        route,
    )
    assert stream.status == "passed"


@pytest.mark.asyncio
async def test_gemini_tool_probe_executes_native_function_response() -> None:
    route = _route("gemini_generate_content")
    client = ScriptedClient(
        responses=[
            _gemini_response([{"functionCall": {"name": "lookup", "args": {"value": "probe"}}}]),
            _gemini_response([{"text": "continued"}]),
        ]
    )
    result = await gemini_tool_probe(
        _context(client, Scenario.TOOLS, "gemini_generate_content"), route
    )
    assert result.status == "passed"
    response = client.requests[1]["contents"][-1]["parts"][0]["functionResponse"]
    assert response["name"] == "lookup"


@pytest.mark.asyncio
async def test_gemini_schema_reasoning_and_image_probes_use_native_fields() -> None:
    route = _route("gemini_generate_content")
    client = ScriptedClient(
        responses=[
            _gemini_response([{"text": '{"answer":"ok"}'}]),
            _gemini_response(
                [
                    {"text": "reason", "thought": True, "thoughtSignature": "sig"},
                    {"text": "answer"},
                ]
            ),
            _gemini_response([{"text": "blue square"}]),
        ]
    )
    assert (
        await gemini_json_schema_probe(
            _context(client, Scenario.JSON_SCHEMA, "gemini_generate_content"), route
        )
    ).status == "passed"
    assert (
        await gemini_reasoning_probe(
            _context(client, Scenario.REASONING, "gemini_generate_content"), route
        )
    ).status == "passed"
    assert (
        await gemini_image_input_probe(
            _context(client, Scenario.IMAGE_INPUT, "gemini_generate_content"), route
        )
    ).status == "passed"
    assert client.requests[0]["generationConfig"]["responseJsonSchema"]
    assert client.requests[1]["generationConfig"]["thinkingConfig"]["includeThoughts"] is True
    image_part = client.requests[2]["contents"][-1]["parts"][1]["inlineData"]
    assert image_part["mimeType"] == "image/png"
    assert base64.b64decode(image_part["data"]).endswith(b"IEND\xaeB`\x82")


def _responses_response(output: list[dict[str, object]]) -> dict[str, object]:
    return {
        "id": "resp-1",
        "status": "completed",
        "output": output,
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }


def _responses_text(text: str) -> dict[str, object]:
    return _responses_response(
        [
            {
                "id": "msg-1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ]
    )


@pytest.mark.asyncio
async def test_responses_text_stream_and_tool_probes_use_responses_contract() -> None:
    route = _route("openai_responses")
    text_client = ScriptedClient(responses=[_responses_text("answer")])
    assert (
        await openai_responses_text_probe(
            _context(text_client, Scenario.TEXT, "openai_responses"), route
        )
    ).status == "passed"
    assert text_client.requests[0]["model"] == route.route_id
    assert "input" in text_client.requests[0]

    stream_client = ScriptedClient(
        stream=[
            {"type": "response.output_text.delta", "delta": "ok"},
            {
                "type": "response.completed",
                "response": _responses_response([]),
            },
        ]
    )
    assert (
        await openai_responses_stream_probe(
            _context(stream_client, Scenario.TEXT_STREAM, "openai_responses"),
            route,
        )
    ).status == "passed"

    tool_client = ScriptedClient(
        responses=[
            _responses_response(
                [
                    {
                        "id": "fc-1",
                        "type": "function_call",
                        "call_id": "call-1",
                        "name": "lookup",
                        "arguments": '{"value":"probe"}',
                    }
                ]
            ),
            _responses_text("continued"),
        ]
    )
    assert (
        await openai_responses_tool_probe(
            _context(tool_client, Scenario.TOOLS, "openai_responses"), route
        )
    ).status == "passed"
    assert tool_client.requests[1]["input"][-1]["type"] == "function_call_output"


@pytest.mark.asyncio
async def test_responses_schema_reasoning_and_image_probes_use_native_items() -> None:
    route = _route("openai_responses")
    client = ScriptedClient(
        responses=[
            _responses_text('{"answer":"ok"}'),
            _responses_response(
                [
                    {
                        "id": "rs-1",
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": "reason"}],
                    },
                    *_responses_text("answer")["output"],
                ]
            ),
            _responses_text("blue square"),
        ]
    )
    assert (
        await openai_responses_json_schema_probe(
            _context(client, Scenario.JSON_SCHEMA, "openai_responses"), route
        )
    ).status == "passed"
    assert (
        await openai_responses_reasoning_probe(
            _context(client, Scenario.REASONING, "openai_responses"), route
        )
    ).status == "passed"
    assert (
        await openai_responses_image_input_probe(
            _context(client, Scenario.IMAGE_INPUT, "openai_responses"), route
        )
    ).status == "passed"
    assert client.requests[0]["text"]["format"]["type"] == "json_schema"
    assert client.requests[1]["reasoning"]["summary"] == "auto"
    image = client.requests[2]["input"][-1]["content"][1]
    assert image["type"] == "input_image"
    assert base64.b64decode(image["image_url"].partition(",")[2]).endswith(
        b"IEND\xaeB`\x82"
    )


class RawScriptedClient:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = list(responses)
        self.requests: list[tuple[str, str, dict[str, object]]] = []
        self.audio_fixture: bytes | None = None
        self.websocket_event: object = {"type": "response.done"}

    async def request(self, method: str, path: str, **kwargs: object) -> httpx.Response:
        self.requests.append((method, path, kwargs))
        return self.responses.pop(0)

    async def websocket_exchange(
        self, path: str, event: dict[str, object]
    ) -> object:
        self.requests.append(("WS", path, {"event": event}))
        return self.websocket_event


def _raw_context(client: RawScriptedClient, scenario: Scenario) -> ProbeContext:
    return ProbeContext(
        snapshot_sha256="7" * 64,
        source_revision="abc1234",
        protocol="openai_chat_completions",
        scenario=scenario,
        attempt=1,
        client=client,
    )


@pytest.mark.asyncio
async def test_embedding_probe_validates_finite_stable_vectors() -> None:
    client = RawScriptedClient(
        [
            httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2]}]}),
            httpx.Response(200, json={"data": [{"embedding": [0.3, 0.4]}]}),
        ]
    )

    result = await embedding_probe(
        _raw_context(client, Scenario.EMBEDDING), _route()
    )

    assert result.status == "passed"
    assert [request[1] for request in client.requests] == ["/embeddings"] * 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "responses",
    [
        [
            httpx.Response(200, json={"data": [{"embedding": [0.1]}]}),
            httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2]}]}),
        ],
        [
            httpx.Response(200, content=b'{"data":[{"embedding":[NaN]}]}'),
        ],
    ],
)
async def test_embedding_probe_rejects_invalid_or_unstable_vectors(
    responses: list[httpx.Response],
) -> None:
    result = await embedding_probe(
        _raw_context(RawScriptedClient(responses), Scenario.EMBEDDING), _route()
    )

    assert result.status == "failed"
    assert result.error_kind.value == "invalid_response"


@pytest.mark.asyncio
async def test_image_generation_and_edit_validate_png_and_multipart() -> None:
    png = base64.b64decode(_PNG)
    output_client = RawScriptedClient(
        [httpx.Response(200, json={"data": [{"b64_json": _PNG}]})]
    )
    assert (
        await image_output_probe(
            _raw_context(output_client, Scenario.IMAGE_OUTPUT), _route()
        )
    ).status == "passed"

    edit_client = RawScriptedClient(
        [httpx.Response(200, json={"data": [{"b64_json": _PNG}]})]
    )
    assert (
        await image_edit_probe(
            _raw_context(edit_client, Scenario.IMAGE_EDIT), _route()
        )
    ).status == "passed"
    assert edit_client.requests[0][1] == "/images/edits"
    assert edit_client.requests[0][2]["files"]["image"][1] == png


@pytest.mark.asyncio
async def test_image_generation_downloads_and_validates_url_result() -> None:
    client = RawScriptedClient(
        [
            httpx.Response(200, json={"data": [{"url": "https://media.test/a.png"}]}),
            httpx.Response(200, content=base64.b64decode(_PNG)),
        ]
    )

    result = await image_output_probe(
        _raw_context(client, Scenario.IMAGE_OUTPUT), _route()
    )

    assert result.status == "passed"
    assert client.requests[1][:2] == ("GET", "https://media.test/a.png")


@pytest.mark.asyncio
async def test_audio_output_feeds_transcription_dependency() -> None:
    wave = b"RIFF" + b"\x00" * 40 + b"WAVE"
    client = RawScriptedClient(
        [
            httpx.Response(200, content=wave),
            httpx.Response(200, json={"text": "probe audio"}),
        ]
    )
    assert (
        await audio_output_probe(
            _raw_context(client, Scenario.AUDIO_OUTPUT), _route()
        )
    ).status == "passed"
    assert client.audio_fixture == wave
    assert (
        await audio_input_probe(
            _raw_context(client, Scenario.AUDIO_INPUT), _route()
        )
    ).status == "passed"
    assert client.requests[1][1] == "/audio/transcriptions"


@pytest.mark.asyncio
async def test_audio_input_reports_missing_generated_dependency() -> None:
    client = RawScriptedClient([])
    result = await audio_input_probe(
        _raw_context(client, Scenario.AUDIO_INPUT), _route()
    )

    assert result.status == "failed"
    assert result.error_kind.value == "dependency_failed"


@pytest.mark.asyncio
async def test_realtime_probe_exchanges_one_legal_event() -> None:
    client = RawScriptedClient([])
    result = await realtime_probe(
        _raw_context(client, Scenario.REALTIME), _route()
    )

    assert result.status == "passed"
    assert client.requests[0][0] == "WS"


@pytest.mark.asyncio
async def test_video_probe_polls_until_success_and_validates_media() -> None:
    video = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 16
    client = RawScriptedClient(
        [
            httpx.Response(200, json={"id": "video-1", "status": "queued"}),
            httpx.Response(200, json={"id": "video-1", "status": "processing"}),
            httpx.Response(
                200,
                json={
                    "id": "video-1",
                    "status": "completed",
                    "b64_json": base64.b64encode(video).decode(),
                },
            ),
        ]
    )

    result = await video_output_probe(
        _raw_context(client, Scenario.VIDEO_OUTPUT), _route()
    )

    assert result.status == "passed"
    assert [item[1] for item in client.requests] == [
        "/videos",
        "/videos/video-1",
        "/videos/video-1",
    ]


@pytest.mark.asyncio
async def test_video_probe_reports_failed_task_and_bounded_timeout() -> None:
    failed = RawScriptedClient(
        [httpx.Response(200, json={"id": "video-1", "status": "failed"})]
    )
    failed_result = await video_output_probe(
        _raw_context(failed, Scenario.VIDEO_OUTPUT), _route()
    )
    assert failed_result.error_kind.value == "capability_mismatch"

    processing = {"id": "video-1", "status": "processing"}
    timed_out = RawScriptedClient(
        [httpx.Response(200, json=processing) for _ in range(21)]
    )
    timeout_result = await video_output_probe(
        _raw_context(timed_out, Scenario.VIDEO_OUTPUT), _route()
    )
    assert timeout_result.error_kind.value == "timeout"


@pytest.mark.asyncio
async def test_serpapi_probe_requires_at_least_one_result() -> None:
    client = RawScriptedClient(
        [httpx.Response(200, json={"organic_results": [{"title": "result"}]})]
    )
    route = route_from_mapping(
        {
            "route_id": "serpapi-google",
            "kind": "external_service",
            "canonical_provider": None,
            "canonical_model_id": None,
            "protocols": ["serpapi_search"],
            "required_scenarios": ["search"],
            "catalog_eligible": False,
        }
    )
    context = ProbeContext(
        snapshot_sha256="7" * 64,
        source_revision="abc1234",
        protocol="serpapi_search",
        scenario=Scenario.SEARCH,
        attempt=1,
        client=client,
    )

    assert (await serpapi_search_probe(context, route)).status == "passed"

    empty_client = RawScriptedClient(
        [httpx.Response(200, json={"organic_results": []})]
    )
    empty_context = ProbeContext(
        snapshot_sha256="7" * 64,
        source_revision="abc1234",
        protocol="serpapi_search",
        scenario=Scenario.SEARCH,
        attempt=1,
        client=empty_client,
    )
    empty = await serpapi_search_probe(empty_context, route)
    assert empty.status == "failed"
    assert empty.error_kind.value == "invalid_response"


def test_builtin_probe_registry_covers_every_manifest_protocol_scenario() -> None:
    manifest = load_manifest()
    expected = {
        (protocol, scenario)
        for route in manifest.routes
        for protocol in route.protocols
        for scenario in route.required_scenarios
    }

    assert set(builtin_probe_registry().probes) == expected
