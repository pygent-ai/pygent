from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator, Sequence

import httpx
import pytest

from pygent.core import FrozenJsonObject, freeze_json_object
from pygent.llm import ModelSpec
from tests.live.az_conformance.anthropic_probes import (
    anthropic_image_input_probe,
    anthropic_json_schema_probe,
    anthropic_reasoning_probe,
    anthropic_stream_probe,
    anthropic_text_probe,
    anthropic_tool_choice_probe,
    anthropic_tool_probe,
)
from tests.live.az_conformance.gemini_probes import (
    gemini_audio_output_probe,
    gemini_image_edit_probe,
    gemini_image_input_probe,
    gemini_image_output_probe,
    gemini_json_schema_probe,
    gemini_reasoning_probe,
    gemini_stream_probe,
    gemini_text_probe,
    gemini_tool_probe,
)
from tests.live.az_conformance.media_fixtures import PNG_BASE64
from tests.live.az_conformance.media_probes import (
    audio_input_probe,
    audio_output_probe,
    dashscope_image_edit_probe,
    dashscope_image_output_probe,
    embedding_probe,
    image_edit_probe,
    image_output_probe,
    realtime_probe,
    video_input_probe,
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
from tests.live.az_conformance.search_probes import search_probe

_PNG = PNG_BASE64


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


def _route(
    protocol: str = "openai_chat_completions",
    *,
    provider: str = "openai",
    model_id: str = "gpt-5.4",
    route_id: str = "gpt-5.4-urg",
):
    return route_from_mapping(
        {
            "route_id": route_id,
            "kind": "gateway_alias",
            "canonical_provider": provider,
            "canonical_model_id": model_id,
            "protocols": [
                {"protocol": protocol, "required_scenarios": ["text"]}
            ],
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
    assert client.requests[0]["max_tokens"] == 1024
    assert "answer" not in str(result.to_public_mapping())


@pytest.mark.asyncio
async def test_deepseek_probes_isolate_thinking_from_other_capabilities() -> None:
    route = _route(
        provider="deepseek",
        model_id="deepseek-v4-flash",
        route_id="deepseek-v4-flash",
    )
    text_client = ScriptedClient(responses=[_openai_response("answer")])
    reasoning_client = ScriptedClient(
        responses=[_openai_response("answer", reasoning_content="reasoning")]
    )

    assert (
        await openai_text_probe(_context(text_client, Scenario.TEXT), route)
    ).status == "passed"
    assert (
        await openai_reasoning_probe(
            _context(reasoning_client, Scenario.REASONING), route
        )
    ).status == "passed"

    assert text_client.requests[0]["thinking"] == {"type": "disabled"}
    assert reasoning_client.requests[0]["thinking"] == {"type": "enabled"}
    assert reasoning_client.requests[0]["reasoning_effort"] == "low"


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
    assert "reply with exactly DONE" in str(client.requests[0])
    assert client.requests[1]["messages"][0]["role"] == "user"
    assert client.requests[1]["messages"][-1]["role"] == "tool"


@pytest.mark.asyncio
async def test_openai_audio_probes_supply_required_audio_output_contract() -> None:
    tool_call = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"value":"probe"}'},
    }
    client = ScriptedClient(
        responses=[
            _openai_response("", audio={"transcript": "probe"}),
            _openai_response("", tool_calls=[tool_call]),
            _openai_response("", audio={"transcript": "probe"}),
            _openai_response("", tool_calls=[tool_call]),
        ],
        stream=[
            {
                "choices": [
                    {"delta": {"audio": {"transcript": "probe"}}}
                ]
            },
            {"done": True},
        ],
    )
    route = _route(
        provider="openai",
        model_id="gpt-audio-1.5",
        route_id="gpt-audio-1.5",
    )

    assert (
        await openai_text_probe(_context(client, Scenario.TEXT), route)
    ).status == "passed"
    assert (
        await openai_stream_probe(_context(client, Scenario.TEXT_STREAM), route)
    ).status == "passed"
    assert (
        await openai_tool_probe(_context(client, Scenario.TOOLS), route)
    ).status == "passed"
    assert (
        await openai_tool_choice_probe(
            _context(client, Scenario.TOOL_CHOICE), route
        )
    ).status == "passed"

    assert len(client.requests) == 5
    for request in client.requests:
        assert request["modalities"] == ["text", "audio"]
        assert request["audio"]["voice"] == "alloy"
    assert client.requests[1]["audio"]["format"] == "pcm16"
    assert all(
        request["audio"]["format"] == "wav"
        for index, request in enumerate(client.requests)
        if index != 1
    )


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
async def test_alibaba_named_tool_choice_probe_disables_thinking() -> None:
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
            )
        ]
    )
    route = route_from_mapping(
        {
            "route_id": "qwen3.8-max",
            "kind": "official_model",
            "canonical_provider": "alibaba_cloud",
            "canonical_model_id": "qwen3.8-max",
            "protocols": [
                {
                    "protocol": "openai_chat_completions",
                    "required_scenarios": ["tool_choice"],
                }
            ],
            "catalog_eligible": True,
        }
    )

    result = await openai_tool_choice_probe(
        _context(client, Scenario.TOOL_CHOICE), route
    )

    assert result.status == "passed"
    assert client.requests[0]["enable_thinking"] is False


@pytest.mark.asyncio
async def test_alibaba_tool_continuation_probe_disables_thinking() -> None:
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
            _openai_response("DONE"),
        ]
    )
    route = route_from_mapping(
        {
            "route_id": "qwen3-vl-plus",
            "kind": "official_model",
            "canonical_provider": "alibaba_cloud",
            "canonical_model_id": "qwen3-vl-plus",
            "protocols": [
                {
                    "protocol": "openai_chat_completions",
                    "required_scenarios": ["tools"],
                }
            ],
            "catalog_eligible": True,
        }
    )

    result = await openai_tool_probe(_context(client, Scenario.TOOLS), route)

    assert result.status == "passed"
    assert client.requests[0]["enable_thinking"] is False
    assert client.requests[1]["enable_thinking"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "protocol",
    ["openai_chat_completions", "anthropic_messages"],
)
async def test_kimi_k3_tool_choice_probe_uses_required(protocol: str) -> None:
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
            )
            if protocol == "openai_chat_completions"
            else {
                "id": "message-1",
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call-1",
                        "name": "lookup",
                        "input": {},
                    }
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        ]
    )
    route = _route(
        protocol,
        provider="moonshot",
        model_id="kimi-k3",
        route_id="kimi-k3",
    )

    if protocol == "openai_chat_completions":
        result = await openai_tool_choice_probe(
            _context(client, Scenario.TOOL_CHOICE), route
        )
        assert client.requests[0]["tool_choice"] == "required"
    else:
        result = await anthropic_tool_choice_probe(
            _context(client, Scenario.TOOL_CHOICE, protocol), route
        )
        assert client.requests[0]["tool_choice"] == {"type": "any"}

    assert result.status == "passed"


@pytest.mark.asyncio
async def test_kimi_k26_named_tool_probe_disables_thinking() -> None:
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
            )
        ]
    )
    route = _route(
        provider="moonshot", model_id="kimi-k2.6", route_id="kimi-k2.6"
    )

    result = await openai_tool_choice_probe(
        _context(client, Scenario.TOOL_CHOICE), route
    )

    assert result.status == "passed"
    assert client.requests[0]["thinking"] == {"type": "disabled"}


@pytest.mark.asyncio
async def test_kimi_reasoning_probes_use_model_specific_controls() -> None:
    openai_client = ScriptedClient(
        responses=[_openai_response("answer", reasoning_content="reasoning")]
    )
    k26 = _route(
        provider="moonshot", model_id="kimi-k2.6", route_id="kimi-k2.6"
    )
    assert (
        await openai_reasoning_probe(
            _context(openai_client, Scenario.REASONING), k26
        )
    ).status == "passed"
    assert openai_client.requests[0]["thinking"] == {"type": "enabled"}

    anthropic_client = ScriptedClient(
        responses=[
            {
                "id": "message-1",
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "reasoning", "signature": "sig"},
                    {"type": "text", "text": "answer"},
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        ]
    )
    k3 = _route(
        "anthropic_messages",
        provider="moonshot",
        model_id="kimi-k3",
        route_id="kimi-k3",
    )
    assert (
        await anthropic_reasoning_probe(
            _context(anthropic_client, Scenario.REASONING, "anthropic_messages"),
            k3,
        )
    ).status == "passed"
    assert anthropic_client.requests[0]["output_config"] == {"effort": "low"}
    assert "thinking" not in anthropic_client.requests[0]


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
async def test_official_openai_reasoning_probe_accepts_normal_text_response() -> None:
    client = ScriptedClient(responses=[_openai_response("answer")])
    route = _route(provider="openai", model_id="gpt-5.4", route_id="gpt-5.4")

    result = await openai_reasoning_probe(
        _context(client, Scenario.REASONING), route
    )

    assert result.status == "passed"
    assert client.requests[0]["reasoning_effort"] == "low"


@pytest.mark.asyncio
async def test_qvq_reasoning_probe_uses_stream_without_a_control_option() -> None:
    client = ScriptedClient(
        stream=[
            {"choices": [{"delta": {"reasoning_content": "thinking"}}]},
            {"choices": [{"delta": {"content": "answer"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"done": True},
        ]
    )
    route = route_from_mapping(
        {
            "route_id": "qvq-max",
            "kind": "official_model",
            "canonical_provider": "alibaba_cloud",
            "canonical_model_id": "qvq-max",
            "protocols": [
                {
                    "protocol": "openai_chat_completions",
                    "required_scenarios": ["reasoning"],
                }
            ],
            "catalog_eligible": True,
        }
    )

    result = await openai_reasoning_probe(
        _context(client, Scenario.REASONING), route
    )

    assert result.status == "passed"
    assert "reasoning_effort" not in client.requests[0]
    assert "enable_thinking" not in client.requests[0]


@pytest.mark.asyncio
async def test_qvq_image_probe_uses_its_stream_only_transport() -> None:
    client = ScriptedClient(
        stream=[
            {"choices": [{"delta": {"content": "blue"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"done": True},
        ]
    )
    route = route_from_mapping(
        {
            "route_id": "qvq-max",
            "kind": "official_model",
            "canonical_provider": "alibaba_cloud",
            "canonical_model_id": "qvq-max",
            "protocols": [
                {
                    "protocol": "openai_chat_completions",
                    "required_scenarios": ["image_input"],
                }
            ],
            "catalog_eligible": True,
        }
    )

    result = await openai_image_input_probe(
        _context(client, Scenario.IMAGE_INPUT), route
    )

    assert result.status == "passed"
    parts = client.requests[0]["messages"][-1]["content"]
    assert parts[1]["type"] == "image_url"


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
    assert "reply with exactly DONE" in str(tool_client.requests[0])
    assert tool_client.requests[1]["messages"][0]["role"] == "user"
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
        ]
    )
    route = _route("anthropic_messages")
    probes = (
        (anthropic_text_probe, Scenario.TEXT),
        (anthropic_tool_choice_probe, Scenario.TOOL_CHOICE),
        (anthropic_json_schema_probe, Scenario.JSON_SCHEMA),
    )
    for probe, scenario in probes:
        result = await probe(_context(client, scenario, "anthropic_messages"), route)
        assert result.status == "passed"
    assert client.requests[1]["tool_choice"] == {"type": "tool", "name": "lookup"}
    assert client.requests[2]["output_config"]["format"]["type"] == "json_schema"


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
    assert "reply with exactly DONE" in str(client.requests[0])
    assert client.requests[1]["contents"][0]["role"] == "user"
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


@pytest.mark.asyncio
async def test_gemini_output_probes_use_native_inline_media_contracts() -> None:
    route = _route("gemini_generate_content")
    client = ScriptedClient(
        responses=[
            _gemini_response(
                [{"inlineData": {"mimeType": "image/png", "data": _PNG}}]
            ),
            _gemini_response(
                [{"inlineData": {"mimeType": "image/png", "data": _PNG}}]
            ),
            _gemini_response(
                [{"inlineData": {"mimeType": "audio/L16", "data": "AAE="}}]
            ),
        ]
    )

    assert (
        await gemini_image_output_probe(
            _context(client, Scenario.IMAGE_OUTPUT, "gemini_generate_content"), route
        )
    ).status == "passed"
    assert (
        await gemini_image_edit_probe(
            _context(client, Scenario.IMAGE_EDIT, "gemini_generate_content"), route
        )
    ).status == "passed"
    assert (
        await gemini_audio_output_probe(
            _context(client, Scenario.AUDIO_OUTPUT, "gemini_generate_content"), route
        )
    ).status == "passed"

    assert client.requests[0]["generationConfig"]["responseModalities"] == ["IMAGE"]
    image_input = client.requests[1]["contents"][0]["parts"][1]["inlineData"]
    assert image_input["mimeType"] == "image/png"
    assert client.requests[2]["generationConfig"]["responseModalities"] == ["AUDIO"]
    voice = client.requests[2]["generationConfig"]["speechConfig"]["voiceConfig"]
    assert voice["prebuiltVoiceConfig"]["voiceName"] == "Kore"


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
    assert "reply with exactly DONE" in str(tool_client.requests[0])
    assert tool_client.requests[1]["input"][0]["role"] == "user"
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
        self.websocket_events: tuple[object, ...] = (
            {"type": "response.output_text.delta", "delta": "probe"},
            {"type": "response.done"},
        )

    async def request(self, method: str, path: str, **kwargs: object) -> httpx.Response:
        self.requests.append((method, path, kwargs))
        return self.responses.pop(0)

    async def websocket_exchange(
        self,
        path: str,
        events: Sequence[dict[str, object]],
        terminal_event_types: frozenset[str],
    ) -> tuple[object, ...]:
        self.requests.append(
            (
                "WS",
                path,
                {
                    "events": list(events),
                    "terminal_event_types": terminal_event_types,
                },
            )
        )
        return self.websocket_events


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
async def test_image_generation_rejects_signature_only_png() -> None:
    malformed = b"\x89PNG\r\n\x1a\nIEND\xaeB\x60\x82"
    client = RawScriptedClient(
        [
            httpx.Response(
                200,
                json={"data": [{"b64_json": base64.b64encode(malformed).decode()}]},
            )
        ]
    )

    result = await image_output_probe(
        _raw_context(client, Scenario.IMAGE_OUTPUT), _route()
    )

    assert result.status == "failed"
    assert result.error_kind.value == "invalid_response"


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
async def test_dashscope_image_generation_and_edit_use_native_contract() -> None:
    png = base64.b64decode(_PNG)
    response = {
        "output": {
            "choices": [
                {
                    "message": {
                        "content": [{"image": "https://media.test/qwen.png"}]
                    }
                }
            ]
        }
    }
    route = route_from_mapping(
        {
            "route_id": "qwen-image-3.0",
            "kind": "official_model",
            "canonical_provider": "alibaba_cloud",
            "canonical_model_id": "qwen-image-3.0",
            "protocols": [
                {
                    "protocol": "dashscope_multimodal_generation",
                    "required_scenarios": ["image_output", "image_edit"],
                }
            ],
            "catalog_eligible": True,
        }
    )
    output_client = RawScriptedClient(
        [httpx.Response(200, json=response), httpx.Response(200, content=png)]
    )
    edit_client = RawScriptedClient(
        [httpx.Response(200, json=response), httpx.Response(200, content=png)]
    )
    output_context = ProbeContext(
        snapshot_sha256="7" * 64,
        source_revision="abc1234",
        protocol="dashscope_multimodal_generation",
        scenario=Scenario.IMAGE_OUTPUT,
        attempt=1,
        client=output_client,
    )
    edit_context = ProbeContext(
        snapshot_sha256="7" * 64,
        source_revision="abc1234",
        protocol="dashscope_multimodal_generation",
        scenario=Scenario.IMAGE_EDIT,
        attempt=1,
        client=edit_client,
    )

    assert (await dashscope_image_output_probe(output_context, route)).status == "passed"
    assert (await dashscope_image_edit_probe(edit_context, route)).status == "passed"
    assert output_client.requests[0][1] == (
        "/api/v1/services/aigc/multimodal-generation/generation"
    )
    output_content = output_client.requests[0][2]["json"]["input"]["messages"][0][
        "content"
    ]
    assert output_content == [{"text": "A single blue square."}]
    edit_content = edit_client.requests[0][2]["json"]["input"]["messages"][0][
        "content"
    ]
    assert edit_content[0]["image"].startswith("data:image/png;base64,")
    assert edit_content[1] == {"text": "Keep the square blue."}


@pytest.mark.asyncio
async def test_dashscope_edit_only_model_uses_image_for_output_probe() -> None:
    png = base64.b64decode(_PNG)
    response = {
        "output": {
            "choices": [
                {
                    "message": {
                        "content": [{"image": "https://media.test/edit.png"}]
                    }
                }
            ]
        }
    }
    route = route_from_mapping(
        {
            "route_id": "qwen-image-edit",
            "kind": "official_model",
            "canonical_provider": "alibaba_cloud",
            "canonical_model_id": "qwen-image-edit",
            "protocols": [
                {
                    "protocol": "dashscope_multimodal_generation",
                    "required_scenarios": ["image_output"],
                }
            ],
            "catalog_eligible": True,
        }
    )
    client = RawScriptedClient(
        [httpx.Response(200, json=response), httpx.Response(200, content=png)]
    )
    context = ProbeContext(
        snapshot_sha256="7" * 64,
        source_revision="abc1234",
        protocol="dashscope_multimodal_generation",
        scenario=Scenario.IMAGE_OUTPUT,
        attempt=1,
        client=client,
    )

    assert (await dashscope_image_output_probe(context, route)).status == "passed"
    content = client.requests[0][2]["json"]["input"]["messages"][0]["content"]
    assert content[0]["image"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_dashscope_missing_gateway_endpoint_is_protocol_mismatch() -> None:
    route = route_from_mapping(
        {
            "route_id": "qwen-image-3.0",
            "kind": "official_model",
            "canonical_provider": "alibaba_cloud",
            "canonical_model_id": "qwen-image-3.0",
            "protocols": [
                {
                    "protocol": "dashscope_multimodal_generation",
                    "required_scenarios": ["image_output"],
                }
            ],
            "catalog_eligible": True,
        }
    )
    context = ProbeContext(
        snapshot_sha256="7" * 64,
        source_revision="abc1234",
        protocol="dashscope_multimodal_generation",
        scenario=Scenario.IMAGE_OUTPUT,
        attempt=1,
        client=RawScriptedClient([httpx.Response(404, json={"error": {}})]),
    )

    result = await dashscope_image_output_probe(context, route)

    assert result.status == "failed"
    assert result.error_kind.value == "protocol_mismatch"


@pytest.mark.asyncio
async def test_audio_output_feeds_transcription_dependency() -> None:
    wave = b"RIFF" + b"\x00" * 40 + b"WAVE"
    client = RawScriptedClient(
        [
            httpx.Response(200, content=wave),
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "probe audio",
                            }
                        }
                    ]
                },
            ),
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
    assert client.requests[1][1] == "/chat/completions"
    content = client.requests[1][2]["json"]["messages"][0]["content"]
    assert content[1]["type"] == "input_audio"
    assert content[1]["input_audio"]["format"] == "wav"


@pytest.mark.asyncio
async def test_conversational_audio_output_uses_chat_modalities() -> None:
    wave = b"RIFF" + b"\x00" * 40 + b"WAVE"
    route = route_from_mapping(
        {
            "route_id": "gpt-audio",
            "kind": "official_model",
            "canonical_provider": "openai",
            "canonical_model_id": "gpt-audio",
            "protocols": [
                {
                    "protocol": "openai_chat_completions",
                    "required_scenarios": ["audio_output"],
                }
            ],
            "catalog_eligible": True,
        }
    )
    client = RawScriptedClient(
        [
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "audio": {
                                    "data": base64.b64encode(wave).decode("ascii")
                                }
                            }
                        }
                    ]
                },
            )
        ]
    )

    result = await audio_output_probe(
        _raw_context(client, Scenario.AUDIO_OUTPUT), route
    )

    assert result.status == "passed"
    assert client.audio_fixture == wave
    body = client.requests[0][2]["json"]
    assert client.requests[0][1] == "/chat/completions"
    assert body["modalities"] == ["text", "audio"]
    assert body["audio"] == {"voice": "alloy", "format": "wav"}


@pytest.mark.asyncio
async def test_qwen_omni_audio_output_uses_streaming_chat_contract() -> None:
    pcm = b"\x00\x01\x02\x03"
    event = {
        "choices": [
            {
                "delta": {
                    "audio": {"data": base64.b64encode(pcm).decode("ascii")}
                }
            }
        ]
    }
    route = route_from_mapping(
        {
            "route_id": "qwen3.5-omni-plus",
            "kind": "official_model",
            "canonical_provider": "alibaba_cloud",
            "canonical_model_id": "qwen3.5-omni-plus",
            "protocols": [
                {
                    "protocol": "openai_chat_completions",
                    "required_scenarios": ["audio_output"],
                }
            ],
            "catalog_eligible": True,
        }
    )
    response = b"data: " + json.dumps(event).encode("utf-8") + b"\n\ndata: [DONE]\n\n"
    client = RawScriptedClient([httpx.Response(200, content=response)])

    result = await audio_output_probe(
        _raw_context(client, Scenario.AUDIO_OUTPUT), route
    )

    assert result.status == "passed"
    assert client.audio_fixture is not None
    assert client.audio_fixture.startswith(b"RIFF")
    body = client.requests[0][2]["json"]
    assert body["stream"] is True
    assert body["audio"] == {"voice": "Tina", "format": "wav"}


@pytest.mark.asyncio
async def test_qwen_omni_turbo_uses_its_supported_voice() -> None:
    event = {
        "choices": [
            {
                "delta": {
                    "audio": {"data": base64.b64encode(b"RIFFWAVE").decode("ascii")}
                }
            }
        ]
    }
    route = route_from_mapping(
        {
            "route_id": "qwen-omni-turbo",
            "kind": "official_model",
            "canonical_provider": "alibaba_cloud",
            "canonical_model_id": "qwen-omni-turbo",
            "protocols": [
                {
                    "protocol": "openai_chat_completions",
                    "required_scenarios": ["audio_output"],
                }
            ],
            "catalog_eligible": True,
        }
    )
    response = b"data: " + json.dumps(event).encode("utf-8") + b"\n\ndata: [DONE]\n\n"
    client = RawScriptedClient([httpx.Response(200, content=response)])

    result = await audio_output_probe(
        _raw_context(client, Scenario.AUDIO_OUTPUT), route
    )

    assert result.status == "passed"
    assert client.requests[0][2]["json"]["audio"] == {
        "voice": "Chelsie",
        "format": "wav",
    }


@pytest.mark.asyncio
async def test_qwen_omni_audio_input_uses_data_url_and_streaming_text() -> None:
    event = {"choices": [{"delta": {"content": "probe audio"}}]}
    response = b"data: " + json.dumps(event).encode("utf-8") + b"\n\ndata: [DONE]\n\n"
    route = route_from_mapping(
        {
            "route_id": "qwen3.5-omni-plus",
            "kind": "official_model",
            "canonical_provider": "alibaba_cloud",
            "canonical_model_id": "qwen3.5-omni-plus",
            "protocols": [
                {
                    "protocol": "openai_chat_completions",
                    "required_scenarios": ["audio_input"],
                }
            ],
            "catalog_eligible": True,
        }
    )
    client = RawScriptedClient([httpx.Response(200, content=response)])
    client.audio_fixture = b"RIFF" + b"\x00" * 40 + b"WAVE"

    result = await audio_input_probe(
        _raw_context(client, Scenario.AUDIO_INPUT), route
    )

    assert result.status == "passed"
    body = client.requests[0][2]["json"]
    audio = body["messages"][0]["content"][1]["input_audio"]
    assert audio["data"].startswith("data:;base64,")
    assert body["modalities"] == ["text"]
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_audio_input_reports_missing_generated_dependency() -> None:
    client = RawScriptedClient([])
    result = await audio_input_probe(
        _raw_context(client, Scenario.AUDIO_INPUT), _route()
    )

    assert result.status == "failed"
    assert result.error_kind.value == "dependency_failed"


@pytest.mark.asyncio
async def test_audio_captioner_receives_audio_without_a_text_part() -> None:
    route = route_from_mapping(
        {
            "route_id": "qwen3-omni-30b-a3b-captioner",
            "kind": "official_model",
            "canonical_provider": "alibaba_cloud",
            "canonical_model_id": "qwen3-omni-30b-a3b-captioner",
            "protocols": [
                {
                    "protocol": "openai_chat_completions",
                    "required_scenarios": ["audio_input"],
                }
            ],
            "catalog_eligible": True,
        }
    )
    client = RawScriptedClient(
        [
            httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "ambient workshop noise"}}
                    ]
                },
            )
        ]
    )
    client.audio_fixture = b"RIFF" + b"\x00" * 40 + b"WAVE"

    result = await audio_input_probe(
        _raw_context(client, Scenario.AUDIO_INPUT), route
    )

    assert result.status == "passed"
    content = client.requests[0][2]["json"]["messages"][0]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "input_audio"
    assert content[0]["input_audio"]["data"].startswith("data:;base64,")


@pytest.mark.asyncio
async def test_realtime_probe_exchanges_a_complete_text_turn() -> None:
    client = RawScriptedClient([])
    result = await realtime_probe(
        _raw_context(client, Scenario.REALTIME), _route()
    )

    assert result.status == "passed"
    assert client.requests[0][0] == "WS"
    assert [
        event["type"] for event in client.requests[0][2]["events"]
    ] == ["session.update", "conversation.item.create", "response.create"]


@pytest.mark.asyncio
async def test_realtime_probe_classifies_websocket_transport_failure() -> None:
    class FailingWebSocketClient(RawScriptedClient):
        async def websocket_exchange(
            self,
            path: str,
            events: Sequence[dict[str, object]],
            terminal_event_types: frozenset[str],
        ) -> tuple[object, ...]:
            raise RuntimeError("connection closed without a close frame")

    result = await realtime_probe(
        _raw_context(FailingWebSocketClient([]), Scenario.REALTIME), _route()
    )

    assert result.status == "failed"
    assert result.error_kind.value == "protocol_mismatch"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scenario", "evidence_type"),
    [
        (Scenario.IMAGE_INPUT, "response.output_text.delta"),
        (Scenario.AUDIO_INPUT, "response.output_text.delta"),
        (Scenario.AUDIO_OUTPUT, "response.output_audio.delta"),
        (Scenario.TOOLS, "response.function_call_arguments.done"),
        (Scenario.TOOL_CHOICE, "response.function_call_arguments.done"),
        (Scenario.REASONING, "response.output_text.delta"),
    ],
)
async def test_realtime_probe_uses_scenario_specific_native_events(
    scenario: Scenario, evidence_type: str
) -> None:
    client = RawScriptedClient([])
    client.websocket_events = (
        (
            {"type": "conversation.item.created"},
            {"type": "response.done"},
        )
        if scenario is Scenario.AUDIO_INPUT
        else (
            {"type": evidence_type, "delta": "probe"},
            {"type": "response.done"},
        )
    )
    context = ProbeContext(
        snapshot_sha256="7" * 64,
        source_revision="abc1234",
        protocol="openai_realtime",
        scenario=scenario,
        attempt=1,
        client=client,
    )

    result = await realtime_probe(context, _route())

    assert result.status == "passed"
    events = client.requests[0][2]["events"]
    session = events[0]["session"]
    if scenario is Scenario.AUDIO_INPUT:
        assert session["input_audio_format"] == "pcm16"
        assert events[1]["type"] == "conversation.item.create"
        content = events[1]["item"]["content"]
        assert content[0]["type"] == "input_audio"
        audio = base64.b64decode(content[0]["audio"])
        assert audio
        assert any(audio)
    if scenario is Scenario.AUDIO_OUTPUT:
        assert session["modalities"] == ["text", "audio"]
        assert session["voice"] == "alloy"
        assert session["output_audio_format"] == "pcm16"
    if scenario in {Scenario.TOOLS, Scenario.TOOL_CHOICE}:
        assert session["tools"][0]["name"] == "lookup"
    if scenario is Scenario.REASONING:
        assert session["reasoning"] == {"effort": "low"}


@pytest.mark.asyncio
async def test_video_input_uses_openai_compatible_image_sequence() -> None:
    event = {"choices": [{"delta": {"content": "No change."}}]}
    client = RawScriptedClient(
        [
            httpx.Response(
                200,
                content=(
                    b"data: "
                    + json.dumps(event).encode("utf-8")
                    + b"\n\ndata: [DONE]\n\n"
                ),
            )
        ]
    )

    result = await video_input_probe(
        _raw_context(client, Scenario.VIDEO_INPUT), _route()
    )

    assert result.status == "passed"
    body = client.requests[0][2]["json"]
    video = body["messages"][0]["content"][0]
    assert video["type"] == "video"
    assert len(video["video"]) == 4
    assert all(
        frame.startswith("data:image/png;base64,") for frame in video["video"]
    )
    assert body["modalities"] == ["text"]
    assert body["stream"] is True


@pytest.mark.asyncio
async def test_minimax_video_input_uses_an_mp4_data_url() -> None:
    event = {"choices": [{"delta": {"content": "Static blue video."}}]}
    client = RawScriptedClient(
        [
            httpx.Response(
                200,
                content=(
                    b"data: "
                    + json.dumps(event).encode("utf-8")
                    + b"\n\ndata: [DONE]\n\n"
                ),
            )
        ]
    )
    route = route_from_mapping(
        {
            "route_id": "MiniMax-M3",
            "kind": "official_model",
            "canonical_provider": "minimax",
            "canonical_model_id": "MiniMax-M3",
            "protocols": [
                {
                    "protocol": "openai_chat_completions",
                    "required_scenarios": ["video_input"],
                }
            ],
            "catalog_eligible": True,
        }
    )

    result = await video_input_probe(
        _raw_context(client, Scenario.VIDEO_INPUT), route
    )

    assert result.status == "passed"
    video = client.requests[0][2]["json"]["messages"][0]["content"][0]
    assert video["type"] == "video_url"
    assert video["video_url"]["url"].startswith("data:video/mp4;base64,")
    assert video["video_url"]["detail"] == "low"


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
            "protocols": [
                {"protocol": "serpapi_search", "required_scenarios": ["search"]}
            ],
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

    assert (await search_probe(context, route)).status == "passed"
    assert client.requests[0][0:2] == ("POST", "/alpha/search")
    assert client.requests[0][2]["json"] == {
        "model": "serpapi-google",
        "query": "Pygent agent framework",
    }

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
    empty = await search_probe(empty_context, route)
    assert empty.status == "failed"
    assert empty.error_kind.value == "invalid_response"


@pytest.mark.asyncio
async def test_openai_search_probe_uses_the_advertised_chat_endpoint() -> None:
    client = RawScriptedClient(
        [
            httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"role": "assistant", "content": "result"}}
                    ]
                },
            )
        ]
    )
    route = route_from_mapping(
        {
            "route_id": "serpapi-google",
            "kind": "external_service",
            "canonical_provider": None,
            "canonical_model_id": None,
            "protocols": [
                {
                    "protocol": "openai_chat_completions",
                    "required_scenarios": ["search"],
                }
            ],
            "catalog_eligible": False,
        }
    )
    context = ProbeContext(
        snapshot_sha256="7" * 64,
        source_revision="abc1234",
        protocol="openai_chat_completions",
        scenario=Scenario.SEARCH,
        attempt=1,
        client=client,
    )

    assert (await search_probe(context, route)).status == "passed"
    assert client.requests[0][0:2] == ("POST", "/chat/completions")
    assert client.requests[0][2]["json"]["model"] == "serpapi-google"


def test_builtin_probe_registry_covers_every_manifest_protocol_scenario() -> None:
    manifest = load_manifest()
    expected = {
        (requirements.protocol, scenario)
        for route in manifest.routes
        for requirements in route.protocols
        for scenario in requirements.required_scenarios
    }

    assert set(builtin_probe_registry().probes) == expected
