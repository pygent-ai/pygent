"""Multimodal model output: final message, streaming, wire, and routing."""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import replace

import pytest

from pygent import (
    Context,
    GenerationConfig,
    ModelCallError,
    ModelGroup,
    RetryPolicy,
    UserMessage,
)
from pygent.core import AIMessage, FrozenJsonObject, freeze_json_object
from pygent.llm import (
    BuiltinModelProtocol,
    DefaultModelInvoker,
    GeminiGenerateContentAdapter,
    ModelErrorKind,
    ModelModalities,
    ModelStreamingCapabilities,
    OpenAICompatibleAdapter,
)
from pygent.llm._adapter_contracts import ModelProviderRequest, ModelProviderStreamPart
from pygent.llm._media_routing import (
    assistant_media_blocks,
    assistant_media_replay_gaps,
    project_request_for_model,
)
from pygent.llm._stream_accumulator import ModelStreamAccumulator
from pygent.llm.types import ModelFailureReason, ModelProviderError
from pygent.runtime.codec import WireCodecError, message_from_dict, message_to_dict
from pygent.tool import MediaBlock, MediaSource
from tests.support.model_specs import model_entry


def _image_block(data: bytes = b"image-bytes") -> MediaBlock:
    return MediaBlock(
        media_type="image",
        mime_type="image/png",
        source=MediaSource.inline(data),
    )


def _audio_block(data: bytes = b"audio-bytes") -> MediaBlock:
    return MediaBlock(
        media_type="audio",
        mime_type="audio/mpeg",
        source=MediaSource.inline(data),
        duration_seconds=1.5,
    )


def _gemini_image_spec(name: str = "main"):
    entry = model_entry(name, "google", "gemini-3.7-flash-image")
    return replace(
        entry.spec,
        protocol="gemini_generate_content",
        capabilities=replace(
            entry.spec.capabilities,
            modalities=ModelModalities(input=("text", "image"), output=("image",)),
            streaming=ModelStreamingCapabilities(output=()),
        ),
    )


def _gemini_tts_spec(name: str = "main"):
    entry = model_entry(name, "google", "gemini-3.1-flash-tts-preview")
    return replace(
        entry.spec,
        protocol="gemini_generate_content",
        capabilities=replace(
            entry.spec.capabilities,
            modalities=ModelModalities(input=("text",), output=("audio",)),
            streaming=ModelStreamingCapabilities(output=()),
        ),
    )


def _openai_spec(name: str = "main"):
    entry = model_entry(name, "openai", "gpt-test")
    return replace(entry.spec, protocol="openai_chat_completions")


def _gemini_request(
    *,
    context: Context | None = None,
    generation: GenerationConfig | None = None,
    model=None,
) -> ModelProviderRequest:
    return ModelProviderRequest(
        model_key="main",
        model=model or _gemini_image_spec(),
        message=UserMessage(content="question"),
        context=context or Context(),
        generation=generation or GenerationConfig(max_output_tokens=64),
        tools=(),
    )


# --- core types ---------------------------------------------------------


def test_audio_media_block_accepts_duration_and_rejects_video_metadata() -> None:
    block = _audio_block()
    assert block.duration_seconds == 1.5
    with pytest.raises(ValueError):
        MediaBlock(
            media_type="audio",
            mime_type="audio/mpeg",
            source=MediaSource.inline(b"audio"),
            fps=24.0,
        )


def test_assistant_message_carries_ordered_media() -> None:
    message = AIMessage(
        content="generated",
        media=(_image_block(), _audio_block()),
    )
    assert tuple(block.media_type for block in message.media) == ("image", "audio")
    with pytest.raises(TypeError):
        AIMessage(content="bad", media=("not-a-block",))  # type: ignore[arg-type]


# --- wire codec ---------------------------------------------------------


def test_assistant_media_survives_wire_roundtrip() -> None:
    message = AIMessage(
        content="generated",
        media=(_image_block(), _audio_block()),
        metadata={"model_key": "main"},
    )
    restored = message_from_dict(message_to_dict(message))
    assert type(restored) is AIMessage
    assert restored == message


def test_wire_codec_reads_legacy_assistant_records_without_media() -> None:
    record = message_to_dict(AIMessage(content="legacy"))
    assert record["media"] == []
    restored = message_from_dict(record)
    assert type(restored) is AIMessage
    assert restored.media == ()


def test_wire_codec_rejects_unknown_assistant_fields() -> None:
    record = message_to_dict(AIMessage(content="legacy"))
    record["unexpected"] = True
    with pytest.raises(WireCodecError):
        message_from_dict(record)


# --- model layer effects -------------------------------------------------


def test_model_layer_effect_roundtrips_assistant_media() -> None:
    from pygent.llm.layer import _message_effect_value, _message_from_effect

    message = AIMessage(content="generated", media=(_image_block(),))
    restored = _message_from_effect(
        freeze_json_object(
            {
                "outcome": "succeeded",
                "message": _message_effect_value(message),
                "usage": {},
                "provider_request_id": None,
            }
        )
    )
    assert restored == message


def test_model_layer_effect_rejects_unknown_message_fields() -> None:
    from pygent.llm.layer import _message_from_effect

    with pytest.raises(TypeError):
        _message_from_effect(
            freeze_json_object(
                {
                    "outcome": "succeeded",
                    "message": {"role": "assistant", "mystery": True},
                    "usage": {},
                    "provider_request_id": None,
                }
            )
        )


# --- stream accumulator --------------------------------------------------


def _consume(
    accumulator: ModelStreamAccumulator, *parts: ModelProviderStreamPart
) -> list[tuple[str, dict[str, object]]]:
    events: list[tuple[str, dict[str, object]]] = []

    async def run() -> None:
        async def sink(kind: str, data: FrozenJsonObject) -> None:
            events.append((kind, dict(data)))

        for part in parts:
            await accumulator.consume(part, sink)

    asyncio.run(run())
    return events


def _finish(
    accumulator: ModelStreamAccumulator,
) -> tuple[object, list[tuple[str, dict[str, object]]]]:
    events: list[tuple[str, dict[str, object]]] = []

    async def run() -> object:
        async def sink(kind: str, data: FrozenJsonObject) -> None:
            events.append((kind, dict(data)))

        return await accumulator.finish(sink)

    return asyncio.run(run()), events


def _media_part(**overrides: object) -> ModelProviderStreamPart:
    data: dict[str, object] = {
        "model_key": "main",
        "attempt": 1,
        "provider": "google",
        "model_id": "gemini-3.7-flash-image",
        "media_type": "image",
        "mime_type": "image/png",
        "data": _image_block(b"png-data").source.base64_data,
    }
    data.update(overrides)
    return ModelProviderStreamPart("media", data)


def test_accumulator_collects_media_parts_into_final_message() -> None:
    accumulator = ModelStreamAccumulator(generation=GenerationConfig(), tools=())
    block = _image_block(b"png-data")
    events = _consume(
        accumulator,
        _media_part(),
        ModelProviderStreamPart(
            "finish", {"model_key": "main", "attempt": 1, "finish_reason": "stop"}
        ),
    )
    response, finish_events = _finish(accumulator)
    media_events = [
        data for kind, data in (*events, *finish_events) if kind == "model.media"
    ]
    assert len(media_events) == 1
    assert media_events[0]["media_index"] == 0
    assert media_events[0]["mime_type"] == "image/png"
    assert media_events[0]["sha256"] == block.source.sha256
    assert media_events[0]["size_bytes"] == len(b"png-data")
    assert media_events[0]["data"] == block.source.base64_data
    assert media_events[0]["mime_parameters"] == ""
    assert response.message.media == (block,)  # type: ignore[attr-defined]
    assert accumulator.media_provider == "google"


def test_accumulator_reset_clears_media_state() -> None:
    accumulator = ModelStreamAccumulator(generation=GenerationConfig(), tools=())
    _consume(
        accumulator,
        _media_part(),
        ModelProviderStreamPart("reset", {"public_output": False}),
    )
    assert accumulator.media == []
    assert accumulator.media_provider is None


def test_accumulator_warns_when_media_arrives_with_schema() -> None:
    accumulator = ModelStreamAccumulator(
        generation=GenerationConfig(
            response_schema={"type": "object", "properties": {}}
        ),
        tools=(),
    )
    _consume(
        accumulator,
        ModelProviderStreamPart(
            "text", {"text": json.dumps({"schema": True}, separators=(",", ":"))}
        ),
        _media_part(),
    )
    _response, events = _finish(accumulator)
    warnings = [data for kind, data in events if kind == "model.capability.warning"]
    assert warnings
    assert warnings[0]["missing_capabilities"] == ("structured_output.media_present",)


def test_accumulator_merges_media_shards_into_one_block() -> None:
    accumulator = ModelStreamAccumulator(generation=GenerationConfig(), tools=())
    first = base64.b64encode(b"pcm-chunk-one-").decode("ascii")
    second = base64.b64encode(b"pcm-chunk-two").decode("ascii")
    events = _consume(
        accumulator,
        ModelProviderStreamPart(
            "media_delta",
            {
                "model_key": "main",
                "attempt": 1,
                "provider": "google",
                "model_id": "tts",
                "media_index": 0,
                "fragment_index": 0,
                "media_type": "audio",
                "mime_type": "audio/l16",
                "mime_parameters": "rate=24000; channels=1",
                "data": first,
                "final": False,
            },
        ),
        ModelProviderStreamPart(
            "media_delta",
            {
                "model_key": "main",
                "attempt": 1,
                "provider": "google",
                "model_id": "tts",
                "media_index": 0,
                "fragment_index": 1,
                "media_type": "audio",
                "mime_type": "audio/l16",
                "mime_parameters": "rate=24000; channels=1",
                "data": second,
                "final": True,
            },
        ),
    )
    response, finish_events = _finish(accumulator)
    block = response.message.media[0]  # type: ignore[attr-defined]
    assert block.media_type == "audio"
    assert block.mime_type == "audio/l16"
    assert block.mime_parameters == "rate=24000; channels=1"
    assert block.source.size_bytes == len(b"pcm-chunk-one-pcm-chunk-two")
    assert accumulator.media_shards == {}
    delta_events = [data for kind, data in events if kind == "model.media.delta"]
    assert [data["fragment_index"] for data in delta_events] == [0, 1]
    assert [data["final"] for data in delta_events] == [False, True]
    complete_events = [
        data for kind, data in (*events, *finish_events) if kind == "model.media"
    ]
    assert len(complete_events) == 1
    assert complete_events[0]["data"] == base64.b64encode(
        b"pcm-chunk-one-pcm-chunk-two"
    ).decode("ascii")
    assert complete_events[0]["mime_parameters"] == "rate=24000; channels=1"


def test_accumulator_rejects_out_of_order_media_fragments() -> None:
    accumulator = ModelStreamAccumulator(generation=GenerationConfig(), tools=())
    with pytest.raises(TypeError):
        _consume(
            accumulator,
            ModelProviderStreamPart(
                "media_delta",
                {
                    "media_index": 0,
                    "fragment_index": 3,
                    "media_type": "audio",
                    "mime_type": "audio/l16",
                    "data": "aaaa",
                    "final": False,
                },
            ),
        )


def test_accumulator_reset_discards_pending_media_shards() -> None:
    accumulator = ModelStreamAccumulator(generation=GenerationConfig(), tools=())
    _consume(
        accumulator,
        ModelProviderStreamPart(
            "media_delta",
            {
                "media_index": 0,
                "fragment_index": 0,
                "media_type": "audio",
                "mime_type": "audio/l16",
                "data": "aaaa",
                "final": False,
            },
        ),
        ModelProviderStreamPart("reset", {"public_output": False}),
    )
    assert accumulator.media_shards == {}
    assert accumulator.media == []


def test_finish_rejects_unfinished_media_shards() -> None:
    accumulator = ModelStreamAccumulator(generation=GenerationConfig(), tools=())
    _consume(
        accumulator,
        ModelProviderStreamPart(
            "media_delta",
            {
                "model_key": "main",
                "attempt": 1,
                "media_index": 0,
                "fragment_index": 0,
                "media_type": "audio",
                "mime_type": "audio/l16",
                "data": "aaaa",
                "final": False,
            },
        ),
    )
    with pytest.raises(ModelCallError) as raised:
        _finish(accumulator)
    assert raised.value.kind is ModelErrorKind.INVALID_RESPONSE


def test_finish_succeeds_after_reset_discards_pending_shards() -> None:
    accumulator = ModelStreamAccumulator(generation=GenerationConfig(), tools=())
    _consume(
        accumulator,
        ModelProviderStreamPart(
            "media_delta",
            {
                "model_key": "main",
                "attempt": 1,
                "media_index": 0,
                "fragment_index": 0,
                "media_type": "audio",
                "mime_type": "audio/l16",
                "data": "aaaa",
                "final": False,
            },
        ),
        ModelProviderStreamPart("reset", {"public_output": False}),
        # The retry attempt re-establishes attempt identity on its first part.
        ModelProviderStreamPart(
            "text", {"model_key": "main", "attempt": 2, "text": "retry"}
        ),
    )
    response, _events = _finish(accumulator)
    assert response.message.media == ()  # type: ignore[attr-defined]
    assert accumulator.media_shards == {}


def test_stream_delivers_complete_video_block() -> None:
    accumulator = ModelStreamAccumulator(generation=GenerationConfig(), tools=())
    video = MediaBlock(
        media_type="video",
        mime_type="video/mp4",
        source=MediaSource.inline(b"mp4-bytes"),
    )
    events = _consume(
        accumulator,
        ModelProviderStreamPart(
            "media",
            {
                "model_key": "main",
                "attempt": 1,
                "provider": "google",
                "model_id": "video-model",
                "media_type": "video",
                "mime_type": "video/mp4",
                "data": video.source.base64_data,
            },
        ),
    )
    response, finish_events = _finish(accumulator)
    assert response.message.media == (video,)  # type: ignore[attr-defined]
    complete_events = [
        data for kind, data in (*events, *finish_events) if kind == "model.media"
    ]
    assert complete_events[0]["media_type"] == "video"
    assert complete_events[0]["size_bytes"] == len(b"mp4-bytes")


def test_model_media_delta_event_payload_validates() -> None:
    from pygent.llm import ModelStreamEvent

    event = ModelStreamEvent(
        "model.media.delta",
        {
            "model_key": "main",
            "attempt": 1,
            "media_index": 0,
            "fragment_index": 0,
            "media_type": "audio",
            "mime_type": "audio/l16",
            "data": "aaaa",
            "final": False,
        },
    )
    assert event.kind == "model.media.delta"
    with pytest.raises(ValueError):
        ModelStreamEvent(
            "model.media.delta",
            {
                "model_key": "main",
                "attempt": 1,
                "media_index": 0,
                "fragment_index": 0,
                "media_type": "audio",
                "mime_type": "audio/l16",
                "final": False,
            },
        )


def test_accumulator_rejects_mismatched_media_part() -> None:
    accumulator = ModelStreamAccumulator(generation=GenerationConfig(), tools=())
    with pytest.raises(TypeError):
        _consume(
            accumulator,
            ModelProviderStreamPart(
                "media",
                {"media_type": "image", "mime_type": "audio/mpeg", "data": "aa"},
            ),
        )


# --- gemini non-stream and stream ----------------------------------------


def _gemini_response_payload(*parts: dict[str, object]) -> FrozenJsonObject:
    return freeze_json_object(
        {
            "candidates": [
                {
                    "content": {"role": "model", "parts": list(parts)},
                    "finishReason": "STOP",
                }
            ]
        }
    )


def test_gemini_non_stream_decodes_mixed_text_image_audio() -> None:
    adapter = GeminiGenerateContentAdapter()
    response = adapter.parse_response(
        _gemini_request(),
        _gemini_response_payload(
            {"text": "draft"},
            {"inlineData": {"mimeType": "image/png", "data": "cG5n"}},
            {"inlineData": {"mimeType": "audio/mpeg", "data": "bXAz"}},
        ),
    )
    assert response.message.content == "draft"
    assert tuple(block.media_type for block in response.message.media) == (
        "image",
        "audio",
    )
    assert response.message.media[0].source.sha256


def test_gemini_non_stream_decodes_parameterized_audio_mime() -> None:
    adapter = GeminiGenerateContentAdapter()
    response = adapter.parse_response(
        _gemini_request(),
        _gemini_response_payload(
            {
                "inlineData": {
                    "mimeType": "audio/l16; rate=24000; channels=1",
                    "data": "bXAz",
                }
            },
        ),
    )
    assert tuple(block.media_type for block in response.message.media) == ("audio",)
    assert response.message.media[0].mime_type == "audio/l16"
    assert response.message.media[0].mime_parameters == "rate=24000; channels=1"


def test_gemini_request_maps_output_modalities_to_response_modalities() -> None:
    adapter = GeminiGenerateContentAdapter()
    payload = adapter.build_request(
        _gemini_request(
            generation=GenerationConfig(output_modalities=("image",)),
        )
    ).to_dict()
    assert payload["generationConfig"]["responseModalities"] == ["IMAGE", "TEXT"]
    tts_payload = adapter.build_request(
        _gemini_request(
            generation=GenerationConfig(output_modalities=("audio",)),
            model=_gemini_tts_spec(),
        )
    ).to_dict()
    assert tts_payload["generationConfig"]["responseModalities"] == ["AUDIO"]
    plain = adapter.build_request(_gemini_request()).to_dict()
    assert "responseModalities" not in plain.get("generationConfig", {})


def test_gemini_request_maps_speech_config_provider_option() -> None:
    adapter = GeminiGenerateContentAdapter()
    spec = replace(
        _gemini_tts_spec(),
        provider_options={
            "speech_config": {
                "voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}}
            }
        },
    )
    payload = adapter.build_request(
        _gemini_request(
            generation=GenerationConfig(output_modalities=("audio",)),
            model=spec,
        )
    ).to_dict()
    assert payload["generationConfig"]["speechConfig"] == {
        "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}
    }
    with pytest.raises(ValueError):
        adapter.validate_model(
            replace(
                spec,
                provider_options={
                    "speech_config": {"voice_config": {"prebuilt_voice_config": {}}}
                },
            )
        )


def test_gemini_rejects_no_image_finish_reason() -> None:
    adapter = GeminiGenerateContentAdapter()
    with pytest.raises(ModelProviderError) as raised:
        adapter.parse_response(
            _gemini_request(),
            freeze_json_object(
                {
                    "candidates": [
                        {"finishReason": "NO_IMAGE", "index": 0},
                    ]
                }
            ),
        )
    assert raised.value.reason_code is ModelFailureReason.OUTPUT_MEDIA_NOT_PRODUCED


def test_gemini_stream_decoder_shards_streaming_audio() -> None:
    adapter = GeminiGenerateContentAdapter()
    decoder = adapter.create_stream_decoder(_gemini_request())
    parts: list[ModelProviderStreamPart] = []
    for chunk in (
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "audio/l16; rate=24000; channels=1",
                                    "data": "AAAA",
                                }
                            }
                        ],
                    }
                }
            ]
        },
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "audio/l16; rate=24000; channels=1",
                                    "data": "QQ==",
                                }
                            }
                        ],
                    },
                    "finishReason": "STOP",
                }
            ]
        },
    ):
        parts.extend(decoder.feed(freeze_json_object(chunk)))
    kinds = [part.kind for part in parts]
    assert kinds == [
        "media_delta",
        "media_delta",
        "media_delta",
        "finish",
    ]
    deltas = [part for part in parts if part.kind == "media_delta"]
    assert [delta.data["fragment_index"] for delta in deltas] == [0, 1, 2]
    assert [delta.data["final"] for delta in deltas] == [False, False, True]
    assert deltas[0].data["data"] == "AAAA"
    assert deltas[1].data["data"] == "QQ=="
    assert deltas[2].data["data"] == ""
    assert deltas[0].data["mime_parameters"] == "rate=24000; channels=1"

    accumulator = ModelStreamAccumulator(generation=GenerationConfig(), tools=())
    # The invoker injects attempt identity and provider identity into every
    # part before the accumulator sees it; mirror that here.
    injected = []
    for part in parts:
        data = dict(part.data)
        data.setdefault("model_key", "main")
        data.setdefault("attempt", 1)
        if part.kind == "media_delta":
            data.setdefault("provider", "google")
            data.setdefault("model_id", "tts")
        injected.append(ModelProviderStreamPart(part.kind, data))
    _consume(accumulator, *injected)
    response, _events = _finish(accumulator)
    block = response.message.media[0]  # type: ignore[attr-defined]
    assert block.mime_type == "audio/l16"
    assert block.mime_parameters == "rate=24000; channels=1"
    assert block.source.size_bytes == len(b"\x00\x00\x00" + b"A")


def test_gemini_stream_decoder_emits_media_parts() -> None:
    adapter = GeminiGenerateContentAdapter()
    decoder = adapter.create_stream_decoder(_gemini_request())
    parts = decoder.feed(
        freeze_json_object(
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {"text": "draft"},
                                {
                                    "inlineData": {
                                        "mimeType": "image/png",
                                        "data": "cG5n",
                                    }
                                },
                            ],
                        },
                        "finishReason": "STOP",
                    }
                ]
            }
        )
    )
    kinds = [part.kind for part in parts]
    assert kinds == ["text", "media", "finish"]
    media_part = parts[1]
    assert media_part.data["media_type"] == "image"
    assert media_part.data["mime_type"] == "image/png"
    assert media_part.data["data"] == "cG5n"


def test_gemini_stream_decoder_rejects_unknown_parts() -> None:
    adapter = GeminiGenerateContentAdapter()
    decoder = adapter.create_stream_decoder(_gemini_request())
    with pytest.raises(ModelProviderError):
        decoder.feed(
            freeze_json_object(
                {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [{"fileData": {"mimeType": "image/png"}}],
                            }
                        }
                    ]
                }
            )
        )


_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAE"
    "hQGAhKmMIQAAAABJRU5ErkJggg=="
)


def test_gemini_request_replays_assistant_media_for_compatible_models() -> None:
    adapter = GeminiGenerateContentAdapter()
    media = _image_block(base64.b64decode(_PNG_BASE64))
    request = _gemini_request(
        context=Context(messages=(AIMessage(content="draft", media=(media,)),)),
    )
    payload = adapter.build_request(request).to_dict()
    history = payload["contents"][0]
    assert history["role"] == "model"
    assert history["parts"][0] == {"text": "draft"}
    assert history["parts"][1]["inlineData"] == {
        "mimeType": "image/png",
        "data": media.source.base64_data,
    }


# --- replay routing -------------------------------------------------------


def test_assistant_media_replay_gaps_distinguish_protocol_support() -> None:
    image = _image_block(b"png-data")
    assert (
        assistant_media_replay_gaps(
            (image,), model=_gemini_image_spec(), adapter=GeminiGenerateContentAdapter()
        )
        == ()
    )
    assert assistant_media_replay_gaps(
        (_audio_block(),),
        model=_gemini_image_spec(),
        adapter=GeminiGenerateContentAdapter(),
    ) == (
        "assistant_media_replay.media_transport.model",
        "modalities.input.audio",
    )
    assert assistant_media_replay_gaps(
        (image,), model=_openai_spec(), adapter=OpenAICompatibleAdapter()
    ) == ("assistant_media_replay.protocol", "modalities.input.image")


def test_assistant_media_blocks_collect_only_assistant_history() -> None:
    context = Context(
        messages=(
            UserMessage(content="hi", media=(_image_block(),)),
            AIMessage(content="done", media=(_audio_block(),)),
        )
    )
    blocks = assistant_media_blocks(context)
    assert tuple(block.media_type for block in blocks) == ("audio",)


def test_fallback_projection_replaces_assistant_media_with_not_viewed_notes() -> None:
    media = _image_block(b"png-data")
    message = UserMessage(content="question")
    context = Context(messages=(AIMessage(content="draft", media=(media,)),))
    projected_message, projected_context = project_request_for_model(
        message,
        context,
        model=_openai_spec(),
        adapter=OpenAICompatibleAdapter(),
    )
    assert projected_context.messages[0].media == ()
    assert "not_viewed" in projected_context.messages[0].content
    assert projected_message is message


# --- capability gate ------------------------------------------------------


def test_missing_capabilities_honors_requested_output_modalities() -> None:
    from pygent.llm.invoker import _missing_capabilities

    model = _gemini_image_spec()
    assert (
        _missing_capabilities(
            model,
            generation=GenerationConfig(output_modalities=("image",)),
            tools=(),
        )
        == ()
    )
    assert "modalities.output.text" in _missing_capabilities(
        model,
        generation=GenerationConfig(),
        tools=(),
    )
    assert "modalities.output.video" in _missing_capabilities(
        model,
        generation=GenerationConfig(output_modalities=("video",)),
        tools=(),
    )


def test_invoker_rejects_schema_combined_with_media_output() -> None:
    entry = model_entry("main", "openai", "gpt-test")
    invoker = DefaultModelInvoker(
        adapters={BuiltinModelProtocol.OPENAI_CHAT_COMPLETIONS: OpenAICompatibleAdapter()},
        clients={},
    )

    async def run() -> None:
        execution = invoker.execute(
            model_group=ModelGroup("main", (entry,)),
            retry_policy=RetryPolicy(),
            generation=GenerationConfig(
                response_schema={"type": "object", "properties": {}},
                output_modalities=("image",),
            ),
            message=UserMessage(content="hi"),
            context=Context(),
        )
        async with execution.subscribe() as subscription:
            async for _event in subscription:
                pass
        await execution.result()

    with pytest.raises(ModelCallError) as raised:
        asyncio.run(run())
    assert raised.value.kind is ModelErrorKind.INVALID_REQUEST
