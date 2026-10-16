"""Tests for OpenAI chat-completions audio output encoding and decoding."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

import pytest

from pygent.core import Context, UserMessage, freeze_json_object
from pygent.llm import (
    DefaultModelInvoker,
    GenerationConfig,
    ModelEntry,
    ModelGroup,
    ModelModalities,
    ModelProviderError,
    OpenAICompatibleAdapter,
)
from pygent.llm._adapter_contracts import ModelProviderRequest
from pygent.llm.types import ModelFailureReason, RetryPolicy
from tests.support.model_specs import model_entry

_WAV_BYTES = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 8
_WAV_B64 = base64.b64encode(_WAV_BYTES).decode("ascii")


def _provider_request(
    *,
    entry: ModelEntry,
    message: UserMessage,
    context: Context,
    generation: GenerationConfig,
    tools: tuple[Any, ...] = (),
) -> ModelProviderRequest:
    return ModelProviderRequest(
        model_key=entry.key,
        model=entry.spec,
        message=message,
        context=context,
        generation=generation,
        tools=tools,
    )


def _audio_entry(*, with_options: bool = True):
    entry = model_entry("main", "openai", "gpt-audio")
    capabilities = replace(
        entry.spec.capabilities,
        modalities=ModelModalities(
            input=("text", "audio"),
            output=("text", "audio"),
        ),
    )
    spec = replace(
        entry.spec,
        capabilities=capabilities,
        provider_options=(
            {"audio": {"voice": "alloy", "format": "wav"}} if with_options else {}
        ),
    )
    return replace(entry, spec=spec)


def _request(entry=None, *, output_modalities: tuple[str, ...] = ()):
    return _provider_request(
        entry=entry or _audio_entry(),
        message=UserMessage(content="Say hello."),
        context=Context(),
        generation=GenerationConfig(output_modalities=output_modalities),
        tools=(),
    )


def test_chat_audio_request_encodes_modalities_and_audio_params() -> None:
    payload = OpenAICompatibleAdapter().build_request(
        _request(output_modalities=("audio",))
    ).to_dict()

    assert payload["modalities"] == ["text", "audio"]
    assert payload["audio"] == {"voice": "alloy", "format": "wav"}


def test_chat_audio_request_without_audio_opt_in_sends_no_audio_params() -> None:
    payload = OpenAICompatibleAdapter().build_request(_request()).to_dict()

    assert "modalities" not in payload
    assert "audio" not in payload


def test_chat_audio_request_requires_configured_voice() -> None:
    entry = _audio_entry(with_options=False)
    with pytest.raises(ModelProviderError, match="audio voice and format"):
        OpenAICompatibleAdapter().build_request(
            _request(entry, output_modalities=("audio",))
        )


def test_chat_audio_request_rejects_undeclared_audio_output() -> None:
    entry = model_entry("main", "openai", "gpt-test")
    with pytest.raises(ModelProviderError, match="does not declare audio output"):
        OpenAICompatibleAdapter().build_request(
            _provider_request(
                entry=entry,
                message=UserMessage(content="Say hello."),
                context=Context(),
                generation=GenerationConfig(output_modalities=("audio",)),
                tools=(),
            )
        )


def test_chat_audio_non_stream_decodes_inline_audio_and_transcript() -> None:
    payload = freeze_json_object(
        {
            "id": "chatcmpl-1",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "audio": {
                            "id": "audio-1",
                            "data": _WAV_B64,
                            "transcript": "Hello there.",
                        },
                    },
                }
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 16},
        }
    )
    response = OpenAICompatibleAdapter().parse_response(_request(), payload)

    assert response.finish_reason == "stop"
    assert response.message.content == "Hello there."
    assert [block.media_type for block in response.message.media] == ["audio"]
    block = response.message.media[0]
    assert block.mime_type == "audio/wav"
    assert block.source.base64_data is not None
    decoded = base64.b64decode(block.source.base64_data)
    assert decoded == _WAV_BYTES
    assert block.source.sha256


def test_chat_audio_non_stream_still_rejects_undeclared_audio() -> None:
    payload = freeze_json_object(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "audio": {"data": _WAV_B64},
                    },
                }
            ]
        }
    )
    with pytest.raises(ModelProviderError) as raised:
        OpenAICompatibleAdapter().parse_response(
            _request(model_entry("main", "openai", "gpt-test")), payload
        )
    assert raised.value.reason_code is ModelFailureReason.MEDIA_TRANSPORT_UNSUPPORTED


def test_chat_audio_non_stream_rejects_missing_audio_data() -> None:
    payload = freeze_json_object(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "audio": {"id": "audio-1"}},
                }
            ]
        }
    )
    with pytest.raises(ModelProviderError) as raised:
        OpenAICompatibleAdapter().parse_response(_request(), payload)
    assert raised.value.reason_code is ModelFailureReason.MEDIA_CONTENT_INVALID


def _stream_chunk(delta: dict[str, Any]) -> dict[str, object]:
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]}


def test_chat_audio_stream_emits_fragments_and_final_marker() -> None:
    adapter = OpenAICompatibleAdapter()
    request = _request(output_modalities=("audio",))
    decoder = adapter.create_stream_decoder(request)

    first = decoder.feed(freeze_json_object(_stream_chunk({"audio": {"data": _WAV_B64}})))
    second = decoder.feed(freeze_json_object(_stream_chunk({})))
    final_payload = {
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 8, "completion_tokens": 16},
    }
    third = decoder.feed(freeze_json_object(final_payload))
    tail = decoder.finish()

    deltas = [
        cast(Mapping[str, Any], part.data)
        for part in (*first, *second, *third)
        if part.kind == "media_delta"
    ]
    assert [delta["final"] for delta in deltas] == [False, True]
    assert deltas[0]["media_type"] == "audio"
    assert deltas[0]["mime_type"] == "audio/wav"
    assert deltas[0]["data"] == _WAV_B64
    assert deltas[-1]["data"] == ""
    assert any(part.kind == "finish" for part in (*third, *tail))
    assert tail == ()
    assert deltas[0]["media_index"] == 0
    assert deltas[0]["fragment_index"] == 0
    assert deltas[-1]["fragment_index"] == 1


async def test_chat_audio_streams_media_through_the_invoker() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls = 0

        async def invoke(self, model, payload):  # pragma: no cover
            raise AssertionError("invoke must not be called")

        async def aclose(self) -> None:
            return None

        async def stream(self, model, payload):
            self.calls += 1
            yield freeze_json_object(
                _stream_chunk({"audio": {"data": _WAV_B64}})
            )
            yield freeze_json_object(
                {
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 8, "completion_tokens": 16},
                }
            )

    client = FakeClient()
    invoker = DefaultModelInvoker(
        adapters={"openai_chat_completions": OpenAICompatibleAdapter()},
        clients={"main": client},
    )
    execution = invoker.execute(
        model_group=ModelGroup("voice", (_audio_entry(),)),
        retry_policy=RetryPolicy(),
        generation=GenerationConfig(output_modalities=("audio",)),
        message=UserMessage(content="Say hello."),
        context=Context(),
    )
    result = await execution.result()

    assert client.calls == 1
    assert [block.media_type for block in result.message.media] == ["audio"]
    block = result.message.media[0]
    assert block.mime_type == "audio/wav"
    assert block.source.base64_data is not None
    assert base64.b64decode(block.source.base64_data) == _WAV_BYTES


def test_chat_audio_request_options_validation_rejects_bad_audio_options() -> None:
    entry = model_entry(
        "main",
        "openai",
        "gpt-audio",
        provider_options={"audio": {"voice": ""}},
    )
    adapter = OpenAICompatibleAdapter()
    with pytest.raises(ValueError, match="audio.voice"):
        adapter.validate_model(entry.spec)
