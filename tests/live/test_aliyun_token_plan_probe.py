from __future__ import annotations

import json

import httpx
import pytest

from pygent.llm import ModelCapabilityCatalog
from tests.live.aliyun_token_plan_probe import (
    ProbeResult,
    TokenPlanProbeRunner,
    probe_inventory,
    summarize_result,
)


def _catalog_model_ids() -> set[str]:
    return {
        model_id
        for provider, model_id, _protocol in ModelCapabilityCatalog.builtin().models
        if provider == "aliyun_token_plan"
    }


def test_probe_inventory_matches_builtin_token_plan_catalog() -> None:
    inventory = probe_inventory()

    assert len(inventory) == 18
    assert len({probe.model_id for probe in inventory}) == 18
    assert {probe.model_id for probe in inventory} == _catalog_model_ids()


def test_probe_summary_never_contains_secret_or_response_content() -> None:
    summary = summarize_result(
        ProbeResult(
            model_id="qwen3.8-max",
            protocol="openai_chat_completions",
            status="passed",
            error_kind=None,
            private_detail="secret output",
        )
    )

    assert "secret output" not in json.dumps(summary)
    assert set(summary) == {"model_id", "protocol", "status", "error_kind"}


@pytest.mark.asyncio
async def test_http_probe_paths_and_minimal_payloads() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = request.url.path
        if path.endswith("/chat/completions"):
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
        if path.endswith("/v1/messages"):
            return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}]})
        if path.endswith("/video-synthesis"):
            return httpx.Response(200, json={"output": {"task_id": "task-1"}})
        if path.endswith("/tasks/task-1"):
            return httpx.Response(200, json={"output": {"task_status": "SUCCEEDED", "video_url": "https://example.test/video.mp4"}})
        if path.endswith("/SpeechSynthesizer"):
            return httpx.Response(
                200,
                json={"output": {"audio": {"url": "http://example.test/audio.wav"}}},
            )
        if path.endswith("/audio.wav"):
            return httpx.Response(200, content=b"audio")
        if path.endswith("/multimodal-generation/generation"):
            body = json.loads(request.content)
            if body["model"] == "qwen-audio-3.0-asr-flash":
                return httpx.Response(200, json={"output": {"text": "ok"}})
            return httpx.Response(200, json={"output": {"choices": [{"message": {"content": [{"image": "https://example.test/image.png"}]}}]}})
        raise AssertionError(f"unexpected path: {path}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        runner = TokenPlanProbeRunner(
            client=client,
            api_key="private",
            anthropic_api_key="private-anthropic",
            openai_base_url="https://example.test/compatible-mode/v1",
            anthropic_base_url="https://example.test/apps/anthropic",
            poll_interval=0,
        )
        assert (await runner.probe_openai("qwen3.8-max")).status == "passed"
        assert (await runner.probe_anthropic("qwen3.8-max")).status == "passed"
        image_result, image_url = await runner.probe_image("wan2.7-image")
        assert image_result.status == "passed"
        assert image_url == "https://example.test/image.png"
        assert (await runner.probe_video("happyhorse-1.1-t2v")).status == "passed"
        tts_result, audio = await runner.probe_tts()
        assert tts_result.status == "passed"
        assert (await runner.probe_asr(audio)).status == "passed"

    paths = [request.url.path for request in requests]
    assert paths == [
        "/compatible-mode/v1/chat/completions",
        "/apps/anthropic/v1/messages",
        "/api/v1/services/aigc/multimodal-generation/generation",
        "/api/v1/services/aigc/video-generation/video-synthesis",
        "/api/v1/tasks/task-1",
        "/api/v1/services/audio/tts/SpeechSynthesizer",
        "/audio.wav",
        "/api/v1/services/aigc/multimodal-generation/generation",
    ]
    payloads = [json.loads(request.content) for request in requests if request.content]
    assert payloads[0] == {
        "model": "qwen3.8-max",
        "messages": [{"role": "user", "content": "Reply OK."}],
        "max_tokens": 8,
        "stream": False,
    }
    assert payloads[1]["max_tokens"] == 8
    assert payloads[2]["parameters"] == {"size": "1024*1024"}
    assert payloads[3]["parameters"] == {
        "resolution": "480P",
        "ratio": "1:1",
        "duration": 3,
        "watermark": False,
    }
    assert payloads[-1]["model"] == "qwen-audio-3.0-asr-flash"
    assert payloads[-1]["input"]["messages"][0]["content"][0]["type"] == "input_audio"


class _FakeWebSocket:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback

    async def recv(self) -> str:
        return '{"type":"session.created"}'


@pytest.mark.asyncio
async def test_realtime_probe_uses_injected_websocket_connector() -> None:
    calls: list[tuple[str, dict[str, str]]] = []

    def connect(url: str, headers: dict[str, str]):
        calls.append((url, headers))
        return _FakeWebSocket()

    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: None)) as client:
        runner = TokenPlanProbeRunner(
            client=client,
            api_key="private",
            anthropic_api_key="private-anthropic",
            openai_base_url="https://example.test/compatible-mode/v1",
            anthropic_base_url="https://example.test/apps/anthropic",
            websocket_connect=connect,
        )
        result = await runner.probe_realtime()

    assert result.status == "passed"
    assert calls == [
        (
            "wss://example.test/api-ws/v1/realtime?model=qwen-audio-3.0-realtime-plus",
            {"Authorization": "Bearer private"},
        )
    ]
