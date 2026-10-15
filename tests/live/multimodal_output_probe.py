"""Opt-in live smoke probe for multimodal model OUTPUT through the AZ gateway.

Exercises the three changed paths against real Gemini image routes:
1. non-stream: real inlineData decodes into AIMessage.media;
2. replay: assistant media is encoded back into model-role history;
3. stream: inlineData produces MEDIA stream parts and a media-bearing final message.

Prints only sanitized output (mime types, hashes, sizes - never keys or base64).
Loads credentials from the process environment and then the repository .env.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import replace
from pathlib import Path

from pygent import Context, GenerationConfig, UserMessage
from pygent.core import AIMessage
from pygent.llm import (
    GeminiGenerateContentAdapter,
    GeminiGenerateContentClient,
    ModelModalities,
    ModelProviderStreamPart,
    ModelSpec,
    ModelStreamingCapabilities,
)
from pygent.llm._stream_accumulator import ModelStreamAccumulator
from pygent.tool import MediaBlock


def _load_dotenv() -> None:
    path = Path(__file__).resolve().parents[2] / ".env"
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if name and name not in os.environ:
            os.environ[name] = value.strip().strip("\"'")


def _spec(model_id: str) -> ModelSpec:
    from pygent.llm import CapabilityPresetCatalog

    capabilities = CapabilityPresetCatalog.builtin().presets[
        "text"
    ].materialize(context_tokens=1_000_000, max_output_tokens=32768)
    capabilities = replace(
        capabilities,
        modalities=ModelModalities(input=("text", "image"), output=("image", "audio")),
        streaming=ModelStreamingCapabilities(output=()),
    )
    provider_options: dict[str, object] = {}
    if "tts" in model_id.lower():
        provider_options["speech_config"] = {
            "voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}}
        }
    return ModelSpec(
        provider="google",
        model_id=model_id,
        protocol="gemini_generate_content",
        provider_options=provider_options,
        capabilities=capabilities,
    )


def _prompt(model_id: str) -> str:
    if "tts" in model_id.lower():
        return "Hello! This is a live multimodal output test from Pygent."
    return (
        "Draw one solid red circle on a white background. "
        "No text in the image."
    )


def _summary(block: MediaBlock) -> str:
    source = block.source
    return (
        f"{block.media_type}/{block.mime_type} sha256={str(source.sha256)[:12]} "
        f"size={source.size_bytes}"
    )


async def _step_non_stream(
    adapter: GeminiGenerateContentAdapter,
    client: GeminiGenerateContentClient,
    model_id: str,
) -> AIMessage:
    request = _build_request(
        adapter,
        _spec(model_id),
        UserMessage(content=_prompt(model_id)),
        model_id=model_id,
    )
    payload = await client.invoke(request.model, request.payload)
    response = adapter.parse_response(request, payload)
    print(f"[1] non-stream finish={response.finish_reason}")
    for block in response.message.media:
        print(f"    media: {_summary(block)}")
    print(f"    content: {response.message.content[:80]!r}")
    if not response.message.media:
        raise SystemExit("[1] FAIL: no media decoded from real response")
    print("[1] PASS")
    return response.message


def _generation(model_id: str) -> GenerationConfig:
    modality = "audio" if "tts" in model_id.lower() else "image"
    return GenerationConfig(output_modalities=(modality,))


def _build_request(
    adapter: GeminiGenerateContentAdapter,
    model: ModelSpec,
    message,
    context: Context | None = None,
    model_id: str = "",
):
    from pygent.llm import ModelProviderRequest

    request = ModelProviderRequest(
        model_key="live-probe",
        model=model,
        message=message,
        context=context or Context(),
        generation=_generation(model_id),
        tools=(),
    )
    return _FrozenRequest(
        model_key=request.model_key,
        model=request.model,
        message=request.message,
        context=request.context,
        generation=request.generation,
        tools=request.tools,
        media_projections=request.media_projections,
        payload=adapter.build_request(request),
    )


class _FrozenRequest:
    """Adapter request plus its built wire payload for direct client calls."""

    def __init__(
        self,
        *,
        model_key: str,
        model: ModelSpec,
        message,
        context,
        generation,
        tools,
        media_projections,
        payload,
    ) -> None:
        self.model_key = model_key
        self.model = model
        self.message = message
        self.context = context
        self.generation = generation
        self.tools = tools
        self.media_projections = media_projections
        self.payload = payload

    def __getattr__(self, name: str):  # pragma: no cover - delegation
        raise AttributeError(name)


async def _step_replay(
    adapter: GeminiGenerateContentAdapter,
    client: GeminiGenerateContentClient,
    model_id: str,
    assistant: AIMessage,
) -> None:
    if "tts" in model_id.lower():
        print("[2] SKIP: TTS models cannot accept audio input; "
              "routing would filter this replay before the adapter")
        return
    request = _build_request(
        adapter,
        _spec(model_id),
        UserMessage(content="Thanks. Now briefly describe in text what you just drew."),
        context=Context(messages=(assistant,)),
        model_id=model_id,
    )
    payload = await client.invoke(request.model, request.payload)
    response = adapter.parse_response(request, payload)
    print(f"[2] replay finish={response.finish_reason}")
    print(f"    content: {response.message.content[:120]!r}")
    print("[2] PASS: endpoint accepted model-role inlineData history")


async def _step_stream(
    adapter: GeminiGenerateContentAdapter,
    client: GeminiGenerateContentClient,
    model_id: str,
) -> None:
    request = _build_request(
        adapter,
        _spec(model_id),
        UserMessage(content="Draw one solid blue triangle. No text in the image."),
        model_id=model_id,
    )
    decoder = adapter.create_stream_decoder(request)
    accumulator = ModelStreamAccumulator(
        generation=GenerationConfig(), tools=()
    )
    media_parts = 0
    media_delta_parts = 0
    delta_sizes: list[int] = []

    async def sink(kind: str, data: object) -> None:
        nonlocal media_parts, media_delta_parts
        if kind == "model.media":
            media_parts += 1
        elif kind == "model.media.delta":
            media_delta_parts += 1
            payload = dict(data)  # type: ignore[arg-type]
            if not payload.get("final"):
                delta_sizes.append(len(str(payload.get("data", ""))))

    async for payload in client.stream(request.model, request.payload):
        for part in decoder.feed(payload):
            part = _inject(part)
            await accumulator.consume(part, sink)
    for part in decoder.finish():
        part = _inject(part)
        await accumulator.consume(part, sink)
    response = await accumulator.finish(sink)
    print(f"[3] stream media_parts={media_parts} media_delta_parts={media_delta_parts}")
    if delta_sizes:
        print(
            f"    delta base64 sizes: n={len(delta_sizes)} "
            f"first={delta_sizes[:6]}"
        )
    for block in response.message.media:
        print(f"    media: {_summary(block)}")
    print(f"    content: {response.message.content[:80]!r}")
    if not response.message.media:
        raise SystemExit("[3] FAIL: stream produced no final media")
    print("[3] PASS")


def _inject(part: ModelProviderStreamPart) -> ModelProviderStreamPart:
    if part.kind == "reset" or part.kind == "continuation":
        return part
    data = dict(part.data)
    data.setdefault("model_key", "live-probe")
    data.setdefault("attempt", 1)
    if part.kind == "media":
        data.setdefault("provider", "google")
        data.setdefault("model_id", "live-probe")
    return ModelProviderStreamPart(part.kind, data)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-2.5-flash-image")
    parser.add_argument("--skip-stream", action="store_true")
    args = parser.parse_args()

    _load_dotenv()
    base_url = os.environ.get("AZ_BASE_URL", "").strip().rstrip("/")
    api_key = os.environ.get("AZ_API_KEY", "").strip()
    if not base_url or not api_key:
        raise SystemExit("AZ_BASE_URL / AZ_API_KEY are not configured")
    root = base_url.removesuffix("/v1")
    client = GeminiGenerateContentClient(
        base_url=root + "/v1beta", api_key=api_key
    )
    adapter = GeminiGenerateContentAdapter()
    try:
        assistant = await _step_non_stream(adapter, client, args.model)
        await _step_replay(adapter, client, args.model, assistant)
        if not args.skip_stream:
            await _step_stream(adapter, client, args.model)
    finally:
        await client.aclose()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
