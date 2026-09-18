from __future__ import annotations

from dataclasses import replace
from io import BytesIO

from PIL import Image

from pygent import Context, ToolMessage, ToolResult
from pygent.llm import (
    AnthropicMessagesAdapter,
    DefaultModelInvoker,
    GeminiGenerateContentAdapter,
    MediaTransportCapabilities,
    ModelModalities,
    OpenAICompatibleAdapter,
    OpenAIResponsesAdapter,
)
from pygent.llm._media_tokens import (
    anthropic_media_input_tokens,
    gemini_media_input_tokens,
    generic_media_input_tokens,
    openai_media_input_tokens,
)
from pygent.tool import MediaSource, ToolResultMedia
from tests.support.model_specs import model_entry, model_group


def image(width: int, height: int, *, detail: str = "high") -> ToolResultMedia:
    stream = BytesIO()
    Image.new("RGB", (width, height), color=(10, 20, 30)).save(stream, format="PNG")
    return ToolResultMedia(
        media_type="image",
        mime_type="image/png",
        source=MediaSource.inline(stream.getvalue()),
        detail=detail,  # type: ignore[arg-type]
    )


def video(*, duration_seconds: float | None) -> ToolResultMedia:
    return ToolResultMedia(
        media_type="video",
        mime_type="video/mp4",
        source=MediaSource.inline(b"\x00\x00\x00\x18ftypmp42fixture"),
        width=1280,
        height=720,
        duration_seconds=duration_seconds,
        fps=24,
        has_audio=True,
    )


def test_openai_patch_image_estimates_follow_model_budget() -> None:
    model = model_entry("main", "openai", "gpt-5.6-sol").spec
    block = image(1024, 1024)

    assert (block.width, block.height) == (1024, 1024)
    assert openai_media_input_tokens(block, model.model_id) == 1229
    assert openai_media_input_tokens(image(2048, 2048), model.model_id) == 3000


def test_openai_tile_image_estimates_honor_detail() -> None:
    model = model_entry("main", "openai", "gpt-4o").spec

    assert (
        openai_media_input_tokens(image(1024, 1024, detail="low"), model.model_id) == 85
    )
    assert openai_media_input_tokens(image(1024, 1024), model.model_id) == 765


def test_generic_rules_cover_images_and_videos_conservatively() -> None:
    assert generic_media_input_tokens(image(1024, 1024)) == 2048
    assert generic_media_input_tokens(video(duration_seconds=2.5)) == 2500
    assert generic_media_input_tokens(video(duration_seconds=None)) == 2_000_000


def test_gemini_rules_cover_images_and_static_video() -> None:
    assert gemini_media_input_tokens(image(384, 384)) == 258
    assert gemini_media_input_tokens(image(1024, 1024)) == 1032
    assert gemini_media_input_tokens(video(duration_seconds=2.5)) == 750
    assert gemini_media_input_tokens(video(duration_seconds=None)) is None


def test_anthropic_rules_apply_current_patch_tiers() -> None:
    assert anthropic_media_input_tokens(image(1000, 1000), "claude-sonnet-4-6") == 1296
    assert anthropic_media_input_tokens(image(1920, 1080), "claude-sonnet-4-6") == 1560
    assert anthropic_media_input_tokens(image(1920, 1080), "claude-sonnet-4-7") == 2691
    assert anthropic_media_input_tokens(image(1920, 1080), "claude-opus-5") == 2691


def test_every_builtin_protocol_has_media_aware_accounting() -> None:
    block = image(1000, 1000)
    entries_and_adapters = (
        (
            "openai_chat_completions",
            model_entry("chat", "openai", "gpt-5.6-sol"),
            OpenAICompatibleAdapter(
                media_transport=MediaTransportCapabilities(
                    enabled=True, modalities=("image",)
                )
            ),
            1229,
        ),
        (
            "openai_responses",
            model_entry("responses", "openai", "gpt-5.6-sol"),
            OpenAIResponsesAdapter(),
            1229,
        ),
        (
            "anthropic_messages",
            model_entry("anthropic", "anthropic", "claude-sonnet-4-7"),
            AnthropicMessagesAdapter(),
            1296,
        ),
        (
            "gemini_generate_content",
            model_entry("gemini", "google", "gemini-3.7-flash"),
            GeminiGenerateContentAdapter(),
            1032,
        ),
    )
    for protocol, source_entry, adapter, expected in entries_and_adapters:
        entry = replace(
            source_entry,
            spec=replace(
                source_entry.spec,
                protocol=protocol,
                capabilities=replace(
                    source_entry.spec.capabilities,
                    modalities=ModelModalities(
                        input=("text", "image"), output=("text",)
                    ),
                ),
            ),
        )
        invoker = DefaultModelInvoker(
            adapters={protocol: adapter},
            clients={entry.key: object()},  # type: ignore[arg-type]
        )

        assert (
            invoker.estimate_media_input_tokens(
                model_group=model_group(protocol, (entry,)),
                message=_media_message(block),
                context=Context(),
            )
            == expected
        )


def test_gemini_pre_v3_function_results_project_media_away() -> None:
    source_entry = model_entry("gemini", "google", "gemini-2.5-pro")
    entry = replace(
        source_entry,
        spec=replace(
            source_entry.spec,
            protocol="gemini_generate_content",
            capabilities=replace(
                source_entry.spec.capabilities,
                modalities=ModelModalities(input=("text", "image"), output=("text",)),
            ),
        ),
    )
    invoker = DefaultModelInvoker(
        adapters={"gemini_generate_content": GeminiGenerateContentAdapter()},
        clients={entry.key: object()},  # type: ignore[arg-type]
    )

    assert (
        invoker.estimate_media_input_tokens(
            model_group=model_group("gemini", (entry,)),
            message=_media_message(image(1000, 1000)),
            context=Context(),
        )
        == 0
    )


def test_invoker_uses_provider_rules_on_the_actual_eligible_routes() -> None:
    patch_entry = model_entry("patch", "openai", "gpt-5.6-sol")
    tile_entry = model_entry("tile", "openai", "gpt-4o")
    entries = tuple(
        replace(
            entry,
            spec=replace(
                entry.spec,
                capabilities=replace(
                    entry.spec.capabilities,
                    modalities=ModelModalities(
                        input=("text", "image"), output=("text",)
                    ),
                ),
            ),
        )
        for entry in (patch_entry, tile_entry)
    )
    adapter = OpenAICompatibleAdapter(
        media_transport=MediaTransportCapabilities(
            enabled=True,
            modalities=("image",),
        )
    )
    invoker = DefaultModelInvoker(
        adapters={"openai_chat_completions": adapter},
        clients={entry.key: object() for entry in entries},  # type: ignore[arg-type]
    )
    message = _media_message(image(1024, 1024))

    assert (
        invoker.estimate_media_input_tokens(
            model_group=model_group("vision", entries),
            message=message,
            context=Context(),
        )
        == 1229
    )


def test_invoker_uses_generic_video_rule_for_compatible_vendor_adapter() -> None:
    entry = model_entry("video", "aliyun", "qwen-vl-plus")
    entry = replace(
        entry,
        spec=replace(
            entry.spec,
            capabilities=replace(
                entry.spec.capabilities,
                modalities=ModelModalities(input=("text", "video"), output=("text",)),
            ),
        ),
    )
    invoker = DefaultModelInvoker(
        adapters={
            "openai_chat_completions": OpenAICompatibleAdapter(
                media_transport=MediaTransportCapabilities(
                    enabled=True, modalities=("video",)
                )
            )
        },
        clients={entry.key: object()},  # type: ignore[arg-type]
    )

    assert (
        invoker.estimate_media_input_tokens(
            model_group=model_group("video", (entry,)),
            message=_media_message(video(duration_seconds=3)),
            context=Context(),
        )
        == 3000
    )


def test_invoker_counts_no_media_tokens_when_routing_projects_media_away() -> None:
    entry = model_entry("text", "openai", "gpt-5.6-sol")
    invoker = DefaultModelInvoker(
        adapters={"openai_chat_completions": OpenAICompatibleAdapter()},
        clients={entry.key: object()},  # type: ignore[arg-type]
    )

    assert (
        invoker.estimate_media_input_tokens(
            model_group=model_group("text", (entry,)),
            message=_media_message(video(duration_seconds=3)),
            context=Context(),
        )
        == 0
    )


def _media_message(block: ToolResultMedia) -> ToolMessage:
    return ToolMessage(
        results=(
            ToolResult(
                call_id="media",
                name="read_media",
                status="succeeded",
                content=(block,),
            ),
        )
    )
