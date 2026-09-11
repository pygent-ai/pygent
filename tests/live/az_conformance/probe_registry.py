from __future__ import annotations

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
    gemini_json_object_probe,
    gemini_json_schema_probe,
    gemini_reasoning_probe,
    gemini_stream_probe,
    gemini_text_probe,
    gemini_tool_choice_probe,
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
    openai_stream_probe,
    openai_text_probe,
    openai_tool_choice_probe,
    openai_tool_probe,
)
from tests.live.az_conformance.runner import ProbeRegistry
from tests.live.az_conformance.schemas import Scenario
from tests.live.az_conformance.search_probes import search_probe


def builtin_probe_registry() -> ProbeRegistry:
    return ProbeRegistry(
        {
            ("openai_chat_completions", Scenario.TEXT): openai_text_probe,
            ("openai_chat_completions", Scenario.TEXT_STREAM): openai_stream_probe,
            ("openai_chat_completions", Scenario.TOOLS): openai_tool_probe,
            ("openai_chat_completions", Scenario.TOOL_CHOICE): openai_tool_choice_probe,
            ("openai_chat_completions", Scenario.JSON_OBJECT): openai_json_object_probe,
            ("openai_chat_completions", Scenario.JSON_SCHEMA): openai_json_schema_probe,
            ("openai_chat_completions", Scenario.REASONING): openai_reasoning_probe,
            ("openai_chat_completions", Scenario.IMAGE_INPUT): openai_image_input_probe,
            ("openai_chat_completions", Scenario.IMAGE_OUTPUT): image_output_probe,
            ("openai_chat_completions", Scenario.IMAGE_EDIT): image_edit_probe,
            ("openai_chat_completions", Scenario.VIDEO_OUTPUT): video_output_probe,
            ("openai_chat_completions", Scenario.AUDIO_OUTPUT): audio_output_probe,
            ("openai_chat_completions", Scenario.AUDIO_INPUT): audio_input_probe,
            ("openai_chat_completions", Scenario.REALTIME): realtime_probe,
            ("openai_chat_completions", Scenario.EMBEDDING): embedding_probe,
            ("openai_chat_completions", Scenario.SEARCH): search_probe,
            ("anthropic_messages", Scenario.TEXT): anthropic_text_probe,
            ("anthropic_messages", Scenario.TEXT_STREAM): anthropic_stream_probe,
            ("anthropic_messages", Scenario.TOOLS): anthropic_tool_probe,
            ("anthropic_messages", Scenario.TOOL_CHOICE): anthropic_tool_choice_probe,
            ("anthropic_messages", Scenario.JSON_OBJECT): anthropic_json_object_probe,
            ("anthropic_messages", Scenario.JSON_SCHEMA): anthropic_json_schema_probe,
            ("anthropic_messages", Scenario.REASONING): anthropic_reasoning_probe,
            ("anthropic_messages", Scenario.IMAGE_INPUT): anthropic_image_input_probe,
            ("gemini_generate_content", Scenario.TEXT): gemini_text_probe,
            ("gemini_generate_content", Scenario.TEXT_STREAM): gemini_stream_probe,
            ("gemini_generate_content", Scenario.TOOLS): gemini_tool_probe,
            ("gemini_generate_content", Scenario.TOOL_CHOICE): gemini_tool_choice_probe,
            ("gemini_generate_content", Scenario.JSON_OBJECT): gemini_json_object_probe,
            ("gemini_generate_content", Scenario.JSON_SCHEMA): gemini_json_schema_probe,
            ("gemini_generate_content", Scenario.REASONING): gemini_reasoning_probe,
            ("gemini_generate_content", Scenario.IMAGE_INPUT): gemini_image_input_probe,
            ("gemini_generate_content", Scenario.IMAGE_OUTPUT): image_output_probe,
            ("gemini_generate_content", Scenario.IMAGE_EDIT): image_edit_probe,
            ("gemini_generate_content", Scenario.AUDIO_OUTPUT): audio_output_probe,
            ("serpapi_search", Scenario.SEARCH): search_probe,
        }
    )


__all__ = ["builtin_probe_registry"]
