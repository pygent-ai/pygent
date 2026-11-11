"""Build a ready-to-use ReActAgent from a model and two prompts."""

from __future__ import annotations

from pygent import (
    CapabilityPresetCatalog,
    GenerationConfig,
    ModelCallLayer,
    ModelEntry,
    ModelGroup,
    ModelSpec,
    ReActAgent,
    RetryPolicy,
    ToolCallLayer,
)
from pygent.llm.spi import ModelInvoker

_MODEL_GROUP_NAME = "basic-agent"
_MODEL_ENTRY_KEY = "primary"
_CONTEXT_WINDOW_TOKENS = 128_000

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Answer concisely and honestly."
DEFAULT_COMPRESSION_PROMPT = (
    "Create a compact continuation for the foreground agent. "
    "Preserve goals, constraints, decisions, unfinished work, relevant files "
    "and tool state. Return only the continuation text."
)


def build_agent(
    *,
    model_name: str,
    invoker: ModelInvoker | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    compression_prompt: str = DEFAULT_COMPRESSION_PROMPT,
) -> ReActAgent:
    """Assemble a ready-to-use ReActAgent.

    Only the model name and the two prompts are visible knobs. Retry,
    generation, compression, and ReAct budgets are fixed internally so the
    example stays deterministic and easy to understand.
    """

    capabilities = (
        CapabilityPresetCatalog.builtin()
        .presets["text_tools_structured_reasoning"]
        .materialize(
            context_tokens=_CONTEXT_WINDOW_TOKENS,
            max_output_tokens=8_192,
        )
    )
    model_group = ModelGroup(
        name=_MODEL_GROUP_NAME,
        models=(
            ModelEntry(
                _MODEL_ENTRY_KEY,
                ModelSpec(
                    provider="openai",
                    model_id=model_name,
                    protocol="openai_chat_completions",
                    capabilities=capabilities,
                ),
            ),
        ),
    )
    foreground = ModelCallLayer(
        model_group=model_group,
        retry_policy=RetryPolicy(max_attempts_per_model=2),
        generation=GenerationConfig(
            temperature=0.2,
            max_output_tokens=8192,
        ),
        tools=(),
        invoker=invoker,
    )
    compressor = ModelCallLayer(
        model_group=model_group,
        retry_policy=RetryPolicy(max_attempts_per_model=2),
        generation=GenerationConfig(
            temperature=0.0,
            max_output_tokens=4096,
        ),
        tools=(),
        invoker=invoker,
    )
    return ReActAgent(
        system_prompt=system_prompt,
        compression_prompt=compression_prompt,
        model=foreground,
        compressor=compressor,
        tools=ToolCallLayer(tools=()),
        context_window_tokens=_CONTEXT_WINDOW_TOKENS,
        compression_trigger_ratio=0.9,
        compression_context_window_tokens=_CONTEXT_WINDOW_TOKENS,
        max_compressions=4,
        max_steps=16,
        max_model_calls=16,
        max_tool_calls=64,
    )


__all__ = [
    "DEFAULT_COMPRESSION_PROMPT",
    "DEFAULT_SYSTEM_PROMPT",
    "build_agent",
]
