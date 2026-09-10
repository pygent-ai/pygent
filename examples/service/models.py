"""Application-owned LLM declarations using the public Pygent SDK."""

from pygent import (
    CapabilityPresetCatalog,
    ExponentialBackoff,
    GenerationConfig,
    ModelCallLayer,
    ModelEntry,
    ModelErrorKind,
    ModelGroup,
    ModelSpec,
    RetryPolicy,
    ToolDefinition,
)
from pygent.llm.spi import ModelInvoker


def _model(name: str, provider: str, model_id: str) -> ModelEntry:
    return ModelEntry(
        name,
        ModelSpec(
            provider=provider,
            model_id=model_id,
            protocol="openai_compatible",
            capabilities=CapabilityPresetCatalog.builtin()
            .presets["text_tools_structured_reasoning"]
            .materialize(context_tokens=128_000, max_output_tokens=8_192),
        ),
    )


def build_assistant_model(
    *,
    invoker: ModelInvoker | None = None,
    tools: tuple[ToolDefinition, ...] = (),
) -> ModelCallLayer:
    return ModelCallLayer(
        model_group=ModelGroup(
            name="assistant",
            models=(
                _model(
                    "assistant-primary",
                    provider="openai",
                    model_id="assistant",
                ),
                _model(
                    "assistant-fallback",
                    provider="qwen",
                    model_id="assistant-backup",
                ),
            ),
        ),
        retry_policy=RetryPolicy(
            max_attempts_per_route=2,
            retry_on=(
                ModelErrorKind.TIMEOUT,
                ModelErrorKind.RATE_LIMIT,
                ModelErrorKind.UNAVAILABLE,
                ModelErrorKind.INCOMPLETE_RESPONSE,
            ),
            backoff=ExponentialBackoff(initial=0.2, maximum=2.0),
        ),
        generation=GenerationConfig(
            temperature=0.2,
            max_output_tokens=2048,
        ),
        tools=tools,
        invoker=invoker,
    )


def build_reviewer_model(*, invoker: ModelInvoker | None = None) -> ModelCallLayer:
    return ModelCallLayer(
        model_group=ModelGroup(
            name="reviewer",
            models=(_model("review", provider="openai", model_id="reviewer"),),
        ),
        retry_policy=RetryPolicy(
            max_attempts_per_route=1,
            retry_on=(ModelErrorKind.TIMEOUT, ModelErrorKind.UNAVAILABLE),
            backoff=ExponentialBackoff(initial=0.1, maximum=1.0),
        ),
        generation=GenerationConfig(
            temperature=0.0,
            max_output_tokens=1024,
        ),
        invoker=invoker,
    )


__all__ = ["build_assistant_model", "build_reviewer_model"]
