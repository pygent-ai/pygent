"""Application-owned LLM declarations using the public Pygent SDK."""

from pygent import (
    ExponentialBackoff,
    FallbackPolicy,
    GenerationConfig,
    ModelCallLayer,
    ModelErrorKind,
    ModelGroupConfig,
    ModelRoute,
    RetryPolicy,
    ToolDefinition,
)
from pygent.llm.spi import ModelInvoker


def build_assistant_model(
    *,
    invoker: ModelInvoker | None = None,
    tools: tuple[ToolDefinition, ...] = (),
) -> ModelCallLayer:
    return ModelCallLayer(
        model_group=ModelGroupConfig(
            name="assistant",
            routes=(
                ModelRoute(
                    "assistant-primary",
                    provider="openai",
                    model="assistant",
                ),
                ModelRoute(
                    "assistant-fallback",
                    provider="qwen",
                    model="assistant-backup",
                ),
            ),
            fallback=FallbackPolicy(order=("assistant-primary", "assistant-fallback")),
            max_concurrency=32,
        ),
        retry_policy=RetryPolicy(
            max_attempts_per_route=2,
            retry_on=(
                ModelErrorKind.TIMEOUT,
                ModelErrorKind.RATE_LIMIT,
                ModelErrorKind.UNAVAILABLE,
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
        model_group=ModelGroupConfig(
            name="reviewer",
            routes=(ModelRoute("review", provider="openai", model="reviewer"),),
            fallback=FallbackPolicy(order=("review",)),
            max_concurrency=8,
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
