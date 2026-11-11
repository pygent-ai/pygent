"""Runnable single-turn and interactive paths for the basic agent example."""

from __future__ import annotations

from pygent import ReActAgentContext, UserMessage

from .agent import (
    DEFAULT_COMPRESSION_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    build_agent,
)
from .providers import LiveModelConfig, build_live_invoker


async def run_once(
    *,
    model_config: LiveModelConfig,
    question: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    compression_prompt: str = DEFAULT_COMPRESSION_PROMPT,
) -> str:
    """Answer one question and return the agent text."""

    invoker = build_live_invoker(model_config)
    try:
        agent = build_agent(
            model_name=model_config.model_name,
            invoker=invoker,
            system_prompt=system_prompt,
            compression_prompt=compression_prompt,
        )
        context = agent.new_context()
        answer, _ = await agent.invoke(UserMessage(content=question), context)
        return answer.content
    finally:
        await invoker.aclose()


async def run_interactive(
    *,
    model_config: LiveModelConfig,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    compression_prompt: str = DEFAULT_COMPRESSION_PROMPT,
) -> None:
    """Run a REPL that accumulates context and reports compression state."""

    invoker = build_live_invoker(model_config)
    try:
        agent = build_agent(
            model_name=model_config.model_name,
            invoker=invoker,
            system_prompt=system_prompt,
            compression_prompt=compression_prompt,
        )
        context = agent.new_context()
        print("Enter questions; an empty line or 'quit' exits.")
        while True:
            try:
                line = input("you> ").strip()
            except EOFError:
                break
            if not line or line.lower() in {"quit", "exit"}:
                break
            answer, context = await agent.invoke(
                UserMessage(content=line),
                context,
            )
            print(f"agent> {answer.content}")
            if isinstance(context, ReActAgentContext):
                print(
                    "  [last_input_tokens="
                    f"{context.last_input_tokens}, "
                    f"compressions={context.compression_count}]"
                )
    finally:
        await invoker.aclose()


__all__ = ["run_interactive", "run_once"]
