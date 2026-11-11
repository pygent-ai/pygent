"""Command-line entry point for the basic ReActAgent example."""

from __future__ import annotations

import argparse
import asyncio
import os
from collections.abc import Sequence

from .agent import DEFAULT_COMPRESSION_PROMPT, DEFAULT_SYSTEM_PROMPT
from .providers import LiveModelConfig
from .runner import run_interactive, run_once


def _prompt(name: str, default: str) -> str:
    return os.environ.get(name, "").strip() or default


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run ReActAgent with a user-provided OpenAI-compatible model, "
            "system prompt, and compression prompt"
        )
    )
    parser.add_argument(
        "message",
        nargs="?",
        default=None,
        help="single-turn question; omit to start an interactive session",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="override PYGENT_SYSTEM_PROMPT",
    )
    parser.add_argument(
        "--compression-prompt",
        default=None,
        help="override PYGENT_COMPRESSION_PROMPT",
    )
    return parser


async def _run(args: argparse.Namespace) -> None:
    config = LiveModelConfig.from_environment()
    system_prompt = args.system_prompt or _prompt(
        "PYGENT_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT
    )
    compression_prompt = args.compression_prompt or _prompt(
        "PYGENT_COMPRESSION_PROMPT", DEFAULT_COMPRESSION_PROMPT
    )
    if args.message:
        answer = await run_once(
            model_config=config,
            question=args.message,
            system_prompt=system_prompt,
            compression_prompt=compression_prompt,
        )
        print(f"agent> {answer}")
    else:
        await run_interactive(
            model_config=config,
            system_prompt=system_prompt,
            compression_prompt=compression_prompt,
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    asyncio.run(_run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
