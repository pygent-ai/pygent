"""Command-line entry point for the progressive tutorial."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence

from .providers import LiveModelConfig, OfflineModelInvoker, build_live_invoker
from .runner import close_invoker, run_direct_demo, run_managed_demo


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Pygent Agent tutorial")
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("offline", "stream", "live", "managed"),
        default="offline",
    )
    parser.add_argument(
        "--profile",
        choices=("quick", "quality"),
        default="quality",
        help="managed mode profile override",
    )
    return parser


async def _run(mode: str, profile: str) -> None:
    if mode == "managed":
        result, quick, quality = await run_managed_demo(selected_profile=profile)
        print(f"answer: {result.answer}")
        print(f"profiles closed: quick={quick.closed}, quality={quality.closed}")
        return

    if mode == "live":
        config = LiveModelConfig.from_environment()
        invoker = build_live_invoker(config)
        model_name = config.model_name
        stream = False
    else:
        invoker = OfflineModelInvoker()
        model_name = "offline-tutorial"
        stream = mode == "stream"

    try:
        result = await run_direct_demo(
            invoker,
            model_name=model_name,
            stream=stream,
        )
    finally:
        await close_invoker(invoker)
    print(f"answer: {result.answer}")
    print(f"context messages: {len(result.context.messages)}")
    if result.event_kinds:
        print(f"events: {', '.join(result.event_kinds)}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    asyncio.run(_run(args.mode, args.profile))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
