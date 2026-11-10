from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from pygent import AIMessage, Message, Module
from pygent.core import EffectSafety, ExecutionRequirements, RecoverySafety
from pygent.lara.core.prompt import Prompt
from pygent.lara.runtime.context import LaraContext
from pygent.lara.runtime.context_compression import (
    TRANSCRIPT_POINTER,
    collect_recent_file_reads,
    parse_summary,
    render_file_read_block,
)
from pygent.lara.schema import RunConfig

from .common import _session_dir_for_run


class LaraCompressorModule(Module[Message, AIMessage]):
    """Add Lara file evidence to Pygent's native compression result."""

    execution_requirements = ExecutionRequirements(
        recovery_safety=RecoverySafety.MODULE_BOUNDARY_RETRY,
        effect_safety=EffectSafety.MANAGED_EFFECTS,
    )
    trusted_live_resource_attributes = ("config", "model")

    def __init__(self, config: RunConfig, model: Module[Message, AIMessage]) -> None:
        super().__init__()
        self.config = config
        self.model = model

    async def forward(self, request: Message, context: LaraContext) -> tuple[AIMessage, LaraContext]:
        for _ in range(10):
            answer, returned = await self.model(request, context)
            if returned != context:
                raise RuntimeError("Lara compressor model must preserve its fork context")
            summary = parse_summary(answer.content, has_tool_call=bool(answer.tool_calls))
            if summary is not None:
                break
        else:
            raise RuntimeError("context compression summary parsing failed after 10 attempts")

        session_dir = _session_dir_for_run(Path(context.run_dir))
        file_read_xml, file_read_metadata = render_file_read_block(
            collect_recent_file_reads(
                session_dir,
                count=self.config.context_compression_file_read_count,
            ),
            max_chars=self.config.context_compression_file_read_max_chars,
        )
        continuation = str(
            Prompt(
                "\n".join(
                    (
                        summary,
                        "",
                        file_read_xml,
                        "",
                        TRANSCRIPT_POINTER,
                        f"{session_dir / 'session.json'} (the authoritative transcript is in its history field)",
                    )
                )
            ).wrap("session-context")
        )
        await self.emit(
            kind="lara.context.compressed",
            data={
                "compression_number": context.compression_count + 1,
                "source_projection_revision": context.projection_revision,
                "file_reads": file_read_metadata,
            },
        )
        return replace(answer, content=continuation, tool_calls=()), context
