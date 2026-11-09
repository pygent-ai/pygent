"""Session-derived fork context windows.

Both fork paths need a block-1 history prefix, but they must not share a
lifecycle: realtime fork-replies read the live session projection (durable
checkpoint plus uncovered tail), while memory extractors consume a frozen
session range selected by the eternal-conversation cursor. :func:`session_fork_window`
is the single runtime entry point for both modes.

The returned :class:`~lara.agents.context.ContextWindow` is the fork
primitive owned by the agents scaffold; this module lives in the runtime so
``lara.agents`` never gains a runtime dependency.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

from pygent import freeze_json_object
from pygent.core import JsonValue
from pygent.lara.agents.context import ContextWindow, messages_from_history
from pygent.lara.schema import AgentSession, CaseRunRef, RunConfig

from .agent.common import _initial_lara_context
from .agent.pipeline import render_model_request_text
from .context import LaraContext
from .eternal_conversation import eternal_enabled_for, load_projection

ForkContextMode = Literal["live", "frozen"]


def restore_session_context(
    *,
    session: AgentSession,
    run_ref: CaseRunRef,
    turn_id: str,
    config: RunConfig,
    carry_context: bool = True,
) -> LaraContext:
    """Rebuild one session projection from its durable checkpoint and tail."""

    memory_projection = (
        load_projection(session.session_dir) if carry_context else {}
    )
    lara_context = LaraContext(
        session_id=session.session_id,
        case_id=run_ref.case_id,
        case_run_id=run_ref.case_run_id,
        run_dir=run_ref.run_dir,
        turn_id=turn_id,
        system_prompt=session.system_prompt,
        eternal_memory_enabled=eternal_enabled_for(session, config),
        memory_covered_through=int(memory_projection.get("covered_through") or 0),
        memory_projection=freeze_json_object(memory_projection),
        raw_history_location=str(
            Path(session.session_dir) / "raw-history" / "events.jsonl"
        ),
    )
    history, _ = _initial_lara_context(
        context=lara_context,
        history=session.history,
        checkpoint=session.metadata.get("agent_context"),
        checkpoint_history_count=session.metadata.get(
            "agent_context_history_count"
        ),
    )
    return replace(
        history,
        metadata={
            "session_id": run_ref.session_id,
            "case_run_id": run_ref.case_run_id,
            "persist_conversation_history": carry_context,
            "model_group_name": session.model_group_name,
            "selected_model_key": session.selected_model_key,
        },
    )


def session_fork_window(
    *,
    mode: ForkContextMode,
    config: RunConfig | None = None,
    session: AgentSession | None = None,
    run_ref: CaseRunRef | None = None,
    turn_id: str | None = None,
    system_prompt: str | None = None,
    history_range: Sequence[Mapping[str, Any]] = (),
) -> ContextWindow:
    """Build the fork block-1 window for the live or frozen lifecycle.

    ``live`` reads the current session projection and renders the exact
    model-request system prompt the foreground would use. Its
    ``system_prompt`` and ``messages`` are byte-identical to calling
    :func:`restore_session_context` and
    :func:`~lara.runtime.agent.pipeline.render_model_request_text` directly;
    this preserves the fork-reply cache prefix.

    ``frozen`` pairs the persisted ``session.json`` system prompt supplied by
    the caller with an explicitly selected persisted history range. It reads no
    live session state, so repeated calls for the same input are deterministic
    and safe for the memory extractor's frozen coverage interval.
    """

    if mode == "live":
        if config is None or session is None or run_ref is None or turn_id is None:
            raise ValueError(
                "live mode requires config, session, run_ref, and turn_id"
            )
        context = restore_session_context(
            session=session,
            run_ref=run_ref,
            turn_id=turn_id,
            config=config,
        )
        return ContextWindow(
            system_prompt=render_model_request_text(config, context, None),
            messages=tuple(context.messages),
            # LaraContext.metadata 类型标注是 Mapping[str, object]，但实际只存
            # FrozenJsonObject（JSON）值；与 AgentRunResult.window() 同一处理。
            metadata=cast("Mapping[str, JsonValue]", dict(context.metadata)),
        )
    if mode == "frozen":
        if system_prompt is None or not system_prompt.strip():
            raise ValueError("frozen mode requires a non-empty system_prompt")
        return ContextWindow(
            system_prompt=system_prompt,
            messages=messages_from_history(history_range),
        )
    raise ValueError(f"unsupported fork context mode: {mode!r}")


@dataclass(frozen=True)
class SessionForkWindowSource:
    """Structural ``ContextSource`` for a frozen session window.

    Runtime code owns this adapter, so the agents scaffold only sees the
    ``resolve()``/``describe()`` protocol and never imports runtime state.
    ``describe()`` is deterministic and feeds the existing
    :class:`~lara.agents.fork.ForkSnapshot` audit record.
    """

    window: ContextWindow

    def resolve(self) -> ContextWindow:
        return self.window

    def describe(self) -> str:
        return f"session-frozen:{len(self.window.messages)}"


__all__ = [
    "ForkContextMode",
    "SessionForkWindowSource",
    "restore_session_context",
    "session_fork_window",
]
