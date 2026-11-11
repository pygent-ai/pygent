from __future__ import annotations

from pathlib import Path
from typing import Any

from pygent.agent.pygent_v1.core.paths import project_pygent_root

from .prompt_models import PromptRenderContext


def _prompt_render_context_payload(ctx: PromptRenderContext) -> dict[str, Any]:
    return {
        "session_id": ctx.session_id,
        "workspace_root": str(ctx.workspace_root),
        "session_dir": str(ctx.session_dir),
        "user_pygent_root": str(_ctx_user_pygent_root(ctx)),
        "project_pygent_root": str(_ctx_project_pygent_root(ctx)),
        "user_skills_dir": str(_ctx_user_skills_dir(ctx)),
        "project_skills_dir": str(_ctx_project_skills_dir(ctx)),
        "turn_id": ctx.turn_id,
        "projection": ctx.projection,
        "tool_names": ctx.tool_names,
        "request_id": ctx.request_id,
        "request_type": ctx.request_type,
    }


def _ctx_project_pygent_root(ctx: PromptRenderContext) -> Path:
    return (
        (
            ctx.project_pygent_root
            or project_pygent_root(ctx.workspace_root, _ctx_user_pygent_root(ctx))
        )
        .expanduser()
        .resolve()
    )


def _ctx_user_pygent_root(ctx: PromptRenderContext) -> Path:
    return (ctx.user_pygent_root or Path.home() / ".pygent_v1").expanduser().resolve()


def _ctx_project_skills_dir(ctx: PromptRenderContext) -> Path:
    return (
        (ctx.project_skills_dir or _ctx_project_pygent_root(ctx) / "skills")
        .expanduser()
        .resolve()
    )


def _ctx_user_skills_dir(ctx: PromptRenderContext) -> Path:
    return (
        (ctx.user_skills_dir or _ctx_user_pygent_root(ctx) / "skills")
        .expanduser()
        .resolve()
    )
