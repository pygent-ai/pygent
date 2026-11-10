from __future__ import annotations

import hashlib
from datetime import datetime
from html import escape

from pygent.lara.core.prompt import Prompt
from pygent.lara.graph.common import _now

from .cli_context import collect_initial_cli
from .git_context import capture_git_snapshot, render_git_section, snapshot_state
from .models import InitialSnapshot, ReminderScope, ReminderSection
from .rendering import render_context_body
from .skills_context import collect_initial_skills


def build_initial_snapshot(
    scope: ReminderScope,
) -> tuple[InitialSnapshot, dict[str, object]]:
    prepared_at = _now()
    cli_section, cli_state = collect_initial_cli(scope.cli_bash_presets)
    skills_section, skills_state = collect_initial_skills(scope)
    git_snapshot = capture_git_snapshot(scope.workspace_root)
    git_section = (
        render_git_section(git_snapshot, reason="session-baseline")
        if git_snapshot is not None
        else None
    )
    environment_body = "\n".join(
        (
            f'  <session id="{escape(scope.session_id, quote=True)}" />',
            f"  <workspace-root>{escape(str(scope.workspace_root), quote=False)}</workspace-root>",
        )
    )
    environment_section = ReminderSection(
        source="runtime.context",
        order=0,
        lines=tuple(str(Prompt(environment_body).wrap("runtime-context")).splitlines()),
        include_time=True,
    )
    sections = [
        section
        for section in (environment_section, cli_section, skills_section, git_section)
        if section is not None
    ]
    content = (
        render_context_body(sections, rendered_at=datetime.fromisoformat(prepared_at)) or ""
    )
    snapshot_id = (
        "initial-"
        + hashlib.sha256(
            f"{scope.session_id}:{prepared_at}:{content}".encode()
        ).hexdigest()[:16]
    )
    observations: dict[str, object] = {
        "cli": cli_state,
        "skills": skills_state,
        "git": (
            {
                "repository_root": str(git_snapshot["repository_root"]),
                "last_observed_hash": str(git_snapshot["fingerprint"]),
                "last_injected_hash": str(git_snapshot["fingerprint"]),
                "last_checked_at": str(git_snapshot["captured_at"]),
                "last_checked_epoch": 0.0,
                "last_check_duration_ms": 0.0,
                "last_snapshot": snapshot_state(git_snapshot),
            }
            if git_snapshot is not None
            else {}
        ),
    }
    return InitialSnapshot(
        snapshot_id, scope.session_id, prepared_at, content
    ), observations
