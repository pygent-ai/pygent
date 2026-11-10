from __future__ import annotations

from html import escape
from typing import Any

from pygent.lara.core.prompt import Prompt
from pygent.lara.graph.skill_catalog import (
    scan_multilevel_skills,
    skill_selection_changed,
    skills_fingerprint,
)

from .models import ReminderScope, ReminderSection


def collect_initial_skills(
    scope: ReminderScope,
) -> tuple[ReminderSection | None, dict[str, Any]]:
    skills = scan_multilevel_skills(
        user_skills_dir=scope.user_skills_dir,
        project_skills_dir=scope.project_skills_dir,
    )
    known = {str(skill["name"]): skill for skill in skills}
    lines = _render_skills(skills, tag="available-skills")
    if lines:
        lines = list(_wrap_skills(lines))
    state = {
        "fingerprint": skills_fingerprint(
            scope.user_skills_dir, scope.project_skills_dir
        ),
        "known": known,
    }
    return (
        ReminderSection("skills.context", 20, tuple(lines)) if lines else None
    ), state


def detect_skill_changes(
    scope: ReminderScope, state: dict[str, Any]
) -> ReminderSection | None:
    fingerprint = skills_fingerprint(scope.user_skills_dir, scope.project_skills_dir)
    if fingerprint == state.get("fingerprint"):
        return None
    known = dict(state.get("known") or {})
    current = scan_multilevel_skills(
        user_skills_dir=scope.user_skills_dir,
        project_skills_dir=scope.project_skills_dir,
    )
    changed: list[dict[str, Any]] = []
    for skill in current:
        name = str(skill.get("name") or "")
        if name and (name not in known or skill_selection_changed(known[name], skill)):
            changed.append(skill)
        if name:
            known[name] = skill
    state.update({"fingerprint": fingerprint, "known": known})
    lines = _render_skills(changed, tag="new-skills")
    if lines:
        lines = list(_wrap_skills(lines))
    return ReminderSection("skills.context", 20, tuple(lines)) if lines else None


def _wrap_skills(lines: list[str]) -> tuple[str, ...]:
    return tuple(
        str(Prompt("\n".join(lines)).wrap("skills-context", indent="  ")).splitlines()
    )


def _render_skills(skills: list[dict[str, Any]], *, tag: str) -> list[str]:
    if not skills:
        return []
    lines = [f"<{tag}>"]
    for skill in skills:
        name = escape(str(skill.get("name") or ""), quote=True)
        description = escape(str(skill.get("description") or ""), quote=False)
        path = escape(str(skill.get("path") or ""), quote=False)
        source = escape(str(skill.get("source") or ""), quote=True)
        lines.extend(
            [
                f'  <skill name="{name}" source="{source}">',
                f"    <description>{description}</description>",
                f"    <path>{path}</path>",
                "  </skill>",
            ]
        )
    lines.append(f"</{tag}>")
    return lines
