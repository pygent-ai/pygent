from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from .common import _hash_json, _hash_text, _now


def skill_selection_changed(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    keys = ("scope", "uri", "path", "content_hash")
    return any(str(previous.get(key) or "") != str(current.get(key) or "") for key in keys) or (
        _hash_json(previous.get("shadowed") or []) != _hash_json(current.get("shadowed") or [])
    )


def skills_fingerprint(user_skills_dir: Path, project_skills_dir: Path) -> str:
    return _hash_json(
        {"user": _skills_dir_fingerprint(user_skills_dir), "project": _skills_dir_fingerprint(project_skills_dir)}
    )


def scan_multilevel_skills(*, user_skills_dir: Path, project_skills_dir: Path) -> list[dict[str, Any]]:
    selected = {str(skill["name"]): skill for skill in _scan_scoped_skills(user_skills_dir, scope="user")}
    for project_skill in _scan_scoped_skills(project_skills_dir, scope="project"):
        name = str(project_skill["name"])
        existing = selected.get(name)
        shadowed = [] if existing is None else [
            {key: existing.get(key) for key in ("scope", "uri", "path", "content_hash")}
        ]
        selected[name] = {**project_skill, "shadowed": shadowed}
    return [selected[name] for name in sorted(selected)]


def infer_installed_cli_names(command: str) -> list[str]:
    parts = command.split()
    if len(parts) >= 4 and parts[:3] == ["npm", "install", "-g"]:
        return [part for part in parts[3:] if not part.startswith("-")]
    if len(parts) >= 3 and parts[:2] == ["npm", "i"] and "-g" in parts:
        return [part for part in parts[parts.index("-g") + 1 :] if not part.startswith("-")]
    return []


def _skills_dir_fingerprint(skills_dir: Path) -> str:
    if not skills_dir.exists():
        return _hash_json([])
    rows = [
        {"name": child.name, "has_skill": (child / "SKILL.md").is_file()}
        for child in sorted(skills_dir.iterdir(), key=lambda item: item.name)
        if child.is_dir()
    ]
    return _hash_json(rows)


def _scan_standard_skills(skills_dir: Path) -> list[dict[str, Any]]:
    if not skills_dir.exists():
        return []
    skills: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for child in sorted(skills_dir.iterdir(), key=lambda item: item.name):
        skill = _read_skill_definition(child / "SKILL.md") if child.is_dir() else None
        if skill is None or skill["name"] in seen_names:
            continue
        seen_names.add(skill["name"])
        skills.append(skill)
    return skills


def _scan_scoped_skills(skills_dir: Path, *, scope: Literal["user", "project"]) -> list[dict[str, Any]]:
    return [
        {
            **skill,
            "scope": scope,
            "uri": f"{scope}://skills/{skill['name']}/SKILL.md",
            "path": str(Path(str(skill["path"])).expanduser().resolve()),
            "shadowed": [],
        }
        for skill in _scan_standard_skills(skills_dir)
    ]


def _read_skill_definition(path: Path) -> dict[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    frontmatter = _parse_frontmatter(text)
    name = str(frontmatter.get("name") or "").strip()
    description = str(frontmatter.get("description") or "").strip()
    if not name or not description:
        return None
    return {
        "name": name,
        "description": description,
        "path": str(path),
        "content_hash": _hash_text(text),
        "detected_at": _now(),
    }


def _parse_frontmatter(text: str) -> dict[str, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    data: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            return data
        if ":" in line:
            key, value = line.split(":", 1)
            data[key.strip()] = value.strip().strip("\"'")
    return {}


__all__ = ["infer_installed_cli_names", "scan_multilevel_skills", "skill_selection_changed", "skills_fingerprint"]
