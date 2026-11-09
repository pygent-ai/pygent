from __future__ import annotations

import shutil
from collections.abc import Sequence
from html import escape
from typing import Any

from pygent.lara.core.prompt import Prompt
from pygent.lara.runtime.agent.common import _now
from pygent.lara.runtime.agent.skill_catalog import infer_installed_cli_names
from pygent.lara.schema import BashCliPreset

from .models import ReminderSection


def collect_initial_cli(
    presets: tuple[BashCliPreset, ...],
) -> tuple[ReminderSection | None, dict[str, Any]]:
    known = {preset.name: _record(preset) for preset in presets}
    lines = _render_entries(list(presets), tag="available-bash-cli")
    if lines:
        lines = list(_wrap_cli(lines))
    section = ReminderSection("cli.context", 10, tuple(lines), True) if lines else None
    return section, {"known": known}


def detect_cli_changes(
    state: dict[str, Any],
    presets: tuple[BashCliPreset, ...],
    *,
    command: str,
) -> ReminderSection | None:
    known = dict(state.get("known") or {})
    candidates = list(presets)
    preset_names = {preset.name for preset in candidates}
    for name in infer_installed_cli_names(command):
        if name not in preset_names:
            candidates.append(
                BashCliPreset(
                    name=name,
                    command=f"{name} --help",
                    description="Newly installed CLI.",
                )
            )
    new_entries: list[dict[str, Any]] = []
    for preset in candidates:
        previous = known.get(preset.name)
        installed = shutil.which(preset.name) is not None
        if previous and bool(previous.get("installed")) == installed:
            continue
        record = _record(preset)
        known[preset.name] = record
        if installed:
            new_entries.append(record)
    state["known"] = known
    lines = _render_entries(new_entries, tag="new-bash-cli")
    if lines:
        lines = list(_wrap_cli(lines))
    return ReminderSection("cli.context", 10, tuple(lines), True) if lines else None


def _wrap_cli(lines: list[str]) -> tuple[str, ...]:
    return tuple(
        str(Prompt("\n".join(lines)).wrap("cli-context", indent="  ")).splitlines()
    )


def _record(preset: BashCliPreset) -> dict[str, Any]:
    return {
        "name": preset.name,
        "command": preset.command,
        "description": preset.description,
        "installed": shutil.which(preset.name) is not None,
        "detected_at": _now(),
    }


def _render_entries(
    values: Sequence[BashCliPreset | dict[str, Any]], *, tag: str
) -> list[str]:
    if not values:
        return []
    lines = [f"<{tag}>"]
    for value in values:
        name = (
            value.name
            if isinstance(value, BashCliPreset)
            else str(value.get("name") or "")
        )
        if not name:
            continue
        description = (
            value.description
            if isinstance(value, BashCliPreset)
            else str(value.get("description") or "")
        )
        command = (
            value.command
            if isinstance(value, BashCliPreset)
            else str(value.get("command") or "")
        )
        installed = (
            shutil.which(name) is not None
            if isinstance(value, BashCliPreset)
            else bool(value.get("installed"))
        )
        status = "Status: installed." if installed else "Status: not installed."
        if str(command).strip().startswith("uv run "):
            status = "Available via uv run in this workspace."
        lines.extend([f"  <{name}>", f"    {escape(description, quote=False)}"])
        if command:
            lines.append(f"    Command: {escape(str(command), quote=False)}")
        lines.extend([f"    {status}", f"  </{name}>"])
    lines.append(f"</{tag}>")
    return lines
