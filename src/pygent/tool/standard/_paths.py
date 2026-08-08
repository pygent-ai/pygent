"""Workspace-scoped path resolution shared by standard tool adapters."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

from pygent.tool.executors import ToolExecutionError

_MSYS_DRIVE_PATH_RE = re.compile(r"^/([a-zA-Z])(?:/(.*))?$")


@dataclass(frozen=True, slots=True)
class ToolPathContext:
    workspace_root: Path
    restrict_to_workspace: bool = True

    @classmethod
    def from_workspace_root(
        cls,
        workspace_root: str | Path,
        *,
        restrict_to_workspace: bool = True,
    ) -> ToolPathContext:
        return cls(
            workspace_root=Path(workspace_root).expanduser().resolve(),
            restrict_to_workspace=restrict_to_workspace,
        )


def normalize_desktop_path(path: str) -> str:
    """Normalize common model-generated desktop aliases."""

    value = str(path).strip().replace("\\", "/")
    if value.startswith("/Users/Desktop") or value.lower().startswith(
        "c:/users/desktop"
    ):
        prefix_length = 17 if value.lower().startswith("c:/users/desktop") else 14
        rest = value[prefix_length:].lstrip("/")
        return "~/Desktop/" + rest if rest else "~/Desktop"
    return str(path).strip()


def normalize_msys_drive_path(path: str) -> str:
    """Convert Git Bash/MSYS paths such as ``/c/Users/me`` on Windows."""

    value = str(path).strip()
    if os.name != "nt":
        return value
    match = _MSYS_DRIVE_PATH_RE.match(value.replace("\\", "/"))
    if match is None:
        return value
    drive = match.group(1).upper()
    rest = match.group(2) or ""
    return f"{drive}:/{rest}" if rest else f"{drive}:/"


def normalize_tool_path(path: str, base: str | None = None) -> Path:
    value = normalize_desktop_path(path)
    value = normalize_msys_drive_path(os.path.expanduser(value))
    resolved = Path(value)
    if not resolved.is_absolute() and base:
        resolved = Path(base).expanduser() / resolved
    return resolved.resolve()


def is_absolute_tool_path(path: str) -> bool:
    value = str(path).strip()
    if not value:
        return False
    normalized = normalize_msys_drive_path(os.path.expanduser(value))
    return Path(normalized).is_absolute() or PureWindowsPath(normalized).is_absolute()


def _is_within_workspace(path: Path, workspace_root: Path) -> bool:
    try:
        path.relative_to(workspace_root)
    except ValueError:
        return False
    return True


def resolve_tool_path(
    path: str | None,
    context: ToolPathContext,
    *,
    default: str | None = None,
) -> Path:
    if path is None or not str(path).strip():
        if default is None:
            raise ToolExecutionError(
                "path must not be empty",
                kind="filesystem_error",
                code="invalid_path",
                side_effect_committed=False,
            )
        input_path = default
    else:
        input_path = str(path).strip()

    resolved = normalize_tool_path(input_path, str(context.workspace_root))
    if context.restrict_to_workspace and not _is_within_workspace(
        resolved, context.workspace_root
    ):
        raise ToolExecutionError(
            f"path is outside workspace_root: {resolved}",
            kind="filesystem_error",
            code="path_outside_workspace",
            side_effect_committed=False,
        )
    return resolved


def resolve_file_path(path: str, context: ToolPathContext) -> Path:
    return resolve_tool_path(path, context)


def resolve_dir_path(
    path: str | None,
    context: ToolPathContext,
    *,
    default: str = ".",
) -> Path:
    return resolve_tool_path(path, context, default=default)


__all__ = [
    "ToolPathContext",
    "is_absolute_tool_path",
    "normalize_desktop_path",
    "normalize_msys_drive_path",
    "normalize_tool_path",
    "resolve_dir_path",
    "resolve_file_path",
    "resolve_tool_path",
]
