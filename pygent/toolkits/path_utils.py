from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Optional


_MSYS_DRIVE_PATH_RE = re.compile(r"^/([a-zA-Z])(?:/(.*))?$")


class ToolPathError(ValueError):
    """Structured path resolution error raised before a tool touches the file system."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str = "InvalidPathError",
        input_path: Optional[str] = None,
        path: Optional[Path] = None,
        workspace_root: Optional[Path] = None,
    ):
        super().__init__(message)
        self.error_type = error_type
        self.details = {
            key: str(value)
            for key, value in {
                "input_path": input_path,
                "path": path,
                "workspace_root": workspace_root,
            }.items()
            if value is not None
        }


@dataclass(frozen=True)
class ToolPathContext:
    """Path resolution settings shared by workspace-scoped tools."""

    workspace_root: Path
    restrict_to_workspace: bool = True

    @classmethod
    def from_workspace_root(
        cls,
        workspace_root: str | Path,
        *,
        restrict_to_workspace: bool = True,
    ) -> "ToolPathContext":
        return cls(
            workspace_root=Path(workspace_root).expanduser().resolve(),
            restrict_to_workspace=restrict_to_workspace,
        )


def normalize_desktop_path(path: str) -> str:
    """Normalize common mistaken desktop paths to the current user's desktop."""
    s = str(path).strip().replace("\\", "/")
    if s.startswith("/Users/Desktop") or s.lower().startswith("c:/users/desktop"):
        if s.lower().startswith("c:/users/desktop"):
            rest = s[17:].lstrip("/")
        else:
            rest = s[14:].lstrip("/")
        return "~/Desktop/" + rest if rest else "~/Desktop"
    return str(path).strip()


def normalize_msys_drive_path(path: str) -> str:
    """Convert Git Bash/MSYS drive paths like /c/Users/me/file on Windows."""
    path_str = str(path).strip()
    if os.name != "nt":
        return path_str

    match = _MSYS_DRIVE_PATH_RE.match(path_str.replace("\\", "/"))
    if not match:
        return path_str

    drive = match.group(1).upper()
    rest = match.group(2) or ""
    return f"{drive}:/{rest}" if rest else f"{drive}:/"


def normalize_tool_path(path: str, base: Optional[str] = None) -> Path:
    """Resolve a user/tool path, including MSYS drive paths on Windows."""
    path_str = normalize_desktop_path(path)
    path_str = normalize_msys_drive_path(os.path.expanduser(path_str))
    resolved_path = Path(path_str)
    if not resolved_path.is_absolute() and base:
        resolved_path = Path(base).expanduser() / resolved_path
    return resolved_path.resolve()


def is_absolute_tool_path(path: str) -> bool:
    """Return whether a path is absolute after applying supported aliases."""
    raw = str(path).strip()
    if not raw:
        return False
    normalized = normalize_msys_drive_path(os.path.expanduser(raw))
    return Path(normalized).is_absolute() or PureWindowsPath(normalized).is_absolute()


def _is_within_workspace(path: Path, workspace_root: Path) -> bool:
    try:
        path.relative_to(workspace_root)
        return True
    except ValueError:
        return False


def resolve_tool_path(
    path: Optional[str],
    context: ToolPathContext,
    *,
    default: Optional[str] = None,
) -> Path:
    """Resolve a tool path against workspace_root and enforce workspace bounds."""
    if path is None or not str(path).strip():
        if default is None:
            raise ToolPathError(
                "path must not be empty",
                input_path=path,
                workspace_root=context.workspace_root,
            )
        input_path = default
    else:
        input_path = str(path).strip()

    resolved = normalize_tool_path(input_path, str(context.workspace_root))
    if context.restrict_to_workspace and not _is_within_workspace(resolved, context.workspace_root):
        raise ToolPathError(
            f"path is outside workspace_root: {resolved}",
            error_type="PathOutsideWorkspaceError",
            input_path=input_path,
            path=resolved,
            workspace_root=context.workspace_root,
        )
    return resolved


def resolve_file_path(path: str, context: ToolPathContext) -> Path:
    """Resolve a file path. Existence and file-kind checks are left to callers."""
    return resolve_tool_path(path, context, default=None)


def resolve_dir_path(
    path: Optional[str],
    context: ToolPathContext,
    *,
    default: str = ".",
) -> Path:
    """Resolve a directory path. Existence and directory-kind checks are left to callers."""
    return resolve_tool_path(path, context, default=default)
