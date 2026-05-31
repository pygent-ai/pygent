from __future__ import annotations

import os
import re
from pathlib import Path, PureWindowsPath
from typing import Optional


_MSYS_DRIVE_PATH_RE = re.compile(r"^/([a-zA-Z])(?:/(.*))?$")


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
