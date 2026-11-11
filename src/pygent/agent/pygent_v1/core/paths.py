from __future__ import annotations

import hashlib
import os
from pathlib import Path


def user_pygent_root() -> Path:
    """Return the user-owned pygent_v1 data root shared by all projects."""
    return (Path.home() / ".pygent_v1").expanduser().resolve()


def project_pygent_root(workspace_root: str | Path, user_root: str | Path | None = None) -> Path:
    """Return the user-owned data directory for a canonical workspace path."""
    workspace = os.path.normcase(str(Path(workspace_root).expanduser().resolve()))
    project_id = hashlib.sha256(workspace.encode("utf-8")).hexdigest()[:24]
    root = (
        Path(user_root).expanduser().resolve()
        if user_root is not None
        else user_pygent_root()
    )
    return root / "projects" / project_id
