from __future__ import annotations

import hashlib
import os
from pathlib import Path



def user_lara_root() -> Path:
    """Return the user-owned Lara data root shared by all projects."""
    return (Path.home() / ".lara").expanduser().resolve()


def project_lara_root(workspace_root: str | Path, user_root: str | Path | None = None) -> Path:
    """Return the user-owned data directory for a canonical workspace path."""
    workspace = os.path.normcase(str(Path(workspace_root).expanduser().resolve()))
    project_id = hashlib.sha256(workspace.encode("utf-8")).hexdigest()[:24]
    root = (
        Path(user_root).expanduser().resolve()
        if user_root is not None
        else user_lara_root()
    )
    return root / "projects" / project_id
