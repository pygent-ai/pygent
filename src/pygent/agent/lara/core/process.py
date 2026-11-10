"""Cross-process ownership identity for artifacts shared by several hosts.

Lara hosts (CLI, desktop API) share one project's sessions root, so a case run
created by one process is visible to the others. Each process stamps artifacts
with its PID, process token, and recording time.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

from pygent.agent.lara.core.io import utc_now

# Stable for the lifetime of this process; a restarted host gets a new token.
PROCESS_TOKEN = uuid.uuid4().hex


def current_owner() -> dict[str, Any]:
    return {
        "pid": os.getpid(),
        "process_token": PROCESS_TOKEN,
        "recorded_at": utc_now(),
    }


__all__ = ["PROCESS_TOKEN", "current_owner"]
