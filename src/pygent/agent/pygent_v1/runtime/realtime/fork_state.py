"""In-process per-session fork-reply state for supersede-and-merge."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionForkState:
    """One realtime session's current fork and queued interrupt messages."""

    handle: Any | None = None
    finished: bool = False
    pending: list[str] = field(default_factory=list)
    generation: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


__all__ = ["SessionForkState"]
