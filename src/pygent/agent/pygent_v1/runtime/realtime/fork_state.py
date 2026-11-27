"""In-process per-session fork-reply state for supersede-and-merge."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from pygent.agent.pygent_v1.runtime.attachments import ResolvedAttachment


@dataclass
class SessionForkState:
    """One realtime session's current fork and queued interrupt messages.

    ``attachments`` is the merged batch's attachment set: a superseded
    generation's files must stay visible to the fork that replaces it.
    """

    handle: Any | None = None
    finished: bool = False
    pending: list[str] = field(default_factory=list)
    input_ids: list[str] = field(default_factory=list)
    attachments: list[ResolvedAttachment] = field(default_factory=list)
    generation: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def merge_attachments(self, items: Iterable[ResolvedAttachment]) -> None:
        """Append attachments not already in the batch (by workspace path)."""

        known = {item.relative for item in self.attachments}
        for item in items:
            if item.relative not in known:
                known.add(item.relative)
                self.attachments.append(item)

    def reset(self) -> None:
        """Drop the failed fork generation's bookkeeping.

        ``generation`` stays: it is the monotonic supersede counter. ``lock``
        is the coordination primitive itself.
        """

        self.handle = None
        self.finished = False
        self.pending = []
        self.input_ids = []
        self.attachments = []


__all__ = ["SessionForkState"]
