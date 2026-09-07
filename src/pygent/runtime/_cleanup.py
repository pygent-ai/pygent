"""Owned cleanup: bounded caller waits, with strict shutdown joining every task."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Coroutine
from typing import Any, TypeVar

from ._history_types import HistoryStoreError

_T = TypeVar("_T")


def track_cleanup(
    tasks: set[asyncio.Task[Any]], operation: Coroutine[Any, Any, _T], name: str
) -> asyncio.Task[_T]:
    task = asyncio.create_task(operation, name=name)
    tasks.add(task)
    task.add_done_callback(tasks.discard)
    task.add_done_callback(
        lambda done: done.exception() if not done.cancelled() else None
    )
    return task


async def wait_cleanup(task: asyncio.Task[_T], deadline: float) -> _T:
    while not task.done():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise HistoryStoreError(
                "execution cleanup has not completed within its budget"
            )
        try:
            await asyncio.wait((task,), timeout=remaining)
        except asyncio.CancelledError:
            # The caller may stop waiting, but repeated cancellation cannot
            # detach resource cleanup from its owning Runtime or Worker.
            continue
    return task.result()
