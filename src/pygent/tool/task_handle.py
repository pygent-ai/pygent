"""A live observation/control reference to a manager-owned tool task."""

from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .executors import ToolTaskManager
    from .types import ToolResult, ToolTask


class ToolTaskHandle:
    def __init__(self, manager: ToolTaskManager, task_id: str) -> None:
        self.manager = manager
        self.task_id = task_id

    async def snapshot(self) -> ToolTask:
        task = await self.manager.get_task(self.task_id)
        if task is None:
            raise KeyError(self.task_id)
        return task

    async def result(self) -> ToolResult:
        result = await self.manager.get_result(self.task_id, wait=True)
        if result is None:
            raise KeyError(self.task_id)
        return result

    async def cancel(self) -> bool:
        return await self.manager.cancel(self.task_id)

    async def wait(self, timeout: float | None = None) -> ToolResult | None:
        if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
            raise ValueError("timeout must be finite and non-negative")
        result = await self.manager.get_result(self.task_id)
        if result is not None or timeout == 0:
            return result
        try:
            async with asyncio.timeout(timeout):
                return await self.manager.get_result(self.task_id, wait=True)
        except TimeoutError:
            return None
