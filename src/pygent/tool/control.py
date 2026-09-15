"""Trusted deployment adapter for observing and cancelling existing tasks."""

from .executors import ToolExecutionContext, ToolExecutor
from .types import ToolCall, ToolSpec


class ToolTaskControlExecutor:
    """Run task control under Execution capacity, outside the work it controls.

    This is a deployment assertion, never a model-provided parameter. Control
    still passes ordinary visibility, authorization, deadlines and result checks.
    """

    def __init__(self, executor: ToolExecutor) -> None:
        self.executor = executor

    async def execute(
        self, spec: ToolSpec, call: ToolCall, context: ToolExecutionContext
    ) -> object:
        return await self.executor.execute(spec, call, context)
