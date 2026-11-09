from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

from pygent import freeze_json_object
from pygent.llm import DefaultModelInvoker, ModelExecution


class LaraModelInvoker(DefaultModelInvoker):
    """Retain streamed reasoning as display metadata on the completed message."""

    def execute(self, **kwargs: Any) -> ModelExecution:
        start = super().execute

        async def invoke(emit):
            execution = start(**kwargs)
            reasoning: list[str] = []

            async def relay():
                async with execution.subscribe() as events:
                    async for event in events:
                        data = freeze_json_object(event.data)
                        if event.kind == "model.output.reset":
                            reasoning.clear()
                        elif event.kind == "model.reasoning.delta":
                            text = data.get("text")
                            if isinstance(text, str):
                                reasoning.append(text)
                        await emit(event.kind, data)

            task = asyncio.create_task(relay(), name="lara-model-display-events")
            try:
                response = await execution.result()
                await task
            except BaseException:
                await execution.cancel()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                raise
            if not reasoning:
                return response
            return replace(response, message=replace(
                response.message,
                metadata={**dict(response.message.metadata), "reasoning_content": "".join(reasoning)},
            ))

        return ModelExecution(invoke)
