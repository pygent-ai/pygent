from __future__ import annotations

import time

import pytest

from pygent import (
    Context,
    ExecutionOptions,
    FallbackPolicy,
    GenerationConfig,
    LocalRuntime,
    ModelCallError,
    ModelCallLayer,
    ModelErrorKind,
    ModelExecution,
    ModelGroupConfig,
    ModelRoute,
    RetryPolicy,
    UserMessage,
)
from pygent.runtime import SQLiteHistoryStore


class FailingInvoker:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, **kwargs):
        async def operation(emit):
            self.calls += 1
            raise ModelCallError("provider unavailable", kind=ModelErrorKind.UNAVAILABLE)

        return ModelExecution(operation)


def failing_model(invoker: FailingInvoker) -> ModelCallLayer:
    return ModelCallLayer(
        model_group=ModelGroupConfig(
            "durable",
            (ModelRoute("primary", "openai", "test"),),
            FallbackPolicy(("primary",)),
        ),
        retry_policy=RetryPolicy(),
        generation=GenerationConfig(),
        invoker=invoker,
    )


@pytest.mark.asyncio
async def test_terminal_model_failure_and_event_cursor_replay_without_provider_call(
    tmp_path,
):
    invoker = FailingInvoker()
    path = tmp_path / "history.sqlite3"
    async with SQLiteHistoryStore(path) as history:
        runtime = LocalRuntime(history=history)
        handle = await runtime.bind(failing_model(invoker)).start(
            UserMessage(content="hello"),
            Context(),
            execution=ExecutionOptions(
                request_id="first-attempt", deadline=time.monotonic() + 3
            ),
        )
        with pytest.raises(ModelCallError, match="provider unavailable"):
            await handle.result()
        execution_id = handle.execution_id
        await runtime.close()

    async with SQLiteHistoryStore(path) as history:
        runtime = LocalRuntime(history=history)
        recovered = await runtime.recover(
            runtime.bind(failing_model(invoker)),
            execution_id,
            deadline=time.monotonic() + 3,
        )
        with pytest.raises(ModelCallError, match="provider unavailable"):
            await recovered.result()
        events = await history.events_after(execution_id=execution_id)
        await runtime.close()

    assert invoker.calls == 1
    assert [event["sequence"] for event in events] == list(range(len(events)))
