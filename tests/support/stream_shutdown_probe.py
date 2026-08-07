from __future__ import annotations

import asyncio
import time

from pygent import Context, UserMessage
from pygent.core import freeze_json_object
from pygent.llm import (
    DefaultModelInvoker,
    FallbackPolicy,
    GenerationConfig,
    ModelCallError,
    ModelErrorKind,
    ModelGroupConfig,
    ModelProviderCapabilities,
    ModelRoute,
    OpenAICompatibleAdapter,
    RetryPolicy,
)
from pygent.llm import adapter as adapter_module


class NestedCancellationResistantClient:
    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.started = asyncio.Event()
        self.iterator = None

    async def invoke(self, route, payload):
        del route, payload
        raise AssertionError("streaming probe must not invoke")

    def stream(self, route, payload):
        del route, payload

        async def inner():
            self.started.set()
            while not self.release.is_set():
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    continue
            yield freeze_json_object(
                {"choices": [{"delta": {"content": "released"}}]}
            )
            yield freeze_json_object({"done": True})

        async def outer():
            async for item in inner():
                yield item

        self.iterator = outer()
        return self.iterator

    async def aclose(self) -> None:
        if self.iterator is not None:
            await self.iterator.aclose()


async def main() -> None:
    adapter_module._CANCELLATION_CLEANUP_GRACE_SECONDS = 0.01
    client = NestedCancellationResistantClient()
    invoker = DefaultModelInvoker(
        adapters={"openai": OpenAICompatibleAdapter()},
        clients={"primary": client},
        capabilities={"primary": ModelProviderCapabilities(streaming=True)},
    )
    execution = invoker.execute(
        model_group=ModelGroupConfig(
            name="shutdown-probe",
            routes=(ModelRoute("primary", "openai", "probe"),),
            fallback=FallbackPolicy(("primary",)),
        ),
        retry_policy=RetryPolicy(
            max_attempts_per_route=1,
            attempt_timeout_seconds=0.005,
        ),
        generation=GenerationConfig(),
        message=UserMessage(content="probe"),
        context=Context(),
        deadline=time.monotonic() + 1,
    )
    await asyncio.wait_for(client.started.wait(), timeout=0.2)

    async def release_later() -> None:
        await asyncio.sleep(0.05)
        client.release.set()

    asyncio.create_task(release_later(), name="probe-provider-release")
    try:
        await execution.result()
    except ModelCallError as exc:
        assert exc.kind is ModelErrorKind.OUTCOME_UNKNOWN
    else:
        raise AssertionError("probe must reach outcome-unknown cleanup")
    await invoker.aclose()
    pending = {
        task.get_name()
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and task.get_name().startswith("pygent-model-")
    }
    assert not pending, pending


if __name__ == "__main__":
    asyncio.run(main())
