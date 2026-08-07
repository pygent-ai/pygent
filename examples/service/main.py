"""Run the service example through real invoke, stream, and CAS commits."""

from __future__ import annotations

import asyncio

from pygent import (
    AIMessage,
    LocalRuntime,
    ModelEventKind,
    ModelExecution,
    ModelProviderResponse,
    freeze_json_object,
)

from .app import create_service
from .domain import ChatRequest, InMemoryConversationStore


class DemoModelInvoker:
    """Deterministic provider boundary for an offline runnable example."""

    async def _invoke(self, emit, **kwargs):
        group = kwargs["model_group"].name
        message = kwargs["message"]
        route_id = "demo"
        attempt = 1
        await emit(
            ModelEventKind.STARTED.value,
            freeze_json_object({"model_group": group}),
        )
        prefix = "reviewed" if group == "reviewer" else "draft"
        content = f"{prefix}: {message.content}"
        await emit(
            ModelEventKind.ATTEMPT_STARTED.value,
            freeze_json_object({"route_id": route_id, "attempt": attempt}),
        )
        await emit(
            ModelEventKind.TEXT_DELTA.value,
            freeze_json_object(
                {"route_id": route_id, "attempt": attempt, "text": content}
            ),
        )
        await emit(
            ModelEventKind.USAGE.value,
            freeze_json_object(
                {
                    "route_id": route_id,
                    "attempt": attempt,
                    "mode": "cumulative",
                    "final": True,
                    "available": True,
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                    "cached_input_tokens": None,
                    "reasoning_tokens": None,
                }
            ),
        )
        await emit(
            ModelEventKind.ATTEMPT_SUCCEEDED.value,
            freeze_json_object({"route_id": route_id, "attempt": attempt}),
        )
        await emit(
            ModelEventKind.COMPLETED.value,
            freeze_json_object(
                {
                    "route_id": route_id,
                    "attempt": attempt,
                    "finish_reason": "stop",
                    "provider_request_id": f"demo-{group}",
                }
            ),
        )
        return ModelProviderResponse(
            AIMessage(content=content),
            usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            provider_request_id=f"demo-{group}",
        )

    def execute(self, **kwargs):
        return ModelExecution(lambda emit: self._invoke(emit, **kwargs))


async def run_demo() -> None:
    runtime = LocalRuntime()
    store = InMemoryConversationStore()
    service = create_service(
        runtime,
        store,
        model_invoker=DemoModelInvoker(),
    )
    response = await service.invoke(
        ChatRequest(
            session_id="invoke-session",
            user_id="demo-user",
            text="hello",
            permissions=("weather:read",),
        )
    )
    print(f"invoke revision={response.revision}: {response.text}")

    events = [
        event.kind
        async for event in service.stream(
            ChatRequest(
                session_id="stream-session",
                user_id="demo-user",
                text="stream hello",
                permissions=("weather:read",),
            )
        )
    ]
    snapshot = await store.read("stream-session")
    print(f"stream revision={snapshot.revision}: {events}")
    await runtime.close()


def main() -> None:
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
