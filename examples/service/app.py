"""Application service boundary around a stateless Module graph."""

from __future__ import annotations

from collections.abc import AsyncIterator
from time import monotonic
from typing import cast
from uuid import uuid4

from pygent import (
    AIMessage,
    Binding,
    BoundModule,
    CapacityPolicy,
    CapacityScope,
    Context,
    ExecutionCapacityPolicy,
    ExecutionEvent,
    ExecutionOptions,
    ModelInvoker,
    Runtime,
    ToolCallLayer,
    UserMessage,
)
from pygent.tool import ExecutorRegistry, ToolTaskManager

from .agents import CoordinatorAgent, build_agent
from .domain import ChatRequest, ChatResponse, ConversationStore


class AgentService:
    def __init__(
        self,
        agent: CoordinatorAgent,
        runtime: Runtime,
        store: ConversationStore,
    ):
        binding = Binding(
            name="agent-service",
            execution_capacity=ExecutionCapacityPolicy(
                # This example owns one in-process Runtime. Multi-worker
                # deployments must inject a shared capacity coordinator.
                scope=CapacityScope.RUNTIME_INSTANCE,
                max_live_executions=128,
                max_runnable_executions=8,
                max_queue_size=64,
                max_waiters=128,
                max_child_depth=8,
                max_children_per_execution=32,
            ),
            model_capacity=CapacityPolicy.passthrough(),
            tool_capacity=CapacityPolicy.limited(
                max_concurrency=32,
                max_queue_size=128,
            ),
        )
        self.agent: BoundModule[UserMessage, AIMessage] = agent.bind(
            runtime, binding=binding
        )
        tool_layer = cast(ToolCallLayer, agent.react.tools)
        self.tool_definitions = tool_layer.definitions
        self.store = store

    async def _prepare(self, request: ChatRequest):
        snapshot = await self.store.read(request.session_id)
        context = Context(
            system_prompt="你是支持工具调用并经过审核的服务 Agent。",
            messages=snapshot.messages,
            tools=self.tool_definitions,
            metadata=(
                ("session_id", request.session_id),
                ("skills", ("answer-quality@1",)),
                ("permissions", request.permissions),
            ),
        )
        message = UserMessage(content=request.text)
        run = ExecutionOptions(
            request_id=uuid4().hex,
            identity=request.user_id,
            context_ref=f"session:{request.session_id}",
            deadline=monotonic() + 60.0,
        )
        return snapshot, message, context, run

    async def invoke(self, request: ChatRequest) -> ChatResponse:
        snapshot, message, context, run = await self._prepare(request)
        output, next_context = await self.agent.invoke(message, context, execution=run)
        revision = await self.store.commit(
            request.session_id,
            snapshot.revision,
            next_context.messages,
        )
        return ChatResponse(
            request.session_id,
            revision,
            output.content,
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[ExecutionEvent]:
        snapshot, message, context, run = await self._prepare(request)
        async with self.agent.stream(message, context, execution=run) as stream:
            async for event in stream:
                yield event
            _, next_context = await stream.final_result()
        await self.store.commit(
            request.session_id,
            snapshot.revision,
            next_context.messages,
        )


def create_service(
    runtime: Runtime,
    store: ConversationStore,
    *,
    model_invoker: ModelInvoker | None = None,
    reviewer_invoker: ModelInvoker | None = None,
    executor_registry: ExecutorRegistry | None = None,
    task_manager: ToolTaskManager | None = None,
) -> AgentService:
    return AgentService(
        build_agent(
            model_invoker=model_invoker,
            reviewer_invoker=reviewer_invoker,
            executor_registry=executor_registry,
            task_manager=task_manager,
        ),
        runtime,
        store,
    )


__all__ = ["AgentService", "create_service"]
