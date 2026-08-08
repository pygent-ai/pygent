"""User-authored Module graph following the Pygent 0.2 SDK contracts."""

from __future__ import annotations

from pygent import (
    AIMessage,
    Context,
    ModelCallLayer,
    Module,
    ReActLayer,
    UserMessage,
)
from pygent.llm.spi import ModelInvoker
from pygent.tool import ExecutorRegistry, ToolTaskManager

from .models import build_assistant_model, build_reviewer_model
from .tools import build_tool_layer


class ReviewAgent(Module[AIMessage, AIMessage]):
    """Review a draft while preserving the draft committed by ReAct."""

    def __init__(self, *, model: ModelCallLayer):
        super().__init__()
        self.model = model

    async def forward(
        self, message: AIMessage, context: Context
    ) -> tuple[AIMessage, Context]:
        review_context = Context(
            system_prompt="你是回答质量审核 Agent。",
            metadata=context.metadata,
        )
        review_message = UserMessage(content=f"请审核并改写：{message.content}")
        reviewed, _ = await self.model(review_message, review_context)
        return reviewed, context


class CoordinatorAgent(Module[UserMessage, AIMessage]):
    """A user-defined Agent that composes ReAct and another Agent."""

    def __init__(self, *, react: ReActLayer, reviewer: ReviewAgent):
        super().__init__()
        self.react = react
        self.reviewer = reviewer

    async def forward(
        self, message: UserMessage, context: Context
    ) -> tuple[AIMessage, Context]:
        draft, next_context = await self.react(message, context)

        await self.emit(
            kind="coordinator.review.started",
            data={"stage": "review"},
        )
        reviewed, next_context = await self.reviewer(draft, next_context)
        await self.emit(
            kind="coordinator.review.completed",
            data={"stage": "review"},
        )

        # The Coordinator owns the application-visible commit decision.
        return reviewed, next_context + reviewed


def build_agent(
    *,
    model_invoker: ModelInvoker | None = None,
    reviewer_invoker: ModelInvoker | None = None,
    executor_registry: ExecutorRegistry | None = None,
    task_manager: ToolTaskManager | None = None,
) -> CoordinatorAgent:
    tool_layer = build_tool_layer(
        executor_registry=executor_registry,
        task_manager=task_manager,
    )
    react = ReActLayer(
        model=build_assistant_model(
            invoker=model_invoker,
            tools=tool_layer.definitions,
        ),
        tools=tool_layer,
        max_steps=4,
        max_model_calls=4,
        max_tool_calls=16,
    )
    reviewer = ReviewAgent(
        model=build_reviewer_model(invoker=reviewer_invoker or model_invoker)
    )
    return CoordinatorAgent(react=react, reviewer=reviewer)


__all__ = ["CoordinatorAgent", "ReviewAgent", "build_agent"]
