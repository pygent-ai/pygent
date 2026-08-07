"""Bounded ReAct composition built entirely from ordinary Modules."""

from pygent.core import (
    AIMessage,
    Context,
    EffectSafety,
    ExecutionRequirements,
    Message,
    Module,
    RecoverySafety,
    ToolMessage,
    UserMessage,
)


class ReActBudgetExceeded(RuntimeError):
    """Raised before a ReAct action that would exceed its declared budget."""

    def __init__(self, budget: str, *, limit: int, requested: int = 1) -> None:
        self.budget = budget
        self.limit = limit
        self.requested = requested
        super().__init__(
            f"ReAct {budget} budget exhausted "
            f"(limit={limit}, requested={requested})"
        )


class ReActLayer(Module[UserMessage, AIMessage]):
    execution_requirements = ExecutionRequirements(
        requires_finite_deadline=True,
        recovery_safety=RecoverySafety.MODULE_BOUNDARY_RETRY,
        effect_safety=EffectSafety.EFFECT_FREE,
    )

    def __init__(
        self,
        *,
        model: Module[Message, AIMessage],
        tools: Module[AIMessage, ToolMessage],
        max_steps: int = 8,
        max_model_calls: int = 8,
        max_tool_calls: int = 32,
    ):
        super().__init__()
        limits = {
            "max_steps": max_steps,
            "max_model_calls": max_model_calls,
            "max_tool_calls": max_tool_calls,
        }
        for name, value in limits.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.model = model
        self.tools = tools
        self.max_steps = max_steps
        self.max_model_calls = max_model_calls
        self.max_tool_calls = max_tool_calls

    async def forward(
        self, message: UserMessage, context: Context
    ) -> tuple[AIMessage, Context]:
        # All counters and history live in this invocation.  A ReActLayer can
        # therefore be reused concurrently without sharing run state.
        steps = 0
        model_calls = 0
        tool_calls = 0
        current: Message = message
        history = context

        while True:
            _admit("max_steps", used=steps, requested=1, limit=self.max_steps)
            _admit(
                "max_model_calls",
                used=model_calls,
                requested=1,
                limit=self.max_model_calls,
            )

            # One completed invocation of the model Module is one inference
            # step. Provider retries are internal to that Module and do not
            # pass through this accounting boundary.
            answer, model_context = await self.model(current, history)
            steps += 1
            model_calls += 1

            calls_this_turn = len(answer.tool_calls)
            if calls_this_turn == 0:
                return answer, model_context + current + answer

            _admit(
                "max_tool_calls",
                used=tool_calls,
                requested=calls_this_turn,
                limit=self.max_tool_calls,
            )

            # The current input is now consumed.  The AI message remains the
            # current increment passed separately to ToolCallLayer.
            tool_context = model_context + current
            tool_message, tool_context = await self.tools(answer, tool_context)
            tool_calls += calls_this_turn

            # Keep the ToolMessage separate from history until the next model
            # call; this preserves the Module (message, context) convention and
            # commits User/AI/Tool in deterministic order exactly once.
            history = tool_context + answer
            current = tool_message


def _admit(budget: str, *, used: int, requested: int, limit: int) -> None:
    if used + requested > limit:
        raise ReActBudgetExceeded(budget, limit=limit, requested=requested)


__all__ = ["ReActBudgetExceeded", "ReActLayer"]
