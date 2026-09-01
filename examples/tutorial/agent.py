"""One Agent graph shared by every step of the progressive tutorial."""

from __future__ import annotations

from pygent import (
    AIMessage,
    Context,
    FallbackPolicy,
    GenerationConfig,
    ModelCallLayer,
    ModelCallPolicy,
    ModelGroupConfig,
    ModelRoute,
    Module,
    ReActLayer,
    RetryPolicy,
    ToolAuthorizationDecision,
    ToolAuthorizationRequest,
    ToolKit,
    ToolSideEffect,
    UserMessage,
    tool,
)
from pygent.llm.spi import ModelInvoker

TUTORIAL_MODEL_GROUP = "tutorial-assistant"
TUTORIAL_PERMISSION = "calculator:use"


@tool(
    tool_id="tutorial.add",
    version="1.0.0",
    name="add",
    side_effect=ToolSideEffect.PURE,
    required_permissions=(TUTORIAL_PERMISSION,),
)
def add_numbers(a: int, b: int) -> dict[str, int]:
    """Add two integers."""

    return {"sum": a + b}


TOOLKIT = ToolKit(add_numbers)


class TutorialAuthorization(
    Module[ToolAuthorizationRequest, ToolAuthorizationDecision]
):
    """Keep business authorization explicit even for a local pure tool."""

    async def forward(
        self,
        request: ToolAuthorizationRequest,
        context: Context,
    ) -> tuple[ToolAuthorizationDecision, Context]:
        allowed = TUTORIAL_PERMISSION in request.permissions
        return (
            ToolAuthorizationDecision(
                call_id=request.call.call_id,
                allowed=allowed,
                reason_code="allowed" if allowed else "missing_permission",
            ),
            context,
        )


class TutorialAgent(Module[UserMessage, AIMessage]):
    """A stateless Agent is an ordinary Module that composes child Modules."""

    def __init__(self, react: ReActLayer) -> None:
        super().__init__()
        self.react = react

    async def forward(
        self,
        message: UserMessage,
        context: Context,
    ) -> tuple[AIMessage, Context]:
        return await self.react(message, context)


def fixed_model_group(model_name: str = "offline-tutorial") -> ModelGroupConfig:
    """Build the fixed route used by direct/offline and direct/live execution."""

    return ModelGroupConfig(
        name=TUTORIAL_MODEL_GROUP,
        routes=(ModelRoute("primary", provider="openai", model=model_name),),
        fallback=FallbackPolicy(("primary",)),
        max_concurrency=8,
        capacity_key="tutorial-model",
    )


def deferred_model_group() -> ModelGroupConfig:
    """Declare a deployment requirement without embedding a concrete model."""

    return ModelGroupConfig.deferred(
        name=TUTORIAL_MODEL_GROUP,
        max_concurrency=8,
        capacity_key="tutorial-model",
    )


def build_agent(
    model_group: ModelGroupConfig,
    *,
    invoker: ModelInvoker | None = None,
) -> TutorialAgent:
    """Assemble the same bounded ReAct graph for direct or managed execution."""

    model = ModelCallLayer(
        model_group=model_group,
        policy=(
            ModelCallPolicy(
                allow_profile_override=True,
                overridable_generation=frozenset(
                    {"temperature", "max_output_tokens"}
                ),
            )
            if model_group.is_deferred
            else ModelCallPolicy()
        ),
        retry_policy=RetryPolicy(attempt_idle_timeout_seconds=30.0),
        generation=GenerationConfig(
            temperature=0.0,
            max_output_tokens=256,
            tool_choice="auto",
        ),
        tools=TOOLKIT.definitions,
        invoker=invoker,
    )
    tools = TOOLKIT.local_layer(
        authorization=TutorialAuthorization(),
        max_concurrency=4,
    )
    return TutorialAgent(
        ReActLayer(
            model=model,
            tools=tools,
            max_steps=3,
            max_model_calls=3,
            max_tool_calls=2,
        )
    )


def build_context() -> Context:
    """Create one immutable request snapshot owned by the application."""

    return TOOLKIT.make_visible_in(
        Context(
            system_prompt=(
                "You are a calculator assistant. Call the add tool once, "
                "then answer with its result."
            ),
            metadata={"permissions": [TUTORIAL_PERMISSION]},
        )
    )


__all__ = [
    "TOOLKIT",
    "TUTORIAL_MODEL_GROUP",
    "TUTORIAL_PERMISSION",
    "TutorialAgent",
    "TutorialAuthorization",
    "add_numbers",
    "build_agent",
    "build_context",
    "deferred_model_group",
    "fixed_model_group",
]
