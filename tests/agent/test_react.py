from __future__ import annotations

import asyncio

import pytest

from pygent.agent import ReActBudgetExceeded, ReActLayer
from pygent.core import AIMessage, Context, Message, Module, ToolMessage, UserMessage
from pygent.tool import ToolCall, ToolResult


def call(call_id: str) -> ToolCall:
    return ToolCall(call_id=call_id, name="lookup", arguments={"id": call_id})


class _ScriptState:
    def __init__(self, scripts: dict[str, tuple[AIMessage, ...]]) -> None:
        self.scripts = scripts
        self.observed: dict[str, list[tuple[Message, Context]]] = {}


class _ToolObservations:
    def __init__(self) -> None:
        self.items: list[tuple[AIMessage, Context]] = []


class ScriptedModel(Module[Message, AIMessage]):
    trusted_live_resource_attributes = ("_state",)

    def __init__(self, scripts: dict[str, tuple[AIMessage, ...]]) -> None:
        super().__init__()
        self._state = _ScriptState(scripts)

    @property
    def observed(self) -> dict[str, list[tuple[Message, Context]]]:
        return self._state.observed

    async def forward(
        self, message: Message, context: Context
    ) -> tuple[AIMessage, Context]:
        run = str(dict(context.metadata)["run"])
        observations = self._state.observed.setdefault(run, [])
        observations.append((message, context))
        return self._state.scripts[run][len(observations) - 1], context


class RecordingTools(Module[AIMessage, ToolMessage]):
    trusted_live_resource_attributes = ("_observations",)

    def __init__(self) -> None:
        super().__init__()
        self._observations = _ToolObservations()

    @property
    def observed(self) -> list[tuple[AIMessage, Context]]:
        return self._observations.items

    async def forward(
        self, message: AIMessage, context: Context
    ) -> tuple[ToolMessage, Context]:
        self._observations.items.append((message, context))
        results = tuple(
            ToolResult(
                call_id=item.call_id,
                name=item.name,
                status="succeeded",
            )
            for item in message.tool_calls
        )
        return ToolMessage(results=results), context


@pytest.mark.asyncio
async def test_no_tool_answer_commits_user_then_final_ai() -> None:
    answer = AIMessage(content="done")
    model = ScriptedModel({"one": (answer,)})
    react = ReActLayer(model=model, tools=RecordingTools())
    user = UserMessage(content="question")
    initial = Context(metadata={"run": "one"})

    actual, context = await react.invoke(user, initial)

    assert actual is answer
    assert context.messages == (user, answer)
    assert model.observed["one"] == [(user, initial)]


@pytest.mark.asyncio
async def test_multiple_tool_turns_preserve_exact_history_order() -> None:
    first = AIMessage(content="first", tool_calls=(call("a"), call("b")))
    second = AIMessage(content="second", tool_calls=(call("c"),))
    final = AIMessage(content="final")
    model = ScriptedModel({"many": (first, second, final)})
    tools = RecordingTools()
    react = ReActLayer(
        model=model,
        tools=tools,
        max_steps=3,
        max_model_calls=3,
        max_tool_calls=3,
    )
    user = UserMessage(content="question")

    actual, context = await react.invoke(
        user, Context(metadata={"run": "many"})
    )

    first_tool = model.observed["many"][1][0]
    second_tool = model.observed["many"][2][0]
    assert isinstance(first_tool, ToolMessage)
    assert isinstance(second_tool, ToolMessage)
    assert actual is final
    assert context.messages == (
        user,
        first,
        first_tool,
        second,
        second_tool,
        final,
    )
    assert [result.call_id for result in first_tool.results] == ["a", "b"]
    assert [result.call_id for result in second_tool.results] == ["c"]
    assert len(tools.observed) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "budget"),
    [
        ({"max_steps": 1, "max_model_calls": 2}, "max_steps"),
        ({"max_steps": 2, "max_model_calls": 1}, "max_model_calls"),
    ],
)
async def test_model_action_is_rejected_before_exceeding_budget(
    kwargs: dict[str, int], budget: str
) -> None:
    model = ScriptedModel(
        {
            "budget": (
                AIMessage(tool_calls=(call("a"),)),
                AIMessage(content="unreachable"),
            )
        }
    )
    tools = RecordingTools()
    react = ReActLayer(model=model, tools=tools, max_tool_calls=1, **kwargs)

    with pytest.raises(ReActBudgetExceeded, match=budget) as caught:
        await react.invoke(
            UserMessage(content="question"),
            Context(metadata={"run": "budget"}),
        )

    assert caught.value.budget == budget
    assert len(model.observed["budget"]) == 1
    assert len(tools.observed) == 1


@pytest.mark.asyncio
async def test_tool_batch_is_rejected_atomically_before_execution() -> None:
    model = ScriptedModel(
        {"tools": (AIMessage(tool_calls=(call("a"), call("b"))),)}
    )
    tools = RecordingTools()
    react = ReActLayer(
        model=model,
        tools=tools,
        max_steps=1,
        max_model_calls=1,
        max_tool_calls=1,
    )

    with pytest.raises(ReActBudgetExceeded) as caught:
        await react.invoke(
            UserMessage(content="question"),
            Context(metadata={"run": "tools"}),
        )

    assert caught.value.budget == "max_tool_calls"
    assert caught.value.requested == 2
    assert tools.observed == []


class BlockingModel(Module[Message, AIMessage]):
    trusted_live_resource_attributes = ("started", "cancelled")

    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def forward(
        self, message: Message, context: Context
    ) -> tuple[AIMessage, Context]:
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        raise AssertionError("unreachable")


@pytest.mark.asyncio
async def test_cancellation_propagates_without_translation() -> None:
    model = BlockingModel()
    react = ReActLayer(model=model, tools=RecordingTools())
    task = asyncio.create_task(
        react.invoke(UserMessage(content="question"), Context())
    )
    await model.started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert model.cancelled.is_set()


@pytest.mark.asyncio
async def test_direct_react_obeys_caller_owned_root_timeout() -> None:
    model = BlockingModel()
    react = ReActLayer(model=model, tools=RecordingTools())

    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.01):
            await react.invoke(UserMessage(content="question"), Context())

    assert model.cancelled.is_set()


@pytest.mark.asyncio
async def test_shared_react_definition_has_no_cross_execution_state() -> None:
    model = ScriptedModel(
        {
            "left": (AIMessage(content="L"),),
            "right": (AIMessage(content="R"),),
        }
    )
    react = ReActLayer(model=model, tools=RecordingTools())

    (left, left_context), (right, right_context) = await asyncio.gather(
        react.invoke(UserMessage(content="left"), Context(metadata={"run": "left"})),
        react.invoke(
            UserMessage(content="right"), Context(metadata={"run": "right"})
        ),
    )

    assert left.content == "L"
    assert right.content == "R"
    assert [item.content for item in left_context.messages] == ["left", "L"]
    assert [item.content for item in right_context.messages] == ["right", "R"]


@pytest.mark.asyncio
async def test_draft_and_reviewer_final_are_both_retained() -> None:
    draft = AIMessage(content="draft")
    reviewed = AIMessage(content="reviewed")
    react = ReActLayer(
        model=ScriptedModel({"review": (draft,)}), tools=RecordingTools()
    )

    class Reviewer(Module[AIMessage, AIMessage]):
        async def forward(
            self, message: AIMessage, context: Context
        ) -> tuple[AIMessage, Context]:
            assert message is draft
            return reviewed, context + reviewed

    class Coordinator(Module[UserMessage, AIMessage]):
        def __init__(self) -> None:
            super().__init__()
            self.react = react
            self.reviewer = Reviewer()

        async def forward(
            self, message: UserMessage, context: Context
        ) -> tuple[AIMessage, Context]:
            answer, context = await self.react(message, context)
            return await self.reviewer(answer, context)

    actual, context = await Coordinator().invoke(
        UserMessage(content="question"), Context(metadata={"run": "review"})
    )

    assert actual is reviewed
    assert [item.content for item in context.messages] == [
        "question",
        "draft",
        "reviewed",
    ]
