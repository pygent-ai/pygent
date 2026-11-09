"""Unit tests for the LARA agent scaffold (src/lara/agents).

Covers the happy paths and the architectural invariants from the 2026-11-02
review: prefix continuation, per-job limits propagation, cancellation
isolation, frame-complete slicing, fork snapshots, and the single-source
tool profile resolution.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import pytest

from pygent import (
    AIMessage,
    Context,
    IdempotencyPolicy,
    Message,
    Module,
    ReActBudgetExceeded,
    ToolCall,
    ToolDefinition,
    ToolKit,
    ToolMessage,
    ToolResult,
    ToolSideEffect,
    UserMessage,
    tool,
)
from pygent.core import JsonValue
from pygent.lara.agents import (
    AgentLimits,
    AgentRunner,
    ChildAgent,
    ChildSpec,
    ContextWindow,
    ForkAgent,
    ForkSpec,
    MainAgent,
    MessageSlice,
    ReminderTransform,
    SystemPromptTransform,
    ToolkitToolSurface,
    ToolProfile,
    ToolRegistry,
    align_slice_bounds,
    messages_from_history,
    system_reminder,
)
from pygent.runtime.codec import message_to_dict


@dataclass(frozen=True)
class _WindowSource:
    """Test-local ``ContextSource`` over an explicit context window."""

    window: ContextWindow

    @classmethod
    def of(
        cls, system_prompt: str, messages: tuple[Message, ...] = ()
    ) -> _WindowSource:
        return cls(ContextWindow(system_prompt=system_prompt, messages=messages))

    def resolve(self) -> ContextWindow:
        return self.window

    def describe(self) -> str:
        return f"static-messages:{len(self.window.messages)}"


class _ScriptState:
    """Opaque adapter holding the scripted model's mutable test state."""

    __slots__ = ("answers", "received_definitions", "seen")

    def __init__(self, answers: Sequence[AIMessage]) -> None:
        self.answers = list(answers)
        self.seen: list[tuple[Message, Context]] = []
        self.received_definitions: tuple[object, ...] = ()


class _ScriptedModel(Module[Message, AIMessage]):
    """Model module returning scripted answers; the last one repeats."""

    trusted_live_resource_attributes = ("state",)

    def __init__(self, script: Sequence[AIMessage]) -> None:
        super().__init__()
        if not script:
            raise ValueError("script must contain at least one answer")
        self.state = _ScriptState(script)

    async def forward(
        self, message: Message, context: Context
    ) -> tuple[AIMessage, Context]:
        self.state.seen.append((message, context))
        if len(self.state.answers) > 1:
            return self.state.answers.pop(0), context
        return self.state.answers[0], context


class _ExplodingModel(Module[Message, AIMessage]):
    async def forward(
        self, message: Message, context: Context
    ) -> tuple[AIMessage, Context]:
        raise RuntimeError("model unavailable")


class _BlockingModel(Module[Message, AIMessage]):
    async def forward(
        self, message: Message, context: Context
    ) -> tuple[AIMessage, Context]:
        await asyncio.Event().wait()  # never set; only cancellation ends this
        raise AssertionError("unreachable")  # pragma: no cover


class _GatedModel(Module[Message, AIMessage]):
    """Model that waits on a gate the test controls, then finishes."""

    trusted_live_resource_attributes = ("gate",)

    def __init__(self, gate: asyncio.Event) -> None:
        super().__init__()
        self.gate = gate

    async def forward(
        self, message: Message, context: Context
    ) -> tuple[AIMessage, Context]:
        await self.gate.wait()
        return AIMessage(content="finished"), context


@tool(
    tool_id="lara.test.echo",
    version="1.0.0",
    side_effect=ToolSideEffect.PURE,
    idempotency=IdempotencyPolicy.INHERENT,
    name="echo",
    description="Echo the given text back.",
    timeout=30,
    resource_key="lara-agents-test",
    required_permissions=(),
)
async def echo(text: str) -> str:
    return f"echo:{text}"


def _echo_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("echo", ToolkitToolSurface(ToolKit(echo)))
    return registry


def _runner(
    script: Sequence[AIMessage],
    *,
    limits: AgentLimits | None = None,
    with_registry: bool = True,
) -> tuple[AgentRunner, _ScriptedModel]:
    resolved_model = _ScriptedModel(script)
    registry = _echo_registry() if with_registry else None
    runner = AgentRunner(
        model_factory=lambda _definitions: resolved_model,
        registry=registry,
        default_tools=ToolProfile("echo") if with_registry else None,
        limits=limits,
    )
    return runner, resolved_model


def _tool_call_answer(call_id: str = "t1", text: str = "hi") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=(ToolCall(call_id=call_id, name="echo", arguments={"text": text}),),
    )


def _tool_result_message(call_id: str = "t1") -> ToolMessage:
    return ToolMessage(
        content="",
        results=(
            ToolResult(
                call_id=call_id,
                name="echo",
                status="succeeded",
                output="echo:hi",
            ),
        ),
    )


@pytest.mark.asyncio
async def test_main_agent_turn_runs_react_and_returns_answer() -> None:
    runner, model = _runner([AIMessage(content="done")])
    main = MainAgent(runner, system_prompt="You are the main agent.")

    result = await main.run_turn("hello")

    assert result.text == "done"
    assert result.label == "lara-main"
    message, context = model.state.seen[0]
    assert isinstance(message, UserMessage)
    assert message.content == "hello"
    assert context.system_prompt == "You are the main agent."
    assert context.messages == ()


@pytest.mark.asyncio
async def test_main_agent_turn_continues_fork_prefix() -> None:
    runner, model = _runner([AIMessage(content="continued")])
    main = MainAgent(runner, system_prompt="You are the main agent.")
    prefix = ContextWindow(
        system_prompt="fork prompt",
        messages=(UserMessage(content="earlier work"),),
    )

    result = await main.run_turn("hello", prefix=prefix)

    assert result.text == "continued"
    message, context = model.state.seen[0]
    assert context.system_prompt == "fork prompt"
    assert context.messages == prefix.messages
    assert isinstance(message, UserMessage)
    assert message.content == "hello"


@pytest.mark.asyncio
async def test_runner_requires_exactly_one_context_origin() -> None:
    runner, model = _runner([AIMessage(content="never")])

    with pytest.raises(ValueError, match="exactly one"):
        await runner.run(message="hello")
    with pytest.raises(ValueError, match="exactly one"):
        await runner.run(
            message="hello",
            system_prompt="fresh",
            prefix=ContextWindow(system_prompt="forked"),
        )

    assert model.state.seen == []


def test_model_factory_receives_resolved_surface_definitions() -> None:
    registry = _echo_registry()
    received: list[tuple[ToolDefinition, ...]] = []

    def factory(
        definitions: tuple[ToolDefinition, ...],
    ) -> Module[Message, AIMessage]:
        received.append(definitions)
        return _ScriptedModel([AIMessage(content="done")])

    runner = AgentRunner(
        model_factory=factory,
        registry=registry,
        default_tools=ToolProfile("echo"),
    )
    surface = registry.resolve(ToolProfile("echo"))

    asyncio.run(runner.run(message="go", system_prompt="p"))

    assert len(received) == 1
    assert received[0] == surface.definitions
    assert any(definition.name == "echo" for definition in received[0])


@pytest.mark.asyncio
async def test_fork_agent_sees_prefix_and_directive_with_uniform_tools() -> None:
    prefix_messages: tuple[Message, ...] = (
        UserMessage(content="fix the parser bug"),
        AIMessage(content="reading the parser"),
    )
    runner, model = _runner([_tool_call_answer(), AIMessage(content="extracted")])
    fork = ForkAgent(
        runner,
        ForkSpec(
            source=_WindowSource.of("foreground prompt", prefix_messages)
        ),
        label="lara-fork.extractor",
    )

    result = await fork.run(system_reminder("extract memories"))

    first_message, first_context = model.state.seen[0]
    assert isinstance(first_message, UserMessage)
    assert first_message.content == system_reminder("extract memories")
    assert first_context.system_prompt == "foreground prompt"
    # the steering message is passed as the current message; the model-visible
    # history is exactly the fork prefix (block 1)
    assert first_context.messages == prefix_messages
    # block 3 really ran a tool through the uniform tool surface
    last_message = model.state.seen[-1][0]
    assert isinstance(last_message, ToolMessage)
    assert last_message.results[0].status == "succeeded"
    assert last_message.results[0].output == "echo:hi"
    assert result.text == "extracted"
    assert result.label == "lara-fork.extractor"
    assert result.context.metadata["lara.agent.label"] == "lara-fork.extractor"


def test_fork_plan_applies_slice_and_transforms_without_model_calls() -> None:
    runner, model = _runner([AIMessage(content="unused")])
    source = _WindowSource.of(
        "foreground prompt",
        (
            UserMessage(content="one"),
            UserMessage(content="two"),
            UserMessage(content="three"),
        ),
    )
    fork = ForkAgent(
        runner,
        ForkSpec(
            source=source,
            slice=MessageSlice(start=1),
            transforms=(
                ReminderTransform(text="mid-stream note", position=0),
                SystemPromptTransform(system_prompt="overridden prompt"),
            ),
        ),
    )

    window = fork.plan()

    assert window.system_prompt == "overridden prompt"
    assert [entry.content for entry in window.messages] == [
        system_reminder("mid-stream note"),
        "two",
        "three",
    ]
    assert model.state.seen == []


def test_slice_completes_interaction_frames() -> None:
    messages: tuple[Message, ...] = (
        UserMessage(content="task"),
        AIMessage(
            content="",
            tool_calls=(ToolCall(call_id="t1", name="echo", arguments={"text": "hi"}),),
        ),
        _tool_result_message(),
        AIMessage(content="final"),
    )

    # a cut that starts on a tool result walks back to its assistant call
    start, end = align_slice_bounds(messages, 2, None)
    assert (start, end) == (1, 4)

    # a cut between an assistant tool call and its results extends forward
    start, end = align_slice_bounds(messages, 0, 2)
    assert (start, end) == (0, 3)

    # clean boundaries pass through unchanged
    start, end = align_slice_bounds(messages, 1, 4)
    assert (start, end) == (1, 4)

    fork = ForkAgent(
        _runner([AIMessage(content="unused")])[0],
        ForkSpec(
            source=_WindowSource.of("p", messages),
            slice=MessageSlice(start=2),
        ),
    )
    window = fork.plan()
    assert window.messages == messages[1:]


def test_fork_snapshot_records_freeze_point() -> None:
    runner, _ = _runner([AIMessage(content="unused")])
    source_messages: tuple[Message, ...] = (
        UserMessage(content="task"),
        AIMessage(content="working"),
    )
    fork = ForkAgent(
        runner,
        ForkSpec(
            source=_WindowSource.of("foreground prompt", source_messages)
        ),
    )

    _window, snapshot = fork.materialize()

    assert snapshot.source == "static-messages:2"
    assert (snapshot.start_index, snapshot.end_index) == (0, 2)
    assert snapshot.message_count == 2
    assert len(snapshot.source_hash) == 64
    assert snapshot.source_hash == snapshot.effective_context_hash
    assert snapshot.transforms == ()
    # the hashes track content: a different history hashes differently
    other = ForkAgent(
        runner,
        ForkSpec(
            source=_WindowSource.of(
                "foreground prompt", (*source_messages, AIMessage(content="more"))
            )
        ),
    )
    assert other.materialize()[1].source_hash != snapshot.source_hash
    # a transform changes the effective hash while the source hash stays put
    transformed = ForkAgent(
        runner,
        ForkSpec(
            source=_WindowSource.of("foreground prompt", source_messages),
            transforms=(ReminderTransform(text="note"),),
        ),
    )
    transformed_window, transformed_snapshot = transformed.materialize()
    assert (
        transformed_snapshot.source_hash != transformed_snapshot.effective_context_hash
    )
    assert transformed_snapshot.transforms == (
        {
            "kind": "reminder",
            "position": None,
            "text_hash": hashlib.sha256(
                system_reminder("note").encode("utf-8")
            ).hexdigest(),
        },
    )
    assert "system-reminder" in transformed_window.messages[-1].content


@pytest.mark.asyncio
async def test_fork_run_attaches_snapshot_to_result_metadata() -> None:
    runner, _ = _runner([AIMessage(content="done")])
    fork = ForkAgent(
        runner,
        ForkSpec(
            source=_WindowSource.of(
                "foreground prompt", (UserMessage(content="task"),)
            )
        ),
        label="lara-fork.snap",
    )

    result = await fork.run(system_reminder("extract"))

    snapshot = cast(
        "dict[str, JsonValue]", result.context.metadata["lara.fork.snapshot"]
    )
    assert snapshot["source"] == "static-messages:1"
    assert snapshot["message_count"] == 1
    assert snapshot["source_hash"] == snapshot["effective_context_hash"]


@pytest.mark.asyncio
async def test_fork_chains_on_a_previous_run_product() -> None:
    first_runner, _ = _runner([AIMessage(content="first answer")])
    first = await first_runner.run(
        message="start",
        system_prompt="chain prompt",
        label="lara-fork.first",
    )
    second_runner, second_model = _runner([AIMessage(content="second answer")])
    chained = ForkAgent(
        second_runner,
        ForkSpec(source=_WindowSource(first.window())),
        label="lara-fork.second",
    )

    result = await chained.run("continue from here")

    _, context = second_model.state.seen[0]
    assert context.system_prompt == "chain prompt"
    assert context.messages == first.window().messages
    assert result.text == "second answer"


@pytest.mark.asyncio
async def test_child_agent_runs_independently_with_fresh_context() -> None:
    runner, model = _runner([AIMessage(content="watching")])
    child = ChildAgent(
        runner,
        ChildSpec(name="watcher", system_prompt="independent prompt"),
    )

    result = await child.run("watch the workspace")

    _, context = model.state.seen[0]
    assert context.system_prompt == "independent prompt"
    assert context.messages == ()
    assert result.text == "watching"
    assert result.label == "lara-child.watcher"


@pytest.mark.asyncio
async def test_main_agent_collects_background_outcomes() -> None:
    runner, _ = _runner([AIMessage(content="done")])
    main = MainAgent(runner, system_prompt="You are the main agent.")
    main.spawn_fork(
        ForkSpec(
            source=_WindowSource.of(
                "foreground prompt", (UserMessage(content="work"),)
            )
        ),
        directive=system_reminder("extract"),
    )
    main.spawn_child(
        ChildSpec(name="reviewer", system_prompt="review prompt"),
        directive="review",
    )

    outcomes = await main.collect()

    assert len(outcomes) == 2
    texts: set[str] = set()
    for outcome in outcomes:
        assert outcome.error is None
        assert outcome.result is not None
        texts.add(outcome.result.text)
    assert texts == {"done"}
    assert all(outcome.job_id for outcome in outcomes)
    # the registry drains after collect
    assert await main.collect() == ()


@pytest.mark.asyncio
async def test_spawn_fork_limits_are_enforced() -> None:
    # control: the same script completes under runner defaults
    completing_runner, _ = _runner([_tool_call_answer("t1"), AIMessage(content="done")])
    completing_main = MainAgent(completing_runner, system_prompt="p")
    completing_main.spawn_fork(
        ForkSpec(source=_WindowSource.of("p")),
        directive="go",
    )
    ok_outcomes = await completing_main.collect()
    ok_result = ok_outcomes[0].result
    assert ok_outcomes[0].error is None
    assert ok_result is not None
    assert ok_result.text == "done"

    # the per-job override reaches the engine: max_steps=2 fails a sticky loop
    looping_runner, _ = _runner([_tool_call_answer("t1"), _tool_call_answer("t2")])
    looping_main = MainAgent(looping_runner, system_prompt="p")
    looping_main.spawn_fork(
        ForkSpec(source=_WindowSource.of("p")),
        directive="loop",
        limits=AgentLimits(max_steps=2, max_model_calls=8, max_tool_calls=8),
    )
    outcomes = await looping_main.collect()

    assert len(outcomes) == 1
    assert outcomes[0].result is None
    assert isinstance(outcomes[0].error, ReActBudgetExceeded)
    assert "max_steps" in str(outcomes[0].error)


@pytest.mark.asyncio
async def test_collect_survives_cancellation_and_frees_the_label() -> None:
    runner = AgentRunner(model_factory=lambda _definitions: _BlockingModel())
    main = MainAgent(runner, system_prompt="p")
    handle = main.spawn_child(
        ChildSpec(name="slow", system_prompt="s"),
        directive="go",
        label="lara-child.slow",
    )

    await asyncio.sleep(0.05)
    assert handle.cancel() is True

    outcomes = await main.collect()

    assert len(outcomes) == 1
    assert outcomes[0].cancelled is True
    assert outcomes[0].result is None
    assert outcomes[0].error is None
    # the label is reusable after cancellation: no ghost registry entry
    again = main.spawn_child(
        ChildSpec(name="slow", system_prompt="s"),
        directive="go",
        label="lara-child.slow",
    )
    again.cancel()
    second_round = await main.collect()
    assert second_round[0].cancelled is True


@pytest.mark.asyncio
async def test_background_failure_is_collected_as_outcome() -> None:
    runner = AgentRunner(
        model_factory=lambda _definitions: _ExplodingModel(),
        registry=_echo_registry(),
        default_tools=ToolProfile("echo"),
    )
    main = MainAgent(runner, system_prompt="p")
    main.spawn_child(
        ChildSpec(name="doomed", system_prompt="doomed prompt"),
        directive="boom",
    )

    failures = await main.collect()

    assert len(failures) == 1
    assert failures[0].result is None
    assert failures[0].cancelled is False
    assert isinstance(failures[0].error, RuntimeError)


@pytest.mark.asyncio
async def test_spawn_rejects_duplicate_in_flight_labels() -> None:
    runner, _ = _runner([AIMessage(content="done")])
    main = MainAgent(runner, system_prompt="p")
    spec = ForkSpec(source=_WindowSource.of("p"))

    main.spawn_fork(spec, directive="go", label="lara-fork.custom")

    with pytest.raises(ValueError, match="already in flight"):
        main.spawn_fork(spec, directive="go", label="lara-fork.custom")

    outcomes = await main.collect()
    assert len(outcomes) == 1 and outcomes[0].error is None


@pytest.mark.asyncio
async def test_unknown_tool_profile_is_rejected_before_any_model_call() -> None:
    runner, model = _runner([AIMessage(content="never")])

    with pytest.raises(ValueError, match="not registered"):
        await runner.run(message="go", system_prompt="p", tools=ToolProfile("missing"))

    assert model.state.seen == []


@pytest.mark.asyncio
async def test_runner_without_tool_surface_rejects_stray_tool_calls() -> None:
    runner, _ = _runner([_tool_call_answer()], with_registry=False)

    with pytest.raises(RuntimeError, match="no tool surface"):
        await runner.run(message="call something", system_prompt="p")


@pytest.mark.asyncio
async def test_runner_enforces_step_budget() -> None:
    runner, _ = _runner(
        [_tool_call_answer("t1"), _tool_call_answer("t2")],
        limits=AgentLimits(max_steps=2, max_model_calls=8, max_tool_calls=8),
    )

    with pytest.raises(ReActBudgetExceeded):
        await runner.run(message="loop forever", system_prompt="p")


def test_messages_from_history_converts_session_entries() -> None:
    history = [
        message_to_dict(UserMessage(content="hello")),
        message_to_dict(AIMessage(content="hi there")),
        {"role": "system", "content": "dropped"},
    ]

    messages = messages_from_history(history)

    assert [entry.content for entry in messages] == ["hello", "hi there"]


def test_context_window_requires_prompt_and_reminder_requires_text() -> None:
    with pytest.raises(ValueError):
        ContextWindow(system_prompt="  ")
    with pytest.raises(ValueError):
        system_reminder("   ")


def test_fork_snapshot_metadata_does_not_leak_transform_content() -> None:
    runner, _ = _runner([AIMessage(content="unused")])
    secret = "internal-token-value-XYZ"
    fork = ForkAgent(
        runner,
        ForkSpec(
            source=_WindowSource.of("p"),
            transforms=(
                ReminderTransform(text=secret),
                SystemPromptTransform(system_prompt=f"prompt holding {secret}"),
            ),
        ),
    )

    window, snapshot = fork.materialize()
    metadata_json = json.dumps(snapshot.to_metadata(), ensure_ascii=False)

    # hashes verify the content, but the content itself never enters
    # snapshot metadata that durable stores and exports would copy
    assert secret not in metadata_json
    assert system_reminder(secret) not in metadata_json
    reminder_descriptor = snapshot.transforms[0]
    assert reminder_descriptor["kind"] == "reminder"
    assert len(reminder_descriptor["text_hash"]) == 64  # type: ignore[literal-required]
    prompt_descriptor = snapshot.transforms[1]
    assert prompt_descriptor["kind"] == "system_prompt"  # type: ignore[literal-required]
    assert len(prompt_descriptor["system_prompt_hash"]) == 64  # type: ignore[literal-required]
    # the hashes are still content-sensitive: verification remains possible
    same_content = ForkAgent(
        runner,
        ForkSpec(
            source=_WindowSource.of("p"),
            transforms=(
                ReminderTransform(text=secret),
                SystemPromptTransform(system_prompt=f"prompt holding {secret}"),
            ),
        ),
    )
    assert same_content.materialize()[1].transforms == snapshot.transforms
    different_content = ForkAgent(
        runner,
        ForkSpec(
            source=_WindowSource.of("p"),
            transforms=(ReminderTransform(text="other"),),
        ),
    )
    assert (
        different_content.materialize()[1].transforms[0]["text_hash"]
        != reminder_descriptor["text_hash"]
    )
    # the effective window itself still carries the real content for the run
    assert secret in window.messages[-1].content
    assert window.system_prompt == f"prompt holding {secret}"


def test_spawn_outside_event_loop_does_not_leak_label() -> None:
    runner, _ = _runner([AIMessage(content="done")])
    main = MainAgent(runner, system_prompt="p")
    spec = ForkSpec(source=_WindowSource.of("p"))

    with pytest.raises(RuntimeError, match="no running event loop"):
        main.spawn_fork(spec, directive="go", label="lara-fork.leak")

    # the reserved label was rolled back, so it is usable inside a loop
    async def verify() -> str:
        handle = main.spawn_fork(spec, directive="go", label="lara-fork.leak")
        outcomes = await main.collect()
        assert outcomes[0].result is not None
        return handle.job_id

    job_id = asyncio.run(verify())
    assert len(job_id) == 32


def test_slice_on_out_of_range_start_returns_empty() -> None:
    messages: tuple[Message, ...] = (
        UserMessage(content="one"),
        UserMessage(content="two"),
    )

    # Python slice semantics: an out-of-range start is an empty slice, not an
    # IndexError.
    assert MessageSlice(start=999).apply(messages) == ()
    assert MessageSlice(start=999, end=1000).apply(messages) == ()
    assert align_slice_bounds(messages, 999, None) == (2, 2)


def test_slice_extends_through_consecutive_tool_results() -> None:
    assistant_call = AIMessage(
        content="",
        tool_calls=(ToolCall(call_id="t1", name="echo", arguments={"text": "hi"}),),
    )
    messages: tuple[Message, ...] = (
        UserMessage(content="task"),
        assistant_call,
        _tool_result_message("t1"),
        _tool_result_message("t1"),
        AIMessage(content="final"),
    )

    # a cut after the first of two consecutive results still completes the
    # frame: both results are pulled into the slice
    start, end = align_slice_bounds(messages, 1, 3)
    assert (start, end) == (1, 4)


def test_reminder_transform_snaps_to_frame_boundaries() -> None:
    messages: tuple[Message, ...] = (
        UserMessage(content="task"),
        AIMessage(
            content="",
            tool_calls=(ToolCall(call_id="t1", name="echo", arguments={"text": "hi"}),),
        ),
        _tool_result_message("t1"),
    )
    fork = ForkAgent(
        _runner([AIMessage(content="unused")])[0],
        ForkSpec(
            source=_WindowSource.of("p", messages),
            transforms=(ReminderTransform(text="mid-frame", position=2),),
        ),
    )

    window = fork.plan()

    # position 2 would split the tool call from its result; the reminder is
    # snapped back to the boundary in front of the assistant call
    assert [entry.content for entry in window.messages] == [
        "task",
        system_reminder("mid-frame"),
        "",
        "",
    ]
    tool_frame = [
        entry
        for entry in window.messages
        if isinstance(entry, AIMessage) and entry.tool_calls
    ]
    assert len(tool_frame) == 1
    assistant_index = window.messages.index(tool_frame[0])
    assert isinstance(window.messages[assistant_index + 1], ToolMessage)


@pytest.mark.asyncio
async def test_collect_cancellation_stops_waiting_without_killing_jobs() -> None:
    gate = asyncio.Event()
    model = _GatedModel(gate)
    runner = AgentRunner(
        model_factory=lambda _definitions: model,
        registry=_echo_registry(),
        default_tools=ToolProfile("echo"),
    )
    main = MainAgent(runner, system_prompt="p")
    handle = main.spawn_child(ChildSpec(name="work", system_prompt="s"), directive="go")

    collecting = asyncio.create_task(main.collect())
    await asyncio.sleep(0.05)
    collecting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await collecting

    # the shielded job kept running: not cancelled, still in the registry
    assert handle.cancelled is False
    assert handle.done is False

    gate.set()
    outcomes = await main.collect()

    assert len(outcomes) == 1
    assert outcomes[0].cancelled is False
    assert outcomes[0].error is None
    assert outcomes[0].result is not None
    assert outcomes[0].result.text == "finished"


@pytest.mark.asyncio
async def test_collect_cancellation_keeps_label_reserved_and_job_registered() -> None:
    gate = asyncio.Event()
    runner = AgentRunner(
        model_factory=lambda _definitions: _GatedModel(gate),
        registry=_echo_registry(),
        default_tools=ToolProfile("echo"),
    )
    main = MainAgent(runner, system_prompt="p")
    handle = main.spawn_fork(
        ForkSpec(source=_WindowSource.of("p")),
        directive="go",
        label="lara-fork.same",
    )

    collecting = asyncio.create_task(main.collect())
    await asyncio.sleep(0.05)
    collecting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await collecting

    # while the job still runs, its label cannot be re-reserved and the old
    # job is not lost from the registry
    with pytest.raises(ValueError, match="already in flight"):
        main.spawn_fork(
            ForkSpec(source=_WindowSource.of("p")),
            directive="again",
            label="lara-fork.same",
        )
    assert handle.job_id in main._background

    gate.set()
    outcomes = await main.collect()
    assert outcomes[0].job_id == handle.job_id
    assert outcomes[0].result is not None

    # only after the outcome was delivered is the label free again
    released = main.spawn_fork(
        ForkSpec(source=_WindowSource.of("p")),
        directive="again",
        label="lara-fork.same",
    )
    assert released.job_id != handle.job_id
    await main.collect()


@pytest.mark.asyncio
async def test_concurrent_collect_calls_are_serialized() -> None:
    runner, _ = _runner([AIMessage(content="done")])
    main = MainAgent(runner, system_prompt="p")
    expected_labels: set[str] = set()
    for index in range(2):
        main.spawn_fork(
            ForkSpec(source=_WindowSource.of("p")),
            directive=f"go-{index}",
            label=f"lara-fork.{index}",
        )
        expected_labels.add(f"lara-fork.{index}")

    first, second = await asyncio.gather(main.collect(), main.collect())

    # no KeyError, no lost outcomes: both jobs are delivered exactly once
    collected = (*first, *second)
    assert len(collected) == 2
    assert {outcome.label for outcome in collected} == expected_labels
    assert all(outcome.error is None for outcome in collected)
    assert len({outcome.job_id for outcome in collected}) == 2
    assert await main.collect() == ()


@pytest.mark.asyncio
async def test_job_ids_are_unique_across_reused_labels() -> None:
    runner, _ = _runner([AIMessage(content="done")])
    main = MainAgent(runner, system_prompt="p")
    spec = ForkSpec(source=_WindowSource.of("p"))

    first = main.spawn_fork(spec, directive="go", label="lara-fork.same")
    first_outcomes = await main.collect()
    second = main.spawn_fork(spec, directive="go", label="lara-fork.same")
    second_outcomes = await main.collect()

    assert first.job_id != second.job_id
    assert first_outcomes[0].job_id == first.job_id
    assert second_outcomes[0].job_id == second.job_id
    assert first_outcomes[0].label == second_outcomes[0].label
