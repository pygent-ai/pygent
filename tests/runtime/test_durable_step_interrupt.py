"""Durable immediate steering: in-flight interruption, journaling, and replay.

These tests pin the durable execution contract that a committed immediate
steering input interrupts the in-flight model/tool step, is consumed at the
next deterministic drain, and that the interrupted step is a recoverable
journal fact rather than a live-signal side effect.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import sqlite3
from pathlib import Path
from typing import Any

import aiosqlite
import pytest

from pygent.agent import (
    REACT_PROJECTION_OPERATION_KIND,
    ReActLayer,
    StandaloneUserMessage,
    SteeringMode,
    encode_react_projection_operation,
)
from pygent.core import (
    AIMessage,
    Context,
    EffectIdempotency,
    EffectRecoveryUnknown,
    EffectRetryPolicy,
    EffectSafety,
    EffectSideEffect,
    EffectSpec,
    ExecutionRequirements,
    ExecutionStatus,
    JsonValue,
    Message,
    Module,
    RecoverySafety,
    ToolMessage,
    UserMessage,
    current_infrastructure,
)
from pygent.runtime import (
    CapacityPolicy,
    CapacityScope,
    ExecutionCapacityPolicy,
    ExecutionOptions,
    LocalRuntime,
    SQLiteHistoryStore,
)
from pygent.runtime.codec import invocation_to_dict
from pygent.tool import (
    ExecutorRegistry,
    LocalToolExecutor,
    ToolAuthorizationDecision,
    ToolAuthorizationRequest,
    ToolCall,
    ToolCallLayer,
    ToolDefinition,
    ToolSideEffect,
    ToolSpec,
)


def _options() -> ExecutionOptions:
    return ExecutionOptions(deadline=asyncio.get_running_loop().time() + 60)


def _immediate(content: str = "new direction") -> JsonValue:
    operation = StandaloneUserMessage(UserMessage(content=content), mode=SteeringMode.IMMEDIATE)
    return encode_react_projection_operation(operation)


def _react_context() -> Context:
    return Context(
        tools=(
            ToolDefinition(
                name="probe",
                description="probe work",
                parameters={
                    "type": "object",
                    "properties": {"tag": {"type": "string"}},
                    "required": ["tag"],
                    "additionalProperties": False,
                },
            ),
        ),
        metadata={"permissions": ["tool:use"]},
    )


def _durable_bind(runtime: LocalRuntime, react: ReActLayer):
    binding = runtime.create_binding(
        name="durable-steering",
        execution_capacity=ExecutionCapacityPolicy(
            scope=CapacityScope.RUNTIME_INSTANCE,
            max_live_executions=2,
            max_runnable_executions=2,
            max_queue_size=2,
            max_waiters=2,
            max_child_depth=8,
            max_children_per_execution=16,
        ),
        model_capacity=CapacityPolicy.passthrough(),
        tool_capacity=CapacityPolicy.passthrough(),
    )
    return binding.bind(react)


async def _wait_for_step_interrupt(
    db: Path, *, timeout: float = 30.0, minimum: int = 1
) -> None:
    """Barrier: the journaled interrupt exists only after the child joined."""

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            async with aiosqlite.connect(db) as conn:
                row = await (
                    await conn.execute(
                        "SELECT COUNT(*) FROM execution_step_interrupts"
                    )
                ).fetchone()
            if row is not None and int(row[0]) >= minimum:
                return
        except sqlite3.OperationalError:
            pass
        await asyncio.sleep(0.01)
    pytest.fail("durable step interrupt was not journaled")


class _GateState:
    def __init__(self) -> None:
        self.calls = 0
        self.aborts = 0
        self.messages: list[Message] = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()


class _GatedModel(Module[Message, AIMessage]):
    """Every call blocks until released; cancellations are counted."""

    execution_requirements = ExecutionRequirements(
        recovery_safety=RecoverySafety.MODULE_BOUNDARY_RETRY,
        effect_safety=EffectSafety.EFFECT_FREE,
    )
    trusted_live_resource_attributes = ("state",)

    def __init__(self) -> None:
        super().__init__()
        self.state = _GateState()

    async def forward(self, message: Message, context: Context):
        self.state.calls += 1
        self.state.messages.append(message)
        self.state.entered.set()
        try:
            await self.state.release.wait()
        except asyncio.CancelledError:
            self.state.aborts += 1
            raise
        return AIMessage(content=f"answer-{self.state.calls}"), context


class _CrashModelState:
    def __init__(self) -> None:
        self.calls = 0


class _CrashModel(Module[Message, AIMessage]):
    """Turn 1 emits a tool batch; turn 2 parks on a flag file when gated."""

    execution_requirements = ExecutionRequirements(
        recovery_safety=RecoverySafety.MODULE_BOUNDARY_RETRY,
        effect_safety=EffectSafety.EFFECT_FREE,
    )
    # Process-local deployment knobs must not enter the portable plan hash.
    trusted_live_resource_attributes = ("state", "_workspace", "_gate_turn2")

    def __init__(self, workspace: Path, *, gate_turn2: bool) -> None:
        super().__init__()
        self._workspace = str(workspace)
        self._gate_turn2 = gate_turn2
        self.state = _CrashModelState()

    async def forward(self, message: Message, context: Context):
        self.state.calls += 1
        if self.state.calls == 1:
            return (
                AIMessage(
                    content="",
                    tool_calls=(
                        ToolCall(
                            call_id="fast", name="probe", arguments={"tag": "fast"}
                        ),
                        ToolCall(
                            call_id="slow", name="probe", arguments={"tag": "slow"}
                        ),
                    ),
                ),
                context,
            )
        if self._gate_turn2:
            workspace = Path(self._workspace)
            while not (workspace / "turn2.flag").exists():
                await asyncio.sleep(0.02)
        return AIMessage(content="final"), context


class _AllowAuthorization(Module[ToolAuthorizationRequest, ToolAuthorizationDecision]):
    """Recoverable authorization: keeps the tool graph durability-eligible."""

    execution_requirements = ExecutionRequirements(
        recovery_safety=RecoverySafety.MODULE_BOUNDARY_RETRY,
        effect_safety=EffectSafety.EFFECT_FREE,
    )

    async def forward(
        self, request: ToolAuthorizationRequest, context: Context
    ) -> tuple[ToolAuthorizationDecision, Context]:
        return (
            ToolAuthorizationDecision(
                call_id=request.call.call_id,
                allowed=True,
                reason_code="allowed",
                lifecycle="sync",
            ),
            context,
        )


def _tool_layer(workspace: Path) -> ToolCallLayer:
    tool = ToolSpec(
        tool_id="probe.work",
        version="1",
        definition=ToolDefinition(
            name="probe",
            description="probe work",
            parameters={
                "type": "object",
                "properties": {"tag": {"type": "string"}},
                "required": ["tag"],
                "additionalProperties": False,
            },
        ),
        side_effect=ToolSideEffect.READ,
    )

    async def work(arguments: Any) -> Any:
        tag = arguments["tag"]
        if tag == "fast":
            _append_line(workspace / "effects.log", "fast")
            return {"tag": tag}
        # The slow call signals the orchestrator, then parks until released.
        _append_line(workspace / "slow.entered", "1")
        while not (workspace / "slow.release").exists():
            await asyncio.sleep(0.02)
        _append_line(workspace / "effects.log", "slow")
        return {"tag": tag}

    registry = ExecutorRegistry()
    registry.register(tool.tool_id, tool.version, LocalToolExecutor(work))
    return ToolCallLayer(
        tools=(tool,),
        authorization=_AllowAuthorization(),
        executor_registry=registry,
        max_concurrency=2,
    )


def _append_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{line}\n")


# ---------------------------------------------------------------------------
# In-process durable steering behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_durable_duplicate_input_is_rejected_without_new_interrupt(
    tmp_path,
) -> None:
    async with SQLiteHistoryStore(tmp_path / "history.sqlite3") as history:
        runtime = LocalRuntime(history=history)
        model = _GatedModel()
        bound = _durable_bind(runtime, ReActLayer(model=model, tools=None))
        handle = await bound.start(
            UserMessage(content="initial"), Context(), execution=_options()
        )
        await model.state.entered.wait()
        first = await handle.send_input(
            input_id="steer-1",
            kind=REACT_PROJECTION_OPERATION_KIND,
            value=_immediate(),
        )
        assert first.status == "accepted"
        await _wait_for_step_interrupt(tmp_path / "history.sqlite3")
        while model.state.calls < 2:
            await asyncio.sleep(0.01)
        await model.state.entered.wait()

        duplicate = await handle.send_input(
            input_id="steer-1",
            kind=REACT_PROJECTION_OPERATION_KIND,
            value=_immediate(),
        )
        assert duplicate.status == "duplicate"
        await asyncio.sleep(0.1)
        assert model.state.aborts == 1

        model.state.release.set()
        answer, _ = await asyncio.wait_for(handle.result(), timeout=15)
        assert answer.content == "answer-2"
        # The decision carries the verified trigger identity of the input the
        # module steered with.
        stored = await history.get_step_interrupt(
            execution_id=handle.execution_id,
            module_path=bound.plan.root,
            step_index=0,
        )
        assert stored is not None
        assert stored.trigger_input_sequence == 0
        await runtime.close()


@pytest.mark.asyncio
async def test_notification_covers_commit_during_dispatch_peek(
    tmp_path, monkeypatch
) -> None:
    # A commit that lands while the dispatch task already holds its peek read
    # must still be dispatched: its own notify finds the task running, so the
    # clear-peek-wait loop -- not the task-done check -- carries the wakeup.
    async with SQLiteHistoryStore(tmp_path / "history.sqlite3") as history:
        runtime = LocalRuntime(history=history)
        model = _GatedModel()
        handle = await _durable_bind(runtime, ReActLayer(model=model, tools=None)).start(
            UserMessage(content="initial"), Context(), execution=_options()
        )
        await model.state.entered.wait()

        original_peek = history.peek_execution_inputs
        peeks = 0
        second_peek_read = asyncio.Event()
        release_second_peek = asyncio.Event()

        async def gated_peek(*, execution_id, kinds):
            nonlocal peeks
            peeks += 1
            result = await original_peek(execution_id=execution_id, kinds=kinds)
            if peeks == 2:
                # Park after the read so the next commit lands in the window
                # between the observed-empty peek and the dispatch exiting.
                second_peek_read.set()
                await release_second_peek.wait()
            return result

        monkeypatch.setattr(history, "peek_execution_inputs", gated_peek)

        first = await handle.send_input(
            input_id="steer-1",
            kind=REACT_PROJECTION_OPERATION_KIND,
            value=_immediate("first"),
        )
        assert first.status == "accepted"
        await _wait_for_step_interrupt(tmp_path / "history.sqlite3")
        while model.state.calls < 2:
            await asyncio.sleep(0.01)
        await second_peek_read.wait()

        second = await handle.send_input(
            input_id="steer-2",
            kind=REACT_PROJECTION_OPERATION_KIND,
            value=_immediate("second"),
        )
        assert second.status == "accepted"
        release_second_peek.set()

        await _wait_for_step_interrupt(
            tmp_path / "history.sqlite3", minimum=2
        )
        model.state.release.set()
        answer, context = await asyncio.wait_for(handle.result(), timeout=15)

        assert model.state.aborts == 2
        assert answer.content == "answer-3"
        assert [message.content for message in model.state.messages] == [
            "initial",
            "first",
            "second",
        ]
        committed = [message.content for message in context.messages]
        assert committed.index("first") < committed.index("second")
        await runtime.close()


@pytest.mark.asyncio
async def test_durable_multiple_immediates_apply_in_sequence_order(tmp_path) -> None:
    async with SQLiteHistoryStore(tmp_path / "history.sqlite3") as history:
        runtime = LocalRuntime(history=history)
        model = _GatedModel()
        handle = await _durable_bind(runtime, ReActLayer(model=model, tools=None)).start(
            UserMessage(content="initial"), Context(), execution=_options()
        )
        await model.state.entered.wait()
        first = await handle.send_input(
            input_id="steer-1",
            kind=REACT_PROJECTION_OPERATION_KIND,
            value=_immediate("first"),
        )
        second = await handle.send_input(
            input_id="steer-2",
            kind=REACT_PROJECTION_OPERATION_KIND,
            value=_immediate("second"),
        )
        assert (first.status, second.status) == ("accepted", "accepted")
        await _wait_for_step_interrupt(tmp_path / "history.sqlite3")
        while model.state.calls < 2:
            await asyncio.sleep(0.01)
        model.state.release.set()
        _, context = await asyncio.wait_for(handle.result(), timeout=15)

        assert model.state.aborts == 1
        # Both inputs were consumed in sequence order; the last one is the
        # message the next model step actually sees.
        assert model.state.messages[-1].content == "second"
        committed = [message.content for message in context.messages]
        assert committed.index("first") < committed.index("second")
        await runtime.close()


# ---------------------------------------------------------------------------
# Replay adjudication of a journaled step interrupt
# ---------------------------------------------------------------------------


class _GatedEffectProbe(Module[UserMessage, AIMessage]):
    """One managed effect whose operation parks until the test releases it."""

    execution_requirements = ExecutionRequirements(
        recovery_safety=RecoverySafety.MODULE_BOUNDARY_RETRY,
        effect_safety=EffectSafety.MANAGED_EFFECTS,
    )

    def __init__(self, workspace: Path) -> None:
        super().__init__()
        self._workspace = str(workspace)

    async def forward(self, message: UserMessage, context: Context):
        infrastructure = current_infrastructure()
        workspace = Path(self._workspace)

        async def operation():
            while not (workspace / "effect.release").exists():
                await asyncio.sleep(0.02)
            return {"released": True}

        await infrastructure.execute_effect(
            spec=EffectSpec(
                effect_type="probe.effect",
                side_effect=EffectSideEffect.READ,
                idempotency=EffectIdempotency.INHERENT,
                retry_policy=EffectRetryPolicy.REPLAY_SAFE,
            ),
            request={"tag": "gated"},
            operation=operation,
        )
        return AIMessage(content="done"), context


class _FailClosedProbe(Module[UserMessage, AIMessage]):
    execution_requirements = ExecutionRequirements(
        recovery_safety=RecoverySafety.MODULE_BOUNDARY_RETRY,
        effect_safety=EffectSafety.MANAGED_EFFECTS,
    )

    async def forward(self, message: UserMessage, context: Context):
        infrastructure = current_infrastructure()

        async def operation():
            return {"content": message.content}

        await infrastructure.execute_effect(
            spec=EffectSpec(
                effect_type="probe.write",
                side_effect=EffectSideEffect.EXTERNAL,
                idempotency=EffectIdempotency.NOT_IDEMPOTENT,
                retry_policy=EffectRetryPolicy.FAIL_CLOSED,
            ),
            request={"content": message.content},
            operation=operation,
        )
        return AIMessage(content="written"), context


async def _wait_for_effect_status(
    db: Path, status: str, *, timeout: float = 30.0
) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            async with aiosqlite.connect(db) as conn:
                row = await (
                    await conn.execute(
                        "SELECT COUNT(*) FROM effects WHERE status=?", (status,)
                    )
                ).fetchone()
            if row is not None and int(row[0]) >= 1:
                return
        except sqlite3.OperationalError:
            pass
        await asyncio.sleep(0.01)
    pytest.fail(f"no effect reached status {status!r}")


@pytest.mark.asyncio
async def test_deadline_during_effect_leaves_started_state_not_interrupted(
    tmp_path,
) -> None:
    # Lifecycle cancellations are not steering interrupts: the effect keeps
    # its started row and no step decision is journaled.
    async with SQLiteHistoryStore(tmp_path / "history.sqlite3") as history:
        runtime = LocalRuntime(history=history)
        bound = runtime.bind(_GatedEffectProbe(tmp_path))
        handle = await bound.start(
            UserMessage(content="go"),
            Context(),
            execution=ExecutionOptions(
                deadline=asyncio.get_running_loop().time() + 0.5
            ),
        )
        await _wait_for_effect_status(tmp_path / "history.sqlite3", "started")
        outcome = await asyncio.wait_for(handle.outcome(), timeout=15)

        assert outcome.status is ExecutionStatus.DEADLINE_EXCEEDED
        rows = await history._db().execute_fetchall("SELECT status FROM effects")
        assert [row[0] for row in rows] == ["started"]
        interrupts = await history._db().execute_fetchall(
            "SELECT COUNT(*) FROM execution_step_interrupts"
        )
        assert int(interrupts[0][0]) == 0
        await runtime.close()


@pytest.mark.asyncio
async def test_execution_cancel_during_effect_leaves_started_state(
    tmp_path,
) -> None:
    async with SQLiteHistoryStore(tmp_path / "history.sqlite3") as history:
        runtime = LocalRuntime(history=history)
        bound = runtime.bind(_GatedEffectProbe(tmp_path))
        handle = await bound.start(
            UserMessage(content="go"), Context(), execution=_options()
        )
        await _wait_for_effect_status(tmp_path / "history.sqlite3", "started")
        assert await handle.cancel() is True
        outcome = await handle.outcome()

        assert outcome.status is ExecutionStatus.CANCELLED
        rows = await history._db().execute_fetchall("SELECT status FROM effects")
        assert [row[0] for row in rows] == ["started"]
        interrupts = await history._db().execute_fetchall(
            "SELECT COUNT(*) FROM execution_step_interrupts"
        )
        assert int(interrupts[0][0]) == 0
        await runtime.close()


@pytest.mark.asyncio
async def test_started_effect_without_decision_recovers_by_original_rules(
    tmp_path,
) -> None:
    # Crash before a step decision was journaled: the interrupt never became
    # durable, so a started FAIL_CLOSED effect follows the original recovery
    # rules instead of claiming a steering interrupt.
    async with SQLiteHistoryStore(tmp_path / "history.sqlite3") as history:
        runtime = LocalRuntime(history=history)
        bound = runtime.bind(_FailClosedProbe())
        await history.create_execution(
            execution_id="gap-run",
            request_id="gap-request",
            plan_id=bound.plan.plan_id,
            input=invocation_to_dict(UserMessage(content="go"), Context()),
            status="running",
        )
        await history.begin_effect(
            execution_id="gap-run",
            module_path=bound.plan.root,
            call_index=0,
            effect_type="probe.write",
            request={"content": "go"},
            spec={
                "side_effect": "external",
                "idempotency": "not_idempotent",
                "retry_policy": "fail_closed",
                "idempotency_key": None,
            },
        )

        recovered = await runtime.recover(
            bound,
            "gap-run",
            deadline=asyncio.get_running_loop().time() + 30,
        )
        with pytest.raises(EffectRecoveryUnknown):
            await asyncio.wait_for(recovered.result(), timeout=30)
        await runtime.close()


class _GatherProbe(Module[UserMessage, AIMessage]):
    """Two parallel managed effects; reports how many children kept results."""

    execution_requirements = ExecutionRequirements(
        recovery_safety=RecoverySafety.MODULE_BOUNDARY_RETRY,
        effect_safety=EffectSafety.MANAGED_EFFECTS,
    )

    async def forward(self, message: UserMessage, context: Context):
        infrastructure = current_infrastructure()

        async def one(tag: str, pre_delay: float):
            if pre_delay:
                await asyncio.sleep(pre_delay)

            async def operation():
                return {"tag": tag}

            outcome = await infrastructure.execute_effect(
                spec=EffectSpec(
                    effect_type="probe.effect",
                    side_effect=EffectSideEffect.READ,
                    idempotency=EffectIdempotency.INHERENT,
                    retry_policy=EffectRetryPolicy.REPLAY_SAFE,
                ),
                request={"tag": tag},
                operation=operation,
            )
            return outcome.value

        _, completed, _ = await infrastructure.gather_until(
            (
                lambda: one("a", 0.0),
                lambda: one("b", 0.05),
            ),
            abort=asyncio.Event(),
        )
        return AIMessage(content=f"completed={sum(completed)}"), context


@pytest.mark.asyncio
async def test_replayed_interrupt_discards_children_the_journal_cancelled(
    tmp_path,
) -> None:
    # A child whose last effect committed before cancellation was applied must
    # still replay as cancelled: the journaled step decision is authoritative.
    async with SQLiteHistoryStore(tmp_path / "history.sqlite3") as history:
        runtime = LocalRuntime(history=history)
        bound = runtime.bind(_GatherProbe())
        execution_id = "probe-run"
        await history.create_execution(
            execution_id=execution_id,
            request_id="probe-request",
            plan_id=bound.plan.plan_id,
            input=invocation_to_dict(UserMessage(content="go"), Context()),
            status="running",
        )
        await history.record_effect(
            execution_id=execution_id,
            module_path=bound.plan.root,
            call_index=0,
            effect_type="probe.effect",
            request={"tag": "a"},
            result={"tag": "a"},
        )
        await history.begin_effect(
            execution_id=execution_id,
            module_path=bound.plan.root,
            call_index=1,
            effect_type="probe.effect",
            request={"tag": "b"},
            spec={
                "side_effect": "read",
                "idempotency": "inherent",
                "retry_policy": "replay_safe",
                "idempotency_key": None,
            },
        )
        await history.interrupt_effect(
            execution_id=execution_id,
            module_path=bound.plan.root,
            call_index=1,
        )
        stored = await history.mark_step_interrupted(
            execution_id=execution_id,
            module_path=bound.plan.root,
            step_index=0,
            cancelled=(0, 1),
        )
        assert stored.cancelled == (0, 1)

        recovered = await runtime.recover(
            bound,
            execution_id,
            deadline=asyncio.get_running_loop().time() + 60,
        )
        output, _ = await asyncio.wait_for(recovered.result(), timeout=30)
        await runtime.close()

    assert output.content == "completed=0"


# ---------------------------------------------------------------------------
# Owner crash after a journaled interrupt
# ---------------------------------------------------------------------------


def _crash_child_main(workspace: str) -> None:
    asyncio.run(_crash_child(workspace))


async def _crash_child(workspace: str) -> None:
    ws = Path(workspace)
    history = await SQLiteHistoryStore(ws / "history.sqlite3").open()
    runtime = LocalRuntime(history=history)
    runtime._recovery_lease_ttl = 0.1
    tools = _tool_layer(ws)
    model = _CrashModel(ws, gate_turn2=True)
    bound = _durable_bind(runtime, ReActLayer(model=model, tools=tools))
    handle = await bound.start(
        UserMessage(content="initial"), _react_context(), execution=_options()
    )
    (ws / "execution_id.txt").write_text(handle.execution_id, encoding="ascii")

    async def steer() -> None:
        while not (ws / "slow.entered").exists():
            await asyncio.sleep(0.01)
        await handle.send_input(
            input_id="steer-1",
            kind=REACT_PROJECTION_OPERATION_KIND,
            value=_immediate(),
        )

    asyncio.create_task(steer())
    # Park: the parent kills this process once the journal shows the interrupt.
    await asyncio.Event().wait()


async def _wait_for_crash_point(db: Path, timeout: float) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            async with aiosqlite.connect(db) as conn:
                step = await (
                    await conn.execute(
                        "SELECT COUNT(*) FROM execution_step_interrupts"
                    )
                ).fetchone()
                interrupted = await (
                    await conn.execute(
                        "SELECT COUNT(*) FROM effects WHERE status='interrupted'"
                    )
                ).fetchone()
                receipts = await (
                    await conn.execute(
                        "SELECT COUNT(*) FROM execution_input_receives"
                    )
                ).fetchone()
            if (
                step is not None
                and int(step[0]) >= 1
                and interrupted is not None
                and int(interrupted[0]) >= 1
                and receipts is not None
                and int(receipts[0]) >= 1
            ):
                return
        except sqlite3.OperationalError:
            pass
        await asyncio.sleep(0.05)
    pytest.fail("crash child never reached the journaled interrupt state")


@pytest.mark.asyncio
async def test_durable_step_interrupt_recovers_after_owner_crash(tmp_path) -> None:
    ws = tmp_path
    context = multiprocessing.get_context("spawn")
    proc = context.Process(target=_crash_child_main, args=(str(ws),), daemon=True)
    proc.start()
    try:
        await _wait_for_crash_point(ws / "history.sqlite3", timeout=120)
    finally:
        proc.kill()
        await asyncio.to_thread(proc.join, 30)
    assert proc.exitcode != 0, "the crash child should have been killed, not finished"
    # The killed owner's claim lease must lapse before another attempt may run.
    await asyncio.sleep(0.3)

    execution_id = (ws / "execution_id.txt").read_text(encoding="ascii").strip()
    async with SQLiteHistoryStore(ws / "history.sqlite3") as history:
        runtime = LocalRuntime(history=history)
        tools = _tool_layer(ws)
        model = _CrashModel(ws, gate_turn2=False)
        bound = _durable_bind(runtime, ReActLayer(model=model, tools=tools))
        recovered = await runtime.recover(
            bound,
            execution_id,
            deadline=asyncio.get_running_loop().time() + 60,
        )
        answer, recovered_context = await asyncio.wait_for(
            recovered.result(), timeout=60
        )
        stored = await history.get_execution(execution_id)
        await runtime.close()

    assert answer.content == "final"
    assert stored is not None and stored.status == "succeeded"
    tool_messages = [
        message
        for message in recovered_context.messages
        if isinstance(message, ToolMessage)
    ]
    assert len(tool_messages) == 1
    results = {result.call_id: result for result in tool_messages[0].results}
    # The finished call keeps its committed result; the cancelled one does not
    # re-execute on replay.
    assert results["fast"].status == "succeeded"
    assert results["slow"].status == "cancelled"
    assert results["slow"].error_code == "interrupted_by_steering"
    committed = [message.content for message in recovered_context.messages]
    assert "new direction" in committed
    side_effects = (ws / "effects.log").read_text(encoding="utf-8").splitlines()
    assert side_effects.count("fast") == 1
    assert "slow" not in side_effects
