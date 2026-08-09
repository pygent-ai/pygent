from __future__ import annotations

import asyncio

import pytest

from pygent import (
    AIMessage,
    Context,
    Module,
    UserMessage,
)
from pygent.core import (
    EffectIdempotency,
    EffectRetryPolicy,
    EffectSafety,
    EffectSideEffect,
    EffectSpec,
    ExecutionRequirements,
    RecoverySafety,
)
from pygent.core._module_contracts import _execution_scope
from pygent.runtime import (
    ExecutionAdmissionError,
    ExecutionOptions,
    ExecutionStatus,
    HistoryConflictError,
    LocalRuntime,
    SQLiteHistoryStore,
)
from pygent.runtime.codec import invocation_to_dict


class _Counter:
    def __init__(self) -> None:
        self.calls = 0


class CountingEcho(Module[UserMessage, AIMessage]):
    execution_requirements = ExecutionRequirements(
        recovery_safety=RecoverySafety.MODULE_BOUNDARY_RETRY,
        effect_safety=EffectSafety.EFFECT_FREE,
    )
    trusted_live_resource_attributes = ("counter",)

    def __init__(self, counter: _Counter) -> None:
        super().__init__()
        self.counter = counter

    async def forward(self, message, context):
        self.counter.calls += 1
        output = AIMessage(content=message.content.upper())
        return output, context + message + output


class EffectModule(Module[UserMessage, AIMessage]):
    execution_requirements = ExecutionRequirements(
        recovery_safety=RecoverySafety.MODULE_BOUNDARY_RETRY,
        effect_safety=EffectSafety.MANAGED_EFFECTS,
    )
    trusted_live_resource_attributes = ("counter",)

    def __init__(self, counter: _Counter) -> None:
        super().__init__()
        self.counter = counter

    async def forward(self, message, context):
        scope = _execution_scope.get()
        assert scope is not None

        async def operation():
            self.counter.calls += 1
            return {"content": message.content.upper()}

        result = await scope.execute_effect(
            spec=EffectSpec(
                effect_type="test.provider",
                side_effect=EffectSideEffect.READ,
                idempotency=EffectIdempotency.INHERENT,
                retry_policy=EffectRetryPolicy.REPLAY_SAFE,
            ),
            request={"content": message.content},
            operation=operation,
        )
        content = result.value["content"]
        assert isinstance(content, str)
        output = AIMessage(content=content)
        return output, context + message + output


class BlockingRecover(Module[UserMessage, AIMessage]):
    execution_requirements = ExecutionRequirements(
        recovery_safety=RecoverySafety.MODULE_BOUNDARY_RETRY,
        effect_safety=EffectSafety.EFFECT_FREE,
    )
    trusted_live_resource_attributes = ("entered", "release")

    def __init__(self, entered: asyncio.Event, release: asyncio.Event) -> None:
        super().__init__()
        self.entered = entered
        self.release = release

    async def forward(self, message, context):
        self.entered.set()
        await self.release.wait()
        return AIMessage(content=message.content), context


class _GatedTerminalHistory(SQLiteHistoryStore):
    def __init__(self, path) -> None:
        super().__init__(path)
        self.terminal_update_entered = asyncio.Event()
        self.release_terminal_update = asyncio.Event()

    async def finalize_execution(self, execution_id, **kwargs) -> None:
        self.terminal_update_entered.set()
        await self.release_terminal_update.wait()
        await super().finalize_execution(execution_id, **kwargs)


class EmitsThenReturns(Module[UserMessage, AIMessage]):
    async def forward(self, message, context):
        await self.emit(kind="test.progress", data={})
        output = AIMessage(content=message.content.upper())
        return output, context + message + output


@pytest.mark.asyncio
async def test_live_subscription_observes_terminal_events_only_after_atomic_finalization(
    tmp_path,
):
    async with _GatedTerminalHistory(tmp_path / "history.sqlite3") as history:
        runtime = LocalRuntime(history=history)
        handle = await runtime.bind(EmitsThenReturns()).start(
            UserMessage(content="hello"), Context()
        )
        live_events = []
        progress_seen = asyncio.Event()

        async def consume_events() -> None:
            async with handle.subscribe() as subscription:
                async for event in subscription:
                    live_events.append(event)
                    if event.kind == "test.progress":
                        progress_seen.set()

        consumer = asyncio.create_task(consume_events())
        await asyncio.wait_for(history.terminal_update_entered.wait(), timeout=1)
        await asyncio.wait_for(progress_seen.wait(), timeout=1)
        await asyncio.sleep(0)

        assert handle.status is ExecutionStatus.RUNNING
        assert consumer.done() is False
        claim = await (
            await history._db().execute(
                "SELECT owner_id FROM execution_claims WHERE execution_id=?",
                (handle.execution_id,),
            )
        ).fetchone()
        assert claim is not None
        assert (await handle.snapshot()).attempt_id in claim[0]

        history.release_terminal_update.set()
        await asyncio.wait_for(handle.result(), timeout=1)
        await asyncio.wait_for(consumer, timeout=1)
        persisted_events = await history.events_after(
            execution_id=handle.execution_id, after=-1
        )

        live_kinds = [event.kind for event in live_events]
        assert live_kinds[-2:] == ["span.completed", "execution.completed"]
        assert live_kinds == [event["kind"] for event in persisted_events]
        released_claim = await (
            await history._db().execute(
                "SELECT owner_id FROM execution_claims WHERE execution_id=?",
                (handle.execution_id,),
            )
        ).fetchone()
        assert released_claim is None
        await runtime.close()


@pytest.mark.asyncio
async def test_completed_durable_run_restores_without_reexecuting_forward(tmp_path):
    counter = _Counter()
    path = tmp_path / "history.sqlite3"
    async with SQLiteHistoryStore(path) as history:
        runtime = LocalRuntime(history=history)
        bound = runtime.bind(CountingEcho(counter))
        first = await bound.start(
            UserMessage(content="hello"),
            Context(),
            execution=ExecutionOptions(request_id="request-1"),
        )
        expected = await first.result()
        execution_id = first.execution_id
        await runtime.close()
    assert counter.calls == 1

    async with SQLiteHistoryStore(path) as history:
        restored_runtime = LocalRuntime(history=history)
        restored = await restored_runtime.get_execution_handle(execution_id)
        assert await restored.result() == expected
        snapshot = await restored.snapshot()
        outcome = await restored.outcome()
        assert snapshot.submitted_at_unix_ns <= snapshot.updated_at_unix_ns
        assert snapshot.terminal_sequence == outcome.terminal_sequence
        async with restored.subscribe() as subscription:
            restored_events = [event async for event in subscription]
        assert restored_events[-1].kind == "execution.completed"
        with pytest.raises(ExecutionAdmissionError, match="attach to its existing outcome"):
            await restored_runtime.recover(
                restored_runtime.bind(CountingEcho(counter)), execution_id
            )
        await restored_runtime.close()
    assert counter.calls == 1


@pytest.mark.asyncio
async def test_unfinished_durable_run_restarts_at_module_boundary(tmp_path):
    counter = _Counter()
    path = tmp_path / "history.sqlite3"
    async with SQLiteHistoryStore(path) as history:
        runtime = LocalRuntime(history=history)
        bound = runtime.bind(CountingEcho(counter))
        await history.create_execution(
            execution_id="crashed-run",
            request_id="request-crashed",
            plan_id=bound.plan.plan_id,
            input=invocation_to_dict(UserMessage(content="again"), Context()),
            status="running",
        )
        recovered = await runtime.recover(bound, "crashed-run")
        output, _ = await recovered.result()
        stored = await history.get_execution("crashed-run")
        await runtime.close()

    assert output.content == "AGAIN"
    assert counter.calls == 1
    assert stored is not None and stored.attempt == 2
    assert stored.status == "succeeded"


@pytest.mark.asyncio
async def test_durable_recovery_claim_rejects_concurrent_owner(tmp_path):
    path = tmp_path / "claimed.sqlite3"
    entered, release = asyncio.Event(), asyncio.Event()
    async with (
        SQLiteHistoryStore(path) as first_history,
        SQLiteHistoryStore(path) as second_history,
    ):
        first_runtime = LocalRuntime(history=first_history)
        second_runtime = LocalRuntime(history=second_history)
        first_bound = first_runtime.bind(BlockingRecover(entered, release))
        second_bound = second_runtime.bind(BlockingRecover(entered, release))
        await first_history.create_execution(
            execution_id="shared-run",
            request_id="request",
            plan_id=first_bound.plan.plan_id,
            input=invocation_to_dict(UserMessage(content="x"), Context()),
            status="running",
        )
        first = await first_runtime.recover(first_bound, "shared-run")
        await entered.wait()
        with pytest.raises(ExecutionAdmissionError, match="owned"):
            await second_runtime.recover(second_bound, "shared-run")
        release.set()
        await first.result()
        await first_runtime.close()
        await second_runtime.close()


@pytest.mark.asyncio
async def test_unfinished_run_replays_committed_effect_without_reexecution(tmp_path):
    counter = _Counter()
    path = tmp_path / "history.sqlite3"
    async with SQLiteHistoryStore(path) as history:
        runtime = LocalRuntime(history=history)
        bound = runtime.bind(EffectModule(counter))
        await history.create_execution(
            execution_id="effect-run",
            request_id="effect-request",
            plan_id=bound.plan.plan_id,
            input=invocation_to_dict(UserMessage(content="once"), Context()),
            status="running",
        )
        await history.record_effect(
            execution_id="effect-run",
            module_path=bound.plan.root,
            call_index=0,
            effect_type="test.provider",
            request={"content": "once"},
            result={"content": "ONCE"},
        )

        recovered = await runtime.recover(bound, "effect-run")
        output, _ = await recovered.result()
        await runtime.close()

    assert output.content == "ONCE"
    assert counter.calls == 0


@pytest.mark.asyncio
async def test_idempotency_key_reuses_run_but_request_id_does_not(tmp_path):
    counter = _Counter()
    async with SQLiteHistoryStore(tmp_path / "history.sqlite3") as history:
        runtime = LocalRuntime(history=history)
        bound = runtime.bind(CountingEcho(counter))
        first = await bound.start(
            UserMessage(content="same"),
            Context(),
            execution=ExecutionOptions(
                request_id="transport-1",
                identity="user-1",
                idempotency_key="operation-1",
            ),
        )
        await first.result()
        repeated = await bound.start(
            UserMessage(content="same"),
            Context(),
            execution=ExecutionOptions(
                request_id="transport-2",
                identity="user-1",
                idempotency_key="operation-1",
            ),
        )
        await repeated.result()
        assert repeated.execution_id == first.execution_id

        independent = await bound.start(
            UserMessage(content="same"),
            Context(),
            execution=ExecutionOptions(request_id="transport-1"),
        )
        await independent.result()
        assert independent.execution_id != first.execution_id

        with pytest.raises(HistoryConflictError):
            await bound.start(
                UserMessage(content="different"),
                Context(),
                execution=ExecutionOptions(
                    identity="user-1",
                    idempotency_key="operation-1",
                ),
            )
        await runtime.close()

    assert counter.calls == 2
