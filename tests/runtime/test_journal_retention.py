from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from pygent import AIMessage, Context, Module, UserMessage
from pygent.core import EffectSafety, ExecutionRequirements, RecoverySafety
from pygent.runtime import LocalRuntime, SQLiteHistoryStore
from pygent.runtime.codec import invocation_to_dict

RETENTION_SECONDS = 3600.0


class Echo(Module[UserMessage, AIMessage]):
    execution_requirements = ExecutionRequirements(
        recovery_safety=RecoverySafety.MODULE_BOUNDARY_RETRY,
        effect_safety=EffectSafety.EFFECT_FREE,
    )

    async def forward(self, message, context):
        output = AIMessage(content=message.content.upper())
        return output, context + message + output


async def _age(history: SQLiteHistoryStore, execution_id: str, *, seconds: float) -> None:
    db = history._db()
    await db.execute(
        "UPDATE executions SET updated_at_unix_ns=? WHERE execution_id=?",
        (time.time_ns() - int(seconds * 1e9), execution_id),
    )
    await db.commit()


async def _row_counts(history: SQLiteHistoryStore, execution_id: str) -> dict[str, int]:
    db = history._db()
    counts: dict[str, int] = {}
    for table in (
        "executions",
        "events",
        "effects",
        "execution_claims",
        "execution_inboxes",
        "execution_model_admissions",
    ):
        rows = await db.execute_fetchall(
            f"SELECT COUNT(*) FROM {table} WHERE execution_id=?", (execution_id,)
        )
        counts[table] = int(rows[0][0])
    return counts


async def _run(runtime: LocalRuntime, bound, content: str) -> str:
    handle = await bound.start(UserMessage(content=content), Context())
    await handle.result()
    return handle.execution_id


@pytest.mark.asyncio
async def test_prune_deletes_expired_terminal_executions_and_keeps_recent(
    tmp_path: Path,
) -> None:
    async with SQLiteHistoryStore(tmp_path / "history.sqlite3") as history:
        runtime = LocalRuntime(
            history=history, journal_retention_seconds=RETENTION_SECONDS
        )
        bound = runtime.bind(Echo())
        expired = await _run(runtime, bound, "old")
        recent = await _run(runtime, bound, "new")
        assert (await _row_counts(history, expired))["events"] > 0
        await _age(history, expired, seconds=RETENTION_SECONDS + 1)

        deleted = await runtime.prune()

        assert deleted == 1
        assert all(count == 0 for count in (await _row_counts(history, expired)).values())
        assert await history.get_execution(recent) is not None
        with pytest.raises(KeyError):
            await runtime.get_execution_handle(expired)
        with pytest.raises(KeyError):
            await runtime.recover(bound, expired)
        await runtime.close()


@pytest.mark.asyncio
async def test_prune_treats_abandoned_running_execution_like_any_other(
    tmp_path: Path,
) -> None:
    async with SQLiteHistoryStore(tmp_path / "history.sqlite3") as history:
        runtime = LocalRuntime(
            history=history, journal_retention_seconds=RETENTION_SECONDS
        )
        bound = runtime.bind(Echo())
        await history.create_execution(
            execution_id="abandoned",
            request_id="request-abandoned",
            plan_id=bound.plan.plan_id,
            input=invocation_to_dict(UserMessage(content="x"), Context()),
            status="running",
        )
        token = await history.claim_execution(
            execution_id="abandoned", owner_id="dead-process", lease_ttl=0.01
        )
        assert token is not None
        await asyncio.sleep(0.05)
        await _age(history, "abandoned", seconds=RETENTION_SECONDS + 1)

        assert await runtime.prune() == 1

        assert all(
            count == 0 for count in (await _row_counts(history, "abandoned")).values()
        )
        await runtime.close()


@pytest.mark.asyncio
async def test_prune_keeps_running_execution_with_live_claim(tmp_path: Path) -> None:
    async with SQLiteHistoryStore(tmp_path / "history.sqlite3") as history:
        runtime = LocalRuntime(
            history=history, journal_retention_seconds=RETENTION_SECONDS
        )
        bound = runtime.bind(Echo())
        await history.create_execution(
            execution_id="owned",
            request_id="request-owned",
            plan_id=bound.plan.plan_id,
            input=invocation_to_dict(UserMessage(content="x"), Context()),
            status="running",
        )
        token = await history.claim_execution(
            execution_id="owned", owner_id="live-process", lease_ttl=60.0
        )
        assert token is not None
        await _age(history, "owned", seconds=RETENTION_SECONDS + 1)

        assert await runtime.prune() == 0

        assert await history.get_execution("owned") is not None
        assert (await _row_counts(history, "owned"))["execution_claims"] == 1
        await runtime.close()


@pytest.mark.asyncio
async def test_prune_skips_execution_claimed_after_listing(tmp_path: Path) -> None:
    async with SQLiteHistoryStore(tmp_path / "history.sqlite3") as history:
        runtime = LocalRuntime(
            history=history, journal_retention_seconds=RETENTION_SECONDS
        )
        bound = runtime.bind(Echo())
        for execution_id in ("contested", "uncontested"):
            await history.create_execution(
                execution_id=execution_id,
                request_id=f"request-{execution_id}",
                plan_id=bound.plan.plan_id,
                input=invocation_to_dict(UserMessage(content="x"), Context()),
                status="running",
            )
            await _age(history, execution_id, seconds=RETENTION_SECONDS + 1)
        listing = history.list_prunable_executions

        async def list_then_claim(**kwargs):
            batch = await listing(**kwargs)
            if "contested" in batch:
                assert await history.claim_execution(
                    execution_id="contested", owner_id="other-process", lease_ttl=60.0
                )
            return batch

        history.list_prunable_executions = list_then_claim  # type: ignore[method-assign]

        assert await runtime.prune() == 1

        assert await history.get_execution("contested") is not None
        assert await history.get_execution("uncontested") is None
        await runtime.close()


@pytest.mark.asyncio
async def test_prune_without_retention_keeps_everything(tmp_path: Path) -> None:
    async with SQLiteHistoryStore(tmp_path / "history.sqlite3") as history:
        runtime = LocalRuntime(history=history, journal_retention_seconds=None)
        bound = runtime.bind(Echo())
        execution_id = await _run(runtime, bound, "forever")
        await _age(history, execution_id, seconds=365 * 24 * 3600)

        assert await runtime.prune() == 0

        assert await history.get_execution(execution_id) is not None
        await runtime.close()


@pytest.mark.asyncio
async def test_prune_returns_freed_pages(tmp_path: Path) -> None:
    async with SQLiteHistoryStore(tmp_path / "history.sqlite3") as history:
        runtime = LocalRuntime(
            history=history, journal_retention_seconds=RETENTION_SECONDS
        )
        bound = runtime.bind(Echo())
        for index in range(20):
            execution_id = await _run(runtime, bound, f"bulk-{index}")
            await _age(history, execution_id, seconds=RETENTION_SECONDS + 1)

        assert await runtime.prune() == 20

        rows = await history._db().execute_fetchall("PRAGMA freelist_count")
        assert int(rows[0][0]) == 0
        await runtime.close()


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True, "7d"])
def test_journal_retention_rejects_non_positive_or_non_numeric(value: object) -> None:
    with pytest.raises(ValueError, match="journal_retention_seconds"):
        LocalRuntime(journal_retention_seconds=value)  # type: ignore[arg-type]
