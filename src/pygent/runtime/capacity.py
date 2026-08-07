"""SQLite-backed deployment capacity coordination.

The tables in this module are deliberately independent from Runtime history and
Job storage.  A coordinator may therefore share a SQLite database file with a
``SQLiteHistoryStore`` without coupling either schema or transaction lifecycle.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import threading
import time
import uuid
import weakref
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

from pygent.core import CapacityPermit

from .api import CapacityPolicy, ExecutionAdmissionError, ExecutionCapacityPolicy

_SUBMISSION_LOCKS: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[str, asyncio.Lock]
] = weakref.WeakKeyDictionary()
_TRANSACTION_LOCKS: weakref.WeakValueDictionary[str, threading.Lock] = (
    weakref.WeakValueDictionary()
)
_TRANSACTION_LOCKS_GUARD = threading.Lock()


def _submission_lock(path: Path) -> asyncio.Lock:
    """Serialize ticket creation before work is handed to the thread pool.

    SQLite serializes transactions, but two ``to_thread`` calls may start in
    the reverse of their event-loop submission order.  One lock per event loop
    and resolved database path preserves that observable order across distinct
    coordinator instances in the same Worker process.  Across processes the
    order in which ``BEGIN IMMEDIATE`` succeeds remains the global authority.
    """

    loop = asyncio.get_running_loop()
    locks = _SUBMISSION_LOCKS.setdefault(loop, {})
    return locks.setdefault(str(path.resolve()), asyncio.Lock())


def _transaction_lock(path: Path) -> threading.Lock:
    """Return one in-process writer lock for every resolved database path."""

    key = str(path.resolve())
    with _TRANSACTION_LOCKS_GUARD:
        current = _TRANSACTION_LOCKS.get(key)
        if current is None:
            current = threading.Lock()
            _TRANSACTION_LOCKS[key] = current
        return current


def _policy_hash(value: object) -> str:
    if isinstance(value, ExecutionCapacityPolicy):
        payload: dict[str, object] = {
            "kind": "execution",
            "scope": value.scope.value,
            "max_live_executions": value.max_live_executions,
            "max_runnable_executions": value.max_runnable_executions,
            "max_queue_size": value.max_queue_size,
            "max_waiters": value.max_waiters,
            "max_child_depth": value.max_child_depth,
            "max_children_per_execution": value.max_children_per_execution,
            "max_external_wait_seconds": value.max_external_wait_seconds,
        }
    elif isinstance(value, CapacityPolicy):
        payload = {
            "kind": "resource",
            "scope": value.scope.value,
            "max_concurrency": value.max_concurrency,
            "max_queue_size": value.max_queue_size,
            "capacity_key": value.capacity_key,
        }
    else:  # pragma: no cover - internal construction invariant
        raise TypeError("unsupported capacity policy")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


class _SQLiteLeasePool:
    def __init__(
        self,
        coordinator: SQLiteCapacityCoordinator,
        *,
        owner_key: str,
        policy_hash: str,
        limit: int,
        max_queue: int,
        label: str,
    ) -> None:
        self._coordinator = coordinator
        self._owner_key = owner_key
        self._policy_hash = policy_hash
        self._limit = limit
        self._max_queue = max_queue
        self._label = label
        self._held: dict[asyncio.Task[Any], tuple[str, int, asyncio.Task[None]]] = {}

    async def acquire(self, *, resume: bool = False) -> CapacityPermit:
        del resume  # SQLite coordination is strict FIFO across START and RESUME.
        task = asyncio.current_task()
        if task is None:  # pragma: no cover - asyncio API invariant
            raise RuntimeError("capacity acquire requires an asyncio Task")
        if task in self._held:
            raise RuntimeError(f"{self._label} capacity is already held")
        waiter_id = uuid.uuid4().hex
        queued = False
        try:
            async with _submission_lock(self._coordinator.path):
                outcome = await self._run_admission(
                    self._coordinator._join_or_acquire,
                    self._owner_key,
                    self._policy_hash,
                    self._limit,
                    self._max_queue,
                    waiter_id,
                )
            if outcome is not None:
                self._hold(task, outcome)
                return CapacityPermit(self._owner_key, outcome[1])
            queued = True
            while True:
                await asyncio.sleep(self._coordinator.poll_interval)
                outcome = await self._run_admission(
                    self._coordinator._poll_waiter,
                    self._owner_key,
                    self._policy_hash,
                    self._limit,
                    self._max_queue,
                    waiter_id,
                )
                if outcome is not None:
                    queued = False
                    self._hold(task, outcome)
                    return CapacityPermit(self._owner_key, outcome[1])
        except BaseException:
            if queued:
                await asyncio.shield(
                    asyncio.to_thread(
                        self._coordinator._delete_waiter,
                        self._owner_key,
                        waiter_id,
                    )
                )
            raise

    async def _run_admission(
        self,
        operation: Callable[..., tuple[str, int] | None],
        *args: object,
    ) -> tuple[str, int] | None:
        """Finish a SQLite admission transaction before propagating cancellation."""

        admission = asyncio.create_task(asyncio.to_thread(operation, *args))
        try:
            return await asyncio.shield(admission)
        except asyncio.CancelledError:
            await asyncio.wait((admission,))
            if admission.cancelled() or admission.exception() is not None:
                # A failed transaction rolls itself back, so it has no row to clean.
                raise
            outcome = admission.result()
            if outcome is None:
                await asyncio.shield(
                    asyncio.to_thread(
                        self._coordinator._delete_waiter,
                        self._owner_key,
                        cast(str, args[-1]),
                    )
                )
            else:
                await asyncio.shield(
                    asyncio.to_thread(
                        self._coordinator._delete_lease,
                        self._owner_key,
                        outcome[0],
                        outcome[1],
                    )
                )
            raise

    def release(self) -> None:
        task = asyncio.current_task()
        if task is None:  # pragma: no cover - asyncio API invariant
            raise RuntimeError("capacity release requires an asyncio Task")
        held = self._held.pop(task, None)
        if held is None:
            raise RuntimeError(f"{self._label} capacity is not held")
        lease_id, fence, heartbeat = held
        heartbeat.cancel()
        self._coordinator._schedule_release(self._owner_key, lease_id, fence)

    async def release_async(self) -> None:
        task = asyncio.current_task()
        if task is None:  # pragma: no cover - asyncio API invariant
            raise RuntimeError("capacity release requires an asyncio Task")
        held = self._held.pop(task, None)
        if held is None:
            raise RuntimeError(f"{self._label} capacity is not held")
        lease_id, fence, heartbeat = held
        heartbeat.cancel()
        await asyncio.to_thread(
            self._coordinator._delete_lease,
            self._owner_key,
            lease_id,
            fence,
        )

    def _hold(self, task: asyncio.Task[Any], lease: tuple[str, int]) -> None:
        lease_id, fence = lease
        heartbeat = asyncio.create_task(
            self._heartbeat(task, lease_id, fence),
            name=f"pygent-capacity-heartbeat-{fence}",
        )
        self._held[task] = (lease_id, fence, heartbeat)

    async def _heartbeat(
        self,
        owner_task: asyncio.Task[Any],
        lease_id: str,
        fence: int,
    ) -> None:
        try:
            while True:
                await asyncio.sleep(self._coordinator.lease_ttl / 3)
                renewed = await asyncio.to_thread(
                    self._coordinator._renew_lease,
                    self._owner_key,
                    lease_id,
                    fence,
                )
                if not renewed:
                    # A reclaimed lease is fenced.  Stop the stale execution
                    # instead of allowing it to continue without ownership.
                    owner_task.cancel()
                    return
        except asyncio.CancelledError:
            return

    async def close(self, *, release_leases: bool) -> None:
        held = tuple(self._held.values())
        self._held.clear()
        for _, _, heartbeat in held:
            heartbeat.cancel()
        if release_leases:
            await asyncio.gather(
                *(
                    asyncio.to_thread(
                        self._coordinator._delete_lease,
                        self._owner_key,
                        lease_id,
                        fence,
                    )
                    for lease_id, fence, _ in held
                )
            )


class _SQLiteRunnableGate:
    def __init__(self, pool: _SQLiteLeasePool) -> None:
        self._pool = pool

    async def acquire(self, *, resume: bool = False) -> None:
        await self._pool.acquire(resume=resume)

    def release(self) -> None:
        self._pool.release()


class _SQLiteExecutionCapacityState:
    def __init__(
        self,
        coordinator: SQLiteCapacityCoordinator,
        name: str,
        policy: ExecutionCapacityPolicy,
    ) -> None:
        fingerprint = _policy_hash(policy)
        self._live = coordinator._pool(
            owner_key=f"execution:{name}:live",
            policy_hash=fingerprint,
            limit=policy.max_live_executions,
            max_queue=policy.max_queue_size,
            label=f"deployment execution {name!r} START",
        )
        runnable_pool = coordinator._pool(
            owner_key=f"execution:{name}:runnable",
            policy_hash=fingerprint,
            limit=policy.max_runnable_executions,
            max_queue=policy.max_live_executions,
            label=f"deployment execution {name!r} runnable",
        )
        self.runnable = _SQLiteRunnableGate(runnable_pool)
        self._waiters = coordinator._pool(
            owner_key=f"execution:{name}:waiters",
            policy_hash=fingerprint,
            limit=policy.max_waiters,
            max_queue=0,
            label=f"deployment execution {name!r} external waiters",
        )

    async def admit(self) -> None:
        await self._live.acquire()

    async def release_live(self) -> None:
        await self._live.release_async()

    @asynccontextmanager
    async def waiter_slot(self) -> AsyncIterator[CapacityPermit]:
        permit = await self._waiters.acquire()
        try:
            yield permit
        finally:
            await self._waiters.release_async()


class _SQLiteResourceGate:
    def __init__(self, pool: _SQLiteLeasePool, public_owner_key: str) -> None:
        self._pool = pool
        self._public_owner_key = public_owner_key

    @asynccontextmanager
    async def permit(self) -> AsyncIterator[CapacityPermit]:
        permit = await self._pool.acquire()
        try:
            yield CapacityPermit(
                owner_key=self._public_owner_key,
                fencing_token=permit.fencing_token,
            )
        finally:
            await self._pool.release_async()


class SQLiteCapacityCoordinator:
    """Cross-process deployment capacity owner backed by SQLite leases.

    Acquisition is FIFO and work-conserving.  Every permit has a monotonically
    increasing fencing token and an expiring lease renewed by its owning event
    loop.  If a process crashes, another coordinator reclaims the permit after
    ``lease_ttl``; if renewal loses its fence, the stale owner Task is cancelled.
    The token becomes a strong side-effect fence only when the protected external
    resource atomically rejects stale tokens at commit time.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        lease_ttl: float = 30.0,
        poll_interval: float = 0.02,
    ) -> None:
        if lease_ttl <= 0:
            raise ValueError("lease_ttl must be greater than zero")
        if poll_interval <= 0 or poll_interval >= lease_ttl:
            raise ValueError("poll_interval must be positive and less than lease_ttl")
        self.path = Path(path)
        self._transaction_lock = _transaction_lock(self.path)
        self.lease_ttl = float(lease_ttl)
        self.poll_interval = float(poll_interval)
        self._execution_states: dict[
            str, tuple[ExecutionCapacityPolicy, _SQLiteExecutionCapacityState]
        ] = {}
        self._resource_gates: dict[
            tuple[str, str], tuple[CapacityPolicy, _SQLiteResourceGate]
        ] = {}
        self._pools: dict[str, _SQLiteLeasePool] = {}
        self._background: set[asyncio.Task[None]] = set()
        self._initialize()

    def execution_state(
        self, name: str, policy: ExecutionCapacityPolicy
    ) -> _SQLiteExecutionCapacityState:
        current = self._execution_states.get(name)
        if current is None:
            state = _SQLiteExecutionCapacityState(self, name, policy)
            self._execution_states[name] = (policy, state)
            return state
        current_policy, state = current
        if current_policy != policy:
            raise ValueError(
                f"deployment Binding {name!r} has conflicting execution capacity"
            )
        return state

    def resource_gate(
        self,
        kind: str,
        capacity_key: str,
        policy: CapacityPolicy,
    ) -> _SQLiteResourceGate:
        key = (kind, capacity_key)
        current = self._resource_gates.get(key)
        if current is None:
            assert policy.max_concurrency is not None
            gate = _SQLiteResourceGate(
                self._pool(
                    owner_key=f"resource:{kind}:{capacity_key}",
                    policy_hash=_policy_hash(policy),
                    limit=policy.max_concurrency,
                    max_queue=policy.max_queue_size or 0,
                    label=f"deployment {kind} capacity {capacity_key!r}",
                ),
                f"{kind}:{capacity_key}",
            )
            self._resource_gates[key] = (policy, gate)
            return gate
        current_policy, gate = current
        if current_policy != policy:
            raise ValueError(
                f"deployment {kind} capacity {capacity_key!r} has conflicting policies"
            )
        return gate

    async def close(self, *, release_leases: bool = True) -> None:
        """Stop renewal; graceful close also deletes this process' leases.

        ``release_leases=False`` models abrupt process loss and is primarily
        useful for failure testing.  Other coordinators reclaim those fenced
        leases only after their TTL expires.
        """

        await asyncio.gather(
            *(
                pool.close(release_leases=release_leases)
                for pool in self._pools.values()
            )
        )
        if self._background:
            await asyncio.gather(*tuple(self._background), return_exceptions=True)

    def _pool(
        self,
        *,
        owner_key: str,
        policy_hash: str,
        limit: int,
        max_queue: int,
        label: str,
    ) -> _SQLiteLeasePool:
        self._ensure_policy(owner_key, policy_hash)
        current = self._pools.get(owner_key)
        if current is None:
            current = _SQLiteLeasePool(
                self,
                owner_key=owner_key,
                policy_hash=policy_hash,
                limit=limit,
                max_queue=max_queue,
                label=label,
            )
            self._pools[owner_key] = current
        return current

    def _ensure_policy(self, owner_key: str, policy_hash: str) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            self._prepare(connection, owner_key, policy_hash)

        self._transaction(operation)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            isolation_level=None,
            timeout=5.0,
        )
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._transaction_lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS pygent_capacity_policies (
                    owner_key TEXT PRIMARY KEY,
                    policy_hash TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pygent_capacity_sequence (
                    fence INTEGER PRIMARY KEY AUTOINCREMENT
                );
                CREATE TABLE IF NOT EXISTS pygent_capacity_waiters (
                    waiter_id TEXT PRIMARY KEY,
                    owner_key TEXT NOT NULL,
                    ticket INTEGER NOT NULL,
                    expires_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_pygent_capacity_waiters_owner
                    ON pygent_capacity_waiters(owner_key, ticket);
                CREATE TABLE IF NOT EXISTS pygent_capacity_leases (
                    fence INTEGER PRIMARY KEY AUTOINCREMENT,
                    lease_id TEXT NOT NULL UNIQUE,
                    owner_key TEXT NOT NULL,
                    expires_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_pygent_capacity_leases_owner
                    ON pygent_capacity_leases(owner_key, expires_at);
                """
            )

    def _transaction(self, operation: Any) -> Any:
        with self._transaction_lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                result = operation(connection)
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            connection.execute("COMMIT")
            return result

    def _prepare(
        self,
        connection: sqlite3.Connection,
        owner_key: str,
        policy_hash: str,
    ) -> float:
        now = time.time()
        connection.execute(
            "DELETE FROM pygent_capacity_leases WHERE expires_at <= ?",
            (now,),
        )
        connection.execute(
            "DELETE FROM pygent_capacity_waiters WHERE expires_at <= ?",
            (now,),
        )
        connection.execute(
            "INSERT OR IGNORE INTO pygent_capacity_policies(owner_key, policy_hash) "
            "VALUES (?, ?)",
            (owner_key, policy_hash),
        )
        row = connection.execute(
            "SELECT policy_hash FROM pygent_capacity_policies WHERE owner_key = ?",
            (owner_key,),
        ).fetchone()
        if row is None or row[0] != policy_hash:
            raise ValueError(f"capacity owner {owner_key!r} has conflicting policies")
        return now

    def _insert_lease(
        self,
        connection: sqlite3.Connection,
        owner_key: str,
        now: float,
    ) -> tuple[str, int]:
        lease_id = uuid.uuid4().hex
        cursor = connection.execute(
            "INSERT INTO pygent_capacity_leases(lease_id, owner_key, expires_at) "
            "VALUES (?, ?, ?)",
            (lease_id, owner_key, now + self.lease_ttl),
        )
        if cursor.lastrowid is None:  # pragma: no cover - SQLite invariant
            raise RuntimeError("SQLite did not allocate a fencing token")
        return lease_id, int(cursor.lastrowid)

    def _join_or_acquire(
        self,
        owner_key: str,
        policy_hash: str,
        limit: int,
        max_queue: int,
        waiter_id: str,
    ) -> tuple[str, int] | None:
        def operation(connection: sqlite3.Connection) -> tuple[str, int] | None:
            now = self._prepare(connection, owner_key, policy_hash)
            leases = int(
                connection.execute(
                    "SELECT COUNT(*) FROM pygent_capacity_leases WHERE owner_key = ?",
                    (owner_key,),
                ).fetchone()[0]
            )
            waiting = int(
                connection.execute(
                    "SELECT COUNT(*) FROM pygent_capacity_waiters WHERE owner_key = ?",
                    (owner_key,),
                ).fetchone()[0]
            )
            if leases < limit and waiting == 0:
                return self._insert_lease(connection, owner_key, now)
            if waiting >= max_queue:
                raise ExecutionAdmissionError(f"{owner_key} capacity queue is full")
            cursor = connection.execute(
                "INSERT INTO pygent_capacity_sequence DEFAULT VALUES"
            )
            if cursor.lastrowid is None:  # pragma: no cover - SQLite invariant
                raise RuntimeError("SQLite did not allocate a waiter ticket")
            ticket = int(cursor.lastrowid)
            connection.execute(
                "INSERT INTO pygent_capacity_waiters"
                "(waiter_id, owner_key, ticket, expires_at) VALUES (?, ?, ?, ?)",
                (waiter_id, owner_key, ticket, now + self.lease_ttl * 3),
            )
            return None

        return self._transaction(operation)

    def _poll_waiter(
        self,
        owner_key: str,
        policy_hash: str,
        limit: int,
        max_queue: int,
        waiter_id: str,
    ) -> tuple[str, int] | None:
        def operation(connection: sqlite3.Connection) -> tuple[str, int] | None:
            now = self._prepare(connection, owner_key, policy_hash)
            updated = connection.execute(
                "UPDATE pygent_capacity_waiters SET expires_at = ? "
                "WHERE owner_key = ? AND waiter_id = ?",
                (now + self.lease_ttl * 3, owner_key, waiter_id),
            )
            if updated.rowcount == 0:
                waiting = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM pygent_capacity_waiters "
                        "WHERE owner_key = ?",
                        (owner_key,),
                    ).fetchone()[0]
                )
                if waiting >= max_queue:
                    raise ExecutionAdmissionError(
                        f"{owner_key} capacity queue is full after waiter expiry"
                    )
                cursor = connection.execute(
                    "INSERT INTO pygent_capacity_sequence DEFAULT VALUES"
                )
                if cursor.lastrowid is None:  # pragma: no cover - SQLite invariant
                    raise RuntimeError("SQLite did not allocate a waiter ticket")
                connection.execute(
                    "INSERT INTO pygent_capacity_waiters"
                    "(waiter_id, owner_key, ticket, expires_at) VALUES (?, ?, ?, ?)",
                    (
                        waiter_id,
                        owner_key,
                        int(cursor.lastrowid),
                        now + self.lease_ttl * 3,
                    ),
                )
            first = connection.execute(
                "SELECT waiter_id FROM pygent_capacity_waiters "
                "WHERE owner_key = ? ORDER BY ticket LIMIT 1",
                (owner_key,),
            ).fetchone()
            leases = int(
                connection.execute(
                    "SELECT COUNT(*) FROM pygent_capacity_leases WHERE owner_key = ?",
                    (owner_key,),
                ).fetchone()[0]
            )
            if first is None or first[0] != waiter_id or leases >= limit:
                return None
            lease = self._insert_lease(connection, owner_key, now)
            connection.execute(
                "DELETE FROM pygent_capacity_waiters "
                "WHERE owner_key = ? AND waiter_id = ?",
                (owner_key, waiter_id),
            )
            return lease

        return self._transaction(operation)

    def _delete_waiter(self, owner_key: str, waiter_id: str) -> None:
        with self._transaction_lock, self._connect() as connection:
            connection.execute(
                "DELETE FROM pygent_capacity_waiters "
                "WHERE owner_key = ? AND waiter_id = ?",
                (owner_key, waiter_id),
            )

    def _renew_lease(self, owner_key: str, lease_id: str, fence: int) -> bool:
        with self._transaction_lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE pygent_capacity_leases SET expires_at = ? "
                "WHERE owner_key = ? AND lease_id = ? AND fence = ?",
                (time.time() + self.lease_ttl, owner_key, lease_id, fence),
            )
            return cursor.rowcount == 1

    def _delete_lease(self, owner_key: str, lease_id: str, fence: int) -> None:
        with self._transaction_lock, self._connect() as connection:
            connection.execute(
                "DELETE FROM pygent_capacity_leases "
                "WHERE owner_key = ? AND lease_id = ? AND fence = ?",
                (owner_key, lease_id, fence),
            )

    def _schedule_release(self, owner_key: str, lease_id: str, fence: int) -> None:
        async def release() -> None:
            await asyncio.to_thread(
                self._delete_lease,
                owner_key,
                lease_id,
                fence,
            )

        task = asyncio.create_task(release())
        self._background.add(task)
        task.add_done_callback(self._background.discard)


__all__ = ["SQLiteCapacityCoordinator"]
