"""Durable history records, errors, serialization, and write guard."""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import sqlite3
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from functools import wraps
from typing import Any, Concatenate, Literal, ParamSpec, Protocol, TypeVar, cast

import aiosqlite

from pygent.core import FrozenJsonObject, JsonValue, freeze_json

_P = ParamSpec("_P")
_R = TypeVar("_R")


class _SerializedWriter(Protocol):
    _write_lock: asyncio.Lock

    def _db(self) -> aiosqlite.Connection: ...


_WriterT = TypeVar("_WriterT", bound=_SerializedWriter)


class HistoryStoreError(RuntimeError):
    """Base class for durable history failures."""


class HistoryConflictError(HistoryStoreError):
    """Raised when a stable identity is committed with different content."""


class NonDeterministicReplayError(HistoryStoreError):
    """Raised when replay reaches a boundary different from committed history."""


@dataclass(frozen=True, slots=True)
class StoredExecution:
    execution_id: str
    request_id: str
    status: str
    plan_id: str
    input: JsonValue
    output: JsonValue | None
    error: JsonValue | None
    attempt: int
    binding_id: str = ""
    identity: str = ""
    idempotency_key: str | None = None
    model_calls: JsonValue | None = None
    model_admission_id: str | None = None
    model_admission_digest: str | None = None
    model_admission_status: str = "none"
    trace_id: str = ""
    phase: str = "submitting"
    attempt_id: str | None = None
    terminal_sequence: int | None = None
    submitted_at_unix_ns: int = 0
    updated_at_unix_ns: int = 0


@dataclass(frozen=True, slots=True)
class ExecutionState:
    status: str
    phase: str
    attempt_id: str | None
    terminal_sequence: int | None
    trace_id: str
    submitted_at_unix_ns: int
    updated_at_unix_ns: int
    error: JsonValue | None
    owner_active: bool
    last_sequence: int


@dataclass(frozen=True, slots=True)
class StoredTask:
    task_id: str
    kind: Literal["job", "tool_task"]
    status: str
    request: JsonValue
    result: JsonValue | None
    error: JsonValue | None


@dataclass(frozen=True, slots=True)
class StoredJob:
    job_id: str
    task_id: str
    logical_key: str
    status: str
    binding_id: str
    plan_id: str
    resource_key: str | None
    required_capabilities: tuple[str, ...]
    request: JsonValue
    result: JsonValue | None
    error: JsonValue | None
    attempt: int


@dataclass(frozen=True, slots=True)
class StoredEffect:
    execution_id: str
    module_path: str
    call_index: int
    effect_type: str
    request_digest: str
    status: Literal["started", "completed", "unknown"]
    spec: JsonValue | None
    result: JsonValue | None


@dataclass(frozen=True, slots=True)
class _PreparedJson:
    frozen: JsonValue
    payload: str


def _prepare_json(value: object) -> _PreparedJson:
    """Freeze once and retain the exact canonical SQLite representation."""

    frozen = value if isinstance(value, FrozenJsonObject) else freeze_json(value)
    return _PreparedJson(frozen=frozen, payload=_json_frozen(frozen))


def _json(value: object) -> str:
    return _prepare_json(value).payload


def _json_frozen(value: JsonValue) -> str:
    """Serialize an already validated immutable JSON value without re-freezing it."""

    return _CANONICAL_ENCODER.encode(value)


def _json_frozen_object(value: Mapping[str, JsonValue]) -> str:
    """Serialize a validated object whose child values are already immutable JSON."""

    return _CANONICAL_ENCODER.encode(value)


def _json_container(value: object) -> dict[str, object]:
    """Let the native JSON encoder traverse children instead of pre-thawing them."""
    if type(value) is FrozenJsonObject:
        return dict(value._items)
    if isinstance(value, FrozenJsonObject):
        return value.to_dict()
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"unsupported JSON container: {type(value).__name__}")


# JSONEncoder keeps traversal/cycle state local to each encode call. Reuse only
# its fixed configuration; never retain execution values or serialized payloads.
_CANONICAL_ENCODER = json.JSONEncoder(
    default=_json_container, sort_keys=True, separators=(",", ":")
)


def _load(value: str | None) -> JsonValue | None:
    if value is None:
        return None
    return freeze_json(json.loads(value))


def effect_digest(request: object) -> str:
    """Return a stable digest for a provider-neutral effect request."""

    return _prepared_json_digest(_prepare_json(request))


def _prepared_json_digest(prepared: _PreparedJson) -> str:
    return hashlib.sha256(prepared.payload.encode("utf-8")).hexdigest()


def _serialized_access(
    method: Callable[Concatenate[_WriterT, _P], Awaitable[_R]],
) -> Callable[Concatenate[_WriterT, _P], Awaitable[_R]]:
    """Serialize access to the Store connection so readers cannot observe uncommitted writes."""

    @wraps(method)
    async def wrapped(self: _WriterT, /, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        async with self._write_lock:
            return await method(self, *args, **kwargs)

    return wrapped


# SQLITE_BUSY/SQLITE_LOCKED primary codes; extended variants (517
# SQLITE_BUSY_SNAPSHOT, 261 SQLITE_BUSY_RECOVERY, 518 SQLITE_LOCKED_SHAREDCACHE)
# collapse onto these through & 0xFF. Other OperationalErrors are not transient.
_LOCK_RETRY_PRIMARY_CODES = frozenset({5, 6})
_LOCK_RETRY_BASE_DELAY = 0.05
_LOCK_RETRY_MAX_DELAY = 1.0
_LOCK_RETRY_MAX_ATTEMPTS = 5
_LOCK_RETRY_DEADLINE_SECONDS = 60.0


def _lock_primary_code(exc: BaseException) -> int | None:
    code = getattr(exc, "sqlite_errorcode", None)
    if not isinstance(code, int):
        return None
    primary = code & 0xFF
    return primary if primary in _LOCK_RETRY_PRIMARY_CODES else None


async def _retry_store_lock(
    store: Any, operation: Callable[[], Awaitable[_R]]
) -> _R:
    """Run one transaction attempt, retrying while SQLite reports lock contention.

    ``operation`` must be a full transaction boundary that rolls back before
    raising, so a retry re-enters cleanly and revalidates fencing inside its
    own transaction. Sleeping happens outside the store serialization lock.
    """

    deadline = time.monotonic() + _LOCK_RETRY_DEADLINE_SECONDS
    attempt = 0
    while True:
        try:
            return await operation()
        except sqlite3.OperationalError as exc:
            if _lock_primary_code(exc) is None:
                raise
            attempt += 1
            if attempt >= _LOCK_RETRY_MAX_ATTEMPTS or time.monotonic() >= deadline:
                enriched = sqlite3.OperationalError(
                    f"{getattr(store, 'path', 'history store')} write still "
                    f"lock-blocked after {attempt} attempt(s): "
                    f"{getattr(exc, 'sqlite_errorname', None) or exc}"
                )
                code = getattr(exc, "sqlite_errorcode", None)
                if isinstance(code, int):
                    enriched.sqlite_errorcode = code
                name = getattr(exc, "sqlite_errorname", None)
                if isinstance(name, str):
                    enriched.sqlite_errorname = name
                raise enriched from exc
            await asyncio.sleep(
                min(_LOCK_RETRY_BASE_DELAY * 2 ** (attempt - 1), _LOCK_RETRY_MAX_DELAY)
                * (1.0 + random.random())
            )


_LockRetryMethodT = TypeVar(
    "_LockRetryMethodT", bound=Callable[..., Awaitable[Any]]
)


def _lock_retry(method: _LockRetryMethodT) -> _LockRetryMethodT:
    """Retry a whole transaction method while SQLite reports lock contention.

    Must be the outermost decorator: retries must sleep without holding the
    store serialization lock, and each attempt re-acquires it.
    """

    @wraps(method)
    async def wrapped(self: Any, /, *args: Any, **kwargs: Any) -> Any:
        return await _retry_store_lock(self, lambda: method(self, *args, **kwargs))

    return cast(_LockRetryMethodT, wrapped)


def _execution_write(
    method: Callable[Concatenate[_WriterT, _P], Awaitable[_R]],
) -> Callable[Concatenate[_WriterT, _P], Awaitable[_R]]:
    """Serialize an execution mutation with its authority check in one transaction."""

    @wraps(method)
    async def wrapped(self: _WriterT, /, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        from ._history_ownership import validate_writers, write_authority

        execution_id = kwargs.get("execution_id", args[0] if args else None)
        if not isinstance(execution_id, str):
            raise TypeError("execution write requires an execution identity")
        authority = write_authority(execution_id)

        async def attempt() -> _R:
            async with self._write_lock:
                db = self._db()
                await db.execute("BEGIN IMMEDIATE")
                try:
                    await validate_writers(db, (authority,))
                    result = await method(self, *args, **kwargs)
                    await db.commit()
                    return result
                except BaseException:
                    await db.rollback()
                    raise

        return await _retry_store_lock(self, attempt)

    return wrapped
