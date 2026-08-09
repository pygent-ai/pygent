"""Durable history records, errors, serialization, and write guard."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import wraps
from typing import Concatenate, Literal, ParamSpec, Protocol, TypeVar

from pygent.core import JsonValue, freeze_json, thaw_json

_P = ParamSpec("_P")
_R = TypeVar("_R")


class _SerializedWriter(Protocol):
    _write_lock: asyncio.Lock


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


def _json(value: object) -> str:
    frozen = freeze_json(value)
    return json.dumps(thaw_json(frozen), sort_keys=True, separators=(",", ":"))


def _load(value: str | None) -> JsonValue | None:
    if value is None:
        return None
    return freeze_json(json.loads(value))


def effect_digest(request: object) -> str:
    """Return a stable digest for a provider-neutral effect request."""

    return hashlib.sha256(_json(request).encode("utf-8")).hexdigest()


def _serialized_write(
    method: Callable[Concatenate[_WriterT, _P], Awaitable[_R]],
) -> Callable[Concatenate[_WriterT, _P], Awaitable[_R]]:
    """Serialize transactions on the Store's single writer connection."""

    @wraps(method)
    async def wrapped(
        self: _WriterT, /, *args: _P.args, **kwargs: _P.kwargs
    ) -> _R:
        async with self._write_lock:
            return await method(self, *args, **kwargs)

    return wrapped
