"""Bounded execution-input validation and in-memory inbox state."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

from pygent.core import (
    ExecutionInput,
    ExecutionInputDelivery,
    JsonValue,
    freeze_json,
    thaw_json,
)
from pygent.core.execution import EXECUTION_INPUT_MAX_BYTES

MAX_EXECUTION_INPUT_BYTES = EXECUTION_INPUT_MAX_BYTES
MAX_PENDING_EXECUTION_INPUTS = 256
MAX_EXECUTION_INPUT_RECEIVE = 256


class ExecutionInputConsumerError(RuntimeError):
    """A kind is already owned by a different Module path."""


def prepare_execution_input(input_id: str, kind: str, value: JsonValue) -> JsonValue:
    for name, item in (("input_id", input_id), ("kind", kind)):
        if not isinstance(item, str) or not item:
            raise ValueError(f"{name} must be a non-empty string")
    frozen = freeze_json(value)
    encoded = json.dumps(
        thaw_json(frozen), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    if len(encoded) > MAX_EXECUTION_INPUT_BYTES:
        raise ValueError("execution input exceeds the 64 KiB limit")
    return frozen


def validate_receive(kinds: tuple[str, ...], limit: int, seal_if_empty: bool) -> None:
    if not isinstance(kinds, tuple) or not kinds or any(
        not isinstance(kind, str) or not kind for kind in kinds
    ):
        raise ValueError("kinds must be a non-empty tuple of non-empty strings")
    if len(kinds) != len(set(kinds)):
        raise ValueError("kinds must be unique")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 256:
        raise ValueError("limit must be between 1 and 256")
    if not isinstance(seal_if_empty, bool):
        raise TypeError("seal_if_empty must be a bool")


@dataclass(slots=True)
class MemoryExecutionInbox:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    inputs: list[ExecutionInput] = field(default_factory=list)
    by_id: dict[str, ExecutionInput] = field(default_factory=dict)
    consumers: dict[str, str] = field(default_factory=dict)
    cursors: dict[str, int] = field(default_factory=dict)
    receives: dict[tuple[str, int], tuple[ExecutionInput, ...]] = field(default_factory=dict)
    sealed: bool = False
    next_sequence: int = 0

    async def send(
        self, execution_id: str, *, input_id: str, kind: str, value: JsonValue
    ) -> ExecutionInputDelivery:
        frozen = prepare_execution_input(input_id, kind, value)
        async with self.lock:
            existing = self.by_id.get(input_id)
            if existing is not None:
                return ExecutionInputDelivery(
                    "duplicate", execution_id, input_id, existing.sequence
                )
            if self.sealed:
                return ExecutionInputDelivery("execution_finished", execution_id, input_id)
            pending = sum(
                item.sequence > self.cursors.get(item.kind, -1) for item in self.inputs
            )
            if pending >= MAX_PENDING_EXECUTION_INPUTS:
                raise OverflowError("execution input inbox is full")
            item = ExecutionInput(input_id, self.next_sequence, kind, frozen)
            self.next_sequence += 1
            self.inputs.append(item)
            self.by_id[input_id] = item
            return ExecutionInputDelivery("accepted", execution_id, input_id, item.sequence)

    async def receive(
        self,
        *,
        module_path: str,
        receive_index: int,
        kinds: tuple[str, ...],
        limit: int,
        seal_if_empty: bool,
    ) -> tuple[ExecutionInput, ...]:
        validate_receive(kinds, limit, seal_if_empty)
        async with self.lock:
            receipt = self.receives.get((module_path, receive_index))
            if receipt is not None:
                return receipt
            for kind in kinds:
                owner = self.consumers.get(kind)
                if owner is not None and owner != module_path:
                    raise ExecutionInputConsumerError(
                        f"execution input kind {kind!r} is owned by {owner!r}"
                    )
            for kind in kinds:
                self.consumers[kind] = module_path
            selected = tuple(
                item
                for item in self.inputs
                if item.kind in kinds
                and item.sequence > self.cursors.get(item.kind, -1)
            )[:limit]
            for item in selected:
                self.cursors[item.kind] = item.sequence
            if not selected and seal_if_empty:
                self.sealed = True
            self.receives[(module_path, receive_index)] = selected
            return selected

    async def seal(self) -> None:
        async with self.lock:
            self.sealed = True
