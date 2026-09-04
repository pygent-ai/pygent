"""Immutable JSON-compatible values used by public Pygent contracts."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias, cast

JsonScalar: TypeAlias = str | int | float | bool | None

# Bound recursive nesting; payload byte size and node count are unrestricted.
MAX_JSON_DEPTH = 64


class JsonValueError(TypeError):
    """Raised when a public value cannot be represented as strict JSON."""


@dataclass(slots=True)
class _FreezeState:
    active_container_ids: set[int]

    def visit(self, *, depth: int) -> None:
        if depth > MAX_JSON_DEPTH:
            raise JsonValueError(
                f"JSON value exceeds maximum depth of {MAX_JSON_DEPTH}"
            )


@dataclass(frozen=True, slots=True)
class FrozenJsonObject(Mapping[str, "JsonValue"]):
    """Insertion-ordered, immutable JSON object."""

    _items: tuple[tuple[str, JsonValue], ...] = ()

    def __post_init__(self) -> None:
        state = _FreezeState(set())
        state.visit(depth=0)
        object.__setattr__(
            self,
            "_items",
            _freeze_pairs(self._items, state=state, depth=1),
        )

    def __getitem__(self, key: str) -> JsonValue:
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def to_dict(self) -> dict[str, object]:
        """Return a mutable built-in representation suitable for JSON encoders."""

        return {key: thaw_json(value) for key, value in self._items}


JsonValue: TypeAlias = JsonScalar | tuple["JsonValue", ...] | FrozenJsonObject
JsonObjectInput: TypeAlias = (
    Mapping[str, object] | Iterable[tuple[str, object]] | FrozenJsonObject
)


def freeze_json(value: object) -> JsonValue:
    """Validate and recursively freeze one strict JSON-compatible value."""

    return _freeze(value, state=_FreezeState(set()), depth=0)


def freeze_json_object(value: JsonObjectInput = ()) -> FrozenJsonObject:
    """Validate and freeze a mapping or iterable of key/value pairs."""

    state = _FreezeState(set())
    state.visit(depth=0)
    container_id = id(value)
    state.active_container_ids.add(container_id)
    try:
        pairs = cast(
            Iterable[tuple[str, object]],
            value.items() if isinstance(value, Mapping) else value,
        )
        items = _freeze_pairs(pairs, state=state, depth=1)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, JsonValueError):
            raise
        raise JsonValueError("JSON objects must contain key/value pairs") from exc
    finally:
        state.active_container_ids.discard(container_id)

    # The private constructor avoids validating the already-frozen tree a
    # second time.
    frozen = object.__new__(FrozenJsonObject)
    object.__setattr__(frozen, "_items", items)
    return frozen


def _freeze_json_object_with_default(
    value: JsonObjectInput,
    key: str,
    default: object,
) -> FrozenJsonObject:
    state = _FreezeState(set())
    state.visit(depth=0)
    container_id = id(value)
    state.active_container_ids.add(container_id)
    try:
        pairs = cast(
            Iterable[tuple[str, object]],
            value.items() if isinstance(value, Mapping) else value,
        )
        frozen_items: list[tuple[str, JsonValue]] = []
        seen: set[str] = set()
        for pair in pairs:
            candidate, item = pair
            if not isinstance(candidate, str):
                raise JsonValueError("JSON object keys must be strings")
            if candidate in seen:
                raise JsonValueError(f"duplicate JSON object key: {candidate!r}")
            seen.add(candidate)
            frozen_items.append(
                (candidate, _freeze(item, state=state, depth=1))
            )
        if key not in seen:
            frozen_items.append((key, _freeze(default, state=state, depth=1)))
    except (TypeError, ValueError) as exc:
        if isinstance(exc, JsonValueError):
            raise
        raise JsonValueError("JSON objects must contain key/value pairs") from exc
    finally:
        state.active_container_ids.discard(container_id)

    frozen = object.__new__(FrozenJsonObject)
    object.__setattr__(frozen, "_items", tuple(frozen_items))
    return frozen


def _patch_frozen_json_object(
    value: FrozenJsonObject,
    updates: Mapping[str, object],
    *,
    overwrite: bool,
) -> FrozenJsonObject:
    """Patch one validated root while reusing its immutable child projections."""

    if not updates:
        return value
    update_items = tuple(updates.items())
    for key, _ in update_items:
        if not isinstance(key, str):
            raise JsonValueError("JSON object keys must be strings")
    update_by_key = dict(update_items)
    existing_keys = {key for key, _ in value._items}
    if not overwrite and all(key in existing_keys for key, _ in update_items):
        return value

    state = _FreezeState(set())
    state.visit(depth=0)
    patched_items: list[tuple[str, JsonValue]] = []
    for key, existing in value._items:
        if overwrite and key in update_by_key:
            patched = _freeze(update_by_key[key], state=state, depth=1)
        else:
            _validate_frozen_depth(existing, state=state)
            patched = existing
        patched_items.append((key, patched))
    for key, update in update_items:
        if key not in existing_keys:
            patched_items.append((key, _freeze(update, state=state, depth=1)))

    patched = object.__new__(FrozenJsonObject)
    object.__setattr__(patched, "_items", tuple(patched_items))
    return patched


def _validate_frozen_depth(value: JsonValue, *, state: _FreezeState) -> None:
    pending = [(value, 1)]
    while pending:
        current, depth = pending.pop()
        state.visit(depth=depth)
        if isinstance(current, FrozenJsonObject):
            pending.extend(
                (item, depth + 1) for _, item in reversed(current._items)
            )
        elif isinstance(current, tuple):
            pending.extend((item, depth + 1) for item in reversed(current))


def _freeze(value: object, *, state: _FreezeState, depth: int) -> JsonValue:
    state.visit(depth=depth)

    if isinstance(value, Enum):
        return _freeze(value.value, state=state, depth=depth)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise JsonValueError("JSON numbers must be finite")
        return value
    if isinstance(value, FrozenJsonObject):
        # Existing frozen values still have to participate in the current
        # depth and cycle checks.  Treating them as trusted leaves
        # lets repeated shallow wrapping create a value that construction
        # accepts but wire thawing cannot traverse safely.
        return _freeze_mapping(
            cast(Mapping[object, object], value), state=state, depth=depth
        )
    if isinstance(value, Mapping):
        return _freeze_mapping(value, state=state, depth=depth)
    if isinstance(value, (list, tuple)):
        return _freeze_sequence(value, state=state, depth=depth)
    raise JsonValueError(f"unsupported JSON value type: {type(value).__name__}")


def _freeze_mapping(
    value: Mapping[object, object], *, state: _FreezeState, depth: int
) -> FrozenJsonObject:
    container_id = id(value)
    if container_id in state.active_container_ids:
        raise JsonValueError("cyclic JSON value")
    state.active_container_ids.add(container_id)
    try:
        items = _freeze_pairs(value.items(), state=state, depth=depth + 1)
    finally:
        state.active_container_ids.remove(container_id)
    frozen = object.__new__(FrozenJsonObject)
    object.__setattr__(frozen, "_items", items)
    return frozen


def _freeze_sequence(
    value: list[object] | tuple[object, ...], *, state: _FreezeState, depth: int
) -> tuple[JsonValue, ...]:
    container_id = id(value)
    if container_id in state.active_container_ids:
        raise JsonValueError("cyclic JSON value")
    state.active_container_ids.add(container_id)
    try:
        return tuple(_freeze(item, state=state, depth=depth + 1) for item in value)
    finally:
        state.active_container_ids.remove(container_id)


def _freeze_pairs(
    pairs: Iterable[tuple[object, object]], *, state: _FreezeState, depth: int
) -> tuple[tuple[str, JsonValue], ...]:
    frozen_items: list[tuple[str, JsonValue]] = []
    seen: set[str] = set()
    try:
        for pair in pairs:
            key, value = pair
            if not isinstance(key, str):
                raise JsonValueError("JSON object keys must be strings")
            if key in seen:
                raise JsonValueError(f"duplicate JSON object key: {key!r}")
            seen.add(key)
            frozen_items.append((key, _freeze(value, state=state, depth=depth)))
    except (TypeError, ValueError) as exc:
        if isinstance(exc, JsonValueError):
            raise
        raise JsonValueError("JSON objects must contain key/value pairs") from exc
    return tuple(frozen_items)


def thaw_json(value: JsonValue | Mapping[str, object]) -> object:
    """Convert an immutable JSON value to built-in dict/list containers."""

    if isinstance(value, FrozenJsonObject):
        return value.to_dict()
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


__all__ = [
    "MAX_JSON_DEPTH",
    "FrozenJsonObject",
    "JsonObjectInput",
    "JsonScalar",
    "JsonValue",
    "JsonValueError",
    "freeze_json",
    "freeze_json_object",
    "thaw_json",
]
