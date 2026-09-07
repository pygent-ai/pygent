"""Attempt authority captured by writers and checked inside SQLite transactions."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import aiosqlite

from ._history_types import HistoryConflictError


@dataclass(frozen=True, slots=True)
class WriteAuthority:
    execution_id: str
    owner_id: str | None = None
    fencing_token: int | None = None


_owner: ContextVar[tuple[str, WriteAuthority] | None] = ContextVar(
    "pygent_history_owner", default=None
)


@contextmanager
def owner_scope(
    execution_id: str,
    owner_id: str,
    fencing_token: int,
    *,
    journal_id: str | None = None,
) -> Iterator[None]:
    token = _owner.set(
        (
            journal_id or execution_id,
            WriteAuthority(execution_id, owner_id, fencing_token),
        )
    )
    try:
        yield
    finally:
        _owner.reset(token)


def write_authority(execution_id: str) -> WriteAuthority:
    owner = _owner.get()
    return (
        owner[1]
        if owner is not None and execution_id in (owner[0], owner[1].execution_id)
        else WriteAuthority(execution_id)
    )


async def validate_writers(
    db: aiosqlite.Connection, authorities: Sequence[WriteAuthority]
) -> None:
    """Validate the whole batch under its BEGIN IMMEDIATE write lock."""
    unique = tuple(dict.fromkeys(authorities))
    if not unique:
        return
    placeholders = ",".join("?" for _ in unique)
    async with db.execute(
        "SELECT execution_id,owner_id,fencing_token,expires_at>unixepoch('subsec') "
        f"FROM execution_claims WHERE execution_id IN ({placeholders})",
        tuple(item.execution_id for item in unique),
    ) as cursor:
        claims = {row[0]: row[1:] for row in await cursor.fetchall()}
    for authority in unique:
        claim = claims.get(authority.execution_id)
        if authority.fencing_token is None:
            valid = claim is None
        else:
            valid = claim is not None and claim == (
                authority.owner_id,
                authority.fencing_token,
                1,
            )
        if not valid:
            raise HistoryConflictError("execution writer no longer owns a valid lease")
