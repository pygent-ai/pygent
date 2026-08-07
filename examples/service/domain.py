"""Application-owned values and persistence interface."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol

from pygent import Message


@dataclass(frozen=True, slots=True)
class ChatRequest:
    session_id: str
    user_id: str
    text: str
    permissions: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ChatResponse:
    session_id: str
    revision: int
    text: str


@dataclass(frozen=True, slots=True)
class Conversation:
    revision: int = 0
    messages: tuple[Message, ...] = ()


class ConversationStore(Protocol):
    """Implemented by the consuming service, never by an Agent Module."""

    async def read(self, session_id: str) -> Conversation: ...

    async def commit(
        self,
        session_id: str,
        expected_revision: int,
        messages: tuple[Message, ...],
    ) -> int: ...


class ConversationConflict(RuntimeError):
    """Raised when a caller commits against a stale conversation revision."""


class InMemoryConversationStore:
    """Small CAS store used by the runnable release example."""

    def __init__(self) -> None:
        self._values: dict[str, Conversation] = {}
        self._lock = asyncio.Lock()

    async def read(self, session_id: str) -> Conversation:
        async with self._lock:
            return self._values.get(session_id, Conversation())

    async def commit(
        self,
        session_id: str,
        expected_revision: int,
        messages: tuple[Message, ...],
    ) -> int:
        async with self._lock:
            current = self._values.get(session_id, Conversation())
            if current.revision != expected_revision:
                raise ConversationConflict(
                    f"stale revision {expected_revision}; current={current.revision}"
                )
            revision = current.revision + 1
            self._values[session_id] = Conversation(revision, tuple(messages))
            return revision


__all__ = [
    "ChatRequest",
    "ChatResponse",
    "Conversation",
    "ConversationConflict",
    "ConversationStore",
    "InMemoryConversationStore",
]
