"""Fork agents: background agents derived from a context window.

A fork agent materializes its block-1 prefix from a
:class:`~lara.agents.context.ContextSource` (slice + transforms), then runs a
multi-turn ReAct execution steered by one ``<system-reminder>`` directive
(blocks 2 and 3). Tools come from the runner's registered
:class:`~lara.agents.runner.ToolProfile` — forks never carry dedicated tools
or submit executable modules (lara-architecture D2/G4).

Every fork records a :class:`ForkSnapshot` (source, frame-aligned cursor
range, content hash) so the freeze point a fork is based on is auditable;
durable session/cursor/revision sources plug in through ContextSource
implementations backed by runtime state.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import cast

from pygent import Message
from pygent.core import JsonValue
from pygent.runtime.codec import message_to_dict

from .context import (
    ContextSource,
    ContextTransform,
    ContextWindow,
    MessageSlice,
    align_slice_bounds,
)
from .runner import AgentLimits, AgentRunner, AgentRunResult


@dataclass(frozen=True)
class ForkSnapshot:
    """The auditable freeze point a fork run is based on.

    ``source_hash`` covers the sliced source window before transforms;
    ``effective_context_hash`` covers the window after transforms — what the
    agent actually sees. ``transforms`` records each transform's structured,
    content-addressed description in application order (payloads as hashes,
    never text, so snapshot metadata cannot spread directive or prompt
    content into durable stores). Together with ``source`` this is enough to
    audit what a fork saw; the effective content itself is carried by the
    run result's context. Replaying a fork from durable state additionally
    needs a session-backed source.
    """

    source: str
    start_index: int
    end_index: int
    message_count: int
    source_hash: str
    effective_context_hash: str
    transforms: tuple[dict[str, JsonValue], ...]

    def to_metadata(self) -> dict[str, JsonValue]:
        return cast(
            "dict[str, JsonValue]",
            {"lara.fork.snapshot": asdict(self)},
        )


def _content_hash(system_prompt: str, messages: tuple[Message, ...]) -> str:
    payload = json.dumps(
        {
            "system_prompt": system_prompt,
            "messages": [message_to_dict(message) for message in messages],
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ForkSpec:
    """Declarative description of one fork: source, slice, transforms."""

    source: ContextSource
    slice: MessageSlice = field(default_factory=MessageSlice)
    transforms: tuple[ContextTransform, ...] = ()


class ForkAgent:
    """A background agent that continues an existing context window."""

    def __init__(
        self,
        runner: AgentRunner,
        spec: ForkSpec,
        *,
        label: str = "lara-fork",
    ) -> None:
        self._runner = runner
        self._spec = spec
        self._label = label

    def materialize(self) -> tuple[ContextWindow, ForkSnapshot]:
        """Materialize the block-1 prefix and its snapshot.

        Resolve the source, apply the frame-aligned slice, hash that source
        window, then apply transforms in order and hash the effective window.
        Exposed separately so callers (and tests) can inspect exactly what
        the fork will see before spending model calls.
        """

        window = self._spec.source.resolve()
        start, end = align_slice_bounds(
            window.messages, self._spec.slice.start, self._spec.slice.end
        )
        window = replace(window, messages=window.messages[start:end])
        source_hash = _content_hash(window.system_prompt, window.messages)
        transform_descriptors: list[dict[str, JsonValue]] = []
        for transform in self._spec.transforms:
            window = transform.apply(window)
            transform_descriptors.append(transform.describe())
        snapshot = ForkSnapshot(
            source=self._spec.source.describe(),
            start_index=start,
            end_index=end,
            message_count=end - start,
            source_hash=source_hash,
            effective_context_hash=_content_hash(window.system_prompt, window.messages),
            transforms=tuple(transform_descriptors),
        )
        return window, snapshot

    def plan(self) -> ContextWindow:
        """Materialize the block-1 prefix without running the agent."""

        return self.materialize()[0]

    async def run(
        self,
        directive: str,
        *,
        limits: AgentLimits | None = None,
    ) -> AgentRunResult:
        """Run the fork: block 1 prefix + directive steering (block 2)."""

        window, snapshot = self.materialize()
        window = replace(window, metadata={**window.metadata, **snapshot.to_metadata()})
        return await self._runner.run(
            message=directive,
            prefix=window,
            limits=limits,
            label=self._label,
        )
