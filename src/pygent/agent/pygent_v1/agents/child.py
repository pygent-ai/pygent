"""Independent child agents for the pygent_v1 architecture.

A child agent owns its system prompt and starts from a fresh context — no
inherited history. It runs through the same :class:`AgentRunner` seam as the
main and fork agents, referencing registered tool surfaces via
:class:`~pygent_v1.agents.runner.ToolProfile` (lara-architecture G2: two
background origins, no privilege difference, no side-channel tools).
"""

from __future__ import annotations

from dataclasses import dataclass

from .runner import AgentRunner, AgentRunResult


@dataclass(frozen=True)
class ChildSpec:
    """Declarative description of one independent background agent.

    A child runs against the runner's default tool profile and limits; tool
    surfaces enter through the uniform :class:`ToolProfile` mechanism, never
    as per-child tool modules.
    """

    name: str
    system_prompt: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("ChildSpec.name must be non-empty")
        if not self.system_prompt.strip():
            raise ValueError("ChildSpec.system_prompt must be non-empty")


class ChildAgent:
    """An independent background agent with its own context."""

    def __init__(
        self,
        runner: AgentRunner,
        spec: ChildSpec,
        *,
        label: str | None = None,
    ) -> None:
        self._runner = runner
        self._spec = spec
        self._label = label or f"pygent.v1-child.{spec.name}"

    async def run(self, directive: str) -> AgentRunResult:
        """Run the child agent against its directive."""

        return await self._runner.run(
            message=directive,
            system_prompt=self._spec.system_prompt,
            label=self._label,
        )
