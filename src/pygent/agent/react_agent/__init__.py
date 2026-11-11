"""The built-in ReAct agent implementation entry.

``pygent.agent`` hosts both reusable base capabilities and independent agent
implementations. This package is the entry for the built-in ReAct individual:
it re-exports the ``ReActAgent`` implementation from the base modules — the
same class object, no copy. The lara individual lives at
``pygent.agent.lara``; future implementations land as sibling subpackages
without touching the base capabilities.
"""

from pygent.agent.pygent_agent import ReActAgent

__all__ = ["ReActAgent"]
