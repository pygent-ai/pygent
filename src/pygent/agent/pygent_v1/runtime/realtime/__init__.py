"""Real-time interaction harness (D-RT series, see docs/design/realtime-interaction).

The agent works silently: content/thinking stay out of the user-facing stream and
``speak`` is the only channel the user sees. Inactive until a run config enables
``realtime_interaction``.

- ``speak``          — the user-visible channel tool + renderer/source abstraction
- ``fork_reply``     — fork-snapshot request construction and speak extraction
- ``fork_agent``     — the fork reply as a minimal managed execution graph
- ``fork_reasoning`` — fork-only model profiles with reduced reasoning effort
- ``fork_state``     — per-session supersede-and-merge state
- ``envelope``       — conversation-exchange injection at the next tool-result boundary
- ``posting``        — envelope delivery through the agent-message channel
- ``nudge``          — N-rounds-without-speak reminder rule and text
- ``injection``      — the tool-result boundary nudge module
- ``turn_finalizer`` — end-of-turn bottom-line speak decision rule

Wiring: the envelope rides the existing agent-message queue (claimed by the
runtime's reminder boundary); the nudge module composes into
``PreparedToolModule`` behind ``realtime_interaction_enabled``; the fork reply
runs as a pygent managed execution via ``PygentRuntimeService.send_interrupt``;
the finalizer round is queued by the session coordinator's lane at turn
completion.
"""
