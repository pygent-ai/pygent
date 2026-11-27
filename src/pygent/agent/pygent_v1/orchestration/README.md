# Orchestration

Coordinates application-level work across persisted sessions and managed runtime
executions. This package is independent of HTTP, SSE, Electron, and agent tool
policy so API, CLI, and future message-delivery adapters can share one execution
control plane.

- `models.py`: transport-independent turn commands and lifecycle states.
- `managed_turn.py`: one submitted turn's state, readiness/finalization signals,
  result, cancellation, and raw execution subscription.
- `session_execution.py`: submits and recovers turns while owning per-session
  execution lanes, active indexes, and coordinated shutdown. It also routes
  realtime exchange envelopes into the turn running on the lane or into a
  continuation turn. SQLite retains the envelope across host restarts and
  records its intended execution before attempting either handoff.
- `runtime_keys.py`: stable workspace scope and configuration-generation keys.
- `runtime_lease.py`: the runtime bundle and idempotent ownership lease.
- `runtime_pool.py`: creates, reuses, retires, and closes runtime generations.
- `execution_host.py`: the composition root and unified user-message intake.
  It commits input before routing to a foreground turn, realtime fork, or
  host-provided steering hook. Recovery replays unreceipted deliveries and
  reconciles their targets with the execution journal.


Session identity includes the resolved sessions root as well as the session ID.
This prevents equal IDs in different desktop project scopes from sharing a lane.
Case-run creation and finalization both occur while the lane is held, so the next
turn cannot observe or overwrite an incompletely finalized predecessor.

The package may depend on `sessions` and `runtime`; neither lower-level package
should import orchestration. Transport adapters belong in `lara_api`.

`WorkspaceRuntimePool` is the sole owner of `PygentRuntimeService` instances.
`SessionExecutionCoordinator` accepts a lease rather than independent manager and
runtime arguments, and every managed turn releases that lease after finalization.
The workspace delivery database contains pending inputs, fork exchanges and
per-destination speak receipts. A stable input key deduplicates transport
retries. A session lock serializes foreground work across processes, and
stable execution and inbox identities bridge the commit boundary between
the delivery database and the execution journal. The runtime retries pending
external replies with their original reply IDs.
