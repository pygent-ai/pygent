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
  realtime exchange envelopes: into the turn running on the lane, or held in
  memory for the next turn's opening context (pending envelopes share the fate
  of lane-queued turns).
- `runtime_keys.py`: stable workspace scope and configuration-generation keys.
- `runtime_lease.py`: the runtime bundle and idempotent ownership lease.
- `runtime_pool.py`: creates, reuses, retires, and closes runtime generations.
- `execution_host.py`: short-lived CLI composition root with ordered shutdown;
  `deliver_user_message` is the single user-message entry — it routes realtime
  sessions with a busy lane through the fork interrupt and everything else
  through turn submission (lease, session creation, and title persistence
  included).

Session identity includes the resolved sessions root as well as the session ID.
This prevents equal IDs in different desktop project scopes from sharing a lane.
Case-run creation and finalization both occur while the lane is held, so the next
turn cannot observe or overwrite an incompletely finalized predecessor.

The package may depend on `sessions` and `runtime`; neither lower-level package
should import orchestration. Transport adapters belong in `lara_api`.

`WorkspaceRuntimePool` is the sole owner of `PygentRuntimeService` instances.
`SessionExecutionCoordinator` accepts a lease rather than independent manager and
runtime arguments, and every managed turn releases that lease after finalization.
The shared SQLite message store is the durable delivery contract: expiring
message claims allow an interrupted tool-boundary delivery to be retried.
