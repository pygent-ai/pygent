# Orchestration

Coordinates application-level work across persisted sessions and managed runtime
executions. This package is independent of HTTP, SSE, Electron, and agent tool
policy so API, CLI, and future message-delivery adapters can share one execution
control plane.

- `models.py`: transport-independent turn commands and lifecycle states.
- `managed_turn.py`: one submitted turn's state, readiness/finalization signals,
  result, cancellation, and raw execution subscription.
- `session_execution.py`: submits and recovers turns while owning per-session
  execution lanes, active indexes, and coordinated shutdown.
- `runtime_keys.py`: stable workspace scope and configuration-generation keys.
- `runtime_lease.py`: the runtime bundle and idempotent ownership lease.
- `runtime_pool.py`: creates, reuses, retires, and closes runtime generations.
- `session_turns.py`: submits ordinary turns for transport adapters without
  duplicating lease, session creation, or title persistence logic.
- `session_collaboration.py`: persists idempotent background-start operations and
  Agent messages. Start operations are dispatched asynchronously and serialized
  per target across processes. A sourced operation atomically enqueues its terminal
  result back to the source session; sent and completion messages stay queued until
  the target runtime claims them at a tool boundary. Both expose
  transport-independent status/wait. The host injects this same service into
  workspace runtimes, so model tools and CLI commands share one control plane.
- `execution_host.py`: short-lived CLI composition root with ordered shutdown.

Session identity includes the resolved sessions root as well as the session ID.
This prevents equal IDs in different desktop project scopes from sharing a lane.
Case-run creation and finalization both occur while the lane is held, so the next
turn cannot observe or overwrite an incompletely finalized predecessor.

The package may depend on `sessions` and `runtime`; neither lower-level package
should import orchestration. Transport adapters belong in `lara_api`.

`WorkspaceRuntimePool` is the sole owner of `LaraRuntimeService` instances.
`SessionExecutionCoordinator` accepts a lease rather than independent manager and
runtime arguments, and every managed turn releases that lease after finalization.
`SessionCollaborationService` owns only start-operation dispatcher and heartbeat
tasks. Its SQLite store is the durable query contract: expiring operation leases
allow pending starts to be resumed, while expiring message claims allow an
interrupted tool-boundary delivery to be retried. Operation finalization and its
completion-message outbox write share one transaction, so recovery cannot expose a
terminal child without the corresponding parent notification.
