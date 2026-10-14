## Worker and Runtime cleanup budgets

- Execution finalization (terminal journal commit) stays bounded by the execution deadline plus an explicit cleanup grace: `HTTPWorkerApp` and `LocalRuntime` accept `cleanup_budget_seconds` (default `10.0`, must be a positive finite number), replacing the hard-coded 1-second grace. Executions that end at their deadline no longer lose the terminal record to a cleanup budget that had already expired before the commit started.
- The owner-claim release wait has its own hard-capped budget measured from its own start instead of reusing the finalization window, so a slow finalization write can no longer starve claim release (the deterministic "cleanup over budget" retry loop).
- A claim-release failure is recorded instead of raised: `LocalRuntime` keeps it on the execution record (`release_error`) and the HTTP Worker reports it in `GET /v1/executions/{id}` metadata as an additive `release_error` field. A committed terminal outcome is never replaced by a cleanup error.

## Unchanged contracts

- An unconfirmed journal commit still fails closed: the execution stays non-terminal, the Worker answers `503` with the `persistence_error` failure (`outcome_unknown=true`), and a late-landing commit still converges the stored record to its real terminal status.
- Worker shutdown still joins claim-release tasks before closing the history store; `wait_cleanup` still ignores caller cancellation so cleanup cannot be detached from its owner.
