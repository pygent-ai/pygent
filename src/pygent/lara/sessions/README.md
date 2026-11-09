# Sessions

Owns session creation, loading, mutation, and on-disk persistence through `SessionManager`.

It depends on core helpers and schema contracts, and stays independent of runtime orchestration.

Project sessions live below `~/.lara/projects/<canonical-path-hash>/sessions`;
projectless sessions use `~/.lara/conversations/sessions`. Runtime databases and
project skills share the same data root. Creating a session does not create a
`.lara` directory inside the workspace.

Starting a case records its history boundary and publishes the running case ID
in session metadata. Native assistant messages preserve usage counters and
reasoning display metadata across checkpoints and application restarts.

Cross-session collaboration operations and Agent-message inboxes are stored in
`runtime/session-collaboration-v1.sqlite3`. Submission IDs are unique and bound to
a request fingerprint. Renewable target-session leases serialize background-start
operations across processes; FIFO message claims provide durable, retryable
delivery into a target execution.

Cross-project addressing runs through the user-level directory
`~/.lara/index/session-directory-v1.sqlite3` (`SessionDirectory`). Sessions
register themselves on create, fork, and load; collaboration items register
when enqueued. A message for a session in another project is written directly
into the target project's store, so delivery on the receiving side is
unchanged. Any session recorded in the directory is addressable; unknown
sessions fail closed.
