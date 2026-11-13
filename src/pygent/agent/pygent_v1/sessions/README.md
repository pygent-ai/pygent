# Sessions

Owns session creation, loading, mutation, and on-disk persistence through `SessionManager`.

It depends on core helpers and schema contracts, and stays independent of runtime orchestration.

Project sessions live below `~/.pygent_v1/projects/<canonical-path-hash>/sessions`;
projectless sessions use `~/.pygent_v1/conversations/sessions`. Runtime databases and
project skills share the same data root. Creating a session does not create a
`.pygent_v1` directory inside the workspace.

Starting a case records its history boundary and publishes the running case ID
in session metadata. Native assistant messages preserve usage counters and
reasoning display metadata across checkpoints and application restarts.

Session addressing is project-local: a session must live in the caller's
project store, and unknown sessions fail closed.
