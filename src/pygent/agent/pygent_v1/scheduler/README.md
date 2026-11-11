# Scheduler

Independent, single-agent scheduler daemon. It has no in-process coupling to
an agent session: `lara scheduler run` owns its own process, polls SQLite, and
invokes `lara session run <scheduler-session> -m <prompt>` as a child process.

## Layers

- `store.py`: one SQLite file (`<pygent_root>/scheduler/pygent.v1-scheduler.sqlite3`)
  with append-only signed `submissions` and daemon-owned `tasks` / `runs`.
  WAL, busy timeout, and lease-checked atomic claims are always enabled.
- `schedule.py`: deterministic `every`, `at`, `daily`, and `on` parsing and
  advancement.
- `keys.py`: per-source HMAC-SHA256 keys under the user pygent_v1 directory. The
  two-key format is intentionally replaceable by Ed25519 later.
- `daemon.py`: verification, promotion, `needs_review`/approval gates, due and
  missed execution, prompt-hash rechecks, audit runs, and failure alerts.
- `message.py`: builds the `<task-context>` execution message with the core
  Prompt API and task-specific instructions.
- `tools.py`: `schedule_task` / `emit_event`; they sign and append submissions,
  never write tasks directly.
- `../cli/scheduler.py`: `run`, `add`, `list`, `show`, approval/control,
  `init-keys`, and Windows autostart commands.

## Trigger bus

The `tasks` table is the unified trigger bus. It carries timer triggers, event
triggers, and agent-registered intents; a user's direct chat message is served
by the unchanged foreground runtime and never enters the bus.

- `origin` is `user` (signed user/agent registration), `timer`, or `event`
  (daemon-generated triggers; task `source` becomes `self`).
- `delivery` is `sync-reply`, `speak`, or `fire-and-forget`. Agent submissions
  are always fire-and-forget; user submissions default to fire-and-forget.
- `priority` is `high`, `normal`, or `low`. Due tasks are claimed in priority
  order, then by `next_run_at`. Agent requests for `high` are normalized to
  `normal` with an `agent-requested-high normalized` audit note.

## Trust boundary

CLI and agent writers can only append `submissions`. The daemon verifies the
signature and claimed source, then alone writes `tasks` and `runs`. User
submissions default to `approved`; agent submissions default to
`needs_review` unless `scheduler.agent_create_enabled` is enabled.
