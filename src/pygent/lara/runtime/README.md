# Runtime

Owns model execution, tool execution, context management, and the high-level runtime service.

- `agent/`: agent orchestration split by concern.
  - `core.py`: `LaraAgent` construction and run lifecycle.
  - `model_invoker.py`: native model execution with per-call reasoning display
    metadata; output resets discard failed-attempt reasoning before persistence.
  - `pipeline.py`: model/tool middleware and authorization.
  - `prompt_models.py`: prompt contracts and render context.
  - `prompts.py`: prompt registry, composition, injection policy, and cache.
  - `prompt_sources.py`: built-in prompt renderers and reminder state.
  - `skill_catalog.py`: project/user Skill discovery and shadowing rules.
  - `common.py`: Pygent message conversion and small persistence helpers.
- `context.py`: portable `LaraContext`, a Pygent `Context` subclass and the single
  source of execution-scoped facts. It carries session/case/run/turn identity,
  model projection, complete persisted history, and deferred file-effect jobs.
- `context_compression.py`: model-context compaction.
- `fork_context.py`: shared live/frozen session fork windows for realtime
  fork-replies and memory extraction.
- `reminders/`: session bootstrap, Git/CLI/Skill observations, and the persistent
  Agent-message inbox. Rendered runtime context and Agent messages are nested XML,
  so the reminder module appends both directly to `ToolMessage.content`; Pygent's
  `format_context` envelope would escape that markup into inert text. Raw
  ToolResults remain unchanged, and the outer conversation checkpoint persists the
  exact projection the model receives. There is no legacy system-reminder renderer
  or v1 adapter.
- `tools.py`: tool observation and file-effect discovery.
- `file_effect_models.py`: dependency-light file-effect contracts.
- `file_effects.py`: deferred file-effect persistence and execution.
- `deployment.py`: model-resource and workspace executor adapters.
- `agent_collaboration.py`: model-visible session collaboration tool definitions
  and visibility policy.
- `service.py`: session-oriented runtime facade used by API and CLI adapters.

Runtime callers import concrete owning modules; package-level compatibility
re-exports are intentionally not provided.

## Media file tools

Pygent 0.3.19 `read(file_path, limit=None, offset=None, pages=None)` is registered
in both the default Agent toolkit and Runtime executor registry. It reads text,
extracts PDF text (or renders selected `pages`), and returns native structured
media blocks for images and MP4 video. Separate `read_image` / `read_video`
tools are not exposed. Text-only `limit` / `offset` cannot be used with media.
Lara's audit projection leaves Pygent's `ToolResult.output` untouched: ordinary
text reads reach the model as the file text over the provider's plain output
path, while media reads keep their native structured blocks.
The OpenAI Chat Completions adapter enables inline image/video tool results with
a 20 MiB per-media limit; the selected model must also declare the corresponding
input modality. Endpoint support still depends on the configured provider.
Other protocols retain Pygent's unsupported-content handling.
Native image processing corrects orientation and bounds the delivered image to
2048 px on its longest edge and 3,500,000 bytes. The video extra supplies PyAV;
native video processing enforces a 120-second, 12,000,000-byte, 1280-px and
15-FPS delivery budget. History media is not automatically evicted.

## Runtime-state boundary

`LaraAgent` is a reusable, run-independent Pygent module graph. It must not retain
the current session, case run, turn, observer, or a mutable side-effect queue.
Each managed execution receives those portable facts through `LaraContext`.

Modules may retain only definition-scoped collaborators such as configuration,
prompt registries, model invokers, and workspace adapters. Run-bound services
(`EventStore`, `SessionManager`, prompt context views, and `DiffTool`) are rebuilt
from `LaraContext` at the point of use. This keeps concurrent executions isolated
without creating a new Agent graph for every turn.

Deferred file effects flow explicitly through `LaraContext.pending_file_effects`:
tool observation appends jobs, and the persisted-diff module drains them into the
managed Pygent tool task. There is no hidden observer-owned queue.

Checkpoint restoration delegates directly to Pygent's current context codec
registry without Lara-owned version detection or migration branches.

## Eternal conversation projection

Before the asynchronous extractor publishes its first snapshot, each turn restores
SessionManager history. Once a snapshot is available, native replacement delivers
that snapshot plus every uncovered message. The suffix starts at a user-turn
boundary to retain matching tool calls and results; a fully covered history keeps
its latest exchange for follow-up references. Stored history is not removed.

## File-effect scan limits

Lara observes file effects after tool execution. Explicit write/edit jobs snapshot
only their declared targets. Shell write detection ignores quoted text, comments,
and heredoc bodies; it remains a heuristic, not a complete shell interpreter.
Full scans prune dependency, build, temporary, and runtime directories before
traversal, including `.tmp`, `node_modules`, and `.venv-*`. Explicit targets inside
excluded directories remain observable.

Scans check a 5-second deadline cooperatively between directory/file operations
and hash chunks, with limits of 20,000 files and 64 MiB of reads. This is not a hard
OS I/O timeout: an individual blocked filesystem call cannot be interrupted by
these checks. An incomplete scan emits `runtime.file_scan.incomplete`, preserves
the prior baseline, and returns without blocking the next model step on more scan
work. Partial scans never infer deletions. Target-only baselines carry
`complete: false` until a successful full scan seeds the workspace baseline.
