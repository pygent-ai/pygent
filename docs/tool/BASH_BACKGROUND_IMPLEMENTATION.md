# Bash Background Implementation Plan

**Goal:** Implement the approved Bash background design as one task lifecycle, with direct handles, model snapshots and restart-readable records.

**Spec:** [Bash background design](BASH_BACKGROUND_DESIGN.md)

**Architecture:** ToolTaskManager owns execution and snapshots. ToolRunner exposes the current tool execution context to local adapters; Bash executes the process once under that owner. ToolCallLayer admits independent tasks and observes them for the declared wait timeout. Direct Bash uses the same manager with a locally owned facility.

**Constraints:** Default wait 600 seconds; no competing Bash execution timeout. No log pagination, long polling, quota system, process recovery or new permission model. Native Python handles never enter portable results. Existing sandbox, identity, cancellation and capacity boundaries apply. Write as a coherent new API, not compatibility branches.

## Tasks

- [x] Task facilities: Add cancellation-isolated ToolTaskHandle, manager output snapshots, task-local execution context and restart-readable output using existing history. Tests cover observer cancellation and preserved output after terminalization/restart.
- [x] Bash: One process execution path, constructor wait timeout, auto-owned direct task facility, explicit background handles, capture snapshots and shared get/stop tools. Tests use real processes for completion, background, stopping and cleanup.
- [x] Composition: Portable wait policy, declaration/schema support, independent admission and finite waiting, runtime query SPI and lease handoff. Tests cover authorization, ordered results, direct JSON projection and single-lease managed execution.
- [x] Integrate and validate: Run focused tests, type/lint checks and relevant broader regressions. Review cancellation, ownership, persistence and portable-value boundaries.
- [x] Documentation: Update SDK examples and affected existing project map claims to verified behavior; record exact validation and limitations.

## Execution

Use test-driven-development and subagent-driven-development for independently owned task facilities and Bash implementation, with integration in the main agent. Tests precede implementation. Changes remain reviewable on codex/bash-background; no release or push.

## Validation

Full repository: 1319 tests passed. The subsequent native-handle projection case passed in the 6-test composition suite. Mypy passed for 89 source files; Ruff and diff whitespace checks passed. Validation ran locally on Windows; no release or cross-platform CI was run.
