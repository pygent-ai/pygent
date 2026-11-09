# Core

Stable, dependency-light helpers shared by every feature domain.

- `io.py`: JSON/JSONL, path validation, timestamps, and plain-data conversion.
- `prompt.py`: streaming XML prompt fragments and the known-envelope registry.
- `process.py`: process ownership stamps for artifacts shared by multiple
  hosts.
- `redaction.py`: secret redaction.

Core must not import runtime, evaluation, repair, CLI, API, or UI code.
