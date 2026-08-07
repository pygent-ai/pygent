# Pygent native load system

The harness measures framework overhead with a deterministic success provider and
end-to-end behavior with the configured GLM-compatible endpoint. It directly
exercises Module, LocalRuntime, ReAct/Tool, SQLite durability, and HTTP Worker
boundaries; it does not require Locust or k6.

```powershell
uv run --extra performance python -m benchmarks dry-run synthetic-smoke
uv run --extra performance python -m benchmarks run synthetic-smoke
uv run --extra performance --env-file .env python -m benchmarks dry-run live-long
uv run --extra performance --env-file .env python -m benchmarks run live-smoke --confirm-live
uv run --extra performance --env-file .env python -m benchmarks run live-long --confirm-live
```

For step-by-step long pressure, use one of the twelve `live-long-<scenario>` aliases.
Each alias contains exactly one execution path and takes about 3.861 hours by
configuration, including warmup and cooldown, so every step stays below 10 hours:

```powershell
$steps = @(
  "direct-invoke", "direct-stream", "local-invoke", "local-stream",
  "dynamic-model-invoke", "dynamic-model-stream",
  "react-tool-invoke", "react-tool-stream",
  "sqlite-durable-invoke", "sqlite-durable-stream",
  "http-worker-invoke", "http-worker-stream"
)

uv run --extra performance --env-file .env python -m benchmarks dry-run live-long-direct-invoke
uv run --extra performance --env-file .env python -m benchmarks run live-long-direct-invoke --confirm-live
```

Replace `direct-invoke` with the next value in `$steps` after inspecting the
previous report. Each step writes an independent result directory and exits at
the first invalid external-provider response. `--scenario http-worker-stream`
remains available for selecting one path from the aggregate profile. Live
profiles create exactly one valid route with no fallback. Timeout and unavailable
failures may use the configured bounded retry; rejection-class and
`OUTCOME_UNKNOWN` failures terminate the stage without another Provider request.
Any terminal provider failure invalidates the stage and returns exit code 3; it
is not classified as a framework performance regression.
The result directory is created before execution begins. Completed stages and
samples remain available when a live campaign stops early, while `summary.json`
uses `outcome = "incomplete"` and identifies the failed phase and error type.
Run `live-smoke` successfully before starting `live-long`.

Results are written below `.benchmarks/results/<run-id>/`:

- `samples.jsonl`: incremental sanitized records with `warmup` or `measurement` phase;
- `summary.json`: atomic checkpoint with outcome, completed stages, partial stage,
  termination type, metrics, and environment fingerprint;
- `stages.csv`: flat stage table;
- `summary.md`: concise human report.

No output contains endpoint, model name, credential, prompt, response content, or
raw provider payload. Warmup samples are retained for failure diagnosis but are
excluded from measured stage summaries. `synthetic-full` is the repeatable performance profile,
`synthetic-soak` runs the three stateful paths for 30 minutes, and aggregate
`live-long` runs every selected path at concurrency 1/2/4/8/16/32 for 30 minutes
per level, then executes calibrated fixed-RPS stages for 10 minutes each. A
`live-long-<scenario>` alias applies that same load shape to only one path.
`dynamic-model-invoke` and `dynamic-model-stream` create two runtime-owned model
profiles per stage and alternate between the default and an explicit per-execution
selection. The invoke path also participates in `synthetic-soak` to pressure
admission/profile lookup over time. `lifecycle-stress` separately churns queued
cancellation and finite deadlines without injecting model failures.

For base/head gates, run the same profile three times on the same machine, reduce
each matching stage to its median, then use `compare` with the reduced JSON
objects. The default gate rejects throughput loss greater than 15% or P95 growth
greater than 25%. Real-model reports are trend evidence only.
