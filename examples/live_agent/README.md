# Live Agent concurrency benchmark

This opt-in example exercises a real OpenAI-compatible endpoint through the
Pygent 0.2 Module, Runtime, LLM, Tool, and ReAct contracts. It deliberately
uses a fresh random invalid API key for the primary route and the configured
key only for the fallback route.

The program never prints the API base, API key, model name, prompts, responses,
HTTP headers, or raw provider exceptions. Its JSON output contains only
aggregate counts and timing data.

## Configuration

The repository-local `.env` must define:

```text
GLM_API_BASE=https://open.bigmodel.cn/api/paas/v4
GLM_API_KEY=...
GLM_MODEL_NAME=...
```

`GLM_API_BASE` is the API root. Do not include `/chat/completions` because the
OpenAI-compatible client appends that path.

No dotenv package is required. Ask `uv` to load the file into the child process:

```powershell
uv run --env-file .env python -m examples.live_agent.benchmark
```

The conservative defaults issue eight logical requests with four concurrent
Executions and at most two provider calls at once. Tune them explicitly when cost and
provider limits are understood:

```powershell
uv run --env-file .env python -m examples.live_agent.benchmark `
  --requests 20 `
  --concurrency 5 `
  --model-concurrency 3 `
  --deadline-seconds 60
```

## What is measured

Each request asks the same stateless Agent definition to call the pure
`benchmark_add` Tool. Tool authorization is an application-owned Module;
executor and model clients are attached to `LocalRuntime`. The Binding limits
live/runnable Executions, shared model calls, and shared Tool calls independently.

The report includes:

- completed/succeeded/failed Executions and sanitized failure type counts;
- wall time, throughput, and min/mean/P50/P95/P99/max latency;
- observed peak provider calls and total provider attempts;
- invalid-primary to configured-fallback count and normalized failure kinds;
- provider token usage when returned;
- Tool result status counts and Context isolation failures.

The public OpenAI-compatible API currently cannot force a Tool choice. A model
may therefore answer without calling the Tool even though the prompt asks it to
do so; this appears as an empty Tool status map and is not silently counted as a
Tool success. Network latency and provider load are external variables, so the
example reports observations rather than imposing a universal performance
threshold.

The normal test suite never contacts the configured endpoint. Its tests use
`httpx.MockTransport` and synthetic responses.

If the configured endpoint does not enforce Authorization itself, use the
local authentication-boundary probe. It rejects the random primary key with a
real HTTP 401 and forwards only the configured fallback key to the real model:

```powershell
uv run --env-file .env python -m examples.live_agent.fallback_probe
```

The proxy binds only to an ephemeral `127.0.0.1` port, never logs credentials,
and exits with the probe. A successful report must contain two authentication
rejections, two forwarded real model calls, one successful fallback Execution, and
one successful Tool result.
