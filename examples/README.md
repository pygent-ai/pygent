# Pygent Examples

`basic_agent/` is the smallest live `ReActAgent`: one OpenAI-compatible model,
a system prompt, and a compression prompt. It needs provider credentials and
runs single-turn or as an interactive REPL; see
[`basic_agent/README.md`](basic_agent/README.md).

```bash
uv run python -m examples.basic_agent "用一句话介绍你自己"
```

`tutorial/` is the progressive, zero-credential starting point. It reuses one
Agent graph across direct invoke, stream, an opt-in OpenAI-compatible provider,
managed Runtime execution, and dynamic model profiles:

```bash
uv run python -m examples.tutorial
uv run python -m examples.tutorial stream
uv run python -m examples.tutorial managed
```

`service/` is application code that consumes the public contract in
`src/pygent`. Framework code never imports example code.

It demonstrates the second-level SDK contracts documented under `docs/`:

- immutable model group, retry, fallback, and generation declarations;
- tool declarations, application authorization, and a shared ToolCallLayer;
- a user-defined CoordinatorAgent composed from ReAct and ReviewAgent;
- explicit Message/Context commit flow and service-owned persistence;
- finite ReAct budgets, Execution deadline, and Binding capacity declarations;
- inherited `emit()` plus the same Module graph for invoke and stream.

Run the managed invoke/stream flow with:

```bash
uv sync --all-extras
uv run python -m examples.service.main
```

The offline demo uses a deterministic ModelInvoker, `LocalRuntime`, ordered events, and revision-based CAS persistence, so it requires no external credentials.

`live_agent/` is the opt-in real OpenAI-compatible GLM Agent and concurrency benchmark. It reads `GLM_API_BASE`, `GLM_API_KEY`, and `GLM_MODEL_NAME`, sends a fresh random invalid key to the primary route, verifies normalized fallback to the configured route, runs a bounded ReAct tool loop, and prints only aggregate latency/throughput/fallback/usage/isolation metrics.

```bash
uv run --env-file .env python -m examples.live_agent.benchmark \
  --requests 8 --concurrency 4 --model-concurrency 2
```

The report never includes the API key, endpoint, model name, prompt, response text, or raw Provider error body.
