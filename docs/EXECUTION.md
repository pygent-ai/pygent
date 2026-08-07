# Pygent 0.2 Execution contract

Pygent 0.2 has one owner for every business execution. `start()` creates that owner and returns an `ExecutionHandle`; `invoke()` is exactly `start() + result()`, while `stream()` is an owned `start() + subscribe()` view. None of these entry points may execute `forward()`, a provider request, or a tool operation a second time.

## Control, result, and observation

```python
handle = await module.start(message, context)

async def observe():
    async with handle.subscribe(after=None) as events:
        async for event in events:
            print(event.kind, event.sequence, event.data)

result, _ = await asyncio.gather(handle.result(), observe())
```

`ExecutionHandle` exposes `execution_id`, `trace_id`, `status`, `result()`, `cancel()`, and `subscribe(after=...)`. Multiple subscriptions own independent cursors. Closing a subscription only stops that observer. Exiting a stream created by `module.stream()` before consuming its result cancels that stream's execution because the stream owns it.

Every public `ExecutionEvent` is strict JSON and contains `schema_version`, globally unique `event_id`, `execution_id`, `trace_id`, `span_id`, optional `parent_span_id`, per-execution `sequence`, `timestamp_unix_ns`, `module_path`, `kind`, and `data`. Sequence is monotonic only inside one observable execution stream. Child modules and tool calls create spans; each root execution and span emits exactly one terminal lifecycle event.

Remote events are imported into the parent execution and receive a new parent-stream `sequence`. Their `event_id` is preserved and `data` includes `origin_execution_id` and `origin_sequence`, allowing reconnect deduplication without claiming global clock ordering.

## Model and tool ownership

`ModelInvoker.execute(...) -> ModelExecution` is the only model SPI. `ModelExecution` owns provider routing, retry/fallback, cancellation, events, usage, and the final `ModelProviderResponse`. Native streaming is preferred when the provider capability declares it; a non-streaming provider is projected into text/tool-call/usage/finish events. Once output is visible, a later failure is terminal and is not hidden by retry or fallback.

A new provider attempt may start only after the previous attempt is confirmed finished. Cancellation cleanup is bounded by an internal one-second grace and the remaining execution deadline. If cleanup cannot be confirmed, the attempt fails as `OUTCOME_UNKNOWN` with the sanitized reason `cancellation_cleanup_timeout`; retry and fallback are prohibited, and the affected client object is quarantined so no overlapping provider request can start. Quarantine is released automatically when the background task exits. Explicit caller cancellation remains cancellation, while a managed execution deadline remains `ExecutionDeadlineExceeded` at the Execution boundary. Each native Provider stream has one owner task that alone advances and closes its iterator. Deployment shutdown is stricter than request cleanup: `DefaultModelInvoker.aclose()` joins active executions and all quarantined stream owners before closing shared clients and returning.

Model events use a closed vocabulary. Every logical call emits `model.started` and exactly one of `model.completed`, `model.failed`, or `model.cancelled`. Attempts emit started plus succeeded or failed. Provider-visible reasoning uses `model.reasoning.delta`; answer text uses `model.text.delta`; streamed Tool intent uses `model.tool_call.started`, `.delta`, and `.completed`. `model.tool_call.*` describes generation only, while `tool.*` describes authorization and execution. `model.completed` is emitted only after complete Tool arguments, structured output, final usage, and the final `AIMessage` have all validated.

`model.usage` is an attempt-scoped cumulative snapshot. Its fixed counters are `input_tokens`, `output_tokens`, `total_tokens`, `cached_input_tokens`, and `reasoning_tokens`; unavailable counters are `null`. Consumers take the last snapshot for one `(span_id, route_id, attempt)` and sum final snapshots across attempts. Provider-specific usage objects never cross this boundary.

`ToolRunner.execute(...) -> ToolExecution` is the only tool-operation owner. It owns timeout, cancellation, executor failure normalization, output freezing, and `ToolResult`. `ToolCallLayer` continues to own visibility, schema validation, authorization, detach choice, and batch ordering. Executors receive `ToolExecutionContext` with the effective deadline and event emitter.

Managed effects return `EffectOutcome(value, disposition, effect_id, attempt)`. `disposition` is `executed`, `replayed`, or `retried`. Replay returns the committed value and emits an effect replay event; it never fabricates provider/tool deltas or counts usage again.

## Compatibility boundary

This is a breaking 0.2 contract. Public `Run*` names, the former split ModelInvoker `invoke/stream` SPI, `/runs` Worker routes, pre-0.2 Runtime plans and Worker payloads, and pre-0.2 SQLite history are rejected without aliases, adapters, or migration. The stable business contract remains `forward(message, context) -> (message, context)`, with direct execution still excluding Binding capacity, remote placement, and durable recovery.

Trace persistence, trace queries, cost aggregation, and reasoning capture are deliberately outside this release. The event and span identities added here are their foundation, not a TraceStore product.
