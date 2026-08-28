# Pygent 0.2 Execution contract

Pygent has one logical identity for every business execution and one fenced owner for every actual attempt. `start()` creates the logical execution plus its first owner task and returns an `ExecutionHandle` without waiting for admission; `invoke()` is exactly `start() + result()`, while `stream()` is an owned `start() + subscribe()` view. None of these entry points may execute `forward()`, a provider request, or a tool operation a second time.

## Control, result, and observation

```python
handle = await module.start(*args, **kwargs)

async def observe():
    async with handle.subscribe(after=None) as events:
        async for event in events:
            print(event.kind, event.sequence, event.data)

result, _ = await asyncio.gather(handle.result(), observe())
```

`ExecutionHandle` exposes `execution_id`, `snapshot()`, `outcome()`, `result()`, `cancel()`, `subscribe(after=...)`, and `send_input(input_id=..., kind=..., value=...)`. It is a stable reference backed by an `ExecutionBackend`; it does not own the coroutine. `snapshot()` returns the canonical `ExecutionSnapshot` including logical status, detailed phase, active `attempt_id`, owner state, trace identity, last journal sequence, optional terminal sequence, and timestamps; `outcome()` returns the frozen terminal identity and final journal cursor. Multiple subscriptions own independent cursors. Closing a subscription only stops that observer. Exiting a stream created by `module.stream()` before consuming its result cancels that stream's execution because the stream owns it.

Managed Runtime inputs are portable `ExecutionInput` values. A Module receives them with `receive_execution_inputs(kinds=..., limit=..., seal_if_empty=...)`; Runtime owns only bounded transport, ordering, idempotency, one consumer per kind, durable receive replay, and the finalization race. Reusing `input_id` returns `duplicate` with the original input sequence. A sealed or terminal inbox returns `execution_finished`. The atomic `seal_if_empty=True` read closes the final empty window. Runtime never interprets `kind` or `value`; direct execution receives no inputs and rejects `send_input()`.

`ExecutionStatus` is the coarse logical state: `PENDING`, `RUNNING`, `SUCCEEDED`, `FAILED`, `CANCELLED`, or `DEADLINE_EXCEEDED`. `ExecutionPhase` explains non-terminal progress such as `SUBMITTING`, `PREPARING`, `WAITING_ADMISSION`, `STARTING`, `RUNNING`, and `FINALIZING`. A submitted execution always reaches one terminal `ExecutionOutcome`; admission errors are outcomes of that logical execution, not exceptions that occur before an execution exists.

Every public `ExecutionEvent` is strict JSON and contains `schema_version`, globally unique `event_id`, `execution_id`, `attempt_id`, `trace_id`, `span_id`, optional `parent_span_id`, per-execution `sequence`, `timestamp_unix_ns`, `module_path`, `kind`, and `data`. Sequence is monotonic only inside one observable execution stream. `execution.submitted` is the first lifecycle event; an admitted attempt emits `execution.admitted` before `forward()` begins. Child modules and tool calls create spans; each root execution and span emits exactly one terminal lifecycle event.

Finalization appends all terminal span events, the one Execution terminal event, the frozen `ExecutionOutcome`, terminal snapshot, and `terminal_sequence` atomically. A subscription ends only after its cursor has yielded `terminal_sequence`. Reading a terminal status is never sufficient to stop a live or durable subscription.

`runtime.get_execution_handle(execution_id)` attaches to an existing execution and never creates an attempt. `runtime.recover(execution_id, ...)` is a separate privileged operation that validates recovery eligibility, obtains a fenced owner lease, and creates a new `attempt_id`.

Remote events are imported into the parent execution and receive a new parent-stream `sequence`. Their `event_id` is preserved and `data` includes `origin_execution_id` and `origin_sequence`, allowing reconnect deduplication without claiming global clock ordering.

## Model and tool ownership

`ModelInvoker.execute(...) -> ModelExecution` is the only model SPI. `ModelExecution` owns provider routing, retry/fallback, cancellation, events, usage, and the final `ModelProviderResponse`. Native streaming is preferred when the provider capability declares it; a non-streaming provider is projected into text/tool-call/usage/finish events. Once output is visible, a later failure is terminal and is not hidden by retry or fallback.

A new provider attempt may start only after the previous attempt is confirmed finished. Cancellation cleanup is bounded by an internal one-second grace and the remaining execution deadline. If cleanup cannot be confirmed, the attempt fails as `OUTCOME_UNKNOWN` with the sanitized reason `cancellation_cleanup_timeout`; retry and fallback are prohibited, and the affected client object is quarantined so no overlapping provider request can start. Quarantine is released automatically when the background task exits. Explicit caller cancellation remains cancellation, while a managed execution deadline remains `ExecutionDeadlineExceeded` at the Execution boundary. Each native Provider stream has one owner task that alone advances and closes its iterator. Deployment shutdown is stricter than request cleanup: `DefaultModelInvoker.aclose()` joins active executions and all quarantined stream owners before closing shared clients and returning.

Model events use a closed vocabulary. Every logical call emits `model.started` and exactly one of `model.completed`, `model.failed`, or `model.cancelled`. Attempts emit started plus succeeded or failed. Provider-visible reasoning uses `model.reasoning.delta`; answer text uses `model.text.delta`; streamed Tool intent uses `model.tool_call.started`, `.delta`, and `.completed`. `model.tool_call.*` describes generation only, while `tool.*` describes authorization and execution. `model.completed` is emitted only after complete Tool arguments, structured output, final usage, and the final `AIMessage` have all validated.

Terminal model failures expose Provider-neutral diagnostics through `ModelCallError` and its portable `ExecutionFailure.details`. Each failed attempt may retain a closed sanitized `reason_code` and a validated numeric `http_status`; these values survive managed effects, durable replay and Worker transport. Provider response messages, arbitrary Provider codes or headers, raw bodies, credentials, endpoints and internal exception chains never cross that boundary. The fixed `model.*` event payloads remain aggregate lifecycle observations and do not carry Provider diagnostics.

`model.usage` is an attempt-scoped cumulative snapshot. Its fixed counters are `input_tokens`, `output_tokens`, `total_tokens`, `cached_input_tokens`, and `reasoning_tokens`; unavailable counters are `null`. Consumers take the last snapshot for one `(span_id, route_id, attempt)` and sum final snapshots across attempts. Provider-specific usage objects never cross this boundary.

`ToolRunner.execute(...) -> ToolExecution` is the only tool-operation owner. It owns timeout, cancellation, executor failure normalization, output freezing, and `ToolResult`. `ToolCallLayer` continues to own visibility, schema validation, authorization, detach choice, and batch ordering. Executors receive `ToolExecutionContext` with the effective deadline and event emitter.

Managed effects return `EffectOutcome(value, disposition, effect_id, attempt)`. `disposition` is `executed`, `replayed`, or `retried`. Replay returns the committed value and emits an effect replay event; it never fabricates provider/tool deltas or counts usage again.

## Contract boundary

There are no aliases, adapters, dual state models, or history migrations for superseded Runtime control-plane contracts. Each Module keeps the business call contract declared by its own `forward()`; RecurrentModule additionally declares explicit state recurrence. Direct execution may use free Python values. Managed, remote, and durable execution accept only call contracts supported by that Runtime; this change does not define a universal wire codec for arbitrary Module values.

Trace persistence, trace queries, cost aggregation, and reasoning capture are deliberately outside this release. The event and span identities added here are their foundation, not a TraceStore product.
