# Pygent 0.3.10 Cross-Framework Resource Benchmark

> Benchmark snapshot: 2026-09-08. This report measures framework overhead on
> Windows, using a fresh process for each repetition and a local
> OpenAI-compatible mock. It is not a model-quality evaluation or a guarantee
> of production capacity, real-API latency, or reliability.

## Executive summary

Pygent 0.3.10 with `DurabilityPolicy.PREFERRED` had the highest composite score
in the durable execution runtime lane at both 200 and 1,000 concurrent
requests. Among the qualifying durable configurations, it recorded the lowest
run time, P95 latency, process-tree CPU time, and peak RSS at both loads. This
does not mean that Pygent led every metric.

Disk writes were Pygent's main cost. Relative to LangGraph with SQLite, Pygent
wrote 40.884 MB more at concurrency 200 (+74.2%) and 438.864 MB more at
concurrency 1,000 (+156.8%, approximately 2.57 times the total). Writes per
execution increased from approximately 0.480 MB to 0.719 MB for Pygent, while
LangGraph remained near 0.275 MB and 0.280 MB. The result is consistent with
write amplification in Pygent's journal or transaction coordination under
higher concurrency. LangGraph was materially more efficient on disk writes.

## Scope and method

- Each request required three sequential tool calls followed by one final
  answer. Returning only a final answer did not qualify as success.
- The concurrency-200 workload submitted 200 requests simultaneously. Each
  repetition produced 800 model HTTP round trips and 600 tool executions.
- The concurrency-1,000 workload submitted 1,000 requests simultaneously. Each
  repetition produced 4,000 model HTTP round trips and 3,000 tool executions.
- Every accepted sample ran in a fresh process. High-load values are medians of
  three repetitions. Capacity-confirmation failures that were run once are
  identified separately and are not presented as three-repetition results.
- The local OpenAI-compatible mock was excluded from framework resource totals.
  Removing provider latency and variance makes framework overhead more visible.
- CPU is cumulative process-tree CPU time, not CPU percentage. RSS is peak
  process-tree resident memory. Disk write is cumulative process-tree write I/O.
  Framework-owned services and child processes were included.
- High-load runs used a 4 GB RSS guard. Service requests used a 180-second
  client deadline; process-level capacity confirmation used a separate bounded
  interval of 300 to 600 seconds.
- Qualification required every request in every accepted repetition to
  succeed, the exact tool sequence `[1, 2, 3]` to be validated, and no process
  failure, deadline, or resource-guard termination.

These measurements came from a dedicated cross-framework harness, not Pygent's
native load system. The workloads serve different purposes, so this report
should not be used as the regression baseline for Pygent's native benchmarks.

## Why the lanes are separate

| Lane | What is measured | Appropriate interpretation | Unsupported interpretation |
|---|---|---|---|
| Agent loop | In-process model/tool loop without a durable runtime | Baseline loop overhead | Cost of durable recovery |
| Durable execution runtime | Official journal, checkpoint, snapshot, or workflow/step persistence enabled | Relative overhead of persistent execution paths | Identical persistence semantics |
| Service/deployment runtime | HTTP transport, session management, routing, lifecycle, service processes, and configured persistence | Resource use and throughput of a complete service surface | Direct ranking against in-process durable step runtimes |

Pi Agent Core and the raw OpenAI SDK did not provide an official runtime path
for this test and therefore appear only in the loop lane. AgentScope Runtime,
Agno AgentOS, and Hayhooks include service and deployment overhead, so they are
reported separately. Conversely, low overhead from Pygent direct or
runtime-disabled execution is not evidence about durable runtime cost.

## Version matrix

| Lane | Framework or component | Tested version and mode |
|---|---|---|
| Durable | Pygent | 0.3.10, `DurabilityPolicy.PREFERRED`, SQLite journal |
| Durable | LangGraph | 1.2.10, `langgraph-checkpoint-sqlite` 3.1.1, `AsyncSqliteSaver` |
| Durable | Mastra | 1.64.0, `@mastra/libsql` 1.22.3, Durable Agent with LibSQL |
| Durable | OpenAI Agents with DBOS | OpenAI Agents 0.22.0, DBOS 2.31.0, `dbos-openai-agents` 0.3.0, SQLite |
| Durable | Vercel WorkflowAgent | `@ai-sdk/workflow` 1.0.70, Workflow SDK 4.8.5 Local World |
| Service | Agno AgentOS | Agno 3.0.6, SQLite events, `tool-batch` checkpoint |
| Service | AgentScope Runtime | Runtime 1.1.6.post2, AgentScope 2.0.7.post1 |
| Service | Hayhooks | Hayhooks 1.24.0, Haystack 3.1.1 |
| Loop | Pygent direct | 0.3.10 |
| Loop | Vercel AI SDK | 7.0.58 |
| Loop | Pi Agent Core | 0.73.1 |
| Loop | OpenAI SDK | 2.53.0 |
| Loop | OpenAI Agents | 0.19.4 |
| Loop | LangChain | 1.3.14 |
| Loop | LangGraph | 1.2.10 |
| Loop | Mastra | 1.57.0 |
| Loop | AgentScope | 2.0.6 |
| Loop | Agno | 2.8.7 |
| Loop | Microsoft Agent Framework core | 1.13.0 |
| Loop | Haystack | 3.0.0 |
| Loop | Pydantic AI | 2.27.0 |
| Loop | Strands | 1.51.0 |
| Loop | smolagents | 1.26.0 |
| Loop | LlamaIndex | 0.14.23 |
| Loop | Qwen-Agent | 0.0.34 |
| Loop | Google ADK | 2.6.3 |
| Loop | OpenJiuwen | 0.1.16 |
| Loop | OpenHands SDK | 1.41.0 |

`@ai-sdk/workflow` 2.0.24 could not be installed because its published
dependency graph requested the unavailable `@workflow/nest@5.0.0-beta.48`.
Version 1.0.70 was the newest installable release in the test environment.

## Scoring method

The published composite scores can be reproduced as follows. Durable and loop
configurations are scored only against qualifying candidates in the same lane
and at the same load. Run time, P95 latency, CPU, RSS, and disk writes are all
lower-is-better metrics with equal 20% weights. For metric \(m\):

```text
metric_score(x) = 100 × (max(m) - x) / (max(m) - min(m))
overall_score = arithmetic mean of the five metric scores
```

Throughput and thread count are included for interpretation but are not scored
again. The method reproduces Pygent's scores of 98.6 at concurrency 200 and
80.0 at concurrency 1,000. At concurrency 1,000, Pygent received the highest
score on four metrics and the lowest score on disk writes. Scoring uses
unrounded medians; calculations made from displayed values can differ by 0.1.

Min-max scores depend on both the candidate set and outliers. For example,
Mastra's write volume expands the disk-write range at concurrency 200. Scores
are therefore meaningful only within the same table. They should not be
compared across loads, lanes, or changed candidate sets, and they should not
replace inspection of the underlying measurements.

## Results at concurrency 200

### Durable execution runtime

| Rank | Official persistent mode | Success | Run s | req/s | P95 s | CPU s | RSS MB | Write MB | Peak threads | Score |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Pygent 0.3.10 preferred | 600/600 (3/3 runs) | 3.828 | 52.2 | 3.758 | 4.906 | 102.2 | 95.958 | 23 | 98.6 |
| 2 | LangGraph 1.2.10 with SQLite | 600/600 (3/3 runs) | 10.844 | 18.4 | 10.814 | 11.922 | 169.3 | 55.074 | 37 | 85.5 |
| 3 | OpenAI Agents 0.22.0 with DBOS 2.31.0 | 600/600 (3/3 runs) | 38.844 | 5.1 | 38.059 | 42.062 | 220.0 | 68.876 | 639 | 35.0 |
| 4 | Mastra 1.64.0 durable with LibSQL | 600/600 (3/3 runs) | 22.699 | 8.8 | 22.453 | 23.266 | 628.1 | 621.132 | 32 | 28.4 |

Pygent recorded the lowest run time, P95, CPU, and RSS. LangGraph wrote less
data. DBOS used less memory and disk than Mastra but reached 639 peak threads.
Mastra completed faster than DBOS, while its RSS and write volume reduced its
equal-weight composite score.

Vercel WorkflowAgent did not qualify. Two runs completed 200/200 requests, with
a 147.292-second median run time, 262.672 CPU-seconds, and approximately 3.89 GB
RSS. The third run crossed the 4 GB RSS guard. The service started in about 0.5
seconds, so the observed failure was associated with memory growth under load,
not the earlier startup race.

### Service/deployment runtime

This table is not scored against the durable table.

| Runtime | Success | Run s | P95 s | CPU s | RSS MB | Write MB | Peak threads | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Agno AgentOS 3.0.6 | 600/600 (3/3 runs) | 11.041 | 10.794 | 14.953 | 254.3 | 37.742 | 40 | Qualified |
| AgentScope Runtime 1.1.6.post2 | 0/600 (0/3 runs) | 182.953 | 182.824 | 13.031 | 228.7 | 0.851 | 39 | Reached the 180-second request deadline in every run |
| Hayhooks 1.24.0 | 0/600 (0/3 runs) | 181.896 | 181.808 | 10.766 | 206.5 | 0.857 | 53 | Reached the 180-second request deadline in every run |

Agno was the only qualifying service runtime at this load. The test used its
working `tool-batch` checkpoint option. The publicly exposed
`checkpoint="tools"` option raised `NotImplementedError` at startup and was not
used.

### Agent-loop baseline

This table describes loop overhead without a durable runtime.

| Rank | Framework | Run s | req/s | P95 s | CPU s | RSS MB | Write MB | Score |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Vercel AI SDK | 0.760 | 263.3 | 0.735 | 1.328 | 164.5 | 0.026 | 92.2 |
| 2 | Pi Agent Core | 1.162 | 172.1 | 1.157 | 2.070 | 154.1 | 0.026 | 91.1 |
| 3 | Pygent direct | 1.234 | 162.1 | 1.213 | 2.125 | 94.1 | 0.029 | 90.0 |
| 4 | Mastra | 2.063 | 96.9 | 2.045 | 4.023 | 406.4 | 0.026 | 72.2 |
| 5 | AgentScope | 4.427 | 45.2 | 3.807 | 6.516 | 171.8 | 0.033 | 66.7 |
| 6 | OpenAI SDK | 4.479 | 44.7 | 3.823 | 5.883 | 148.0 | 0.036 | 61.7 |
| 7 | OpenAI Agents | 3.952 | 50.6 | 3.911 | 6.531 | 171.0 | 0.035 | 60.6 |
| 8 | LangGraph | 3.772 | 53.0 | 3.736 | 7.031 | 175.7 | 0.035 | 57.2 |
| 9 | Agno | 5.118 | 39.1 | 4.182 | 7.094 | 158.8 | 0.034 | 52.8 |
| 10 | Haystack | 5.161 | 38.8 | 4.273 | 6.812 | 173.6 | 0.034 | 50.0 |
| 11 | Microsoft Agent Framework | 5.076 | 39.4 | 4.322 | 6.945 | 148.8 | 0.036 | 49.4 |
| 12 | Pydantic AI | 4.465 | 44.8 | 4.394 | 6.812 | 192.2 | 0.035 | 45.6 |
| 13 | LangChain | 4.440 | 45.0 | 4.406 | 8.062 | 178.0 | 0.035 | 41.7 |
| 14 | smolagents | 6.187 | 32.3 | 4.565 | 8.562 | 129.1 | 0.377 | 37.8 |
| 15 | Google ADK | 9.849 | 20.3 | 9.838 | 10.859 | 492.3 | 0.030 | 30.0 |
| 16 | OpenJiuwen | 10.723 | 18.7 | 9.778 | 13.844 | 297.1 | 8.222 | 16.7 |
| 17 | Strands | 35.355 | 5.7 | 34.257 | 36.531 | 859.9 | 0.031 | 15.6 |
| 18 | OpenHands | 31.226 | 6.4 | 27.643 | 34.594 | 338.1 | 0.409 | 10.0 |
| 19 | Qwen-Agent | 13.375 | 15.0 | 11.427 | 70.734 | 790.5 | 0.104 | 8.9 |

Pygent direct ranked third and recorded the lowest RSS. Vercel AI SDK had the
highest loop-lane composite score. These results do not establish that either
Vercel AI SDK or Pi Agent Core provides a lower-cost durable runtime.

## Results at concurrency 1,000

### Durable execution runtime

| Rank | Official persistent mode | Success | Run s | req/s | P95 s | CPU s | RSS MB | Write MB | Peak threads | Score |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Pygent 0.3.10 preferred | 3,000/3,000 (3/3 runs) | 23.070 | 43.3 | 22.603 | 23.453 | 211.1 | 718.838 | 23 | 80.0 |
| 2 | LangGraph 1.2.10 with SQLite | 3,000/3,000 (3/3 runs) | 50.540 | 19.8 | 50.288 | 42.188 | 264.3 | 279.974 | 33 | 69.9 |
| 3 | OpenAI Agents 0.22.0 with DBOS 2.31.0 | 3,000/3,000 (3/3 runs) | 86.767 | 11.5 | 84.909 | 84.719 | 374.5 | 352.741 | 2,434 | 16.7 |

Pygent again recorded the lowest values on four metrics and the highest disk
write total. DBOS completed reliably but reached 2,434 peak threads, which
requires separate deployment assessment.

Mastra did not qualify in any of its three runs. Each process exited after 65
to 73 seconds with Node exit code 13 (`unsettled top-level await`) before
producing an adapter result. Vercel WorkflowAgent also did not qualify: its
capacity-confirmation run crossed the 4 GB RSS guard before producing a result.

### Service/deployment runtime

| Runtime | Success | Run s | P95 s | CPU s | RSS MB | Write MB | Peak threads | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Agno AgentOS 3.0.6 | 3,000/3,000 (3/3 runs) | 60.788 | 48.657 | 81.359 | 285.0 | 189.232 | 40 | Qualified |
| AgentScope Runtime 1.1.6.post2 | 0/1,000 (0/1 run) | 195.719 | 195.646 | 36.422 | 238.6 | 4.221 | 38 | Reached the 180-second request deadline |
| Hayhooks 1.24.0 | 0/1,000 (0/1 run) | 197.180 | 197.097 | 34.984 | 220.5 | 4.236 | 55 | Reached the 180-second request deadline |

AgentScope and Hayhooks recorded relatively low CPU and write totals. Their
primary observed symptom was waiting or throughput collapse rather than CPU
saturation. Because both repeatedly reached the deadline at concurrency 200,
one capacity-confirmation run was retained for each at concurrency 1,000.

### Agent-loop baseline

This batch retested the leading candidates. All seven listed entries completed
3,000/3,000 requests. Scores were recalculated across these seven candidates
and are not directly comparable with the 19-entry concurrency-200 scores.

| Rank | Framework | Run s | req/s | P95 s | CPU s | RSS MB | Write MB | Peak threads | Score |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Vercel AI SDK | 3.901 | 256.3 | 3.846 | 4.266 | 390.9 | 0.129 | 16 | 81.7 |
| 2 | Pygent direct | 8.398 | 119.1 | 8.203 | 11.406 | 195.4 | 0.140 | 22 | 73.3 |
| 3 | Pi Agent Core | 4.393 | 227.6 | 4.371 | 6.219 | 417.0 | 0.129 | 16 | 68.3 |
| 4 | OpenAI SDK | 28.171 | 35.5 | 27.318 | 24.625 | 249.6 | 0.147 | 41 | 50.0 |
| 5 | AgentScope | 31.461 | 31.8 | 30.041 | 30.359 | 286.0 | 0.146 | 34 | 40.0 |
| 6 | LangGraph | 33.067 | 30.2 | 32.858 | 32.750 | 292.1 | 0.147 | 45 | 20.0 |
| 7 | OpenAI Agents | 37.503 | 26.7 | 37.316 | 36.828 | 279.7 | 0.147 | 48 | 16.7 |

Pygent direct ranked second and recorded the lowest RSS. Vercel AI SDK recorded
lower run time and CPU.

## Qualification, retries, and excluded results

| Configuration | Load | Observed result | Treatment |
|---|---:|---|---|
| Pygent, LangGraph, and DBOS durable | 200 and 1,000 | 3/3 runs completed all requests with exact tool counts at both loads | Ranked |
| Mastra durable | 200 | 3/3 runs, 600/600 requests | Ranked |
| Mastra durable | 1,000 | 3/3 runs exited with Node code 13 and no adapter result | Excluded |
| Vercel WorkflowAgent | 200 | Two 200/200 runs; third run crossed the 4 GB RSS guard | Entire configuration excluded; successful runs were not substituted |
| Vercel WorkflowAgent | 1,000 | Capacity-confirmation run crossed the 4 GB RSS guard | Excluded |
| Agno AgentOS | 200 and 1,000 | 3/3 runs completed all requests at both loads | Reported only in the service lane |
| AgentScope Runtime and Hayhooks | 200 | Each reached the 180-second request deadline in 3/3 runs, 0/600 requests | Excluded |
| AgentScope Runtime and Hayhooks | 1,000 | Each reached the deadline in its single capacity-confirmation run, 0/1,000 requests | Excluded |
| LlamaIndex loop | 200 | All three runs reached 4,097–4,102 MB before completion | Excluded; no older value substituted |
| Strands loop | 200 | First two runs completed 200/200; third completed 195/200; a clean replacement completed 200/200 | Failed run retained; three qualifying samples used |
| Mastra loop | 1,000 | Three runs completed 574/1,000, 784/1,000, and 723/1,000; backlog-2,048 confirmation completed 534/1,000 | Excluded due to `ECONNREFUSED` failures |

## Pygent observations and limitations

Observed strengths: Pygent's durable path recorded the lowest wall/P95, CPU,
and RSS values for this short, deterministic mock workload. Its peak thread
count remained 23 when load increased from 200 to 1,000, unlike DBOS's thread
growth. Pygent direct also recorded the lowest RSS in both loop tables.

Observed cost: durable journal writes increased from 95.958 MB to 718.838 MB,
a factor of 7.49 while the request count increased by a factor of five. Writes
per execution increased by approximately 49.8%, compared with approximately
1.7% for LangGraph. The data supports the statement that Pygent had the highest
composite score in these durable tables. It does not support a claim that
Pygent led every resource metric or had the lowest overall persistence cost.

Semantic limitation: persistence behavior was not identical across frameworks.
Pygent recorded an execution journal, LangGraph and Mastra stored checkpoints
or snapshots, and DBOS recorded workflow and step state. Equal tool sequences
do not imply equal persisted content, recovery boundaries, or consistency
guarantees.

Measurement limitation: the latest refresh ran on a non-idle host. Three-run
medians are not sufficient to distinguish small regressions from all host
variance. Values should be compared within the same table and test batch;
separate A/B measurements should not replace cells in these ranking tables.

## Actionable optimization targets

Future Pygent durable optimizations should preserve recovery correctness and
exact tool-count validation as hard requirements. Suggested acceptance targets
are:

1. Reduce total writes at concurrency 1,000 from 718.838 MB to no more than
   350 MB, a reduction of at least 51.3%.
2. Keep `write_mb / successful_execution` approximately stable from
   concurrency 200 to 1,000, with no more than 10% growth. Report database, WAL,
   and other journal files separately.
3. Do not exchange disk improvements for material regressions elsewhere. At
   concurrency 1,000, CPU, RSS, run time, and P95 should each remain within 10%
   of the current baseline, with all three repetitions passing exact tool-count
   and recovery checks.
4. Add byte and transaction accounting by event type and payload type, including
   WAL bytes and checkpoint or journal bytes, before selecting an optimization.
5. Evaluate event coalescing, redundant state-update elimination, batched
   transactions, and journal-payload compression or normalization as separate
   A/B changes. Include crash-point recovery tests for each change so fewer
   required persistence boundaries cannot be mistaken for an improvement.
6. Use fresh-process, three-repetition concurrency-200 and concurrency-1,000
   runs as the optimization gate. Retain every raw repetition and interleave
   baseline and candidate runs on the same host and harness.

Compression can increase CPU, transaction batching can alter cancellation or
crash windows, and event coalescing can reduce observability. An optimization
is acceptable only if the durable contract remains intact.

## Evidence and reproducibility boundary

The report was cross-checked against a benchmark methodology document, its
published ranking graphic, and the per-run raw results for all four high-load
groups. The raw artifacts included `result.json`, `adapter.json`, stderr, and
process-tree resource measurements. They are not copied into this repository
and are therefore not represented as Pygent-native reproducibility fixtures.

No real model API was called as part of publishing this report. A separate
live benchmark is required to evaluate provider TLS and network
distance, token generation, rate limits, retries, cost, and long-duration
steady-state capacity. Production conclusions also require fault injection,
recovery-semantics validation, data-growth tests, filesystem testing, and
deployment-topology validation.
