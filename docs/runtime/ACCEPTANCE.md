# Pygent 0.2.x 第一原则验收矩阵

本页把第一原则映射到可重复执行的验收证据。它不是新的契约；发生冲突时仍按
[文档层级](README.md)服从总第一原则、模块第一原则与 SDK。

| 契约切片 | 主要可执行证据 |
| --- | --- |
| Module 定义冻结、direct/managed 配置一致、plan drift fail-closed | `tests/core/test_module_direct_execution.py`、`tests/runtime/test_definition_drift.py`、`tests/runtime/test_execution_plan.py` |
| 领域 Message 的严格 JSON、wire 往返、approval/handoff/termination 组合 | `tests/integration/test_module_context_contract.py`、`tests/runtime/test_wire_codec.py` |
| Module 自由参数/结果、RecurrentModule 显式 state、本地 direct Root/Child 保持用户声明的结果类型 | `tests/core/test_module_direct_execution.py`、`tests/integration/test_public_api.py`；managed/remote call contract 仍保持既有 Message/Context 范围 |
| 基础 Context 不可变、模型投影、显式历史、slot、严格有限 JSON 与 wire 往返 | `tests/integration/test_module_context_contract.py`、`tests/common_module/test_json_values.py`、`tests/runtime/test_wire_codec.py` |
| AgentContext 子类校验、`ContextCodec.dataclass()`、具体类型保留、通用信封 discriminator、计划/Worker codec allowlist、durable history/reattach 与版本拒绝 | `tests/integration/test_agent_context.py`、`tests/runtime/test_wire_codec.py`、`tests/runtime/test_execution_plan.py`、`tests/runtime/test_http_worker.py`、`tests/runtime/test_durable_runtime.py` |
| Binding、父子 lineage、取消、deadline、live/runnable、有限队列与 RESUME | `tests/runtime/test_local_runtime.py`、`tests/runtime/test_agent_boundaries.py` |
| Model/Tool 独立共享容量、稳定资源身份、无 hold-and-wait | `tests/runtime/test_shared_capacity.py` |
| deployment Execution/Model/Tool 跨 Runtime 与 SQLite 协调、FIFO、取消、TTL/fencing | `tests/runtime/test_shared_capacity.py` |
| ExecutionPlan 稳定身份、placement、部署 manifest/artifact resolver 与 durability capability | `tests/runtime/test_execution_plan.py`、`tests/runtime/test_http_worker.py`、`tests/runtime/test_durability_eligibility.py` |
| SQLite effect 的 `started/completed/unknown`、安全重放、任务恢复、HTTP Worker/SSE | `tests/integration/test_durable_effect_replay.py`、`tests/runtime/test_public_infrastructure.py`、`tests/runtime/test_durable_runtime.py`、`tests/runtime/test_sqlite_history.py`、`tests/runtime/test_http_worker.py` |
| durable Tool Job occurrence 身份、原子 admission、owner claim、恢复资格与计划/版本/capability 校验 | `tests/runtime/test_durable_tool_jobs.py`、`tests/runtime/test_durable_tool_tasks.py`、`tests/runtime/test_http_worker.py` |
| REQUIRED durability 的逐节点 recovery/effect 资格与明确降级报告 | `tests/runtime/test_durability_eligibility.py` |
| 内置与用户 Infrastructure Module 共用 effect/resource/resolver SPI | `tests/runtime/test_public_infrastructure.py` |
| LLM route/retry/fallback、托管有限 deadline、stream/usage/工具定义投影 | `tests/llm/`、`tests/runtime/test_execution_plan.py` |
| Provider 取消清理有界、`OUTCOME_UNKNOWN` fail-closed、client 隔离与显式取消/managed deadline 终态 | `tests/llm/test_invoker.py`、`tests/performance/test_model_retry.py`、`tests/runtime/` |
| Tool 定义、授权、执行、顺序、detach、副作用不确定性 | `tests/tool/` |
| MCP stdio/SSE 适配与错误边界 | `tests/mcp/` |
| ReAct 三重预算、工具循环、draft 与 reviewed 历史 | `tests/agent/` |
| 共享 Agent 并发无状态、真实 Agent 组合及随机无效密钥 fallback | `tests/examples/test_live_agent.py`、`tests/runtime/test_agent_boundaries.py` |

## 固定门禁

```bash
uv run pytest -q
uv run ruff check .
uv run mypy src examples
uv build
```

发布验证还应在干净虚拟环境安装构建出的 wheel，并运行发布版示例。CI 在 Windows 与
Linux 上重复核心门禁。不得用 `xfail` 隐藏尚未实现的第一原则。

## 可选在线验收

`examples.live_agent.benchmark` 从环境读取 `GLM_API_BASE`、`GLM_API_KEY` 与 `GLM_MODEL_NAME`，为
首路由生成随机无效密钥，并以配置密钥作为 fallback。程序只输出聚合延迟、吞吐、
fallback、usage、错误分类和上下文隔离数据，不输出端点、模型名或密钥。

```bash
uv run --env-file .env python -m examples.live_agent.benchmark --concurrency 1 --requests 8
uv run --env-file .env python -m examples.live_agent.benchmark --concurrency 2 --requests 8
uv run --env-file .env python -m examples.live_agent.benchmark --concurrency 4 --requests 8
uv run --env-file .env python -m examples.live_agent.fallback_probe
```

在线验收受外部端点可用性影响。HTTP 5xx、网络故障或配额错误必须被如实记录，不能用
Mock 成功替代；无网络的确定性回归由 `tests/examples/test_live_agent.py` 覆盖。
若上游兼容端点本身不校验 Authorization，`fallback_probe` 会在临时
`127.0.0.1` HTTP 边界拒绝随机 Key，并且只把配置 Key 转发给真实模型；成功验收必须同时
观察到 authentication failure、fallback success、真实模型结果与 Tool 结果。

## 明确的能力边界

- direct execution 不提供跨 Root 的框架级容量治理或持久恢复。
- durable Runtime 只在声明并验证过的 effect/checkpoint 边界恢复，不序列化任意 Python coroutine。
- 外部副作用是否 exactly-once 取决于外部系统的幂等契约；框架不会静默作出该承诺。
- `Context.tools` 只控制模型可见性，不构成授权；授权由应用 Module 或受信 executor 决定。
