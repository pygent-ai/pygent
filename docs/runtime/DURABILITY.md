# Runtime 持久化与恢复边界

本文定义 Runtime 在本地执行、分布式 placement 与 durable recovery 之间的能力边界。它从属于 [Pygent 0.2 第一原则](../FEATURES.md) 和 [Runtime 第一原则](FEATURES.md)。0.2.x 参考实现提供 `LocalRuntime`、`SQLiteHistoryStore`、HTTP Worker/SSE、稳定 Execution 恢复和受管 Model/Tool effect 重放；本页同时明确这些能力不能推出任意 coroutine 或外部副作用 exactly-once。

## 能力分级

Runtime 必须显式声明自己支持的能力，调用方不得从“分布式”推断“可持久恢复”。

| 能力 | 最低保证 | 不自动保证 |
|---|---|---|
| Local execution | 在一个 Runtime 实例内执行 Root、Child、lease 与事件 | 进程故障恢复 |
| Distributed placement | Root 或声明的 Module 边界可以放置到远程 Worker | 活跃 coroutine 迁移、持久恢复 |
| Durable recovery | 从声明的 retry/checkpoint 边界创建新 attempt 或恢复状态 | 从任意 Python 源码位置 exactly-once 续跑 |

一个 Runtime 可以同时支持多个能力，但必须分别报告 capability、限制和失败语义。

## Binding eligibility 与有效能力

Runtime 实现必须在 bind/compile 阶段结合 ExecutionPlan、adapter 和部署设施验证 durable 要求。扩展 Runtime 可以提供以下策略级别；`LocalRuntime(history=...)` 当前以是否配置 durable store 显式选择能力，不伪造更强保证：

| 要求 | bind/compile 行为 |
|---|---|
| `disabled` | 不建立 durable history，不获得故障恢复保证 |
| `preferred` | 尝试启用请求的恢复级别；无法满足时可以降级，但必须在绑定结果中明确报告 |
| `required` | 任一必需 capability、受管 adapter 或兼容策略缺失时拒绝绑定 |

Binding 检查结果至少应报告：实际 placement/recovery 级别、checkpoint/replay policy、外部副作用保证、事件重连能力、容量有效作用域、未满足项和降级原因。`required` 不允许静默从 deterministic replay 降级为整 Module retry，也不允许把未受管第三方 `await`、未知副作用或不可移植计划伪装成已验证 durable。

普通 Python 行为无法仅靠类型系统完全证明安全。Durable Runtime 必须结合显式声明、ExecutionPlan capability、受管 Model/Tool/Activity adapter、bind-time validation 和运行时 non-determinism 检测；无法验证的部分应拒绝、隔离为可重试 Module 边界，或在 `preferred` 模式下明确降级。绑定成功只证明所报告的有效能力，不代表任意用户代码自动获得 exactly-once 或任意位置恢复。

### Module 恢复资格声明

配置 SQLite 或其他 HistoryStore 只证明 Runtime 能持久保存记录，不证明用户 `forward()` 可以安全重放。每个参与 durable 图的本地 Module 都必须通过严格不可变的 `ExecutionRequirements` 同时声明：

- `recovery_safety=RecoverySafety.MODULE_BOUNDARY_RETRY`：允许 Runtime 使用相同边界输入创建新的 Module attempt；
- `effect_safety=EffectSafety.EFFECT_FREE`：Module 本身不执行非确定性操作或外部副作用；或者
- `effect_safety=EffectSafety.MANAGED_EFFECTS`：所有需要重放的非确定性结果和副作用都经过 Runtime 受管 effect 边界。

默认值 `UNDECLARED` 表示普通 Module **不承诺可重放**。声明会进入 ExecutionPlan 的模块 metadata、配置摘要和 `graph_hash`。组合 Module 的声明不能替代 Child 声明；Runtime 在 bind 时逐节点验证整张图。任意未声明恢复资格的节点、任意 `effect_safety=UNDECLARED` 的节点、以及无法验证的 trusted callback/第三方 `await` 都使 `required` 绑定失败。

`preferred` 可以继续建立本地执行与 durable 事件记录，但必须在 `DurabilityReport.recovery_undeclared_modules`、`effect_unverified_modules` 和 `degraded_reasons` 中列出具体路径，并报告 `recovery_level="none"`；不能仅因为 SQLite 存在就报告 `module_boundary_retry`。此时 `checkpoint_policy="run_history_only"`，`runtime.recover()` 也必须拒绝。`disabled` 不写 durable history，Module 声明本身不会偷偷启用恢复。

框架内置 `ModelCallLayer` 把 Provider 调用作为受管 effect，`ReActLayer` 只组合 Child，`ToolCallLayer` 把工具执行作为受管 effect，因此它们可以做相应声明。配置任意 `authorization_adapter` 的 ToolCallLayer 会降为 effect 未验证；若要获得 required durability，应使用同样声明资格的应用授权 Module，或未来具有显式 durable adapter capability 的受信适配器。声明只描述可验证的恢复条件，不提升外部副作用为 exactly-once。

## 三种继续执行

以下术语不得混用：

### RESUME

RESUME 是调度恢复。原 owner Task、Python coroutine 栈和局部变量仍然存在；执行流只是在受管等待期间释放 execution lease，完成等待后重新参与公平调度。

RESUME 不跨越 Worker 或进程故障，也不读取持久化 checkpoint。

### RETRY

RETRY 在 Root、Module 调用或显式 Step 边界创建新的 execution attempt。旧 attempt 不再继续；新 attempt 使用持久化或重新构造的边界输入重新执行，因此可能重放 Model、Tool、`emit()` 和用户逻辑。

### CHECKPOINT RESTORE

CHECKPOINT RESTORE 从已提交的持久化状态重建执行边界。checkpoint policy 必须声明恢复单位、保存状态、兼容版本和恢复后需要重放的范围。没有显式状态机或 checkpointable contract 时，不得承诺恢复任意 `forward()` 局部变量或 coroutine continuation。

## 默认 checkpoint 边界

Pygent 普通 Module 的默认可恢复边界只能是：

- Root 调用开始前；
- Module 调用开始前；
- Module 调用成功且其输出已经可靠提交后；
- 未来显式 Step/Checkpoint API 声明的边界。

Message 与 Context 可以构成边界数据，但不足以描述完整执行状态。普通局部变量、第三方 `await`、打开的连接、锁、iterator、文件描述符和未提交事件都不是默认 checkpoint 内容。

如果未来提供 `CheckpointableModule` 或显式 Step API，该能力必须是普通 `forward()` 之上的附加契约，不能假设任意 Python coroutine 可自动序列化。

0.2.x 参考实现不要求用户显式声明 Step，但要求 Module 显式声明上述边界重试与 effect 安全资格：Runtime 在自身可观察的 Execution、Model、Tool 和事件边界记录 durable history，并通过重新执行合格的 `forward()`、返回已提交 effect 结果的方式重建局部状态。该方案仍不序列化 coroutine，也不恢复任意 Python 指令；详细设计见 [透明恢复与确定性重放](REPLAY.md)。

## 故障矩阵

| 故障位置 | 可恢复状态 | 允许行为 |
|---|---|---|
| 逻辑 Execution 尚未 durable commit | 仅有进程内 `SUBMITTING` snapshot | 在 deadline 内提交；失败则原子终结，进程崩溃不承诺可恢复 |
| Execution 已提交、admission 未完成 | Root 输入、计划身份与 admission journal | 回滚部分资源并终结，或由新 owner 创建新 attempt 继续 admission |
| admission 已提交、`forward()` 未开始 | Root 输入、计划身份与精确模型/资源 pins | 在新 Worker 创建新 attempt |
| `forward()` 普通本地计算中 | 最近已提交边界 | 从该边界重试，丢弃未提交局部状态 |
| Child/Model/Tool 等待中 | 最近边界及已记录调用状态 | 查询结果、重试或失败，取决于 adapter 能力 |
| `wait_external()` 等待中 | 外部 waiter 可能仍存在，但本地 coroutine continuation 不可恢复 | 注销或过期旧 waiter，从最近已提交边界重试，或等待反馈后创建新 Execution |
| Provider 已接收请求、结果未提交 | 请求可能已经产生费用或结果 | 允许重复请求；必须关联和去重事件/usage |
| Tool 副作用已发生、结果未提交 | 外部状态可能已改变 | 只能依赖幂等键、去重存储或补偿策略 |
| 最终 Context 生成、业务 Store 未提交 | Runtime 与业务状态不一致 | 使用 finalization/outbox 或由业务提交幂等处理 |
| 业务 Store 已提交、Runtime 未确认 | 结果可能已对外可见 | 按 run identity 查询并确认，不得盲目重放 |

## 身份与幂等

Durable Runtime 至少需要区分：

- `request_id`：一次传输或 SDK 请求尝试的身份，用于 tracing；客户端重试时可以变化；
- `idempotency_key`：调用方提供的逻辑业务操作身份，同一操作的重试保持不变；
- `execution_id`：逻辑 Execution 身份，跨恢复保持稳定；
- `call_id`：逻辑 Module、Model 或 Tool 调用身份，跨同一逻辑调用的重试保持可关联；
- `attempt_id`：一次实际执行尝试，每次 retry 都不同；
- `event_id` 或稳定事件游标：用于流式重连和事件去重；
- `binding_id`、`graph_hash` 和代码制品版本：用于恢复兼容性检查；
- 外部副作用 idempotency key：通常由 run/call identity 派生。

`request_id` 不参与业务去重。要求客户端重试去重、durable recovery 或 finalization 协调时，调用方必须提供 `idempotency_key`；Runtime 在 `(binding_id, tenant/identity, idempotency_key)` 作用域内执行 get-or-create。相同 key、输入摘要和 ExecutionPlan 的重复请求关联到同一 `execution_id`；相同 key 携带不同输入、计划或关键策略时必须返回幂等冲突，不能覆盖旧 Execution。Runtime 生成 `execution_id` 后，后续 RETRY、restore、事件重连和最终确认都继续使用该逻辑身份，仅创建新的 `attempt_id`。

Runtime 默认只能提供 at-least-once 执行语义。只有底层 Provider、Tool 和业务 Store 都参与同一幂等或事务协议时，才可以对特定边界声明更强保证；不得笼统宣称 exactly-once Agent 执行。

每一个 attempt（包括首次执行）都必须先为逻辑 Execution 原子获取带 TTL 与单调 fencing token 的 owner claim；同一时刻只有一个 attempt owner 可以执行，heartbeat 丢失会取消旧 owner。恢复不是特殊 owner 模式，而是取得新 claim 并创建新 `attempt_id`。claim 只避免并发双执行，不提升外部副作用保证；外部提交仍必须按 EffectSpec、幂等键或资源 fencing 处理。

附着与恢复严格分离。`get_execution_handle(execution_id)` 只读取 snapshot、outcome 和 journal，或向当前 owner 写入取消请求；它不得取得 claim 或启动 `forward()`。只有显式 `recover()` 能在验证 ExecutionPlan、恢复资格和原 admission manifest 后取得新 claim。

## Model、Tool 与事件重放

- Model 调用可能在响应丢失后重复产生费用和非确定性输出；Runtime 应记录 route、attempt、请求关联身份和已提交结果。
- 延迟模型组还必须把 admission 时选择的具体部署 manifest 与不可变资源 revision 作为恢复事实保存；已提交 effect 可以脱离 live invoker 重放，尚未提交的模型工作只能重新取得原 revision，不能改用当前最新部署。完整契约见 [延迟与动态模型组规范](../llm/DYNAMIC_MODEL_GROUP_SPEC.md)。
- Tool 调用必须声明幂等、可查询、可补偿或不可安全重试；不可安全重试的调用在状态不明时必须进入人工或业务决策状态。
- `emit()` 事件必须携带稳定事件身份。新 attempt 重放相同逻辑事件时，事件系统必须能够去重或明确标记为新 attempt。
- 流式 token 在故障边界前可能已被客户端观察但尚未成为最终结果；重连协议必须说明是否重放、截断或从持久游标继续。

## 原子终结与订阅关闭

History backend 必须提供单个 `finalize_execution(...)` 事务，在一次提交中完成：追加尚未提交的 terminal span events、追加唯一 Execution terminal event、冻结 `ExecutionOutcome`、更新 terminal snapshot、关闭 active attempt/owner claim，并写入 `terminal_sequence`。事务提交前 Execution 仍处于 `FINALIZING`；提交后 journal、snapshot 与 outcome 对终态达成一致。

`terminal_sequence` 是订阅结束的唯一依据。live observer 与断线重连 observer 都必须持续读取，直到自己的 cursor 已经交付该 sequence；不得因为并发状态查询先读到 terminal status 而停止。Local 与 durable backend 服从同一规则，只允许存储事务实现不同。

准入也必须有明确提交点。模型 pins、资源 leases、容量 tickets、attempt claim 与 history 状态由一个 admission transaction 协调；取消、deadline 或任一步骤失败时按逆序释放，并把失败写为同一逻辑 Execution 的结构化 outcome。崩溃恢复只能采用已提交 manifest，不能保留孤儿 pin 或重新读取 current profile。

## 代码与 Binding 兼容

恢复前必须校验 checkpoint 与当前 ExecutionPlan 的兼容性，至少包括 Runtime API 版本、`graph_hash`、代码制品 digest、输入输出 schema、serializer 和 checkpoint policy。

默认情况下，hash 或 schema 不兼容必须拒绝自动恢复。迁移只能通过显式、版本化的 checkpoint migration 完成，不得让新代码直接解释未知旧状态。

## 独立生命周期任务

结构化 Child 始终属于 Parent，Parent 终止前必须 cancel/join 未完成 Child。独立生命周期任务不得通过 `detached=True` 绕过这一规则。普通工具与 Agent-backed Tool 都可以 detach，但 Runtime 必须为调用创建独立 ToolTask 身份与 admission；需要 durable recovery 时，由独立 Job 承载该 ToolTask。转换完成后它不再是 Child，Parent 只保留稳定引用，但新任务仍受 Binding 与资源治理。

进程内 ToolTask 可以只具有 Runtime 生命周期内的独立身份，不自动承诺故障恢复。需要 durable recovery 的 Job 或 Workflow 必须拥有独立身份、持久化 admission、状态机、取消和保留策略。Parent 只能获得 `JobRef` 或等价引用；创建成功的判定点是 Job 已被可靠持久化，而不是后台 coroutine 已在本进程启动。

参考实现把 durable detach 保存为独立 Job 记录。该记录原子携带 Job/ToolTask 两个稳定 ID、可移植请求、Binding/ExecutionPlan/resource/capability 身份与 attempt，不携带 Python callback 或 executor。重启恢复必须通过 `LocalRuntime.recover_tool_jobs(compatible_bound_module)`：Runtime 重新解析部署 registry、校验版本和 capability，并重新获取 Binding Tool capacity/resource gate。缺少兼容声明时 Job 保持可查询但不得执行。崩溃时已经 `RUNNING` 的非幂等写入或外部副作用进入 `unknown`；只有固有幂等，或已提交稳定 idempotency key 的调用，允许新 attempt 重放。

Job admission 还具有稳定 logical key，由逻辑 `execution_id`、Root、Module path、该 Module 在 Execution 内的确定性 occurrence、Tool `call_id` 与幂等身份派生。occurrence 随相同控制流在 recovery replay 中从零重建：同一 occurrence 重新取得原 Job，不同轮次或重复 Module 调用即使复用 `call_id` 也保持独立；不得用随机数或仅用参数 hash 代替调用身份。SQLite 对该 key 执行原子 get-or-create：即使 Parent 在 Job commit 与自身 Execution result commit 之间崩溃，Parent recovery 也只能重新取得原 `job_id/task_id`，不能创建第二个独立 Job。

普通 managed effect 同样先写 `started`，再执行 operation，最后写 `completed + result`。恢复遇到 `started` 时按持久化的 `EffectSpec` 判定：只读、固有幂等或带稳定 key 的 effect 仅在显式 `REPLAY_SAFE` 下重放；非幂等 write/external 标记 `unknown` 并 fail closed。不能因 result 尚未写入就推断外部副作用未发生。

## 业务 Store 与最终提交

Context 和业务 Store 仍由应用服务拥有，但 Durable Runtime 必须明确最终结果与业务提交之间的协议。可选模式包括：

- 明确采用 at-least-once，由业务 Store 按 `execution_id` 和 revision 幂等提交；
- transactional outbox；
- `PREPARED -> application commit -> COMMITTED -> runtime finalized` 的确认协议；
- 领域专用补偿流程。

Runtime checkpoint 不得被描述为业务 source of truth，业务 Store 也不得被假定包含完整执行 checkpoint。

## 当前不承诺的能力

- 任意 Python `await` 后的透明持久恢复；
- 通过 `wait_external()` 跨进程保留当前 Agent coroutine、调用栈或局部变量；
- 活跃 coroutine 或调用栈跨 Worker 迁移；
- 不依赖幂等、事务或补偿的 exactly-once 外部副作用；
- 通过 Context 单独恢复完整 Execution；
- 通过 detached Child 表达 durable Job；
- 在代码或 schema 不兼容时自动猜测迁移。
