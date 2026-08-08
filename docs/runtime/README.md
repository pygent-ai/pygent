# Runtime

阅读顺序：

1. [第一原则](FEATURES.md)
2. [SDK 使用](SDK.md)
3. [持久化与恢复边界](DURABILITY.md)
4. [透明恢复与确定性重放草案](REPLAY.md)
5. 本文的详细契约

Runtime 把不可变 Module 图和 Binding 接入一个有界托管执行域。它负责 admission、并发、排队、父子执行、取消、deadline、事件和关闭；它不拥有 Agent、LLM、Tool 或业务状态的具体逻辑。普通 Module 可以不绑定 Runtime 而直接 `invoke()`/`stream()`；direct execution 的 Root 并发和资源生命周期由调用方管理，不获得本文件的托管保证。

Worker 的协议/registry、远程 handle、服务端、客户端和 Module target 分属独立源码模块；旧的 `runtime.worker` 聚合模块已删除。SQLite history 仍由 `_history_store.SQLiteHistoryStore`、一个连接和同一写锁维护事务边界，execution、Job、effect/checkpoint/event 方法簇分属独立模块；旧的 `runtime.history` 聚合模块同样不再存在。

Module 定义图可以共享节点；一次 Execution 中实际发生的每次调用仍形成独立的临时执行节点。因此同一 Module 被 Agent 直接调用、被 ReActLayer 调用或被重复调用时，分别拥有独立调用身份、结果、事件和取消状态。

## Binding 是可选的托管部署域

direct execution 不创建 Binding。需要托管执行时，Binding 是一组 Module/Agent 共同使用的部署策略与资源治理域，不是 Agent 的身份，也不是每个 Agent 必须单独创建的包装器。通常一个面向服务的 Module 图只创建一个 Binding：Coordinator、Worker、Reviewer、ModelCallLayer 和 ToolCallLayer 都在同一治理域中运行。

原始结构化 Child 不需要执行 `bind()`。Parent 在 `forward()` 中调用原始子 Module 时，Runtime 自动把当前 Binding、lineage、deadline、取消 scope 和容量预算传给 Child。这样 Runtime 才能看到完整执行树，并在 Parent 等待 Child 时正确完成 lease handoff。

需要独立部署边界时，组装者可以把预先创建的 `BoundModule` 作为 Child 依赖，或由 Parent Binding 的 placement policy 把某个原始 Child 固定或动态路由到其他 Runtime。跨 Runtime 只改变 Child 的执行位置和容量归属，不把它提升为新 Root；root/parent identity、deadline、取消、事件和最终结果仍属于同一结构化调用树。

只有以下边界确实需要隔离时才创建不同 Binding：

- 独立的 Execution 容量、队列或调度 SLA；
- 不同租户、权限、credential 或工具可见性；
- 不同模型/工具路由和资源策略；
- 不同服务生命周期、观测归属或发布环境。

同一个 Agent 定义可以在生产、评测或不同租户的多个 Binding 中复用；多个不同 Agent 入口也可以接入同一个 Binding。是否共享 Binding 由部署治理边界决定，不由 Python 类型或 Agent 数量决定。

## Child placement

Binding 与 Runtime 支持三种 Child 放置语义：

| 模式 | 声明方式 | 执行位置 | Binding 语义 |
|---|---|---|---|
| `inherit` | Parent 持有原始 Module | 当前 Runtime | 继承 Parent Binding |
| `pinned` | Parent 持有预绑定 BoundModule，或 Binding 指定固定目标 | 指定 Runtime/Binding | 使用显式 Child Binding |
| `adaptive` | Binding 为原始 Module 声明目标池和约束 | Runtime 调用时选择目标 | Binding 身份稳定，物理目标可变 |

`bind()` 负责建立稳定的部署身份、策略和 ExecutionPlan；placement resolver 只为一次 Child 调用选择物理执行目标。Adaptive placement 不得在每次调用时创建新 Binding，否则会产生新的调度命名空间并绕过原 Binding 的容量、观测和身份契约。

预绑定 Child 是显式硬边界。Parent Binding 可以为原始 Child 选择 `inherit`、`pinned` 或 `adaptive`，但不得静默把已经绑定到 Runtime A/Binding A 的 Child 改绑到另一个目标；冲突必须在 Parent bind/compile 阶段拒绝。

`RemoteModule(binding_ref=..., plan_id=..., graph_hash=..., required_capabilities=..., placement=...)` 用于只有远程部署契约、没有本地 Module 定义实例的场景。远程计划身份必须来自受信部署控制面并在调用端与 Worker 两侧校验；Registry 声明只是第一道筛选，每次启动前还必须请求 endpoint 的 `/health` 复核实时 capability，并由 Worker 对启动请求携带的 required capabilities 再校验。注册 target 的 placement 必须与编译声明完全一致，不能用 adaptive target 覆盖 pinned 声明。只有实时健康、属于已声明逻辑目标且满足 capability 的 endpoint 才能参与 adaptive 选择，pinned 模式不能故障转移到其他 target。预绑定 Child、RemoteModule 与 Binding placement 最终编译成同一种 Child target 描述，共享取消、deadline、事件、错误和结果语义。

### 受约束的动态 Agent 解析

Pygent 不支持绕过 Binding 和 ExecutionPlan 的开放式动态 Agent Registry：`forward()` 不能根据任意字符串发现并调用一个未在当前计划中声明的 Agent，也不能让 Registry 在运行时引入新的逻辑 Child。逻辑依赖、稳定 `binding_ref`、输入输出 schema、授权边界、容量归属和必需 Runtime capability 必须在 bind/compile 阶段确定，并进入 ExecutionPlan 的身份或兼容性检查。

这一限制只针对改变业务调用图的 Agent 查找，不禁止 Runtime 使用精确 `(definition_id, version)` 的实现注册表重建已声明 Module，也不禁止 Tool 执行注册表按已声明 `tool_id/version` 解析 executor；这两类注册表解析的是计划中已有契约，而不是新增逻辑依赖。

Registry 或服务发现可以动态解析一个已经声明的逻辑目标。它可以按健康、地域、亲和性和实时容量改变具体 Worker、endpoint 或部署副本，但不能改变逻辑 Agent、目标 Binding 契约或调用的父子身份。扩缩容和故障转移因此不要求重新生成 ExecutionPlan；更换 schema、权限边界、关键 capability 或逻辑 Agent 则必须产生新版本的 Binding/ExecutionPlan。

解析时没有合格实例必须返回明确的 unavailable 或 capability mismatch，不能退回到名称相似但未获计划授权的 Agent。确实需要调用当前计划未声明的 Agent 时，应用应先生成并验证新的 ExecutionPlan，或通过独立 Root Execution/Job 入口建立新的治理边界。

跨 Binding Child 同时受两类预算约束：Parent Binding 继续计算 lineage depth、直接 child 数、waiter 和结构化 live ownership，防止通过远程 Child 绕过执行树上限；目标 Binding 负责该 Child attempt 的 admission、queue、live/runnable lease 和资源容量。它不是目标 Binding 的 Root，也不额外消耗 Parent 的 runnable lease。

## 对象边界

| 对象 | 保存内容 | 不保存内容 |
|---|---|---|
| `Module` | 定义、配置、子 Module 与 direct-capable 本地 adapter 声明 | 当前请求、queue、lease、Task；可移植计划不得包含活连接 |
| `Binding` | 不可变部署策略、治理域身份与共享容量 key | semaphore、dispatcher、活计数器 |
| `BoundModule` | Module、Binding 与 Runtime 调度命名空间的可执行关联；既可作为 Root 入口，也可作为显式绑定的 Child 依赖 | 业务 Session 状态 |
| `Runtime` | queue、waiter、lease、owner、deadline、Task 和关闭状态 | 跨 Execution 的业务 source of truth |

同一个 `BoundModule` 可以被并发复用，所有 Root 和 Child 执行共享其 Binding 对应的 Execution 容量。在同一 Runtime 中复用同一个 Binding 身份，会复用同一个 Execution 调度命名空间；显式创建另一个 Binding 才产生独立治理域，即使两者的策略值完全相同。

## Binding 编译与 ExecutionPlan

Binding 必须为实际执行使用的 Module 图创建不可变 `ExecutionPlan` 快照。BoundModule 引用该快照；绑定后对原始 Python Module 对象的修改不得悄然改变已绑定部署。相同定义、代码制品和执行契约必须产生相同 `graph_hash`，环境标签等描述性 metadata 不参与图身份。

计划分为两级：

- 本地计划只要求 root、完整可达的无环 Module 图和 Runtime API 版本，可以由同一进程持有的 Module 对象执行；
- 可移植计划还必须声明代码制品、稳定 definition ID、输入/输出 schema 和 serializer，远程 Runtime 只能接纳自身能够解析且 `graph_hash` 校验成功的可移植计划。

`CodeArtifactSpec` 通过 package、version、content digest 和 entrypoint 标识 Worker 应加载的代码。`ModuleSpec` 除路径和子节点外，还可以携带配置引用、资源 key、placement constraint、Worker capability、retry policy 与 checkpoint policy 引用。secret、连接和 handler 不得直接进入计划；它们由 Runtime 根据稳定引用解析。

`type_name` 只用于诊断和可读性，不是远程重建 Module 的充分身份。分布式 Runtime 不得隐式 pickle Module 图，也不得仅凭 Python 类型名加载代码。计划哈希提供内容完整性，不替代代码制品签名、来源验证和策略授权。

当前实现会从 Module 图确定性编译 ExecutionPlan，并把 inherit、pinned、adaptive placement、远程 plan identity 与 capability 约束纳入图身份。正常 HTTP Worker 部署必须向 `LocalRuntime` 显式提供 `CodeArtifactSpec`、输入/输出 schema 和 serializer；这些字段都进入计划哈希，缺少任一项的本地计划会被 Worker adapter 拒绝。artifact/config/policy 的实际解析和物理 endpoint 健康信息仍由具体 Runtime 或受信部署控制面提供；运行时 resolver 只能在已编译约束内选择目标，不能引入新的逻辑 Child。

Model 和 Tool 容量既可以是 Binding-local，也可以通过稳定 `capacity_key` 映射到 Runtime 内的共享资源容量组。因此两个 Binding 的 Execution 容量默认独立，但它们可以因为使用同一模型 endpoint、credential scope、工具资源或部署容量 key 而共享 Model/Tool 计数器。

### 容量作用域

所有启用的容量策略都必须声明作用域，不能仅以 `Binding` 名称推断计数器是否跨进程共享：

| 作用域 | 计数与排队边界 | 运行要求 |
|---|---|---|
| `runtime_instance` | 当前 Runtime 实例；通常对应一个进程或 Worker 副本 | 所有 Runtime 都必须支持 |
| `deployment` | 共享同一 `binding_id` 的整个部署 | Runtime 必须提供集中 admission、分布式 permit 或等价协调能力 |
| `external_resource` | Provider、数据库、浏览器池等外部资源所有者 | 必须通过稳定 `capacity_key` 和受信 adapter 解析 |

Execution/Agent 容量必须在 Binding 创建时显式选择 `runtime_instance` 或 `deployment`。Model/Tool 的 `limited` 策略同样必须声明作用域；`passthrough` 使用 `external_resource` 作用域，表示 permit 由外部资源所有者治理，而不是一个无界的本地或部署队列。LocalRuntime 只有注入共享 `CapacityCoordinator` 后才能接受 `deployment`；没有 coordinator 时即使当前只有一个实例也必须 fail closed，不能静默改成每 Worker 独立计数。`InMemoryCapacityCoordinator` 只覆盖同进程多个 LocalRuntime；SQLite+HTTP 基线为每个 Worker 注入独立 `SQLiteCapacityCoordinator` 并指向同一可靠 SQLite 文件，其独立 `pygent_capacity_*` 表使用 FIFO waiter、可续期 lease 和 fencing token，确保有限队列、取消清理与 crash permit 过期回收；deployment `max_waiters` 也必须由该共享 owner 全局计数，不能按 Worker 相乘。多主机部署需要能兑现同一协议的外部协调器；无法共享可靠 owner 时必须拒绝 deployment scope。

SQLite permit 暴露单调 fencing token，但它只有在受保护的外部资源于提交时原子拒绝旧 token 才构成强 fence。Pygent 取消旧 Task 和回收 permit 不能撤回已经提交给数据库或远程服务的 I/O；未实现 token 校验的外部资源仍可能接受 stale owner 的迟到提交，因此不能据此承诺 exactly-once。

`deployment` 作用域限制的是逻辑部署总量；`runtime_instance` 作用域的上限会随副本数相乘。Runtime 必须在 Binding 检查结果和指标中暴露有效作用域，使扩缩容控制面能够区分“增加 Worker 后总容量增加”和“Worker 数增加但部署总 permit 不变”。

## 三类容量平面

| 容量平面 | 是否必须 | 主要作用域 | 满载行为 |
|---|---:|---|---|
| Execution/Agent | 必须 | 当前 Binding 的 Root、Child 与 RESUME | 有界等待或 admission rejection |
| Model | 可选 Binding 门禁 | 当前 Binding 或共享模型资源 key | passthrough，或有界等待/拒绝 |
| Tool | 可选 Binding 门禁 | 当前 Binding 或共享工具资源 key | passthrough，或有界等待/拒绝 |

三类容量分别计数和观测，不合并为一个 semaphore。Binding 只保存模式、上限、队列预算和可选容量 key；Runtime 创建并拥有实际 limiter、queue、permit 与统计状态。

### Execution/Agent 容量

Execution 容量必须启用，并拆成两个独立上限：

- `max_live_runs`：已经 admission、尚未终止的 Root 和 Child 总数，用于限制执行树、状态与内存规模；
- `max_runnable_runs`：当前持有 execution lease、可以执行 Module 业务逻辑的执行流数量；
- Root START queue、总 waiter、child depth 和直接 child 数量必须分别有界；
- Blocking Child 不消耗第二个 Root slot，但占用 Child live 预算；
- Parent 等待 Child 或资源时仍然是 live execution，但不再是 runnable Execution。

### Model 容量

Binding 的 Model 策略有两种明确模式：

- `passthrough`：不增加 Binding-local 模型 limiter，由模型服务和 Runtime adapter 的资源治理负责流量检测；
- `limited`：对当前 Binding 或 `capacity_key` 对应的共享模型资源组设置本地上限和有界等待。

passthrough 不表示创建无限本地队列。调用仍受 live execution、deadline、取消和模型层 attempt/retry 预算约束。模型 endpoint、credential scope 或部署资源本身的容量所有者可以跨 Binding 共享，但不属于 Execution lease。

### Tool 容量

工具并发同时存在两个独立层次：

- ToolCallLayer 的并发限制单次 Layer 调用、也就是单个 Execution 内的工具 fan-out；
- Binding Tool 策略可选择 `passthrough`，或限制该 Binding 全部 Execution 共享的工具总并发；
- 特定浏览器、进程池、数据库、MCP server 或远程服务仍可以通过资源 key 共享更窄的物理容量。

一次工具调用必须满足所有已启用层次。ToolCallLayer 上限不能代替 Binding 总量控制，Binding 上限也不能放宽 ToolCallLayer 的单 Execution fan-out。

上述容量字段、作用域和公开类型名称已经由 [Runtime SDK](SDK.md#创建部署治理域) 冻结；本节展开它们的容量归属与运行语义。

## 执行关系

### Root Execution

`bound.invoke()` 或 `bound.stream()` 创建托管 Root Execution。未绑定 `module.invoke()`/`module.stream()` 创建 direct Root，不属于 Binding Execution，也不获得本节的 admission、identity、lease 或恢复保证。托管 Root 在获得 START lease 前只保留不可变输入快照和调度元数据，不创建业务 Task，不执行 `forward()`。

Root 的身份至少分为四层：

| 身份 | 生命周期与用途 |
|---|---|
| `request_id` | 一次 HTTP、RPC 或 SDK 调用尝试，用于 tracing；客户端重试时通常变化 |
| `idempotency_key` | 调用方定义的逻辑业务操作身份；同一操作重试时保持不变 |
| `execution_id` | Runtime 接纳后创建的逻辑 Execution 身份；跨 Worker 恢复保持稳定 |
| `attempt_id` | 一次实际执行尝试；每次 RETRY 或 restore 创建的新执行尝试都不同 |

`idempotency_key` 在普通瞬时执行中可以省略；要求请求去重、durable recovery 或最终提交协调时必须提供。其唯一性作用域至少包含 `binding_id` 与租户/调用身份。相同作用域和 key 的重复请求如果输入摘要、ExecutionPlan 或关键策略不一致，Runtime 必须拒绝冲突；若一致，则返回、等待或重新订阅同一个逻辑 Execution，而不是创建第二个 Execution。幂等记录的保留时间和过期行为必须由部署策略声明。

未绑定 Module 和 BoundModule 都只通过 `invoke()` 与 `stream()` 入口创建各自模式的 Root。`forward()` 中对原始 Module、预绑定 BoundModule 或 RemoteModule 的直接调用都表示当前 scope 的 Child；不得在 `forward()` 中调用依赖的 `invoke()` 把子调用伪装成新 Root。direct scope 直接转发本地 Child 且不增加容量治理；managed scope 建立受管 Child 并应用 Binding、lineage、deadline、取消和 placement。

预绑定 BoundModule 在执行树外不使用 `__call__()` 创建 Root；调用者必须使用 `invoke()` 或 `stream()`。它在已有 ExecutionScope 内被直接 `await` 时，由当前 scope 创建 Child，并把 Child 派发给该 BoundModule 的目标 Runtime/Binding：

```text
parent scope -> await bound_child(...)
  -> parent Runtime 创建 Child identity 并释放 Parent lease
  -> target Runtime/Binding 接纳并执行 Child attempt
  -> Child terminal，结果和事件回到原 lineage
  -> Parent 进入 WAITING_RESUME
```

### Blocking Child

Parent 在 `forward()` 中直接等待一个子 Module 时形成 Blocking Child：

1. Runtime 记录 Parent、Root、depth 和有效 deadline。
2. Parent 从 `RUNNING` 进入 `WAITING_CHILD` 并释放 lease。
3. Child 作为 START Item 进入同一调度域。
4. Child 获得 lease 后执行。
5. Child 进入终态并释放 lease。
6. Parent 进入 `WAITING_RESUME`。
7. Parent 通过正常公平调度重新获得 lease，随后继续执行。

因此 `max_runnable_runs=1` 时 Parent 等待 Child 也不能死锁。普通文件、网络或用户 coroutine 的 `await` 不自动释放 lease；只有 Runtime 可识别的 Child、Model、Tool、Handle、容量和调度恢复等待才执行 lease handoff。

### Parallel Child

Parallel Child 使用独立受管执行流：

- Parent 创建 Child Handle 后可以继续执行并继续持有自己的 lease；
- Child 作为有界 START Item 独立竞争 lease；
- Parent 等待未完成 Handle 或 `gather` 时释放 lease；
- Parent coroutine 使用 RESUME Item 重新获得 lease，与其他 Ready Item 一起参与公平调度；
- Parent scope 结束前必须消费、取消或由 Runtime 自动取消并 join 全部结构化 Child。

通用 Parallel Child 的 0.2.x SDK 入口是 Module 继承的 `gather()`；它不是 Agent 专用入口，也不提供 `detached=True` 改变结构化所有权。工具调用的 detach 是另一个边界：普通工具与 Agent-backed Tool 都通过独立 ToolTask admission 获得新身份，需要 durable recovery 时由独立 Job 承载该 ToolTask。调用不再作为 Child，Parent 只保留稳定引用，新任务仍受 Binding 与资源治理。

## Execution Lease

Execution lease 允许一个受管逻辑执行流运行：

```text
0 <= granted_run_leases <= max_runnable_runs
available_run_leases = max_runnable_runs - granted_run_leases
```

必须满足：

- `RUNNING` 持有一个 lease；
- `QUEUED`、`WAITING_CAPACITY`、`WAITING_CHILD`、`WAITING_HANDLE`、`WAITING_MODEL`、`WAITING_TOOL` 和 `WAITING_RESUME` 不持有 Execution lease；
- 终态不持有 lease；
- START、RESUME、异常、取消、deadline 和关闭的所有路径保持 lease 计数守恒；
- 非 owner Task 不能释放、重新授予或转移其他执行流的 lease。

## START 与 RESUME

调度器处理两类 Ready Item：

- `START`：尚未开始 `forward()`、等待首次 lease 的 Root 或 Parallel Child；
- `RESUME`：仍持有活跃 owner Task 和 coroutine continuation、完成受管等待后需要重新获得 lease 的 Parent。

START 和 RESUME 使用同一套 FIFO、公平和 deadline 检查。RESUME 不能绕过其他 Ready Item，也不能因为 Parent 曾经运行过而自动取回 lease。

RESUME 是调度概念，不是持久化恢复概念。它要求原 owner Task、Python coroutine 栈及其局部变量仍然存在；Worker 或进程退出后，不得仅通过构造 RESUME Item 恢复该执行流。本文后续使用“调度恢复”专指这一含义；进程故障后的继续执行使用 RETRY 或 CHECKPOINT RESTORE 表达。

## 执行连续性与故障边界

- Root 可以由外部队列分发到不同 Worker，Module 边界也可以由可移植 ExecutionPlan 支持远程调用。
- 活跃 `forward()` continuation 不保证跨 Worker 迁移；普通 Python 调用栈、局部变量和未提交事件属于当前执行 attempt 的临时状态。
- Worker 故障后只能根据显式 retry/checkpoint 策略重启相应边界，不得声称从任意 `await` 之后透明续跑。
- 未提交 checkpoint 的局部状态会丢失；边界重试可能重新执行 Model、Tool、`emit()` 和用户逻辑，因此必须定义稳定调用身份、幂等和事件去重。
- Runtime 必须声明自身支持的是本地执行、分布式 placement 还是 durable recovery；支持前两者不自动意味着支持后者。

完整能力分级和故障矩阵见 [持久化与恢复边界](DURABILITY.md)；参考实现采用的内部记录点、history/checkpoint 与确定性重放见 [透明恢复与确定性重放](REPLAY.md)。

## Execution 与事件订阅生命周期

托管执行生命周期和事件观察生命周期在 Runtime 内部必须分离。普通用户仍可使用 `invoke()` 与 `stream()`：当前 `stream()` 是“创建一个 Root 并观察它”的便捷接口，因此提前退出时继续取消并清理它所创建的 direct 或 managed Root，不要求普通用户感知 Execution Handle 或订阅对象。

需要 durable 重连、后台继续执行或多个观察者时，Runtime 提供独立的稳定 Execution Handle：

```python
run = await bound.start(message, context, run=options)

async with run.subscribe(after=event_cursor) as events:
    async for event in events:
        ...

message, context = await handle.result()
await run.cancel()
```

关闭 `events` 只注销当前订阅者，不改变 Execution 状态；取消 Execution 必须调用显式控制入口。订阅游标、事件去重与 retention 只在 Runtime 声明相应 durable event capability 时可用。0.2.x 的 `start()`、`subscribe()` 与 Execution Handle 是公开控制面，但不改变 `invoke()`、`stream().final_result()` 与 `handle.result()` 统一返回 `(message, context)` 的最终结果契约。运行元数据保留在 Handle，不使用额外包装业务结果。

## 调度与公平

当前保留以下公共保证：

- 同一优先级按单调 sequence FIFO；
- 已运行的执行流不被新 Item 抢占；
- 存在 Ready Item 和空闲 lease 时必须继续分配，即 work-conserving；
- 默认策略在有限工作和持续可用容量下不得永久饿死某个 Ready lineage；
- 不承诺完成顺序、精确时间片或固定吞吐。

历史规范中的四级优先级和 `16:8:4:1` 权重暂不进入当前冻结契约；如需保留，应作为 Binding 的可选调度策略单独评审。

## Model/Tool 受管等待

Model 和 Tool 的 Module 调用仍先按 Blocking/Parallel Child 规则进入受管执行树；当对应 Layer 等待模型或工具容量及 I/O 时，Runtime 必须避免跨容量平面的 hold-and-wait：

```text
parent RUNNING
  -> 按 Child handoff 进入 ModelCallLayer/ToolCallLayer
  -> layer child 获得 Execution lease并完成校验与提交
  -> layer child 释放 Execution lease，进入 WAITING_MODEL/WAITING_TOOL
  -> adapter 在自己的 limiter、queue 和 permit 下执行
  -> 完成、失败、取消或 timeout
  -> 先释放 Model/Tool permit
  -> layer child 进入 WAITING_RESUME并重新获得 Execution lease
  -> layer child terminal，随后 Parent 按 Child 规则重新获得 lease
```

具体 adapter 可以在 I/O 期间持有所需 Model/Tool permit，但不能持有 layer child 的 Execution lease。结果到达后必须先完成资源清理并释放 permit，再等待 layer child 重新获得 lease；不得持有 Model/Tool permit 无限等待 RESUME。

未经 Runtime 管理的普通 `await` 无法安全进入上述状态机，因此不会自动释放 Execution lease。

## 外部信号受管等待

`wait_external()` 是 Runtime 的通用能力，用于审批、补充参数、验证码或人工选择等短时外部信号；它不把审批语义固化到 Agent 或 Runtime。Module 决定等待的业务含义，并把返回的 JSON 值转换为领域 Message；Runtime 只管理等待身份、反馈匹配、调度和生命周期。

```python
decision = await self.wait_external(
    kind="approval",
    key=approval_id,
    request={"action": action},
)
```

调用必须先原子注册 `(kind, key)`，再发布可观察的 `external.waiting` 事件，避免反馈在 waiter 建立前到达。服务通过同一 Runtime 信号入口或其共享适配器提交反馈：

```python
await runtime.deliver_external(
    kind="approval",
    key=approval_id,
    value={"approved": True, "comment": "同意"},
)
```

等待流程为：

```text
RUNNING
  -> 注册外部 waiter 并发布 external.waiting
  -> WAITING_EXTERNAL，释放 runnable lease
  -> 外部反馈、deadline、取消或关闭完成 waiter
  -> 成功反馈进入 WAITING_RESUME
  -> 重新获得 lease 后恢复原 forward()
```

这里的“等待”会阻塞当前 Agent 调用的业务推进：当前 `forward()` 不会继续，直接或间接等待它的 Parent 也不会继续。它不阻塞 event loop、操作系统线程或其他独立 Execution；同一个无请求级可变状态的 Module 定义仍可被其他 Execution 并发调用。但是，原 owner Task、Python 调用栈、局部变量和 Context 引用始终保留，当前执行树也继续占用 live execution 与 waiter 容量。

因此 `wait_external()` 必须谨慎使用：

- effective deadline 是 Execution deadline、局部 timeout 与 `ExecutionCapacityPolicy.max_external_wait_seconds` 的最早值；部署策略硬上限确保远期 deadline 也不能让 owner Task 无界驻留。
- Binding/Runtime 必须限制 external waiter 总数和每个 lineage 的 waiter 数，避免大量挂起 Execution 耗尽内存与 live admission。
- 相同 `(kind, key)` 的重复注册必须拒绝；重复、过期或取消后的反馈必须返回明确状态，不能恢复两次。
- deadline、调用方取消和 immediate shutdown 必须原子注销 waiter；反馈与取消竞态只能有一个结果获胜。
- 该能力只提供进程存活期间的调度 RESUME，不提供 Worker 故障恢复、调用栈迁移或持久化 continuation。
- 预期等待达到小时或天时，应返回领域 `ApprovalRequiredMessage` 等结果并结束当前 Execution；外部保存业务状态，反馈到达后使用新 Message 和当前有效 Context 创建新 Execution。

`LocalRuntime` 已实现 `wait_external()` 与 `deliver_external()` 的进程内有界等待语义。它释放并重新获取 runnable lease，但不持久化 Python continuation；跨进程长等待仍应结束当前 Execution，待反馈后创建新 Execution。

## 无死锁与稳定性不变量

- 内部锁只保护不跨 `await` 的短状态转换；不得在持锁状态等待 queue、permit、Child、I/O 或 shutdown。
- Parent 不以新 Root 方式递归提交 Child；Blocking Child 使用同一 lineage，并通过 lease handoff 执行。
- 原始 Child 不重新创建 Binding；它继承 Parent Binding。预绑定 Child 使用已经确定的 Child Binding；adaptive placement 只选择已获准的物理 target，不逐次创建 Binding。
- Execution、Model 与 Tool queue 分离，某个模型或工具资源拥塞不能阻塞无关资源的 admission queue。
- RESUME 使用有界优先级以缩短 lease handoff：同一队列内保持 FIFO，连续 RESUME 达到固定 burst 上限后必须选择等待的 START，因此两类工作都不能永久饥饿。
- 固定上限和有界队列是默认策略；瞬时 Provider 429、Tool 延迟或局部队列增长不得自动触发大幅度并发扩缩。
- Runtime 分别观测 live/runnable、各队列等待、permit 使用、rejection、timeout 和 handoff latency，以区分 Agent、Model 与 Tool 瓶颈。

## 背压

START queue 只约束尚未获得首次 lease 的 Root 和 Parallel Child。Blocking Child 与 RESUME 不占 START queue，但仍受 live、depth、child 和 waiter 上限约束。

当 Parent 创建 Parallel Child 时队列已满：

1. Parent 进入 `WAITING_CAPACITY` 并释放 lease；
2. 容量请求进入有界 FIFO waiter；
3. 获得空间后创建 Child START Item；
4. Parent 进入 `WAITING_RESUME`；
5. Parent 重新获得 lease 后返回 Child Handle。

`max_queue_size=0` 表示不等待队列容量，满载时立即拒绝；任何默认值都不得表达无界队列。

## Deadline 与取消

Child 的有效 deadline 为 Parent deadline 与 Child 请求 deadline 中更早者。排队、容量等待、Child 等待、外部信号等待和 RESUME 等待都必须响应 deadline。

结构化取消规则为：

- Parent 取消或超时时向未完成 Child 传播；
- Parent 正常返回、失败、取消或超时前都不能遗留未 join 的 Child；
- Child 自身失败或超时通过结果或 Handle 传播，不自动取消无关 Root；
- 已进入取消或 timeout 清理的 Parent 不再申请普通 RESUME lease；
- 清理完成后才能报告终态和释放仍持有的 lease。

## 状态与关闭

最小执行状态集合为：

```text
CREATED -> QUEUED -> RUNNING
RUNNING -> WAITING_CAPACITY | WAITING_CHILD | WAITING_HANDLE
RUNNING -> WAITING_MODEL | WAITING_TOOL | WAITING_EXTERNAL
WAITING_* -> WAITING_RESUME -> RUNNING
RUNNING | WAITING_* -> CANCELLING | TIMING_OUT
* -> SUCCEEDED | FAILED | CANCELLED | TIMED_OUT
```

每个 execution attempt 只能进入一个终态，且该 attempt 的 `forward()` 最多启动一次。显式 RETRY 会创建新的 attempt，而不是让旧 coroutine 重新进入 `forward()`；CHECKPOINT RESTORE 的恢复单位和重放范围由 durability 契约定义。Runtime graceful shutdown 停止接纳新 Root，继续排空已经接纳的结构化执行树；immediate shutdown 拒绝新 Root 和 Child，取消所有排队、运行和等待项，并在返回前完成 lease 与 registry 清理。

## 非职责

Runtime 并发契约不定义：

- Agent 的 ReAct、委派、身份或业务结果；
- LLM fallback、Provider 自身的流量检测算法或重试；
- Tool 授权、资源选择或结果规范化；
- Session、Context branch 或业务 Store 的创建与提交；
- 通过 detached Child 表达独立生命周期任务。结构化 Child 始终属于 Parent；工具 detach 必须使用独立 ToolTask admission 和身份，并在转换后脱离 Child 所有权。需要 durable recovery 时，必须由独立 Job 通过持久化入口承载该 ToolTask；
- 跨进程容量协调器、分布式 permit 存储或外部资源限流器的具体实现。Binding 仍必须声明容量作用域；具体 Runtime 只能接纳自己能够兑现的作用域。
