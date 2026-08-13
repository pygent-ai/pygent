# 透明恢复与确定性重放

本文描述 Runtime 的透明恢复方案。它是 [持久化与恢复边界](DURABILITY.md) 的可调整实现策略，不是 Pygent 第一原则。0.2.x 参考实现以 `SQLiteHistoryStore` 和 ExecutionScope effect 边界交付了其中的确定性重放核心；suspend/compaction 等扩展控制面仍不是公共契约。

## 定位

透明恢复不新增 Agent 或业务能力域。普通 Module 仍只实现：

```python
async def forward(message, context) -> tuple[Message, Context]: ...
```

用户不声明 checkpoint、step 或源码恢复标签。支持该能力的 Runtime 在自身可观察的执行边界记录 durable history；恢复时创建新的 execution attempt，从头重新进入 `forward()`，并用已提交历史重放此前结果，直到到达第一个尚未完成的受管调用。

该方案不序列化 CPython coroutine、frame 或调用栈。恢复产生新的 Task 和调用栈，局部变量由确定性重放重新构造。

## 能力承诺

可以承诺：

- suspend 请求可以在任意时刻到达；
- Runtime 在下一个可提交的内部边界停止；
- 已可靠提交的受管调用不会在恢复时重复执行；
- 恢复时校验 ExecutionPlan、代码制品、schema、serializer 和策略兼容性；
- 未知状态的外部副作用按 Tool/Activity 策略查询、重试、补偿或转人工。

不能承诺：

- 从任意 Python 字节码的下一条指令继续；
- 保存或迁移原 coroutine、锁、连接、文件描述符、iterator 或第三方 Future；
- 对 Runtime 不可见的 I/O 和外部副作用提供 exactly-once；
- 代码或 schema 不兼容时自动猜测迁移；
- CPU 密集型代码或死循环在没有内部检查点时立即响应 suspend。

因此准确语义是“任意时刻请求中断，在下一个内部边界停止”，不是“任意指令位置快照”。

## 默认内部记录点

Runtime 默认在以下可观察边界记录历史，用户业务代码不出现对应 API：

1. Root admission 完成、`forward()` 启动之前；
2. 每次结构化 Module 调用提交之前和结果可靠返回之后；
3. Model 调用提交、标准化结果或失败确定之后；
4. Tool 调用提交、结果或未知副作用状态确定之后；
5. `wait_external()` waiter 注册、反馈提交、超时或取消之后；
6. `emit()` 事件获得稳定事件身份并可靠提交之后；
7. Module 和 Root 终态提交之后。

普通 Python 表达式、局部纯计算和分支不单独写入 history。恢复时它们会重新计算。未经 Runtime 管理的第三方 `await` 不是默认记录点；durable Binding 必须拒绝、包装为受管 adapter，或把包含它的整个 Module attempt 视为可重复执行边界。

## History 与 Snapshot

恢复事实不能依赖可淘汰的普通 cache。实现应区分：

- **Durable history**：调用意图、稳定身份、输入摘要、结果、错误、事件和副作用状态的顺序日志，是恢复 source of truth；
- **Snapshot**：对已提交 history 的周期性压缩，用于缩短重放时间，可以由 history 重新生成；
- **Transient cache**：反序列化结果或热点 snapshot 的性能优化，不参与正确性判断。

history 记录至少包含：

```python
ExecutionHistoryEvent(
    execution_id=...,
    attempt_id=...,
    sequence=...,
    module_path=...,
    call_index=...,
    effect_kind=...,
    request_hash=...,
    status=...,
    result=...,
    plan_id=...,
    code_digest=...,
)
```

`execution_id` 表示跨恢复稳定的逻辑 Execution；`attempt_id` 表示一次实际执行；`module_path + call_index` 标识该 Parent 调用中的受管 effect。`SQLiteHistoryStore` 使用稳定 effect 身份和请求摘要验证重放；其他后端可以改变物理格式，但不得改变确定性校验语义。

## 执行与重放

首次执行时，Runtime 必须先可靠提交调用结果，再把结果交还给 `forward()`：

```text
forward()
  -> ExecutionScope 收到受管调用
  -> 写入 effect started
  -> 执行 Module / Model / Tool / external wait
  -> 写入 effect completed 或明确失败状态
  -> 把结果返回 forward()
```

恢复时：

```text
读取 Execution history 和可选 snapshot
  -> 校验 ExecutionPlan 与代码兼容性
  -> 创建新 attempt
  -> 从头执行 forward()
  -> 已完成 effect：校验请求身份并返回历史结果
  -> 已记录未完成 effect：按 effect 恢复策略处理
  -> history 耗尽：切换为实时执行
```

如果当前调用与历史中的 `module_path`、`call_index`、effect 类型或请求摘要不一致，Runtime 必须停止并报告 non-deterministic replay，不得跳过记录或猜测继续。

## Suspend 与 Resume

支持 suspend 的扩展 Runtime 状态流为：

```text
RUNNING
  -> SUSPEND_REQUESTED
  -> 到达下一个可提交的 Runtime 边界
  -> 提交 history / snapshot
  -> SUSPENDED
  -> 释放 owner Task、coroutine 栈和 live execution 资源

SUSPENDED
  -> RESTORING
  -> 创建新 attempt 并重放
  -> RUNNING
```

同进程受管等待后的调度 `RESUME` 仍保留原 coroutine；本文的 durable restore 会创建新 attempt。两者必须使用不同状态、事件和指标。

若 suspend 请求到达时正在进行不可立即取消的 Model、Tool 或第三方调用，Runtime 可以等待调用落入明确状态、尝试取消，或把 effect 记录为 unknown；不得把“已发送但结果未知”伪装成“未执行”。

## 确定性边界

透明重放要求同一输入与同一已提交 history 产生相同的受管调用序列。以下值必须由 Runtime 记录并在重放时返回，或通过受管 adapter 获取：

- 时间、随机数、UUID；
- Model 输出和流式最终结果；
- Tool/Activity 输出；
- 外部反馈；
- 影响控制流的配置快照和稳定业务查询结果。

以下行为在 durable Binding 中必须被禁止、静态检查、运行时检测或明确降级为 Module 边界重试：

- 读取可变全局状态后决定调用顺序；
- 裸 `asyncio.create_task()` 或脱离结构化执行树的后台任务；
- 直接执行未声明幂等性的数据库、网络或文件副作用；
- 跨受管边界保留连接、锁、文件句柄、iterator、generator 或任意活 Python 对象；
- 使用普通第三方 `await`，但没有 adapter、查询或重试语义。

普通非 durable Binding 不受这些限制，也不获得透明恢复保证。

## 外部副作用

持久化 intent 和 result 之间仍存在故障窗口：外部系统可能已经执行，但 Runtime 尚未提交结果。恢复策略必须来自部署策略和 ToolSpec/ActivitySpec，而不是由重放引擎猜测：

| 状态 | 恢复策略 |
|---|---|
| 固有幂等或有幂等键 | 查询结果或安全重试 |
| 可查询但不可安全重试 | 查询外部状态并提交结果 |
| 可补偿 | 进入显式补偿流程 |
| 不可查询、不可重试、状态未知 | 暂停并进入人工或业务决策 |

Model Provider 不支持幂等请求时也可能重复计费或生成不同结果。Runtime 应记录逻辑 call、attempt 和 Provider request 的关联身份，不得宣称 exactly-once Model 调用。

## 可调整策略

透明恢复不进入第一原则，具体业务可以通过 Runtime/Binding 部署策略调整，不改变 `forward()`：

- 是否启用 durable history；
- 自动记录哪些受管 effect；
- 每多少事件或多大 history 生成 snapshot；
- suspend 是等待当前 effect、请求取消还是立即记录 unknown；
- history、snapshot 和终态结果的保留时间；
- replay 最大事件数、最大耗时和 non-determinism 处理方式；
- 非受管 `await` 是拒绝绑定、降级为 Module 边界重试还是关闭恢复能力；
- 不同 Tool/Model/Activity 的查询、重试、补偿和人工决策策略。

这些策略必须被解析为版本化引用，并参与 ExecutionPlan 的兼容性检查。默认值可以随产品阶段和业务负载调整，但已经持久化的 Execution 必须按创建时解析出的策略快照恢复。

## 与 Context 的关系

Context 表示显式流转的不可变 Agent 状态快照，可以包含用户定义的 portable 历史视图和领域状态，但不保存 execution cursor、attempt、未完成 effect 或 Runtime 恢复状态。Context 和 Message 可以作为 durable history 中的边界输入输出；Runtime history、Execution snapshot 与权威业务 Store 分别拥有自己的职责和提交协议。

## 参考实现状态与扩展项

当前代码提供具体 `LocalRuntime`、`SQLiteHistoryStore`、Execution/Task/effect/event/checkpoint 表、确定性请求摘要校验、未完成 Execution 新 attempt 恢复、Model/Tool 已提交结果重放、HTTP Worker 与游标 SSE。下列能力仍属于可选扩展，不能从基础 API 推断：

1. snapshot compaction、retention 与 migration；
2. durable Binding 对任意第三方 `await` 的静态检查；
3. suspend 控制面、事件和状态名称；
4. 并行 Child 与 token 级流式事件的重放；
5. 跨进程外部 waiter continuation；
6. 未知 Tool/Model 状态进入人工决策的标准协议。
