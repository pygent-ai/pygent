# Pygent 第一原则

> 每次业务调用只有一个逻辑 Execution，每个实际 attempt 只有一个 fenced owner。`start()` 创建可立即观察和取消的逻辑执行，`invoke()` 固定投影为 `start() + result()`，`stream()` 固定投影为 owned `start() + subscribe()`；结果、事件和控制面不能各自重复运行 `forward()`、模型 Provider 或工具 Executor。统一事件信封、Snapshot、Outcome、Attempt、Span、ModelExecution、ToolExecution、EffectOutcome 与 Worker 转发契约见 [Execution contract](EXECUTION.md)。

模型 attempt 必须串行且取消清理有界；无法确认上一请求已退出时的 `OUTCOME_UNKNOWN`、禁止 retry/fallback 与 client 隔离语义同样由 [Execution contract](EXECUTION.md) 统一定义。

本文是 Pygent 的最高契约。任何文档、测试或实现与本文冲突时，均以本文为准；改变执行语义必须先修改本契约和 SDK 示例，不保留并行的旧控制面。

1. **Binding 是可选的部署域，不是 Agent 身份**：直接执行不创建 Binding；只有接入托管 Runtime 时，Binding 才表示一组 Module/Agent 共同使用的部署策略与资源治理域。托管执行中的结构化父子 Agent 默认继承当前 Binding，只有需要独立治理边界时才创建不同 Binding，包括容量、资源、权限、安全、SLA、服务、部署策略或生命周期隔离。
2. **统一抽象**：Agent、Layer 与用户组合都只是 Module；内置能力不享有特权。
3. **统一传递**：Module 只表达 `(message, context) -> (message, context)`；Message 是 Module 间传递的类型化当前增量，不等同于聊天文本；Context 是显式传递、不可变且可移植的 Agent 状态快照。基础 `Context` 保存当前模型投影，用户可以通过具有稳定 schema 的 `Context` 子类增加历史视图和领域状态；模型只消费基类定义的 prompt、messages、tools 与 metadata 投影。
4. **PyTorch-like**：用户通过声明子 Module 和实现一个 `forward()` 表达计算与层级关系。
5. **无状态**：Module 定义不持有单次运行或业务状态；运行状态只通过输入和返回的不可变 Context 显式流转。跨 Execution 的加载、提交、revision、冲突处理与权威持久状态仍由框架外部管理；支持 durable recovery 的 Runtime 可以保存执行元数据、Context checkpoint 与重放记录，但这些数据不得成为第二个业务状态源。
6. **执行同源且结果统一**：直接或托管的 `invoke()` 与 `stream()` 都运行同一个 `forward()` Module 图；流式只是观察方式，不是另一套实现。普通调用与 `stream().final_result()` 统一返回最终 `(message, context)`，运行身份、状态、usage 与可重连订阅属于独立的高级 Execution 控制面，不改变业务结果形状。
7. **执行分层**：未绑定 Module 的 `invoke()`/`stream()` 使用框架内部的本地 direct execution scope，调用方无需创建 Runtime；该模式不提供框架级容量治理、远程 placement、持久恢复或跨 Execution 调度，Root 并发由调用方管理。`bind()` 把同一 Module 图接入托管 Runtime/Binding；资源、并发、调度、结构化取消、placement 与生命周期才由 Runtime 按声明能力统一负责。两种模式中的用户 `forward()` 与 Child 直接调用保持不变，用户不直接操作 ExecutionScope。
8. **执行位置无关，恢复边界可验证**：Module 的输入输出语义不依赖实例、进程或节点；本地、分布式与弹性 Runtime 不改变 `forward()` 的调用与最终结果契约。普通 `forward()` 的活跃 coroutine 不保证跨进程恢复；持久化 Runtime 只能依据自身声明的可验证边界重放或恢复，并必须遵守对应的幂等、副作用和版本兼容约束。
9. **可恢复执行只属于托管 Runtime**：Runtime 可以按部署策略记录执行进度，并在运行中断或 Worker 故障后从可验证的执行边界重建和继续 Execution；direct execution 不获得该能力。恢复不改变 `forward(message, context)` 契约，也不要求普通业务代码显式声明恢复点。具体记录粒度、恢复方式、兼容检查和副作用处理由 Runtime capability 与部署策略决定；可恢复执行不等同于序列化任意 Python coroutine，也不自动保证外部副作用 exactly-once。
10. **定义可共享**：Module 图允许同一 Module 被多条属性路径引用；定义身份可以共享，每次调用身份必须独立。direct execution 不提供框架级共享容量治理，调用方自行管理并发与本地资源；托管执行按 Runtime 解析的资源身份共享物理资源容量。
11. **控制语义可组合**：handoff、审批请求和领域终止条件可以由用户定义的 Message 与 Module 表达；普通 Module 不因此获得跨进程持久挂起或恢复 Python 调用栈的隐式能力。
12. **公开值可移植**：Message、Context、用户 Context 子类、ToolDefinition、ToolSpec、ToolTask、ToolResult 与 ExecutionEvent 的公开扩展数据必须具有稳定 schema，并由严格、有限且递归冻结的 JSON 值组成，不得携带连接、锁、协程、handler、Store、client 或任意活 Python 对象。Python 类名和 pickle 不是远程或恢复协议。
13. **完整生命周期只有一个预算**：Execution deadline 从提交开始，覆盖准入前初始化、模型/资源 pin、容量排队、业务执行、取消清理和终结；所有等待都必须可取消。配置与发布是独立控制面操作，使用自己的显式 deadline。
14. **Journal 决定终结顺序**：terminal span events、唯一 Execution terminal event、冻结 Outcome、终态 Snapshot 与 terminal sequence 必须原子提交。任何订阅都在交付 terminal sequence 后结束，不能根据另一路先读到的 terminal status 提前退出。
15. **观察不等于取得执行权**：Handle 只是由 backend 支撑的稳定控制面引用。attach 只能观察、等待或请求取消；recover 必须验证资格、获取新 owner lease 和 fencing token，并创建新 attempt。首次 attempt 与恢复 attempt 使用同一所有权协议。

可恢复执行的能力分级与故障边界见 [Runtime Durability](runtime/DURABILITY.md)，参考实现的内部记录与重放方案见 [透明恢复与确定性重放](runtime/REPLAY.md)。
