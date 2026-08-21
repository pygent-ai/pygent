# Pygent 第一原则

> 每次业务调用只有一个逻辑 Execution，每个实际 attempt 只有一个 fenced owner。`start()` 创建可立即观察和取消的逻辑执行，`invoke()` 固定投影为 `start() + result()`，`stream()` 固定投影为 owned `start() + subscribe()`；结果、事件和控制面不能各自重复运行 `forward()`、模型 Provider 或工具 Executor。统一事件信封、Snapshot、Outcome、Attempt、Span、ModelExecution、ToolExecution、EffectOutcome 与 Worker 转发契约见 [Execution contract](EXECUTION.md)。

模型 attempt 必须串行且取消清理有界；无法确认上一请求已退出时的 `OUTCOME_UNKNOWN`、禁止 retry/fallback 与 client 隔离语义同样由 [Execution contract](EXECUTION.md) 统一定义。

本文是 Pygent 的最高契约。任何文档、测试或实现与本文冲突时，均以本文为准；改变执行语义必须先修改本契约和 SDK 示例，不保留并行的旧控制面。

1. **Binding 是可选的部署域，不是 Agent 身份**：直接执行不创建 Binding；只有接入托管 Runtime 时，Binding 才表示一组 Module/Agent 共同使用的部署策略与资源治理域。托管执行中的结构化父子 Agent 默认继承当前 Binding，只有需要独立治理边界时才创建不同 Binding，包括容量、资源、权限、安全、SLA、服务、部署策略或生命周期隔离。
2. **统一抽象**：Agent、Layer 与用户组合都只是 Module；内置能力不享有特权。
3. **调用自由**：Module 不规定业务参数和结果形状；`forward()` 可以接收零个、一个或多个位置参数与关键字参数，并返回该 Module 自己声明的结果。Message 与 Context 是可选公开值，不是所有 Module 的强制端口。
4. **PyTorch-like**：用户通过声明子 Module 和实现一个 `forward()` 表达计算与层级关系。需要显式状态递推时可以使用标准 RecurrentModule，其语义为 `(input, state) -> (output, next_state)`；普通 Module 不被强制采用该形状。
5. **无隐藏调用状态**：Module 可以持有定义配置、子 Module 和显式声明的部署资源，但不持有某次调用的输入、局部结果、recurrent state、请求状态或业务会话状态。调用状态通过参数、局部变量和返回值流转；权威持久状态仍由框架外部管理，Runtime 记录也不得成为第二个业务状态源。
6. **执行同源且结果统一**：一种执行方式支持某个 Module 调用时，`invoke()`、Child 调用和 `stream().final_result()` 都运行同一个 `forward()` 图并返回该 Module 声明的结果；流式只是观察方式，不是另一套实现。本次变更不扩大 managed、remote 或 durable execution 已支持的调用范围。
7. **执行分层**：未绑定 Module 的 `invoke()`/`stream()` 使用框架内部的本地 direct execution scope，调用方无需创建 Runtime；该模式不提供框架级容量治理、远程 placement、持久恢复或跨 Execution 调度，Root 并发由调用方管理。`bind()` 把同一 Module 图接入托管 Runtime/Binding；资源、并发、调度、结构化取消、placement 与生命周期才由 Runtime 按声明能力统一负责。两种模式中的用户 `forward()` 与 Child 直接调用保持不变，用户不直接操作 ExecutionScope。
8. **执行位置无关，恢复边界可验证**：对 Runtime 已支持的调用契约，Module 的输入输出语义不依赖实例、进程或节点。普通 `forward()` 的活跃 coroutine 不保证跨进程恢复；持久化 Runtime 只能依据自身声明的可验证边界重放或恢复，并必须遵守对应的幂等、副作用和版本兼容约束。
9. **可恢复执行只属于托管 Runtime**：Runtime 可以按部署策略记录执行进度，并在运行中断或 Worker 故障后从可验证的执行边界重建和继续 Execution；direct execution 不获得该能力。恢复不要求普通业务代码显式声明恢复点；具体记录粒度、恢复方式、兼容检查和副作用处理由 Runtime capability 与部署策略决定。可恢复执行不等同于序列化任意 Python coroutine，也不自动保证外部副作用 exactly-once。
10. **定义可共享**：Module 图允许同一 Module 被多条属性路径引用；定义身份可以共享，每次调用身份必须独立。direct execution 不提供框架级共享容量治理，调用方自行管理并发与本地资源；托管执行按 Runtime 解析的资源身份共享物理资源容量。
11. **控制语义可组合**：handoff、审批请求和领域终止条件可以由用户定义的 Message 与 Module 表达；普通 Module 不因此获得跨进程持久挂起或恢复 Python 调用栈的隐式能力。
12. **公开值可移植**：本地 direct Module 可以使用普通 Python 值；这不改变既有可移植值规则。Message、Context、用户 Context 子类、ToolDefinition、ToolSpec、ToolTask、ToolResult 与 ExecutionEvent 的公开扩展数据仍必须具有稳定 schema，并由严格、有限且递归冻结的 JSON 值组成，不得携带连接、锁、协程、handler、Store、client 或任意活 Python 对象。Python 类名和 pickle 不是远程或恢复协议。
13. **完整生命周期只有一个预算**：Execution deadline 从提交开始，覆盖准入前初始化、模型/资源 pin、容量排队、业务执行、取消清理和终结；所有等待都必须可取消。配置与发布是独立控制面操作，使用自己的显式 deadline。
14. **Journal 决定终结顺序**：terminal span events、唯一 Execution terminal event、冻结 Outcome、终态 Snapshot 与 terminal sequence 必须原子提交。任何订阅都在交付 terminal sequence 后结束，不能根据另一路先读到的 terminal status 提前退出。
15. **观察不等于取得执行权**：Handle 只是由 backend 支撑的稳定控制面引用。attach 只能观察、等待或请求取消；recover 必须验证资格、获取新 owner lease 和 fencing token，并创建新 attempt。首次 attempt 与恢复 attempt 使用同一所有权协议。

可恢复执行的能力分级与故障边界见 [Runtime Durability](runtime/DURABILITY.md)，参考实现的内部记录与重放方案见 [透明恢复与确定性重放](runtime/REPLAY.md)。
