# Runtime 第一原则

本文从属于 [Pygent 0.2 第一原则](../FEATURES.md)。Runtime 是 Module 图按需接入的托管执行契约，不是 direct execution 的前置条件，也不是新的业务能力域。

模型 attempt 的 retry/fallback 与取消清理边界属于 LLM；Runtime 拥有从提交、准入、执行到原子终结的完整 Execution 生命周期，向所有子系统传播同一个 effective deadline 与取消，并保持 [Execution contract](../EXECUTION.md) 的终态映射。

1. **Binding 是可选的部署与资源治理域**：direct execution 不创建 Binding。托管执行中的 Binding 不代表 Agent 身份，也不要求一个 Agent 一个 Binding；原始子 Module 默认继承 Parent Binding。只有需要独立治理边界时才使用预绑定 Child 或 placement policy，包括容量、资源、权限、安全、SLA、服务、部署策略或生命周期隔离。
2. **Binding 聚合三类容量策略且作用域显式**：不可变 `Binding` 同时声明强制 Execution/Agent 容量、可选 Model 容量和可选 Tool 容量；三类容量独立排队、计数和观测，不合并成一个 semaphore。受控容量必须明确属于单个 Runtime 实例、整个部署或外部资源所有者，不得把每个 Worker 的局部上限描述成部署全局上限。
3. **策略与活状态分离**：`Binding` 只保存部署策略、治理域身份和共享容量 key；queue、waiter、dispatcher、Task、timer、permit、execution lease 和计数器只存在于 Runtime。ModelGroup 的 default/profile current pointer、published snapshot、admission selection、pin、live lease 与 recoverable manifest 都属于 Runtime 或外部部署控制面的状态，不进入不可变 Binding 策略值、Agent 或 Module 定义。
4. **Execution/Agent 容量必须受控**：每个 Binding 必须同时限制 live execution 和 runnable Execution；Root、Blocking Child、Parallel Child 与等待调度恢复的 Parent 都属于同一个有界执行树。
5. **Model 容量可以透传或受控**：Binding 可以不增加本地模型门禁，把流量治理交给模型服务；也可以按 Binding 或稳定模型资源 key 设置共享上限和有界队列。
6. **Tool 容量具有两层约束**：ToolCallLayer 的并发只限制单次 Execution 的工具 fan-out；Binding 可以选择是否再限制所有 Execution 共享的工具总并发，二者不能互相替代。
7. **调度内核与业务类型无关**：Runtime 只识别 Root、Parent、Child、资源等待和调度恢复关系，不包含 ReAct、fallback、工具授权或其他 Agent 具体逻辑。
8. **并发按逻辑执行流计数**：runnable 上限限制持有 execution lease 的受管执行流，而不是 event loop 中存在的 coroutine、Task 或仍在等待 I/O 的 live execution 数量。
9. **受管阻塞必须让出 Execution 容量**：Parent 等待 Child、Handle、Model、Tool、显式外部信号或 Runtime 队列容量时释放 runnable lease，完成等待后通过统一 RESUME 调度重新获得 lease。
10. **跨容量平面禁止 hold-and-wait**：执行流不得持有 Execution lease 无限等待 Model/Tool permit；Model/Tool 完成后先释放资源 permit，再把调用方加入 Execution RESUME 队列。
11. **父子关系结构化，独立任务显式转换身份**：Root、Blocking Child 与 Parallel Child 形成统一临时执行树；原始 Child 继承 Parent Binding，预绑定或按 placement 路由的 Child 可以在另一 Runtime 执行，但仍保留同一 root/parent lineage、deadline、取消与终态契约。结构化 Child 不得 detach；普通工具或 Agent-backed Tool 的 detach 必须创建具有独立身份和 admission 的 ToolTask，需要 durable recovery 时由独立 Job 承载该 ToolTask。它不再是 Child，Parent 只保留稳定引用，但新任务仍必须通过声明的 Binding、资源与 capability 治理。
12. **所有等待有界且稳定公平**：Root、START、资源、waiter、child depth 和 fan-out 都有硬上限；固定容量、work-conserving 和无饥饿调度是默认契约，瞬时 429 或延迟变化不得触发无界扩缩容。
13. **部署位置兼容**：本地、分布式或弹性 Runtime 可以使用不同的 placement、transport 和调度实现，但不得改变 Binding、三类容量、父子 handoff、取消和最终结果契约。
14. **恢复能力显式分级并可验证**：普通 Runtime 的 RESUME 表示活跃 owner coroutine 在受管等待后重新获得 lease，不表示进程故障恢复。支持 durable recovery 的 Runtime 必须声明 checkpoint、重放、副作用、代码版本和恢复边界，并在 bind/compile 后报告实际获得的 durability capability；不能满足调用方必需能力时必须拒绝绑定，不得静默降级或把内存 continuation 伪装成持久恢复。
15. **绑定产物具有可验证身份**：Binding 为实际执行的 Module 图生成不可变、版本化的 `ExecutionPlan`；计划哈希覆盖代码制品、Runtime API、节点定义、schema、资源和执行策略引用，远程 Runtime 不以 Python 类型名或对象 pickle 作为部署协议。计划身份覆盖逻辑 ModelGroup requirement、selection policy、required capability 与容量声明，但不覆盖 Binding 当前的 default profile、profile current pointer 或具体 deployment snapshot；每次 admission 选中的 exact profile snapshot 记录在独立、可验证的 admission manifest 中。
16. **放置策略不改变业务调用**：Child placement 支持 `inherit`、`pinned` 与 `adaptive`；用户在 `forward()` 中始终直接调用 Child。`bind()` 创建稳定部署身份，Runtime 在调用时选择物理执行目标，不得通过逐次重新绑定绕过容量或改变调用身份。
17. **动态发现不改变逻辑执行图**：Pygent 不支持绕过 Binding 和 ExecutionPlan、按任意名称调用未声明 Agent、Module 或 ModelGroup 的开放式 Registry。服务发现可以为已声明的稳定逻辑目标动态选择 Worker、endpoint 或副本；对于 ExecutionPlan 已声明的逻辑 ModelGroup，Runtime 可以在其选择策略内解析不同的已验证 profile snapshot。这些选择只能改变具体模型部署，不能改变 Module graph、schema、授权边界、逻辑容量声明、必需 capability、父子身份或调用结果契约；新增逻辑 Agent、Module 或 ModelGroup requirement 必须生成新的 ExecutionPlan，新增独立任务则使用新的 Root Execution/Job。
18. **外部等待必须显式且谨慎**：`wait_external()` 只为短时、受管、可取消的外部信号提供等待边界；它暂停当前 `forward()` 及其同步等待链并持续占用 live execution、Task、调用栈和内存。每次等待必须受 deadline、waiter 上限和关闭清理约束；小时级或天级交互必须结束当前 Execution，并在反馈到达后创建新 Execution。
19. **逻辑 Execution 与实际 attempt 分离**：`execution_id` 标识一次可查询、可取消、可订阅的逻辑执行；每次实际占有执行权的尝试使用独立 `attempt_id`。首次执行与恢复 attempt 服从同一 owner lease、fencing、heartbeat 和终结规则，不能把首次执行当作无租约的特殊路径。
20. **Handle 是控制面引用，不是执行 owner**：`start()` 在创建逻辑 Execution 和 owner Task 后立即返回稳定 `ExecutionHandle`；准入失败、deadline、取消和恢复都通过同一 Handle 可观察。关闭订阅或丢弃 Handle 不改变 Execution 生命周期；只有显式取消或拥有型便捷流退出才请求取消。
21. **deadline 覆盖完整生命周期**：Execution deadline 从 `start()` 提交时刻开始，覆盖持久化提交、Binding/计划准备、模型 store 打开、profile 解析与 pin、容量 admission、排队、`forward()`、取消清理和 finalization。任何内部等待都必须同时响应 deadline、Execution cancellation 与 Runtime shutdown；配置/发布属于独立控制面操作，使用自己的显式 deadline，不能偷用未来请求的 Execution deadline。
22. **准入是可回滚事务**：Execution 提交后，模型 pin、资源 lease、容量 ticket、attempt owner claim 与 history 状态的获取必须由一个 AdmissionCoordinator 管理。成功只在全部资源提交后产生；失败、取消或 deadline 按相反顺序释放已取得资源，并把逻辑 Execution 原子终结为可查询结果，不能留下孤儿 pin、permit、claim 或没有 Execution 记录的等待。
23. **终态是一次原子提交**：terminal span events、唯一 Execution terminal event、`ExecutionOutcome`、终态 snapshot 与 `terminal_sequence` 必须作为一个 finalization 事务提交。订阅只在游标消费到 `terminal_sequence` 后结束，不能仅因另一路读到 terminal status 就提前退出；journal 是事件顺序的权威来源，snapshot 是其物化视图。
24. **attach 与 recover 是不同权限**：`get_execution_handle(execution_id)` 只附着到既有逻辑 Execution 并观察、等待或请求取消，不创建 attempt；`recover()` 必须验证恢复资格、取得新的 fenced owner lease 并创建新 `attempt_id`。查询、附着、恢复和取消都通过统一 Execution backend 契约实现，本地、SQLite 与远程 transport 不得各自发明终态判断。
25. **控制面初始化与发布必须并发安全**：共享 store/resource 的打开使用 single-flight；相同 profile 内容的并发确保操作按稳定 digest 幂等合并，profile current pointer 与 default pointer 在同一事务发布。失败不得暴露半配置状态，也不得把配置等待隐藏在业务 Execution admission 之外。
