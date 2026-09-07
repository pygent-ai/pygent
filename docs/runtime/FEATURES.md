# Runtime 第一原则

本文从属于 [Pygent 0.3 第一原则](../FEATURES.md)。Runtime 是 Module 图按需接入的托管执行契约，不是 direct execution 的前置条件，也不是新的业务能力域。

模型 attempt 的 retry/fallback 与取消清理边界属于 LLM；Runtime 拥有从提交、准入、执行到原子终结的完整 Execution 生命周期，向所有子系统传播同一个 effective deadline 与取消，并保持 [Execution contract](../EXECUTION.md) 的终态映射。

## 治理域与资源身份

1. **Binding 是可选的部署与资源治理域**：direct execution 不创建 Binding。托管执行中的 Binding 不代表 Agent 身份，也不要求一个 Agent 一个 Binding；原始子 Module 默认继承 Parent Binding。只有需要独立治理边界时才使用预绑定 Child 或 placement policy，包括容量、资源、权限、安全、SLA、服务、部署策略或生命周期隔离。
2. **Binding 聚合三类容量策略且作用域显式**：不可变 `Binding` 同时声明强制 Execution/Agent 容量、可选 Model 容量和可选 Tool 容量；三类容量独立排队、计数和观测，不合并成一个 semaphore。受控容量必须明确属于单个 Runtime 实例、整个部署或外部资源所有者，不得把每个 Worker 的局部上限描述成部署全局上限。
3. **策略与活状态分离**：`Binding` 只保存部署策略、治理域身份和共享容量 key；queue、waiter、dispatcher、Task、timer、permit、execution lease 和计数器只存在于 Runtime。ModelGroup 的 default/profile current pointer、published snapshot、admission selection、pin、live lease 与 recoverable manifest 都属于 Runtime 或外部部署控制面的状态，不进入不可变 Binding 策略值、Agent 或 Module 定义。
## 容量与结构化调度

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
## 部署与恢复能力

14. **恢复能力显式分级并可验证**：普通 Runtime 的 RESUME 表示活跃 owner coroutine 在受管等待后重新获得 lease，不表示进程故障恢复。支持 durable recovery 的 Runtime 必须声明 checkpoint、重放、副作用、代码版本和恢复边界，并在 bind/compile 后报告实际获得的 durability capability；不能满足调用方必需能力时必须拒绝绑定，不得静默降级或把内存 continuation 伪装成持久恢复。
15. **绑定产物具有可验证身份**：Binding 为实际执行的 Module 图生成不可变、版本化的 `ExecutionPlan`；计划哈希覆盖代码制品、Runtime API、节点定义、schema、资源和执行策略引用，远程 Runtime 不以 Python 类型名或对象 pickle 作为部署协议。计划身份覆盖逻辑 ModelGroup requirement、selection policy、required capability 与容量声明，但不覆盖 Binding 当前的 default profile、profile current pointer 或具体 deployment snapshot；每次 admission 选中的 exact profile snapshot 记录在独立、可验证的 admission manifest 中。
16. **放置策略不改变业务调用**：Child placement 支持 `inherit`、`pinned` 与 `adaptive`；用户在 `forward()` 中始终直接调用 Child。`bind()` 创建稳定部署身份，Runtime 在调用时选择物理执行目标，不得通过逐次重新绑定绕过容量或改变调用身份。
17. **动态发现不改变逻辑执行图**：Pygent 不支持绕过 Binding 和 ExecutionPlan、按任意名称调用未声明 Agent、Module 或 ModelGroup 的开放式 Registry。服务发现可以为已声明的稳定逻辑目标动态选择 Worker、endpoint 或副本；对于 ExecutionPlan 已声明的逻辑 ModelGroup，Runtime 可以在其选择策略内解析不同的已验证 profile snapshot。这些选择只能改变具体模型部署，不能改变 Module graph、schema、授权边界、逻辑容量声明、必需 capability、父子身份或调用结果契约；新增逻辑 Agent、Module 或 ModelGroup requirement 必须生成新的 ExecutionPlan，新增独立任务则使用新的 Root Execution/Job。
18. **外部等待必须显式且谨慎**：`wait_external()` 只为短时、受管、可取消的外部信号提供等待边界；它暂停当前 `forward()` 及其同步等待链并持续占用 live execution、Task、调用栈和内存。每次等待必须受 deadline、waiter 上限和关闭清理约束；小时级或天级交互必须结束当前 Execution，并在反馈到达后创建新 Execution。
## 执行所有权与终结

19. **逻辑 Execution 与实际 attempt 分离**：execution_id 标识可查询、可取消、可订阅的逻辑执行，attempt_id 标识实际执行尝试。跨进程所有权通过 owner lease、fencing 和 heartbeat 验证，首次执行与恢复使用同一协议；过期 owner 不得继续发起业务工作，其受管持久提交必须被拒绝。租约失效后的取消不证明外部副作用已经停止。
20. **Handle 与所有权分离**：`start()` 创建逻辑 Execution 和 owner Task 后立即返回 Handle。独立观察者关闭订阅或停止等待不取得或转移 owner，也不主动终结执行；显式取消、拥有型 stream 退出和结构化父子取消按 Execution 契约传播。准入失败、deadline、取消和恢复继续由同一控制面观察。
21. **统一 deadline 与有界终结**：deadline 从 `start()` 提交开始，覆盖持久化、准备、资源 pin、准入、排队和业务执行，不由阶段或流进展延长。内部等待同时响应有效 deadline、取消与关闭；到期后只允许有限 cleanup grace 内的停止、回收和原子终结，不得继续业务工作。未确认清理或提交不得伪装成功；资源级关闭仍负责 join 和回收。配置与发布使用独立显式 deadline。
22. **准入具有唯一回滚责任**：模型 pin、资源 lease、容量 ticket、owner claim 与 history 状态的获取由同一执行所有者协调，成功必须有明确提交点。失败、取消或 deadline 按获取依赖的反向顺序释放资源；清理责任不能因调用方退出而丢失。外部设施故障时保留已提交恢复事实并报告未确认操作，通过声明的回收与恢复协议完成处理，不报告虚假释放。具体协调类与事务批处理方式属于实现。
23. **Journal 决定原子终态**：terminal span events、唯一 Execution terminal event、冻结 Outcome、终态 Snapshot 与 terminal_sequence 作为一个 finalization 事务提交。提交确认前不发布终态成功；设施故障不得生成与 Journal 不一致的第二套终态。订阅正常结束必须已经交付 terminal_sequence，不能仅凭 snapshot 提前退出。事件、游标与 Handle 的保留及读取范围由 Runtime SDK 定义，物理批处理和缓存不能静默丢失可读取事实。
24. **attach 与 recover 是不同权限**：`get_execution_handle(execution_id)` 只附着到既有逻辑 Execution 并观察、等待或请求取消，不创建 attempt；`recover()` 必须验证恢复资格、取得新的 fenced owner lease 并创建新 `attempt_id`。查询、附着、恢复和取消都通过统一 Execution backend 契约实现，本地、SQLite 与远程 transport 不得各自发明终态判断。
## 控制面与运行中输入

25. **控制面初始化与发布必须并发安全**：共享 store/resource 的打开使用 single-flight；相同 profile 内容的并发确保操作按稳定 digest 幂等合并，profile current pointer 与 default pointer 在同一事务发布。失败不得暴露半配置状态，也不得把配置等待隐藏在业务 Execution admission 之外。
26. **Execution Inbox 是托管 Runtime 必选协议**：所有内置 managed Runtime 必须为 `ExecutionHandle.send_input()` 和 `Module.receive_execution_inputs()` 提供一致的有界、幂等、有序、单消费者和可恢复语义。send、receive/seal 与 finalization 使用同一事务或锁顺序；`seal_if_empty=True` 原子完成空读和封闭。Runtime 只保存 opaque `kind/value`，不识别 ReAct operation、压缩或领域事件。direct execution 的 receive 固定为空，send 明确拒绝。
