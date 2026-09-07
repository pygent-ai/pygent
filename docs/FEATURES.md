# Pygent 0.3 第一原则

Pygent 以可共享的 Module 定义组合计算，以显式输入输出传递调用状态，以 Execution 统一一次执行的结果、事件和控制，以可选 Runtime 承担部署治理与可验证恢复。

本文规定架构不变量。模块第一原则展开责任边界，SDK 和 [Execution contract](EXECUTION.md) 定义公开调用及事件协议，详细规范定义部署能力和故障处理，验收矩阵记录实现证据。下级文档不得改变上级语义；内部类名、存储布局和调度算法属于实现设计。文档契约版本为 0.3，包版本和各公开 wire/schema 版本分别管理。

## 定义、调用与状态

1. **统一计算抽象**：Agent、Layer 与用户组合都只是 Module。内置能力与用户 Module 使用同一执行和扩展协议；adapter、codec、Store 与部署控制面按自己的责任提供基础设施。
2. **调用自由**：`forward()` 的位置参数、关键字参数和结果由 Module 自己声明。Message 与 Context 是可选公开值，不是普通 Module 的强制端口。RecurrentModule 表达显式消费 state 并产生 next state 的递推语义，不固定业务值类型或结果结构。
3. **无隐藏调用状态**：Module 可以持有定义配置、子 Module 和显式部署资源，不持有某次调用的输入、局部结果、recurrent state、请求或业务会话状态。调用状态通过参数、局部变量与返回值流转；同一 Module 定义可以并发复用。
4. **定义共享，调用独立**：同一 Module 可以被多条属性路径引用，每次调用拥有独立身份与局部状态。用户通过声明子 Module 和一个 `forward()` 表达计算；Child 由当前执行范围建立调用关系，用户无需操作 ExecutionScope。
5. **业务持久状态外置**：业务服务负责状态加载、权威提交、历史版本与冲突处理。Context 是显式的不可变调用值；Runtime 可以保存恢复所需的输入、结果和执行事实，但不得成为第二个业务会话状态源。
6. **公开值可移植**：Message、Context 及其用户子类、ToolDefinition、ToolSpec、ToolTask、ToolResult 与 ExecutionEvent 的扩展数据具有稳定 schema，并由严格、有限、递归冻结的 JSON 值组成，不携带连接、锁、协程、handler、Store、client 或任意活 Python 对象。本地 direct Module 可以使用普通 Python 值；跨进程与恢复只使用已声明并验证的 codec，不使用 pickle 或按 Python 类名恢复对象。

## 执行与部署

7. **一次执行，同源结果**：每次业务执行只有一个逻辑 Execution；结果等待、事件观察和控制操作不得重复启动计算。`start()` 创建可立即观察和取消的执行，`invoke()` 投影为 `start() + result()`，`stream()` 投影为拥有执行的 `start() + subscribe()`。一种模式支持某个调用时，Root、Child 与流式最终结果都来自同一个 `forward()` 图。
8. **部署按需接入**：direct execution 无需 Runtime 或 Binding，调用方管理 Root 并发、外部 deadline 和本地资源生命周期。managed execution 由 Runtime 统一治理容量、调度、取消、placement 和资源生命周期。不同模式的能力范围必须明确；同一调用进入受支持的部署边界后，业务输入输出语义不因位置改变。
9. **Binding 是治理域**：Binding 聚合部署策略与共享资源身份，不代表 Agent 身份，也不与 Module 一一对应。结构化 Child 默认继承 Parent Binding；独立的容量、权限、资源、SLA 或部署边界通过预绑定 Child 或 placement 声明。策略声明与队列、锁、lease、client 等运行状态分离。
10. **资源责任唯一且作用域明确**：托管 Runtime 按资源身份治理共享容量，区分 Execution、Model 与 Tool 的容量平面。等待、恢复调度和释放遵循结构化所有权；局部容量不冒充部署全局容量。Provider 协议与 retry/fallback 属于 LLM，工具授权属于业务 Module 或受信 adapter，Runtime 不解释这些业务规则。
11. **父子生命周期结构化**：Child 继承执行 lineage、有效 deadline、取消与终态责任，Parent 退出前停止并回收未完成 Child。独立任务必须经过显式 admission 获得独立身份和资源治理，不能以 detached Child 绕过结构化执行。handoff、审批和领域终止可以由 Message 与 Module 组合表达，不隐式赋予 coroutine 持久挂起能力。

## 所有权、终结与恢复

12. **有效 owner 唯一**：每个实际 attempt 只有一个有效执行 owner。需要跨进程恢复时，首次与恢复 attempt 都必须取得 owner lease 与 fencing token；受管持久提交验证有效所有权并拒绝过期 token。失去所有权后必须停止发起新业务工作并取消清理；本地 Task 退出不能证明远端副作用停止，未确认操作仍按幂等和结果未知契约处理。
13. **业务预算统一，清理责任持续**：Execution deadline 从提交开始，覆盖准备、pin、准入、排队、业务执行和终结，不因阶段切换、重试或流式进展重新计时。到期后不再开始新的业务工作；停止、回收与原子终结只使用明确且有硬上限的 cleanup grace，不成为额外业务预算。未确认完成不得报告成功、已释放或安全可重试。资源级关闭仍须履行其 join 与回收责任；配置和发布使用独立显式 deadline。
14. **Journal 是终态权威**：terminal span events、唯一 Execution terminal event、冻结 Outcome、终态 Snapshot 与 terminal sequence 原子提交。提交确认前不得发布成功终态或伪造持久结果；提交失败必须保留故障及已有恢复事实，由声明的恢复协议处理。订阅只有交付 terminal sequence 才能正常结束；读取失败或取消观察不能伪装为正常完成。
15. **观察与执行所有权分离**：Handle 是稳定控制面引用，不拥有业务 coroutine。attach 可以观察、等待或请求取消，不创建 attempt；recover 必须另行验证资格、取得有效 owner 并创建新 attempt。独立观察者退出不改变执行生命周期；拥有型 stream 退出和结构化父子取消按各自所有权传播。具体等待与取消方式由统一 Execution 契约定义。
16. **恢复能力可验证**：恢复只属于声明该能力的 managed Runtime。普通活跃 coroutine 不保证迁移或恢复；Runtime 仅从已验证的边界，以兼容的计划、schema、输入和精确资源版本重建执行。受管 effect 重放已提交结果，未知副作用依据幂等、查询或补偿能力处理，不笼统承诺 exactly-once；不得以不可验证的重试替代声明的恢复保证。
17. **模型 attempt 串行且结果诚实**：新 Provider attempt 只能在上一 attempt 已确认结束后开始。清理未确认时进入既有 `OUTCOME_UNKNOWN` 与 client 隔离流程，禁止 retry/fallback。流式输出后的重试只允许 Execution 契约明确列出的情形，并通过 `model.output.reset` 显式撤回上一 attempt 的暂存增量；不得把失败输出或不确定副作用当作成功结果。
18. **运行中输入分层解释**：managed Runtime 负责有界 Execution Input 的身份、顺序、幂等、单消费者、存储与终态竞态；Module 解释 `kind` 与 `value`。Runtime 不解释 ReAct Projection Operation，也不把 Inbox 变成第二份业务 Context。direct execution 不接收外部输入。

各模式支持的调用形状与能力见 [验收矩阵](runtime/ACCEPTANCE.md)，持久化故障边界见 [Runtime Durability](runtime/DURABILITY.md)。[透明恢复与确定性重放](runtime/REPLAY.md) 是具体恢复策略，不是额外的架构保证。
