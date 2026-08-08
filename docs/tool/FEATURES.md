# Tool 第一原则

本文从属于 [Pygent 0.2 第一原则](../FEATURES.md)。只能澄清，不能与其冲突。

1. **工具定义是声明**：ToolDefinition 只描述模型可见的名称与 schema，不持有权限、执行策略或运行资源。
2. **统一消息端口**：ToolCallLayer 执行 `(AIMessage, Context) -> (ToolMessage, Context)`。
3. **批量结果单消息**：同一模型轮次的多个结果按调用顺序聚合为一个 ToolMessage。
4. **Context 只读**：工具层默认原样返回 Context，是否进入历史由组合层决定。
5. **授权与执行治理分离**：业务授权由用户开发的自定义授权 Module 或受信执行适配器决定；direct execution 中调用方与本地 adapter 负责资源和并发，managed Runtime 负责资源获取、并发、调度、取消、终态传播与清理；两者都不作出业务授权决策。
6. **适配器不分叉**：本地、MCP、Shell 或远程工具遵守同一公开调用协议。
7. **单次调用与 Binding 容量分层**：ToolCallLayer 的并发限制单次调用的工具 fan-out；direct execution 不增加跨 Root 门禁，调用方自行管理总并发。managed Binding 可以选择透传，或对其全部 Execution 增加共享工具总并发，两层限制分别生效。
8. **三层名称固定**：ToolDefinition 描述模型可见接口；ToolSpec 声明稳定身份、版本、副作用、幂等、timeout、资源、沙箱和所需权限，不保存 handler、secret 或活连接；ToolTask 表示一次已接纳的执行实例及其生命周期，并最终产生 ToolResult。ToolCallLayer 是编排这三层的 Module，不是第四层工具值。
9. **可见性不是授权**：Context 中的工具定义只控制本次模型可见集合；是否允许执行由用户开发的自定义授权 Module 或受信执行适配器决定。
10. **拒绝、结果与副作用可判定**：授权拒绝发生在 ToolTask admission 之前，不创建 ToolTask，但必须生成具有原 `call_id` 的 `ToolResult(status="rejected")` 并按调用顺序进入 ToolMessage。ToolResult 还必须区分错误类别、是否可重试以及外部副作用是否已提交；超时不得被等同于副作用未发生。
11. **同步与 detach 对所有工具统一**：普通工具与 Agent-backed Tool 都可以声明同步等待和独立任务两种生命周期，但模型不得通过 ToolCall 参数自行提升为独立任务；生命周期只能由应用或自定义授权 Module 决定。direct execution 只直接支持同步本地调用，独立生命周期由调用方自己的任务设施承载。managed execution 中，同步 Agent-backed Tool 可以作为当前 Execution 的结构化 Child；detach 必须为该调用创建具有独立身份和 admission 的 ToolTask，并立即返回 `ToolResult(status="detached", task=<ToolTask 公开快照>)`。ToolTask 是严格 JSON、不可变的公开任务描述，不是活句柄；后续使用其稳定 `task_id` 查询、取消和取得最终 ToolResult。需要 durable recovery 时由独立 Job 承载该 ToolTask。detach 后的调用不再是 Child，但仍受声明的 Binding、资源与 capability 治理；它不得通过 `detached Child` 绕过父子取消、join、容量和终态约束。
12. **沙箱需求与部署能力分离**：`ToolSpec.sandbox_profile` 只声明工具所需的稳定沙箱 profile，不证明 Runtime、Worker 或 executor 已经实施该隔离，也不得自动授予 `tool.sandbox.<profile>` capability。外部 E2B、Daytona、Modal、自托管容器或其他沙箱必须通过同一 `ToolExecutor` 扩展缝包装；provider client、credential、活 session、连接池和可变映射只属于受信部署 adapter，不进入 ToolSpec、Context、Module 图或 ExecutionPlan。managed Runtime 只能根据为精确 `(tool_id, version)` 注册且声明实际支持该 profile 的 sandbox-aware executor 派生能力，并在注册、bind/compile、detach admission 与 durable recovery 的最早可知阶段验证；`ToolKit.managed_layer(runtime, executor_factory=...)` 只批量缩短这一显式装配，不得把 direct `LocalToolExecutor` 升格为沙箱 executor，且整批必须先验证后注册。缺失时必须报告具体工具与 `tool.sandbox.<profile>`，不得折叠为笼统的 detach 不可用。direct execution 仍由调用方与显式 adapter 承担资源、并发、隔离真实性和清理责任。
