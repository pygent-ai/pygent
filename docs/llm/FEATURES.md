# LLM 第一原则

本文从属于 [Pygent 0.2 第一原则](../FEATURES.md)。只能澄清，不能与其冲突。

1. **模型调用即 Module**：ModelCallLayer 使用统一的 Message、Context、`forward()`、直接执行、绑定和托管执行协议。
2. **定义与运行分离**：模型组需求、模型选择策略、重试策略和生成默认行为是不可变声明；托管 Runtime 可以在 admission 时从已声明模型组中选择一个经过验证的不可变 profile snapshot。允许的调用级生成参数覆盖是本次 Execution 的不可变输入，不修改 Module 或其默认配置。ModelInvoker 拥有 route、retry 与 fallback，ModelProviderAdapter 拥有 Provider 协议转换和错误归一。direct execution 使用显式配置的本地 adapter，连接生命周期和 Root 并发由调用方负责；managed Runtime 只治理选择、client 生命周期、资源、deadline、取消和调度，不解释 Provider 逻辑。
3. **Context 只读**：模型层返回完整 AIMessage 和原 Context，不隐式提交历史；“原 Context”包括具体 AgentContext 类型和全部用户字段，Provider 只消费基础模型投影。
4. **预算统一**：重试、fallback、deadline、attempt 与取消清理必须处于同一有界预算。只有上一 attempt 已确认结束后才能启动下一 attempt；若取消清理无法在预算内确认，结果为 `OUTCOME_UNKNOWN`，必须 fail closed，不得 retry 或 fallback。
5. **容量责任按模式划分**：direct execution 不提供跨调用的框架容量治理；managed execution 中，多个 Layer 指向同一资源时共享 Runtime 容量所有者。
6. **执行同源**：流式与非流式运行同一个 `forward()`，只改变事件观察方式。
7. **边界安全**：secret、连接、Provider 原始响应和内部异常不得泄露到公开值。
8. **Binding 模型门禁只属于托管执行**：Binding 可以选择透传模型流量，或增加 Binding/资源组级共享上限；两种托管模式都必须服从 deadline、取消和有界 live execution 约束。direct execution 的限流与外部 deadline 由调用方或 adapter 负责。
9. **Provider 逻辑不进入 Runtime**：Runtime 不构造 OpenAI、Anthropic 或其他 Provider 请求，也不解释其响应；Provider 差异只能存在于 LLM adapter。
10. **中间事件是固定协议**：reasoning、正文、ToolCall 生成、usage、attempt 与模型终态只能通过封闭的 `model.*` 事件集合输出；未知 Provider 增量必须被拒绝或显式忽略，不能伪装成完成事件。`model.completed` 只能在最终 Message、ToolCall、结构化输出和 usage 完成校验后发布。
11. **模型发现属于部署查询**：可选 ModelCatalog 只查询当前 endpoint 与 credential 可见的模型，不是 ModelCallLayer、ModelInvoker 或 Runtime 的执行能力。查询不得自动新增或更新 ModelGroup profile、改变 default profile 或移动 current pointer，也不参与 retry/fallback、模型容量或 `model.*` 事件。目录结果只有经过显式且受权的部署配置、授权与能力验证后，才能成为可执行 profile snapshot。

这里的“不可变声明”按值和执行边界理解：固定模型组是包含 routes/fallback 的不可变声明；托管模型组在 Agent 中是不可变的逻辑需求和选择边界，Binding 控制面可以为其维护多个命名 profile，每个 profile 的发布结果都是不可变 snapshot。Admission 根据显式 invocation/session 选择、sticky session 选择或 group default 选出一个 snapshot，并在执行开始前固定。选择不会修改 Module、需求、ExecutionPlan、RetryPolicy 或生成默认值；调用级生成覆盖是参与 effect identity 的不可变执行输入。Runtime 不在执行中追踪 default/current，也不承担 Provider 路由逻辑。具体约束见 [延迟与动态模型组规范](DYNAMIC_MODEL_GROUP_SPEC.md)。
