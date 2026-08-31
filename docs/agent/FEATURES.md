# Agent 第一原则

本文从属于 [Pygent 0.2 第一原则](../FEATURES.md)。只能澄清，不能与其冲突。

1. **Agent 即 Module**：内置 Agent 与用户 Agent 不享有不同的执行协议。
2. **组合优先**：Agent 通过子 Module 和 `forward()` 表达业务数据流。
3. **上下文自决**：Agent 明确决定 Message 是否进入模型投影、用户 AgentContext 状态是否演进，以及各自的时机和顺序；框架不自动追加或合并。内置与推荐 Agent 默认把已提交的最终 Message 加入返回 Context，调用方不得重复追加。
4. **循环有界**：ReAct 的步数、调用数、deadline 与取消行为必须有界。
5. **能力归属清晰**：模型 fallback 属于模型层，工具并发属于工具层，Agent 不复制基础设施循环。
6. **定义无状态**：会话加载和提交位于 Agent 外部；请求与领域状态通过不可变 AgentContext 流转，同一 Agent 定义可并发复用。
7. **共享 Layer 安全**：Agent 可以直接调用并向子 Agent 或 Layer 注入同一 Module；每次调用保持独立运行身份。
8. **并发责任按执行模式划分**：direct execution 不启用框架级 Execution 容量，调用方自行使用 `asyncio`、服务限流器或外部设施管理 Root 与本地并发；托管执行中的所有 Agent Root 和 Child 受当前 Binding 的 live/runnable 上限、父子深度、fan-out、waiter 和公平调度约束。Agent 不为托管执行私建锁或 semaphore。
9. **父子 Agent 只在托管执行中继承 Binding**：Binding 是部署与资源治理域，不是 Agent 身份；direct execution 没有 Binding。托管执行中的原始子 Agent 默认继承 Parent Binding，只有需要独立治理边界时才使用预绑定 Child 或 placement policy，包括容量、资源、权限、安全、SLA、服务、部署策略或生命周期隔离。
10. **ReAct 解释自己的 Projection Operation**：Runtime Inbox 只交付 opaque 输入；`ReActLayer` 在每次模型调用前和最终返回前解释 `react.projection.operation.v1`。操作只有 ToolResult content append、独立 UserMessage 与带 revision 的 message projection replacement；拒绝只产生稳定事件，不扩展为公开生命周期 Hook、Handler、Policy 或 Runtime 业务分支。
11. **标准前台 Agent 仍是组合**：`PygentAgent` 只组合标准 ReAct、前台模型、Compressor 和工具 Module。System Prompt 与 Compression Prompt 属于不可变 Agent 定义；Agent 根据上下文窗口触发压缩。Compressor 是普通 Module；框架统一生成 Snapshot，只替换模型消息投影，并保留 pending current 和其他 Context 状态。`PygentAgent` 不负责长期历史的加载或持久化。ReAct 只接受子 model Module 显式返回的单次消息投影替换，要求 projection revision 精确前进并禁止该边界替换 System Prompt 或 tools。
