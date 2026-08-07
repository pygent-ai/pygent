# Agent

阅读顺序：

1. 新用户先走[渐进式开发教程](TUTORIAL.md)
2. [第一原则](FEATURES.md)
3. [SDK 使用](SDK.md)
4. 本文的详细契约

Agent 是 Module 的语义名称，不是另一套框架对象。内置 Agent 和用户 Agent 使用相同的继承、组合、直接执行、绑定与托管执行协议。

## ReActLayer

ReActLayer 是可使用、继承或替换的普通 Module。它接收 `(UserMessage, Context)`，执行有界循环并返回 `(AIMessage, Context)`：

1. 预处理用户消息，把当前 UserMessage 与此前 Context 分开传给 ModelCallLayer。
2. 模型产生工具调用时，把当前 AIMessage 与此前 Context 分开传给 ToolCallLayer。
3. 把已经消费的 UserMessage、AIMessage 和 ToolMessage 按顺序加入新 Context，再进入下一轮。
4. 完成时把最终 AIMessage 加入新 Context 并返回。

ModelCallLayer 和 ToolCallLayer 本身不追加历史。ReAct 拥有循环的上下文演进语义，但只能构造新 Context。

## 职责边界

ReAct 负责循环控制和上下文演进，不重复模型 fallback 或工具并发。循环步数、模型调用数、工具调用数、deadline 和取消必须有界。

Agent 只保存配置与子 Module。业务服务负责加载长期状态、构造 Context，并在成功后提交返回历史。

ReActLayer 直接接收模型层和工具层实例，自己的循环策略由 `max_steps` 等不可变参数声明：

```python
react = ReActLayer(
    model=model,
    tools=tools,
    max_steps=8,
)
```

同一个 `model` 可以既是 Agent 的直接子 Module，又被注入 ReActLayer。direct execution 保留共享定义但不提供框架级共享容量，调用方负责并发与本地资源；managed Runtime 为 Agent 直接调用和 ReAct 内部的每次调用建立独立调用身份，并按解析出的物理资源身份共享容量。

## 组合与调度

Agent 属性形成定义图，但 Agent 不拥有专用调度器或专用父子 API。direct execution 中，内部 direct scope 只保持 Child 调用、事件和本地取消语义，用户自行管理 `asyncio` 并发；managed execution 中，Agent 与其他 Module 使用相同的 [Runtime 父子调度契约](../runtime/README.md)，并形成结构化子 Execution，Blocking Child、Parallel Child、deadline、取消和容量规则由 Binding 与 Runtime 定义。

Binding 不与 Agent 一一对应，且 direct execution 不创建 Binding。托管服务中的 CoordinatorAgent 及其 WorkerAgent、ReviewAgent 通常共同属于服务级 Binding；Coordinator 在 `forward()` 中调用原始子 Agent 时，子调用自动继承当前 Binding。只有需要独立治理边界时，才把预绑定 BoundModule 作为子依赖，或由 Parent Binding placement 固定/动态选择执行目标，包括容量、资源、权限、安全、SLA、服务、部署策略或生命周期隔离；两者都不改变 `forward()` 的直接调用形式，也不把 Child 变成新 Root。

## Handoff 与 human-in-the-loop

Pygent 不为 Agent 冻结一套专用 handoff、审批或终止对象。用户可以声明 `HandoffMessage`、`ApprovalRequiredMessage`、`ApprovalMessage` 等领域 Message，并用普通 Module 决定路由、校验和上下文演进。

已经运行的 `forward()` 不能在任意位置被外部调用者注入新参数；反馈必须到达一个事先声明的等待边界。支持两种模式：

1. **进程内短等待**：自定义 ApprovalModule 创建审批 ID，通过 Runtime 的 [`wait_external()`](../runtime/README.md#外部信号受管等待) 等待；服务收到用户反馈后按审批 ID 完成该等待，`await approval_module(...)` 随后返回 ApprovalMessage。该方式会暂停当前 `forward()` 及同步等待它的 Parent 调用链，持续占用 live execution、Task、调用栈和内存；它不阻塞线程或其他独立 Execution，但要求原进程和调用栈持续存活，只适合有 deadline 和 waiter 上限的短等待，不提供故障恢复。
2. **跨请求长等待**：Module 返回 ApprovalRequiredMessage 或发布审批事件后结束当前 Execution；业务服务外置保存审批事实和当前 Context。用户反馈到达后，服务构造 ApprovalMessage，并以保存的当前有效 Context 启动新 Execution。这是没有 durable Runtime 时的推荐生产模式。

自定义 Module 可以表达上述业务语义，但不能仅靠普通 Python `await` 获得跨进程持久挂起、Worker 故障恢复或调用栈迁移能力。handoff 若只是当前执行树中的子 Module 路由，可以在同一 Execution 内完成；若转交后需要独立生命周期，应结束当前 Execution，并由服务创建新的 Root Execution。
