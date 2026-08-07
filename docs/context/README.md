# Context

阅读顺序：

1. [第一原则](FEATURES.md)
2. [SDK 使用](SDK.md)
3. 本文的详细契约

Context 是随 Module 数据流传递的当前有效上下文快照，不是 Session、版本仓库、依赖容器或服务对象。它只回答“本次 Module 调用当前看见什么”，不保存这些内容的历史版本、revision、审计轨迹或持久化提交状态。

## 数据形态

```python
@dataclass(frozen=True, slots=True)
class Context:
    system_prompt: str = ""
    messages: tuple[Message, ...] = ()
    tools: tuple[ToolDefinition, ...] = ()
    metadata: FrozenJsonObject = FrozenJsonObject()
```

- `system_prompt` 是本次数据流的系统指令。
- `messages` 是已经进入历史的 User、AI 和 Tool 消息。
- `tools` 是本次上下文和模型可见的工具定义，不是可信授权证明。
- `metadata` 是递归冻结、严格 JSON 可序列化的请求事实。

Context 不提供 prompt 构造、工具调用、保存或加载等领域方法。

## 历史演进

唯一基础操作是 `Context + Message -> Context`。不支持字符串、字典、`Context + Context` 或 `Message + Context`，避免产生隐式合并规则。

Message 可以声明可选稳定槽位 `slot`：

- `slot is None`：作为普通会话或处理增量追加，保留此前 Message。
- `slot is not None`：移除旧的同槽位 Message，再把新 Message 追加到末尾。

槽位不是 Agent 名、role 或 Tool 名；普通 UserMessage、AIMessage 和 ToolMessage 默认不设置槽位，因此同一角色的多轮消息不会互相覆盖。检索结果、当前计划或其他只需保留最新值的 `Message(kind=..., data=...)` 领域增量可以使用类似 `retrieval/current` 的领域槽位。

当前 Message 不会因为进入 Module 就自动成为历史：

- ModelCallLayer 和 ToolCallLayer 默认原样返回 Context。
- ReActLayer 或用户 Agent 决定何时追加 User、AI、Tool 消息。
- 服务从外部 Store 加载历史，并在 Execution 成功后显式提交返回值。

历史始终在调用栈中以值传递，不得保存到 Module 实例。

Context 只负责保存工具可见性。执行某个工具是否被允许，必须由用户开发的自定义授权 Module 或受信执行适配器判断；从 Store 或客户端加载的 `context.tools` 不能单独授予权限。

## JSON 值边界

Message metadata、Context metadata、工具参数、工具结果和公开事件数据只接受 JSON 标量、数组和字符串键对象。构造时递归转换为不可变容器，并拒绝 NaN、Infinity、bytes、连接、锁、协程、handler、非字符串对象键和其他任意 Python 对象。业务状态如需引用外部对象，应保存稳定字符串引用，而不是把活对象放入 Context。

Message 与 Context 的公开类型边界是封闭的：用户领域扩展使用 `Message(kind=..., data=...)`，不能通过伪造 Python module 名注册子类。Context 中的 ToolDefinition、ToolCall/ToolResult 等嵌套公开值同样使用封闭的精确类型，禁止子类附加 handler 或其他 codec 会丢失的字段。已冻结 JSON 子树在组合时仍重新计入整体深度、规模与循环校验。

## 与执行 checkpoint 的边界

Context 可以作为持久化 checkpoint 的一部分，但 Context 本身不是 Execution checkpoint、coroutine continuation 或 Tool 副作用日志。仅持久化 Context 不能恢复 `forward()` 的局部变量、当前源码位置、普通 `await` 状态、attempt 身份或尚未提交的外部副作用。

支持 durable recovery 的 Runtime 还必须保存恢复边界、ExecutionPlan 与代码版本、调用和 attempt 身份、必要的边界输入输出以及副作用/事件去重记录。Context 持久化可以支持从 Root 或 Module 边界重新执行；是否支持更细粒度恢复必须由显式 checkpoint policy 声明，详细语义见 [Runtime 持久化边界](../runtime/DURABILITY.md)。
