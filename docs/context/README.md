# Context

阅读顺序：

1. [第一原则](FEATURES.md)
2. [SDK 使用](SDK.md)
3. 本文的详细契约

Context 是框架提供的不可变 Agent 上下文值。它可以作为普通 Module 的输入或输出，也可以被 RecurrentModule 选作 state；Context 本身不依赖 RecurrentModule。基础 `Context` 定义当前模型可见投影；用户可以用具有稳定 schema、版本和 portable 字段的子类增加有限历史视图与领域状态。Context 不是依赖容器或服务对象，也不负责长期完整历史、权威持久化、revision、审计和冲突提交。

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

这些基础字段是模型 Layer 的明确投影。用户 Context 子类的额外字段不会自动进入模型请求。Context 不提供工具调用、保存或加载等服务方法。

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

## 用户 AgentContext

用户 Context 子类是受约束的 portable value，而不是任意 Python 扩展点：必须是 frozen/slots dataclass，声明稳定 `context_schema` 和正整数 `context_schema_version`，全部实例字段都可由严格 JSON 编码。框架的 `replace()`、消息追加、Child、wire 和 checkpoint 必须保留具体类型与全部字段；未知或不兼容 schema 必须在 admission、发送或恢复前失败。

具体字段协议由 `ContextCodec.dataclass()` 规范生成。Agent 通过 `context_type` 显式声明 Context 类型后，`LocalRuntime.bind()` 在编译阶段自动派生并注册 codec；自定义 codec、非 Agent Root 和部署控制面仍可使用低层显式注册。通用 Message/Context wire schema 只提供带 discriminator 的信封，不会抹平用户字段。完整注册和部署规则见 [Context SDK](SDK.md#schema-与-codec-注册)。

用户可以重载 `__add__()`，使继承的 `context += message` 在重新绑定新值时同步演进自定义状态；也可以定义返回新值的 `__iadd__()`。两者都不得原地修改 frozen Context，且必须保持基础 Message/slot、portable 和无隐藏 I/O 契约。普通子类无需重载：基础 `__add__()` 已保留具体 Context 类型和全部字段。

完整示例见 [Context SDK](SDK.md#定义用户-agentcontext)。并行 Child 各自接收同一个不可变输入，Parent 显式合并返回状态；Runtime 不提供默认 last-writer-wins 或字段级自动合并。

## JSON 值边界

Message metadata、Context metadata、工具参数、工具结果和公开事件数据只接受 JSON 标量、数组和字符串键对象。构造时递归转换为不可变容器，并拒绝 NaN、Infinity、bytes、连接、锁、协程、handler、非字符串对象键和其他任意 Python 对象。业务状态如需引用外部对象，应保存稳定字符串引用，而不是把活对象放入 Context。

Message 的公开类型边界保持封闭：用户领域增量使用 `Message(kind=..., data=...)`。Context 则只允许经过 schema/codec 验证的受约束子类；Python 类名或伪造 module 名不能注册协议。Context 中的 ToolDefinition、ToolCall/ToolResult 等嵌套公开值仍使用封闭的精确类型，禁止附加 handler 或其他 codec 会丢失的字段。已冻结 JSON 子树在组合时仍重新计入整体深度、规模与循环校验。

## 与执行 checkpoint 的边界

Context 可以作为持久化 checkpoint 的一部分，但 Context 本身不是 Execution checkpoint、coroutine continuation 或 Tool 副作用日志。仅持久化 Context 不能恢复 `forward()` 的局部变量、当前源码位置、普通 `await` 状态、attempt 身份或尚未提交的外部副作用。

支持 durable recovery 的 Runtime 还必须保存恢复边界、ExecutionPlan 与代码版本、调用和 attempt 身份、必要的边界输入输出以及副作用/事件去重记录。Context 持久化可以支持从 Root 或 Module 边界重新执行；是否支持更细粒度恢复必须由显式 checkpoint policy 声明，详细语义见 [Runtime 持久化边界](../runtime/DURABILITY.md)。
