# Agent SDK

本文是 Agent 的第二级契约，必须服从 [Agent 第一原则](FEATURES.md)。框架演进应保持这些使用方式成立。

## 自定义 Agent

```python
class MyAgent(Module[UserMessage, AIMessage]):
    def __init__(self, model: ModelCallLayer, tools: ToolCallLayer):
        super().__init__()
        self.react = ReActLayer(
            model=model,
            tools=tools,
            max_steps=8,
            max_model_calls=8,
            max_tool_calls=32,
        )

    async def forward(
        self,
        message: UserMessage,
        context: Context,
    ) -> tuple[AIMessage, Context]:
        return await self.react(message, context)
```

## 自定义 AgentContext 与应用状态 Module

Agent 可以声明自己的 portable 生命周期状态，而不把状态挂到 Agent 实例。以下 `ToolState`、`FileState` 和 `StateModule` 全部是应用开发者示例，不是 Pygent 提供的框架类型或固定命名：

```python
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class AgentContext(Context):
    context_schema: ClassVar[str] = "my-agent.context"
    context_schema_version: ClassVar[int] = 1

    tool_state: ToolState = field(default_factory=ToolState)
    file_state: FileState = field(default_factory=FileState)


class StateModule(Module[UserMessage, Message]):
    async def forward(self, message: UserMessage, context: AgentContext):
        tool_prompt = compute_tool_prompt(context.tool_state)
        return Message(kind="tool.prompt.computed", content=tool_prompt), context


class StatefulAgent(Agent[UserMessage, AIMessage]):
    context_type = AgentContext

    def __init__(self, state_module: StateModule, react: ReActLayer):
        super().__init__()
        self.state_module = state_module
        self.react = react

    async def forward(self, message: UserMessage, context: AgentContext):
        tool_info, context = await self.state_module(message, context)
        message = process(message, tool_info)
        return await self.react(message, context)
```

`context_type` 是 managed、Worker 与 durable 部署的显式 Context 契约。`LocalRuntime.bind()` 自动派生并注册规范 dataclass codec；普通应用不需要手写 `ContextCodec.dataclass()` 或向 `LocalRuntime` 传 `context_codecs=`。低层入口仍保留给自定义 codec、非 Agent Root 与部署控制面。

示例中的 `tool_state`、`file_state` 可以替换为应用需要的任意领域字段；它们及对应 State 类型不属于 Pygent API。所有字段仍必须满足 [Context SDK](../context/SDK.md) 的 schema、版本和 portable 值约束。Store、连接、锁、manager、executor 与 provider client 不得进入 AgentContext。上例选择现有的 Message/Context 递推形状；这只是该 Module 的具体契约，不是普通 Module 的统一二元协议。只读取 Context 并返回其他值的本地辅助计算也可以建模为普通 Module。

## 自定义组合 Agent

```python
class CoordinatorAgent(Module[UserMessage, AIMessage]):
    def __init__(self, worker: MyAgent, reviewer: ReviewAgent):
        super().__init__()
        self.worker = worker
        self.reviewer = reviewer

    async def forward(self, message: UserMessage, context: Context):
        draft, context = await self.worker(message, context)
        answer, context = await self.reviewer(draft, context)
        return answer, context
```

是否把最终 Message 写入返回 Context 由用户 Agent 自己决定，框架不自动追加。内置与推荐 Agent 默认把已提交的最终 Message 加入返回 Context；上例中 `reviewer` 已返回包含 `answer` 的 Context，Coordinator 不得重复追加。如果自定义 Agent 选择不提交输出，必须在自身契约中明确说明。

父子关系来自属性声明，调用方式与普通 Layer 一致。direct execution 使用普通 `asyncio` 并发与取消；需要框架治理的阻塞、并行、容量和恢复语义时再阅读 [Runtime SDK](../runtime/SDK.md)。

## 直接调用

不需要分布式部署或框架级并发治理时，直接创建并执行 Agent：

```python
agent = MyAgent(model, tools)
message, context = await agent.invoke(message, context)

async with agent.stream(message, context) as stream:
    async for event in stream:
        ...
    message, context = await stream.final_result()
```

该模式不要求 Runtime、Binding 或 ExecutionOptions。多个 Root Agent 的并发、服务限流和外部 deadline 由调用方自行管理；`forward()` 内的子 Agent 仍以同样的 `await self.child(message, context)` 形式执行。

ReAct 在两种执行模式下都必须具有有限 `max_steps`、总模型调用数和总工具调用数。direct execution 的调用方负责用 `asyncio.timeout()` 或服务 deadline 限制整个 Root 生命周期；托管执行由有效 Execution deadline 与 Runtime 取消 scope 强制治理。

## 托管绑定和调用

只有需要框架级容量、资源、placement、结构化取消或恢复时，才先创建服务级 Binding 并接入 Root Agent：

```python
binding = runtime.create_binding(
    name="assistant-service",
    execution_capacity=run_capacity,
    model_capacity=model_capacity,
    tool_capacity=tool_capacity,
)
agent = binding.bind(MyAgent(model, tools))

message, context = await agent.invoke(message, context)

async with agent.stream(message, context) as stream:
    async for event in stream:
        ...
    message, context = await stream.final_result()
```

普通托管调用不需要显式 ExecutionOptions。需要覆盖 Binding 默认 deadline、提供调用身份或启用幂等/durable 协调时再传入：

```python
message, context = await agent.invoke(
    message,
    context,
    execution=ExecutionOptions(
        request_id=request_id,
        deadline=finite_deadline,
    ),
)
```

工具总调用数不能由 `max_steps` 代替，因为单次模型输出可以包含多个 ToolCall。托管取消继承当前 Runtime scope；Binding 默认策略与 ExecutionOptions 都没有有限 effective deadline 时，Runtime 必须拒绝启动需要 deadline 的 ReAct。

现有形式继续保留：

```python
agent = MyAgent(model, tools).bind(runtime, binding=binding)
```

Binding 属于服务部署域，不属于 `MyAgent` 类型。`MyAgent` 内部的 ReActLayer、ModelCallLayer、ToolCallLayer 和原始子 Agent 默认继承当前 Binding。只有需要独立治理边界时，才预先绑定 Child 或声明 placement policy，包括容量、资源、权限、安全、SLA、服务、部署策略或生命周期隔离。

## 显式绑定子 Agent

同一组合 Agent 可以接收原始或预绑定子 Agent，`forward()` 不区分两者：

```python
reviewer = ReviewAgent(model=review_model).bind(
    reviewer_runtime,
    binding=binding_reviewer,
)

coordinator = CoordinatorAgent(
    worker=MyAgent(model, tools),      # inherit
    reviewer=reviewer,                 # pinned
).bind(
    main_runtime,
    binding=binding_main,
)
```

Coordinator 内部仍然直接执行：

```python
draft, context = await self.worker(message, context)
answer, context = await self.reviewer(draft, context)
```

`worker` 继承 Main Binding；`reviewer` 在自己的 Runtime/Binding 上执行，但保持 Coordinator Execution 的 Child identity、deadline、取消和事件关系。自适应 placement 与只有远程契约的 `RemoteModule` 示例见 [Runtime SDK](../runtime/SDK.md#child-的三种放置方式)。

用户不实现第二个 stream 方法。

## 服务边界

```python
snapshot = await store.read(request.session_id)
context = AgentContext(
    messages=project_model_history(snapshot),
    metadata={"session_id": request.session_id},
    tool_state=snapshot.tool_state,
    file_state=snapshot.file_state,
)
message = UserMessage(content=request.text)

message, next_context = await agent.invoke(message, context)

await store.commit(
    request.session_id,
    snapshot.revision,
    tool_state=next_context.tool_state,
    file_state=next_context.file_state,
)
```

Agent 构造函数不接收当前 Session 或 Store。基础 `Context` 仍适用于只需要模型投影的 Agent；用户 AgentContext 用于显式流转额外的 portable 状态。

## 自定义 handoff 与审批

handoff 和审批使用普通类型化 Message，不要求 Agent 专用控制协议：

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class ApprovalRequiredMessage(Message):
    approval_id: str
    action: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ApprovalMessage(Message):
    approval_id: str
    approved: bool
```

进程内短等待写法如下；`wait_external()` 由 Runtime ExecutionScope 提供，用户不覆盖它：

```python
class ApprovalModule(Module[ApprovalRequiredMessage, ApprovalMessage]):
    async def forward(self, message, context):
        value = await self.wait_external(
            kind="approval",
            key=message.approval_id,
            request={"action": message.action},
            timeout=60.0,
        )

        decision = ApprovalMessage(
            approval_id=message.approval_id,
            approved=value["approved"],
        )
        return decision, context + decision
```

外部反馈按 `approval_id` 通过 Runtime 或共享信号适配器完成等待。`wait_external()` 会暂停当前 `forward()` 和同步等待它的 Parent Agent；虽然不会阻塞线程、event loop 或其他独立 Execution，但会持续保留 live execution、Task、调用栈、局部变量和内存。它必须同时受当前 Execution deadline 与局部 `timeout` 限制，effective deadline 取两者中更早者；如果两者都没有提供有限值，Runtime 必须拒绝注册 waiter。Binding 还必须限制 waiter 与 live execution 数，该能力只用于秒级或分钟级短等待。

长等待应返回 ApprovalRequiredMessage 并结束当前 Execution，服务保存 Context；反馈到达后，以 ApprovalMessage 和保存的当前有效 Context 创建新 Execution。Execution Inbox 可以在短时运行过程中向选择消费该 `kind` 的 Module 追加 portable 输入，但它不保存 Python 调用栈，也不替代长等待的新 Execution。完整调度与竞态要求见 [Runtime 外部信号受管等待](../runtime/README.md#外部信号受管等待)。

## ReAct 运行中 Projection Operation

托管 ReAct 固定消费 `react.projection.operation.v1`。开发者通过 `handle.send_input()` 发送由 `encode_react_projection_operation()` 编码的 `StandaloneUserMessage`、`AppendToolResultContent` 或 `ReplaceMessageProjection`；ReAct 在每次模型调用前以及最终返回前读取并按 input sequence 应用。`ReplaceMessageProjection` 使用 `Context.projection_revision` 做严格替换，或以 `rebase_appended=True` 保留 base revision 后可以证明为完整 Message 追加的尾部。解码、revision、replacement 或 pending ToolResult 校验失败会发出 `react.projection_operation.rejected`，不会终结 Execution。

该能力使 `ReActLayer.execution_requirements.effect_safety` 固定为 `MANAGED_EFFECTS`。direct execution 的 receive 固定为空，因此运行中 Projection Operation 只属于 bound/managed ReAct。
