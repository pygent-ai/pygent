# Module SDK

本文是 Module 的第二级契约，必须服从 [Module 第一原则](FEATURES.md)与[调用与递推状态原则](CALL_CONTRACT_SPEC.md)。框架演进应保持这些使用方式成立。

## 定义和组合

```python
class Normalize(Module):
    async def forward(self, value: float, *, scale: float = 1.0) -> float:
        return value * scale


class MyAgent(RecurrentModule):
    def __init__(self, react: ReActLayer):
        super().__init__()
        self.react = react

    async def forward(
        self,
        message: UserMessage,
        context: Context,
    ) -> tuple[AIMessage, Context]:
        return await self.react(message, context)
```

用户只实现 `forward()`。把 Module 赋给属性即声明依赖关系；同一实例可以被多个属性路径共享。

普通 Module 的参数和结果由自己的 `forward()` 定义。需要显式状态递推时可以继承标准 RecurrentModule；state 可以是任意用户类型，内置 Agent/LLM 组合通常使用基础 `Context` 或满足 [Context SDK](../context/SDK.md#定义用户-agentcontext) 的 `AgentContext`。普通辅助计算可以建模为 Module，不需要伪装成 Message/Context 转移。

Module 图必须先完整构造，再进入执行生命周期。第一次调用 Root 的 `invoke()`、创建 `stream()` 或执行 `bind()` 时，框架会递归冻结全部原始子 Module；之后不能重新赋值或删除任何 Module 属性：

```python
agent = MyAgent(react)
bound = runtime.bind(agent)

# RuntimeError: definition is frozen
agent.react = another_react
```

需要修改策略或依赖时创建一个新 Module 并重新绑定。`forward()` 不能通过 `self.current_message = message` 保存请求态，也不能通过预先创建的 `self.history.append(...)` 绕过：direct 与 managed 在首次 freeze 前共用 validator，public/private raw mutable container 都会拒绝。用户 Module 的 class attributes 同样属于定义：不可变 class policy 进入 identity，mutable/callable class state 被拒绝，freeze 后修改 class policy 会让后续 direct/managed admission fail-closed。

冻结不深入明确声明的外部 adapter。Module 必须逐项列出 trusted 部署资源：

```python
class ModelLayer(Module[UserMessage, AIMessage]):
    trusted_live_resource_attributes = ("client",)

    def __init__(self, client):
        super().__init__()
        self.client = client
```

声明值必须是 opaque adapter/resource，裸 dict/list 不能作为 trusted live resource。框架信任 adapter 只维护共享 client、连接池、telemetry 或测试同步等部署生命周期状态；它不得保存请求 Message、Context、结果或业务状态。存入 Module 属性的 callable 不会获得隐式豁免：它必须列入 `trusted_live_resource_attributes`，或由 `execution_plan_config()` 的严格 snapshot 记录稳定行为声明；否则 direct freeze 与 bind 都会拒绝。可信 callable/adapter 仍不得成为请求状态旁路。

Module stored state 必须能完整验证并进入 ExecutionPlan identity。直接支持的值是不可变、严格可移植的标量、Enum、tuple、`FrozenJsonObject` 和 frozen dataclass。普通 dict/list 或未知 Python 对象会在 direct freeze 与 bind/compile 时报错，而不是被忽略；private 字段也遵守同一规则。测试探针、部署 client/registry 等不属于定义的 instrumentation 使用 opaque adapter，并列入 `trusted_live_resource_attributes`。复杂定义可以提供显式 snapshot：

```python
class RoutedAgent(Module[UserMessage, AIMessage]):
    trusted_live_resource_attributes = ("client",)

    def __init__(self, routes, client):
        super().__init__()
        self._routes = tuple(routes)
        self.client = client

    def execution_plan_config(self):
        return {
            "routes": [
                {"name": route.name, "model": route.model}
                for route in self._routes
            ]
        }
```

Hook 返回值是 freeze/编译时严格 JSON snapshot，允许新建 dict/list 容器，但其存储来源自身仍须不可变，且 snapshot 不能包含锁、连接、handler 或其他 Python 对象。Hook declaration 不会替代普通 stored attributes；二者同时进入 plan identity，因此即使 hook 自身不变，普通不可变策略值变化也会产生新的 `graph_hash`。Runtime 启动前会重新编译并比较 plan identity，因此通过非公开手段替换私有声明源仍会被识别为 definition drift；trusted live-resource 内部状态则明确不属于执行语义 identity。

领域增量使用统一、可移植的 Message 信封，不定义 Python 子类：

```python
approval = Message(
    kind="approval.requested",
    content="请审批发布操作",
    data={"operation_id": "op-42", "required_scopes": ["publish"]},
    slot="approval/current",
)
```

`kind` 必须是非空稳定标识；`data` 必须是 JSON 对象，并会在构造边界防御性复制和递归冻结。该形式能由 Runtime wire codec 无损往返。框架拒绝任意 Message 子类作为 portable 扩展；普通本地 Module 可以传递其他 Python 值，但跨进程边界必须使用显式 schema 与 codec。Agent 生命周期状态可以使用具有稳定 schema、版本、frozen/slots 与 portable 字段的 Context 子类。

内置 Module 使用仅关键字构造参数。组合 Module 直接接收子 Module，自己的策略使用不可变值或标量声明：

```python
model = ModelCallLayer(
    model_group=model_group,
    retry_policy=retry_policy,
    generation=generation,
)
tools = ToolCallLayer(tools=tool_definitions, max_concurrency=16)
react = ReActLayer(model=model, tools=tools, max_steps=8)
```

如果 `model` 同时赋给 Agent 和 `react.model`，它仍是一个共享定义；每次 `await` 调用是独立执行，物理资源容量由 Runtime 按资源身份统一治理。

## 直接执行

不需要框架管理并发、部署或恢复时，Module 本身就是 Root 执行入口：

```python
agent = MyAgent(react)
message, context = await agent.invoke(
    UserMessage(content="北京天气怎么样？"),
    context,
)
```

调用方不创建 Runtime、Binding 或 ExecutionOptions。框架只在本次调用内部建立 direct execution scope，使子 Module 调用、事件和本地取消传播使用同一个 `forward()` 图；Root 并发、外部 deadline 和资源生命周期由调用方使用 `asyncio` 或服务设施管理。

`await module(*args, **kwargs)` 只表示活动 execution scope 内的 Child 调用；Root 必须使用 `module.invoke()` 或 `module.stream()`。这样同一调用语法不会在 scope 内外分别表示 Child 和 Root。

直接执行只承诺本地、瞬时语义，不承诺 Binding 容量、远程 placement、durable retry、跨进程恢复或可重连事件。需要这些能力时使用托管执行。

## 托管非流式执行

```python
binding = runtime.create_binding(
    name="assistant-service",
    execution_capacity=run_capacity,
    model_capacity=model_capacity,
    tool_capacity=tool_capacity,
)
bound = binding.bind(MyAgent(react))

message, context = await bound.invoke(
    UserMessage(content="北京天气怎么样？"),
    context,
)

print(message.content)
```

`runtime.create_binding(...).bind(module)` 是推荐形式。现有写法继续有效：

```python
bound = MyAgent(react).bind(runtime, binding=binding)
```

两种形式都只创建可执行的绑定句柄，不立即创建 Execution。该句柄在执行树外通过 `.invoke()` 或 `.stream()` 调用时创建托管 Root；作为 `ModuleDependency` 在 Parent `forward()` 中直接调用时创建托管 Child。`forward()` 中调用的原始子 Module 默认继承当前 Binding，不创建新的 Root；显式预绑定 Child 与 placement policy 的隔离形式见下文。

普通调用可以省略 `ExecutionOptions`；Runtime 生成请求身份并使用 Binding 默认策略。只有需要显式 deadline、调用身份、幂等或 durable 协调时才传入 `execution=ExecutionOptions(...)`。`execution` 是现有 Root 执行入口的框架控制参数，不属于 `forward()` 的业务参数。

## 原始与预绑定 Child

组合 Module 可以接受统一的 `ModuleDependency`，因此调用代码不区分本地、预绑定或远程依赖：

```python
class ReviewPipeline(Module[UserMessage, AIMessage]):
    def __init__(self, reviewer: ModuleDependency):
        super().__init__()
        self.reviewer = reviewer

    async def forward(self, message: UserMessage, context: Context):
        return await self.reviewer(message, context)
```

传入原始 Module 时默认继承 Parent Runtime/Binding：

```python
pipeline = ReviewPipeline(reviewer=ReviewerAgent())
bound = pipeline.bind(main_runtime, binding=binding_main)
```

仅当 reviewer 需要容量、资源、权限、安全、SLA、服务、部署策略或生命周期等独立治理边界时，才传入 BoundModule 并固定使用显式 Child Runtime/Binding；普通情况应传入原始 Module 并继承 Parent Binding。预绑定句柄在 Parent 中调用时仍保持 Child 身份：

```python
reviewer = ReviewerAgent().bind(
    reviewer_runtime,
    binding=binding_reviewer,
)
pipeline = ReviewPipeline(reviewer=reviewer)
bound = pipeline.bind(main_runtime, binding=binding_main)
```

`forward()` 内只直接调用依赖；`.invoke()` 和 `.stream()` 只用于在执行树外创建 Root。三种 placement 与 RemoteModule 的完整示例见 [Runtime SDK](../runtime/SDK.md#child-的三种放置方式)。

未绑定 Parent 以 direct 模式执行时，显式预绑定 Child 保留其 Runtime/Binding，并作为该部署域中的独立 managed Root 执行；direct Parent 等待该 Child 声明的结果并在取消时清理 Child Execution。该桥接不会为 direct Parent 本身创建 Binding。

## 同型结果与流式执行

```python
# direct execution
async with agent.stream(message, context) as stream:
    async for event in stream:
        ...
    message, context = await stream.final_result()

# managed execution
async with bound.stream(message, context) as stream:
    async for event in stream:
        if event.kind == "model.output.reset":
            reset_rendered_output(event.data["route_id"], event.data["attempt"])
        elif event.kind == "model.text.delta":
            print(event.data["text"], end="")

    message, context = await stream.final_result()
```

direct `invoke()`、本地 Child 调用和 `stream().final_result()` 返回具体 Module 声明的结果类型。上例使用当前 managed Runtime 已支持的 Message/Context 具体契约，因此托管结果仍是 `(message, context)`；普通 Module 可以在本地返回其他类型，但本次变更不扩展 managed/remote 结果协议。

托管执行的 `execution_id`、attempt、状态、usage、取消、后台继续和可重连订阅属于独立 Execution Handle 控制面。需要这些信息的调用方显式进入该高级 API；普通 `invoke()` 不因运行元数据而返回 `ExecutionResult` 包装。

## 自定义流式内容

```python
class SearchAgent(Module[UserMessage, AIMessage]):
    async def forward(
        self,
        message: UserMessage,
        context: Context,
    ) -> tuple[AIMessage, Context]:
        await self.emit(
            kind="search.progress",
            data={"completed": 1, "total": 3},
        )

        answer = AIMessage(content="查询完成")
        return answer, context + message + answer
```

`emit()` 是继承的运行能力，不是第二个业务入口。用户只提供事件类型以及严格、有限的 JSON 值；Runtime 在 `emit()` 边界校验、防御性复制并递归冻结数据，再补充身份、路径和顺序。不得使用 pickle、自定义 encoder 或活对象放宽 ExecutionEvent 的公开值边界。

## Infrastructure Module SPI

普通业务 Module 继续只使用 `forward()`、Child、`emit()`、`gather()` 和 `wait_external()` 等高层能力，不导入或构造 ExecutionScope。实现模型、工具、数据库或其他部署适配层的用户自定义 Infrastructure Module，可以从 `pygent.core` 导入 `Infrastructure`、`current_infrastructure()` 与 `active_infrastructure()`，使用与内置 Model/Tool Layer 完全相同的公共 SPI：

- `deadline`：当前 effective deadline；
- `model_permit()` / `tool_permit()`：遵守 Binding 与物理资源容量，并执行 Execution lease handoff；
- `resolve_model_invoker()` / `resolve_tool_registry()`：只解析部署注入的活资源，不在 Runtime 内实现 Provider 或 executor；
- `tool_idempotency_key()`：从当前稳定 Execution/Module/call identity 派生工具幂等 key；
- `execute_effect()`：接收严格 `EffectSpec` 和 JSON request，按稳定 module path、call index、effect type、副作用、幂等与 retry policy 执行或重放受管 effect；
- `gather()` 与 `submit_tool_task()`：分别接入结构化并行和独立 ToolTask admission。

direct scope 只提供本地边界：permit passthrough，effect 执行当前 operation 且不持久重放，部署 resolver、Execution 幂等身份和 detached admission 不可用。基础设施 Module 不得把 direct 的降级行为描述为 managed durability。创建独立 owner Task 时使用 `independent_execution()` 隔离 Parent execution context；普通应用不需要调用它。

`EffectSpec` 必须声明 `EffectSideEffect`、`EffectIdempotency` 与 `EffectRetryPolicy`。`REQUIRES_KEY` 必须同时提供非空稳定 `idempotency_key`；非幂等 write/external 不能声明 `REPLAY_SAFE`。durable history 在调用 operation 前提交 `started`，成功后原子进入 `completed` 并保存 result。恢复看到未完成 `started` 时，只有 `REPLAY_SAFE` 且实际满足只读/固有幂等/稳定 key 的 effect 可重放；其他 effect 标记 `unknown` 并抛出 `EffectRecoveryUnknown`。因此仅声明 `MANAGED_EFFECTS` 不能把未分类第三方写入提升为可安全重放。
