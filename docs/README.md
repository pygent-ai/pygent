# Pygent 0.2

[FEATURES.md](FEATURES.md) 是 0.2 的最高契约。其余文档只负责展开，不得改变第一原则的语义。统一执行控制面、事件信封与兼容边界见 [Execution contract](EXECUTION.md)。

[第一原则验收矩阵](runtime/ACCEPTANCE.md) 将这些契约映射到可重复执行的测试、发布门禁与可选在线验收。

每个模块独占一个目录，并使用相同的文档层级：

1. 模块 `FEATURES.md`：模块第一原则。
2. 模块 `SDK.md`：用户使用方式，是模块第二级契约。
3. 模块 `README.md`：详细边界和运行语义。

发生冲突时依次服从：总第一原则、模块第一原则、SDK 示例、详细说明、测试与实现。

## 学习路线

第一次使用 Pygent，先阅读[渐进式 Agent 开发教程](agent/TUTORIAL.md)。它通过同一个可运行项目依次介绍 Module/Context、模型、Python 工具、ReAct、stream、Runtime 和动态模型 profile；默认离线运行，不需要模型密钥。

准备构建服务时，再阅读 [`examples/service`](../examples/service/README.md) 和 Runtime SDK。实现 Provider、Runtime 或部署适配器时，才需要进入详细契约与 SPI 文档。

## 文档范围

- [Module](module/README.md)：统一计算、组合、绑定与执行接口。
- [Context](context/README.md)：基础模型投影、用户 AgentContext、不可变 portable 状态及 Message 追加/槽位替换规则。
- [Runtime](runtime/README.md)：Binding、并发、父子调度、取消与生命周期。
- [Durability](runtime/DURABILITY.md)：调度恢复、边界重试、持久化恢复与故障语义。
- [透明恢复与确定性重放](runtime/REPLAY.md)：Runtime 内部默认记录点、history/snapshot、重放限制与可调整策略草案。
- [Agent](agent/README.md)：Agent、ReAct 与上下文演进职责。
- [Agent 渐进式教程](agent/TUTORIAL.md)：从最小 Agent 到托管执行和动态模型配置。
- [LLM](llm/README.md)：模型调用、fallback、连接与流事件。
- [Tool](tool/README.md)：工具声明、批量执行、授权与结果。

0.2 冻结 Agent、LLM、Tool 三个能力域，并新增统一 Execution 控制面；Module 与 Context 是它们共同依赖的基础契约，Runtime 是按需接入的托管执行能力，不是普通本地调用的前置条件，也不是额外的业务能力域。

## 按需要选择接口层

普通应用从直接执行开始；只有需要框架治理的并发、部署、远程放置或持久恢复时才进入 Runtime 文档。

| 层级 | 面向对象 | 需要理解的公开概念 |
|---|---|---|
| Core | 定义、组合和本地执行 Agent | `Module`、`Message`、`Context`、`ContextCodec`、`invoke()`、`stream()` |
| Managed execution | 需要框架管理并发、取消和资源的服务 | `Runtime`、`Binding`、`BoundModule`、`ExecutionOptions` |
| Deployment | 分布式部署和持久恢复 | capacity scope、placement、durability capability、Execution Handle |
| Runtime SPI | Runtime 与 adapter 实现者 | `ExecutionScope`、`ExecutionPlan`、lease、checkpoint、replay |

使用某一层时，不应被迫构造下一层的对象。高级类型应从相应子包导入；普通应用不需要直接导入 `ExecutionScope`、`ExecutionPlan` 或计划完整性类型。

### 公开导入边界

顶层 `pygent` 只承诺日常定义和执行所需的 Application API：`Module`、`Agent`、Message/Context 值、用于声明受约束 AgentContext 的 `ContextCodec`、内置 Agent/LLM/Tool Module，以及构造这些 Module 所需的高层不可变配置。普通 Agent 通过 `context_type` 声明 Context，Runtime 在 bind 阶段自动派生 codec；`ContextCodec` 与显式 Runtime/Worker codec 配置保留为高级扩展入口。当前实现状态见 [验收矩阵](runtime/ACCEPTANCE.md)。以下类型不再作为顶层入门 API：

- `Binding`、`BoundModule`、`ExecutionOptions`、`ExecutionEvent` 与 Runtime 接口从 `pygent.runtime` 导入；
- `ExecutionPlan`、`ModuleSpec`、`CodeArtifactSpec`、schema version 与计划校验异常从 `pygent.runtime.plan` 导入；
- Provider client、adapter、invoker 等扩展协议从 `pygent.llm.spi` 导入；
- `ExecutionScope` 是框架内部或 Runtime SPI 类型，不从顶层导出，普通用户不直接构造。
- 用户自定义基础设施 Module 若需要受管 effect、Model/Tool permit、部署资源解析或稳定幂等身份，从 `pygent.core` 导入 `Infrastructure` 与 `current_infrastructure()`；这些名称不是顶层 Application API，也不要求普通业务 Module 理解 ExecutionScope。

移动这些导入是 0.2 breaking contract 的一部分。自动补全中出现的顶层名称应代表普通用户可以直接理解和构造的对象，不能以 re-export 把 Deployment API 与 Runtime SPI 重新摊平。

旧的顶层基础设施导入不提供别名、警告桥接或懒加载，必须直接从上述规范子包导入；例如 `from pygent import Runtime` 会失败，应改为 `from pygent.runtime import Runtime`。

## Core：直接执行

```python
class Module(Generic[InputMessageT, OutputMessageT]):
    async def forward(
        self,
        message: InputMessageT,
        context: Context,
    ) -> tuple[OutputMessageT, Context]: ...

agent = MyAgent()
message, context = await agent.invoke(message, context)

async with agent.stream(message, context) as stream:
    async for event in stream:
        ...
    message, context = await stream.final_result()
```

直接执行不要求调用方创建 Runtime、Binding 或 ExecutionOptions。`invoke()`/`stream()` 在当前进程中建立框架内部的 direct execution scope，使 `forward()` 内的 `await self.child(message, context)` 与事件发送保持统一；该 scope 不是用户可配置的 Runtime，也不提供框架级容量治理、远程 placement、跨进程恢复或多 Execution 调度。调用方使用 `asyncio`、服务限流器或外部设施自行管理 Root 并发、deadline 与进程生命周期。

Root 只通过 `invoke()` 或 `stream()` 启动。`await module(message, context)` 保留为活动 execution scope 内的 Child 调用；在 Root 外直接使用 `__call__()` 必须给出明确错误并引导调用 `invoke()`，从而避免同一个语法同时表示 Root 和 Child。

直接执行必须支持普通用户 Module 和声明为 direct-capable 的本地 Agent、LLM、Tool 依赖。需要托管资源解析、远程目标、共享容量或持久能力的 Module 在未绑定执行时必须明确拒绝，不得静默伪装为已治理执行。

## Managed execution：按需绑定 Runtime

```python
binding = runtime.create_binding(
    name="interactive-service",
    execution_capacity=run_capacity,
    model_capacity=model_capacity,
    tool_capacity=tool_capacity,
)
bound = binding.bind(module)
message, context = await bound.invoke(message, context)

async with bound.stream(message, context) as stream:
    async for event in stream:
        ...
    message, context = await stream.final_result()
```

`runtime.create_binding(...).bind(module)` 是托管执行的推荐形式，它先显式创建部署与资源治理域，再把一个或多个 Root Module 接入该域。现有 `module.bind(runtime, binding=binding)` 形式继续保留；两种形式必须产生相同的 BoundModule 执行契约。

```python
# 保留形式
bound = module.bind(runtime, binding=binding)
```

`ExecutionOptions` 在普通托管调用中可省略，由 Runtime 生成请求身份并使用 Binding 默认策略。只有调用方需要幂等、调用身份、deadline、持久恢复或最终提交协调时才显式传入；省略不得被解释为获得这些高级保证。

参数顺序类比 PyTorch/LSTM 的 `(x, h) -> (y, h')`：类型化当前增量 `message` 在前，不可变 Agent 状态快照 `context` 在后。基础 Context 定义模型可见投影，用户 AgentContext 可以增加 portable 历史视图和领域状态。Message 不等同于聊天文本，检索、计划、评估、审批和领域结果由相应 Module 转换为 Message 信封。直接与托管的 `invoke()`、`stream()` 执行同一个 Module 图并统一返回最终 `(message, context)`；两个独立调用不保证非确定性输出逐字相同。

普通 `stream()` 继续把一次执行和观察组合成一个便捷入口，调用方无需感知内部 Execution/订阅分离。运行身份、状态、usage、durable 重连、后台继续执行或多观察者不进入普通返回值；支持相应 capability 的 Runtime 通过独立 Execution Handle 与事件订阅控制面提供它们。关闭订阅不等于取消 Execution，显式取消只通过 Execution 控制入口完成。

## 状态与 Runtime

Module 只保存定义、配置和子 Module，不保存会话历史、当前请求或运行结果。同一个 Module 与 BoundModule 必须可被并发复用。direct-capable 的本地 adapter 可以由 Module 声明，但其连接与生命周期不得进入 Message/Context，也不得被误认为可移植 ExecutionPlan；托管绑定必须验证、替换或拒绝不可移植的本地执行依赖。

direct execution 中，调用方管理 Root 并发与外部资源生命周期，框架不声称提供跨调用容量治理。托管 Runtime 统一持有模型 client/连接、工具执行器、容量、队列、取消、deadline 与关闭过程，但 Provider 协议转换、模型 retry/fallback 和 Tool 业务授权不属于 Runtime。Binding 强制声明 Execution/Agent live 与 runnable 容量，并可选择是否增加 Model/Tool 总量门禁；领域 Module 仍可声明模型物理资源容量和单 Execution 工具 fan-out，不私建托管模式的 semaphore、Pool 或 Manager。

所有受控容量必须显式声明 `runtime_instance`、`deployment` 或 `external_resource` 作用域。分布式 Runtime 不能把每 Worker 的局部上限描述成部署总量；无法兑现 deployment 级协调时必须拒绝对应 Binding。

Binding 不与某个 Agent 一一对应。同一个服务中的 Coordinator、Worker、Reviewer 等父子 Agent 通常属于同一 Binding；它们共享部署策略和 Execution 治理域，原始子 Agent 调用默认继承当前 Binding。只有需要独立治理边界时，才使用预绑定 Child、RemoteModule 或 Binding placement，包括容量、资源、权限、安全、SLA、服务、部署策略或生命周期隔离；这不改变用户的 `forward()`。

业务服务负责加载持久状态、构造本次 Context，并在 Execution 成功后显式提交返回的 Agent 状态快照。AgentContext 可以携带完整历史视图、领域状态和来源 revision，但不负责权威提交或冲突处理，框架不暗中维护第二个 Session 状态源。Durable Runtime 可以另行保存执行元数据、Context checkpoint 和重放记录，但不得把这些记录当作业务会话或领域状态的真实源。

## 组合与调度

子 Module 作为属性声明，形成稳定的组合树。`forward()` 中对子 Module 的直接调用统一经过当前 execution scope：direct scope 只维持调用边界、事件与本地取消传播，managed scope 按 [Runtime 父子调度契约](runtime/README.md) 建立受管父子执行关系。用户代码不分支判断执行模式，也不直接操作 scope。

`bind()` 把 Module 定义接入稳定的逻辑 Runtime/Binding 执行域。原始 Child 继承 Parent Binding；预绑定 Child 保留自己的 Runtime/Binding；placement policy 可以为原始 Child 固定或自适应选择物理目标。三种方式都通过当前 ExecutionScope 建立 Child，不把它提交成新 Root，也不改变用户 `forward()` 的输入输出接口。活跃 Python 调用栈不在进程或 Worker 之间迁移；进程故障后的继续执行只能发生在 Runtime 明确支持的 retry 或 checkpoint 边界，详细语义见 [Durability](runtime/DURABILITY.md)。

Pygent 不支持在 `forward()` 中通过任意名称动态发现未声明 Agent 的开放式 Registry。Registry 或服务发现只负责把 ExecutionPlan 中已声明的稳定逻辑目标解析到当前健康实例，因此可以支持无感扩缩容和故障转移；新增逻辑 Agent、改变 schema、权限或关键 capability 时必须生成新的 Binding/ExecutionPlan，或使用独立 Root Execution/Job。
