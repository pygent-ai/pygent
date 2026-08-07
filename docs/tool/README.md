# Tool

阅读顺序：

1. [第一原则](FEATURES.md)
2. [SDK 使用](SDK.md)
3. 本文的详细契约

Tool 域的公开数据模型固定为三层：模型可见的 ToolDefinition、可移植执行声明 ToolSpec，以及表示执行实例与结果的 ToolTask/ToolResult。ToolCallLayer 是编排这三层的 Module，不是第四层工具值。用户开发的自定义授权 Module 或受信执行适配器负责授权决策；Runtime 只负责资源获取、调度、取消与关闭。

## 输入与输出

```python
async def forward(
    self,
    message: AIMessage,
    context: Context,
) -> tuple[ToolMessage, Context]: ...
```

ToolCallLayer 根据自身 ToolSpec 和 `context.tools` 的可见名称交集选择候选工具，再要求用户开发的自定义授权 Module 或受信执行适配器完成实际授权。`context.tools` 本身不是授权证明。获准的 ToolCall 被接纳为 ToolTask；被拒绝的 ToolCall 不创建 ToolTask，但仍产生具有原 `call_id` 的 rejected ToolResult。同一 AIMessage 中的所有结果按原 ToolCall 顺序聚合为一个 ToolMessage，不受完成顺序影响。

每个 ToolResult 显式区分成功、拒绝、失败、取消和不确定终态，并携带标准错误码、可重试性和可选的副作用提交状态。完整执行结果与再次发送给模型的安全投影分离。ToolCallLayer 原样返回 Context。

## ToolDefinition、ToolSpec 与 ToolTask/ToolResult

ToolDefinition 只包含名称、描述、输入 schema 和可选输出 schema，用于向模型描述逻辑工具接口。ToolSpec 包含稳定 `tool_id`、版本、副作用类别、幂等策略、timeout、resource key、sandbox profile 和所需权限。ToolSpec 不保存部署环境的并发数字；单 Execution fan-out 由 ToolCallLayer 声明，跨 Execution 和物理资源容量由 Binding/resource capacity 治理。

ToolSpec 不保存 handler、client、credential、secret、连接、线程池或进程池。执行注册表按 `(tool_id, version)` 解析实际 executor；部署可以替换 executor 而不改变模型可见 ToolDefinition。

`@tool` 与 `ToolKit` 是 Python 函数工具的构建及部署辅助器，不增加第四层公开工具值。decorator 从函数签名和 docstring 生成现有 ToolDefinition/ToolSpec，ToolKit 显式收集这些声明并可在部署侧建立本地 ExecutorRegistry。ToolKit、Python handler、Pydantic adapter 与应用 client 都是非可移植对象，不得进入 Message、Context、ToolSpec、Module 定义状态或 ExecutionPlan；进入模型与执行图的仍只有 `ToolKit.definitions` 和 `ToolKit.specs`。

ToolTask 是通过 schema 校验与可信授权后被接纳的单次执行实例，其公开快照至少包含 `task_id`、`call_id`、`tool_id`、`version` 和 `state`，但不暴露 handler、credential 或活连接。ToolResult 表示调用结果；它可以是已接纳 ToolTask 的终态值，也可以是 admission 之前的 rejected 结果。

## 执行边界

一次调用经过 schema 校验、可信授权、admission、调度、执行、结果构造和清理。timeout、资源、沙箱、副作用与幂等要求来自 ToolSpec；单 Execution fan-out 来自 ToolCallLayer；跨 Execution 总并发和物理资源容量来自 Binding/resource capacity。Layer 不创建私有线程池、进程池、连接池、semaphore 或 ToolManager。

ToolCallLayer 的并发值只限制本次 Layer 调用中的工具 fan-out。direct execution 不增加跨 Root 门禁，调用方自行管理总并发。managed Binding 可以选择不增加共享 Tool 门禁，也可以限制该 Binding 下所有 Execution 的工具总并发；具体工具资源还可以按稳定 resource key 共享更窄容量。托管调用必须满足所有已启用层次，等待 Tool 容量或执行完成时调用方释放 Execution lease，Tool permit 在调用方进入 RESUME 前释放。

Manager、grant、permit、lease 和 Pool 是 Runtime 内部协议。文件、Shell、浏览器、MCP 或远程工具只是不同 adapter，不能建立第二套公开接口。

## 同步与 detach 生命周期

普通工具与 Agent-backed Tool 使用同一生命周期契约。direct execution 只直接支持同步本地调用，独立后台生命周期由调用方自己的任务设施承载。managed 同步调用中，ToolTask 属于当前 Execution；Agent-backed Tool 可以作为结构化 Child 执行，并随 Parent 取消、join 和收敛。

detach 是显式的生命周期转换，只能由应用或自定义授权 Module 选择，不能由模型在 ToolCall 参数中自行提升。Runtime 必须把调用接纳为拥有独立身份、admission、取消与保留策略的 ToolTask，并立即返回带公开 ToolTask 快照的 detached ToolResult；需要 durable recovery 时，由独立 Job 承载该 ToolTask。Agent-backed Tool 在这一边界后不再是结构化 Child，不继续伪装成 `detached Child`；新任务仍受声明的 Binding、资源与 capability 治理。

进程内 ToolTask 只保证当前 Runtime 生命周期内可重新获得。需要跨进程故障恢复或持久保留时，Runtime 必须提供具有持久化 admission 和状态机的 Job/durable task capability；无法满足调用方必需 capability 时必须拒绝 detach，不得用后台 coroutine 静默降级。

## 流式观察

工具只有一个 `forward()`。长任务可以向当前 Root 发布进度事件；`invoke()` 排空事件，`stream()` 暴露事件。两者共享同一次工具执行，并统一返回最终 `(ToolMessage, Context)`。
