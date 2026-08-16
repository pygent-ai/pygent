# LLM

阅读顺序：

1. [第一原则](FEATURES.md)
2. [SDK 使用](SDK.md)
3. 本文的详细契约
4. 按需阅读 [Provider 路由选项规范](PROVIDER_OPTIONS_SPEC.md) 与 [延迟及动态模型组规范](DYNAMIC_MODEL_GROUP_SPEC.md)；动态模型组实现设计见 [独立实现文档](DYNAMIC_MODEL_GROUP_IMPLEMENTATION.md)

LLM 域向用户提供 ModelCallLayer、ModelInvoker 与 ModelProviderAdapter。Layer 声明调用哪个模型组以及如何调用；Invoker 负责 route、retry、fallback 和标准化结果；Provider adapter 负责请求/响应转换与错误归一。Runtime 只拥有 Provider client/连接池的生命周期、容量计数、deadline、取消与执行调度，不拥有 Provider 协议逻辑。

源码边界与这些职责一致：`llm.spi` 是 Provider 扩展契约，`llm.invoker` 拥有 retry/fallback、取消和关闭，`llm.openai_compatible` 拥有 HTTP/SSE transport 与 wire codec。旧的 `llm.adapter` 聚合模块已删除，不提供兼容导入。

## 输入与输出

ModelCallLayer 不引入 ModelInput 或 ModelOutput：

```python
async def forward(
    self,
    message: Message,
    context: Context,
) -> tuple[AIMessage, Context]: ...
```

ModelProviderAdapter 从系统提示词、已提交历史、当前 Message、可见工具定义和 Layer 生成配置物化 Provider 请求。

当前 Message 单独传入，不会被隐式追加。ModelCallLayer 返回完整 AIMessage 和原 Context；若输入是用户 AgentContext，返回值必须是同一实例及具体类型，不能只重建基础字段。Provider adapter 只读取 `system_prompt`、`messages`、`tools` 与 `metadata` 模型投影。ReActLayer 或用户 Agent 决定追加和状态演进时机。这避免模型成功但业务状态尚未提交时出现半完成状态。

usage、实际 route、attempt 和耗时通过类型化 ExecutionEvent 暴露；确需保存的安全事实可以进入 AIMessage metadata。Provider 原始响应、secret、连接对象与内部异常不得进入 Message、Context 或公开事件。

## 配置边界

- ModelGroupConfig 描述逻辑模型组、解析状态和组级最大并发声明；固定组同时包含候选 route 与 fallback 顺序，延迟组只包含托管部署需求。
- ModelRoute 的仅关键字 `provider_options` 描述 Provider 私有但稳定的严格 JSON 路由语义；值会防御性复制并递归冻结，非空值参与定义、部署和 effect 身份，空值保持原 canonical payload。
- RetryPolicy 描述同一 route 内的重试条件、次数与退避，不决定 fallback 顺序。
- GenerationConfig 描述与一个 ModelCallLayer 定义绑定的生成行为。

这些值必须在绑定前完成确定性校验：

- 模型组名称非空；route ID 唯一，fallback 只能引用本组 route，且不能重复。
- 固定组 routes 非空；显式延迟组 routes 和 fallback 都为空；`max_concurrency` 为空或大于零。
- `max_attempts_per_route` 至少为一；退避满足 `0 <= initial <= maximum`。
- temperature 为空或为有限的非负数；`max_output_tokens` 为空或大于零。

校验只验证定义自身，不探测网络、credential 或 Provider 能力。Provider 支持的 temperature 上限等部署相关验证属于 Runtime 的 bind/prepare 阶段。

`/v1/models` 一类模型目录是可选的部署查询能力，不改变上述确定性配置边界。应用可以在配置界面或显式启动预检中通过 Provider client 查询当前 credential 可见的模型，再用选定 ID 构造新的 ModelRoute；ModelCallLayer、bind/compile 和 Runtime 不会自动查询目录、修改现有 ModelGroupConfig 或把目录结果写入 ExecutionPlan。

## Adapter 与调用边界

```text
ModelCallLayer
  -> ModelInvoker
     -> ModelProviderAdapter
        -> ModelProviderClient
```

- ModelInvoker 选择 route，并在统一 attempt/deadline 预算内执行 retry 与 fallback。
- ModelProviderRouteValidator 是 adapter 的公开预检 SPI；第三方 adapter 若未实现它，只能继续处理空 `provider_options`，非空选项在 Provider I/O 前 fail closed。
- ModelProviderAdapter 构造 Provider JSON 请求、解析完整或流式响应，并把 Provider 错误归一为 ModelErrorKind。
- ModelProviderClient 只负责实际网络传输；Runtime 可以创建、缓存和关闭 client，但不解释 Provider payload。
- LLM 域产生 route、attempt、usage 和错误事实，Runtime 将它们包装为当前 Execution 的有序事件。

ModelCatalog 是与 ModelProviderClient 分离的可选协议。OpenAICompatibleClient 同时提供 `client.models.list()` 便利入口，但自定义推理 client 不需要实现模型目录；目录请求不创建 ModelExecution，不消耗 Model permit，也不发布 `model.*` 事件。

Provider adapter、client 和 invoker 都不得写入 Context，也不得把 secret、连接或 Provider 原始对象放入公开值。

内置 OpenAI-compatible adapter 会再次验证 route，并把选项浅合并到请求顶层。框架保留字段以及 secret、认证、连接、endpoint、retry、deadline、stream 等类别始终拒绝；DeepSeek `thinking.type` 只接受 `enabled` 或 `disabled`。未知 OpenAI-compatible Provider 默认允许其余严格 JSON 结构透传，但这不是 Provider 能力证明。

## 调度、重试与 fallback

ModelInvoker 在当前执行模式提供的总 deadline、取消信号和资源边界内执行一次模型调用。direct execution 使用调用方 deadline 与本地 adapter，managed execution 使用 Runtime scope：

1. 按 FallbackPolicy 选择 route。
2. direct execution 从本地 adapter 获得 client；managed execution 请求 Runtime 获取物理资源对应的容量和 client。
3. 通过对应 ModelProviderAdapter 调用 Provider，并把错误归一化为 ModelErrorKind。
4. RetryPolicy 允许时在同一 route 重试；预算耗尽后考虑下一 route。
5. 全部失败时抛出脱敏的 ModelCallError，不伪造成功 AIMessage。

ModelGroupConfig 的 `max_concurrency` 是模型物理资源约束声明，不是 Layer 内的 semaphore，也不是 Binding 的 Execution/Agent 上限。direct execution 不提供跨 Root 的框架共享计数器；managed Runtime 使用显式 `capacity_key`（省略时为模型组名）解析共享容量所有者，相同物理资源必须共享计数器。

Binding 可以选择不增加本地模型门禁，由模型服务承担流量检测；也可以对当前 Binding 或共享 `capacity_key` 增加模型总并发。受控模式必须同时满足 Binding 模型门禁和物理资源容量，透传模式也仍受 Execution live 上限、deadline、取消及 attempt 预算约束。模型等待期间调用方释放 Execution lease，模型调用完成并释放资源 permit 后再进入 RESUME。分布式实现不能改变 ModelCallLayer 的公开 `forward()` 输入输出契约，但远程模型调用不会让调用方 coroutine 自动获得持久化能力。

如果 Worker 在 Provider 已接收请求、但 Runtime 尚未可靠记录结果时失败，durable retry 可能重复发起模型请求。Runtime 必须为逻辑调用、执行 attempt 和 Provider 请求分别保留可关联身份，并定义流式事件与 usage 的去重规则；Provider 不支持幂等请求时不得宣称 exactly-once 模型调用。完整故障与重放语义见 [Runtime 持久化边界](../runtime/DURABILITY.md)。

SDK adapter 的隐式重试必须关闭，或纳入同一 attempt 与 deadline 预算。

## 流式与非流式

模型层只有一个 `forward()`。direct scope 或 managed Runtime 都可以使用 Provider 流式传输，并把文本、工具调用、usage 和 attempt 增量转成有序事件；最终仍返回完整 AIMessage 与原具体 Context 值。

公开事件使用固定集合：

- 生命周期：`model.started`，以及恰好一个 `model.completed`、`model.failed` 或 `model.cancelled`。
- Provider attempt：`model.attempt.started`、`model.attempt.succeeded`、`model.attempt.failed`。
- 内容：`model.reasoning.delta` 和 `model.text.delta`。reasoning 仅表示 Provider 明确允许公开的 reasoning/thinking 内容，不承诺暴露隐藏思维链。
- ToolCall 生成：`model.tool_call.started`、`model.tool_call.delta`、`model.tool_call.completed`。这组事件不表示工具已经授权或执行；实际执行使用 `tool.*` 事件。
- Token：`model.usage`。字段固定为 `input_tokens`、`output_tokens`、`total_tokens`、`cached_input_tokens` 和 `reasoning_tokens`，缺失值为 `null`；每个 route/attempt 使用累计快照。

一个 Provider SSE 数据块可以拆成多个标准增量，例如同时产生 reasoning、正文、多个 ToolCall 和 usage。`model.completed` 不等同于 Provider 的 `[DONE]`；只有完整 ToolCall 参数、结构化输出、最终 usage 与 AIMessage 全部校验后才能发布。

- `invoke()` 消费或转交事件，只返回最终 `(AIMessage, Context)`。
- `stream()` 暴露同一路径的事件，通过 `final_result()` 返回相同元组。
- 提前退出当前一次性 `stream()` 便捷入口时，direct scope 或 managed Runtime 必须取消该入口创建的 Root 并等待资源清理。durable Runtime 的可重连事件订阅与 Execution 生命周期分离；关闭订阅不等于取消 Execution，具体契约见 [Runtime](../runtime/README.md#run-与事件订阅生命周期)。

ModelCallLayer 可以独立绑定，也可以作为 ReActLayer 或用户 Agent 的子 Module；组合方式不会产生第二套模型接口。
