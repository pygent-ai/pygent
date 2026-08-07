# LLM SDK

本文是 LLM 的第二级契约，必须服从 [LLM 第一原则](FEATURES.md)。框架演进应保持这些使用方式成立。

## 定义模型层

```python
from time import monotonic

from pygent import (
    AIMessage,
    Context,
    ExponentialBackoff,
    FallbackPolicy,
    GenerationConfig,
    ModelCallOptions,
    ModelCallLayer,
    ModelCallPolicy,
    ModelErrorKind,
    ModelGroupConfig,
    ModelInvoker,
    ModelRoute,
    Module,
    RetryPolicy,
    ExecutionOptions,
    UserMessage,
)


def build_model(invoker: ModelInvoker | None = None) -> ModelCallLayer:
    return ModelCallLayer(
        model_group=ModelGroupConfig(
            name="assistant",
            routes=(
                ModelRoute("primary", provider="openai", model="gpt-5"),
                ModelRoute("fallback", provider="qwen", model="qwen-plus"),
            ),
            fallback=FallbackPolicy(order=("primary", "fallback")),
            max_concurrency=32,
            capacity_key="account-a/chat-model",
        ),
        retry_policy=RetryPolicy(
            max_attempts_per_route=2,
            retry_on=(
                ModelErrorKind.TIMEOUT,
                ModelErrorKind.RATE_LIMIT,
                ModelErrorKind.UNAVAILABLE,
            ),
            backoff=ExponentialBackoff(initial=0.2, maximum=2.0),
            attempt_timeout_seconds=10.0,
        ),
        generation=GenerationConfig(
            temperature=0.2,
            max_output_tokens=2048,
            tool_choice="auto",
        ),
        invoker=invoker,
    )


model = build_model(invoker)
run = ExecutionOptions(deadline=monotonic() + 30.0)
```

这些对象都是不可变、可移植的声明值。只有 `ModelCallLayer` 是 Module 定义；`ModelGroupConfig`、`RetryPolicy`、`FallbackPolicy` 和 `GenerationConfig` 是该 Module 引用的配置值，不具有独立 Module 身份或调用身份。ModelCallLayer 使用仅关键字构造参数，避免把模型组、重试与生成策略按位置传错。需要两种稳定语义时声明两个 ModelCallLayer 子 Module；延迟托管模型组可以按下文的显式 policy 开放少量请求级生成参数。

`ModelGroupConfig.max_concurrency` 是模型物理资源约束声明，不是 Layer 私有 semaphore。`capacity_key` 是多个逻辑模型组共享同一物理 endpoint/credential/model 配额时使用的稳定身份；省略时退回模型组名称。direct execution 不负责跨 Root 协调该声明，调用方或本地 adapter 自行限流；managed execution 中，相同 `capacity_key` 的 Layer 共享同一 Runtime 容量所有者，不得将各 Layer 的数值累加成更高物理并发。

`GenerationConfig.tool_choice` 为 `None`、`"auto"`、`"required"`、`"none"` 或当前可见工具名；指定名称但该工具不在 Layer 声明与 `Context.tools` 的交集中时，adapter 在发请求前拒绝。OpenAI-compatible adapter 会把 portable 工具名稳定映射为 Provider 允许的 wire name，并在 ToolCall 返回时还原，应用不必把 `weather.lookup` 之类的业务名称改成 Provider 私有格式。

## 查询 OpenAI-compatible 模型目录

模型选择界面或显式启动预检可以复用同一个 Provider client 查询当前 endpoint 与 credential 可见的模型：

```python
from pygent.llm import OpenAICompatibleClient

client = OpenAICompatibleClient(
    base_url="http://localhost:8000/v1",
    api_key="token",
)

try:
    available = await client.models.list()
    available_ids = {model.id for model in available}
finally:
    await client.aclose()
```

`list()` 默认使用独立的十秒有限超时，可以通过 `timeout=<positive seconds>` 调整，或显式传 `None` 交给调用方取消边界。返回顺序与 Provider 一致；每个 ModelInfo 只保留稳定的 `id` 以及可选 `created`、`owned_by`。认证、限流、超时、不可用和非法响应通过脱敏的 ModelProviderError/ModelErrorKind 报告。

目录查询不是 ModelCallLayer 调用：它不执行 route、retry/fallback，不获取 Runtime Model permit，也不产生 `model.*` 事件。`ModelProviderClient` 协议保持不变，自定义推理 client 不需要实现目录。

应用显式选择一个可见 ID 后构造新的不可变声明：

```python
selected_model = "qwen3-32b"
if selected_model not in available_ids:
    raise ValueError("selected model is unavailable")

route = ModelRoute(
    route_id="primary",
    provider="openai",
    model=selected_model,
)
```

固定模型的生产部署可以继续直接声明 ModelRoute，完全不查询目录。不要在 `forward()` 中覆盖模型、修改已经冻结的 Layer，或在 bind/compile 时自动请求 `/v1/models`。固定路径的每请求模型选择从服务端允许的预构造 Layer/BoundModule 集合中完成；延迟托管路径则从 Binding 句柄已经验证发布的 profile 中选择。模型目录表示查询当时的可见性，不是长期可用性或能力证明。

0.2 不在 `forward()` 中接受 provider 私有参数字典、stream 开关、client、credential 或 route 强制覆盖。direct execution 使用 ModelCallLayer 声明的本地 adapter 配置，调用方管理其连接生命周期与外部 deadline；managed execution 的本次请求信息进入可选 ExecutionOptions，secret、endpoint 与 client 等部署资源由 Runtime 根据 Binding 中的稳定资源引用解析。`ModelRoute` 选择、retry 与 fallback 始终由 ModelInvoker 决定，Runtime 不解释 Provider 路由逻辑。

所有 route、retry、fallback、容量等待和 attempt 必须消耗同一有限 effective deadline 与取消预算；adapter 不得建立隐藏的第二套重试或 deadline 预算。`ModelCallLayer` 声明 `requires_finite_deadline=True`，因此 managed Root 或任意包含它的 Module 图在没有有限 `ExecutionOptions.deadline` 时必须于 admission 阶段 fail closed，不能等到 Provider I/O 后才失败。direct execution 不启用该 Runtime 门禁，外部 deadline 仍由调用方或 adapter 负责。

attempt timeout 取消 Provider task 后，ModelInvoker 最多使用内部 1 秒 cleanup grace，并进一步受剩余 effective deadline 限制。只有 task 已确认退出，`TIMEOUT` 才能按 `RetryPolicy` 进入 retry/fallback；清理未确认时公开错误为 `ModelErrorKind.OUTCOME_UNKNOWN`，`model.attempt.failed` 固定携带脱敏的 `reason="cancellation_cleanup_timeout"`，本次模型调用立即终止。Invoker 按 client 对象身份隔离仍未退出的 task；隔离期间同一 client 的新逻辑 attempt fail-fast，不发送 Provider 请求，后台 task 退出并被安全回收后自动解除隔离。调用方显式取消仍传播 `CancelledError`，不会转换为模型失败。

`ModelCallLayer` 通过公开 `pygent.core.current_infrastructure()` 获取 effective deadline、Model permit、部署 `ModelInvoker` resolver 和 managed effect replay；它不导入私有 execution ContextVar。用户自定义模型基础设施 Module 可以使用同一 SPI。Runtime 只解析部署注入的 invoker 并治理执行，不拥有 route、retry、fallback、HTTP client 或 Provider 解析逻辑。

## 作为子 Module 调用

```python
class AnswerLayer(Module[UserMessage, AIMessage]):
    def __init__(self, model: ModelCallLayer):
        super().__init__()
        self.model = model

    async def forward(self, message: UserMessage, context: Context):
        answer, unchanged_context = await self.model(message, context)
        return answer, unchanged_context
```

ModelCallLayer 不会把当前 Message 或输出 AIMessage 自动加入历史。

## 作为 Root 直接调用

```python
ai_message, context = await model.invoke(message, context)

async with model.stream(message, context) as stream:
    async for event in stream:
        if event.kind == "model.reasoning.delta":
            render_reasoning(event.data["text"])
        elif event.kind == "model.text.delta":
            render_answer(event.data["text"])
        elif event.kind == "model.tool_call.delta":
            render_tool_arguments(event.data["item_id"], event.data["arguments_delta"])
        elif event.kind == "model.usage" and event.data["final"]:
            record_attempt_usage(event.data)
    ai_message, context = await stream.final_result()
```

事件消费者应按 `event.kind` 处理固定 payload，不解释 Provider 私有字段。`model.usage` 是 route/attempt 级累计快照，同一 attempt 取最后一条；`model.tool_call.completed` 才包含已解析的完整 `arguments` 对象。执行工具时观察独立的 `tool.*` 事件。

直接调用要求 ModelCallLayer 具有可用的本地 adapter 配置；调用方负责 Root 并发、连接生命周期和外部 deadline。只声明了托管资源引用、没有本地 adapter 的模型层必须明确拒绝 direct execution。

## 作为 Root 使用已有 Binding

```python
bound_model = model.bind(runtime, binding=binding)

ai_message, context = await bound_model.invoke(message, context, run=run)

async with bound_model.stream(message, context, run=run) as stream:
    async for event in stream:
        ...
    ai_message, context = await stream.final_result()
```

请求级 attempt timeout 仍只等待有限 cleanup grace；未确认退出时返回 `OUTCOME_UNKNOWN` 并隔离对应 client。每个原生 Provider stream 由唯一 owner task 从创建、读取到关闭全程持有，其他任务不直接调用其 `__anext__()` 或 `aclose()`。资源级 `DefaultModelInvoker.aclose()` 使用严格关闭语义：它会等待所有 active execution、stream owner 与 quarantine cleanup 终止，再关闭共享 client 并返回，因此 `aclose()` 返回后不会遗留后台异步生成器。

这里是把 ModelCallLayer 作为 Root 接入已有 Binding。作为子 Module 时它默认继承 Parent Binding；只有需要独立治理边界时才为模型层使用不同 Binding。

部署希望由 Runtime 统一拥有 client 生命周期时，Layer 可以省略 `invoker`，并在启动服务时按模型组注册：`runtime.register_model_invoker("assistant", invoker)`。`LocalRuntime.close()` 会关闭注册 invoker 暴露的 `aclose()`；未注册的模型组在托管调用时明确失败。

直接与托管入口使用相同 Module 图和最终 `(message, context)` 契约；只有托管入口获得 Binding 调度规则。两个独立调用不保证非确定性模型的文本逐字相同。

## 延迟配置的托管模型组

需要先定义 Agent、再由应用或用户配置具体模型时，声明一个延迟模型组。它仍是不可变、可移植的 Layer 配置，只表达部署需求；Agent 不保存可变句柄、route、credential 或本地 invoker：

```python
assistant_group = ModelGroupConfig.deferred(
    name="assistant",
    max_concurrency=8,
    capacity_key="assistant-model",
)

model = ModelCallLayer(
    model_group=assistant_group,
    policy=ModelCallPolicy(
        allow_profile_override=True,
        overridable_generation=frozenset({"temperature", "max_output_tokens"}),
    ),
    retry_policy=RetryPolicy(),
    generation=GenerationConfig(max_output_tokens=2048),
)
```

`policy` 只声明 Root 调用是否可以选择 profile、可以覆盖哪些生成参数；它不区分框架 Session 和单次 invocation，也不包含当前选择。应用需要会话粘性时，自行保存 profile 并在每次 Root 调用中重复传入。调用方不能覆盖 route、credential、client、retry 或 fallback；未声明的覆盖在 admission 阶段拒绝。延迟模型组只支持托管执行；给它传入 `invoker=` 或直接调用 `model.invoke()`/`model.stream()` 都必须在 Provider 工作开始前明确失败。需要 direct execution 时继续使用本文开头的固定模型组示例。

应用从 `RuntimeBinding` 取得该声明对应的 `ModelGroupHandle`，在执行树外配置若干命名 profile。句柄是 control-plane 对象，不进入 Agent、Context 或 ExecutionPlan：

```python
binding = runtime.create_binding(...)
group = binding.model_groups.get(assistant_group)

await group.configure(
    profile="balanced",
    routes=(
        ModelRoute("primary", provider="openai", model=selected_model),
    ),
    fallback=FallbackPolicy(("primary",)),
    invoker=invoker,
    resource_ref=resource_resolver.ref(
        "tenant-42/openai-primary",
        revision=credential_revision,
    ),
)

await group.configure(
    profile="quality",
    routes=(
        ModelRoute("primary", provider="openai", model="gpt-5"),
        ModelRoute("fallback", provider="openai", model=selected_model),
    ),
    fallback=FallbackPolicy(("primary", "fallback")),
    invoker=quality_invoker,
    resource_ref=resource_resolver.ref(
        "tenant-42/openai-quality",
        revision=quality_credential_revision,
    ),
)

await group.set_default("balanced")
```

`configure()` 会先做 Provider 配置验证，再让 Runtime 检查声明匹配、资源引用、容量归属与当前执行安全性；任一检查失败都不会产生可选快照。重复配置同名 profile 会创建新的不可变快照，只影响之后选择该 profile 的 admission。开发者不传递或维护版本号。资源引用必须指向不可变 revision，不能只是会被静默改写的 credential 别名。

正常调用使用默认 profile；临时选择和生成参数放在本次不可变 `ExecutionOptions` 中，不修改 Agent 或全局默认：

```python
bound_agent = binding.bind(agent)

message, context = await bound_agent.invoke(message, context, execution=run)

message, context = await bound_agent.invoke(
    message,
    context,
    execution=ExecutionOptions(
        deadline=deadline,
        model_calls={
            "assistant": ModelCallOptions(
                profile="quality",
                temperature=0.1,
                max_output_tokens=4096,
            ),
        },
    ),
)
```

如果应用希望一个会话持续使用同一 profile，应在应用自己的会话记录中保存 profile 名称，并在该会话的每次 Root 调用中传入相同的 `ModelCallOptions`。这不要求为每个会话创建 ModelGroup，也不要求修改 Agent 实例。多个 Agent 实例可以共享同一个延迟声明和 Binding 下的 profile 集合；只有确实需要独立权限、容量或部署生命周期时才创建另一个 Binding。

调用开始时，Runtime 优先使用本次 `ExecutionOptions` 中的选择（它可以来自应用保存的会话状态），否则使用 group 默认值，并立即 pin 其不可变快照。一次调用内的 retry/fallback 始终在这个快照内执行，不会因失败跳到另一个 profile，也不会重新读取默认值。`RetryPolicy` 仍由 `ModelCallLayer` 声明；`ModelCallOptions` 不能覆盖它。

更新默认值也只通过句柄完成：

```python
await group.set_default("quality")
```

这只影响之后未显式选择 profile 的 admission。已经开始的 Root，以及继承同一 Binding admission 的结构化 Child，继续使用已 pin 的快照。句柄可额外提供 `current()`（默认 profile）、`current(profile)`、`list_profiles()` 和 `available_models()` 供控制台展示与预检，但目录结果只表示查询当时可见，不会自动创建 profile 或改变默认值。

Runtime 在 managed effect replay 确认需要新的 Provider 工作后才获取模型容量和 live invoker lease。已提交结果的重放不要求 Provider client 存活。完整的 pin、替换、恢复与生命周期契约见 [延迟与动态模型组规范](DYNAMIC_MODEL_GROUP_SPEC.md)。
