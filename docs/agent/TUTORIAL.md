# 从零开发一个 Pygent Agent

这是一条面向 Python 开发者的渐进式学习路线。你不需要先理解 Runtime、ExecutionPlan 或分布式部署：先在本地跑通一个 Agent，再逐步加入模型、工具、流式输出和托管能力。

配套代码位于 [`examples/tutorial`](../../examples/tutorial/)，默认使用确定性的离线模型边界，不需要 API Key：

```bash
uv sync --all-extras
uv run python -m examples.tutorial
```

预期输出：

```text
answer: offline: 2 + 3 = 5
context messages: 4
```

## 1. 先记住三个概念

Pygent 的 Agent 开发可以先缩减成三个概念：

1. **Agent 就是 Module**：在 `forward()` 中接收 `(message, context)`，返回新的 `(message, context)`。
2. **Context 是不可变快照**：Agent 不保存会话状态；谁加载 Context，谁就在成功后提交新 Context。
3. **子 Module 直接调用**：模型、工具、审核 Agent 都是属性，在 `forward()` 中使用 `await self.child(...)` 组合。

最小 Agent 完全不需要模型服务：

```python
from pygent import (
    AIMessage,
    Context,
    Module,
    UserMessage,
)


class EchoAgent(Module[UserMessage, AIMessage]):
    async def forward(self, message, context):
        output = AIMessage(content=message.content.upper())
        return output, context + message + output


answer, context = await EchoAgent().invoke(
    UserMessage(content="hello"),
    Context(),
)
```

Root 调用使用 `invoke()` 或 `stream()`；只有 `forward()` 内的 Child 调用才写成 `await self.child(message, context)`。

## 2. 接入一个模型

模型由三个部分组成：

- `ModelGroupConfig` 声明 route 和 fallback 顺序；
- `RetryPolicy` 与 `GenerationConfig` 声明稳定策略；
- `ModelCallLayer` 把这些声明变成可组合的 Module。

配套示例在 [`agent.py`](../../examples/tutorial/agent.py) 中组装模型层：

```python
model = ModelCallLayer(
    model_group=fixed_model_group(model_name),
    retry_policy=RetryPolicy(attempt_timeout_seconds=30.0),
    generation=GenerationConfig(
        temperature=0.0,
        max_output_tokens=256,
        tool_choice="auto",
    ),
    tools=TOOLKIT.definitions,
    invoker=invoker,
)
```

模型配置属于 Agent 定义，endpoint、credential、HTTP client 等活资源属于应用或部署。不要把密钥、client、会话历史或本次调用结果保存在 Agent 属性里。

默认命令使用 [`OfflineModelInvoker`](../../examples/tutorial/providers.py)，但执行的仍是完整的 ModelCallLayer → ReAct → ToolCallLayer 路径，因此适合本地开发和 CI。

## 3. 用 Python 函数声明工具

使用 `@tool` 保留普通 Python 函数，同时生成模型可见 schema 和可移植 ToolSpec：

```python
@tool(
    tool_id="tutorial.add",
    version="1.0.0",
    name="add",
    side_effect=ToolSideEffect.PURE,
    required_permissions=("calculator:use",),
)
def add_numbers(a: int, b: int) -> dict[str, int]:
    """Add two integers."""
    return {"sum": a + b}


toolkit = ToolKit(add_numbers)
```

工具可见性和执行授权是两件事：

- `toolkit.make_visible_in(context)` 只允许模型看到工具定义；
- `ToolCallLayer` 仍要求应用提供授权 Module 或受信 adapter；
- 权限等请求事实放在 `Context.metadata`，模型参数不能提升权限。

教程中的 `TutorialAuthorization` 检查 `calculator:use`，然后 `toolkit.local_layer(...)` 组装本地 executor。WRITE/EXTERNAL 工具还应显式声明幂等策略，不能照搬 PURE 工具的默认值。

## 4. 用 ReAct 组成真正的 Agent

`ReActLayer` 负责有界的“模型 → 工具 → 模型”循环以及历史顺序：

```python
react = ReActLayer(
    model=model,
    tools=toolkit.local_layer(
        authorization=TutorialAuthorization(),
        max_concurrency=4,
    ),
    max_steps=3,
    max_model_calls=3,
    max_tool_calls=2,
)
```

再用普通 Module 包装它：

```python
class TutorialAgent(Module[UserMessage, AIMessage]):
    def __init__(self, react: ReActLayer):
        super().__init__()
        self.react = react

    async def forward(self, message, context):
        return await self.react(message, context)
```

一次离线运行得到的 Context 顺序是：

```text
UserMessage → AIMessage(tool call) → ToolMessage → AIMessage(final)
```

不要在外层再次追加 ReAct 已提交的最终消息，否则历史会重复。

## 5. 切换到流式输出

Agent 图不需要修改，只替换 Root 入口：

```python
async with agent.stream(message, context) as stream:
    async for event in stream:
        if event.kind == "model.text.delta":
            render(event.data["text"])
    answer, next_context = await stream.final_result()
```

运行完整示例：

```bash
uv run python -m examples.tutorial stream
```

事件用于观察，不是另一份业务状态。只有 `final_result()` 返回的 Context 才能在 Execution 成功后提交。使用 `async with`，并在离开 stream 后再关闭模型 client 或 Runtime。

## 6. 接入真实 OpenAI-compatible 模型

真实模型是显式可选路径。配置 API 根路径，不要填写 `/chat/completions`：

```powershell
$env:PYGENT_API_BASE = "https://your-provider.example/v1"
$env:PYGENT_API_KEY = "your-secret"
$env:PYGENT_MODEL_NAME = "your-model"
uv run python -m examples.tutorial live
```

Linux/macOS：

```bash
export PYGENT_API_BASE="https://your-provider.example/v1"
export PYGENT_API_KEY="your-secret"
export PYGENT_MODEL_NAME="your-model"
uv run python -m examples.tutorial live
```

[`build_live_invoker()`](../../examples/tutorial/providers.py) 使用 `OpenAICompatibleClient`、`OpenAICompatibleAdapter` 和 `DefaultModelInvoker`。调用结束后 runner 显式执行 `aclose()`；配置对象的 `repr` 不包含密钥。

## 7. 什么时候使用 Runtime

本地脚本和简单服务先用 direct execution。需要统一管理并发、队列、deadline、取消、远程 placement 或持久能力时，再把同一 Agent 图接入 Binding：

```python
runtime = LocalRuntime()
binding = runtime.create_binding(
    name="tutorial-service",
    execution_capacity=execution_capacity,
    model_capacity=model_capacity,
    tool_capacity=tool_capacity,
)
bound_agent = binding.bind(agent)

try:
    answer, context = await bound_agent.invoke(
        message,
        context,
        execution=ExecutionOptions(deadline=monotonic() + 30.0),
    )
finally:
    await runtime.close()
```

包含 ModelCallLayer 或 ReActLayer 的 managed Root 必须提供有限 deadline。Binding 是服务的部署与治理域，不是 Agent 身份，也不需要一个 Agent 创建一个 Binding。

## 8. 动态配置模型 profile

当 Agent 定义要先于具体模型部署创建时，声明 deferred ModelGroup：

```python
requirement = ModelGroupConfig.deferred(
    name="tutorial-assistant",
    max_concurrency=8,
    capacity_key="tutorial-model",
)
agent = build_agent(requirement)
bound = binding.bind(agent)
group = bound.model_groups.get(requirement)
```

在执行树外发布并选择 profile：

```python
await group.configure(
    profile="quick",
    routes=(ModelRoute("primary", "openai", "fast-model"),),
    fallback=FallbackPolicy(("primary",)),
    invoker=quick_invoker,
)
await group.configure(
    profile="quality",
    routes=(ModelRoute("primary", "openai", "quality-model"),),
    fallback=FallbackPolicy(("primary",)),
    invoker=quality_invoker,
)
await group.set_default("quick")
```

单次调用可以在 Agent 允许的范围内覆盖 profile 和生成参数：

```python
execution = ExecutionOptions(
    deadline=monotonic() + 30.0,
    model_calls={
        "tutorial-assistant": ModelCallOptions(
            profile="quality",
            temperature=0.1,
        )
    },
)
```

运行不依赖外部模型的动态配置示例：

```bash
uv run python -m examples.tutorial managed
uv run python -m examples.tutorial managed --profile quick
```

profile 在 admission 时被固定；执行中的 retry/fallback 不会重新读取默认值。profile 句柄、client 和 credential 都不进入 Agent 或 Context。

## 9. 从示例走向服务

上线前至少确认：

- Agent 只保存配置和子 Module，可以被并发复用；
- 服务负责加载与提交会话 Context，并处理并发写冲突；
- ReAct 步数、模型次数、工具次数、Provider attempt 和 Execution deadline 都有界；
- 工具可见性、业务授权、副作用与幂等策略分别声明；
- stream 正常完成、异常和取消后都先 join，再关闭共享资源；
- secret、原始 Provider 错误和 client 不进入 Message、Context 或日志；
- managed 服务在 shutdown 时停止接收新请求，再关闭 Runtime。

完整的会话 CAS、审核 Agent、invoke/stream 服务边界见 [`examples/service`](../../examples/service/README.md)。

## 常见错误

| 错误 | 正确做法 |
|---|---|
| 在 Root 外写 `await agent(message, context)` | Root 使用 `invoke()`/`stream()`；`await child(...)` 只用于 `forward()` 内 |
| 把历史、client 或当前 profile 存到 Agent | 历史放 Context/业务 Store；client 与 profile 句柄由应用或 Runtime 持有 |
| 模型能看到工具就默认允许执行 | 始终提供应用授权 Module，并把权限作为可信请求事实注入 |
| managed 模型调用不传 deadline | 使用有限的 `ExecutionOptions.deadline` |
| stream 未结束就关闭 client | 使用 `async with`，等待 `final_result()` 或取消清理完成后再关闭 |
| ReAct 返回后再次追加最终 AIMessage | 尊重子 Agent 的 Context 提交契约，避免重复历史 |

## 下一步

- [Agent SDK](SDK.md)：自定义组合、direct/managed 与 handoff。
- [LLM SDK](../llm/SDK.md)：route、retry、fallback、事件与动态模型组。
- [Tool SDK](../tool/SDK.md)：Python 工具、授权、detach 与 Agent-backed Tool。
- [Runtime SDK](../runtime/SDK.md)：Binding、父子执行、并发、恢复与 Worker。
- [Execution contract](../EXECUTION.md)：统一执行和事件语义。
