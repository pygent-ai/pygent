# Tool SDK

本文是 Tool 的第二级契约，必须服从 [Tool 第一原则](FEATURES.md)。框架演进应保持这些使用方式成立。

## Python 函数工具

普通同步函数、异步函数和显式传入的实例绑定方法可以用 `@tool` 声明，再由 `ToolKit` 显式组装。decorator 保持原 callable 与类型签名不变，不在 import 时全局注册工具。

```python
from typing import Annotated, Literal

from pydantic import BaseModel, Field
from pygent import (
    Context,
    IdempotencyPolicy,
    ToolKit,
    ToolSideEffect,
    tool,
)


class WeatherResult(BaseModel):
    city: str
    temperature: float
    unit: Literal["c", "f"]


@tool(
    tool_id="weather.lookup",
    version="1.0.0",
    side_effect=ToolSideEffect.READ,
    timeout=10,
    resource_key="weather-api",
    required_permissions=("weather:read",),
)
async def lookup_weather(
    city: Annotated[str, Field(description="需要查询的城市")],
    unit: Literal["c", "f"] = "c",
) -> WeatherResult:
    """查询指定城市的天气。

    Args:
        city: 城市名。
        unit: 温度单位。
    """
    return WeatherResult(city=city, temperature=20, unit=unit)


toolkit = ToolKit(lookup_weather)
```

`tool_id`、`version` 和 `side_effect` 必须显式声明。PURE/READ 工具未指定 `idempotency` 时默认使用 `INHERENT`；WRITE/EXTERNAL 工具必须显式选择 `INHERENT`、`REQUIRES_KEY` 或 `NOT_IDEMPOTENT`，避免便利 API 把外部写操作误标为可安全重放。

Pydantic 根据类型注解生成 Draft 2020-12 input/output schema，并负责把已验证 JSON 参数还原为 Python 参数及把返回模型序列化为严格 JSON。参数不允许缺少类型注解，也不允许 positional-only、`*args`、`**kwargs` 或生成器。工具描述来自显式 `description` 或 docstring 首段；参数描述按显式 Pydantic `Annotated`/`Field` 优先、Google/NumPy/Sphinx docstring 次之的顺序生成。

### Direct/local 组装

```python
tool_layer = toolkit.local_layer(
    authorization=WeatherAuthorization(),
    max_concurrency=16,
)

model_layer = ModelCallLayer(
    model_group=model_group,
    retry_policy=retry_policy,
    generation=generation,
    tools=toolkit.definitions,
)

context = toolkit.make_visible_in(Context(messages=history))
agent = ReActLayer(model=model_layer, tools=tool_layer)
```

`local_layer()` 只缩短本地 ExecutorRegistry 与 ToolCallLayer 的组装，不改变授权规则；未传授权 Module 或受信 adapter 时仍然 fail closed。`make_visible_in()` 返回新的不可变 Context；相同定义重复加入是幂等操作，同名不同定义会拒绝，而且可见性仍不代表授权。

### 绑定方法与 managed 部署

```python
class WeatherService:
    def __init__(self, client):
        self.client = client

    @tool(
        tool_id="weather.current",
        version="1",
        side_effect=ToolSideEffect.READ,
    )
    async def current(self, city: str) -> WeatherResult:
        """查询当前天气。"""
        return await self.client.current(city)


service = WeatherService(client)
toolkit = ToolKit(service.current)
tool_layer = ToolCallLayer(
    tools=toolkit.specs,
    authorization=WeatherAuthorization(),
)

runtime.attach_executor_registry(toolkit.build_registry())
bound_tools = tool_layer.bind(runtime, binding=binding)
```

绑定实例、client 和 handler 只存在于 Worker/Runtime 启动侧；managed 图与 ExecutionPlan 只携带 `toolkit.specs`。已有 Registry 可以用 `toolkit.register_into(registry)` 安装；重复身份默认拒绝，只有部署代码显式传入 `replace_existing=True` 才替换 executor。detach、远程 executor、MCP 与自定义 ToolTaskManager 继续使用下文的低层显式组装接口。

## ToolDefinition 与 ToolSpec

```python
weather = ToolSpec(
    tool_id="weather.lookup",
    version="1.0.0",
    definition=ToolDefinition(
        name="weather.lookup",
        description="查询城市天气",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    ),
    side_effect=ToolSideEffect.READ,
    idempotency=IdempotencyPolicy.INHERENT,
    timeout=10.0,
    resource_key="weather-api",
    required_permissions=("weather:read",),
)
```

ToolDefinition 只描述模型可见接口；ToolSpec 描述可移植执行语义。handler、credential 和连接不进入两者。direct execution 由显式本地 adapter 解析 executor，调用方负责连接生命周期与跨 Root 并发；managed execution 由执行注册表和 Runtime 资源层解析。单次调用 fan-out 由 ToolCallLayer 声明，托管的跨 Execution 与物理资源容量由 Binding/resource capacity 声明。

`IdempotencyPolicy.REQUIRES_KEY` 要求 executor 收到稳定的 `ToolCall.idempotency_key`。direct execution 必须由应用显式提供，否则在 executor admission 前拒绝；managed execution 由 Runtime 使用稳定 Execution、Module 路径与 `call_id` 派生，恢复时保持不变。模型生成的参数不能覆盖该字段。

Python DSL 中的 dict、list、tuple 和 Enum 在构造时必须被防御性复制、规范化为严格 JSON 并递归冻结；构造后修改原 schema dict 不得改变 ToolDefinition 或 ToolSpec。NaN、Infinity、bytes、handler 和其他任意 Python 对象必须在构造边界被拒绝。

ToolDefinition、ToolSpec、ToolCall、ToolTask 与 ToolResult 是封闭的 portable value 类型，不支持 Python 子类追加字段。领域扩展放入已有的严格 JSON schema、arguments、metadata 或 output 槽位；codec 对未知 subtype fail-closed，不能静默丢弃 handler 或自定义字段。

## 自定义授权 Module

授权 Module 接收包含 ToolCall、已解析 ToolSpec 与请求事实的不可变 `ToolAuthorizationRequest`，返回 `ToolAuthorizationDecision`。决定至少包含原 `call_id`、`allowed`、稳定 `reason_code` 与由应用选择的 `lifecycle`（`"sync"` 或 `"detach"`）。模型提供的 ToolCall 参数不得设置或覆盖 lifecycle。

这两个框架公开 Message 类型具有显式 wire discriminator 和严格字段 codec；其 ToolCall、ToolSpec、permissions、decision 字段及基础 Message metadata 在 Context、持久化和 HTTP Worker 边界无损往返。未知字段或不可移植值必须 fail-closed，不能因远程 placement 丢失授权事实。

```python
class WeatherAuthorization(
    Module[ToolAuthorizationRequest, ToolAuthorizationDecision]
):
    def __init__(self, *, lifecycle="sync"):
        super().__init__()
        self.lifecycle = lifecycle

    async def forward(self, request, context):
        allowed = "weather:read" in request.permissions
        decision = ToolAuthorizationDecision(
            call_id=request.call.call_id,
            allowed=allowed,
            reason_code="allowed" if allowed else "missing_permission",
            lifecycle=self.lifecycle,
        )
        return decision, context
```

ToolCallLayer 必须显式接收自定义授权 Module 或受信执行适配器。两者都未配置时默认拒绝，不得默认放行，也不得把业务授权委托给 Runtime。

## ToolTask 与 ToolResult

ToolCall 是 AIMessage 中的调用请求，不代表已经获得授权或已被执行。授权允许后，调用才被接纳执行。被拒绝的调用不创建 ToolTask，但必须生成 `ToolResult(status="rejected", call_id=<原 call_id>)`，以便 ToolMessage 保留原批次的身份与顺序。direct execution 只在当前 Root 同步执行本地调用，由调用方与 adapter 管理资源、并发和清理；managed Runtime 才把已接纳调用建模为受管 ToolTask。两种模式都不把业务授权决策交给 Runtime。

同一 `AIMessage` 内的 `ToolCall.call_id` 必须唯一。重复 ID 的所有调用在 visibility、authorization 和 ToolTask admission 之前明确返回 `error_code="duplicate_call_id"` 的 validation rejection；框架不能依赖 Provider 通常生成唯一 ID，也不能让两个调用共享同一 ToolTask。

ToolTask 的公开快照至少包含 `task_id`、`call_id`、`tool_id`、`version` 和 `state`。ToolResult 至少包含 `call_id`、`status`、可选 `task`、可选的严格 JSON `output`、`error_kind`、`retryable` 与 `side_effect_committed`。`timeout` 或 `unknown` 状态不得默认把 `side_effect_committed` 设为 false。direct execution 只直接支持同步本地调用；独立生命周期任务由调用方自己的任务设施承载，不伪装成 Runtime ToolTask。

以下同步 Child 与 detach 语义只属于 managed execution。同步 Agent-backed Tool 可以作为当前 Execution 的结构化 Child；detach 时 Runtime 创建独立 ToolTask，ToolCallLayer 立即返回 `ToolResult(status="detached", task=<ToolTask 公开快照>)`。该调用自此不再是 Child，但仍受声明的 Binding、资源与 capability 治理。需要故障后重新获得时，由独立 Job 承载该 ToolTask，调用方必须要求并获得相应 durable task capability。

```python
detached_tools = ToolCallLayer(
    tools=(weather,),
    authorization=WeatherAuthorization(lifecycle="detach"),
    executor_registry=executor_registry,
    max_concurrency=16,
)
runtime.attach_executor_registry(executor_registry)
runtime.attach_tool_task_manager(tool_task_manager)
bound_detached_tools = detached_tools.bind(runtime, binding=binding)
tool_message, _ = await bound_detached_tools.invoke(ai_message, context)
detached_result = tool_message.results[0]
task = detached_result.task

snapshot = await runtime.get_tool_task(task.task_id)
await runtime.cancel_tool_task(task.task_id)
final_result = await runtime.get_tool_result(task.task_id)
```

`ToolTask` 是不可变 JSON 快照，不携带 Runtime 对象或活 handler。查询、取消和取结果都使用稳定 `task_id`。

当 Binding 的 durable capability 实际生效时，detach admission 必须按稳定 logical key 原子 get-or-create 一个独立 Job；该 key 绑定 run/Root/Module path、该 Module 在 Execution 内的确定性 occurrence、call 与 idempotency identity，确保 Parent recovery 不会重复创建 Job，同时保证跨轮或重复 Module 调用即使复用 `call_id` 也不会折叠成同一 Job。occurrence 必须由可重放调用顺序派生，不能使用随机数或仅使用参数 hash。返回的 `ToolTask.job_id` 指向该 Job。`JobRef(job_id, task_id)`、`JobSnapshot` 与 `JobState` 都是严格、不可变的公开值：Job 保存 logical key、Binding、ExecutionPlan、resource 与 required-capability 身份以及可移植 ToolSpec/ToolCall 请求，但不保存 callback、handler、registry、连接或 Runtime 对象。

```python
task = detached_result.task
job = await runtime.get_job(task.job_id)
assert job.ref == JobRef(task.job_id, task.task_id)

# 进程重启后，先重建同一声明并 bind，再显式恢复：
recovered = await runtime.recover_tool_jobs(bound_detached_tools)
```

`recover_tool_jobs()` 在启动 executor 前重新验证 Binding identity、ExecutionPlan identity、ToolSpec version、resource identity 与 required capabilities，并重新进入该 Binding 的共享 Tool capacity/resource gates。任一身份或 capability 不匹配都拒绝恢复，不得回落为直接调用 registry。`RUNNING` 的非幂等副作用在崩溃后变为 `unknown`；固有幂等，或携带稳定 idempotency key 的调用，才允许创建新的 Job attempt。

`ToolSpec.sandbox_profile="restricted"` 对应必需 capability `tool.sandbox.restricted`；其他 profile 使用同样的 `tool.sandbox.<profile>` 稳定命名。Runtime 未声明该 capability 时，durable detach admission 与恢复都必须 fail closed。

部署执行注册表按 `(tool_id, version)` 解析 executor。普通 executor 与 Agent-backed executor 可以有不同内部适配器，但必须共享上述 ToolCall、ToolTask、ToolResult、sync/detach 与授权契约；ToolSpec 不携带 executor 或 Agent 对象。

Agent-backed executor 使用普通 ModuleDependency 和应用转换函数，不创建第二套 Agent API：

```python
executor = AgentToolExecutor(
    agent=bound_or_raw_agent,
    request_builder=lambda spec, call: (to_message(call), to_context(call)),
    result_builder=lambda message, context: to_json_result(message),
)
executor_registry.register(tool_id, version, executor)
```

managed sync 调用通过当前 ExecutionScope 成为结构化 Child，继承 lineage、deadline、取消和 join；detach admission 会先可靠建立独立 ToolTask，再在 Parent 可以返回 detached 引用后启动 Agent Root，因此 `max_runnable_runs=1` 也不能形成 Parent RESUME 与新 Root 的互等死锁。detach 创建后台 Task 时不得继承原 Parent execution scope。

`ToolCallLayer` 与 `AgentToolExecutor` 只通过公开 `pygent.core` Infrastructure SPI 取得 Tool permit、executor registry resolver、幂等 key、managed effect 和当前执行状态，不导入私有 `_execution_scope`。用户自定义工具基础设施 Module 使用同一 SPI；Runtime 仍只负责治理与解析，不实现授权、schema、executor 或 Agent 业务逻辑。

## 构造工具层

```python
tools = ToolCallLayer(
    tools=(weather,),
    authorization=WeatherAuthorization(),
    executor_registry=executor_registry,
    max_concurrency=16,
)
```

ToolCallLayer 使用仅关键字构造参数。Layer 的 `max_concurrency` 只限制单次调用的工具 fan-out。direct execution 不增加跨 Root 门禁；managed execution 中，该值不替代 Binding 跨 Execution 总并发或 resource capacity。

managed 部署也可以让 Layer 省略 `executor_registry`，并在服务启动时调用 `runtime.attach_executor_registry(executor_registry)`；Runtime 按 `(tool_id, version)` 解析 executor 并治理其 permit。业务授权仍必须由 Layer 的应用授权 Module/适配器完成。

## 在组合层中调用

```python
class ToolStep(Module[AIMessage, ToolMessage]):
    def __init__(self, tools: ToolCallLayer):
        super().__init__()
        self.tools = tools

    async def forward(self, message: AIMessage, context: Context):
        tool_message, unchanged_context = await self.tools(message, context)
        return tool_message, unchanged_context
```

ToolCallLayer 从 AIMessage 的 `tool_calls` 读取调用，并返回聚合 ToolMessage。即使并发调用的完成顺序不同，或结果混合 success、rejected、failed 与 detached，`tool_message.results` 也必须与原 `message.tool_calls` 顺序一致：

```python
assert [r.call_id for r in tool_message.results] == [
    call.call_id for call in ai_message.tool_calls
]
```

## 作为 Root 直接调用

```python
tool_message, context = await tools.invoke(ai_message, context)
```

直接调用要求 ToolCallLayer 具有可用的本地执行 adapter，只支持当前 Root 内同步等待；调用方自行管理跨 Root 并发和独立后台任务。

## 作为 Root 使用已有 Binding

```python
bound_tools = tools.bind(runtime, binding=binding)
tool_message, context = await bound_tools.invoke(ai_message, context)
```

这里是把 ToolCallLayer 作为 Root 接入已有 Binding。作为子 Module 时默认继承 Parent Binding；只有需要独立治理边界时才使用不同 Binding。

长任务的进度通过同一 Execution 的事件通道观察：

```python
async with bound_tools.stream(ai_message, context) as stream:
    async for event in stream:
        ...
    tool_message, context = await stream.final_result()
```
