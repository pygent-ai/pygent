# PygentAgent 前台 ReAct SDK

> 状态：已实现的公共 SDK 契约。
>
> 本文服从 [Pygent 第一原则](../FEATURES.md)以及正式的
> [Agent](../agent/FEATURES.md)、[Context](../context/FEATURES.md)、
> [Runtime](../runtime/FEATURES.md)、[LLM](../llm/FEATURES.md)和
> [Tool](../tool/FEATURES.md)第一原则。

## 1. 定位

`PygentAgent` 是标准前台 ReAct Agent。它复用 `ReActLayer`、`ModelCallLayer`、
`ToolCallLayer` 和 Runtime Execution Inbox，不维护第二套循环。

```text
PygentAgent
└─ ReActLayer
   ├─ model: context-compression Module
   │  ├─ model: foreground Model Module
   │  └─ compressor: compression Module
   └─ tools: Tool Module
```

框架固定 System Prompt，执行有界压缩，返回本次调用正式提交的消息增量，消费三种 ReAct
Projection Operation，并在真实 Provider attempt 前发布最终请求快照。开发者负责 UserMessage、
workspace reminder、ToolResult reminder、压缩提示词正文和长期历史持久化。

业务能力、提示词正文以及 session 和审计 Store 都由开发者在 Agent 外部管理。

## 2. 公共 API

```python
from pygent import PygentAgent, PygentAgentContext
from pygent.agent import (
    ContextCompressionLimitExceeded,
    ContextCompressionUnavailable,
)
```

`PygentAgentContext` 是 portable、frozen、slots Context：

```python
@dataclass(frozen=True, slots=True)
class PygentAgentContext(Context):
    context_schema = "pygent.agent-context"
    context_schema_version = 3

    committed_messages: tuple[Message, ...] = ()
    compression_count: int = 0
    input_token_scale_ppm: int = 1_100_000
    last_input_tokens: int | None = None
```

- `messages` 是下一次模型调用看到的投影；
- `committed_messages` 只包含本次 `PygentAgent.forward()` 已正式提交的消息；
- `projection_revision` 是唯一模型投影 revision；
- `compression_count` 是本次 Agent 调用已完成的压缩次数；
- `input_token_scale_ppm` 是根据真实 usage 单调提高的估算系数；
- `last_input_tokens` 是最近一次可用的前台模型输入 token 数。

## 3. 构造 Agent

```python
from pygent import (
    GenerationConfig,
    ModelCallLayer,
    PygentAgent,
    RetryPolicy,
    ToolCallLayer,
)

SYSTEM_PROMPT = """You are a careful coding agent.
Inspect evidence before changing code and keep execution bounded.
"""

COMPRESSION_PROMPT = """Create a compact continuation for the foreground agent.
Preserve goals, constraints, decisions, unfinished work, relevant files and tool state.
Return only the continuation text.
"""

foreground_model = ModelCallLayer(
    model_group=foreground_model_group,
    retry_policy=RetryPolicy(max_attempts_per_route=2),
    generation=GenerationConfig(temperature=0.2, max_output_tokens=8_000),
    tools=all_tool_definitions,
)

compressor = ModelCallLayer(
    model_group=compression_model_group,
    retry_policy=RetryPolicy(max_attempts_per_route=2),
    generation=GenerationConfig(temperature=0.0, max_output_tokens=4_000),
    tools=(),
)

agent = PygentAgent(
    system_prompt=SYSTEM_PROMPT,
    compression_prompt=COMPRESSION_PROMPT,
    model=foreground_model,
    compressor=compressor,
    tools=ToolCallLayer(
        tools=all_tool_specs,
        authorization=tool_authorization,
    ),
    context_window_tokens=128_000,
    compression_trigger_ratio=0.9,
    compression_context_window_tokens=128_000,
    max_compressions=4,
    max_steps=16,
    max_model_calls=16,
    max_tool_calls=64,
)
```

两个 Prompt 在 Agent 初始化时固定。`new_context()` 创建初始 Context，不在每次
`forward()` 中重新替换或校验 System Prompt。

`ModelCallLayer.tools` 是部署允许的工具集合；`Context.tools` 是本次请求可见的子集；
`ToolCallLayer.authorization` 在实际执行前完成授权。

## 4. 创建 Context 与首次执行

```python
context = agent.new_context(
    tools=all_tool_definitions,
    metadata={"workspace_mode": "write"},
)

initial_message = UserMessage(
    content="检查项目并修复失败的测试。",
    kind="application.user_context",
    metadata={"audit_id": "user-input-1842"},
)

bound_agent = runtime.bind(agent)
handle = await bound_agent.start(
    initial_message,
    context,
    execution=execution_options,
)

answer, final_context = await handle.result()

await business_history.append(
    execution_id=handle.execution_id,
    messages=final_context.committed_messages,
)
```

`committed_messages` 已包含 Initial/Mid-run UserMessage、ToolCall AIMessage、ToolMessage
和最终 AIMessage，业务服务不得再次追加 `answer`。只有成功结果具有可提交的消息增量；
业务 Store 负责以 Execution 或 invocation identity 实现幂等追加、revision、审计和冲突处理。
原始用户输入审计、session 加载和最终 Context 提交也属于 Agent 外部的业务服务。

## 5. 运行中 UserMessage

Runtime 接收 opaque Execution Input；ReAct 只解释固定 kind：

```python
from pygent.agent import (
    REACT_PROJECTION_OPERATION_KIND,
    StandaloneUserMessage,
    encode_react_projection_operation,
)

operation = StandaloneUserMessage(
    UserMessage(
        content="先分析根因，再决定修改范围。",
        kind="application.user_context",
        metadata={"audit_id": "user-input-1843"},
    )
)

delivery = await handle.send_input(
    input_id="chat-message-1843",
    kind=REACT_PROJECTION_OPERATION_KIND,
    value=encode_react_projection_operation(operation),
)
```

`accepted` 表示进入当前 Execution；`duplicate` 表示相同 `input_id` 已投递；
`execution_finished` 表示调用方应把该 UserMessage 作为下一 turn 的 Initial UserMessage。

工作区变化等补充上下文使用 `InjectionKind.RUNTIME_CONTEXT.value` 标记，ReAct 会统一包装 XML；也可以由普通 [Reminder Module](../agent/SDK.md#reminder) 显式生成：

```python
from pygent.agent import InjectionKind

workspace_change = StandaloneUserMessage(
    UserMessage(
        content="pyproject.toml 和 src/app.py 已发生变化。",
        kind=InjectionKind.RUNTIME_CONTEXT.value,
        metadata={"event_kind": "workspace_changed", "revision": 37},
    )
)
```

## 6. ToolResult 后附加内容

开发者可以把最终提示词附在当前尚未提交的 ToolMessage：

```python
from pygent.agent import AppendToolResultContent

operation = AppendToolResultContent(
    content="工具执行后发现工作区配置发生变化。"
)

await handle.send_input(
    input_id="tool-reminder-17",
    kind=REACT_PROJECTION_OPERATION_KIND,
    value=encode_react_projection_operation(operation),
)
```

ReAct 将附加内容包装为 `<runtime-context>...</runtime-context>`，已规范化的包装不会重复嵌套。
原始 `ToolResult.output` 不变。附加内容保存在 `ToolMessage.content`，Provider 投影时追加到
最后一个 ToolResult 的模型可见内容。

## 7. 自动压缩

每次前台模型调用前，PygentAgent 对实际 provider-neutral 投影进行保守 token 估算：

```text
System Prompt + Context.messages + current + Context.tools
```

ASCII 内容、非 ASCII 内容和消息/工具结构分别计入估算。第一次使用 10% 安全系数；
后续成功请求使用真实 `AIMessage.usage.input_tokens` 单调提高估算系数，不因较低 usage
降低已经观察到的高水位。

当前台请求或压缩请求达到各自
`context_window_tokens × compression_trigger_ratio` 时：

1. 当前 `Context.messages` 成为 Compressor 的完整投影历史；
2. 保留固定 System Prompt，tools 固定为空；
3. `compression_prompt` 作为最后一条独立 UserMessage；
4. 非空、无 ToolCall 的结果成为 `pygent.context.snapshot` UserMessage；
5. Snapshot 替换 `Context.messages`，pending current 保持原文；
6. `compression_count` 和 `projection_revision` 各增加一次；
7. 本次已经产生的 `committed_messages`、metadata 和具体 Context 类型保持不变。

压缩调用不计入 ReAct `max_model_calls`，但服从同一 Execution deadline、Runtime model
capacity 和 managed-effect 规则。

- 达到 `max_compressions`：`ContextCompressionLimitExceeded`；
- 没有投影历史、压缩请求超过压缩窗口、Snapshot 后前台请求仍超限：
  `ContextCompressionUnavailable`；
- Compressor 修改 fork Context、返回空结果或带 ToolCall：
  `ContextCompressionUnavailable`。

### 自定义 Compressor

`compressor` 是普通 Module，不要求专用基类。开发者可以组合多个模型、检索或文件读取，
最终返回一个摘要 AIMessage：

```python
class ReviewingCompressor(Module[Message, AIMessage]):
    def __init__(self, *, summarizer, reviewer):
        super().__init__()
        self.summarizer = summarizer
        self.reviewer = reviewer

    async def forward(self, request, context):
        draft, context = await self.summarizer(request, context)
        final, context = await self.reviewer(
            UserMessage(content=f"Review and rewrite this snapshot:\n{draft.content}"),
            context,
        )
        return final, context
```

需要长期历史的自定义 Compressor 应通过开发者自己的外部服务或有界输入获得它，不能依赖
`PygentAgentContext` 跨调用累积完整历史。返回的 fork Context 必须与输入完全相等；
Snapshot kind、slot、当前调用提交增量和 revision 仍由 PygentAgent 统一处理。

## 8. 手动替换投影

```python
from pygent.agent import ReplaceMessageProjection

replacement = ReplaceMessageProjection(
    messages=(
        UserMessage(
            content=snapshot_text,
            kind="application.context_snapshot",
        ),
    ),
    expected_revision=captured_context.projection_revision,
    rebase_appended=True,
)

await handle.send_input(
    input_id="context-replacement-7",
    kind=REACT_PROJECTION_OPERATION_KIND,
    value=encode_react_projection_operation(replacement),
)
```

严格模式要求 revision 完全相等。`rebase_appended=True` 只允许把 base revision 后的完整
追加消息接到 replacement 尾部；期间发生自动压缩、其他
`ReplaceMessageProjection` 或 `AppendToolResultContent` 时，ReAct 以
`revision_conflict` 拒绝。

替换投影前尚未提交的 current 会先进入本次 `committed_messages`；最终 AIMessage 只记录
一次。replacement 中仅作为投影前缀的消息不会成为本次业务增量。完全清空会话时结束当前
Execution，再调用 `agent.new_context()`。

## 9. 最终模型请求快照

每个真实 Provider attempt 发出 I/O 前依次产生：

```text
model.attempt.started
model.request.prepared
Provider I/O
```

```python
async with handle.subscribe() as events:
    async for event in events:
        if event.kind == "model.request.prepared":
            await request_audit_store.record(
                execution_id=event.execution_id,
                span_id=event.span_id,
                request_id=event.data["request_id"],
                request_digest=event.data["request_digest"],
                request=event.data["request"],
            )
```

事件包含 `route_id`、`attempt`、唯一 `request_id`、稳定 `request_digest` 和完整
provider-neutral request。Request 包含 provider/model、System Prompt、历史消息、current、
effective tools、有效 generation settings 和 projection revision。

快照不包含 Context metadata、client、credential、endpoint、headers、Provider 原始响应
或内部异常。用户消息正文和 Tool Definition 是实际请求内容，不会被改写；审计服务必须
自行实施访问控制。

请求快照不设固定字节大小上限，保留完整请求投影和 digest。retry 使用新的 `request_id`，请求不变时 digest 相同。effect replay
不伪造没有真实发生的 attempt 或请求快照。

## 10. 不变量

1. PygentAgent 只使用标准 ReActLayer。
2. System Prompt 和 Compression Prompt 属于不可变 Agent 定义。
3. Initial 与 Mid-run 输入都是开发者构造的真实 UserMessage。
4. Runtime 只管理 Execution Input，不解释 Projection Operation。
5. ReAct 在每次模型调用前和最终返回前消费输入。
6. 自动压缩和手动 `ReplaceMessageProjection` 只替换模型投影，不删除本次已经记录的提交增量。
7. 每次有效投影变化只增加一次 `projection_revision`。
8. 压缩后的 replacement 会阻止过期 append-only rebase。
9. 每个真实 Provider attempt 都有一份完整、有界的请求快照。
10. direct execution 可以压缩和产生请求快照，但不接收运行中外部输入。
11. `committed_messages` 在每次 Agent 调用入口清空；长期完整历史只由外部业务 Store 保存。
