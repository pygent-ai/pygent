# Context SDK

本文是 Context 的第二级契约，必须服从 [Context 第一原则](FEATURES.md)。框架演进应保持这些使用方式成立。本文中的用户 Context 子类是目标 SDK 契约；实现、wire codec 与 Runtime 必须收敛到该契约，不得用 pickle、类名导入或静默丢弃字段代替稳定 schema。

## 基础 Context 与模型投影

```python
context = Context(
    system_prompt="你是天气助手。",
    messages=tuple(model_history),
    tools=(weather_definition,),
    metadata={"session_id": str(session_id)},
)
```

基础字段组成模型请求投影：`system_prompt`、`messages`、`tools` 与 `metadata`。构造器立即防御性复制 tuple 字段并递归冻结 JSON 数据。`Context.tools` 只接受完整 `ToolDefinition`；工具可见性不构成执行授权。

`Context` 不再等同于整个模型请求。它是 Agent 状态快照的基础类型，模型 Layer 只消费上述基础字段。用户 Context 子类中的领域字段不会自动进入 prompt、模型 metadata 或工具参数。

## 定义用户 AgentContext

用户通过 frozen、slots dataclass 扩展 Context，并提供稳定 schema。下面的 `ToolState`、`FileState` 只是应用开发者自定义领域状态的示例名称和形状，不是 Pygent 提供或预留的公共类型：

```python
from dataclasses import dataclass, field, replace
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class ToolState:
    enabled: tuple[str, ...] = ()
    summaries: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FileState:
    workspace_id: str = "default"
    paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AgentContext(Context):
    context_schema: ClassVar[str] = "example.agent-context"
    context_schema_version: ClassVar[int] = 1

    tool_state: ToolState = field(default_factory=ToolState)
    file_state: FileState = field(default_factory=FileState)
```

Context 扩展必须满足以下条件：

- 继承 `Context`，使用 frozen 与 slots 值语义；
- `context_schema` 是稳定的应用协议标识，`context_schema_version` 是正整数；
- 所有实例字段都能规范化为有限、递归冻结的 JSON 值；
- 不覆盖基础字段的语义，不改变 `(message, context)` 的调用顺序；
- 不包含连接、锁、Store、manager、client、handler、coroutine 或其他 live object；
- wire、ExecutionPlan 和 durable checkpoint 依据注册的 schema/codec 验证，不依据 Python 类名或 pickle。

框架必须在 Root admission、远程发送或恢复开始前拒绝未知 schema、版本不兼容或不可移植字段，不得运行到深层 Module 后才静默丢失状态。`Context()` 仍是无需领域扩展时的完整 SDK 路径。

## Schema 与 codec 注册

`context_schema` 和 `context_schema_version` 只声明逻辑协议身份；它们本身不足以定义字段结构或恢复 Python 值。Pygent 的目标 SDK 提供 `ContextCodec.dataclass()`，从受约束的 frozen/slots dataclass 递归生成规范 JSON schema、编码器、解码器和稳定 digest：

```python
from pygent import ContextCodec


agent_context_codec = ContextCodec.dataclass(AgentContext)

assert agent_context_codec.schema == "example.agent-context"
assert agent_context_codec.version == 1
```

生成过程必须包含基础 Context 字段和全部用户实例字段，并拒绝未支持的 annotation、可变默认值、非 portable 字段、重复 schema/version 和递归不封闭的类型。相同 `(schema, version)` 只能对应一个规范 schema 与 codec digest；同名同版本但字段结构或 codec 不同必须 fail closed，不能以后注册者覆盖。

普通 direct execution 不经过序列化，可以从实际 AgentContext 类使用同一 dataclass validator 校验输入和返回值，不要求进程全局注册。任何可能经过 managed history、Worker、checkpoint 或恢复边界的部署都必须显式注册 codec：

```python
runtime = LocalRuntime(
    context_codecs=(agent_context_codec,),
)
```

注册表属于 Runtime/Worker 部署，不是全局可变 registry，也不进入 Context。`ContextCodec` 中的 Python constructor 只用于当前已验证代码制品内的本地重建；wire 上只出现 schema、version、规范 codec 名、codec digest 与严格 JSON data。基础 `Context` 使用 Pygent 内置 codec，无需应用注册。

`message-context-input@0.2` 与 `message-context-output@0.2` 是稳定的通用信封 schema，不等同于某个具体 AgentContext schema。其 Context 部分固定携带 discriminator：

```json
{
  "context": {
    "schema": "example.agent-context",
    "version": 1,
    "codec": "pygent-dataclass-json-v1",
    "codec_digest": "sha256:...",
    "data": {}
  }
}
```

ExecutionPlan 和 Worker deployment manifest 必须列出该部署允许的 Context codec identities/digests；每次 Root admission 把实际选中的精确 codec identity 固定到 admission manifest。Worker 在解码 data 前同时验证通用信封、部署允许列表和精确 digest。Child 若保持同一具体 Context 类型则继承该 identity；显式改变 Context schema 的 Module 边界必须在计划中声明输入/输出 codec 转换，不能靠返回任意子类动态改变 wire 契约。

## 不可变状态演进

```python
next_context = replace(
    context,
    tool_state=replace(
        context.tool_state,
        summaries=context.tool_state.summaries + ("天气工具可用",),
    ),
)
```

任何演进都返回同一具体 Context 类型的新值。旧值不变。框架提供的消息追加也必须保留用户字段：

```python
next_context = context + UserMessage(content="北京天气怎么样？")

assert type(next_context) is type(context)
assert next_context.tool_state is context.tool_state
```

普通 Message 的 `slot` 为 `None` 时按顺序追加；带稳定 `slot` 的 Message 替换基础模型投影中旧的同槽位值。当前 Message 不会因为进入 Module 就自动写入 `messages` 或用户状态。

### 自定义 `+` 与 `+=`

Python 在类型没有定义 `__iadd__()` 时，会让 `context += message` 调用 `context.__add__(message)`，再把返回的新值重新绑定给变量。因此基础 Context 的 `+=` 从来不是原地修改；对于普通 AgentContext，继承的消息追加已经通过 `dataclasses.replace(self, ...)` 保留具体子类和用户字段，通常不需要重载。

应用可以在 AgentContext 中重载 `__add__()`，让一次消息追加同时演进自定义 portable 状态；`+=` 会自动采用该行为：

```python
@dataclass(frozen=True, slots=True)
class AgentContext(Context):
    context_schema: ClassVar[str] = "example.agent-context"
    context_schema_version: ClassVar[int] = 1

    full_history: tuple[Message, ...] = ()

    def __add__(self, value: object):
        updated = super().__add__(value)
        if updated is NotImplemented:
            return NotImplemented
        assert isinstance(value, Message)
        return replace(
            updated,
            full_history=updated.full_history + (value,),
        )


previous = context
context += new_message

assert context is not previous
assert previous.full_history != context.full_history
```

用户也可以显式定义 `__iadd__()`，但它仍必须返回同一具体 Context 类型的新值，不得修改 `self`。无论重载哪个运算符，都必须保留基础消息的 slot 替换规则、拒绝非 Message 输入、递归 portable/frozen 约束以及旧值不变；不得把 `+=` 变成隐藏 I/O、Store 提交或 live resource 修改入口。Runtime 和 codec 只持久化运算后的 Context 值，不执行或传输用户运算符代码。

## 应用自定义状态 Module 示例

下面的 `StateModule` 只是应用开发者如何读取自定义 AgentContext 的示例，不是 Pygent 内置 Module、基类或约定名称。开发者选择把该计算建模为 Pygent Module 时，必须保持统一的二元输入输出协议：

```python
class StateModule(Module[UserMessage, Message]):
    async def forward(
        self,
        message: UserMessage,
        context: AgentContext,
    ) -> tuple[Message, AgentContext]:
        prompt = compute_tool_prompt(context.tool_state)
        return Message(kind="tool.prompt.computed", content=prompt), context
```

调用形式仍是：

```python
tool_info, context = await self.state_module(message, context)
```

`forward(context) -> value` 可以是普通 Python 纯函数，但不是 Pygent Module。保持统一协议才能让 Child lineage、事件、取消、placement 和 direct/managed 行为继续一致。

## 在 Agent 中使用

```python
class MyAgent(Module[UserMessage, AIMessage]):
    def __init__(self, state_module: StateModule, react: ReActLayer):
        super().__init__()
        self.state_module = state_module
        self.react = react

    async def forward(
        self,
        message: UserMessage,
        context: AgentContext,
    ) -> tuple[AIMessage, AgentContext]:
        tool_info, context = await self.state_module(message, context)
        next_message = process(message, tool_info)
        return await self.react(next_message, context)
```

内置 Model、Tool 与 ReAct Layer 必须接受满足契约的 Context 子类，并在未修改状态时原样返回该实例；需要追加消息时返回保留全部用户字段的同类型新值。模型 Provider 只收到基础字段构成的模型投影。

## 外部持久化

```python
snapshot = await store.read(session_id)
context = AgentContext(
    system_prompt=project_prompt(snapshot),
    messages=project_model_history(snapshot),
    metadata={"session_id": session_id},
    tool_state=snapshot.tool_state,
    file_state=snapshot.file_state,
)

message, next_context = await agent.invoke(message, context)

await store.commit(
    session_id,
    expected_revision=snapshot.revision,
    tool_state=next_context.tool_state,
    file_state=next_context.file_state,
)
```

Context 可以携带完整历史视图和领域快照，但不提供隐式 `save()`、`load()` 或全局 Session。业务 Store 仍然负责权威状态、revision、审计与冲突处理。Runtime checkpoint 只负责恢复对应 Execution，不能成为另一份可独立写入的业务状态源。

## 并行 Child

串行 Child 可以依次传递返回的 Context。并行 Child 各自看到同一个不可变输入快照；Runtime 不猜测用户字段的合并语义。Parent 必须显式选择或合并 Child 返回的状态：

```python
(left_message, left_context), (right_message, right_context) = await self.gather(
    self.left(message, context),
    self.right(message, context),
)

next_context = merge_agent_contexts(
    base=context,
    left=left_context,
    right=right_context,
)
```

默认隐式 last-writer-wins、字段级自动合并或共享可变 Context 都不属于 SDK。
