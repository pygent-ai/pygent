# Context SDK

本文是 Context 的第二级契约，必须服从 [Context 第一原则](FEATURES.md)。框架演进应保持这些使用方式成立。

## 构造

```python
context = Context(
    system_prompt="你是天气助手。",
    messages=tuple(history),
    tools=(weather_definition,),
    metadata=(("session_id", str(session_id)),),
)
```

`Context` 构造器会立即把 `messages` 与 `tools` 防御性复制为 tuple，不保留调用方可变容器的引用；`metadata` 中的 dict/list 会递归冻结，并拒绝非 JSON 值。稳定外部引用必须先规范化为字符串或其他 JSON 标量。

0.2 的 `Context.tools` 只接受完整 `ToolDefinition`，不接受字符串引用、注册表代理、handler 或连接。工具可见性不构成执行授权；实际授权只能由用户开发的自定义授权 Module 或受信执行适配器决定。

## 追加消息

```python
context1 = context + UserMessage(content="北京天气怎么样？")
```

原 Context 不变，`context1` 是新值。也可以使用重新绑定写法：

```python
context += user_message
```

这等价于 `context = context + user_message`，不是原地修改。

普通 Message 的 `slot` 默认为 `None`，因此按顺序追加。只需保留当前有效值的领域 Message 可以声明稳定槽位：

```python
latest_retrieval = Message(
    kind="retrieval.completed",
    slot="retrieval/current",
    content="当前检索结果",
    data={"document_ids": ["doc-1", "doc-2"]},
)
context = context + latest_retrieval
```

再次追加相同 `slot` 时，旧值会被移除，新值放在末尾。槽位不是 role 或作者名；普通多轮 UserMessage/AIMessage 不设置槽位，不会互相覆盖。

`Context` 本身是封闭的可移植值，不能通过子类增加连接、Store、handler 或其他字段。额外请求事实必须规范化为严格 JSON 后写入 `metadata`；领域消息扩展使用上述 `Message(kind=..., data=...)`。

## 在 Agent 中推进历史

```python
async def forward(self, message: UserMessage, context: Context):
    answer, unchanged_context = await self.model(message, context)
    next_context = unchanged_context + message + answer
    return answer, next_context
```

当前 Message 在调用时与历史分开；调用者在追加前可以完成内容校验、标准化、内容权限检查或裁剪。这些历史写入检查不等同于工具执行授权。

## 外部持久化

```python
snapshot = await store.read(session_id)
context = Context(messages=snapshot.messages)

message, context = await agent.invoke(message, context)

await store.commit(
    session_id,
    expected_revision=snapshot.revision,
    messages=context.messages,
)
```

Context 本身不提供 `save()`、`load()`、版本、revision、审计或隐式 Session；它只表示本次调用当前有效的数据快照。
