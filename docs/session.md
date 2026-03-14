# Session 管理设计文档

## 1. 概述

本文档从架构师视角定义 pygent 框架的 **Session（会话）管理** 方案，面向需要长时对话、状态恢复、审计追溯的关键业务场景（如医疗辅助决策）。所有设计均基于 pygent 既有的持久化能力（`PygentOperator.save/load`），确保与现有模块无缝集成。

### 1.1 设计目标

- **会话隔离**：不同 Session 之间的对话历史、工作空间、缓存互不干扰
- **可恢复性**：支持 checkpoint、暂停/恢复，降低断线或异常造成的数据丢失风险
- **可审计**：完整保留对话历史与关键元数据，满足合规与追溯需求
- **最小侵入**：在不破坏现有 Agent/Context/Tool 架构的前提下扩展 Session 能力

---

## 2. react_agent 现状分析

### 2.1 当前实现要点

`examples/react_agent.py` 中的 `ReactAgent` 典型用法如下：

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ReactAgent                                   │
│  - llm: AsyncRequestsClient                                          │
│  - tool_manager: ToolManager                                         │
│    - FileToolkits(session_id="react", workspace_root=root_dir)       │
│    - TerminalToolkits(session_id="react", workspace_root=root_dir)   │
│    - WebSearchToolkits(session_id="react", workspace_root=root_dir)  │
│    - WebFetchToolkits(session_id="react", workspace_root=root_dir)   │
└─────────────────────────────────────────────────────────────────────┘
```

**调用流程：**

1. `agent.stream(user_input)` 或 `agent.forward(user_input)`
2. 每次调用内部创建新的 `BaseContext(system_prompt=...)`
3. `context.add_message(UserMessage(content=user_input))`
4. 调用 LLM，若有 tool_calls 则循环执行工具并继续调用 LLM
5. 返回最后一条 Assistant 消息 content

### 2.2 现有问题

| 问题 | 影响 | 严重性 |
|------|------|--------|
| **每次请求新建 Context** | 无跨轮对话记忆，无法实现多轮问诊、上下文连续推理 | 高 |
| **session_id 硬编码为 "react"** | 多用户/多会话时工作空间、缓存、临时文件混用 | 高 |
| **Context 不持久化** | 进程崩溃或重启后对话历史丢失 | 高 |
| **Agent 每次 main() 新建** | 无法复用已加载配置、工具状态 | 中 |

### 2.3 session_id 在 Toolkits 中的用途

所有 Toolkits（`FileToolkits`, `TerminalToolkits`, `WebSearchToolkits`, `WebFetchToolkits`）在 `__init__` 中接收 `session_id` 和 `workspace_root`：

- **session_id**：用于隔离同一工作空间下不同会话的：
  - 临时文件
  - 缓存（如搜索结果）
  - 日志与审计标识
- **workspace_root**：工具执行的根路径，通常按项目或用户隔离

当前 `session_id` 在 ReactAgent 中写死，无法实现“每 Session 一个 ID”的隔离策略。

---

## 3. Pygent 模块持久化能力

### 3.1 继承关系与 save/load

pygent 中所有可持久化模块均继承 `PygentOperator`，具备：

```python
state_dict() -> Dict[str, Any]   # 导出状态
load_state_dict(state_dict)      # 恢复状态
save(path, format='json'|'yaml'|'pickle')  # 持久化到文件
load(path, format='auto')        # 从文件加载
```

**关键约束**：`state_dict` 仅导出通过 **类型注解** 声明的 `PygentData` 字段。未在类型注解中声明的属性（如 `self.llm`、`self.tool_manager`）不会被持久化。

### 3.2 各模块持久化现状

| 模块 | 基类 | 可持久化字段 | 备注 |
|------|------|--------------|------|
| `BaseContext` | PygentOperator | `history: PygentList[BaseMessage]` | `system_prompt` 未在注解中，当前不会被 save |
| `BaseAgent` | PygentOperator | 无显式 PygentData 字段 | 子类需显式声明 |
| `BaseClient` / `BaseAsyncClient` | PygentOperator | base_url, api_key, model_name, timeout, temperature 等 | LLM 配置可保存 |
| `ToolManager` | PygentModule | tools, tool_categories 等子模块 | 工具定义可保存 |
| `BaseTool` | PygentModule | metadata, parameters, config, enabled 等 | 工具配置与状态可保存 |
| `FileToolkits` 等 | ToolClassBase → PygentOperator | `session_id: PygentString`, `workspace_root: PygentString` | 可通过 save 持久化 |

### 3.3 Context.history 序列化注意事项

`BaseContext.history` 是 `PygentList[BaseMessage]`，其 `data` 是原始 Python list，元素为 `BaseMessage` 子类实例。在 `json.dumps(state_dict, default=str)` 时：

- 非 JSON 原生类型会走 `default=str`，导致 `BaseMessage` 被转成字符串，结构丢失
- 如需可靠持久化，建议在 Session 层对 `history` 做显式序列化：每条消息调用 `msg.to_dict()` 或类似方法，存储为结构化 JSON

---

## 4. Session 架构设计

### 4.1 Session 的定义与职责

**Session** 表示一次完整的对话会话，是管理 Agent 对话生命周期的最小单位。

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              Session                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ session_id: str           # 唯一标识，用于工具隔离与存储路径          │ │
│  │ created_at: datetime      # 创建时间                                 │ │
│  │ updated_at: datetime      # 最后更新时间                             │ │
│  │ metadata: dict            # 业务元数据（用户ID、场景标签等）          │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ context: BaseContext      # 对话历史（system_prompt + history）      │ │
│  │   - system_prompt                                                   │ │
│  │   - history: PygentList[BaseMessage]                                │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ agent_ref: str | None     # 可选：关联的 Agent 配置/快照路径         │ │
│  │ workspace_root: str       # 工具工作空间根路径                       │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Session 与 Agent 的关系

```
                    ┌─────────────────────────────────┐
                    │           SessionManager         │
                    │  - 创建 / 加载 / 删除 Session    │
                    │  - 按 session_id 索引            │
                    └─────────────────────────────────┘
                                        │
          ┌─────────────────────────────┼─────────────────────────────┐
          │                             │                             │
          ▼                             ▼                             ▼
   ┌──────────────┐             ┌──────────────┐             ┌──────────────┐
   │  Session A   │             │  Session B   │             │  Session C   │
   │  session_id  │             │  session_id  │             │  session_id  │
   │  context     │             │  context     │             │  context     │
   └──────────────┘             └──────────────┘             └──────────────┘
          │                             │                             │
          └─────────────────────────────┼─────────────────────────────┘
                                        │
                                        ▼
                    ┌─────────────────────────────────┐
                    │            BaseAgent             │
                    │  (单例或按需创建，可被多 Session │
                    │   共享，但 context 由 Session 持  │
                    │   有，每次 forward 注入)         │
                    └─────────────────────────────────┘
```

**推荐模式**：

- **Agent**：负责 LLM 调用、工具调度，可跨 Session 复用（配置不变时）
- **Session**：持有 `BaseContext`，每次 `agent.forward(session.context, ...)` 时注入
- **session_id**：由 Session 生成并传给 Toolkits，确保工具侧隔离

### 4.3 推荐的 Session 使用流程

```
1. 创建 Session
   session = session_manager.create(session_id="user_123_001", workspace_root="/data/workspaces/user_123")

2. 注册工具时绑定 session_id
   file_toolkits = FileToolkits(session_id=session.session_id, workspace_root=session.workspace_root)
   agent.tool_manager.register_tool(...)

3. 对话循环
   session.context.add_message(UserMessage(content=user_input))
   result = await agent.forward(session.context)   # 或 agent.stream(session.context, ...)
   # context 已被 LLM 更新，history 中新增 AssistantMessage / ToolMessage

4. 保存 checkpoint
   session.save()   # 自动保存到 workspace_root/sessions/{session_id}/session.json

5. 恢复
   session = Session.load(workspace_root, session_id)   # 从对应文件夹加载
   # 继续对话
```

---

## 5. Session 持久化设计

### 5.1 持久化内容

**注意**：Context 的对话历史使用 `PygentList[BaseMessage]`，序列化时需将每条消息转为 dict 再存储。

| 字段 | 说明 | 格式建议 |
|------|------|----------|
| `session_id` | 会话唯一标识 | string |
| `created_at` | 创建时间 | ISO 8601 |
| `updated_at` | 最后更新时间 | ISO 8601 |
| `workspace_root` | 工作空间根路径 | string |
| `system_prompt` | 系统提示词 | string |
| `history` | 消息列表（来自 PygentList） | 每条为 `{role, content, tool_call_id?, tool_calls?, ...}` |
| `metadata` | 业务元数据 | object |

### 5.1.1 存储路径

Session 的 `save()` 自动将数据保存到 **session 对应的文件夹**：

```
{workspace_root}/sessions/{session_id}/session.json
```

调用 `save()` 时无需传入路径；`load()` 通过 `workspace_root` 和 `session_id` 定位文件。

### 5.2 存储格式

推荐使用 **JSON**，便于人工审计与跨语言解析。若需二进制体积优化，可选用 `pickle` 或 `yaml`。

**示例 JSON 结构：**

```json
{
  "version": "1.0",
  "session_id": "user_123_001",
  "created_at": "2025-03-14T10:00:00Z",
  "updated_at": "2025-03-14T10:30:00Z",
  "workspace_root": "/data/workspaces/user_123",
  "system_prompt": "你是一个医疗辅助助手...",
  "history": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "患者主诉头痛"},
    {"role": "assistant", "content": "建议...", "tool_calls": []}
  ],
  "metadata": {"user_id": "user_123", "scene": "triage"}
}
```

### 5.3 保存策略

| 策略 | 触发时机 | 适用场景 |
|------|----------|----------|
| **显式保存** | 用户/系统调用 `session.save()` | 关键节点、对话结束 |
| **按轮次** | 每 N 轮 user-assistant 交互后自动 save | 长对话防丢失 |
| **定时** | 每隔 T 秒后台 save | 长时间连续对话 |
| **Checkpoint** | 工具调用链完成后 | 多步推理的可回滚点 |

医疗等关键场景建议：**显式保存 + 按轮次自动保存**，并存多份（如 `session_id.json` 与 `session_id.backup.json`）以应对写入中断。

---

## 6. 与 Agent 集成的最佳实践

### 6.1 修改 ReactAgent 以支持 Session

**原逻辑**：每次 `forward/stream` 内部新建 `BaseContext`。

**推荐**：将 Context 作为参数传入，由调用方（Session）持有：

```python
async def forward(self, context: BaseContext) -> str:
    result = await self.llm.forward(context, tools=self._tools_param())
    while getattr(context.last_message, "tool_calls", None):
        # ... tool loop ...
    return context.last_message.content

async def stream(self, context: BaseContext, max_steps: int = 20) -> AsyncIterator[...]:
    for step in range(max_steps):
        async for chunk in self.llm.stream_forward(context, tools=self._tools_param()):
            yield chunk
        # ... tool handling ...
```

**创建 Session 时传入 session_id**：

```python
def __init__(self, session_id: str, root_dir: str = "."):
    self.session_id = session_id
    file_toolkits = FileToolkits(session_id=session_id, workspace_root=root_dir)
    # ...
```

这样，每个 Session 使用独立的 session_id，工具侧自然隔离。

### 6.2 Session 创建与 Agent 绑定

```python
# 方案 A：每次新建 Agent（简单，适合 session_id 固定）
agent = ReactAgent(session_id=session.session_id, root_dir=session.workspace_root)

# 方案 B：Agent 单例 + 动态更新 Toolkits 的 session_id（需 Toolkits 支持 set_session_id）
# 更复杂，一般不推荐
```

推荐 **方案 A**：Session 与 Agent 一一对应，或按 Session 创建 Agent，逻辑清晰、隔离彻底。

### 6.3 多轮对话示例

```python
session = Session(session_id="pat_001", workspace_root="/workspace/pat_001")
session.context = BaseContext(system_prompt="你是医疗辅助助手...")
agent = ReactAgent(session_id=session.session_id, root_dir=session.workspace_root)

# 第 1 轮
session.context.add_message(UserMessage(content="患者男，45岁，主诉胸痛2小时"))
await agent.forward(session.context)
session.save()   # 保存到对应文件夹

# 第 2 轮（可从文件恢复后继续）
session.context.add_message(UserMessage(content="既往有高血压病史"))
await agent.forward(session.context)
session.save()
```

---

## 7. 医疗/关键场景的特殊考虑

### 7.1 审计与合规

- **完整历史**：Session 的 `history` 必须完整保存，不得裁剪或脱敏（脱敏应在展示层做）
- **时间戳**：`created_at`、`updated_at` 及消息级 `timestamp`（若扩展）需可靠
- **不可篡改**：持久化文件建议配合只增日志（WAL）或哈希校验，便于事后审计

### 7.2 数据隔离

- 每个患者/每次问诊使用独立 `session_id`
- `workspace_root` 按患者或项目划分，避免跨会话文件访问
- 工具内的缓存、临时文件均以 session_id 为命名空间

### 7.3 恢复与容错

- 支持从任意已保存的 Session 文件恢复
- 恢复后应能继续 `add_message` + `forward`，行为与未中断时一致
- 建议实现 `session.export_audit_log()`，导出为合规要求的审计格式

---

## 8. 实现路线图（建议）

| 阶段 | 内容 | 优先级 |
|------|------|--------|
| P0 | 定义 `Session` 类（位于 `pygent/session`），包含 session_id、context、workspace_root、metadata | 高 |
| P0 | 实现 `Session.save()`（保存到对应文件夹）/ `Session.load()`，history 使用 PygentList 与 `to_dict` 序列化 | 高 |
| P0 | 修改 `ReactAgent`：Context 由外部传入，session_id 由构造函数传入 | 高 |
| P1 | 实现 `SessionManager`：create、get、delete、list | 中 |
| P1 | `BaseContext` 增加 `system_prompt: PygentString` 类型注解，纳入 state_dict | 中 |
| P2 | 自动 checkpoint（按轮次/定时） | 中 |
| P2 | 审计日志导出 `export_audit_log()` | 中 |

---

## 9. 总结

- **Session** 是管理对话生命周期、实现多轮对话与状态恢复的核心抽象。
- **Context** 由 Session 持有，Agent 的 `forward` / `stream` 接收 Context，不再内部新建。
- **session_id** 由 Session 生成并传入 Agent/Toolkits，实现工具侧隔离。
- 所有 pygent 模块均支持 `save`/`load`，Session 层可组合 Context 的显式序列化与自身元数据，实现完整持久化。
- 医疗等关键场景需特别关注：完整历史、时间戳、数据隔离与审计能力。

按本设计实施后，可满足“按 Session 管理 Agent 对话”的需求，并为后续扩展（多模态、知识库、工作流）预留清晰接口。
