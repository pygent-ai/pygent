# Agents SDK

本目录保存完整的 Agent SDK 使用文档与端到端示例。

- [ReActAgent 前台 ReAct SDK](REACT_AGENT_SDK.md)：内置前台 ReAct 个体 `ReActAgent`
  （`pygent.agent.react_agent`，顶层 `pygent` 直接导出），只组合标准 ReAct、前台模型、
  Compressor 和工具 Module；最小可运行示例见 [`examples/basic_agent`](../../examples/basic_agent/README.md)。

`pygent.agent` 还延迟导出 `PygentAgent`：它属于独立 Agent 个体 `pygent.agent.pygent_v1`，
在 `ReActAgent` 之上组装完整的会话、调度、实时对话与检查点管线，依赖该个体自己的
`RunConfig`/`SessionManager` 等配置与会话管理接口，不是上述 SDK 的替代入口。普通应用从
`ReActAgent` 开始；只有需要 pygent_v1 个体的完整运行环境时才导入 `PygentAgent`。

本目录使用复数 `agents` 保存完整 Agent 组合示例；[`docs/agent`](../agent/README.md)
继续保存 Agent/ReAct 基础模块契约。
