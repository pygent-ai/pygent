# Basic PygentAgent Example

最小可用的 `PygentAgent` 示例：你只需要提供**模型**、**系统提示词**和**上下文压缩提示词**，
其余（ReAct 循环、模型层、压缩器、工具占位层）由 `build_agent()` 固定组装。

## 前置条件

一个 OpenAI-compatible 服务端点（OpenAI、GLM、Qwen 等）和密钥。设置环境变量：

```powershell
$env:PYGENT_API_BASE = "https://your-provider.example/v1"
$env:PYGENT_API_KEY = "your-secret"
$env:PYGENT_MODEL_NAME = "your-model"
```

Linux/macOS：

```bash
export PYGENT_API_BASE="https://your-provider.example/v1"
export PYGENT_API_KEY="your-secret"
export PYGENT_MODEL_NAME="your-model"
```

## 单轮问答

```bash
uv run python -m examples.basic_agent "用一句话介绍你自己"
```

## 交互式会话

```bash
uv run python -m examples.basic_agent
```

每轮回复后会打印 `last_input_tokens` 和 `compressions`（本次调用触发的压缩次数）。
空行或 `quit` 退出。

## 提供提示词

提示词可用环境变量或命令行参数提供，命令行优先于环境变量；两者都不提供时使用仓库内置默认值。

```powershell
$env:PYGENT_SYSTEM_PROMPT = "You are a coding assistant. Inspect evidence before changing code."
$env:PYGENT_COMPRESSION_PROMPT = "Create a compact continuation. Preserve goals, constraints, and unfinished work."
```

或：

```bash
uv run python -m examples.basic_agent "帮我审查这段代码" `
  --system-prompt "You are a coding assistant." `
  --compression-prompt "Create a compact continuation. Preserve goals and unfinished work."
```

## 代码结构

- `agent.py` — 默认提示词、`build_agent()`（只需模型名和两个提示词）
- `providers.py` — 环境变量配置 `LiveModelConfig` 和 `build_live_invoker()`
- `runner.py` — `run_once()` 单轮、`run_interactive()` 交互循环
- `__main__.py` — 命令行入口

## 加工具

`PygentAgent` 需要工具 Module；当前示例使用空 `ToolCallLayer(tools=())`。
要加工具，用 `@tool` 声明函数、`ToolKit` 收集定义，再把
`ToolCallLayer(tools=kit.specs, authorization=...)` 传给 `build_agent()`，
并把 `ModelCallLayer` 的 `tools` 换成 `kit.definitions`。完整写法见
[教程示例](../tutorial/agent.py)。
