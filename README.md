# Pygent

A Python framework for building LLM-powered agents with modular state management, composable operators, and LLM tool-calling integration (including MCP).

## Features

- **Modular architecture** — Agents, context, tools, memory, and plans as composable modules
- **State management** — Consistent save/load and serialization via `PygentOperator`
- **LLM-native** — Messages and tools align with OpenAI-style APIs
- **MCP support** — Use Model Context Protocol (SSE and stdio) tools via `ToolManager`
- **Toolkits** — Built-in toolkits such as `RestrictedTerminal` for safe shell access

## Requirements

- Python 3.11+

## Installation

From the project root:

```bash
pip install -e .
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv pip install -e .
```

## Quick Start

```python
from pygent.agent import BaseAgent
from pygent.context import BaseContext
from pygent.llm import AsyncOpenAIClient
from pygent.message import UserMessage
from pygent.module.tool import ToolManager
from pygent.toolkits import RestrictedTerminal

class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.llm = AsyncOpenAIClient(
            base_url="https://api.openai.com/v1",
            api_key="YOUR_API_KEY",
            model_name="gpt-4",
        )
        self.tool_manager = ToolManager()
        terminal = RestrictedTerminal(root_dir=".")
        self.tool_manager.add_module("terminal", terminal)
        self.tool_manager.register_tools(terminal.get_tools())

    async def run(self, user_input: str):
        context = BaseContext(system_prompt="You are a helpful assistant.")
        context.add_message(UserMessage(content=user_input))
        # ... tool loop and LLM calls (see examples/react_agent)
        return context
```

See [examples/react_agent/react_agent.py](examples/react_agent/react_agent.py) for a full ReAct-style agent with tool calling.

## Project Layout

- **pygent/** — Core package: agents, context, LLM client, messages, modules (tool, plan, memory, etc.), toolkits
- **examples/** — Example agents (e.g. ReAct agent)
- **MCPs/** — Example MCP servers (e.g. LocalMemoryMCP)
- **docs/** — [Architecture](docs/ARCHITECTURE.md) and [API](docs/API.md) documentation
- **tests/** — Pytest tests

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — Design, layers, and components
- [API](docs/API.md) — API reference

## License

Apache-2.0. See [LICENSE](LICENSE) for details.
