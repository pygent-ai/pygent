# Pygent

A Python framework for building LLM-powered agents with modular state management, composable operators, and LLM tool-calling integration (including MCP).

## Features

- **Modular architecture** — Agents, sessions, context, tools, and plans as composable modules
- **State management** — Consistent save/load and serialization via `PygentOperator`
- **LLM-native** — Messages and tools align with OpenAI-style APIs
- **MCP support** — Use Model Context Protocol (SSE and stdio) tools via `ToolManager`
- **Toolkits** — Built-in toolkits such as `BashToolkits` for bash shell access; `TerminalToolkits` and `RestrictedTerminal` remain as compatibility aliases.

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

Examples that read `.env` files use `python-dotenv`:

```bash
pip install -e ".[examples]"
```

## Quick Start

```python
from pygent.agent import BaseAgent
from pygent.context import BaseContext
from pygent.llm import AsyncRequestsClient
from pygent.message import UserMessage
from pygent.module.tool import ToolManager
from pygent.toolkits import BashToolkits

class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.llm = AsyncRequestsClient(
            base_url="https://api.openai.com/v1",
            api_key="YOUR_API_KEY",
            model_name="gpt-4",
        )
        self.tool_manager = ToolManager()
        bash = BashToolkits(session_id="quickstart", workspace_root=".")
        self.tool_manager.add_module("bash", bash)
        self.tool_manager.register_tools(bash.get_all_tools())

    async def run(self, user_input: str):
        context = BaseContext(system_prompt="You are a helpful assistant.")
        context.add_message(UserMessage(content=user_input))
        # ... tool loop and LLM calls (see examples/react_agent)
        return context
```

See [examples/react_agent.py](examples/react_agent.py) for a ReAct-style agent with tool calling, and
[examples/multi_session_agent.py](examples/multi_session_agent.py) for a session-aware variant.

Compatibility note: `AsyncOpenAIClient` is still exported as an alias of
`AsyncRequestsClient`. `TerminalToolkits` and `RestrictedTerminal(root_dir=...)`
still delegate to the bash toolkit for compatibility, while the registered tool
name is `bash`.

## Project Layout

- **pygent/** — Core package: agents, sessions, context, LLM clients, messages, modules, and toolkits
- **examples/** — Example agents (e.g. ReAct agent)
- **docs/** — [Architecture](docs/ARCHITECTURE.md) and [API](docs/API.md) documentation
- **scripts/** — Local smoke tests and utility scripts
- **tests/** — Pytest tests

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — Design, layers, and components
- [API](docs/API.md) — API reference

## License

Apache-2.0. See [LICENSE](LICENSE) for details.
