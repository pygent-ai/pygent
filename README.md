# Pygent 0.2

Pygent is a PyTorch-like framework for composing stateless LLM agents from reusable `Module` graphs. Version 0.2 provides a single Execution owner for direct and managed calls, an asyncio Runtime, observable model/tool events, OpenAI-compatible adapters, portable tools and MCP adapters, ReAct, SQLite recovery, and HTTP/SSE workers.

## First principles

[Pygent 0.2 第一原则](docs/FEATURES.md) is the repository's highest authority. Module-specific first principles, SDK contracts, tests, examples, and code derive from it.

## Quickstart

```python
from pygent import AIMessage, Context, Module, UserMessage


class Echo(Module[UserMessage, AIMessage]):
    async def forward(self, message, context):
        output = AIMessage(content=message.content.upper())
        return output, context + message + output


message, context = await Echo().invoke(UserMessage(content="hello"), Context())
```

Direct `invoke()`, `stream()`, and `start()` need no Runtime. Bind the same graph to `LocalRuntime` when the application needs bounded concurrency, cancellation, deadlines, remote placement, durable history, or advanced Execution control. Both modes execute the same `forward()` graph and return `(message, context)`; see the [0.2 Execution contract](docs/EXECUTION.md).

## Documentation

- [渐进式 Agent 开发教程](docs/agent/TUTORIAL.md) — 从零密钥离线示例到工具、流式、Runtime 与动态模型配置
- [Layered documentation](docs/README.md)
- [Execution contract](docs/EXECUTION.md)
- [Module](docs/module/README.md), [Context](docs/context/README.md), [Runtime](docs/runtime/README.md)
- [Agent](docs/agent/README.md), [LLM](docs/llm/README.md), [Tool](docs/tool/README.md)
- [Executionnable service example](examples/service/README.md)
- [Runnable tutorial example](examples/tutorial/)
- [Native full-path load system](benchmarks/README.md)

## Development

```bash
uv sync --all-extras
uv run pytest -q
uv run ruff check src tests examples benchmarks
uv run mypy src benchmarks
uv build
python -m examples.service.main
uv run --extra performance python -m benchmarks run synthetic-smoke
```

Python 3.11+ is required. Apache-2.0; see [LICENSE](LICENSE).
