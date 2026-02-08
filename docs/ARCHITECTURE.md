# Pygent Architecture

## Overview

Pygent is a Python framework for building LLM-powered agents. It provides a modular architecture centered on state management, composable operators, and LLM tool-calling integration. The framework uses a layered design from primitive data types up through agents, with clear abstractions at each level.

---

## Core Design Principles

1. **State Management**: All components inherit from `PygentOperator`, enabling consistent state serialization (save/load) across the stack.
2. **Composability**: `PygentModule` allows nesting submodules and propagating state through the hierarchy.
3. **LLM-Native**: Messages, contexts, and tools align with OpenAI-style APIs for easy integration.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Application Layer                               │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         BaseAgent (PygentOperator)                      │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Module Layer                                    │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────────────────────┐ │
│  │   Memory    │ │  Knowledge  │ │    Plan     │ │   ToolManager          │ │
│  │   (placeholder) │ (placeholder) │ (placeholder) │  ┌──────────────────┐ │ │
│  │             │ │             │ │             │ │  │ BaseTool(s)       │ │ │
│  │             │ │             │ │             │ │  │ MCPToolAdapter(s) │ │ │
│  │             │ │             │ │             │ │  └──────────────────┘ │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └────────────────────────┘ │
│                              PygentModule                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Infrastructure Layer                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │   BaseContext   │  │   BaseClient    │  │       Message Types         │  │
│  │   (history)     │  │   (LLM client)  │  │ System/User/Assistant/Tool  │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Foundation Layer                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      PygentOperator                                  │   │
│  │  (state_dict, load_state_dict, save, load, parameters)               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                        │                                    │
│                                        ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      PygentData Types                                │   │
│  │  PygentString, PygentInt, PygentDict, PygentList, PygentBool, ...    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Layer Details

### 1. Foundation Layer (`pygent.common.base`)

#### PygentData

Base class for all data types used in Pygent. Provides:

- `data`: The underlying value
- `to_json()`, `to_dict()`: Serialization
- `copy()`: Cloning
- `from_json()`: Deserialization

Concrete types extend Python primitives:

| Type           | Base   | Purpose                      |
|----------------|--------|------------------------------|
| `PygentString` | `str`  | Mutable string with `.data`  |
| `PygentInt`    | `int`  | Integer with helpers         |
| `PygentFloat`  | `float`| Float with math helpers      |
| `PygentBool`   | -      | Boolean wrapper              |
| `PygentList`   | `list` | List with filter/map         |
| `PygentDict`   | `dict` | Dictionary wrapper           |
| `PygentTuple`  | `tuple`| Tuple wrapper                |
| `PygentSet`    | `set`  | Set with set operations      |
| `PygentBytes`  | `bytes`| Bytes with base64/hex        |
| `PygentDateTime` | `datetime` | DateTime utilities       |
| `PygentDate`   | `date` | Date utilities               |
| `PygentTime`   | `time` | Time utilities               |
| `PygentDecimal`| `Decimal` | High-precision decimal   |
| `PygentEnum`   | -      | Enum wrapper                 |
| `PygentNone`   | -      | Explicit None                |
| `PygentAny`    | -      | Generic wrapper              |

#### PygentOperator

Base class for components that hold state. Uses type hints to auto-discover `PygentData` fields and provides:

- `state_dict()`: Export all PygentData fields
- `load_state_dict()`: Restore state
- `save(path, format='json'|'yaml'|'pickle')`: Persist to disk
- `load(path)`: Load from disk
- `parameters()`, `named_parameters()`: PyTorch-style parameter access

---

### 2. Infrastructure Layer

#### BaseContext (`pygent.context.base`)

Holds conversation history and optional system prompt.

- `history: PygentList[BaseMessage]`: Ordered message list
- `add_message(message)`: Append to history
- Accepts `system_prompt` to initialize with a `SystemMessage`

#### Message Types (`pygent.message.base`)

- `BaseMessage`: role, content, name, metadata
- `SystemMessage`, `UserMessage`, `AssistantMessage`: Standard chat roles
- `ToolMessage`: Tool result (tool_call_id)
- `FunctionMessage`: Legacy function-call format
- `ToolCall`, `FunctionCall`: Tool invocation metadata
- `to_openai_format()`: Convert to OpenAI API format

#### LLM Clients (`pygent.llm`)

- `BaseClient`: Abstract sync LLM client
- `BaseAsyncClient`: Abstract async LLM client
- `AsyncOpenAIClient`: OpenAI-compatible implementation (uses `openai` SDK)
- `forward(context: BaseContext) -> BaseContext`: In-place update of context with model response

---

### 3. Module Layer (`pygent.module`)

#### PygentModule (`pygent.module.base`)

Extends `PygentOperator` with submodule composition:

- `_modules`: Dict of child modules
- `add_module(name, module)`: Register submodule
- `modules()`, `named_modules()`: Iterate children
- `state_dict()`: Includes `_modules` state
- `load_state_dict()`: Recursively loads children
- `save()`: Includes `module_names` in metadata

#### Tool System (`pygent.module.tool`)

**BaseTool**:

- Subclasses `PygentModule`
- Fields: metadata, parameters, config, enabled, call_count, last_called, error_count
- `forward(*args, **kwargs)`: Implement in subclasses
- `__call__()`: Validates, executes, returns `{success, result, metadata, status}`
- `_discover_parameters()`: Infers parameters from `forward` signature
- `validate_parameters()`: Type and constraint checking
- `to_openai_function()`: OpenAI function-calling schema
- `get_schema()`: Full tool description

**ToolManager**:

- Registers tools by name
- `register_tool(tool)`: Add tool, categorize by metadata
- `call_tool(name, *args, **kwargs)`: Dispatch to tool
- `get_openai_functions()`: All tools as OpenAI format
- `add_mcp_server_stdio()`: Connect MCP via stdio, register tools
- `add_mcp_server_sse()`: Connect MCP via SSE, register tools

**Decorators** (`utils.py`):

- `@tool()`: Turn a function into a `BaseTool`
- `@auto_tool()`: Extract params from docstring (Google/Numpy/RST)
- `@tool_method()`, `@tool_class()`: Class-based tools
- `@async_tool()`, `@async_auto_tool()`: Async support
- `ToolRegistry`: Global registry, `@register_tool()`

**MCP Integration** (`pygent.module.tool.mcp`):

- `BaseMCPClient`: Abstract MCP client, sync wrappers over async MCP SDK
- `StdioMCPClient`: Spawn process, communicate over stdio
- `SSEMCPClient`: Connect to MCP server via HTTP SSE
- `MCPToolAdapter`: Wrap MCP tool as `BaseTool`, delegate to client

#### Placeholder Modules

- `memory.base_memory`: Empty
- `knowledge.base_knowledge`: Empty
- `plan.base_plan`: Empty
- `reasoning.base_reasoning`: Empty
- `skill.base_skill`: Empty
- `tool_search.base_tool_search`: Empty

---

### 4. Application Layer

#### BaseAgent (`pygent.agent.base_agent`)

Minimal agent stub that subclasses `PygentOperator`. Intended to compose modules (context, LLM client, tools, memory, etc.) and orchestrate the agent loop.

---

## Data Flow

### Typical Agent Loop (Conceptual)

```
User Input → BaseContext.add_message(UserMessage(...))
    → BaseClient.forward(context)  # LLM returns AssistantMessage
    → If tool_calls: ToolManager.call_tool(name, **args)
    → BaseContext.add_message(ToolMessage(...))
    → BaseClient.forward(context)  # Continue until final response
    → Output
```

### State Persistence

```
PygentOperator.state_dict() → JSON/YAML/Pickle
    ↓
File on disk
    ↓
PygentOperator.load(path) → load_state_dict()
```

---

## External Dependencies

- **openai**: For `AsyncOpenAIClient`
- **mcp**: For MCP clients (stdio, SSE) and tool adapters
- **anyio**: For sync wrappers over async MCP SDK
- **yaml**: For YAML save/load

---

## Directory Structure

```
pygent/
├── __init__.py
├── common/
│   └── base.py          # PygentData, PygentOperator
├── agent/
│   └── base_agent.py    # BaseAgent
├── context/
│   └── base.py          # BaseContext
├── message/
│   └── base.py          # Message types, ToolCall
├── llm/
│   ├── base.py          # BaseClient, BaseAsyncClient
│   └── openai_client.py # AsyncOpenAIClient
├── module/
│   ├── base.py          # PygentModule
│   ├── tool/
│   │   ├── base.py      # BaseTool, ToolParameter, ToolMetadata
│   │   ├── tool_manager.py
│   │   ├── utils.py     # Decorators, ToolRegistry
│   │   └── mcp/         # MCP clients, tool adapter
│   ├── memory/
│   ├── knowledge/
│   ├── plan/
│   ├── reasoning/
│   ├── skill/
│   └── tool_search/
└── toolkits/
    └── terminal.py      # RestrictedTerminal (sandboxed fs)
```

---

## Extensibility

1. **New LLM backends**: Subclass `BaseClient` or `BaseAsyncClient`, implement `forward(context)`.
2. **New tools**: Subclass `BaseTool` and implement `forward()`, or use `@tool` / `@auto_tool`.
3. **MCP servers**: Use `ToolManager.add_mcp_server_stdio()` or `add_mcp_server_sse()`.
4. **New modules**: Subclass `PygentModule`, add submodules, implement behavior.
5. **New agent logic**: Extend `BaseAgent` with orchestration over context, LLM, and tools.
