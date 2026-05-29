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
│  ┌─────────────┐ ┌─────────────┐ ┌──────────────────────────────────────┐ │
│  │   Session   │ │    Plan     │ │             ToolManager              │ │
│  │             │ │ InMemoryPlan│ │  ┌────────────────────────────────┐  │ │
│  │             │ │             │ │  │ BaseTool / MCPToolAdapter      │  │ │
│  │             │ │             │ │  └────────────────────────────────┘  │ │
│  └─────────────┘ └─────────────┘ └──────────────────────────────────────┘ │
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
- `save(path, format='json'|'pickle')`: Persist to disk
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
- `to_openai_format()`: Compatibility method that delegates to provider adapters
- `pygent.message.adapters`: OpenAI/Ollama wire-format adapters

#### LLM Clients (`pygent.llm`)

- `BaseClient`: Abstract sync LLM client
- `BaseAsyncClient`: Abstract async LLM client
- `AsyncRequestsClient`: OpenAI-compatible implementation using urllib
- `AsyncOpenAIClient`: Backward-compatible alias of `AsyncRequestsClient`
- `OllamaAsyncClient`: Optional Ollama backend; requires the third-party `ollama` package only when used
- `forward(context: BaseContext) -> AssistantMessage`: appends to context and returns the new assistant message

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
- `to_openai_tool()`: OpenAI tools wrapper format
- `get_static_schema()`: Schema/config without runtime counters
- `get_status()`: Runtime status
- `get_schema()`: Backward-compatible full tool description, with status by default

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

- `BaseMCPClient`: Abstract JSON-RPC client implemented with the Python standard library
- `StdioMCPClient`: Spawn process, communicate over stdio
- `SSEMCPClient`: Send JSON-RPC over HTTP endpoints compatible with the local test server shape
- `MCPToolAdapter`: Wrap MCP tool as `BaseTool`, delegate to client

#### Implemented and Planned Modules

- `pygent.session`: JSON session persistence and recovery
- `pygent.module.plan`: `BasePlan` and `InMemoryPlan`
- `pygent.toolkits.base_tool_search`: Placeholder base for future toolkit search work
- Memory, knowledge, reasoning, and skill modules are design-level extension points, not current package directories

---

### 4. Application Layer

#### BaseAgent (`pygent.agent.base`)

Minimal agent stub that subclasses `PygentOperator`. Intended to compose context, LLM clients, tools, sessions, plans, and future extension modules.

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
PygentOperator.state_dict() → JSON/Pickle
    ↓
File on disk
    ↓
PygentOperator.load(path) → load_state_dict()
```

---

## External Dependencies

- **stdlib urllib**: For `AsyncRequestsClient`
- **python-dotenv**: Used by examples only, available through the `examples` extra
- **ollama**: Optional runtime dependency only when using `OllamaAsyncClient`; it is not installed by default

---

## Directory Structure

```
pygent/
├── __init__.py
├── common/
│   └── base.py          # PygentData, PygentOperator
├── agent/
│   └── base.py          # BaseAgent
├── context/
│   └── base.py          # BaseContext
├── session/
│   └── base.py          # Session
├── message/
│   └── base.py          # Message types, ToolCall
├── llm/
│   ├── base.py          # BaseClient, BaseAsyncClient
│   ├── requests_client.py # AsyncRequestsClient
│   └── ollama_client.py # OllamaAsyncClient
├── module/
│   ├── base.py          # PygentModule
│   ├── tool/
│   │   ├── base.py      # BaseTool, ToolParameter, ToolMetadata
│   │   ├── tool_manager.py
│   │   ├── utils.py     # Decorators, ToolRegistry
│   │   └── mcp/         # MCP clients, tool adapter
│   ├── plan/
│   │   ├── base.py
│   │   └── in_memory_plan.py
└── toolkits/
    ├── file_operations.py # FileToolkits
    ├── bash.py # BashToolkits
    ├── web_search.py # WebSearchToolkits
    └── web_fetch.py # WebFetchToolkits
```

---

## Extensibility

1. **New LLM backends**: Subclass `BaseClient` or `BaseAsyncClient`, implement `forward(context)`.
2. **New tools**: Subclass `BaseTool` and implement `forward()`, or use `@tool` / `@auto_tool`.
3. **MCP servers**: Use `ToolManager.add_mcp_server_stdio()` or `add_mcp_server_sse()`.
4. **New modules**: Subclass `PygentModule`, add submodules, implement behavior.
5. **New agent logic**: Extend `BaseAgent` with orchestration over context, LLM, and tools.
