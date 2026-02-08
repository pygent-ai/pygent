# Pygent API Reference

This document describes the public APIs of the Pygent framework.

---

## Table of Contents

1. [Common — PygentData & PygentOperator](#1-common--pygentdata--pygentoperator)
2. [Context](#2-context)
3. [Message](#3-message)
4. [LLM](#4-llm)
5. [Module](#5-module)
6. [Tool](#6-tool)
7. [MCP](#7-mcp)
8. [Toolkits](#8-toolkits)

---

## 1. Common — PygentData & PygentOperator

**Module:** `pygent.common.base`

### PygentData

Base class for all Pygent data types.

```python
class PygentData:
    data: Any
```

| Method / Attribute | Description |
|--------------------|-------------|
| `data` | The underlying value |
| `to_json() -> str` | Serialize to JSON string |
| `to_dict() -> Any` | Convert to Python dict (if applicable) |
| `copy() -> PygentData` | Create a shallow copy |
| `from_json(json_str: str) -> PygentData` | Class method to deserialize |

### PygentData Types

| Class | Base | Key Methods |
|-------|------|-------------|
| `PygentString` | `str` | `upper()`, `lower()`, `strip()`, `replace()`, `length()`, `contains()` |
| `PygentInt` | `int` | `to_float()`, `to_binary()`, `to_hex()`, `is_even()`, `is_odd()` |
| `PygentFloat` | `float` | `to_int()`, `round()`, `ceil()`, `floor()`, `is_integer()` |
| `PygentBool` | - | `__and__`, `__or__`, `__invert__` |
| `PygentList` | `list` | `filter(func)`, `map(func)`, `copy()` |
| `PygentDict` | `dict` | `set(key, value)` |
| `PygentTuple` | `tuple` | Standard tuple operations |
| `PygentSet` | `set` | `union()`, `intersection()`, `difference()`, `symmetric_difference()` |
| `PygentBytes` | `bytes` | `to_base64()`, `from_base64()`, `to_hex()`, `from_hex()`, `decode()` |
| `PygentDateTime` | `datetime` | `now()`, `from_timestamp()`, `from_isoformat()`, `to_timestamp()`, `format()` |
| `PygentDate` | `date` | `today()`, `from_isoformat()`, `to_isoformat()`, `format()` |
| `PygentTime` | `time` | `now()`, `format()` |
| `PygentDecimal` | `Decimal` | Standard Decimal operations |
| `PygentEnum` | - | `name`, `value` |
| `PygentNone` | - | `is_none()` |
| `PygentAny` | - | `get_type()`, `isinstance()` |

### create_pygent_data

```python
def create_pygent_data(data: Any) -> PygentData
```

Factory: creates the appropriate `PygentData` subclass from a Python value.

---

### PygentOperator

Base class for stateful components.

```python
class PygentOperator:
```

| Method | Description |
|--------|-------------|
| `state_dict() -> Dict[str, Any]` | Export all PygentData fields |
| `load_state_dict(state_dict, strict=True)` | Restore state |
| `save(path, format='json', include_metadata=True) -> str` | Persist to file |
| `load(path, format='auto', strict=True)` | Load from file |
| `parameters() -> Dict[str, Any]` | PyTorch-style parameters |
| `named_parameters() -> List[tuple]` | Named parameters |
| `to(*args, **kwargs) -> self` | Chainable (no-op) |
| `train(mode=True) -> self` | Chainable (no-op) |
| `eval() -> self` | Chainable (no-op) |

---

## 2. Context

**Module:** `pygent.context.base`

### BaseContext

```python
class BaseContext(PygentOperator):
    history: PygentList[BaseMessage]

    def __init__(self, system_prompt=None, *args, **kwargs)
    def add_message(self, message: BaseMessage) -> None
```

| Attribute / Method | Description |
|--------------------|-------------|
| `history` | List of messages in the conversation |
| `add_message(message)` | Append a message to history |

---

## 3. Message

**Module:** `pygent.message.base`

### Message Roles

```python
class MessageRole(PygentString):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"
```

### BaseMessage

```python
class BaseMessage(PygentData):
    role: PygentString
    content: PygentString
    name: Optional[PygentString] = None
    metadata: Optional[PygentDict] = None

    def __init__(self, role, content, name=None, metadata=None, **kwargs)
    def to_dict() -> Dict[str, Any]
    def to_openai_format() -> Dict[str, Any]
    @classmethod
    def from_dict(cls, data: Dict) -> BaseMessage
```

### Message Subclasses

| Class | Role | Extra Parameters |
|-------|------|------------------|
| `SystemMessage` | system | - |
| `UserMessage` | user | `name` |
| `AssistantMessage` | assistant | `name`, `tool_calls` |
| `ToolMessage` | tool | `tool_call_id` |
| `FunctionMessage` | function | `name` |

### ToolCall

```python
class ToolCall(PygentData):
    tool_call_id: PygentString
    tool_name: PygentString
    arguments: PygentDict

    def __init__(self, tool_call_id, tool_name, arguments, **kwargs)
    def to_dict() -> Dict[str, Any]
    @classmethod
    def from_dict(cls, data: Dict) -> ToolCall
```

### FunctionCall

```python
class FunctionCall(PygentData):
    name: PygentString
    arguments: PygentDict
```

---

## 4. LLM

**Module:** `pygent.llm.base`, `pygent.llm.openai_client`

### BaseClient

```python
class BaseClient(PygentOperator, ABC):
    base_url: PygentString
    api_key: PygentString
    model_name: PygentString
    timeout: PygentInt
    max_retries: PygentInt
    temperature: PygentFloat
    max_tokens: PygentInt
    stream: PygentBool

    def __init__(self, base_url, api_key, model_name, timeout=30, max_retries=3,
                 temperature=0.7, max_tokens=None, stream=False, **kwargs)
    def forward(self, context: BaseContext) -> BaseContext  # abstract
```

### BaseAsyncClient

Same structure as `BaseClient`, with async `forward`:

```python
async def forward(self, context: BaseContext) -> BaseContext
```

### AsyncOpenAIClient

```python
class AsyncOpenAIClient(BaseAsyncClient):
    def __init__(self, base_url, api_key, model_name, ...)
    async def forward(self, context: BaseContext, **kwargs) -> BaseContext
```

---

## 5. Module

**Module:** `pygent.module.base`

### PygentModule

```python
class PygentModule(PygentOperator):
    def __init__(self)
    def add_module(self, name: str, module: PygentOperator) -> None
    def modules(self) -> List[PygentOperator]
    def named_modules(self) -> List[tuple]
    def state_dict(self) -> Dict[str, Any]
    def load_state_dict(self, state_dict, strict=True) -> None
    def save(self, path, format='json', include_metadata=True) -> str
```

---

## 6. Tool

**Module:** `pygent.module.tool.base`, `pygent.module.tool.tool_manager`, `pygent.module.tool.utils`

### Enums

```python
class ToolCategory(Enum):
    SEARCH, CALCULATION, DATABASE, FILE, NETWORK, SYSTEM, UTILITY, AI, CUSTOM

class ToolPermission(Enum):
    PUBLIC, LIMITED, PRIVATE, ADMIN
```

### ToolMetadata

```python
@dataclass
class ToolMetadata:
    name: str
    description: str
    version: str = "1.0.0"
    author: str = "unknown"
    category: ToolCategory = ToolCategory.UTILITY
    permission: ToolPermission = ToolPermission.PUBLIC
    tags: List[str] = []
    rate_limit: Optional[int] = None
    timeout: float = 30.0
    requires_auth: bool = False
    deprecated: bool = False
    ...
```

### ToolParameter

```python
class ToolParameter(PygentData):
    def __init__(self, name, type, description="", required=True, default=None,
                 enum=None, min_value=None, max_value=None, pattern=None, **kwargs)
    def to_openai_schema(self) -> Dict[str, Any]
```

### BaseTool

```python
class BaseTool(PygentModule):
    metadata: PygentDict
    parameters: PygentDict
    config: PygentDict
    enabled: PygentBool
    call_count: PygentInt
    last_called: PygentString
    error_count: PygentInt

    def __init__(self, name, description, version="1.0.0", **kwargs)
    def forward(self, *args, **kwargs) -> Any  # override in subclass
    def __call__(self, *args, **kwargs) -> Dict[str, Any]
    def validate_parameters(self, parameters: Dict) -> Dict[str, List[str]]
    def to_openai_function(self) -> Dict[str, Any]
    def to_langchain_tool(self) -> Dict[str, Any]
    def get_schema(self) -> Dict[str, Any]
    def enable(self) -> None
    def disable(self) -> None
    def reset_stats(self) -> None
    def update_config(self, config: Dict) -> None
```

**Call response format:**

```python
{
    "success": bool,
    "result": Any,           # if success
    "error": str,            # if not success
    "metadata": {...},
    "status": {...},
    "details": {...},        # optional validation errors
    "exception": {...}       # optional exception info
}
```

### ToolManager

```python
class ToolManager(PygentModule):
    tools: PygentDict
    tool_categories: PygentDict
    mcp_clients: PygentDict

    def __init__(self)
    def register_tool(self, tool: BaseTool) -> None
    def get_tool(self, name: str) -> Optional[BaseTool]
    def call_tool(self, name: str, *args, **kwargs) -> Dict[str, Any]
    def get_all_schemas(self) -> Dict[str, Any]
    def get_openai_functions(self) -> List[Dict[str, Any]]
    def add_mcp_server_stdio(self, server_id, command, args=None, env=None,
                             tool_name_prefix=None) -> List[BaseTool]
    def add_mcp_server_sse(self, server_id, url, headers=None,
                           tool_name_prefix=None) -> List[BaseTool]
```

### Decorators

| Decorator | Description |
|-----------|-------------|
| `@tool(name, description, version, ...)` | Convert function to `BaseTool` |
| `@auto_tool(name, description, ...)` | Same, with docstring param extraction |
| `@tool_method(name, description, ...)` | Convert class method to tool |
| `@tool_class(...)` | Convert class with `@tool_method` methods to tool set |
| `@async_tool(...)` | Async function → sync wrapper + tool |
| `@async_auto_tool(...)` | Async + auto docstring |
| `@register_tool(name)` | Register to global `ToolRegistry` |

### ToolRegistry

```python
registry = ToolRegistry()  # singleton

def register(self, func_or_tool, name=None) -> Any
def get(self, name: str) -> Optional[BaseTool]
def list_all(self) -> List[str]
def clear(self) -> None
def get_all_tools(self) -> Dict[str, BaseTool]
def get_tool_manager(self) -> ToolManager
```

### Predefined Decorators

- `calculator` — category=CALCULATION
- `searcher` — category=SEARCH
- `converter`, `validator` — category=UTILITY
- `generator` — category=AI
- `file_processor` — category=FILE
- `database` — category=DATABASE
- `network` — category=NETWORK

---

## 7. MCP

**Module:** `pygent.module.tool.mcp`

### BaseMCPClient

```python
class BaseMCPClient(PygentOperator):
    server_id: PygentString

    def __init__(self, server_id: str, **kwargs)
    def list_tools(self) -> List[Any]
    def call_tool(self, name: str, arguments: Optional[Dict] = None) -> Any
```

### StdioMCPClient

```python
class StdioMCPClient(BaseMCPClient):
    command: PygentString
    args: PygentList
    env: PygentDict

    def __init__(self, server_id, command, args=None, env=None, **kwargs)
```

### SSEMCPClient

```python
class SSEMCPClient(BaseMCPClient):
    url: PygentString
    headers: PygentDict

    def __init__(self, server_id, url, headers=None, **kwargs)
```

### MCPToolAdapter

```python
class MCPToolAdapter(BaseTool):
    def __init__(self, client, tool_name, tool_description, input_schema,
                 server_id=None, version="1.0.0")
    def forward(self, *args, **kwargs) -> Any
```

---

## 8. Toolkits

**Module:** `pygent.toolkits.terminal`

### RestrictedTerminal

Sandboxed terminal for file operations within a root directory.

```python
class RestrictedTerminal:
    def __init__(self, root_dir)
    def get_absolute_path(self, path=None)
    def safe_path_check(self, path)
    def change_directory(self, args)
    def list_files(self, args)
    def make_directory(self, args)
    def remove_file_or_dir(self, args)
    def print_working_directory(self, args)
    def copy_file(self, args)
    def move_file(self, args)
    def view_file(self, args)
    def show_help(self)
    def tree_command(self, args)
    def touch_file(self, args)
    def find_files(self, args)
    def process_command(self, cmd_line)
    def run_command(self, cmd: str) -> str
    def run_commands(self, commands: list) -> list
```

Supported commands: `cd`, `ls`, `pwd`, `mkdir`, `rm`, `cp`, `mv`, `cat`, `touch`, `tree`, `find`, `help`, `exit`, plus external commands.

---

## Agent

**Module:** `pygent.agent.base_agent`

### BaseAgent

```python
class BaseAgent(PygentOperator):
    pass
```

Placeholder for future agent orchestration logic.
