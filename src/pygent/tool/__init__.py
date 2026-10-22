from typing import TYPE_CHECKING

from .executors import (
    AgentToolExecutor,
    ExecutorRegistry,
    HttpToolExecutor,
    InMemoryToolTaskManager,
    LocalToolExecutor,
    SandboxExecutorSupport,
    ToolExecution,
    ToolExecutionContext,
    ToolExecutionError,
    ToolExecutor,
    ToolRunner,
    ToolTaskAdmission,
    ToolTaskManager,
)
from .functional import ToolExecutorFactory, ToolKit, tool
from .layer import ToolCallLayer, TrustedAuthorizationAdapter
from .task_handle import ToolTaskHandle
from .types import (
    IdempotencyPolicy,
    MediaBlock,
    MediaSource,
    ToolAuthorizationDecision,
    ToolAuthorizationRequest,
    ToolCall,
    ToolDefinition,
    ToolOutput,
    ToolResult,
    ToolResultContent,
    ToolResultJson,
    ToolResultText,
    ToolSideEffect,
    ToolSpec,
    ToolTask,
    ToolTaskState,
)

if TYPE_CHECKING:
    from .mcp.pool import MCPClientPool
    from .standard import (
        BashTools,
        FileTools,
        PowerShellTools,
        ShellIdentity,
        StandardTools,
        TerminalSessionStore,
        TerminalTools,
        WebFetchTools,
        WebSearchTools,
        ZshTools,
        describe_shell_environment,
    )

_STANDARD_EXPORTS = frozenset(
    {
        "BashTools",
        "FileTools",
        "PowerShellTools",
        "ShellIdentity",
        "StandardTools",
        "TerminalSessionStore",
        "TerminalTools",
        "WebFetchTools",
        "WebSearchTools",
        "ZshTools",
        "describe_shell_environment",
    }
)

_MCP_EXPORTS = frozenset({"MCPClientPool"})


def __getattr__(name: str) -> object:
    if name in _STANDARD_EXPORTS:
        from . import standard

        value = getattr(standard, name)
        globals()[name] = value
        return value
    if name in _MCP_EXPORTS:
        from .mcp.pool import MCPClientPool

        globals()[name] = MCPClientPool
        return MCPClientPool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AgentToolExecutor",
    "BashTools",
    "ExecutorRegistry",
    "FileTools",
    "HttpToolExecutor",
    "IdempotencyPolicy",
    "InMemoryToolTaskManager",
    "LocalToolExecutor",
    "MCPClientPool",
    "MediaBlock",
    "MediaSource",
    "PowerShellTools",
    "SandboxExecutorSupport",
    "ShellIdentity",
    "StandardTools",
    "TerminalSessionStore",
    "TerminalTools",
    "ToolAuthorizationDecision",
    "ToolAuthorizationRequest",
    "ToolCall",
    "ToolCallLayer",
    "ToolDefinition",
    "ToolExecution",
    "ToolExecutionContext",
    "ToolExecutionError",
    "ToolExecutor",
    "ToolExecutorFactory",
    "ToolKit",
    "ToolOutput",
    "ToolResult",
    "ToolResultContent",
    "ToolResultJson",
    "ToolResultText",
    "ToolRunner",
    "ToolSideEffect",
    "ToolSpec",
    "ToolTask",
    "ToolTaskAdmission",
    "ToolTaskHandle",
    "ToolTaskManager",
    "ToolTaskState",
    "TrustedAuthorizationAdapter",
    "WebFetchTools",
    "WebSearchTools",
    "ZshTools",
    "describe_shell_environment",
    "tool",
]
