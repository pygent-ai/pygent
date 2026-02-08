"""
Base MCP client that inherits PygentOperator and uses PygentData.
Provides sync wrappers over the async MCP SDK (list_tools, call_tool).
"""

import asyncio
from typing import Any, Dict, List, Optional

from pygent.common import PygentOperator, PygentString

# MCP SDK imports (optional at module load for lazy use)
try:
    from mcp import ClientSession
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False


class BaseMCPClient(PygentOperator):
    """
    Base MCP client: inherits PygentOperator, uses PygentData for config.
    Subclasses implement transport (stdio vs SSE) via _run_with_transport().
    """

    server_id: PygentString
    """Identifier for this MCP server (e.g. used as tool name prefix)."""

    def __init__(self, server_id: str, **kwargs: Any):
        super().__init__()
        self.server_id = PygentString(server_id)
        if not _MCP_AVAILABLE:
            raise RuntimeError(
                "MCP SDK is not installed. Install with: pip install mcp"
            )

    def _run_async(self, coro: Any) -> Any:
        """Run an async coroutine in a new event loop."""
        return asyncio.run(coro)

    async def _session_flow(
        self,
        read_stream: Any,
        write_stream: Any,
        *,
        list_tools_only: bool = False,
        call_tool_name: Optional[str] = None,
        call_tool_arguments: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Create session, initialize, then either list_tools or call_tool.
        Returns ListToolsResult or CallToolResult.
        """
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            if list_tools_only:
                return await session.list_tools()
            if call_tool_name is not None:
                return await session.call_tool(
                    call_tool_name,
                    arguments=call_tool_arguments,
                )
            return await session.list_tools()

    async def _connect_and_run(
        self,
        list_tools_only: bool = False,
        call_tool_name: Optional[str] = None,
        call_tool_arguments: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Override in subclasses: open transport, run _session_flow, return result.
        """
        raise NotImplementedError("Subclasses must implement _connect_and_run")

    def list_tools(self) -> List[Any]:
        """Sync: list tools from the MCP server. Returns list of mcp.types.Tool."""
        result = self._run_async(
            self._connect_and_run(list_tools_only=True)
        )
        if hasattr(result, "tools"):
            return list(result.tools)
        return []

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        """Sync: call a tool by name. Returns mcp.types.CallToolResult."""
        return self._run_async(
            self._connect_and_run(
                call_tool_name=name,
                call_tool_arguments=arguments,
            )
        )
