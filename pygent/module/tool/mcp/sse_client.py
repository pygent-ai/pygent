"""
MCP client for SSE (Server-Sent Events) transport: connects via HTTP URL.
"""

from typing import Any, Dict, List, Optional

from pygent.common import PygentString, PygentDict

from .base import BaseMCPClient

from mcp.client.sse import sse_client


class SSEMCPClient(BaseMCPClient):
    """
    MCP client for SSE transport.
    Uses PygentData: server_id, url, headers.
    """

    url: PygentString
    headers: PygentDict

    def __init__(
        self,
        server_id: str,
        url: str,
        headers: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(server_id, **kwargs)
        self.url = PygentString(url)
        self.headers = PygentDict(dict(headers) if headers else {})

    async def _connect_and_run(
        self,
        list_tools_only: bool = False,
        call_tool_name: Optional[str] = None,
        call_tool_arguments: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if sse_client is None:
            raise RuntimeError("MCP SSE client not available (pip install mcp)")
        # headers: SDK expects dict[str, Any] for SSE
        h = self.headers.data if self.headers.data else None
        async with sse_client(self.url.data, headers=h) as (read_stream, write_stream):
            return await self._session_flow(
                read_stream,
                write_stream,
                list_tools_only=list_tools_only,
                call_tool_name=call_tool_name,
                call_tool_arguments=call_tool_arguments,
            )
