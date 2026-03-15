"""
Base MCP client using only Python standard library (no mcp package).
Implements JSON-RPC 2.0 over stdio/SSE for list_tools and call_tool.
"""

import json
from typing import Any, Dict, List, Optional

from pygent.common import PygentOperator, PygentString


class _DotDict:
    """Wrapper for dict to support attribute access (getattr)."""

    def __init__(self, d: Dict[str, Any]):
        self._d = {}
        for k, v in d.items():
            if isinstance(v, dict):
                self._d[k] = _DotDict(v)
            elif isinstance(v, list):
                self._d[k] = [_DotDict(x) if isinstance(x, dict) else x for x in v]
            else:
                self._d[k] = v

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return self._d.get(name)

    def get(self, key: str, default: Any = None) -> Any:
        return self._d.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        def _conv(v: Any) -> Any:
            if isinstance(v, _DotDict):
                return v.to_dict()
            if isinstance(v, list):
                return [_conv(x) for x in v]
            return v
        return {k: _conv(v) for k, v in self._d.items()}


class BaseMCPClient(PygentOperator):
    """
    Base MCP client: inherits PygentOperator.
    Subclasses implement transport via _request(method, params).
    """

    server_id: PygentString

    def __init__(self, server_id: str, **kwargs: Any):
        super().__init__()
        self.server_id = PygentString(server_id)

    def _request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Override in subclasses: send JSON-RPC request, return parsed result.
        Raises on error.
        """
        raise NotImplementedError("Subclasses must implement _request")

    def list_tools(self) -> List[Any]:
        """List tools from MCP server. Returns list of tool objects (name, description, inputSchema)."""
        result = self._request("tools/list", params={})
        tools = result.get("tools") or []
        out = []
        for t in tools:
            out.append(_DotDict(t) if isinstance(t, dict) else t)
        return out

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        """Call tool by name. Returns result with .content, .isError, .structuredContent."""
        result = self._request(
            "tools/call",
            params={"name": name, "arguments": arguments or {}},
        )
        return _DotDict(result)
