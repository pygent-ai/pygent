"""
MCP client for SSE transport using only Python standard library.
Uses urllib for HTTP: GET for SSE stream, POST for sending requests.
"""

import json
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from pygent.common import PygentString, PygentDict

from .base import BaseMCPClient


class SSEMCPClient(BaseMCPClient):
    """MCP client for SSE: connects via HTTP URL, POST for requests, SSE for responses."""

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

    def _request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send JSON-RPC via POST, receive response from SSE endpoint."""
        base = self.url.data.rstrip("/")
        if base.endswith("/sse"):
            base = base[:-4]  # http://host:port
        post_url = f"{base}/messages/"
        req_body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or {},
        }
        headers = {"Content-Type": "application/json", **{str(k): str(v) for k, v in self.headers.data.items()}}
        req = Request(post_url, data=json.dumps(req_body).encode("utf-8"), headers=headers, method="POST")
        try:
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, json.JSONDecodeError) as e:
            raise RuntimeError(f"MCP SSE request failed: {e}") from e
        if "error" in data:
            raise RuntimeError(f"MCP error: {data['error']}")
        return data.get("result", {})
