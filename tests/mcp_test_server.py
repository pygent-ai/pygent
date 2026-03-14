"""
Minimal MCP server using only Python standard library (no mcp package).
Run: python tests/mcp_test_server.py           # stdio (default)
     python tests/mcp_test_server.py --sse     # SSE on http://127.0.0.1:8765/sse
Exposes tools: get_memory_statistics, list_memory_files, write_raw_conversation.
"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread


def _tools_list() -> list:
    return [
        {
            "name": "get_memory_statistics",
            "title": "Get Memory Statistics",
            "description": "Returns memory usage statistics",
            "inputSchema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "list_memory_files",
            "title": "List Memory Files",
            "description": "Lists memory files",
            "inputSchema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "write_raw_conversation",
            "title": "Write Raw Conversation",
            "description": "Writes raw conversation to memory",
            "inputSchema": {
                "type": "object",
                "properties": {"content": {"type": "string"}},
                "required": ["content"],
            },
        },
    ]


def _call_tool(name: str, arguments: dict) -> dict:
    args = arguments or {}
    if name == "get_memory_statistics":
        return {"content": [{"type": "text", "text": '{"total_files": 0, "total_size_bytes": 0}'}], "isError": False}
    if name == "list_memory_files":
        return {"content": [{"type": "text", "text": "[]"}], "isError": False}
    if name == "write_raw_conversation":
        content = args.get("content", "")
        return {"content": [{"type": "text", "text": f"Wrote {len(content)} chars"}], "isError": False}
    return {"content": [{"type": "text", "text": f"Unknown tool: {name}"}], "isError": True}


def run_stdio() -> None:
    """Stdio transport: newline-delimited JSON-RPC on stdin/stdout."""
    def read_msg():
        line = sys.stdin.readline()
        return json.loads(line) if line else None

    def write_msg(msg: dict) -> None:
        sys.stdout.write(json.dumps(msg) + "\n")
        sys.stdout.flush()

    while True:
        msg = read_msg()
        if msg is None:
            break
        req_id = msg.get("id")
        method = msg.get("method", "")
        params = msg.get("params") or {}

        if method == "initialize":
            write_msg({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "mcp-test-server", "version": "1.0.0"},
                },
            })
        elif method == "notifications/initialized":
            pass  # no response
        elif method == "tools/list":
            write_msg({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": _tools_list()},
            })
        elif method == "tools/call":
            name = params.get("name", "")
            args = params.get("arguments") or {}
            write_msg({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": _call_tool(name, args),
            })
        else:
            if req_id is not None:
                write_msg({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method not found: {method}"}})


def run_sse(port: int = 8765) -> None:
    """SSE transport: GET /sse for events, POST /messages/ for requests."""

    result_holder: dict = {}  # request_id -> result

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # noqa: N802
            pass  # quiet

        def do_GET(self) -> None:
            if self.path == "/sse":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.flush()
                while True:
                    import time
                    time.sleep(1)
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()

        def do_POST(self) -> None:
            if self.path.startswith("/messages") or self.path == "/messages/":
                n = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(n).decode("utf-8")
                msg = json.loads(body)
                req_id = msg.get("id")
                method = msg.get("method", "")
                params = msg.get("params") or {}

                if method == "initialize":
                    result = {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "mcp-test-server", "version": "1.0.0"},
                    }
                elif method == "tools/list":
                    result = {"tools": _tools_list()}
                elif method == "tools/call":
                    result = _call_tool(params.get("name", ""), params.get("arguments") or {})
                else:
                    result = {}
                resp = {"jsonrpc": "2.0", "id": req_id, "result": result}
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(resp).encode("utf-8"))

            else:
                self.send_error(404)

    with HTTPServer(("127.0.0.1", port), Handler) as httpd:
        httpd.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sse", action="store_true")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.sse:
        run_sse(port=args.port)
    else:
        run_stdio()


if __name__ == "__main__":
    main()
