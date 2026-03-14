"""
MCP client for stdio transport using only Python standard library.
Spawns server process, communicates via newline-delimited JSON-RPC on stdin/stdout.
"""

import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pygent.common import PygentString, PygentDict, PygentList

from .base import BaseMCPClient


def _safe_errlog() -> tuple:
    """Return (errlog, should_close) for subprocess stderr on Windows."""
    try:
        if hasattr(sys.stderr, "fileno") and callable(getattr(sys.stderr, "fileno")):
            sys.stderr.fileno()
            return sys.stderr, False
    except (ValueError, OSError, io.UnsupportedOperation):
        pass
    return open(os.devnull, "w", encoding="utf-8"), True


class StdioMCPClient(BaseMCPClient):
    """MCP client for stdio: spawns server process, JSON-RPC over stdin/stdout."""

    command: PygentString
    args: PygentList
    env: PygentDict

    def __init__(
        self,
        server_id: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ):
        super().__init__(server_id, **kwargs)
        self.command = PygentString(command)
        self.args = PygentList(list(args) if args else [])
        self.env = PygentDict(dict(env) if env else {})
        self._cwd = str(cwd) if cwd else None

    def _request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run one JSON-RPC session: spawn, initialize, send request, return result."""
        cmd = [self.command.data] + list(self.args.data)
        env = os.environ.copy()
        env.update(self.env.data)
        errlog, close_err = _safe_errlog()
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=errlog,
                cwd=self._cwd,
                env=env,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        finally:
            if close_err:
                errlog.close()

        request_id = 1
        req_init = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "pygent", "version": "0.1.0"},
            },
        }
        proc.stdin.write(json.dumps(req_init) + "\n")
        proc.stdin.flush()

        # Read until we get initialize result
        _read_until_response(proc, request_id)

        # Send notifications/initialized if needed (some servers expect it)
        proc.stdin.write(json.dumps({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }) + "\n")
        proc.stdin.flush()

        # Send tools/list or tools/call
        request_id = 2
        req = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        }
        proc.stdin.write(json.dumps(req) + "\n")
        proc.stdin.flush()

        result = _read_until_response(proc, request_id)
        proc.stdin.close()
        proc.stdout.close()
        proc.terminate()
        proc.wait(timeout=2)
        return result


def _read_until_response(proc: subprocess.Popen, expect_id: Any) -> Dict[str, Any]:
    """Read newline-delimited JSON from proc.stdout until we get response with id=expect_id."""
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("MCP server closed connection")
        line = line.rstrip("\n")
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "id" in msg and msg["id"] == expect_id:
            if "error" in msg:
                raise RuntimeError(f"MCP error: {msg['error']}")
            return msg.get("result", {})
        # Skip notifications (no id) and other responses
