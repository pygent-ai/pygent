from __future__ import annotations

import asyncio
import os
import socket
import sys
from pathlib import Path

import httpx
import mcp
import pytest

from pygent import AIMessage, Context, thaw_json
from pygent.tool import (
    ExecutorRegistry,
    ToolAuthorizationDecision,
    ToolCall,
    ToolCallLayer,
)
from pygent.tool.mcp import MCPSseTransport, discover_mcp_tools, register_mcp_tools


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.asyncio
async def test_sse_discovery_and_execution_end_to_end() -> None:
    port = _free_port()
    server = Path(__file__).with_name("stdio_server.py")
    env = dict(os.environ)
    site_packages = str(Path(mcp.__file__).parent.parent)
    env["PYTHONPATH"] = os.pathsep.join((site_packages, *sys.path))
    virtual_env = Path.cwd() / ".venv"
    python = virtual_env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    creationflags = 0x08000000 if os.name == "nt" else 0
    process = await asyncio.create_subprocess_exec(
        str(python),
        str(server),
        "--sse",
        "--port",
        str(port),
        env=env,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        creationflags=creationflags,
    )
    try:
        url = f"http://127.0.0.1:{port}/sse"
        async with httpx.AsyncClient() as client:
            for _ in range(50):
                try:
                    response = await client.get(
                        f"http://127.0.0.1:{port}/health", timeout=0.1
                    )
                    if response.status_code in {200, 404}:
                        break
                except httpx.TransportError:
                    await asyncio.sleep(0.05)
            else:
                raise AssertionError("MCP SSE test server did not start")

        transport = MCPSseTransport(url, sse_read_timeout=5.0)
        specs = await discover_mcp_tools(transport, namespace="test", version="1")
        assert [item.definition.name for item in specs] == ["double"]

        registry = ExecutorRegistry()
        register_mcp_tools(registry, transport, specs)

        def authorize(request, context):
            return ToolAuthorizationDecision(
                call_id=request.call.call_id,
                allowed=True,
                reason_code="allowed",
            )

        layer = ToolCallLayer(
            tools=specs,
            authorization_adapter=authorize,
            executor_registry=registry,
        )
        message, _ = await layer.invoke(
            AIMessage(
                tool_calls=(
                    ToolCall(call_id="mcp-sse-1", name="double", arguments={"value": 4}),
                )
            ),
            Context(tools=tuple(item.definition for item in specs)),
        )
        assert message.results[0].status == "succeeded", message.results[0].error
        assert thaw_json(message.results[0].output) == {"result": 8}
    finally:
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except TimeoutError:
            process.kill()
            await process.wait()
