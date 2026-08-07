import os
import sys
from pathlib import Path

import mcp
import pytest

from pygent import AIMessage, Context
from pygent.tool import (
    ExecutorRegistry,
    ToolAuthorizationDecision,
    ToolCall,
    ToolCallLayer,
)
from pygent.tool.mcp import MCPStdioTransport, discover_mcp_tools, register_mcp_tools


@pytest.mark.asyncio
async def test_stdio_discovery_and_execution_end_to_end() -> None:
    server = Path(__file__).with_name("stdio_server.py")
    env = dict(os.environ)
    site_packages = str(Path(mcp.__file__).parent.parent)
    env["PYTHONPATH"] = os.pathsep.join((site_packages, *sys.path))
    virtual_env = Path.cwd() / ".venv"
    python = virtual_env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    transport = MCPStdioTransport(
        str(python),
        args=(str(server),),
        env=env,
    )
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
                ToolCall(call_id="mcp-1", name="double", arguments={"value": 3}),
            )
        ),
        Context(tools=tuple(item.definition for item in specs)),
    )
    assert message.results[0].status == "succeeded", message.results[0].error
    assert message.results[0].output["result"] == 6
