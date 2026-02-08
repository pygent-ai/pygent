"""Tests for ToolManager MCP integration (add_mcp_server_stdio, add_mcp_server_sse)."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import unittest

from pygent.module.tool import ToolManager, BaseTool

LOCAL_MCP_DIR = _root / "MCPs" / "LocalMemoryMCP"


@unittest.skipUnless(LOCAL_MCP_DIR.is_dir(), "LocalMemoryMCP not found at MCPs/LocalMemoryMCP")
class TestToolManagerMCPStdio(unittest.TestCase):
    """Test ToolManager.add_mcp_server_stdio with LocalMemoryMCP."""

    def test_add_mcp_server_stdio_registers_tools(self):
        manager = ToolManager()
        tools = manager.add_mcp_server_stdio(
            server_id="local_memory",
            command="python",
            args=["main.py"],
            cwd=str(LOCAL_MCP_DIR),
        )
        self.assertIsInstance(tools, list)
        self.assertGreater(len(tools), 0)
        for t in tools:
            self.assertIsInstance(t, BaseTool)
        self.assertIn("local_memory", manager.mcp_clients.data)

    def test_add_mcp_server_stdio_tool_call(self):
        manager = ToolManager()
        manager.add_mcp_server_stdio(
            server_id="local_memory",
            command="python",
            args=["main.py"],
            cwd=str(LOCAL_MCP_DIR),
        )
        tool = manager.get_tool("get_memory_statistics")
        self.assertIsNotNone(tool)
        result = manager.call_tool("get_memory_statistics")
        self.assertTrue(result.get("success"), result.get("error", result))
        self.assertIn("result", result)

    def test_add_mcp_server_stdio_with_prefix(self):
        manager = ToolManager()
        tools = manager.add_mcp_server_stdio(
            server_id="mem",
            command="python",
            args=["main.py"],
            cwd=str(LOCAL_MCP_DIR),
            tool_name_prefix="mem",
        )
        self.assertGreater(len(tools), 0)
        # Registered names should be prefixed
        prefixed = manager.get_tool("mem_get_memory_statistics")
        self.assertIsNotNone(prefixed)
        result = manager.call_tool("mem_get_memory_statistics")
        self.assertTrue(result.get("success"), result.get("error", result))


if __name__ == "__main__":
    unittest.main()
