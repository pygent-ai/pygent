"""Tests for MCP stdio client using LocalMemoryMCP server."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import unittest

from pygent.module.tool.mcp import StdioMCPClient

# LocalMemoryMCP lives at MCPs/LocalMemoryMCP; server must be run with cwd there so imports work
LOCAL_MCP_DIR = _root / "MCPs" / "LocalMemoryMCP"


@unittest.skipUnless(LOCAL_MCP_DIR.is_dir(), "LocalMemoryMCP not found at MCPs/LocalMemoryMCP")
class TestStdioMCPClientLocalMemory(unittest.TestCase):
    """Test StdioMCPClient against the LocalMemoryMCP (DeepMemory) server."""

    def setUp(self):
        self.client = StdioMCPClient(
            server_id="local_memory",
            command="python",
            args=["main.py"],
            cwd=LOCAL_MCP_DIR,
        )

    def test_list_tools(self):
        tools = self.client.list_tools()
        self.assertIsInstance(tools, list)
        self.assertGreater(len(tools), 0)
        names = [getattr(t, "name", None) for t in tools]
        self.assertIn("get_memory_statistics", names)
        self.assertIn("list_memory_files", names)
        self.assertIn("write_raw_conversation", names)

    def test_call_tool_get_memory_statistics(self):
        result = self.client.call_tool("get_memory_statistics")
        self.assertFalse(getattr(result, "isError", True))
        content = getattr(result, "content", [])
        self.assertIsInstance(content, list)
        # Often one text block
        if content:
            first = content[0]
            text = getattr(first, "text", None) or (first.get("text") if isinstance(first, dict) else None)
            self.assertIsNotNone(text)
            self.assertIsInstance(text, str)

    def test_call_tool_list_memory_files(self):
        result = self.client.call_tool("list_memory_files", arguments={})
        self.assertFalse(getattr(result, "isError", True))
        content = getattr(result, "content", [])
        self.assertIsInstance(content, list)


if __name__ == "__main__":
    unittest.main()
