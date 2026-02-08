"""Tests for MCP SSE client using LocalMemoryMCP server (SSE transport)."""

import sys
import time
import subprocess
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import unittest

from pygent.module.tool.mcp import SSEMCPClient

LOCAL_MCP_DIR = _root / "MCPs" / "LocalMemoryMCP"
SSE_URL = "http://127.0.0.1:8765/sse"


def _start_sse_server():
    """Start LocalMemoryMCP with SSE in a subprocess. Returns the process."""
    proc = subprocess.Popen(
        [sys.executable, "run_sse.py"],
        cwd=LOCAL_MCP_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


def _wait_for_sse_port(port: int = 8765, max_wait: float = 10.0):
    """Wait until the SSE server port is open."""
    import socket
    start = time.monotonic()
    while time.monotonic() - start < max_wait:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.3)
    return False


@unittest.skipUnless(LOCAL_MCP_DIR.is_dir(), "LocalMemoryMCP not found at MCPs/LocalMemoryMCP")
class TestSSEMCPClientLocalMemory(unittest.TestCase):
    """Test SSEMCPClient against LocalMemoryMCP run with SSE (run_sse.py)."""

    _process = None

    @classmethod
    def setUpClass(cls):
        if not (LOCAL_MCP_DIR / "run_sse.py").exists():
            raise unittest.SkipTest("run_sse.py not found in LocalMemoryMCP")
        cls._process = _start_sse_server()
        if not _wait_for_sse_port():
            if cls._process:
                cls._process.terminate()
            raise unittest.SkipTest("SSE server did not become ready in time")

    @classmethod
    def tearDownClass(cls):
        if cls._process is not None and cls._process.poll() is None:
            cls._process.terminate()
            cls._process.wait(timeout=5)

    def setUp(self):
        self.client = SSEMCPClient(
            server_id="local_memory_sse",
            url=SSE_URL,
        )

    def test_list_tools(self):
        tools = self.client.list_tools()
        self.assertIsInstance(tools, list)
        self.assertGreater(len(tools), 0)
        names = [getattr(t, "name", None) for t in tools]
        self.assertIn("get_memory_statistics", names)

    def test_call_tool_get_memory_statistics(self):
        result = self.client.call_tool("get_memory_statistics")
        self.assertFalse(getattr(result, "isError", True))
        content = getattr(result, "content", [])
        self.assertIsInstance(content, list)


if __name__ == "__main__":
    unittest.main()
