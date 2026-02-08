"""Tests for ReactAgent in examples/react_agent."""

import sys
import tempfile
import unittest
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

_examples_react = _root / "examples" / "react_agent"
if str(_examples_react) not in sys.path:
    sys.path.insert(0, str(_examples_react))

from pygent.agent import BaseAgent
from react_agent import ReactAgent


class TestReactAgent(unittest.IsolatedAsyncioTestCase):
    """Tests for ReactAgent with real API."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_react_agent_instantiation(self):
        """ReactAgent can be instantiated with tools registered."""
        agent = ReactAgent(root_dir=self.temp_dir)
        self.assertIsInstance(agent, BaseAgent)
        self.assertIsNotNone(agent.llm)
        self.assertIsNotNone(agent.tool_manager)
        tool = agent.tool_manager.get_tool("run_command")
        self.assertIsNotNone(tool)

    def test_tools_param_format(self):
        """_tools_param returns OpenAI-compatible format."""
        agent = ReactAgent(root_dir=self.temp_dir)
        tools = agent._tools_param()
        self.assertIsInstance(tools, list)
        self.assertGreater(len(tools), 0)
        for t in tools:
            self.assertEqual(t["type"], "function")
            self.assertIn("function", t)
            self.assertIn("name", t["function"])
            self.assertIn("parameters", t["function"])

    async def test_forward_simple_greeting(self):
        """forward() returns a response for a simple greeting (no tool use)."""
        agent = ReactAgent(root_dir=self.temp_dir)
        result = await agent.forward("你好")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result.strip()), 0)

    async def test_forward_with_tool_use(self):
        """forward() can use run_command tool when asked to list directory."""
        agent = ReactAgent(root_dir=self.temp_dir)
        result = await agent.forward(
            "请使用 run_command 工具执行 ls 命令，列出当前目录的文件，然后告诉我结果。"
        )
        self.assertIsInstance(result, str)
        self.assertGreater(len(result.strip()), 0)

    async def test_forward_pwd(self):
        """forward() can use run_command for pwd."""
        agent = ReactAgent(root_dir=self.temp_dir)
        result = await agent.forward(
            "请使用 run_command 执行 pwd 命令，告诉我当前工作目录。"
        )
        print(result)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result.strip()), 0)


if __name__ == "__main__":
    unittest.main()
