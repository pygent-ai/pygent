"""Tests for ToolManager with RestrictedTerminal (run_command tool)."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import tempfile
import unittest

from pygent.module.tool import ToolManager, BaseTool
from pygent.toolkits import RestrictedTerminal


class TestToolManagerTerminal(unittest.TestCase):
    """Use ToolManager to add RestrictedTerminal and call run_command."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_register_tools_from_terminal(self):
        """register_tools(terminal.get_tools()) registers run_command."""
        manager = ToolManager()
        terminal = RestrictedTerminal(root_dir=self.temp_dir)
        manager.add_module("terminal", terminal)
        manager.register_tools(terminal.get_tools())

        tool = manager.get_tool("run_command")
        self.assertIsNotNone(tool)
        self.assertIsInstance(tool, BaseTool)
        self.assertEqual(tool.metadata.data["name"], "run_command")

    def test_call_run_command_pwd(self):
        """call_tool('run_command', cmd='pwd') returns current directory output."""
        manager = ToolManager()
        terminal = RestrictedTerminal(root_dir=self.temp_dir)
        manager.add_module("terminal", terminal)
        manager.register_tools(terminal.get_tools())

        result = manager.call_tool("run_command", cmd="pwd")
        self.assertTrue(result["success"], result.get("error", result))
        self.assertIn("result", result)
        # pwd prints current dir (e.g. "当前目录: /." or similar)
        self.assertIsInstance(result["result"], str)
        self.assertGreater(len(result["result"].strip()), 0)

    def test_call_run_command_ls(self):
        """call_tool('run_command', cmd='ls') returns listing."""
        manager = ToolManager()
        terminal = RestrictedTerminal(root_dir=self.temp_dir)
        manager.add_module("terminal", terminal)
        manager.register_tools(terminal.get_tools())

        result = manager.call_tool("run_command", cmd="ls")
        self.assertTrue(result["success"], result.get("error", result))
        self.assertIn("result", result)
        self.assertIsInstance(result["result"], str)

    def test_call_run_command_mkdir_and_ls(self):
        """run_command supports mkdir; subsequent ls sees the directory."""
        manager = ToolManager()
        terminal = RestrictedTerminal(root_dir=self.temp_dir)
        manager.add_module("terminal", terminal)
        manager.register_tools(terminal.get_tools())

        r1 = manager.call_tool("run_command", cmd="mkdir testdir")
        self.assertTrue(r1["success"], r1.get("error", r1))

        r2 = manager.call_tool("run_command", cmd="ls")
        self.assertTrue(r2["success"], r2.get("error", r2))
        self.assertIn("testdir", r2["result"])

    def test_run_command_in_schemas_and_openai_functions(self):
        """run_command appears in get_all_schemas and get_openai_functions."""
        manager = ToolManager()
        terminal = RestrictedTerminal(root_dir=self.temp_dir)
        manager.add_module("terminal", terminal)
        manager.register_tools(terminal.get_tools())

        schemas = manager.get_all_schemas()
        self.assertIn("run_command", schemas["tools"])
        run_cmd_schema = schemas["tools"]["run_command"]
        self.assertIn("parameters", run_cmd_schema)
        self.assertEqual(run_cmd_schema.get("metadata", {}).get("name"), "run_command")

        funcs = manager.get_openai_functions()
        names = [f["name"] for f in funcs]
        self.assertIn("run_command", names)
        run_cmd_func = next(f for f in funcs if f["name"] == "run_command")
        self.assertIn("parameters", run_cmd_func)
        self.assertIn("cmd", run_cmd_func["parameters"].get("properties", {}))

    def test_call_run_command_missing_cmd_fails(self):
        """call_tool without cmd fails validation."""
        manager = ToolManager()
        terminal = RestrictedTerminal(root_dir=self.temp_dir)
        manager.register_tools(terminal.get_tools())

        result = manager.call_tool("run_command")
        self.assertFalse(result["success"])
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
