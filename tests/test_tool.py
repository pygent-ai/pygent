"""Tests for pygent.module.tool (base, tool_manager, utils)."""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import unittest

from pygent.module.tool import (
    BaseTool,
    ToolParameter,
    ToolMetadata,
    ToolCategory,
    ToolPermission,
    ToolManager,
    tool,
    auto_tool,
    ToolRegistry,
)


# --- Concrete tool for testing BaseTool ---
class EchoTool(BaseTool):
    """Echo tool that returns the input message."""

    def __init__(self):
        super().__init__(name="echo", description="Echo the input message", version="1.0.0")

    def forward(self, message: str = "") -> str:
        return message or "(empty)"


class AddTool(BaseTool):
    """Add two numbers."""

    def __init__(self):
        super().__init__(name="add", description="Add two numbers", version="1.0.0")

    def forward(self, a: float, b: float) -> float:
        return a + b


class TestToolParameter(unittest.TestCase):
    """Tests for ToolParameter."""

    def test_init_basic(self):
        p = ToolParameter(name="x", type=str, description="a string")
        self.assertEqual(p.data["name"], "x")
        self.assertEqual(p.data["type"], "string")
        self.assertEqual(p.data["description"], "a string")
        self.assertTrue(p.data["required"])

    def test_init_with_type_string(self):
        """ToolParameter accepts type as str (e.g. when rebuilt from schema)."""
        p = ToolParameter(name="y", type="integer", description="an int")
        self.assertEqual(p.data["type"], "integer")

    def test_init_optional(self):
        p = ToolParameter(name="opt", type=str, required=False, default="default")
        self.assertFalse(p.data["required"])
        self.assertEqual(p.data["default"], "default")

    def test_to_openai_schema(self):
        p = ToolParameter(name="count", type=int, description="count", required=True)
        schema = p.to_openai_schema()
        self.assertEqual(schema["type"], "integer")
        self.assertEqual(schema["description"], "count")

    def test_to_openai_schema_with_enum(self):
        p = ToolParameter(name="mode", type=str, enum=["a", "b"], description="mode")
        schema = p.to_openai_schema()
        self.assertEqual(schema["enum"], ["a", "b"])

    def test_to_openai_schema_with_min_max(self):
        p = ToolParameter(name="n", type=int, min_value=0, max_value=100)
        schema = p.to_openai_schema()
        self.assertEqual(schema["minimum"], 0)
        self.assertEqual(schema["maximum"], 100)


class TestToolMetadata(unittest.TestCase):
    """Tests for ToolMetadata."""

    def test_defaults(self):
        m = ToolMetadata(name="t", description="d")
        self.assertEqual(m.name, "t")
        self.assertEqual(m.description, "d")
        self.assertEqual(m.version, "1.0.0")
        self.assertEqual(m.category, ToolCategory.UTILITY)
        self.assertEqual(m.permission, ToolPermission.PUBLIC)


class TestBaseTool(unittest.TestCase):
    """Tests for BaseTool."""

    def test_forward_not_implemented(self):
        class NoForward(BaseTool):
            pass
        t = NoForward(name="n", description="d")
        with self.assertRaises(NotImplementedError):
            t.forward()

    def test_call_success(self):
        echo = EchoTool()
        out = echo(message="hello")
        self.assertTrue(out["success"])
        self.assertEqual(out["result"], "hello")
        self.assertEqual(echo.call_count.data, 1)

    def test_call_with_empty_message(self):
        echo = EchoTool()
        out = echo(message="")
        self.assertTrue(out["success"])
        self.assertEqual(out["result"], "(empty)")

    def test_call_disabled(self):
        echo = EchoTool()
        echo.disable()
        out = echo(message="x")
        self.assertFalse(out["success"])
        self.assertIn("禁用", out["error"])

    def test_call_validation_fail(self):
        add = AddTool()
        out = add(a=1)  # missing b
        self.assertFalse(out["success"])
        self.assertIn("参数", out["error"])

    def test_call_validation_pass(self):
        add = AddTool()
        out = add(a=1, b=2)
        self.assertTrue(out["success"])
        self.assertEqual(out["result"], 3)

    def test_validate_parameters_unknown(self):
        echo = EchoTool()
        errors = echo.validate_parameters({"message": "hi", "unknown_key": 1})
        self.assertIn("_unknown", errors)
        self.assertTrue(any("未知参数" in str(v) for v in errors["_unknown"]))

    def test_to_openai_function(self):
        echo = EchoTool()
        func = echo.to_openai_function()
        self.assertEqual(func["name"], "echo")
        self.assertEqual(func["description"], "Echo the input message")
        self.assertIn("parameters", func)
        self.assertIn("properties", func["parameters"])
        self.assertIn("message", func["parameters"]["properties"])

    def test_get_schema(self):
        echo = EchoTool()
        schema = echo.get_schema()
        self.assertIn("metadata", schema)
        self.assertIn("parameters", schema)
        self.assertIn("openai_function", schema)
        self.assertIn("status", schema)
        self.assertEqual(schema["metadata"]["name"], "echo")

    def test_enable_disable(self):
        echo = EchoTool()
        self.assertTrue(echo.enabled.data)
        echo.disable()
        self.assertFalse(echo.enabled.data)
        echo.enable()
        self.assertTrue(echo.enabled.data)

    def test_reset_stats(self):
        echo = EchoTool()
        echo(message="x")
        self.assertEqual(echo.call_count.data, 1)
        echo.reset_stats()
        self.assertEqual(echo.call_count.data, 0)
        self.assertEqual(echo.error_count.data, 0)

    def test_forward_exception(self):
        class BadTool(BaseTool):
            def forward(self, x: str) -> str:
                raise ValueError("bad")
        t = BadTool(name="bad", description="d")
        out = t(x="y")
        self.assertFalse(out["success"])
        self.assertIn("bad", out["error"])
        self.assertEqual(t.error_count.data, 1)


class TestToolManager(unittest.TestCase):
    """Tests for ToolManager."""

    def test_register_and_get(self):
        mgr = ToolManager()
        echo = EchoTool()
        mgr.register_tool(echo)
        self.assertIs(mgr.get_tool("echo"), echo)
        self.assertIsNone(mgr.get_tool("nonexistent"))

    def test_call_tool(self):
        mgr = ToolManager()
        mgr.register_tool(EchoTool())
        out = mgr.call_tool("echo", message="hi")
        self.assertTrue(out["success"])
        self.assertEqual(out["result"], "hi")

    def test_call_tool_not_found(self):
        mgr = ToolManager()
        out = mgr.call_tool("missing")
        self.assertFalse(out["success"])
        self.assertIn("未找到", out["error"])

    def test_get_all_schemas(self):
        mgr = ToolManager()
        mgr.register_tool(EchoTool())
        mgr.register_tool(AddTool())
        schemas = mgr.get_all_schemas()
        self.assertIn("tools", schemas)
        self.assertIn("echo", schemas["tools"])
        self.assertIn("add", schemas["tools"])
        self.assertIn("categories", schemas)
        # Category keys should be strings (e.g. "utility")
        for key in schemas["categories"].keys():
            self.assertIsInstance(key, str)

    def test_get_openai_functions(self):
        mgr = ToolManager()
        mgr.register_tool(EchoTool())
        funcs = mgr.get_openai_functions()
        self.assertEqual(len(funcs), 1)
        self.assertEqual(funcs[0]["name"], "echo")


class TestToolDecorator(unittest.TestCase):
    """Tests for @tool decorator from utils."""

    def test_tool_decorator_call(self):
        @tool(name="greet", description="Say hello")
        def greet(name: str) -> str:
            return f"Hello, {name}!"
        self.assertEqual(greet("World"), "Hello, World!")

    def test_tool_decorator_tool_instance(self):
        @tool(name="double", description="Double a number")
        def double(x: float) -> float:
            return x * 2
        self.assertTrue(hasattr(double, "tool"))
        self.assertEqual(double.tool.metadata.data["name"], "double")
        out = double.tool(x=5)
        self.assertTrue(out["success"])
        self.assertEqual(out["result"], 10)

    def test_tool_decorator_validation(self):
        @tool(name="strict_add", description="Add two ints")
        def strict_add(a: int, b: int) -> int:
            return a + b
        out = strict_add.tool(a=1, b=2)
        self.assertTrue(out["success"])
        self.assertEqual(out["result"], 3)
        out_bad = strict_add.tool(a=1)  # missing b
        self.assertFalse(out_bad["success"])


class TestAutoToolDecorator(unittest.TestCase):
    """Tests for @auto_tool decorator and tool schema correctness."""

    def test_auto_tool_basic_call(self):
        """auto_tool without docstring: call works and returns result."""
        @auto_tool()
        def no_doc(x: str) -> str:
            return x.upper()
        self.assertEqual(no_doc("hi"), "HI")
        out = no_doc.tool(x="hi")
        self.assertTrue(out["success"])
        self.assertEqual(out["result"], "HI")

    def test_auto_tool_has_tool_attr(self):
        @auto_tool(name="my_auto")
        def my_auto() -> str:
            return "ok"
        self.assertTrue(hasattr(my_auto, "tool"))
        self.assertEqual(my_auto.tool.metadata.data["name"], "my_auto")

    def test_auto_tool_schema_metadata(self):
        """Schema and OpenAI function have correct name and description."""
        @auto_tool(
            name="meta_tool",
            description="Tool for testing metadata in schema.",
        )
        def meta_tool() -> str:
            return "meta"
        schema = meta_tool.tool.get_schema()
        self.assertEqual(schema["metadata"]["name"], "meta_tool")
        self.assertEqual(schema["metadata"]["description"], "Tool for testing metadata in schema.")
        oai = meta_tool.tool.to_openai_function()
        self.assertEqual(oai["name"], "meta_tool")
        self.assertEqual(oai["description"], "Tool for testing metadata in schema.")

    def test_auto_tool_schema_parameters_structure(self):
        """Schema parameters and OpenAI function have correct structure."""
        @auto_tool(name="param_tool", description="Param tool.")
        def param_tool(a: int, b: str) -> str:
            return f"{a}-{b}"
        schema = param_tool.tool.get_schema()
        params = schema["parameters"]
        self.assertIn("a", params)
        self.assertIn("b", params)
        self.assertEqual(params["a"].get("type"), "integer")
        self.assertEqual(params["b"].get("type"), "string")
        oai = param_tool.tool.to_openai_function()
        self.assertEqual(oai["parameters"]["type"], "object")
        self.assertIn("properties", oai["parameters"])
        self.assertIn("required", oai["parameters"])
        self.assertIn("a", oai["parameters"]["properties"])
        self.assertIn("b", oai["parameters"]["properties"])
        self.assertIn("a", oai["parameters"]["required"])
        self.assertIn("b", oai["parameters"]["required"])

    def test_auto_tool_google_style_docstring(self):
        """auto_tool extracts param descriptions from Google-style docstring."""
        @auto_tool(name="google_tool", description="Tool with Google-style doc.")
        def google_tool(
            query: str,
            limit: int,
        ) -> str:
            """Tool with Google-style doc.
            Args:
                query: The search query string.
                limit: Maximum number of results, between 1 and 100.
            Returns:
                Result string.
            """
            return f"q={query}&n={limit}"
        schema = google_tool.tool.get_schema()
        params = schema["parameters"]
        self.assertEqual(params["query"].get("description"), "The search query string.")
        self.assertEqual(params["limit"].get("description"), "Maximum number of results, between 1 and 100.")
        oai = google_tool.tool.to_openai_function()
        self.assertEqual(oai["parameters"]["properties"]["query"]["description"], "The search query string.")
        self.assertEqual(oai["parameters"]["properties"]["limit"]["description"], "Maximum number of results, between 1 and 100.")

    def test_auto_tool_description_from_docstring(self):
        """When description not given, first line of docstring is used."""
        @auto_tool(name="desc_tool")
        def desc_tool() -> str:
            """Summary line used as tool description."""
            return "ok"
        self.assertEqual(desc_tool.tool.metadata.data["description"], "Summary line used as tool description.")

    def test_auto_tool_optional_param_schema(self):
        """Optional parameter has required=False and default in schema."""
        @auto_tool(name="opt_tool", description="Optional param.")
        def opt_tool(x: int, prefix: str = ">") -> str:
            return f"{prefix}{x}"
        schema = opt_tool.tool.get_schema()
        params = schema["parameters"]
        self.assertTrue(params["x"].get("required", True))
        self.assertFalse(params["prefix"].get("required", True))
        self.assertEqual(params["prefix"].get("default"), ">")
        oai = opt_tool.tool.to_openai_function()
        self.assertNotIn("prefix", oai["parameters"]["required"])
        # OpenAI schema may omit default; full schema has it
        self.assertIn("prefix", oai["parameters"]["properties"])

    def test_auto_tool_openai_function_property_fields(self):
        """Each OpenAI function property has type and description."""
        @auto_tool(name="shape_tool", description="Shape check.")
        def shape_tool(name: str, count: int) -> str:
            return f"{name}:{count}"
        oai = shape_tool.tool.to_openai_function()
        for prop_name, prop_schema in oai["parameters"]["properties"].items():
            self.assertIn("type", prop_schema, msg=f"property {prop_name} missing 'type'")
            self.assertIn("description", prop_schema, msg=f"property {prop_name} missing 'description'")

    def test_auto_tool_call_validation(self):
        """auto_tool instance validates args and executes."""
        @auto_tool(name="valid_tool", description="Validation.")
        def valid_tool(a: float, b: float) -> float:
            return a * b
        out = valid_tool.tool(a=3, b=4)
        self.assertTrue(out["success"])
        self.assertEqual(out["result"], 12)
        bad = valid_tool.tool(a=1)  # missing b
        self.assertFalse(bad["success"])

    def test_auto_tool_get_schema_full_structure(self):
        """get_schema() returns metadata, parameters, config, openai_function, status."""
        @auto_tool(name="full_schema", description="Full.")
        def full_schema() -> str:
            return "x"
        schema = full_schema.tool.get_schema()
        self.assertIn("metadata", schema)
        self.assertIn("parameters", schema)
        self.assertIn("config", schema)
        self.assertIn("openai_function", schema)
        self.assertIn("status", schema)
        self.assertIn("enabled", schema["status"])
        self.assertIn("call_count", schema["status"])
        self.assertIn("error_count", schema["status"])
        self.assertIn("last_called", schema["status"])


class TestToolRegistry(unittest.TestCase):
    """Tests for ToolRegistry."""

    def setUp(self):
        self.registry = ToolRegistry()
        self.registry.clear()

    def test_register_and_get(self):
        @tool(name="reg_echo", description="Echo for registry")
        def reg_echo(msg: str) -> str:
            return msg
        self.registry.register(reg_echo)
        t = self.registry.get("reg_echo")
        self.assertIsNotNone(t)
        self.assertEqual(t.metadata.data["name"], "reg_echo")

    def test_list_all(self):
        @tool(name="r1", description="R1")
        def r1() -> str:
            return "r1"
        self.registry.register(r1)
        names = self.registry.list_all()
        self.assertIn("r1", names)


if __name__ == "__main__":
    unittest.main()
