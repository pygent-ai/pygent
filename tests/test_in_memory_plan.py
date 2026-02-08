"""Tests for pygent.module.plan.in_memory_plan: tool callability and tool schema."""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import unittest

from pygent.module.plan import (
    InMemoryPlan,
    PygentStatus,
    InMemoryTodoItem,
)


class TestInMemoryPlanToolsCallable(unittest.TestCase):
    """Test that each plan tool is callable and returns correct results."""

    def setUp(self):
        self.plan = InMemoryPlan()
        self.tools = {t.metadata.data["name"]: t for t in self.plan.get_tools()}

    def test_create_todo_list_callable(self):
        create = self.tools["create_todo_list"]
        out = create(todo_list=["task A", "task B", "task C"])
        self.assertTrue(out["success"], msg=out.get("error"))
        self.assertIn("Created todo list with 3 item(s).", out["result"])
        self.assertEqual(len(self.plan.todo_list), 3)
        self.assertEqual(self.plan.todo_list[0]["content"], "task A")
        self.assertEqual(self.plan.todo_list[0]["status"], PygentStatus.PENDING)

    def test_create_todo_list_replace_existing(self):
        create = self.tools["create_todo_list"]
        create(todo_list=["old"])
        create(todo_list=["new1", "new2"])
        self.assertEqual(len(self.plan.todo_list), 2)
        self.assertEqual(self.plan.todo_list[0]["content"], "new1")

    def test_mark_current_todo_item_callable(self):
        create = self.tools["create_todo_list"]
        mark = self.tools["mark_current_todo_item"]
        create(todo_list=["first", "second"])
        out = mark()
        self.assertTrue(out["success"], msg=out.get("error"))
        self.assertIn("Marked item 0 as completed.", out["result"])
        self.assertEqual(self.plan.todo_list[0]["status"], PygentStatus.SUCCESS)

    def test_mark_current_todo_item_no_pending(self):
        mark = self.tools["mark_current_todo_item"]
        out = mark()
        self.assertTrue(out["success"])
        self.assertIn("No PENDING or RUNNING", out["result"])

    def test_insert_todo_list_append_callable(self):
        create = self.tools["create_todo_list"]
        insert = self.tools["insert_todo_list"]
        create(todo_list=["a"])
        out = insert(todo_list=["b", "c"], index=-1)
        self.assertTrue(out["success"], msg=out.get("error"))
        self.assertIn("Inserted 2 item(s).", out["result"])
        self.assertEqual(len(self.plan.todo_list), 3)
        self.assertEqual(self.plan.todo_list[1]["content"], "b")

    def test_insert_todo_list_at_index_callable(self):
        create = self.tools["create_todo_list"]
        insert = self.tools["insert_todo_list"]
        create(todo_list=["a", "c"])
        out = insert(todo_list=["b"], index=1)
        self.assertTrue(out["success"], msg=out.get("error"))
        self.assertEqual([self.plan.todo_list[i]["content"] for i in range(3)], ["a", "b", "c"])

    def test_remove_todo_items_callable(self):
        create = self.tools["create_todo_list"]
        remove = self.tools["remove_todo_items"]
        create(todo_list=["a", "b", "c"])
        out = remove(indices=[0, 2])
        self.assertTrue(out["success"], msg=out.get("error"))
        self.assertIn("Removed 2 item(s).", out["result"])
        self.assertEqual(len(self.plan.todo_list), 1)
        self.assertEqual(self.plan.todo_list[0]["content"], "b")

    def test_remove_todo_items_empty_indices(self):
        remove = self.tools["remove_todo_items"]
        out = remove(indices=[])
        self.assertTrue(out["success"])
        self.assertIn("No indices given.", out["result"])

    def test_full_workflow_via_tools(self):
        create = self.tools["create_todo_list"]
        mark = self.tools["mark_current_todo_item"]
        insert = self.tools["insert_todo_list"]
        remove = self.tools["remove_todo_items"]
        create(todo_list=["Step 1", "Step 2", "Step 3"])
        self.assertEqual(len(self.plan.todo_list), 3)
        mark()
        self.assertEqual(self.plan.todo_list[0]["status"], PygentStatus.SUCCESS)
        insert(todo_list=["Step 1.5"], index=1)
        self.assertEqual(len(self.plan.todo_list), 4)
        remove(indices=[3])
        self.assertEqual(len(self.plan.todo_list), 3)
        self.assertEqual(self.plan.todo_list[1]["content"], "Step 1.5")


class TestInMemoryPlanToolSchemaReachable(unittest.TestCase):
    """Test that tool schema is reachable (get_schema, to_openai_function)."""

    def setUp(self):
        self.plan = InMemoryPlan()
        self.tools = {t.metadata.data["name"]: t for t in self.plan.get_tools()}

    def test_each_tool_has_get_schema(self):
        for name, tool in self.tools.items():
            with self.subTest(tool=name):
                schema = tool.get_schema()
                self.assertIsInstance(schema, dict)
                self.assertIn("metadata", schema)
                self.assertIn("parameters", schema)
                self.assertIn("openai_function", schema)
                self.assertIn("status", schema)

    def test_each_tool_has_to_openai_function(self):
        for name, tool in self.tools.items():
            with self.subTest(tool=name):
                oai = tool.to_openai_function()
                self.assertIsInstance(oai, dict)
                self.assertIn("name", oai)
                self.assertIn("description", oai)
                self.assertIn("parameters", oai)
                self.assertEqual(oai["name"], name)


class TestInMemoryPlanToolSchemaCorrect(unittest.TestCase):
    """Test that tool schema content is correct (names, descriptions, parameters)."""

    def setUp(self):
        self.plan = InMemoryPlan()
        self.tools = {t.metadata.data["name"]: t for t in self.plan.get_tools()}

    def test_create_todo_list_schema(self):
        tool = self.tools["create_todo_list"]
        schema = tool.get_schema()
        self.assertEqual(schema["metadata"]["name"], "create_todo_list")
        self.assertIn("todo list", schema["metadata"]["description"].lower())
        self.assertIn("todo_list", schema["parameters"])
        param = schema["parameters"]["todo_list"]
        self.assertEqual(param.get("type"), "array")
        oai = tool.to_openai_function()
        self.assertEqual(oai["name"], "create_todo_list")
        self.assertIn("todo_list", oai["parameters"]["properties"])
        self.assertIn("todo_list", oai["parameters"]["required"])

    def test_mark_current_todo_item_schema(self):
        tool = self.tools["mark_current_todo_item"]
        schema = tool.get_schema()
        self.assertEqual(schema["metadata"]["name"], "mark_current_todo_item")
        self.assertIn("current", schema["metadata"]["description"].lower() or schema["metadata"]["description"])
        oai = tool.to_openai_function()
        self.assertEqual(oai["name"], "mark_current_todo_item")
        self.assertEqual(oai["parameters"].get("properties"), {})
        self.assertEqual(oai["parameters"].get("required"), [])

    def test_insert_todo_list_schema(self):
        tool = self.tools["insert_todo_list"]
        schema = tool.get_schema()
        self.assertEqual(schema["metadata"]["name"], "insert_todo_list")
        self.assertIn("todo_list", schema["parameters"])
        self.assertIn("index", schema["parameters"])
        self.assertEqual(schema["parameters"]["todo_list"].get("type"), "array")
        self.assertEqual(schema["parameters"]["index"].get("type"), "integer")
        oai = tool.to_openai_function()
        self.assertIn("todo_list", oai["parameters"]["properties"])
        self.assertIn("index", oai["parameters"]["properties"])
        self.assertIn("todo_list", oai["parameters"]["required"])
        self.assertNotIn("index", oai["parameters"]["required"])

    def test_remove_todo_items_schema(self):
        tool = self.tools["remove_todo_items"]
        schema = tool.get_schema()
        self.assertEqual(schema["metadata"]["name"], "remove_todo_items")
        self.assertIn("indices", schema["parameters"])
        self.assertEqual(schema["parameters"]["indices"].get("type"), "array")
        oai = tool.to_openai_function()
        self.assertEqual(oai["name"], "remove_todo_items")
        self.assertIn("indices", oai["parameters"]["properties"])
        self.assertIn("indices", oai["parameters"]["required"])

    def test_openai_function_properties_have_type_and_description(self):
        for name, tool in self.tools.items():
            oai = tool.to_openai_function()
            for prop_name, prop_schema in oai["parameters"].get("properties", {}).items():
                with self.subTest(tool=name, param=prop_name):
                    self.assertIn("type", prop_schema, msg=f"{name}.{prop_name} missing 'type'")
                    self.assertIn("description", prop_schema, msg=f"{name}.{prop_name} missing 'description'")


class TestInMemoryTodoItem(unittest.TestCase):
    """Test InMemoryTodoItem and PygentStatus."""

    def test_todo_item_to_dict_from_dict(self):
        item = InMemoryTodoItem(content="test task", status=PygentStatus.PENDING)
        d = item.to_dict()
        self.assertEqual(d["content"], "test task")
        self.assertEqual(d["status"], PygentStatus.PENDING)
        item2 = InMemoryTodoItem.from_dict(d)
        self.assertEqual(item2.content.data, "test task")
        self.assertEqual(item2.status.data, PygentStatus.PENDING)


if __name__ == "__main__":
    unittest.main()
