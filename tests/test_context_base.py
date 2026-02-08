"""Tests for pygent.context.base module (BaseContext save/load)."""
import os
import tempfile
import unittest
import unittest.mock
import sys
from pathlib import Path

# Ensure project root is on path when running tests directly
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from pygent.context import BaseContext
from pygent.message import BaseMessage, SystemMessage, UserMessage, AssistantMessage


class TestBaseContextSaveLoad(unittest.TestCase):
    """Test cases for BaseContext save and load methods."""

    def test_save_load_json_empty_context(self):
        """Empty BaseContext (no system prompt, no messages) round-trips with JSON."""
        ctx = BaseContext()
        self.assertEqual(len(ctx.history.data), 0)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ctx.json")
            with unittest.mock.patch("builtins.print"):
                ctx.save(path, format="json", include_metadata=True)
            self.assertTrue(os.path.exists(path))

            ctx2 = BaseContext()
            with unittest.mock.patch("builtins.print"):
                ctx2.load(path, format="json", strict=True)
            self.assertEqual(len(ctx2.history.data), 0)

    def test_save_load_json_empty_context_no_metadata(self):
        """Empty BaseContext save/load JSON without metadata."""
        ctx = BaseContext()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ctx.json")
            with unittest.mock.patch("builtins.print"):
                ctx.save(path, format="json", include_metadata=False)
            ctx2 = BaseContext()
            with unittest.mock.patch("builtins.print"):
                ctx2.load(path, format="json", strict=True)
            self.assertEqual(len(ctx2.history.data), 0)

    def test_save_load_pickle_with_system_prompt(self):
        """BaseContext with system_prompt round-trips with pickle (message objects preserved)."""
        ctx = BaseContext(system_prompt="You are a helpful assistant.")
        self.assertEqual(len(ctx.history.data), 1)
        self.assertEqual(ctx.history.data[0].role.data, "system")
        self.assertEqual(ctx.history.data[0].content.data, "You are a helpful assistant.")

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ctx.pkl")
            with unittest.mock.patch("builtins.print"):
                ctx.save(path, format="pickle", include_metadata=True)
            self.assertTrue(os.path.exists(path))

            ctx2 = BaseContext()
            with unittest.mock.patch("builtins.print"):
                ctx2.load(path, format="pickle", strict=True)
            self.assertEqual(len(ctx2.history.data), 1)
            self.assertIsInstance(ctx2.history.data[0], BaseMessage)
            self.assertEqual(ctx2.history.data[0].role.data, "system")
            self.assertEqual(ctx2.history.data[0].content.data, "You are a helpful assistant.")

    def test_save_load_pickle_with_multiple_messages(self):
        """BaseContext with multiple messages round-trips with pickle."""
        ctx = BaseContext(system_prompt="System")
        ctx.add_message(UserMessage("Hello"))
        ctx.add_message(AssistantMessage("Hi there!"))
        ctx.add_message(UserMessage("Bye"))
        self.assertEqual(len(ctx.history.data), 4)  # system + 3

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ctx.pkl")
            with unittest.mock.patch("builtins.print"):
                ctx.save(path, format="pickle", include_metadata=True)
            ctx2 = BaseContext()
            with unittest.mock.patch("builtins.print"):
                ctx2.load(path, format="pickle", strict=True)
            self.assertEqual(len(ctx2.history.data), 4)
            self.assertEqual(ctx2.history.data[0].role.data, "system")
            self.assertEqual(ctx2.history.data[0].content.data, "System")
            self.assertEqual(ctx2.history.data[1].role.data, "user")
            self.assertEqual(ctx2.history.data[1].content.data, "Hello")
            self.assertEqual(ctx2.history.data[2].role.data, "assistant")
            self.assertEqual(ctx2.history.data[2].content.data, "Hi there!")
            self.assertEqual(ctx2.history.data[3].role.data, "user")
            self.assertEqual(ctx2.history.data[3].content.data, "Bye")

    def test_save_load_pickle_no_metadata(self):
        """BaseContext save/load pickle without metadata."""
        ctx = BaseContext(system_prompt="Minimal")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ctx.pkl")
            with unittest.mock.patch("builtins.print"):
                ctx.save(path, format="pickle", include_metadata=False)
            ctx2 = BaseContext()
            with unittest.mock.patch("builtins.print"):
                ctx2.load(path, format="pickle", strict=True)
            self.assertEqual(len(ctx2.history.data), 1)
            self.assertEqual(ctx2.history.data[0].content.data, "Minimal")

    def test_save_returns_absolute_path(self):
        """BaseContext.save() returns absolute path."""
        ctx = BaseContext()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ctx.json")
            with unittest.mock.patch("builtins.print"):
                result = ctx.save(path, format="json")
            self.assertIsInstance(result, str)
            self.assertEqual(result, str(Path(path).resolve()))
            self.assertTrue(Path(result).is_absolute())

    def test_save_creates_parent_directories(self):
        """BaseContext.save() creates parent directories."""
        ctx = BaseContext()
        with tempfile.TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, "subdir", "deep", "ctx.json")
            with unittest.mock.patch("builtins.print"):
                ctx.save(nested, format="json")
            self.assertTrue(os.path.isfile(nested))

    def test_load_file_not_found(self):
        """BaseContext.load() raises FileNotFoundError when file does not exist."""
        ctx = BaseContext()
        with self.assertRaises(FileNotFoundError):
            ctx.load("/nonexistent/path/ctx.json")

    def test_load_format_auto_pickle(self):
        """BaseContext load with format='auto' detects pickle from extension."""
        ctx = BaseContext(system_prompt="Auto")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ctx.pickle")
            with unittest.mock.patch("builtins.print"):
                ctx.save(path, format="pickle")
            ctx2 = BaseContext()
            with unittest.mock.patch("builtins.print"):
                ctx2.load(path, format="auto", strict=True)
            self.assertEqual(len(ctx2.history.data), 1)
            self.assertEqual(ctx2.history.data[0].content.data, "Auto")

    def test_load_format_auto_json(self):
        """BaseContext load with format='auto' detects JSON from extension."""
        ctx = BaseContext()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ctx.json")
            with unittest.mock.patch("builtins.print"):
                ctx.save(path, format="json")
            ctx2 = BaseContext()
            with unittest.mock.patch("builtins.print"):
                ctx2.load(path, format="auto", strict=True)
            self.assertEqual(len(ctx2.history.data), 0)


# Fix: BaseMessage is used in test_save_load_pickle_with_system_prompt
from pygent.message import BaseMessage  # noqa: E402

if __name__ == "__main__":
    unittest.main()
