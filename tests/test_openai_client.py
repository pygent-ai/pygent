"""Tests for pygent.llm.openai_client module."""
import sys
import unittest
import unittest.mock
from pathlib import Path

# Ensure project root is on path when running tests directly
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from pygent.llm import AsyncOpenAIClient
from pygent.context import BaseContext
from pygent.message import UserMessage, AssistantMessage


class TestAsyncOpenAIClient(unittest.IsolatedAsyncioTestCase):
    """Tests for AsyncOpenAIClient."""

    def setUp(self):
        self.client = AsyncOpenAIClient(
            base_url="https://api.deepseek.com",
            api_key="API_KEY",
            model_name="deepseek-chat",
        )

    async def test_llm_client(self):
        context = BaseContext()
        context.add_message(UserMessage("你好"))
        context = await self.client.forward(context=context)

        print(context.history[-1].content)



if __name__ == "__main__":
    unittest.main()
