"""Provider message adapters preserve legacy message formatting."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from pygent.context import BaseContext
from pygent.message import AssistantMessage, ToolCall, UserMessage
from pygent.message.adapters import OllamaMessageAdapter, OpenAIMessageAdapter


def test_openai_adapter_matches_legacy_method():
    msg = UserMessage(content="hello")
    assert OpenAIMessageAdapter.to_message_dict(msg) == msg.to_openai_format()


def test_ollama_adapter_converts_tool_arguments_to_dict():
    msg = AssistantMessage(
        content="",
        tool_calls=[ToolCall(tool_call_id="call_1", tool_name="search", arguments={"query": "x"})],
    )

    payload = OllamaMessageAdapter.to_message_dict(msg)

    assert payload["tool_calls"][0]["function"]["arguments"] == {"query": "x"}
    assert payload == msg.to_ollama_format()


def test_adapter_messages_from_context():
    context = BaseContext()
    context.add_message(UserMessage(content="hello"))

    assert OpenAIMessageAdapter.messages_from_context(context) == [{"role": "user", "content": "hello"}]
