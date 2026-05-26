import json

import pytest

from pygent.context import BaseContext
from pygent.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from pygent.session.base import Session, _session_dir, _session_file_path


def test_session_paths_are_workspace_scoped(tmp_path):
    assert _session_dir(str(tmp_path), "abc") == tmp_path / "sessions" / "abc"
    assert _session_file_path(str(tmp_path), "abc") == tmp_path / "sessions" / "abc" / "session.json"


def test_session_save_creates_json_payload_and_directories(tmp_path):
    session = Session(
        session_id="release-check",
        workspace_root=str(tmp_path),
        system_prompt="Be precise",
        metadata={"user": "alice"},
    )
    session.context.add_message(UserMessage("hello"))

    saved_path = session.save()

    path = tmp_path / "sessions" / "release-check" / "session.json"
    assert saved_path == str(path.absolute())
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == "1.0"
    assert payload["session_id"] == "release-check"
    assert payload["workspace_root"] == str(tmp_path.absolute())
    assert payload["system_prompt"] == "Be precise"
    assert payload["metadata"] == {"user": "alice"}
    assert [m["role"] for m in payload["history"]] == ["system", "user"]
    assert payload["updated_at"] >= payload["created_at"]


def test_session_load_restores_message_subclasses_and_tool_calls(tmp_path):
    context = BaseContext(system_prompt="System")
    context.add_message(UserMessage("question", name="ran"))
    context.add_message(
        AssistantMessage(
            "answer",
            tool_calls=[ToolCall(tool_call_id="call_1", tool_name="lookup", arguments={"q": "pygent"})],
        )
    )
    context.add_message(ToolMessage("result", tool_call_id="call_1"))
    session = Session("restore", str(tmp_path), metadata={"env": "test"}, context=context)
    session.save()

    restored = Session.load(str(tmp_path), "restore")

    assert restored.session_id == "restore"
    assert restored.metadata == {"env": "test"}
    assert restored.context.system_prompt.data == "System"
    roles = [msg.role.data for msg in restored.context.history.data]
    assert roles == ["system", "user", "assistant", "tool"]
    assistant = restored.context.history.data[2]
    assert assistant.tool_calls.data[0].tool_name.data == "lookup"
    assert assistant.tool_calls.data[0].arguments.data == {"q": "pygent"}
    tool = restored.context.history.data[3]
    assert tool.tool_call_id.data == "call_1"


def test_session_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Session.load(str(tmp_path), "missing")


def test_session_save_rejects_unsupported_format(tmp_path):
    session = Session("fmt", str(tmp_path))
    with pytest.raises(ValueError, match="Unsupported format"):
        session.save(format="yaml")
