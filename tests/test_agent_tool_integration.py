import asyncio
import json
import sys
from pathlib import Path

import pytest

from pygent.agent import BaseAgent
from pygent.context import BaseContext
from pygent.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from pygent.module.tool import ToolManager
from pygent.toolkits import BashToolkits, FileToolkits, WebFetchToolkits, WebSearchToolkits


DEFAULT_TOOLKIT_TOOL_NAMES = [
    "Edit",
    "Glob",
    "Read",
    "Write",
    "bash",
    "delete_file",
    "edit_notebook",
    "grep",
    "mcp_web_fetch",
    "read_file",
    "read_lints",
    "search_replace",
    "web_search",
    "write",
]


class _FakeResponse:
    def __init__(self, body: bytes, content_type: str = "text/html; charset=utf-8"):
        self._body = body
        self.headers = {"Content-Type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._body


class _QueryDrivenLLM:
    """Fake LLM that turns a JSON query into a ToolCall and then echoes ToolMessage."""

    def __init__(self):
        self.requests = []

    async def forward(self, context: BaseContext, **kwargs):
        last = context.last_message
        role = last.role.data
        tools = kwargs.get("tools") or []
        self.requests.append({"role": role, "tools": tools})
        self.last_tool_names = {
            item["function"]["name"]
            for item in tools
            if item.get("type") == "function" and "function" in item
        }

        if role == "user":
            payload = json.loads(last.content.data)
            tool_name = payload["tool"]
            assert tool_name in self.last_tool_names
            context.add_message(
                AssistantMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            tool_call_id=f"call_{tool_name}",
                            tool_name=tool_name,
                            arguments=payload.get("arguments", {}),
                        )
                    ],
                )
            )
            return context.last_message

        if role == "tool":
            context.add_message(AssistantMessage(content=last.content.data))
            return context.last_message

        context.add_message(AssistantMessage(content=""))
        return context.last_message


class _ToolProbeAgent(BaseAgent):
    def __init__(self, tool_manager: ToolManager):
        super().__init__()
        self.tool_manager = tool_manager
        self.llm = _QueryDrivenLLM()

    def _tools_param(self):
        return self.tool_manager.get_openai_tools()

    async def forward(self, user_input: str, max_steps: int = 3) -> BaseContext:
        context = BaseContext()
        context.add_message(UserMessage(content=user_input))
        await self.llm.forward(context, tools=self._tools_param())

        steps = 0
        while (
            steps < max_steps
            and getattr(context.last_message, "tool_calls", None)
            and context.last_message.tool_calls.data
        ):
            for tool_call in context.last_message.tool_calls.data:
                result = self.tool_manager.call_tool(
                    tool_call.tool_name.data,
                    **dict(tool_call.arguments.data),
                )
                context.add_message(
                    ToolMessage(
                        content=json.dumps(result, ensure_ascii=False),
                        tool_call_id=tool_call.tool_call_id.data,
                    )
                )
            await self.llm.forward(context, tools=self._tools_param())
            steps += 1

        return context


def _query(tool_name: str, arguments: dict) -> str:
    return json.dumps({"tool": tool_name, "arguments": arguments}, ensure_ascii=False)


def _tool_names_from_llm_request(request: dict) -> set[str]:
    return {
        item["function"]["name"]
        for item in request["tools"]
        if item.get("type") == "function" and "function" in item
    }


def _build_default_tool_manager(tmp_path: Path) -> ToolManager:
    manager = ToolManager()
    toolkits = [
        FileToolkits(session_id="agent-test", workspace_root=str(tmp_path)),
        BashToolkits(session_id="agent-test", workspace_root=str(tmp_path)),
        WebSearchToolkits(session_id="agent-test", workspace_root=str(tmp_path)),
        WebFetchToolkits(session_id="agent-test", workspace_root=str(tmp_path)),
    ]
    for toolkit in toolkits:
        manager.register_tools(toolkit.get_all_tools())
    return manager


def _tool_result_from_context(context: BaseContext) -> dict:
    assert context.last_message.role.data == "assistant"
    return json.loads(context.last_message.content.data)


def _prepare_files(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir(exist_ok=True)
    (tmp_path / "legacy").mkdir(exist_ok=True)
    (tmp_path / "docs" / "read.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    (tmp_path / "docs" / "edit.txt").write_text("old value\n", encoding="utf-8")
    (tmp_path / "docs" / "glob_a.py").write_text("print('a')\n", encoding="utf-8")
    (tmp_path / "docs" / "grep.txt").write_text("one\nneedle\nthree\n", encoding="utf-8")
    (tmp_path / "legacy" / "read.txt").write_text("legacy read\n", encoding="utf-8")
    (tmp_path / "legacy" / "replace.txt").write_text("before\n", encoding="utf-8")
    (tmp_path / "legacy" / "delete.txt").write_text("delete me\n", encoding="utf-8")
    (tmp_path / "nb.ipynb").write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": "markdown", "metadata": {}, "source": ["old text\n"]},
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )


def _case_arguments(tool_name: str, tmp_path: Path) -> dict:
    cases = {
        "Edit": {
            "file_path": str(tmp_path / "docs" / "edit.txt"),
            "old_string": "old value",
            "new_string": "new value",
        },
        "Glob": {"pattern": "**/*.py", "path": "docs"},
        "Read": {"file_path": str(tmp_path / "docs" / "read.txt"), "offset": 2, "limit": 1},
        "Write": {"file_path": str(tmp_path / "docs" / "write.txt"), "content": "strict write\n"},
        "bash": {"command": "printf agent-bash", "timeout": 5000},
        "delete_file": {"path": "legacy/delete.txt"},
        "edit_notebook": {
            "target_notebook": "nb.ipynb",
            "cell_idx": 0,
            "is_new_cell": False,
            "cell_language": "markdown",
            "old_string": "old",
            "new_string": "new",
        },
        "grep": {"pattern": "needle", "path": "docs", "output_mode": "content"},
        "mcp_web_fetch": {"url": "https://example.com/tool"},
        "read_file": {"path": "legacy/read.txt"},
        "read_lints": {},
        "search_replace": {
            "path": "legacy/replace.txt",
            "old_string": "before",
            "new_string": "after",
        },
        "web_search": {"search_term": "pygent", "explanation": "agent integration test"},
        "write": {"path": "legacy/write.txt", "contents": "legacy write\n"},
    }
    return cases[tool_name]


def _assert_case_result(tool_name: str, result: dict, tmp_path: Path) -> None:
    assert result["success"], result
    value = str(result.get("result", ""))

    if tool_name == "Edit":
        assert (tmp_path / "docs" / "edit.txt").read_text(encoding="utf-8") == "new value\n"
    elif tool_name == "Glob":
        assert "glob_a.py" in value
    elif tool_name == "Read":
        assert "2|beta" in value
    elif tool_name == "Write":
        assert (tmp_path / "docs" / "write.txt").read_text(encoding="utf-8") == "strict write\n"
    elif tool_name == "bash":
        if "bash executable not found" in value:
            pytest.skip(value)
        assert "agent-bash" in value
    elif tool_name == "delete_file":
        assert not (tmp_path / "legacy" / "delete.txt").exists()
    elif tool_name == "edit_notebook":
        data = json.loads((tmp_path / "nb.ipynb").read_text(encoding="utf-8"))
        assert data["cells"][0]["source"] == ["new text\n"]
    elif tool_name == "grep":
        assert "grep.txt" in value
        assert "2|needle" in value
    elif tool_name == "mcp_web_fetch":
        assert "Agent Page" in value
    elif tool_name == "read_file":
        assert "1|legacy read" in value
    elif tool_name == "read_lints":
        assert value
    elif tool_name == "search_replace":
        assert (tmp_path / "legacy" / "replace.txt").read_text(encoding="utf-8") == "after\n"
    elif tool_name == "web_search":
        assert "Fake Title" in value
    elif tool_name == "write":
        assert (tmp_path / "legacy" / "write.txt").read_text(encoding="utf-8") == "legacy write\n"


def test_default_toolkits_registered_tools_are_all_covered(tmp_path):
    manager = _build_default_tool_manager(tmp_path)

    assert {tool.metadata.data["name"] for tool in manager.get_registered_tools()} == set(
        DEFAULT_TOOLKIT_TOOL_NAMES
    )


@pytest.mark.parametrize("tool_name", DEFAULT_TOOLKIT_TOOL_NAMES)
def test_default_toolkit_tools_can_be_called_by_agent_query(tmp_path, monkeypatch, tool_name):
    _prepare_files(tmp_path)
    monkeypatch.setattr(
        "pygent.toolkits.web_search._search_via_html",
        lambda query: [("Fake Title", "https://example.com/fake", "Fake snippet")],
    )
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: _FakeResponse(
            b"<html><body><h1>Agent Page</h1><p>Fetched by test.</p></body></html>"
        ),
    )
    agent = _ToolProbeAgent(_build_default_tool_manager(tmp_path))

    context = asyncio.run(
        agent.forward(_query(tool_name, _case_arguments(tool_name, tmp_path)))
    )

    first_llm_request = agent.llm.requests[0]
    assert first_llm_request["role"] == "user"
    assert _tool_names_from_llm_request(first_llm_request) == set(DEFAULT_TOOLKIT_TOOL_NAMES)
    assert tool_name in _tool_names_from_llm_request(first_llm_request)

    result = _tool_result_from_context(context)
    _assert_case_result(tool_name, result, tmp_path)


def test_mcp_tools_can_be_called_by_agent_query(tmp_path):
    server = Path(__file__).resolve().parent / "mcp_test_server.py"
    manager = ToolManager()
    manager.add_mcp_server_stdio(
        server_id="agent-mcp",
        command=sys.executable,
        args=[str(server)],
        cwd=str(server.parent.parent),
    )
    agent = _ToolProbeAgent(manager)

    for tool_name, arguments in {
        "get_memory_statistics": {},
        "list_memory_files": {},
        "write_raw_conversation": {"content": "hello"},
    }.items():
        context = asyncio.run(agent.forward(_query(tool_name, arguments)))
        first_llm_request = agent.llm.requests[-2]
        assert first_llm_request["role"] == "user"
        assert tool_name in _tool_names_from_llm_request(first_llm_request)
        result = _tool_result_from_context(context)
        assert result["success"], result
        assert result["result"]
