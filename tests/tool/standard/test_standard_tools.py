from __future__ import annotations

import json

import pytest

from pygent import AIMessage, Context, ToolCall
from pygent.tool.standard import StandardTools

from ._helpers import allow

STANDARD_TOOL_NAMES = [
    "bash",
    "edit",
    "edit_notebook",
    "glob",
    "grep",
    "read",
    "read_lints",
    "web_fetch",
    "web_search",
    "write",
]


class _FakeResponse:
    def __init__(self) -> None:
        self.headers = {"Content-Type": "text/html; charset=utf-8"}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, limit: int = -1):
        return b"<html><body><h1>Agent Page</h1></body></html>"


def _suite(tmp_path) -> StandardTools:
    return StandardTools(
        workspace_root=tmp_path,
        web_searcher=lambda query: [
            ("Fake Title", "https://example.com/fake", "Fake snippet")
        ],
        web_fetcher=lambda request, timeout: _FakeResponse(),
        web_resolver=lambda host, port: [
            (None, None, None, None, ("93.184.216.34", port))
        ],
    )


def _prepare_files(tmp_path) -> None:
    (tmp_path / "docs").mkdir(exist_ok=True)
    (tmp_path / "docs" / "read.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    (tmp_path / "docs" / "edit.txt").write_text("old value\n", encoding="utf-8")
    (tmp_path / "docs" / "glob_a.py").write_text("print('a')\n", encoding="utf-8")
    (tmp_path / "docs" / "grep.txt").write_text(
        "one\nneedle\nthree\n", encoding="utf-8"
    )
    (tmp_path / "nb.ipynb").write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": "markdown", "metadata": {}, "source": ["old text\n"]}
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )


def _arguments(name: str) -> dict[str, object]:
    return {
        "bash": {"command": "printf agent-bash", "timeout": 5000},
        "edit": {
            "file_path": "docs/edit.txt",
            "old_string": "old value",
            "new_string": "new value",
        },
        "edit_notebook": {
            "target_notebook": "nb.ipynb",
            "cell_idx": 0,
            "is_new_cell": False,
            "cell_language": "markdown",
            "old_string": "old",
            "new_string": "new",
        },
        "glob": {"pattern": "**/*.py", "path": "docs"},
        "grep": {"pattern": "needle", "path": "docs", "output_mode": "content"},
        "read": {"file_path": "docs/read.txt", "offset": 2, "limit": 1},
        "read_lints": {"paths": ["docs/glob_a.py"]},
        "web_fetch": {"url": "https://example.com/tool"},
        "web_search": {"search_term": "pygent", "description": "integration"},
        "write": {"file_path": "docs/write.txt", "content": "strict write\n"},
    }[name]


def test_standard_tools_explicitly_assemble_all_ten_model_definitions(tmp_path):
    suite = _suite(tmp_path)

    assert [item.name for item in suite.toolkit.definitions] == STANDARD_TOOL_NAMES
    assert [item.definition.name for item in suite.toolkit.specs] == STANDARD_TOOL_NAMES
    assert suite.toolkit.build_registry() is not None


@pytest.mark.parametrize("tool_name", STANDARD_TOOL_NAMES)
@pytest.mark.asyncio
async def test_every_standard_tool_runs_through_02_tool_call_layer(tmp_path, tool_name):
    _prepare_files(tmp_path)
    suite = _suite(tmp_path)
    layer = suite.toolkit.local_layer(authorization_adapter=allow)
    context = suite.toolkit.make_visible_in(Context())

    message, returned_context = await layer.invoke(
        AIMessage(
            tool_calls=(
                ToolCall(
                    call_id=f"call-{tool_name}",
                    name=tool_name,
                    arguments=_arguments(tool_name),
                ),
            )
        ),
        context,
    )

    result = message.results[0]
    assert returned_context is context
    assert result.status == "succeeded", result
    assert result.name == tool_name
    assert result.tool_id.startswith("standard.")


def test_standard_tools_are_deployment_local_and_not_portable_state(tmp_path):
    suite = _suite(tmp_path)

    assert "toolkit" not in suite.toolkit.specs[0].definition.parameters
    assert "workspace_root" not in suite.toolkit.specs[0].definition.parameters
    assert all(not hasattr(spec, "handler") for spec in suite.toolkit.specs)
