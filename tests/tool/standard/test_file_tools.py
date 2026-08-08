from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path

import pytest

from pygent import IdempotencyPolicy, ToolKit, ToolSideEffect
from pygent.tool.standard import _files as file_module
from pygent.tool.standard._files import FileTools
from pygent.tool.standard._paths import normalize_desktop_path, normalize_tool_path

from ._helpers import invoke_tool, succeeded


def _to_msys_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    rest = resolved.as_posix()[3:]
    return f"/{drive}/{rest}"


def _parameters(handler) -> dict:
    return ToolKit(handler).definitions[0].parameters.to_dict()


def test_resolve_path_handles_relative_and_desktop_alias(tmp_path):
    assert (
        normalize_tool_path("notes.txt", str(tmp_path))
        == (tmp_path / "notes.txt").resolve()
    )
    assert normalize_desktop_path("/Users/Desktop/report.txt") == "~/Desktop/report.txt"
    assert (
        normalize_desktop_path("C:\\Users\\Desktop\\report.txt")
        == "~/Desktop/report.txt"
    )


def test_file_toolkit_exposes_only_lowercase_tool_names(tmp_path):
    tools = FileTools(workspace_root=tmp_path)
    definitions = ToolKit(*tools.handlers).definitions

    assert [item.name for item in definitions] == [
        "edit",
        "edit_notebook",
        "glob",
        "grep",
        "read",
        "read_lints",
        "write",
    ]
    assert not {
        "Edit",
        "Glob",
        "Read",
        "Write",
        "delete_file",
        "read_file",
        "search_replace",
    }.intersection(item.name for item in definitions)


@pytest.mark.asyncio
async def test_write_read_edit_grep_flow(tmp_path):
    tools = FileTools(workspace_root=tmp_path)
    target = tmp_path / "docs" / "example.txt"

    assert (
        await succeeded(
            tools.write, file_path="docs/example.txt", content="alpha\nbeta\nalpha\n"
        )
        == "写入完成"
    )
    assert target.read_text(encoding="utf-8") == "alpha\nbeta\nalpha\n"

    assert (
        await succeeded(tools.read, file_path="docs/example.txt", offset=2, limit=1)
        == "2|beta\n"
    )

    await succeeded(
        tools.edit,
        file_path="docs/example.txt",
        old_string="alpha",
        new_string="gamma",
    )
    await succeeded(
        tools.edit,
        file_path="docs/example.txt",
        old_string="alpha",
        new_string="delta",
        replace_all=True,
    )
    assert target.read_text(encoding="utf-8") == "gamma\nbeta\ndelta\n"

    grep_output = await succeeded(
        tools.grep,
        pattern="DELTA",
        path="docs",
        ignore_case=True,
        output_mode="content",
    )
    assert "example.txt" in grep_output
    assert "3|delta" in grep_output
    assert (
        await succeeded(tools.grep, pattern="delta", path="docs", output_mode="count")
        == "1"
    )


@pytest.mark.asyncio
async def test_file_tools_resolve_relative_paths_from_workspace_root(tmp_path):
    tools = FileTools(workspace_root=tmp_path)
    target = tmp_path / "docs" / "relative.txt"

    await succeeded(tools.write, file_path="docs/relative.txt", content="alpha\nbeta\n")
    assert (
        await succeeded(tools.read, file_path="docs/relative.txt", offset=2, limit=1)
        == "2|beta\n"
    )
    await succeeded(
        tools.edit,
        file_path="docs/relative.txt",
        old_string="beta",
        new_string="gamma",
    )
    assert target.read_text(encoding="utf-8") == "alpha\ngamma\n"


@pytest.mark.asyncio
async def test_file_tools_restrict_paths_to_workspace_by_default(tmp_path):
    outside = tmp_path.parent / "outside-pygent-path.txt"
    outside.write_text("secret\n", encoding="utf-8")
    tools = FileTools(workspace_root=tmp_path)

    result = await invoke_tool(tools.read, {"file_path": str(outside)})

    assert result.status == "failed"
    assert result.error_code == "path_outside_workspace"
    assert str(outside.resolve()) in (result.error or "")
    assert result.side_effect_committed is False


@pytest.mark.asyncio
async def test_glob_rejects_parent_traversal_outside_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("secret\n", encoding="utf-8")
    tools = FileTools(workspace_root=workspace)

    result = await invoke_tool(tools.glob, {"pattern": "../outside/*.txt"})

    assert result.status == "failed"
    assert result.error_code == "path_outside_workspace"
    assert result.side_effect_committed is False


@pytest.mark.asyncio
async def test_cancelled_file_write_finishes_before_cancellation_returns(tmp_path):
    tools = FileTools(workspace_root=tmp_path)
    target = tmp_path / "late.txt"
    started = threading.Event()

    def slow_write(_file_path: str, content: str) -> str:
        started.set()
        time.sleep(0.2)
        target.write_text(content, encoding="utf-8")
        return "done"

    tools._write = slow_write  # type: ignore[method-assign]
    invocation = asyncio.create_task(
        invoke_tool(tools.write, {"file_path": "ignored.txt", "content": "late"})
    )
    assert await asyncio.to_thread(started.wait, 1)

    invocation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await invocation

    assert target.read_text(encoding="utf-8") == "late"


@pytest.mark.asyncio
async def test_concurrent_edits_to_one_file_do_not_lose_updates(
    tmp_path, monkeypatch
):
    target = tmp_path / "shared.txt"
    target.write_text("alpha beta\n", encoding="utf-8")
    tools = FileTools(workspace_root=tmp_path)
    original_atomic_write = file_module._atomic_write_text

    def slow_atomic_write(path: Path, content: str) -> None:
        time.sleep(0.05)
        original_atomic_write(path, content)

    monkeypatch.setattr(file_module, "_atomic_write_text", slow_atomic_write)

    first, second = await asyncio.gather(
        invoke_tool(
            tools.edit,
            {
                "file_path": "shared.txt",
                "old_string": "alpha",
                "new_string": "one",
            },
            call_id="edit-alpha",
        ),
        invoke_tool(
            tools.edit,
            {
                "file_path": "shared.txt",
                "old_string": "beta",
                "new_string": "two",
            },
            call_id="edit-beta",
        ),
    )

    assert first.status == second.status == "succeeded"
    assert target.read_text(encoding="utf-8") == "one two\n"


@pytest.mark.asyncio
async def test_file_tools_can_disable_workspace_restriction(tmp_path):
    outside = tmp_path.parent / "outside-pygent-unrestricted.txt"
    outside.write_text("alpha\n", encoding="utf-8")
    tools = FileTools(workspace_root=tmp_path, restrict_to_workspace=False)

    assert await succeeded(tools.read, file_path=str(outside)) == "1|alpha\n"


@pytest.mark.asyncio
async def test_file_tools_accept_git_bash_msys_paths_on_windows(tmp_path):
    if os.name != "nt":
        pytest.skip("MSYS drive path compatibility is Windows-specific")

    tools = FileTools(workspace_root=tmp_path)
    target = tmp_path / "docs" / "example.txt"
    msys_target = _to_msys_path(target)
    msys_root = _to_msys_path(tmp_path)

    await succeeded(tools.write, file_path=msys_target, content="alpha\nbeta\n")
    assert (
        await succeeded(tools.read, file_path=msys_target, offset=1, limit=1)
        == "1|alpha\n"
    )
    assert (
        await succeeded(tools.glob, pattern="**/*.txt", path=msys_root)
    ).splitlines() == [str(target)]
    grep_output = await succeeded(
        tools.grep, pattern="beta", path=msys_root, output_mode="content"
    )
    assert f"{target}:2|beta" in grep_output
    await succeeded(
        tools.edit,
        file_path=msys_target,
        old_string="beta",
        new_string="gamma",
    )
    assert target.read_text(encoding="utf-8") == "alpha\ngamma\n"


@pytest.mark.asyncio
async def test_file_tool_path_errors_are_structured_tool_results(tmp_path):
    tools = FileTools(workspace_root=tmp_path)
    missing = tmp_path / "missing.txt"

    result = await invoke_tool(tools.read, {"file_path": str(missing)})

    assert result.status == "failed"
    assert result.error_kind == "filesystem_error"
    assert result.error_code == "file_not_found"
    assert "missing.txt" in (result.error or "")
    assert result.output is None


@pytest.mark.asyncio
async def test_glob_grep_and_edit_path_errors_include_standard_codes(tmp_path):
    tools = FileTools(workspace_root=tmp_path)
    missing_dir = tmp_path / "missing-dir"
    missing_file = tmp_path / "missing.txt"

    glob_result = await invoke_tool(
        tools.glob, {"pattern": "*.md", "path": str(missing_dir)}
    )
    grep_result = await invoke_tool(
        tools.grep, {"pattern": "x", "path": str(missing_dir)}
    )
    edit_result = await invoke_tool(
        tools.edit,
        {
            "file_path": str(missing_file),
            "old_string": "x",
            "new_string": "y",
        },
    )

    assert glob_result.error_code == "file_not_found"
    assert grep_result.error_code == "file_not_found"
    assert edit_result.error_code == "file_not_found"
    assert edit_result.side_effect_committed is False
    assert not missing_file.exists()


@pytest.mark.asyncio
async def test_write_uses_workspace_path_schema_and_resolves_relative_paths(tmp_path):
    tools = FileTools(workspace_root=tmp_path)
    target = tmp_path / "strict" / "out.txt"

    await succeeded(tools.write, file_path="strict/out.txt", content="hello\n")
    assert target.read_text(encoding="utf-8") == "hello\n"

    parameters = _parameters(tools.write)
    assert parameters["additionalProperties"] is False
    assert parameters["required"] == ["file_path", "content"]
    assert set(parameters["properties"]) == {"file_path", "content"}
    assert "workspace_root" in parameters["properties"]["file_path"]["description"]


@pytest.mark.asyncio
async def test_read_uses_requested_schema_and_reads_text_ranges(tmp_path):
    tools = FileTools(workspace_root=tmp_path)
    target = tmp_path / "strict" / "input.txt"
    target.parent.mkdir()
    target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    assert (
        await succeeded(tools.read, file_path=str(target), offset=2, limit=1)
        == "2|beta\n"
    )
    assert (
        await succeeded(tools.read, file_path="strict/input.txt", offset=1, limit=1)
        == "1|alpha\n"
    )

    parameters = _parameters(tools.read)
    assert parameters["additionalProperties"] is False
    assert parameters["required"] == ["file_path"]
    assert set(parameters["properties"]) == {"file_path", "limit", "offset", "pages"}

    invalid = await invoke_tool(tools.read, {"file_path": str(target), "limit": 0})
    assert invalid.status == "rejected"
    assert invalid.error_kind == "validation_error"


@pytest.mark.asyncio
async def test_edit_uses_workspace_path_schema_and_exact_replacement(tmp_path):
    tools = FileTools(workspace_root=tmp_path)
    target = tmp_path / "strict" / "edit.txt"
    target.parent.mkdir()
    target.write_text("alpha\nbeta\nalpha\n", encoding="utf-8")

    await succeeded(
        tools.edit, file_path=str(target), old_string="alpha", new_string="gamma"
    )
    await succeeded(
        tools.edit,
        file_path=str(target),
        old_string="alpha",
        new_string="delta",
        replace_all=True,
    )
    await succeeded(
        tools.edit,
        file_path="strict/edit.txt",
        old_string="delta",
        new_string="epsilon",
    )
    assert target.read_text(encoding="utf-8") == "gamma\nbeta\nepsilon\n"

    same = await invoke_tool(
        tools.edit,
        {"file_path": str(target), "old_string": "same", "new_string": "same"},
    )
    assert same.error_code == "identical_replacement"

    parameters = _parameters(tools.edit)
    assert parameters["additionalProperties"] is False
    assert parameters["required"] == ["file_path", "old_string", "new_string"]
    assert set(parameters["properties"]) == {
        "file_path",
        "old_string",
        "new_string",
        "replace_all",
    }
    assert parameters["properties"]["replace_all"]["default"] is False


@pytest.mark.asyncio
async def test_glob_finds_files_sorted_by_mtime_and_exposes_schema(tmp_path):
    tools = FileTools(workspace_root=tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    old_match = tmp_path / "src" / "old.py"
    new_match = tmp_path / "tests" / "new.py"
    non_match = tmp_path / "src" / "note.md"
    old_match.write_text("old", encoding="utf-8")
    new_match.write_text("new", encoding="utf-8")
    non_match.write_text("note", encoding="utf-8")
    os.utime(old_match, (100, 100))
    os.utime(new_match, (200, 200))

    assert (await succeeded(tools.glob, pattern="**/*.py")).splitlines() == [
        str(new_match),
        str(old_match),
    ]
    assert (await succeeded(tools.glob, pattern="*.py", path="src")).splitlines() == [
        str(old_match)
    ]
    not_directory = await invoke_tool(
        tools.glob, {"pattern": "*.py", "path": "src/old.py"}
    )
    assert not_directory.error_code == "not_a_directory"

    parameters = _parameters(tools.glob)
    assert parameters["required"] == ["pattern"]
    assert parameters["additionalProperties"] is False
    assert set(parameters["properties"]) == {"pattern", "path"}


def test_grep_tool_schema_uses_02_python_parameter_names(tmp_path):
    tools = FileTools(workspace_root=tmp_path)
    parameters = _parameters(tools.grep)
    properties = parameters["properties"]

    assert parameters["additionalProperties"] is False
    assert set(properties) == {
        "pattern",
        "path",
        "glob",
        "output_mode",
        "context_before",
        "context_after",
        "context",
        "ignore_case",
        "file_type",
        "head_limit",
        "offset",
        "multiline",
        "show_line_numbers",
    }
    assert parameters["required"] == ["pattern"]
    assert not {"-A", "-B", "-C", "-i", "-n", "type"}.intersection(properties)


@pytest.mark.asyncio
async def test_grep_schema_arguments_work_through_tool_call_layer(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.py").write_text("before\nAlpha\nAfter\n", encoding="utf-8")
    (docs / "b.rs").write_text("Alpha rust\n", encoding="utf-8")
    (docs / "c.txt").write_text("Alpha text\n", encoding="utf-8")
    tools = FileTools(workspace_root=tmp_path)

    output = await succeeded(
        tools.grep,
        pattern="alpha",
        path="docs",
        output_mode="content",
        ignore_case=True,
        context_before=1,
        context_after=1,
        file_type="py",
        head_limit=0,
    )
    assert "a.py" in output
    assert "1|before" in output
    assert "2|Alpha" in output
    assert "b.rs" not in output

    default_output = await succeeded(tools.grep, pattern="Alpha", path="docs")
    assert "a.py" in default_output
    assert "1|Alpha" not in default_output

    brace_glob = await succeeded(
        tools.grep, pattern="Alpha", path="docs", glob="*.{py,rs}"
    )
    assert "a.py" in brace_glob
    assert "b.rs" in brace_glob
    assert "c.txt" not in brace_glob


@pytest.mark.asyncio
async def test_grep_multiline_context_and_line_number_toggle(tmp_path):
    (tmp_path / "notes.txt").write_text(
        "one\nstart\nmiddle\nend\nlast\n", encoding="utf-8"
    )
    tools = FileTools(workspace_root=tmp_path)

    output = await succeeded(
        tools.grep,
        pattern="start.*end",
        path="notes.txt",
        output_mode="content",
        multiline=True,
        context=1,
        show_line_numbers=False,
    )

    assert "notes.txt:one" in output
    assert "notes.txt:start" in output
    assert "notes.txt:middle" in output
    assert "notes.txt:end" in output
    assert "notes.txt:last" in output
    assert "2|start" not in output


@pytest.mark.asyncio
async def test_file_tool_errors_are_non_throwing_tool_results(tmp_path):
    tools = FileTools(workspace_root=tmp_path)

    read = await invoke_tool(tools.read, {"file_path": "missing.txt"})
    edit = await invoke_tool(
        tools.edit,
        {"file_path": "missing.txt", "old_string": "a", "new_string": "b"},
    )
    grep = await invoke_tool(tools.grep, {"pattern": "x", "path": "missing"})

    assert read.error_code == "file_not_found"
    assert edit.error_code == "file_not_found"
    assert grep.error_code == "file_not_found"


@pytest.mark.asyncio
async def test_read_describes_binary_and_image_files(tmp_path):
    tools = FileTools(workspace_root=tmp_path)
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01\x02")
    (tmp_path / "pixel.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00")

    assert "blob.bin" in await succeeded(tools.read, file_path="blob.bin")
    assert "pixel.png" in await succeeded(tools.read, file_path="pixel.png")


@pytest.mark.asyncio
async def test_edit_notebook_insert_replace_and_bounds(tmp_path):
    notebook = tmp_path / "nb.ipynb"
    notebook.write_text(
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
    tools = FileTools(workspace_root=tmp_path)

    await succeeded(
        tools.edit_notebook,
        target_notebook="nb.ipynb",
        cell_idx=0,
        is_new_cell=False,
        cell_language="markdown",
        old_string="old",
        new_string="new",
    )
    data = json.loads(notebook.read_text(encoding="utf-8"))
    assert data["cells"][0]["source"] == ["new text\n"]

    await succeeded(
        tools.edit_notebook,
        target_notebook="nb.ipynb",
        cell_idx=1,
        is_new_cell=True,
        cell_language="python",
        old_string="",
        new_string="print('ok')\n",
    )
    data = json.loads(notebook.read_text(encoding="utf-8"))
    assert data["cells"][1]["cell_type"] == "code"
    assert data["cells"][1]["source"] == ["print('ok')\n"]

    bounds = await invoke_tool(
        tools.edit_notebook,
        {
            "target_notebook": "nb.ipynb",
            "cell_idx": 99,
            "is_new_cell": False,
            "cell_language": "markdown",
            "old_string": "x",
            "new_string": "y",
        },
    )
    assert bounds.error_code == "cell_index_out_of_range"
    assert "0..1" in (bounds.error or "")


@pytest.mark.asyncio
async def test_read_lints_reports_python_syntax_diagnostics(tmp_path):
    (tmp_path / "good.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "bad.py").write_text("if True print('bad')\n", encoding="utf-8")
    tools = FileTools(workspace_root=tmp_path)

    output = await succeeded(tools.read_lints, paths=["good.py", "bad.py"])

    diagnostics = json.loads(output)
    assert diagnostics["tool"] == "python.compile"
    assert len(diagnostics["diagnostics"]) == 1
    assert diagnostics["diagnostics"][0]["path"].endswith("bad.py")
    assert diagnostics["diagnostics"][0]["line"] == 1


def test_file_tools_publish_explicit_02_side_effect_and_permission_policies(tmp_path):
    tools = FileTools(workspace_root=tmp_path)
    specs = {item.definition.name: item for item in ToolKit(*tools.handlers).specs}

    assert specs["read"].side_effect is ToolSideEffect.READ
    assert specs["glob"].side_effect is ToolSideEffect.READ
    assert specs["grep"].side_effect is ToolSideEffect.READ
    assert specs["read_lints"].side_effect is ToolSideEffect.READ
    assert specs["write"].side_effect is ToolSideEffect.WRITE
    assert specs["edit"].side_effect is ToolSideEffect.WRITE
    assert specs["edit_notebook"].side_effect is ToolSideEffect.WRITE
    assert specs["write"].idempotency is IdempotencyPolicy.INHERENT
    assert specs["edit"].idempotency is IdempotencyPolicy.NOT_IDEMPOTENT
    assert specs["edit_notebook"].idempotency is IdempotencyPolicy.NOT_IDEMPOTENT
    assert specs["read"].required_permissions == ("filesystem:read",)
    assert specs["write"].required_permissions == ("filesystem:write",)
    assert all(item.sandbox_profile == "workspace" for item in specs.values())
