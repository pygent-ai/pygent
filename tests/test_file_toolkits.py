import json
import os
from pathlib import Path

import pytest

from pygent.toolkits.file_operations import FileToolkits, _normalize_desktop_path, _resolve_path


def _to_msys_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    rest = resolved.as_posix()[3:]
    return f"/{drive}/{rest}"


def test_resolve_path_handles_relative_and_desktop_alias(tmp_path):
    assert _resolve_path("notes.txt", str(tmp_path)) == (tmp_path / "notes.txt").resolve()
    assert _normalize_desktop_path("/Users/Desktop/report.txt") == "~/Desktop/report.txt"
    assert _normalize_desktop_path("C:\\Users\\Desktop\\report.txt") == "~/Desktop/report.txt"


def test_file_toolkit_exposes_only_lowercase_tool_names(tmp_path):
    tools = FileToolkits(session_id="s", workspace_root=str(tmp_path))

    assert [tool.metadata.data["name"] for tool in tools.get_all_tools()] == [
        "edit",
        "edit_notebook",
        "glob",
        "grep",
        "read",
        "read_lints",
        "write",
    ]
    for removed_name in ["Edit", "Glob", "Read", "Write", "delete_file", "read_file", "search_replace"]:
        assert tools.get_tool(removed_name) is None


def test_write_read_edit_grep_flow(tmp_path):
    tools = FileToolkits(session_id="s", workspace_root=str(tmp_path))
    target = tmp_path / "docs" / "example.txt"

    tools.write(str(target), "alpha\nbeta\nalpha\n")
    assert target.read_text(encoding="utf-8") == "alpha\nbeta\nalpha\n"

    assert tools.read(str(target), offset=2, limit=1) == "2|beta\n"

    tools.edit(str(target), "alpha", "gamma")
    assert target.read_text(encoding="utf-8") == "gamma\nbeta\nalpha\n"

    tools.edit(str(target), "alpha", "delta", replace_all=True)
    assert target.read_text(encoding="utf-8") == "gamma\nbeta\ndelta\n"

    grep_output = tools.grep("DELTA", path="docs", ignore_case=True, output_mode="content")
    assert "example.txt" in grep_output
    assert "3|delta" in grep_output
    assert tools.grep("delta", path="docs", output_mode="count") == "1"


def test_file_tools_resolve_relative_paths_from_workspace_root(tmp_path):
    tools = FileToolkits(session_id="s", workspace_root=str(tmp_path))
    target = tmp_path / "docs" / "relative.txt"

    assert tools.write("docs/relative.txt", "alpha\nbeta\n") == "写入完成"
    assert target.read_text(encoding="utf-8") == "alpha\nbeta\n"

    assert tools.read("docs/relative.txt", offset=2, limit=1) == "2|beta\n"

    assert tools.edit("docs/relative.txt", "beta", "gamma") == "替换完成"
    assert target.read_text(encoding="utf-8") == "alpha\ngamma\n"


def test_file_tools_restrict_paths_to_workspace_by_default(tmp_path):
    tools = FileToolkits(session_id="s", workspace_root=str(tmp_path))
    outside = tmp_path.parent / "outside-pygent-path.txt"
    outside.write_text("secret\n", encoding="utf-8")

    result = tools.call_tool("read", file_path=str(outside))

    assert result["success"] is False
    assert result["error_type"] == "PathOutsideWorkspaceError"
    assert result["details"]["input_path"] == str(outside)
    assert result["details"]["path"] == str(outside.resolve())
    assert result["details"]["workspace_root"] == str(tmp_path.resolve())


def test_file_tools_can_disable_workspace_restriction(tmp_path):
    outside = tmp_path.parent / "outside-pygent-unrestricted.txt"
    outside.write_text("alpha\n", encoding="utf-8")
    tools = FileToolkits(
        session_id="s",
        workspace_root=str(tmp_path),
        restrict_to_workspace=False,
    )

    assert tools.read(str(outside)) == "1|alpha\n"


def test_file_tools_accept_git_bash_msys_paths_on_windows(tmp_path):
    if os.name != "nt":
        pytest.skip("MSYS drive path compatibility is Windows-specific")

    tools = FileToolkits(session_id="s", workspace_root=str(tmp_path))
    target = tmp_path / "docs" / "example.txt"
    msys_target = _to_msys_path(target)
    msys_root = _to_msys_path(tmp_path)

    assert tools.write(msys_target, "alpha\nbeta\n") == "写入完成"
    assert target.read_text(encoding="utf-8") == "alpha\nbeta\n"

    assert tools.read(msys_target, offset=1, limit=1) == "1|alpha\n"

    assert tools.glob("**/*.txt", path=msys_root).splitlines() == [str(target)]

    grep_output = tools.grep("beta", path=msys_root, output_mode="content")
    assert f"{target}:2|beta" in grep_output

    assert tools.edit(msys_target, "beta", "gamma") == "替换完成"
    assert target.read_text(encoding="utf-8") == "alpha\ngamma\n"


def test_file_tool_path_errors_are_structured_through_call_tool(tmp_path):
    tools = FileToolkits(session_id="s", workspace_root=str(tmp_path))
    missing = tmp_path / "missing.txt"

    result = tools.call_tool("read", file_path=str(missing))

    assert result["success"] is False
    assert result["error_type"] == "FileNotFoundError"
    assert "missing.txt" in result["error"]
    assert result["details"]["input_path"] == str(missing)
    assert result["details"]["path"] == str(missing)
    assert "result" not in result


def test_glob_grep_and_edit_path_errors_include_structured_error_type(tmp_path):
    tools = FileToolkits(session_id="s", workspace_root=str(tmp_path))
    missing_dir = tmp_path / "missing-dir"
    missing_file = tmp_path / "missing.txt"

    glob_result = tools.call_tool("glob", pattern="*.md", path=str(missing_dir))
    assert glob_result["success"] is False
    assert glob_result["error_type"] == "FileNotFoundError"
    assert glob_result["details"]["input_path"] == str(missing_dir)
    assert glob_result["details"]["path"] == str(missing_dir)

    grep_result = tools.call_tool("grep", pattern="x", path=str(missing_dir))
    assert grep_result["success"] is False
    assert grep_result["error_type"] == "FileNotFoundError"
    assert grep_result["details"]["input_path"] == str(missing_dir)
    assert grep_result["details"]["path"] == str(missing_dir)

    edit_result = tools.call_tool(
        "edit",
        file_path=str(missing_file),
        old_string="x",
        new_string="y",
    )
    assert edit_result["success"] is False
    assert edit_result["error_type"] == "FileNotFoundError"
    assert edit_result["details"]["input_path"] == str(missing_file)
    assert edit_result["details"]["path"] == str(missing_file)
    assert not missing_file.exists()


def test_write_uses_workspace_path_schema_and_resolves_relative_paths(tmp_path):
    tools = FileToolkits(session_id="s", workspace_root=str(tmp_path))
    target = tmp_path / "strict" / "out.txt"

    tools.write("strict/out.txt", "hello\n")
    assert target.read_text(encoding="utf-8") == "hello\n"

    schema = tools.get_tool("write").to_openai_function()
    assert schema["name"] == "write"
    parameters = schema["parameters"]
    assert parameters["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert parameters["additionalProperties"] is False
    assert parameters["required"] == ["file_path", "content"]
    assert set(parameters["properties"]) == {"file_path", "content"}
    assert "Relative paths are rejected" not in parameters["properties"]["file_path"]["description"]


def test_read_uses_requested_schema_and_reads_text_ranges(tmp_path):
    tools = FileToolkits(session_id="s", workspace_root=str(tmp_path))
    target = tmp_path / "strict" / "input.txt"
    target.parent.mkdir()
    target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    assert tools.read(str(target), offset=2, limit=1) == "2|beta\n"
    assert tools.read("strict/input.txt", offset=1, limit=1) == "1|alpha\n"

    schema = tools.get_tool("read").to_openai_function()
    assert schema["name"] == "read"
    parameters = schema["parameters"]
    assert parameters["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert parameters["additionalProperties"] is False
    assert parameters["required"] == ["file_path"]
    assert set(parameters["properties"]) == {"file_path", "limit", "offset", "pages"}
    assert "Relative paths are rejected" not in parameters["properties"]["file_path"]["description"]
    assert parameters["properties"]["limit"]["exclusiveMinimum"] == 0
    assert parameters["properties"]["offset"]["minimum"] == 0

    invalid = tools.call_tool("read", file_path=str(target), limit=0)
    assert not invalid["success"]
    assert "limit" in invalid["details"]


def test_edit_uses_workspace_path_schema_and_exact_replacement(tmp_path):
    tools = FileToolkits(session_id="s", workspace_root=str(tmp_path))
    target = tmp_path / "strict" / "edit.txt"
    target.parent.mkdir()
    target.write_text("alpha\nbeta\nalpha\n", encoding="utf-8")

    tools.edit(str(target), "alpha", "gamma")
    assert target.read_text(encoding="utf-8") == "gamma\nbeta\nalpha\n"

    tools.edit(str(target), "alpha", "delta", replace_all=True)
    assert target.read_text(encoding="utf-8") == "gamma\nbeta\ndelta\n"
    tools.edit("strict/edit.txt", "delta", "epsilon")
    assert target.read_text(encoding="utf-8") == "gamma\nbeta\nepsilon\n"
    assert "new_string" in tools.edit(str(target), "same", "same")

    schema = tools.get_tool("edit").to_openai_function()
    assert schema["name"] == "edit"
    parameters = schema["parameters"]
    assert parameters["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert parameters["additionalProperties"] is False
    assert parameters["required"] == ["file_path", "old_string", "new_string"]
    assert set(parameters["properties"]) == {"file_path", "old_string", "new_string", "replace_all"}
    assert "Relative paths are rejected" not in parameters["properties"]["file_path"]["description"]
    assert parameters["properties"]["replace_all"]["default"] is False


def test_glob_finds_files_sorted_by_mtime_and_exposes_schema(tmp_path):
    tools = FileToolkits(session_id="s", workspace_root=str(tmp_path))
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

    assert tools.glob("**/*.py").splitlines() == [str(new_match), str(old_match)]
    assert tools.glob("*.py", path="src").splitlines() == [str(old_match)]
    assert "path must be a directory" in tools.glob("*.py", path="src/old.py")

    params = tools.get_tool("glob").to_openai_function()["parameters"]
    assert params["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert params["required"] == ["pattern"]
    assert params["additionalProperties"] is False
    assert "description" not in params["properties"]


def test_grep_tool_schema_uses_rg_style_parameter_names(tmp_path):
    tool = FileToolkits(session_id="s", workspace_root=str(tmp_path)).get_tool("grep")
    parameters = tool.to_openai_function()["parameters"]
    properties = parameters["properties"]

    assert parameters["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert parameters["additionalProperties"] is False
    assert list(properties) == [
        "-A",
        "-B",
        "-C",
        "-i",
        "-n",
        "context",
        "glob",
        "head_limit",
        "multiline",
        "offset",
        "output_mode",
        "path",
        "pattern",
        "type",
    ]
    assert parameters["required"] == ["pattern"]
    assert "description" not in properties
    assert properties["output_mode"]["enum"] == ["content", "files_with_matches", "count"]


def test_grep_schema_arguments_work_through_tool_manager(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.py").write_text("before\nAlpha\nAfter\n", encoding="utf-8")
    (docs / "b.rs").write_text("Alpha rust\n", encoding="utf-8")
    (docs / "c.txt").write_text("Alpha text\n", encoding="utf-8")
    tools = FileToolkits(session_id="s", workspace_root=str(tmp_path))

    result = tools.call_tool(
        "grep",
        pattern="alpha",
        path="docs",
        output_mode="content",
        **{"-i": True, "-B": 1, "-A": 1, "type": "py", "head_limit": 0},
    )
    assert result["success"] is True
    assert "a.py" in result["result"]
    assert "1|before" in result["result"]
    assert "2|Alpha" in result["result"]
    assert "b.rs" not in result["result"]

    default_result = tools.call_tool("grep", pattern="Alpha", path="docs")
    assert default_result["success"] is True
    assert "a.py" in default_result["result"]
    assert "1|Alpha" not in default_result["result"]

    brace_glob = tools.grep("Alpha", path="docs", glob="*.{py,rs}")
    assert "a.py" in brace_glob
    assert "b.rs" in brace_glob
    assert "c.txt" not in brace_glob


def test_grep_multiline_context_and_line_number_toggle(tmp_path):
    (tmp_path / "notes.txt").write_text("one\nstart\nmiddle\nend\nlast\n", encoding="utf-8")
    tools = FileToolkits(session_id="s", workspace_root=str(tmp_path))

    output = tools.call_tool(
        "grep",
        pattern="start.*end",
        path="notes.txt",
        output_mode="content",
        multiline=True,
        context=1,
        **{"-n": False},
    )

    assert output["success"] is True
    assert "notes.txt:one" in output["result"]
    assert "notes.txt:start" in output["result"]
    assert "notes.txt:middle" in output["result"]
    assert "notes.txt:end" in output["result"]
    assert "notes.txt:last" in output["result"]
    assert "2|start" not in output["result"]


def test_file_tool_errors_are_non_throwing(tmp_path):
    tools = FileToolkits(session_id="s", workspace_root=str(tmp_path))

    assert "missing.txt" in tools.read(str(tmp_path / "missing.txt"))
    assert "missing.txt" in tools.edit(str(tmp_path / "missing.txt"), "a", "b")
    assert str(tmp_path / "missing") in tools.grep("x", path="missing")


def test_read_describes_binary_and_image_files(tmp_path):
    tools = FileToolkits(session_id="s", workspace_root=str(tmp_path))
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01\x02")
    (tmp_path / "pixel.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00")

    assert "blob.bin" in tools.read(str(tmp_path / "blob.bin"))
    assert "pixel.png" in tools.read(str(tmp_path / "pixel.png"))


def test_edit_notebook_insert_replace_and_bounds(tmp_path):
    notebook = tmp_path / "nb.ipynb"
    notebook.write_text(
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
    tools = FileToolkits(session_id="s", workspace_root=str(tmp_path))

    tools.edit_notebook("nb.ipynb", 0, False, "markdown", "old", "new")
    data = json.loads(notebook.read_text(encoding="utf-8"))
    assert data["cells"][0]["source"] == ["new text\n"]

    tools.edit_notebook("nb.ipynb", 1, True, "python", "", "print('ok')\n")
    data = json.loads(notebook.read_text(encoding="utf-8"))
    assert data["cells"][1]["cell_type"] == "code"
    assert data["cells"][1]["source"] == ["print('ok')\n"]

    assert "0..1" in tools.edit_notebook("nb.ipynb", 99, False, "markdown", "x", "y")
