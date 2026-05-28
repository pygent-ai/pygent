import json
import os

from pygent.toolkits.file_operations import FileToolkits, _normalize_desktop_path, _resolve_path


def test_resolve_path_handles_relative_and_desktop_alias(tmp_path):
    assert _resolve_path("notes.txt", str(tmp_path)) == (tmp_path / "notes.txt").resolve()
    assert _normalize_desktop_path("/Users/Desktop/report.txt") == "~/Desktop/report.txt"
    assert _normalize_desktop_path("C:\\Users\\Desktop\\report.txt") == "~/Desktop/report.txt"


def test_write_read_replace_grep_and_delete_file(tmp_path):
    tools = FileToolkits(session_id="s", workspace_root=str(tmp_path))

    assert "完成" in tools.write("docs/example.txt", "alpha\nbeta\nalpha\n")
    assert (tmp_path / "docs" / "example.txt").read_text(encoding="utf-8") == "alpha\nbeta\nalpha\n"

    assert tools.read_file("docs/example.txt", offset=2, limit=1) == "2|beta\n"

    assert "完成" in tools.search_replace("docs/example.txt", "alpha", "gamma")
    assert (tmp_path / "docs" / "example.txt").read_text(encoding="utf-8") == "gamma\nbeta\nalpha\n"

    assert "完成" in tools.search_replace("docs/example.txt", "alpha", "delta", replace_all=True)
    assert (tmp_path / "docs" / "example.txt").read_text(encoding="utf-8") == "gamma\nbeta\ndelta\n"

    grep_output = tools.grep("DELTA", path="docs", ignore_case=True, output_mode="content")
    assert "example.txt" in grep_output
    assert "3|delta" in grep_output

    assert tools.grep("delta", path="docs", output_mode="count") == "1"
    assert "删除" in tools.delete_file("docs/example.txt")
    assert not (tmp_path / "docs" / "example.txt").exists()


def test_Write_uses_absolute_path_schema_and_rejects_relative_paths(tmp_path):
    tools = FileToolkits(session_id="s", workspace_root=str(tmp_path))
    target = tmp_path / "strict" / "out.txt"

    assert "完成" in tools.Write(str(target), "hello\n")
    assert target.read_text(encoding="utf-8") == "hello\n"
    assert "绝对路径" in tools.Write("relative.txt", "nope")

    write_tool = tools.get_tool("Write")
    assert write_tool is not None
    schema = write_tool.to_openai_function()

    assert schema["name"] == "Write"
    parameters = schema["parameters"]
    assert parameters["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert parameters["additionalProperties"] is False
    assert parameters["required"] == ["file_path", "content"]
    assert set(parameters["properties"]) == {"file_path", "content"}
    assert parameters["properties"]["file_path"]["type"] == "string"
    assert "Absolute destination file path" in parameters["properties"]["file_path"]["description"]
    assert parameters["properties"]["content"]["type"] == "string"
    assert "Full text content" in parameters["properties"]["content"]["description"]


def test_Read_uses_requested_schema_and_reads_text_ranges(tmp_path):
    tools = FileToolkits(session_id="s", workspace_root=str(tmp_path))
    target = tmp_path / "strict" / "input.txt"
    target.parent.mkdir()
    target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    assert tools.Read(str(target), offset=2, limit=1) == "2|beta\n"
    assert "绝对路径" in tools.Read("relative.txt")

    read_tool = tools.get_tool("Read")
    assert read_tool is not None
    schema = read_tool.to_openai_function()

    assert schema["name"] == "Read"
    parameters = schema["parameters"]
    assert parameters["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert parameters["additionalProperties"] is False
    assert parameters["required"] == ["file_path"]
    assert set(parameters["properties"]) == {"file_path", "limit", "offset", "pages"}
    assert parameters["properties"]["file_path"]["type"] == "string"
    assert parameters["properties"]["limit"]["type"] == "integer"
    assert parameters["properties"]["limit"]["exclusiveMinimum"] == 0
    assert parameters["properties"]["limit"]["maximum"] == 9007199254740991
    assert parameters["properties"]["offset"]["type"] == "integer"
    assert parameters["properties"]["offset"]["minimum"] == 0
    assert parameters["properties"]["offset"]["maximum"] == 9007199254740991
    assert parameters["properties"]["pages"]["type"] == "string"
    assert "PDF" in parameters["properties"]["pages"]["description"]

    invalid = tools.call_tool("Read", file_path=str(target), limit=0)
    assert not invalid["success"]
    assert "limit" in invalid["details"]


def test_Edit_uses_absolute_path_schema_and_exact_replacement(tmp_path):
    tools = FileToolkits(session_id="s", workspace_root=str(tmp_path))
    target = tmp_path / "strict" / "edit.txt"
    target.parent.mkdir()
    target.write_text("alpha\nbeta\nalpha\n", encoding="utf-8")

    assert "完成" in tools.Edit(str(target), "alpha", "gamma")
    assert target.read_text(encoding="utf-8") == "gamma\nbeta\nalpha\n"

    assert "完成" in tools.Edit(str(target), "alpha", "delta", replace_all=True)
    assert target.read_text(encoding="utf-8") == "gamma\nbeta\ndelta\n"
    assert "绝对路径" in tools.Edit("relative.txt", "a", "b")
    assert "不同" in tools.Edit(str(target), "same", "same")

    edit_tool = tools.get_tool("Edit")
    assert edit_tool is not None
    schema = edit_tool.to_openai_function()

    assert schema["name"] == "Edit"
    parameters = schema["parameters"]
    assert parameters["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert parameters["additionalProperties"] is False
    assert parameters["required"] == ["file_path", "old_string", "new_string"]
    assert set(parameters["properties"]) == {"file_path", "old_string", "new_string", "replace_all"}
    assert parameters["properties"]["file_path"]["type"] == "string"
    assert "Absolute path" in parameters["properties"]["file_path"]["description"]
    assert parameters["properties"]["old_string"]["type"] == "string"
    assert "Exact text" in parameters["properties"]["old_string"]["description"]
    assert parameters["properties"]["new_string"]["type"] == "string"
    assert "different from old_string" in parameters["properties"]["new_string"]["description"]
    assert "enum" not in parameters["properties"]["new_string"]
    assert parameters["properties"]["replace_all"]["type"] == "boolean"
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

    assert tools.Glob("**/*.py").splitlines() == [str(new_match), str(old_match)]
    assert tools.Glob("*.py", path="src").splitlines() == [str(old_match)]
    assert "path must be a directory" in tools.Glob("*.py", path="src/old.py")

    glob_tool = tools.get_tool("Glob")
    params = glob_tool.to_openai_function()["parameters"]
    assert params["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert params["required"] == ["pattern"]
    assert params["additionalProperties"] is False
    assert "description" not in params["properties"]
    assert "workspace root" in params["properties"]["path"]["description"]
    assert "Glob pattern" in params["properties"]["pattern"]["description"]


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
    assert "context_before" not in properties
    assert "description" not in properties
    assert properties["-A"]["description"].startswith("Number of lines to show after")
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

    assert "missing.txt" in tools.read_file("missing.txt")
    assert "missing.txt" in tools.search_replace("missing.txt", "a", "b")
    assert "missing.txt" in tools.delete_file("missing.txt")
    assert str(tmp_path / "missing") in tools.grep("x", path="missing")


def test_read_file_describes_binary_and_image_files(tmp_path):
    tools = FileToolkits(session_id="s", workspace_root=str(tmp_path))
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01\x02")
    (tmp_path / "pixel.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00")

    assert "blob.bin" in tools.read_file("blob.bin")
    assert "pixel.png" in tools.read_file("pixel.png")


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

    assert "更新" in tools.edit_notebook("nb.ipynb", 0, False, "markdown", "old", "new")
    data = json.loads(notebook.read_text(encoding="utf-8"))
    assert data["cells"][0]["source"] == ["new text\n"]

    assert "更新" in tools.edit_notebook("nb.ipynb", 1, True, "python", "", "print('ok')\n")
    data = json.loads(notebook.read_text(encoding="utf-8"))
    assert data["cells"][1]["cell_type"] == "code"
    assert data["cells"][1]["source"] == ["print('ok')\n"]

    assert "越界" in tools.edit_notebook("nb.ipynb", 99, False, "markdown", "x", "y")
