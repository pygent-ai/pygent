import json

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
