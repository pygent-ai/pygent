from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest
from pypdf import PdfWriter

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


def test_pdf_reader_uses_current_pypdf_backend(tmp_path) -> None:
    path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with path.open("wb") as stream:
        writer.write(stream)
    assert file_module._read_pdf_text(path, "1") == "--- page 1 ---"


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
        ignoreCase=True,
    )
    assert grep_output == "example.txt:3: delta"


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
async def test_concurrent_edits_to_one_file_do_not_lose_updates(tmp_path, monkeypatch):
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
    ).splitlines() == ["docs/example.txt"]
    grep_output = await succeeded(tools.grep, pattern="beta", path=msys_root)
    assert grep_output == "docs/example.txt:2: beta"
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
async def test_glob_finds_relative_files_and_exposes_pi_schema(tmp_path):
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

    assert set((await succeeded(tools.glob, pattern="**/*.py")).splitlines()) == {
        "src/old.py",
        "tests/new.py",
    }
    assert (await succeeded(tools.glob, pattern="*.py", path="src")).splitlines() == [
        "old.py"
    ]
    not_directory = await invoke_tool(
        tools.glob, {"pattern": "*.py", "path": "src/old.py"}
    )
    assert not_directory.error_code == "not_a_directory"

    parameters = _parameters(tools.glob)
    assert parameters["required"] == ["pattern"]
    assert parameters["additionalProperties"] is False
    assert set(parameters["properties"]) == {"pattern", "path", "limit"}
    assert parameters["properties"]["path"]["type"] == "string"
    assert parameters["properties"]["limit"]["default"] == 1000


def test_grep_tool_exposes_pi_input_schema(tmp_path):
    tools = FileTools(workspace_root=tmp_path)
    parameters = _parameters(tools.grep)
    properties = parameters["properties"]

    assert parameters["additionalProperties"] is False
    assert set(properties) == {
        "pattern",
        "path",
        "glob",
        "ignoreCase",
        "literal",
        "context",
        "limit",
    }
    assert parameters["required"] == ["pattern"]
    assert properties["path"]["type"] == "string"
    assert properties["glob"]["type"] == "string"
    assert properties["ignoreCase"]["default"] is False
    assert properties["literal"]["default"] is False
    assert properties["context"]["default"] == 0
    assert properties["limit"]["default"] == 100


@pytest.mark.asyncio
async def test_grep_pi_arguments_work_through_tool_call_layer(tmp_path):
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
        ignoreCase=True,
        context=1,
        glob="*.py",
    )
    assert "a.py-1- before" in output
    assert "a.py:2: Alpha" in output
    assert "a.py-3- After" in output
    assert "b.rs" not in output

    default_output = await succeeded(tools.grep, pattern="Alpha", path="docs")
    assert "a.py:2: Alpha" in default_output
    assert "b.rs:1: Alpha rust" in default_output

    brace_glob = await succeeded(
        tools.grep, pattern="Alpha", path="docs", glob="*.{py,rs}"
    )
    assert "a.py" in brace_glob
    assert "b.rs" in brace_glob
    assert "c.txt" not in brace_glob


@pytest.mark.asyncio
async def test_grep_literal_limit_and_no_match_messages(tmp_path):
    (tmp_path / "notes.txt").write_text("a.b\naxb\na.b\n", encoding="utf-8")
    tools = FileTools(workspace_root=tmp_path)

    output = await succeeded(
        tools.grep,
        pattern="a.b",
        path="notes.txt",
        literal=True,
        limit=1,
    )
    assert output.startswith("notes.txt:1: a.b")
    assert "1 matches limit reached" in output
    assert await succeeded(tools.grep, pattern="missing") == "No matches found"


@pytest.mark.asyncio
async def test_glob_and_grep_respect_gitignore_but_include_unignored_hidden_files(
    tmp_path,
):
    (tmp_path / ".gitignore").write_text(".venv/\n", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "ignored.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "visible.py").write_text("needle\n", encoding="utf-8")
    tools = FileTools(workspace_root=tmp_path)

    glob_output = await succeeded(tools.glob, pattern="**/*.py")
    grep_output = await succeeded(tools.grep, pattern="needle", glob="*.py")

    assert ".hidden/visible.py" in glob_output
    assert ".venv/ignored.py" not in glob_output
    assert ".hidden/visible.py:1: needle" in grep_output
    assert ".venv/ignored.py" not in grep_output


@pytest.mark.asyncio
async def test_glob_supports_directory_prefixed_patterns_and_limits_results(tmp_path):
    target = tmp_path / "src" / "nested" / "example.spec.py"
    target.parent.mkdir(parents=True)
    target.write_text("pass\n", encoding="utf-8")
    (tmp_path / "src" / "direct.spec.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("pass\n", encoding="utf-8")
    tools = FileTools(workspace_root=tmp_path)

    assert set(
        (await succeeded(tools.glob, pattern="src/**/*.spec.py")).splitlines()
    ) == {"src/direct.spec.py", "src/nested/example.spec.py"}
    assert await succeeded(tools.glob, pattern="src/*.spec.py") == (
        "src/direct.spec.py"
    )
    assert "other.py" in await succeeded(tools.glob, pattern="**/*.py")
    limited = await succeeded(tools.glob, pattern="**/*.py", limit=1)
    assert "1 results limit reached" in limited


@pytest.mark.asyncio
async def test_nested_gitignore_rules_do_not_leak_into_siblings(tmp_path):
    for directory in (tmp_path / "a", tmp_path / "b"):
        directory.mkdir()
        (directory / "ignored.txt").write_text("needle\n", encoding="utf-8")
        (directory / "kept.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / "a" / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    tools = FileTools(workspace_root=tmp_path)

    paths = set((await succeeded(tools.glob, pattern="**/*.txt")).splitlines())
    grep_output = await succeeded(tools.grep, pattern="needle", glob="*.txt")

    assert paths == {"a/kept.txt", "b/ignored.txt", "b/kept.txt"}
    assert "a/ignored.txt" not in grep_output
    assert "b/ignored.txt:1: needle" in grep_output


@pytest.mark.asyncio
async def test_glob_and_grep_fall_back_without_external_search_binaries(
    tmp_path, monkeypatch
):
    (tmp_path / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    (tmp_path / "ignored").mkdir()
    (tmp_path / "ignored" / "hidden.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "visible.py").write_text("Needle\n", encoding="utf-8")
    monkeypatch.setattr(file_module, "_search_executable", lambda *_names: None)
    tools = FileTools(workspace_root=tmp_path)

    assert await succeeded(tools.glob, pattern="**/*.py") == "src/visible.py"
    assert (
        await succeeded(tools.grep, pattern="needle", ignoreCase=True, glob="*.py")
        == "src/visible.py:1: Needle"
    )


@pytest.mark.asyncio
async def test_grep_invalid_regex_is_a_failed_tool_result(tmp_path):
    (tmp_path / "example.txt").write_text("text\n", encoding="utf-8")
    tools = FileTools(workspace_root=tmp_path)

    result = await invoke_tool(tools.grep, {"pattern": "["})

    assert result.status == "failed"
    assert result.error_code == "search_backend_failed"


@pytest.mark.asyncio
async def test_cancelled_grep_terminates_its_owned_process(tmp_path, monkeypatch):
    if file_module._search_executable("rg") is None:
        pytest.skip("ripgrep is required to exercise subprocess cancellation")
    (tmp_path / "example.txt").write_text("text\n", encoding="utf-8")
    started: list[asyncio.subprocess.Process] = []

    async def start_slow_process(*_args, **_kwargs):
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        started.append(process)
        return process

    monkeypatch.setattr(
        FileTools, "_start_search_process", staticmethod(start_slow_process)
    )
    tools = FileTools(workspace_root=tmp_path)
    invocation = asyncio.create_task(tools.grep("text"))
    for _ in range(100):
        if started:
            break
        await asyncio.sleep(0.01)
    assert started

    invocation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await invocation

    assert started[0].returncode is not None


@pytest.mark.asyncio
async def test_search_process_cleanup_drains_full_stdout_pipe():
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        (
            "import os, time\n"
            "chunk = b'x' * 65536\n"
            "for _ in range(64):\n"
            "    os.write(1, chunk)\n"
            "time.sleep(30)\n"
        ),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stderr is not None
    stderr_task = asyncio.create_task(process.stderr.read())
    await asyncio.sleep(0.1)

    await asyncio.wait_for(
        file_module._terminate_search_process(process, stderr_task), timeout=2
    )

    assert process.returncode is not None
    assert stderr_task.done()


@pytest.mark.asyncio
async def test_grep_result_limit_drains_buffered_backend_output(tmp_path, monkeypatch):
    (tmp_path / "example.txt").write_text("text\n", encoding="utf-8")
    monkeypatch.setattr(file_module, "_search_executable", lambda *_names: "rg")
    event = json.dumps(
        {
            "type": "match",
            "data": {
                "path": {"text": "example.txt"},
                "line_number": 1,
                "lines": {"text": "text\n"},
            },
        }
    )
    started: list[asyncio.subprocess.Process] = []

    async def start_noisy_process(*_args, **_kwargs):
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            (
                "import sys\n"
                f"line = {event + chr(10)!r}\n"
                "while True:\n"
                "    sys.stdout.write(line)\n"
                "    sys.stdout.flush()\n"
            ),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        started.append(process)
        await asyncio.sleep(0.1)
        return process

    monkeypatch.setattr(
        FileTools, "_start_search_process", staticmethod(start_noisy_process)
    )
    tools = FileTools(workspace_root=tmp_path)

    output = await asyncio.wait_for(tools.grep("text", limit=1), timeout=2)

    assert output.startswith("example.txt:1: text")
    assert "1 matches limit reached" in output
    assert started[0].returncode is not None


@pytest.mark.asyncio
async def test_cancelled_grep_fallback_joins_its_worker_thread(tmp_path, monkeypatch):
    (tmp_path / "example.txt").write_text("text\n", encoding="utf-8")
    monkeypatch.setattr(file_module, "_search_executable", lambda *_names: None)
    started = threading.Event()
    stopped = threading.Event()
    tools = FileTools(workspace_root=tmp_path)

    def slow_fallback(*args):
        cancelled = args[-1]
        started.set()
        cancelled.wait(30)
        stopped.set()
        return "No matches found"

    monkeypatch.setattr(tools, "_grep_fallback", slow_fallback)
    invocation = asyncio.create_task(tools.grep("text"))
    assert await asyncio.to_thread(started.wait, 1)

    invocation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(invocation, timeout=1)

    assert stopped.is_set()


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
    assert specs["glob"].version == "3.0.0"
    assert specs["grep"].version == "3.0.0"
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
