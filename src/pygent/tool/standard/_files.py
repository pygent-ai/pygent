"""Workspace file adapters expressed through the Pygent 0.2 Tool contract."""

from __future__ import annotations

import asyncio
import fnmatch
import json
import os
import re
import tempfile
import threading
from contextlib import suppress
from itertools import islice
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field

from pygent.tool.executors import ToolExecutionError
from pygent.tool.functional import tool
from pygent.tool.types import IdempotencyPolicy, ToolSideEffect

from ._paths import (
    ToolPathContext,
    is_absolute_tool_path,
    resolve_dir_path,
    resolve_file_path,
    resolve_tool_path,
)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico"}
_GREP_TYPE_SUFFIXES = {
    "c": {".c", ".h"},
    "cpp": {".cc", ".cpp", ".cxx", ".hh", ".hpp", ".hxx"},
    "css": {".css"},
    "go": {".go"},
    "html": {".htm", ".html"},
    "java": {".java"},
    "js": {".cjs", ".js", ".jsx", ".mjs"},
    "json": {".json"},
    "md": {".md", ".markdown"},
    "py": {".py", ".pyw"},
    "rust": {".rs"},
    "rs": {".rs"},
    "ts": {".ts", ".tsx"},
    "txt": {".txt"},
    "yaml": {".yaml", ".yml"},
}


def _fail(
    message: str,
    code: str,
    *,
    committed: bool | None = False,
    retryable: bool = False,
) -> None:
    raise ToolExecutionError(
        message,
        kind="filesystem_error",
        code=code,
        retryable=retryable,
        side_effect_committed=committed,
    )


def _format_file_text(text: str, offset: int | None, limit: int | None) -> str:
    lines = text.splitlines(keepends=True)
    start_line = offset or 1
    start = start_line - 1
    selected = lines[start : start + limit] if limit is not None else lines[start:]
    return "".join(
        f"{index + start_line}|{line}" for index, line in enumerate(selected)
    )


def _parse_pdf_page_range(pages: str | None, total_pages: int) -> list[int]:
    if total_pages <= 0:
        return []
    if not pages:
        return list(range(min(total_pages, 20)))
    page_spec = pages.strip()
    if not page_spec:
        _fail("pages must not be empty", "invalid_page_range")
    try:
        if "-" in page_spec:
            start_text, end_text = page_spec.split("-", 1)
            start_page, end_page = int(start_text.strip()), int(end_text.strip())
        else:
            start_page = end_page = int(page_spec)
    except ValueError:
        _fail("pages must be a page number or range such as 1-5", "invalid_page_range")
    if start_page < 1 or end_page < start_page:
        _fail("PDF page ranges are one-based and increasing", "invalid_page_range")
    if end_page - start_page + 1 > 20:
        _fail("PDF reads are limited to 20 pages", "page_limit_exceeded")
    if start_page > total_pages:
        _fail(f"start page exceeds PDF page count {total_pages}", "page_out_of_range")
    return list(range(start_page - 1, min(end_page, total_pages)))


def _read_pdf_text(path: Path, pages: str | None) -> str:
    try:
        try:
            from pypdf import PdfReader  # type: ignore[import-not-found]
        except ImportError:
            from PyPDF2 import PdfReader  # type: ignore[import-not-found,no-redef]
    except ImportError:
        _fail("reading PDFs requires pypdf or PyPDF2", "pdf_backend_missing")
    try:
        reader = PdfReader(str(path))
        output = []
        for page_index in _parse_pdf_page_range(pages, len(reader.pages)):
            text = reader.pages[page_index].extract_text() or ""
            output.append(f"--- page {page_index + 1} ---\n{text}".rstrip())
        return "\n\n".join(output)
    except ToolExecutionError:
        raise
    except Exception as exc:
        raise ToolExecutionError(
            "PDF extraction failed",
            kind="filesystem_error",
            code="pdf_read_failed",
            side_effect_committed=False,
        ) from exc


def _expand_brace_glob(pattern: str) -> list[str]:
    match = re.search(r"\{([^{}]+)\}", pattern)
    if match is None:
        return [pattern]
    prefix, suffix = pattern[: match.start()], pattern[match.end() :]
    expanded: list[str] = []
    for option in match.group(1).split(","):
        expanded.extend(_expand_brace_glob(prefix + option + suffix))
    return expanded


def _matches_glob(path: Path, root: Path, glob_pattern: str) -> bool:
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        relative = path.name
    patterns = _expand_brace_glob(glob_pattern.lstrip("/"))
    return any(
        fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(path.name, pattern)
        for pattern in patterns
    )


def _file_type_matches(path: Path, file_type: str) -> bool:
    normalized = file_type.lower().lstrip(".")
    return path.suffix.lower() in _GREP_TYPE_SUFFIXES.get(
        normalized, {"." + normalized}
    )


async def _run_owned_thread(function, *args):
    """Do not return cancellation while a blocking adapter still owns resources."""

    operation = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(operation)
    except asyncio.CancelledError:
        with suppress(Exception):
            await operation
        raise


def _atomic_write_text(path: Path, content: str) -> None:
    """Commit complete UTF-8 content with one same-directory replace."""

    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".pygent-tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


class FileTools:
    """Deployment-local workspace file handlers.

    The instance and its workspace configuration stay in the executor registry;
    only projections created by :class:`ToolKit` are portable.
    """

    def __init__(
        self,
        *,
        workspace_root: str | Path,
        restrict_to_workspace: bool = True,
        max_read_bytes: int = 1024 * 1024,
        max_search_files: int = 10_000,
    ) -> None:
        if max_read_bytes <= 0 or max_search_files <= 0:
            raise ValueError("file tool limits must be positive")
        self.path_context = ToolPathContext.from_workspace_root(
            workspace_root, restrict_to_workspace=restrict_to_workspace
        )
        self.workspace_root = self.path_context.workspace_root
        self.max_read_bytes = max_read_bytes
        self.max_search_files = max_search_files
        self._mutation_locks = tuple(threading.Lock() for _ in range(64))

    def _mutation_lock(self, path: Path) -> threading.Lock:
        return self._mutation_locks[hash(path) % len(self._mutation_locks)]

    @property
    def handlers(self) -> tuple[Any, ...]:
        """Return handlers in the stable model-visible order used by 0.1.15."""

        return (
            self.edit,
            self.edit_notebook,
            self.glob,
            self.grep,
            self.read,
            self.read_lints,
            self.write,
        )

    @tool(
        tool_id="standard.files.read",
        version="2.0.0",
        side_effect=ToolSideEffect.READ,
        timeout=30,
        resource_key="filesystem",
        sandbox_profile="workspace",
        required_permissions=("filesystem:read",),
    )
    async def read(
        self,
        file_path: Annotated[
            str,
            Field(
                description="File path resolved from workspace_root and restricted to it by default."
            ),
        ],
        limit: Annotated[int | None, Field(gt=0)] = None,
        offset: Annotated[int | None, Field(gt=0)] = None,
        pages: str | None = None,
    ) -> str:
        """Read text ranges, optional PDF pages, or a bounded binary description."""

        return await asyncio.to_thread(self._read, file_path, limit, offset, pages)

    def _read(
        self, file_path: str, limit: int | None, offset: int | None, pages: str | None
    ) -> str:
        path = resolve_file_path(file_path, self.path_context)
        if not path.exists():
            _fail(f"file does not exist: {path}", "file_not_found")
        if not path.is_file():
            _fail(f"path is not a file: {path}", "not_a_file")
        if path.suffix.lower() == ".pdf":
            return _read_pdf_text(path, pages)
        if pages:
            _fail("pages applies only to PDF files", "invalid_page_range")
        try:
            with path.open("rb") as stream:
                raw = stream.read(self.max_read_bytes + 1)
        except OSError as exc:
            raise ToolExecutionError(
                f"could not read file: {path}",
                kind="filesystem_error",
                code="read_failed",
                retryable=True,
                side_effect_committed=False,
            ) from exc
        truncated = len(raw) > self.max_read_bytes
        raw = raw[: self.max_read_bytes]
        text_bytes = set(range(0x20, 0x100)) | {7, 8, 9, 10, 12, 13, 27}
        if not raw or all(byte in text_bytes for byte in raw):
            text = raw.decode("utf-8", errors="replace")
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            output = _format_file_text(text, offset, limit)
            if truncated:
                output += f"\n[read truncated to {self.max_read_bytes} bytes]"
            return output
        label = "image" if path.suffix.lower() in _IMAGE_SUFFIXES else "binary"
        return f"[{label} file {path.name}, size {path.stat().st_size} bytes]"

    @tool(
        tool_id="standard.files.write",
        version="2.0.0",
        side_effect=ToolSideEffect.WRITE,
        idempotency=IdempotencyPolicy.INHERENT,
        timeout=30,
        resource_key="filesystem",
        sandbox_profile="workspace",
        required_permissions=("filesystem:write",),
    )
    async def write(
        self,
        file_path: Annotated[
            str,
            Field(
                description="Destination resolved from workspace_root and restricted to it by default."
            ),
        ],
        content: str,
    ) -> str:
        """Replace a UTF-8 file atomically enough for a single local process."""

        return await _run_owned_thread(self._write, file_path, content)

    def _write(self, file_path: str, content: str) -> str:
        path = resolve_file_path(file_path, self.path_context)
        if path.exists() and path.is_dir():
            _fail(f"file_path points to a directory: {path}", "not_a_file")
        try:
            with self._mutation_lock(path):
                path.parent.mkdir(parents=True, exist_ok=True)
                _atomic_write_text(path, content)
        except OSError as exc:
            raise ToolExecutionError(
                f"could not write file: {path}",
                kind="filesystem_error",
                code="write_failed",
                retryable=True,
                side_effect_committed=None,
            ) from exc
        return "写入完成"

    @tool(
        tool_id="standard.files.edit",
        version="2.0.0",
        side_effect=ToolSideEffect.WRITE,
        idempotency=IdempotencyPolicy.NOT_IDEMPOTENT,
        timeout=30,
        resource_key="filesystem",
        sandbox_profile="workspace",
        required_permissions=("filesystem:write",),
    )
    async def edit(
        self,
        file_path: Annotated[
            str,
            Field(
                description="File resolved from workspace_root and restricted to it by default."
            ),
        ],
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        """Replace exact text once, or all occurrences when explicitly requested."""

        return await _run_owned_thread(
            self._edit, file_path, old_string, new_string, replace_all
        )

    def _edit(
        self, file_path: str, old_string: str, new_string: str, replace_all: bool
    ) -> str:
        if old_string == new_string:
            _fail("new_string must differ from old_string", "identical_replacement")
        path = resolve_file_path(file_path, self.path_context)
        if not path.exists() or not path.is_file():
            _fail(f"file does not exist: {path}", "file_not_found")
        try:
            with self._mutation_lock(path):
                text = path.read_text(encoding="utf-8")
                if old_string not in text:
                    _fail("exact old_string was not found", "match_not_found")
                updated = text.replace(
                    old_string, new_string, -1 if replace_all else 1
                )
                _atomic_write_text(path, updated)
        except ToolExecutionError:
            raise
        except OSError as exc:
            raise ToolExecutionError(
                f"could not edit file: {path}",
                kind="filesystem_error",
                code="edit_failed",
                retryable=True,
                side_effect_committed=None,
            ) from exc
        return "替换完成"

    @tool(
        tool_id="standard.files.edit_notebook",
        version="2.0.0",
        side_effect=ToolSideEffect.WRITE,
        idempotency=IdempotencyPolicy.NOT_IDEMPOTENT,
        timeout=30,
        resource_key="filesystem",
        sandbox_profile="workspace",
        required_permissions=("filesystem:write",),
    )
    async def edit_notebook(
        self,
        target_notebook: str,
        cell_idx: Annotated[int, Field(ge=0)],
        is_new_cell: bool,
        cell_language: str,
        old_string: str,
        new_string: str,
    ) -> str:
        """Insert or edit one Jupyter notebook cell."""

        return await _run_owned_thread(
            self._edit_notebook,
            target_notebook,
            cell_idx,
            is_new_cell,
            cell_language,
            old_string,
            new_string,
        )

    def _edit_notebook(
        self,
        target_notebook: str,
        cell_idx: int,
        is_new_cell: bool,
        cell_language: str,
        old_string: str,
        new_string: str,
    ) -> str:
        path = resolve_file_path(target_notebook, self.path_context)
        if not path.exists() or not path.is_file():
            _fail(f"notebook does not exist: {path}", "file_not_found")
        try:
            with self._mutation_lock(path):
                notebook = json.loads(path.read_text(encoding="utf-8"))
                cells = notebook.get("cells")
                if not isinstance(cells, list):
                    _fail("notebook cells must be a list", "invalid_notebook")
                if is_new_cell:
                    if cell_idx > len(cells):
                        _fail(
                            f"cell index out of range 0..{len(cells)}",
                            "cell_index_out_of_range",
                        )
                    code_languages = {
                        "python",
                        "javascript",
                        "typescript",
                        "r",
                        "sql",
                        "shell",
                    }
                    cell_type = (
                        "code"
                        if cell_language in code_languages
                        else "markdown"
                        if cell_language == "markdown"
                        else "raw"
                    )
                    cells.insert(
                        cell_idx,
                        {
                            "cell_type": cell_type,
                            "metadata": {},
                            "source": new_string.splitlines(keepends=True),
                        },
                    )
                else:
                    if cell_idx >= len(cells):
                        _fail(
                            f"cell index out of range 0..{len(cells) - 1}",
                            "cell_index_out_of_range",
                        )
                    cell = cells[cell_idx]
                    source = cell.get("source", [])
                    content = (
                        "".join(source) if isinstance(source, list) else str(source)
                    )
                    if old_string not in content:
                        _fail("old_string was not found in the cell", "match_not_found")
                    cell["source"] = content.replace(
                        old_string, new_string, 1
                    ).splitlines(keepends=True)
                notebook["cells"] = cells
                _atomic_write_text(
                    path, json.dumps(notebook, ensure_ascii=False, indent=1)
                )
        except ToolExecutionError:
            raise
        except json.JSONDecodeError as exc:
            raise ToolExecutionError(
                "notebook is not valid JSON",
                kind="filesystem_error",
                code="invalid_notebook",
                side_effect_committed=False,
            ) from exc
        except OSError as exc:
            raise ToolExecutionError(
                f"could not update notebook: {path}",
                kind="filesystem_error",
                code="notebook_write_failed",
                side_effect_committed=None,
            ) from exc
        return "笔记本已更新"

    @tool(
        tool_id="standard.files.read_lints",
        version="2.0.0",
        side_effect=ToolSideEffect.READ,
        timeout=30,
        resource_key="filesystem",
        sandbox_profile="workspace",
        required_permissions=("filesystem:read",),
    )
    async def read_lints(self, paths: list[str] | None = None) -> str:
        """Return bounded Python syntax diagnostics for files or directories."""

        return await asyncio.to_thread(self._read_lints, paths)

    def _read_lints(self, paths: list[str] | None) -> str:
        requested = paths or ["."]
        files: list[Path] = []
        for value in requested:
            path = resolve_tool_path(value, self.path_context)
            if not path.exists():
                _fail(f"lint path does not exist: {path}", "file_not_found")
            if path.is_file() and path.suffix.lower() == ".py":
                files.append(path)
            elif path.is_dir():
                remaining = max(0, self.max_search_files - len(files))
                files.extend(islice(path.rglob("*.py"), remaining))
        diagnostics = []
        for path in sorted(set(files))[: self.max_search_files]:
            try:
                source = path.read_text(encoding="utf-8", errors="replace")
                compile(source, str(path), "exec")
            except SyntaxError as exc:
                diagnostics.append(
                    {
                        "path": str(path),
                        "line": exc.lineno,
                        "column": exc.offset,
                        "message": exc.msg,
                    }
                )
            except OSError:
                continue
        return json.dumps(
            {"tool": "python.compile", "diagnostics": diagnostics},
            ensure_ascii=False,
            sort_keys=True,
        )

    @tool(
        tool_id="standard.files.glob",
        version="2.0.0",
        side_effect=ToolSideEffect.READ,
        timeout=30,
        resource_key="filesystem",
        sandbox_profile="workspace",
        required_permissions=("filesystem:read",),
    )
    async def glob(self, pattern: str, path: str | None = None) -> str:
        """Find files by glob pattern, newest first.

        Args:
            pattern: Glob pattern such as ``*.py`` or ``**/*.ts``.
            path: Directory resolved from workspace_root; defaults to the workspace.
        """

        return await asyncio.to_thread(self._glob, pattern, path)

    def _glob(self, pattern: str, path: str | None) -> str:
        root = resolve_dir_path(path, self.path_context)
        if not root.exists():
            _fail(f"path does not exist: {root}", "file_not_found")
        if not root.is_dir():
            _fail(f"path must be a directory: {root}", "not_a_directory")
        pattern_parts = Path(pattern.replace("\\", "/")).parts
        if is_absolute_tool_path(pattern) or ".." in pattern_parts:
            _fail(
                "glob pattern must stay within workspace_root",
                "path_outside_workspace",
            )
        try:
            matches = []
            for candidate in root.glob(pattern.lstrip("/\\")):
                resolved = resolve_file_path(str(candidate), self.path_context)
                if resolved.is_file():
                    matches.append(resolved)
                if len(matches) >= self.max_search_files:
                    break
        except (OSError, ValueError) as exc:
            raise ToolExecutionError(
                "invalid glob pattern",
                kind="filesystem_error",
                code="invalid_glob",
                side_effect_committed=False,
            ) from exc

        def sort_key(candidate: Path) -> tuple[float, str]:
            try:
                modified = candidate.stat().st_mtime
            except OSError:
                modified = 0.0
            return -modified, str(candidate)

        return "\n".join(str(item) for item in sorted(matches, key=sort_key))

    @tool(
        tool_id="standard.files.grep",
        version="2.0.0",
        side_effect=ToolSideEffect.READ,
        timeout=30,
        resource_key="filesystem",
        sandbox_profile="workspace",
        required_permissions=("filesystem:read",),
    )
    async def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        output_mode: Literal["content", "files_with_matches", "count"] | None = None,
        context_before: Annotated[int | None, Field(ge=0)] = None,
        context_after: Annotated[int | None, Field(ge=0)] = None,
        context: Annotated[int | None, Field(ge=0)] = None,
        ignore_case: bool = False,
        file_type: str | None = None,
        head_limit: Annotated[int | None, Field(ge=0)] = None,
        offset: Annotated[int | None, Field(ge=0)] = None,
        multiline: bool = False,
        show_line_numbers: bool = True,
    ) -> str:
        """Search file text using a regular expression and bounded result modes."""

        return await asyncio.to_thread(
            self._grep,
            pattern,
            path,
            glob,
            output_mode,
            context_before,
            context_after,
            context,
            ignore_case,
            file_type,
            head_limit,
            offset,
            multiline,
            show_line_numbers,
        )

    def _grep(
        self,
        pattern: str,
        path: str | None,
        glob: str | None,
        output_mode: str | None,
        context_before: int | None,
        context_after: int | None,
        context: int | None,
        ignore_case: bool,
        file_type: str | None,
        head_limit: int | None,
        offset: int | None,
        multiline: bool,
        show_line_numbers: bool,
    ) -> str:
        mode = output_mode or "files_with_matches"
        root = resolve_tool_path(path, self.path_context, default=".")
        if not root.exists():
            _fail(f"path does not exist: {root}", "file_not_found")
        flags = re.IGNORECASE if ignore_case else 0
        if multiline:
            flags |= re.DOTALL | re.MULTILINE
        try:
            expression = re.compile(pattern, flags)
        except re.error:
            expression = re.compile(
                re.escape(pattern), re.IGNORECASE if ignore_case else 0
            )
        before = context_before if context_before is not None else context or 0
        after = context_after if context_after is not None else context or 0
        skip = offset or 0
        limit = 250 if head_limit is None else head_limit

        if root.is_file():
            files = [root]
        else:
            files = list(
                islice(
                    (item for item in root.rglob("*") if item.is_file()),
                    self.max_search_files,
                )
            )
            if glob:
                files = [item for item in files if _matches_glob(item, root, glob)]
            if file_type:
                files = [item for item in files if _file_type_matches(item, file_type)]

        content_lines: list[str] = []
        count_pairs: list[tuple[Path, int]] = []
        matched_files: set[Path] = set()
        hard_output_lines = 10_000

        def format_line(file_path: Path, line_no: int, line: str) -> str:
            body = f"{line_no}|{line}" if show_line_numbers else line
            return f"{file_path}:{body}"

        for file_path in sorted(files):
            try:
                raw = file_path.read_bytes()[: self.max_read_bytes]
                text = raw.decode("utf-8", errors="replace")
            except OSError:
                continue
            lines = text.splitlines()
            if multiline:
                matches = list(expression.finditer(text))
                if not matches:
                    continue
                matched_files.add(file_path)
                count_pairs.append((file_path, len(matches)))
                if mode == "content":
                    for match in matches:
                        start_line = text.count("\n", 0, match.start())
                        end_at = max(match.start(), match.end() - 1)
                        end_line = text.count("\n", 0, end_at)
                        start = max(0, start_line - before)
                        end = min(len(lines), end_line + after + 1)
                        remaining = hard_output_lines - len(content_lines)
                        if remaining > 0:
                            content_lines.extend(
                                format_line(file_path, index + 1, lines[index])
                                for index in range(start, min(end, start + remaining))
                            )
                continue
            match_count = 0
            for index, line in enumerate(lines):
                if expression.search(line) is None:
                    continue
                match_count += 1
                matched_files.add(file_path)
                if mode == "content":
                    start = max(0, index - before)
                    end = min(len(lines), index + after + 1)
                    remaining = hard_output_lines - len(content_lines)
                    if remaining > 0:
                        content_lines.extend(
                            format_line(file_path, line_index + 1, lines[line_index])
                            for line_index in range(start, min(end, start + remaining))
                        )
            if match_count:
                count_pairs.append((file_path, match_count))

        if mode == "count":
            if not count_pairs:
                return "0"
            selected = count_pairs[skip:]
            if limit:
                selected = selected[:limit]
            if len(count_pairs) == 1 and skip == 0 and (limit == 0 or limit >= 1):
                return str(count_pairs[0][1])
            return "\n".join(f"{file_path}:{count}" for file_path, count in selected)
        if mode == "files_with_matches":
            selected_paths = sorted(str(item) for item in matched_files)[skip:]
            if limit:
                selected_paths = selected_paths[:limit]
            return "\n".join(selected_paths)
        selected_lines = content_lines[skip:]
        if limit:
            selected_lines = selected_lines[:limit]
        return "\n".join(selected_lines) if selected_lines else "无匹配"


__all__ = ["FileTools"]
