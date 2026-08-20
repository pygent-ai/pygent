"""Workspace file adapters expressed through the Pygent 0.2 Tool contract."""

from __future__ import annotations

import asyncio
import fnmatch
import json
import os
import re
import shutil
import tempfile
import threading
from contextlib import suppress
from itertools import islice
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Never

from pydantic import Field
from pypdf import PdfReader

from pygent.tool.executors import ToolExecutionError
from pygent.tool.functional import tool
from pygent.tool.types import IdempotencyPolicy, ToolSideEffect

from ._file_services import (
    FileDiagnosticsService,
    FileIOService,
    FileSearchService,
    NotebookService,
)
from ._paths import (
    ToolPathContext,
    is_absolute_tool_path,
    resolve_dir_path,
    resolve_file_path,
    resolve_tool_path,
)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico"}
_SEARCH_MAX_BYTES = 50 * 1024
_GREP_MAX_LINE_LENGTH = 500
_DEFAULT_SEARCH_IGNORES = frozenset(
    {
        ".git",
        ".hg",
        ".lora",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".venv",
        "__pycache__",
        "node_modules",
        "venv",
    }
)


def _fail(
    message: str,
    code: str,
    *,
    committed: bool | None = False,
    retryable: bool = False,
) -> Never:
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


def _truncate_search_line(line: str) -> tuple[str, bool]:
    if len(line) <= _GREP_MAX_LINE_LENGTH:
        return line, False
    return f"{line[:_GREP_MAX_LINE_LENGTH]}... [truncated]", True


def _expand_search_glob(pattern: str) -> list[str]:
    match = re.search(r"\{([^{}]+)\}", pattern)
    if match is None:
        return [pattern]
    prefix, suffix = pattern[: match.start()], pattern[match.end() :]
    expanded: list[str] = []
    for option in match.group(1).split(","):
        expanded.extend(_expand_search_glob(prefix + option + suffix))
    return expanded


def _matches_search_glob(relative_path: str, pattern: str) -> bool:
    candidates: set[str] = set()
    pending = _expand_search_glob(pattern.replace("\\", "/").lstrip("/"))
    while pending:
        candidate = pending.pop()
        if candidate in candidates:
            continue
        candidates.add(candidate)
        marker = candidate.find("**/")
        if marker >= 0:
            pending.append(candidate[:marker] + candidate[marker + 3 :])
    path = PurePosixPath(relative_path)
    return any(path.match(candidate) for candidate in candidates)


def _truncate_search_output(lines: list[str]) -> tuple[str, bool]:
    selected: list[str] = []
    size = 0
    for line in lines:
        encoded_size = len(line.encode("utf-8")) + (1 if selected else 0)
        if size + encoded_size > _SEARCH_MAX_BYTES:
            return "\n".join(selected), True
        selected.append(line)
        size += encoded_size
    return "\n".join(selected), False


def _search_executable(*names: str) -> str | None:
    for name in names:
        executable = shutil.which(name)
        if executable:
            return executable
    return None


def _inside_git_tree(path: Path) -> bool:
    return any((parent / ".git").exists() for parent in (path, *path.parents))


def _gitignore_rules(path: Path) -> tuple[tuple[str, bool, bool], ...]:
    ignore_file = path / ".gitignore"
    try:
        lines = ignore_file.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ()
    rules: list[tuple[str, bool, bool]] = []
    for raw in lines:
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        negated = value.startswith("!")
        if negated:
            value = value[1:]
        value = value.replace("\\", "/").lstrip("/")
        directory_only = value.endswith("/")
        value = value.rstrip("/")
        if value:
            rules.append((value, negated, directory_only))
    return tuple(rules)


def _gitignore_matches(
    relative: str,
    *,
    is_dir: bool,
    rules: tuple[tuple[PurePosixPath, tuple[tuple[str, bool, bool], ...]], ...],
) -> bool:
    ignored = False
    candidate = PurePosixPath(relative)
    for base, entries in rules:
        try:
            scoped = candidate.relative_to(base) if base.parts else candidate
        except ValueError:
            continue
        for pattern, negated, directory_only in entries:
            if directory_only and not is_dir:
                continue
            if "/" in pattern:
                matched = scoped.match(pattern) or scoped.match(f"**/{pattern}")
            else:
                matched = any(fnmatch.fnmatchcase(part, pattern) for part in scoped.parts)
            if matched:
                ignored = not negated
    return ignored


async def _terminate_search_process(
    process: asyncio.subprocess.Process,
    stderr_task: asyncio.Task[bytes],
) -> None:
    """Kill a search backend, drain its pipes, and reap it in that order.

    On Windows, waiting for a subprocess that still has buffered PIPE output can
    block even after ``kill()``.  The caller owns stdout while ``stderr_task``
    owns stderr, so both readers must reach EOF before the process is reaped.
    """

    if process.returncode is None:
        with suppress(ProcessLookupError):
            process.kill()

    assert process.stdout is not None
    with suppress(OSError, RuntimeError):
        await process.stdout.read()

    try:
        await stderr_task
    except asyncio.CancelledError:
        # asyncio.run() cancels every task at shutdown, so the independently
        # owned stderr reader may already have been cancelled.  Resume draining
        # only after that reader has released StreamReader's single-reader lock.
        assert process.stderr is not None
        with suppress(OSError, RuntimeError):
            await process.stderr.read()

    with suppress(ProcessLookupError):
        await process.wait()


def _relative_search_path(candidate: Path, root: Path) -> str:
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        relative = Path(os.path.relpath(candidate, root))
    value = relative.as_posix()
    if candidate.is_dir() and not value.endswith("/"):
        value += "/"
    return value


def _walk_search_files(
    root: Path,
    workspace_root: Path,
    limit: int,
    cancelled: threading.Event | None = None,
) -> list[Path]:
    """Bounded fallback traversal with default and hierarchical git ignores."""

    try:
        root_relative = root.relative_to(workspace_root)
        rule_root = workspace_root
    except ValueError:
        root_relative = Path()
        rule_root = root

    inherited: list[
        tuple[PurePosixPath, tuple[tuple[str, bool, bool], ...]]
    ] = []
    current = rule_root
    relative = PurePosixPath()
    for part in root_relative.parts:
        rules = _gitignore_rules(current)
        if rules:
            inherited.append((relative, rules))
        current /= part
        relative /= part

    files: list[Path] = []

    def visit(
        directory: Path,
        relative_directory: PurePosixPath,
        parent_rules: tuple[
            tuple[PurePosixPath, tuple[tuple[str, bool, bool], ...]], ...
        ],
    ) -> None:
        if len(files) >= limit or (cancelled is not None and cancelled.is_set()):
            return
        local_rules = _gitignore_rules(directory)
        rules = parent_rules + (
            ((relative_directory, local_rules),) if local_rules else ()
        )
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError:
            return
        for entry in entries:
            if len(files) >= limit or (cancelled is not None and cancelled.is_set()):
                return
            entry_relative = relative_directory / entry.name
            entry_text = entry_relative.as_posix()
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
                is_file = entry.is_file(follow_symlinks=False)
            except OSError:
                continue
            if is_dir:
                if entry.name in _DEFAULT_SEARCH_IGNORES or _gitignore_matches(
                    entry_text, is_dir=True, rules=rules
                ):
                    continue
                visit(Path(entry.path), entry_relative, rules)
            elif is_file and not _gitignore_matches(
                entry_text, is_dir=False, rules=rules
            ):
                files.append(Path(entry.path))

    visit(root, PurePosixPath(root_relative.as_posix()), tuple(inherited))
    return files


async def _run_owned_thread(function, *args):
    """Do not return cancellation while a blocking adapter still owns resources."""

    operation = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(operation)
    except asyncio.CancelledError:
        with suppress(Exception):
            await operation
        raise


async def _run_cancellable_search_thread(function, *args):
    """Request cooperative stop and join a fallback search before cancellation."""

    cancelled = threading.Event()
    operation = asyncio.create_task(asyncio.to_thread(function, *args, cancelled))
    try:
        return await asyncio.shield(operation)
    except asyncio.CancelledError:
        cancelled.set()
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
        self._io_service = FileIOService(
            lambda *args: self._read(*args),
            lambda *args: self._write(*args),
            lambda *args: self._edit(*args),
        )
        self._notebook_service = NotebookService(
            lambda *args: self._edit_notebook(*args)
        )
        self._diagnostics_service = FileDiagnosticsService(
            lambda *args: self._read_lints(*args)
        )
        self._search_service = FileSearchService(
            lambda *args: self._glob(*args), lambda *args: self._grep(*args)
        )

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

        return await _run_owned_thread(
            self._io_service.read, file_path, limit, offset, pages
        )

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

        return await _run_owned_thread(self._io_service.write, file_path, content)

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
            self._io_service.edit, file_path, old_string, new_string, replace_all
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
            self._notebook_service.edit,
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

        return await _run_owned_thread(self._diagnostics_service.read_lints, paths)

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
        version="3.0.0",
        side_effect=ToolSideEffect.READ,
        timeout=30,
        resource_key="filesystem",
        sandbox_profile="workspace",
        required_permissions=("filesystem:read",),
    )
    async def glob(
        self,
        pattern: Annotated[
            str,
            Field(
                description=(
                    "Glob pattern to match files, e.g. '*.ts', '**/*.json', "
                    "or 'src/**/*.spec.ts'"
                )
            ),
        ],
        path: Annotated[
            str,
            Field(description="Directory to search in (default: current directory)"),
        ] = "",
        limit: Annotated[
            int,
            Field(gt=0, description="Maximum number of results (default: 1000)"),
        ] = 1000,
    ) -> str:
        """Search for files by glob pattern while respecting ignore files."""

        return await self._search_service.glob(pattern, path, limit)

    async def _glob(self, pattern: str, path: str | None, limit: int) -> str:
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

        effective_limit = min(limit, self.max_search_files)
        executable = _search_executable("fd", "fdfind")
        process_cwd: Path | None = None
        if executable:
            arguments = ["--glob", "--color=never", "--hidden"]
            for ignored in sorted(_DEFAULT_SEARCH_IGNORES):
                arguments.extend(("--exclude", ignored))
            if not _inside_git_tree(root):
                arguments.append("--no-require-git")
            arguments.extend(("--max-results", str(effective_limit)))
            effective_pattern = pattern
            if "/" in pattern:
                arguments.append("--full-path")
                if (
                    not pattern.startswith("/")
                    and not pattern.startswith("**/")
                    and pattern != "**"
                ):
                    effective_pattern = f"**/{pattern}"
                if os.name == "nt":
                    effective_pattern = effective_pattern.replace("/", "[/\\\\]")
            arguments.extend(("--", effective_pattern, str(root)))
        else:
            executable = _search_executable("rg")
            if executable is None:
                return await _run_cancellable_search_thread(
                    self._glob_fallback, root, pattern, effective_limit
                )
            arguments = [
                "--files",
                "--hidden",
            ]
            for ignored in sorted(_DEFAULT_SEARCH_IGNORES):
                arguments.extend(("--glob", f"!**/{ignored}/**"))
            if not _inside_git_tree(root):
                arguments.append("--no-require-git")
            arguments.extend(("--", "."))
            process_cwd = root

        process = await self._start_search_process(
            executable, arguments, cwd=process_cwd
        )
        assert process.stderr is not None
        stderr_task = asyncio.create_task(process.stderr.read())
        lines: list[str] = []
        reached_limit = False
        try:
            assert process.stdout is not None
            while raw_line := await process.stdout.readline():
                value = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                if not value:
                    continue
                candidate = Path(value)
                if not candidate.is_absolute():
                    candidate = root / candidate
                candidate = resolve_tool_path(str(candidate), self.path_context)
                relative = _relative_search_path(candidate, root)
                if not _matches_search_glob(relative.rstrip("/"), pattern):
                    continue
                lines.append(relative)
                if len(lines) >= effective_limit:
                    reached_limit = True
                    await _terminate_search_process(process, stderr_task)
                    break
            stderr = await stderr_task
            code = await process.wait()
        except BaseException:
            await _terminate_search_process(process, stderr_task)
            with suppress(asyncio.CancelledError):
                await stderr_task
            raise

        if not reached_limit and code not in (0, 1):
            self._raise_search_failure(stderr, code, "file search backend")
        if not lines:
            return "No files found matching pattern"
        output, bytes_truncated = _truncate_search_output(lines)
        notices = []
        if reached_limit:
            notices.append(
                f"{effective_limit} results limit reached. Use limit="
                f"{effective_limit * 2} for more, or refine pattern"
            )
        if bytes_truncated:
            notices.append("50KB limit reached")
        return output + (f"\n\n[{'. '.join(notices)}]" if notices else "")

    def _glob_fallback(
        self,
        root: Path,
        pattern: str,
        limit: int,
        cancelled: threading.Event | None = None,
    ) -> str:
        candidates = _walk_search_files(
            root, self.workspace_root, self.max_search_files, cancelled
        )
        lines = []
        for candidate in candidates:
            if cancelled is not None and cancelled.is_set():
                break
            relative = _relative_search_path(candidate, root)
            if _matches_search_glob(relative.rstrip("/"), pattern):
                lines.append(relative)
        reached_limit = len(lines) > limit
        lines = lines[:limit]
        if not lines:
            return "No files found matching pattern"
        output, bytes_truncated = _truncate_search_output(lines)
        notices = []
        if reached_limit:
            notices.append(
                f"{limit} results limit reached. Use limit={limit * 2} for more, "
                "or refine pattern"
            )
        if bytes_truncated:
            notices.append("50KB limit reached")
        return output + (f"\n\n[{'. '.join(notices)}]" if notices else "")

    @tool(
        tool_id="standard.files.grep",
        version="3.0.0",
        side_effect=ToolSideEffect.READ,
        timeout=30,
        resource_key="filesystem",
        sandbox_profile="workspace",
        required_permissions=("filesystem:read",),
    )
    async def grep(
        self,
        pattern: Annotated[
            str, Field(description="Search pattern (regex or literal string)")
        ],
        path: Annotated[
            str,
            Field(description="Directory or file to search (default: current directory)"),
        ] = "",
        glob: Annotated[
            str,
            Field(
                description="Filter files by glob pattern, e.g. '*.ts' or '**/*.spec.ts'"
            ),
        ] = "",
        ignoreCase: Annotated[
            bool, Field(description="Case-insensitive search (default: false)")
        ] = False,
        literal: Annotated[
            bool,
            Field(
                description="Treat pattern as literal string instead of regex (default: false)"
            ),
        ] = False,
        context: Annotated[
            int,
            Field(
                ge=0,
                description="Number of lines to show before and after each match (default: 0)",
            ),
        ] = 0,
        limit: Annotated[
            int,
            Field(gt=0, description="Maximum number of matches to return (default: 100)"),
        ] = 100,
    ) -> str:
        """Search file contents with ripgrep while respecting ignore files."""

        return await self._search_service.grep(
            pattern, path, glob, ignoreCase, literal, context, limit
        )

    async def _grep(
        self,
        pattern: str,
        path: str | None,
        glob: str | None,
        ignore_case: bool,
        literal: bool,
        context: int,
        limit: int,
    ) -> str:
        root = resolve_tool_path(path, self.path_context, default=".")
        if not root.exists():
            _fail(f"path does not exist: {root}", "file_not_found")
        executable = _search_executable("rg")
        if executable is None:
            return await _run_cancellable_search_thread(
                self._grep_fallback,
                root,
                pattern,
                glob,
                ignore_case,
                literal,
                context,
                min(limit, self.max_search_files),
            )
        effective_limit = min(limit, self.max_search_files)
        arguments = ["--json", "--line-number", "--color=never", "--hidden"]
        for ignored in sorted(_DEFAULT_SEARCH_IGNORES):
            arguments.extend(("--glob", f"!**/{ignored}/**"))
        if ignore_case:
            arguments.append("--ignore-case")
        if literal:
            arguments.append("--fixed-strings")
        if not _inside_git_tree(root if root.is_dir() else root.parent):
            arguments.append("--no-require-git")
        process_cwd = root if root.is_dir() else root.parent
        search_target = "." if root.is_dir() else root.name
        arguments.extend(("--", pattern, search_target))

        process = await self._start_search_process(
            executable, arguments, cwd=process_cwd
        )
        assert process.stderr is not None
        stderr_task = asyncio.create_task(process.stderr.read())
        matches: list[tuple[Path, int, str]] = []
        reached_limit = False
        try:
            assert process.stdout is not None
            while raw_line := await process.stdout.readline():
                try:
                    event = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if event.get("type") != "match":
                    continue
                data = event.get("data", {})
                raw_path = data.get("path", {}).get("text")
                line_number = data.get("line_number")
                line_text = data.get("lines", {}).get("text")
                if not isinstance(raw_path, str) or not isinstance(line_number, int):
                    continue
                candidate = Path(raw_path)
                if not candidate.is_absolute():
                    candidate = root.parent / candidate if root.is_file() else root / candidate
                candidate = resolve_file_path(str(candidate), self.path_context)
                search_root = root if root.is_dir() else root.parent
                relative = _relative_search_path(candidate, search_root)
                if glob and not _matches_search_glob(relative, glob):
                    continue
                matches.append(
                    (candidate, line_number, line_text if isinstance(line_text, str) else "")
                )
                if len(matches) >= effective_limit:
                    reached_limit = True
                    await _terminate_search_process(process, stderr_task)
                    break
            stderr = await stderr_task
            code = await process.wait()
        except BaseException:
            await _terminate_search_process(process, stderr_task)
            with suppress(asyncio.CancelledError):
                await stderr_task
            raise

        if not reached_limit and code not in (0, 1):
            self._raise_search_failure(stderr, code, "ripgrep")
        if not matches:
            return "No matches found"

        file_cache: dict[Path, list[str]] = {}
        output_lines: list[str] = []
        lines_truncated = False
        search_root = root if root.is_dir() else root.parent
        for file_path, line_number, matched_text in matches:
            relative = _relative_search_path(file_path, search_root)
            if context == 0:
                value = matched_text.replace("\r\n", "\n").replace("\r", "")
                value = value.removesuffix("\n")
                value, truncated = _truncate_search_line(value)
                lines_truncated |= truncated
                output_lines.append(f"{relative}:{line_number}: {value}")
                continue
            lines = file_cache.get(file_path)
            if lines is None:
                try:
                    text = file_path.read_text(encoding="utf-8", errors="replace")
                    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
                except OSError:
                    lines = []
                file_cache[file_path] = lines
            if not lines:
                output_lines.append(f"{relative}:{line_number}: (unable to read file)")
                continue
            start = max(1, line_number - context)
            end = min(len(lines), line_number + context)
            for current in range(start, end + 1):
                value, truncated = _truncate_search_line(lines[current - 1])
                lines_truncated |= truncated
                separator = ":" if current == line_number else "-"
                output_lines.append(f"{relative}{separator}{current}{separator} {value}")

        output, bytes_truncated = _truncate_search_output(output_lines)
        notices = []
        if reached_limit:
            notices.append(
                f"{effective_limit} matches limit reached. Use limit="
                f"{effective_limit * 2} for more, or refine pattern"
            )
        if bytes_truncated:
            notices.append("50KB limit reached")
        if lines_truncated:
            notices.append(
                "Some lines truncated to 500 chars. Use read tool to see full lines"
            )
        return output + (f"\n\n[{'. '.join(notices)}]" if notices else "")

    def _grep_fallback(
        self,
        root: Path,
        pattern: str,
        glob: str | None,
        ignore_case: bool,
        literal: bool,
        context: int,
        limit: int,
        cancelled: threading.Event | None = None,
    ) -> str:
        flags = re.IGNORECASE if ignore_case else 0
        try:
            matcher = re.compile(re.escape(pattern) if literal else pattern, flags)
        except re.error as exc:
            raise ToolExecutionError(
                f"invalid search pattern: {exc}",
                kind="filesystem_error",
                code="search_backend_failed",
                side_effect_committed=False,
            ) from exc

        search_root = root if root.is_dir() else root.parent
        candidates = (
            _walk_search_files(
                root, self.workspace_root, self.max_search_files, cancelled
            )
            if root.is_dir()
            else [root]
        )
        matches: list[tuple[Path, int, str, list[str]]] = []
        reached_limit = False
        for candidate in candidates:
            if cancelled is not None and cancelled.is_set():
                break
            relative = _relative_search_path(candidate, search_root)
            if glob and not _matches_search_glob(relative, glob):
                continue
            try:
                data = candidate.read_bytes()
            except OSError:
                continue
            if b"\x00" in data[:8192]:
                continue
            lines = data.decode("utf-8", errors="replace").replace(
                "\r\n", "\n"
            ).replace("\r", "\n").split("\n")
            for line_number, line in enumerate(lines, start=1):
                if cancelled is not None and cancelled.is_set():
                    break
                if matcher.search(line) is None:
                    continue
                matches.append((candidate, line_number, line, lines))
                if len(matches) >= limit:
                    reached_limit = True
                    break
            if reached_limit:
                break
        if not matches:
            return "No matches found"

        output_lines: list[str] = []
        lines_truncated = False
        for file_path, line_number, matched_text, lines in matches:
            relative = _relative_search_path(file_path, search_root)
            if context == 0:
                value, truncated = _truncate_search_line(matched_text)
                lines_truncated |= truncated
                output_lines.append(f"{relative}:{line_number}: {value}")
                continue
            start = max(1, line_number - context)
            end = min(len(lines), line_number + context)
            for current in range(start, end + 1):
                value, truncated = _truncate_search_line(lines[current - 1])
                lines_truncated |= truncated
                separator = ":" if current == line_number else "-"
                output_lines.append(f"{relative}{separator}{current}{separator} {value}")

        output, bytes_truncated = _truncate_search_output(output_lines)
        notices = []
        if reached_limit:
            notices.append(
                f"{limit} matches limit reached. Use limit={limit * 2} for more, "
                "or refine pattern"
            )
        if lines_truncated:
            notices.append("long lines truncated to 500 characters")
        if bytes_truncated:
            notices.append("50KB limit reached")
        return output + (f"\n\n[{'. '.join(notices)}]" if notices else "")

    @staticmethod
    async def _start_search_process(
        executable: str,
        arguments: list[str],
        *,
        cwd: Path | None = None,
    ) -> asyncio.subprocess.Process:
        try:
            return await asyncio.create_subprocess_exec(
                executable,
                *arguments,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
        except OSError as exc:
            raise ToolExecutionError(
                "could not start search backend",
                kind="filesystem_error",
                code="search_backend_failed",
                retryable=True,
                side_effect_committed=False,
            ) from exc

    @staticmethod
    def _raise_search_failure(stderr: bytes, code: int, backend: str) -> None:
        message = stderr.decode("utf-8", errors="replace").strip()
        raise ToolExecutionError(
            message or f"{backend} exited with code {code}",
            kind="filesystem_error",
            code="search_backend_failed",
            retryable=True,
            side_effect_committed=False,
        )


__all__ = ["FileTools"]
