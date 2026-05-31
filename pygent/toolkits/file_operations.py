"""
文件操作工具集。参考 docs/tools_openai_schema.json 实现所有文件相关工具。
"""
from __future__ import annotations

import json
import fnmatch
import os
import re
from pathlib import Path
from typing import Any, List, Optional

from pygent.common import PygentString
from pygent.module.tool import BaseTool, ToolErrorResult
from pygent.module.tool.utils import ToolClassBase, tool_method, tool_class
from pygent.toolkits.path_utils import (
    is_absolute_tool_path,
    normalize_desktop_path,
    normalize_tool_path,
)


def _normalize_desktop_path(path: str) -> str:
    """
    将大模型常见的「桌面」路径错误归一化为 ~/Desktop/xxx，兼容：
    - /Users/Desktop/xxx（Unix 风格，在 Windows 下会错误解析为 C:\\Users\\Desktop）
    - C:\\Users\\Desktop\\xxx（Windows 错误路径，实际桌面是 C:\\Users\\<用户名>\\Desktop）
    """
    s = str(path).strip().replace("\\", "/")
    # /Users/Desktop/xxx 或 /Users/Desktop -> ~/Desktop/xxx
    if s.startswith("/Users/Desktop") or s.lower().startswith("c:/users/desktop"):
        if s.lower().startswith("c:/users/desktop"):
            rest = s[17:].lstrip("/")  # len("c:/users/desktop") = 17
        else:
            rest = s[14:].lstrip("/")  # len("/Users/Desktop") = 14
        return "~/Desktop/" + rest if rest else "~/Desktop"
    return path.strip()


def _resolve_path(path: str, base: Optional[str] = None) -> Path:
    """
    将路径解析为绝对路径，行为类似 bash/shell：
    - 先归一化常见桌面路径（/Users/Desktop/xxx -> ~/Desktop/xxx）
    - 再展开 ~ 为用户主目录
    - 绝对路径（/xxx 或 C:\\xxx）直接使用
    - 相对路径相对于 base 解析；base 为空时相对于当前工作目录
    """
    resolved = normalize_tool_path(path, base)
    # Windows：若父目录 C:\Users\Desktop 不存在（真实桌面是 C:\Users\<用户名>\Desktop），则使用用户桌面
    if os.name == "nt" and not resolved.parent.exists():
        parent_parts = Path(resolved.parent).parts
        if len(parent_parts) >= 4 and parent_parts[2].lower() == "users" and parent_parts[3].lower() == "desktop":
            real_desktop = Path.home() / "Desktop"
            resolved = real_desktop / resolved.name
    return resolved


def _is_absolute_file_path(path: str) -> bool:
    """Return whether a path is absolute after expanding user-home aliases."""
    raw = str(path).strip()
    if not raw:
        return False
    return is_absolute_tool_path(raw)


def _tool_error(
    message: str,
    error_type: str = "ToolExecutionError",
    **details: Any,
) -> ToolErrorResult:
    input_path = details.pop("input_path", details.pop("input", None))
    path = details.pop("path", details.pop("resolved_path", None))
    normalized_details = dict(details)
    if input_path is not None:
        normalized_details["input_path"] = str(input_path)
    if path is not None:
        normalized_details["path"] = str(path)
    return ToolErrorResult(message, error_type=error_type, details=normalized_details or None)


READ_PARAMETER_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "additionalProperties": False,
    "properties": {
        "file_path": {
            "description": "Absolute file path to read. Accepts Windows paths such as E:\\Projects\\repo\\file.txt, Windows slash paths such as E:/Projects/repo/file.txt, and on Windows Git Bash/MSYS drive paths such as /e/Projects/repo/file.txt. Relative paths are rejected.",
            "type": "string",
        },
        "limit": {
            "description": "要读取的行数；仅当文件过大、无法一次读取时提供。",
            "exclusiveMinimum": 0,
            "maximum": 9007199254740991,
            "type": "integer",
        },
        "offset": {
            "description": "开始读取的行号；仅当文件过大、需要分段读取时提供。",
            "maximum": 9007199254740991,
            "minimum": 0,
            "type": "integer",
        },
        "pages": {
            "description": 'PDF 页码范围，例如 "1-5"、"3"、"10-20"；仅适用于 PDF 文件，每次最多 20 页。',
            "type": "string",
        },
    },
    "required": ["file_path"],
    "type": "object",
}


def _read_file_text(path: Path, offset: Optional[int], limit: Optional[int]) -> str:
    """读取文本文件内容，支持行范围。"""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    if offset is not None or limit is not None:
        start = (offset or 1) - 1
        end = start + (limit or len(lines)) if limit else len(lines)
        lines = lines[start:end]
    return "".join(f"{i + (offset or 1)}|{line}" for i, line in enumerate(lines))


def _parse_pdf_page_range(pages: Optional[str], total_pages: int) -> tuple[Optional[List[int]], Optional[str]]:
    """Parse a 1-based single page or page range into zero-based indices."""
    if total_pages <= 0:
        return [], None

    if not pages:
        end = min(total_pages, 20)
        return list(range(end)), None

    page_spec = pages.strip()
    if not page_spec:
        return None, "错误：pages 不能为空"

    try:
        if "-" in page_spec:
            start_text, end_text = page_spec.split("-", 1)
            start_page = int(start_text.strip())
            end_page = int(end_text.strip())
        else:
            start_page = end_page = int(page_spec)
    except ValueError:
        return None, '错误：pages 必须是页码或页码范围，例如 "1-5"、"3"、"10-20"'

    if start_page < 1 or end_page < 1:
        return None, "错误：PDF 页码从 1 开始"
    if end_page < start_page:
        return None, "错误：pages 的结束页不能小于起始页"
    if end_page - start_page + 1 > 20:
        return None, "错误：PDF 每次最多读取 20 页"
    if start_page > total_pages:
        return None, f"错误：起始页超过 PDF 总页数 {total_pages}"

    end_page = min(end_page, total_pages)
    return list(range(start_page - 1, end_page)), None


def _read_pdf_text(path: Path, pages: Optional[str]) -> str:
    """Read text from a PDF when an optional PDF backend is installed."""
    try:
        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError:
            from PyPDF2 import PdfReader  # type: ignore
    except ImportError:
        return "错误：读取 PDF 需要安装 pypdf 或 PyPDF2"

    try:
        reader = PdfReader(str(path))
        selected_pages, error = _parse_pdf_page_range(pages, len(reader.pages))
        if error:
            return error

        output: List[str] = []
        for page_index in selected_pages or []:
            text = reader.pages[page_index].extract_text() or ""
            output.append(f"--- page {page_index + 1} ---\n{text}".rstrip())
        return "\n\n".join(output) if output else ""
    except Exception as e:
        return f"错误：无法读取 PDF {e}"


GREP_PARAMETER_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "additionalProperties": False,
    "properties": {
        "-A": {
            "description": 'Number of lines to show after each match (rg -A). Requires output_mode: "content", ignored otherwise.',
            "type": "number",
        },
        "-B": {
            "description": 'Number of lines to show before each match (rg -B). Requires output_mode: "content", ignored otherwise.',
            "type": "number",
        },
        "-C": {
            "description": "Alias for context.",
            "type": "number",
        },
        "-i": {
            "description": "Case insensitive search (rg -i)",
            "type": "boolean",
        },
        "-n": {
            "description": 'Show line numbers in output (rg -n). Requires output_mode: "content", ignored otherwise. Defaults to true.',
            "type": "boolean",
        },
        "context": {
            "description": 'Number of lines to show before and after each match (rg -C). Requires output_mode: "content", ignored otherwise.',
            "type": "number",
        },
        "glob": {
            "description": 'Glob pattern to filter files (e.g. "*.js", "*.{ts,tsx}") - maps to rg --glob',
            "type": "string",
        },
        "head_limit": {
            "description": 'Limit output to first N lines/entries, equivalent to "| head -N". Works across all output modes: content (limits output lines), files_with_matches (limits file paths), count (limits count entries). Defaults to 250 when unspecified. Pass 0 for unlimited (use sparingly — large result sets waste context).',
            "type": "number",
        },
        "multiline": {
            "description": "Enable multiline mode where . matches newlines and patterns can span lines (rg -U --multiline-dotall). Default: false.",
            "type": "boolean",
        },
        "offset": {
            "description": 'Skip first N lines/entries before applying head_limit, equivalent to "| tail -n +N | head -N". Works across all output modes. Defaults to 0.',
            "type": "number",
        },
        "output_mode": {
            "description": 'Output mode: "content" shows matching lines (supports -A/-B/-C context, -n line numbers, head_limit), "files_with_matches" shows file paths (supports head_limit), "count" shows match counts (supports head_limit). Defaults to "files_with_matches".',
            "enum": ["content", "files_with_matches", "count"],
            "type": "string",
        },
        "path": {
            "description": "File or directory to search. Omit to use workspace_root. Accepts Windows paths such as E:\\Projects\\repo, Windows slash paths such as E:/Projects/repo, Git Bash/MSYS drive paths on Windows such as /e/Projects/repo, and relative paths such as . or docs resolved from workspace_root.",
            "type": "string",
        },
        "pattern": {
            "description": "The regular expression pattern to search for in file contents",
            "type": "string",
        },
        "type": {
            "description": "File type to search (rg --type). Common types: js, py, rust, go, java, etc. More efficient than include for standard file types.",
            "type": "string",
        },
    },
    "required": ["pattern"],
    "type": "object",
}


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


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _as_non_negative_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _expand_brace_glob(pattern: str) -> list[str]:
    match = re.search(r"\{([^{}]+)\}", pattern)
    if not match:
        return [pattern]
    prefix = pattern[: match.start()]
    suffix = pattern[match.end() :]
    expanded: list[str] = []
    for option in match.group(1).split(","):
        expanded.extend(_expand_brace_glob(prefix + option + suffix))
    return expanded


def _matches_glob(path: Path, root: Path, glob_pattern: str) -> bool:
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        rel = path.name
    patterns = _expand_brace_glob(glob_pattern.lstrip("/"))
    return any(fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(path.name, pattern) for pattern in patterns)


def _file_type_matches(path: Path, file_type: str) -> bool:
    normalized = file_type.lower().lstrip(".")
    suffixes = _GREP_TYPE_SUFFIXES.get(normalized, {"." + normalized})
    return path.suffix.lower() in suffixes


@tool_class(description="文件操作工具集：读取、写入、替换、删除、grep、笔记本编辑、linter 诊断。")
class FileToolkits(ToolClassBase):
    """文件操作工具集：读取、写入、替换、删除、grep、笔记本编辑、linter 诊断。"""

    def __init__(self, session_id: str, workspace_root: Optional[str] = None):
        super().__init__()
        self.session_id = PygentString(session_id)
        self.workspace_root = workspace_root or os.getcwd()

    @tool_method(
        name="read",
        description="Read a file by absolute path. Accepts Windows, Windows slash, and on Windows Git Bash/MSYS drive paths; relative paths are rejected.",
        include_call_description_parameter=False,
        parameters_additional_properties=False,
        parameters_schema_uri="https://json-schema.org/draft/2020-12/schema",
        parameter_schema=READ_PARAMETER_SCHEMA,
    )
    def read(
        self,
        file_path: str,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        pages: Optional[str] = None,
    ) -> str:
        """
        Read file contents from an absolute path.

        Args:
            file_path: Absolute path to the file to read.
            limit: Number of lines to read. Provide only when the file is too large to read at once.
            offset: Line number to start reading from. Provide only when the file is too large to read at once.
            pages: Page range for PDF files, for example "1-5", "3", or "10-20". Applies only to PDF files and is limited to 20 pages per request.
        """
        if not _is_absolute_file_path(file_path):
            return _tool_error("错误：file_path 必须是绝对路径，不能使用相对路径", "ValueError", input=file_path)

        p = _resolve_path(file_path)
        if not p.exists():
            return _tool_error(f"错误：文件不存在 {p}", "FileNotFoundError", input=file_path, resolved_path=str(p))
        if not p.is_file():
            return _tool_error(f"错误：路径不是文件 {p}", "IsADirectoryError", input=file_path, resolved_path=str(p))
        if p.suffix.lower() == ".pdf":
            return _read_pdf_text(p, pages)
        if pages:
            return _tool_error("错误：pages 仅适用于 PDF 文件", "ValueError", input=file_path, resolved_path=str(p))
        return self._read_file(str(p), offset=offset, limit=limit)

    def _read_file(
        self,
        path: str,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> str:
        """
        读取文件内容。返回格式为 行号|内容；若为二进制/图片则返回简短描述。

        Args:
            path: 文件路径。支持：1) 绝对路径（/Users/xxx/file.txt 或 C:\\Users\\xxx\\file.txt）；2) ~ 表示用户主目录（~/Desktop/file.txt）；3) 相对路径相对于工作区根目录。
            offset: 起始行号（从 1 开始）。
            limit: 最多读取的行数。
        """
        p = _resolve_path(path, self.workspace_root)
        if not p.exists():
            return _tool_error(f"错误：文件不存在 {p}", "FileNotFoundError", input=path, resolved_path=str(p))
        if not p.is_file():
            return _tool_error(f"错误：路径不是文件 {p}", "IsADirectoryError", input=path, resolved_path=str(p))
        try:
            # 尝试按文本读取
            with open(p, "rb") as f:
                raw = f.read()
            # 简单检测是否为文本（可打印或常见空白）
            text_chars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7F})
            if not raw or all(b in text_chars for b in raw):
                return _read_file_text(p, offset, limit)
            # 二进制：返回简短描述
            ext = p.suffix.lower()
            if ext in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico"):
                return f"[图片文件 {p.name}, 大小 {len(raw)} 字节]"
            return f"[二进制文件 {p.name}, 大小 {len(raw)} 字节]"
        except OSError as e:
            return _tool_error(f"错误：无法读取文件 {e}", "OSError", input=path, resolved_path=str(p))

    @tool_method(
        name="write",
        description="Write text content to an absolute file path. Accepts Windows, Windows slash, and on Windows Git Bash/MSYS drive paths; relative paths are rejected.",
        include_call_description_parameter=False,
        parameters_additional_properties=False,
        parameters_schema_uri="https://json-schema.org/draft/2020-12/schema",
    )
    def write(self, file_path: str, content: str) -> str:
        """
        Write complete text content to a file, overwriting the file if it already exists.

        Args:
            file_path: Absolute destination file path. Accepts Windows paths such as E:\\Projects\\repo\\file.txt, Windows slash paths such as E:/Projects/repo/file.txt, and on Windows Git Bash/MSYS drive paths such as /e/Projects/repo/file.txt. Relative paths are rejected; parent directories are created when missing.
            content: Full text content to write, replacing any existing file contents.
        """
        if not _is_absolute_file_path(file_path):
            return _tool_error("错误：file_path 必须是绝对路径，不能使用相对路径", "ValueError", input=file_path)

        p = _resolve_path(file_path)
        if p.exists() and p.is_dir():
            return _tool_error(f"错误：file_path 指向目录而不是文件 {p}", "IsADirectoryError", input=file_path, resolved_path=str(p))

        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return "写入完成"
        except OSError as e:
            return _tool_error(f"错误：无法写入 {e}", "OSError", input=file_path, resolved_path=str(p))

    def _search_replace(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        """
        在文件中做精确字符串替换。

        Args:
            path: 文件路径。支持绝对路径、~/Desktop/xxx、或相对于工作区的相对路径。
            old_string: 被替换的字符串，需与文件内容精确匹配（含空格与缩进）。
            new_string: 替换后的新字符串。
            replace_all: 是否替换文件中所有匹配项；默认 false 仅替换第一次。
        """
        p = _resolve_path(path, self.workspace_root)
        if not p.exists() or not p.is_file():
            return _tool_error(f"错误：文件不存在或不是文件 {p}", "FileNotFoundError", input=path, resolved_path=str(p))
        try:
            text = p.read_text(encoding="utf-8")
            if replace_all:
                if old_string not in text:
                    return _tool_error("错误：未找到匹配内容", "ValueError", input=path, resolved_path=str(p))
                new_text = text.replace(old_string, new_string)
            else:
                if old_string not in text:
                    return _tool_error("错误：未找到匹配内容", "ValueError", input=path, resolved_path=str(p))
                new_text = text.replace(old_string, new_string, 1)
            p.write_text(new_text, encoding="utf-8")
            return "替换完成"
        except OSError as e:
            return _tool_error(f"错误：{e}", "OSError", input=path, resolved_path=str(p))

    @tool_method(
        name="edit",
        description="Edit a file by replacing exact text at an absolute file path. Accepts Windows, Windows slash, and on Windows Git Bash/MSYS drive paths; relative paths are rejected.",
        include_call_description_parameter=False,
        parameters_additional_properties=False,
        parameters_schema_uri="https://json-schema.org/draft/2020-12/schema",
    )
    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        """
        Replace exact text in a file.

        Args:
            file_path: Absolute path of the file to modify. Accepts Windows paths such as E:\\Projects\\repo\\file.txt, Windows slash paths such as E:/Projects/repo/file.txt, and on Windows Git Bash/MSYS drive paths such as /e/Projects/repo/file.txt. Relative paths are rejected.
            old_string: Exact text to replace, including whitespace and indentation.
            new_string: Replacement text; use a value different from old_string.
            replace_all: Replace every occurrence when true; defaults to false, which replaces only the first match.
        """
        if not _is_absolute_file_path(file_path):
            return _tool_error("错误：file_path 必须是绝对路径，不能使用相对路径", "ValueError", input=file_path)
        if old_string == new_string:
            return _tool_error("错误：new_string 必须与 old_string 不同", "ValueError", input=file_path)

        return self._search_replace(
            file_path,
            old_string,
            new_string,
            replace_all=replace_all,
        )

    @tool_method(
        name="edit_notebook",
        description="编辑 Jupyter 笔记本的单元格：修改已有单元格或插入新单元格。",
    )
    def edit_notebook(
        self,
        target_notebook: str,
        cell_idx: int,
        is_new_cell: bool,
        cell_language: str,
        old_string: str,
        new_string: str,
    ) -> str:
        """
        编辑 Jupyter 笔记本单元格。

        Args:
            target_notebook: 文件路径。支持绝对路径、~/Desktop/xxx、或相对于工作区的相对路径。
            cell_idx: 要编辑或插入位置的单元格索引（从 0 开始）。
            is_new_cell: true=新建单元格；false=编辑已有单元格。
            cell_language: 单元格语言类型。
            old_string: 要替换的单元格内容；新建单元格时传空字符串。
            new_string: 新内容或新单元格的完整内容。
        """
        p = _resolve_path(target_notebook, self.workspace_root)
        if not p.exists() or not p.is_file():
            return f"错误：笔记本不存在 {p}"
        try:
            nb = json.loads(p.read_text(encoding="utf-8"))
            cells = nb.get("cells", [])
            if is_new_cell:
                cell_type = "code" if cell_language in ("python", "javascript", "typescript", "r", "sql", "shell") else "markdown" if cell_language == "markdown" else "raw"
                new_cell = {"cell_type": cell_type, "metadata": {}, "source": new_string.splitlines(keepends=True) if isinstance(new_string, str) else [new_string]}
                if not new_cell["source"] and new_string:
                    new_cell["source"] = [new_string + "\n"]
                cells.insert(cell_idx, new_cell)
            else:
                if cell_idx < 0 or cell_idx >= len(cells):
                    return f"错误：单元格索引越界 0..{len(cells)-1}"
                cell = cells[cell_idx]
                src = cell.get("source", [])
                if isinstance(src, list):
                    content = "".join(src)
                else:
                    content = str(src)
                if old_string not in content:
                    return "错误：未在单元格中找到要替换的内容"
                new_content = content.replace(old_string, new_string, 1)
                cell["source"] = new_content.splitlines(keepends=True) if new_content else []
            nb["cells"] = cells
            p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
            return "笔记本已更新"
        except (json.JSONDecodeError, OSError) as e:
            return f"错误：{e}"

    def _delete_file(self, path: str) -> str:
        """
        删除指定路径的文件。

        Args:
            path: 文件路径。支持绝对路径、~/Desktop/xxx、或相对于工作区的相对路径。
        """
        p = _resolve_path(path, self.workspace_root)
        if not p.exists():
            return f"错误：文件不存在 {p}"
        if not p.is_file():
            return f"错误：路径不是文件，无法删除 {p}"
        try:
            p.unlink()
            return "已删除"
        except OSError as e:
            return f"错误：无法删除 {e}"

    @tool_method(
        name="read_lints",
        description="读取工作区或指定文件/目录的 linter 诊断信息。",
    )
    def read_lints(self, paths: Optional[list[str]] = None) -> str:
        """
        返回 linter 诊断。本实现不接入 IDE，可返回空列表或尝试调用外部 linter。

        Args:
            paths: 要检查的文件或目录路径列表；不传则返回整个工作区诊断（本实现返回说明）。
        """
        if not paths:
            return "当前实现不接入 IDE 诊断；可传入 paths 指定文件或目录，将返回占位说明。"
        return "当前实现不接入 IDE linter；诊断列表通常由编辑器/语言服务提供。"

    @tool_method(
        name="glob",
        description="Find files by glob pattern under a directory and return matching file paths.",
        include_call_description_parameter=False,
        parameter_schema={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "additionalProperties": False,
            "properties": {
                "path": {
                    "description": "Directory to search. Omit this field to use workspace_root. Accepts Windows paths such as E:\\Projects\\repo, Windows slash paths such as E:/Projects/repo, Git Bash/MSYS drive paths on Windows such as /e/Projects/repo, and relative paths such as . or docs resolved from workspace_root.",
                    "type": "string",
                },
                "pattern": {
                    "description": "Glob pattern used to match file paths, for example \"*.py\" or \"**/*.ts\".",
                    "type": "string",
                },
            },
            "required": ["pattern"],
            "type": "object",
        },
    )
    def glob(self, pattern: str, path: Optional[str] = None) -> str:
        """
        Find files whose relative paths match a glob pattern.

        Args:
            pattern: Glob pattern used to match files, for example "*.py" or "**/*.ts".
            path: Directory to search. Omit this field to search from the workspace root; do not pass "undefined" or "null". If provided, it must be an existing directory path.
        """
        root = _resolve_path(path or ".", self.workspace_root)
        if not root.exists():
            return _tool_error(f"error: path does not exist {root}", "FileNotFoundError", input=path or ".", resolved_path=str(root))
        if not root.is_dir():
            return _tool_error(f"error: path must be a directory {root}", "NotADirectoryError", input=path or ".", resolved_path=str(root))

        normalized_pattern = pattern.lstrip("/\\")
        try:
            matches = [p for p in root.glob(normalized_pattern) if p.is_file()]
        except (OSError, ValueError) as e:
            return _tool_error(f"error: invalid glob pattern {e}", "ValueError", input=path or ".", resolved_path=str(root))

        def sort_key(file_path: Path) -> tuple[float, str]:
            try:
                mtime = file_path.stat().st_mtime
            except OSError:
                mtime = 0.0
            return (-mtime, str(file_path))

        return "\n".join(str(p) for p in sorted(matches, key=sort_key))

    @tool_method(
        name="grep",
        description="按精确文本或正则表达式在文件中搜索，支持上下文行数、文件类型、大小写等选项。",
        parameter_schema=GREP_PARAMETER_SCHEMA,
    )
    def grep(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
        output_mode: Optional[str] = None,
        context_before: Optional[int] = None,
        context_after: Optional[int] = None,
        context_both: Optional[int] = None,
        context: Optional[int] = None,
        ignore_case: bool = False,
        file_type: Optional[str] = None,
        type: Optional[str] = None,
        head_limit: Optional[int] = None,
        offset: Optional[int] = None,
        multiline: bool = False,
        show_line_numbers: bool = True,
        **kwargs: Any,
    ) -> str:
        """
        在文件中按正则或字面量搜索。

        Args:
            pattern: 正则表达式或字面量搜索串。
            path: 要搜索的文件或目录路径。支持绝对路径、~/Desktop/xxx、或相对于工作区的相对路径；不传则默认为工作区根目录。
            glob: 文件名过滤，如 '*.js'、'**/*.ts'。
            output_mode: content=匹配行及上下文；files_with_matches=仅文件路径；count=匹配数量。
            context_before: 匹配行之前显示的上下文行数（-B）。
            context_after: 匹配行之后显示的上下文行数（-A）。
            context_both: 匹配行前后各显示的上下文行数（-C）。
            context: 匹配行前后各显示的上下文行数。
            ignore_case: 是否忽略大小写。
            file_type: 按文件类型过滤，如 js, py, ts。
            type: 按文件类型过滤，与 schema 中的 type 参数对应。
            head_limit: 结果数量上限。
            offset: 跳过前 N 条结果，用于分页。
            multiline: 是否启用跨行匹配。
            show_line_numbers: content 模式下是否显示行号。
        """
        # 兼容 schema 中的 -B, -A, -C, -i, -n, context, type。
        context_before = _first_present(kwargs.get("-B"), context_before)
        context_after = _first_present(kwargs.get("-A"), context_after)
        context_both = _first_present(kwargs.get("-C"), context_both, kwargs.get("context"), context)
        ignore_case = bool(_first_present(kwargs.get("-i"), ignore_case))
        show_line_numbers = bool(_first_present(kwargs.get("-n"), show_line_numbers))
        file_type = _first_present(kwargs.get("type"), type, file_type)
        output_mode = output_mode or "files_with_matches"
        if output_mode not in {"content", "files_with_matches", "count"}:
            return _tool_error(f"错误：不支持的 output_mode {output_mode}", "ValueError")

        root = _resolve_path(path or ".", self.workspace_root)
        if not root.exists():
            return _tool_error(f"错误：路径不存在 {root}", "FileNotFoundError", input=path or ".", resolved_path=str(root))

        flags = re.IGNORECASE if ignore_case else 0
        if multiline:
            flags |= re.DOTALL | re.MULTILINE
        try:
            rx = re.compile(pattern, flags)
        except re.error:
            rx = re.compile(re.escape(pattern), re.IGNORECASE if ignore_case else 0)

        before = _as_non_negative_int(_first_present(context_before, context_both), 0)
        after = _as_non_negative_int(_first_present(context_after, context_both), 0)
        skip = _as_non_negative_int(offset, 0)
        limit = _as_non_negative_int(head_limit, 250) if head_limit is not None else 250

        if root.is_file():
            files = [root]
        else:
            files = [f for f in root.rglob("*") if f.is_file()]
            if glob:
                files = [f for f in files if _matches_glob(f, root, glob)]
            if file_type:
                files = [f for f in files if _file_type_matches(f, file_type)]

        content_lines: list[str] = []
        count_pairs: list[tuple[Path, int]] = []
        file_set: set[Path] = set()

        def format_line(fp: Path, line_no: int, line: str) -> str:
            body = f"{line_no}|{line}" if show_line_numbers else line
            return f"{fp}:{body}"

        for fp in sorted(files):
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            lines = text.splitlines()
            if multiline:
                matches = list(rx.finditer(text))
                if not matches:
                    continue
                file_set.add(fp)
                count_pairs.append((fp, len(matches)))
                if output_mode != "content":
                    continue
                for match in matches:
                    start_line = text.count("\n", 0, match.start())
                    end_at = max(match.start(), match.end() - 1)
                    end_line = text.count("\n", 0, end_at)
                    start = max(0, start_line - before)
                    end = min(len(lines), end_line + after + 1)
                    for j in range(start, end):
                        content_lines.append(format_line(fp, j + 1, lines[j]))
                continue

            match_count = 0
            for i, line in enumerate(lines):
                if not rx.search(line):
                    continue
                match_count += 1
                file_set.add(fp)
                if output_mode == "content":
                    start = max(0, i - before)
                    end = min(len(lines), i + after + 1)
                    for j in range(start, end):
                        content_lines.append(format_line(fp, j + 1, lines[j]))
            if match_count:
                count_pairs.append((fp, match_count))

        if output_mode == "count":
            if not count_pairs:
                return "0"
            limited_pairs = count_pairs[skip:]
            if limit:
                limited_pairs = limited_pairs[:limit]
            if len(count_pairs) == 1 and skip == 0 and (limit == 0 or limit >= 1):
                return str(count_pairs[0][1])
            return "\n".join(f"{fp}:{count}" for fp, count in limited_pairs)
        if output_mode == "files_with_matches":
            out = sorted(str(p) for p in file_set)
            if skip:
                out = out[skip:]
            if limit:
                out = out[:limit]
            return "\n".join(out) if out else ""

        out_lines = content_lines[skip:]
        if limit:
            out_lines = out_lines[:limit]
        return "\n".join(out_lines) if out_lines else "无匹配"
