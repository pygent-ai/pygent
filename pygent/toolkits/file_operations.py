"""
文件操作工具集。参考 docs/tools_openai_schema.json 实现所有文件相关工具。
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, List, Optional

from pygent.common import PygentString
from pygent.module.tool import BaseTool
from pygent.module.tool.utils import tool_method, tool_class


def _resolve_path(path: str, base: Optional[str] = None) -> Path:
    """将路径解析为绝对路径。"""
    p = Path(path)
    if not p.is_absolute() and base:
        p = Path(base) / p
    return p.resolve()


def _read_file_text(path: Path, offset: Optional[int], limit: Optional[int]) -> str:
    """读取文本文件内容，支持行范围。"""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    if offset is not None or limit is not None:
        start = (offset or 1) - 1
        end = start + (limit or len(lines)) if limit else len(lines)
        lines = lines[start:end]
    return "".join(f"{i + (offset or 1)}|{line}" for i, line in enumerate(lines))


@tool_class(description="文件操作工具集：读取、写入、替换、删除、grep、笔记本编辑、linter 诊断。")
class FileToolkits:
    """文件操作工具集：读取、写入、替换、删除、grep、笔记本编辑、linter 诊断。"""

    def __init__(self, session_id: str, workspace_root: Optional[str] = None):
        self.session_id = PygentString(session_id)
        self.workspace_root = PygentString(workspace_root) or PygentString(os.getcwd())

    @tool_method(
        name="read_file",
        description="从本地读取文件内容，支持按行范围读取；也可读取图片等二进制文件的描述。",
    )
    def read_file(
        self,
        path: str,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> str:
        """
        读取文件内容。返回格式为 行号|内容；若为二进制/图片则返回简短描述。

        Args:
            path: 文件的绝对路径或相对工作区的路径。
            offset: 起始行号（从 1 开始）。
            limit: 最多读取的行数。
        """
        p = _resolve_path(path, self.workspace_root)
        if not p.exists():
            return f"错误：文件不存在 {p}"
        if not p.is_file():
            return f"错误：路径不是文件 {p}"
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
            return f"错误：无法读取文件 {e}"

    @tool_method(
        name="write",
        description="将内容写入指定路径；若文件已存在则完整覆盖。",
    )
    def write(self, path: str, contents: str) -> str:
        """
        将完整内容写入文件，已存在则覆盖。

        Args:
            path: 要写入的文件路径。
            contents: 文件的完整内容。
        """
        p = _resolve_path(path, self.workspace_root)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(contents, encoding="utf-8")
            return "写入完成"
        except OSError as e:
            return f"错误：无法写入 {e}"

    @tool_method(
        name="search_replace",
        description="在指定文件中做精确字符串替换，可单次或全部替换。",
    )
    def search_replace(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        """
        在文件中做精确字符串替换。

        Args:
            path: 要修改的文件绝对路径。
            old_string: 被替换的字符串，需与文件内容精确匹配（含空格与缩进）。
            new_string: 替换后的新字符串。
            replace_all: 是否替换文件中所有匹配项；默认 false 仅替换第一次。
        """
        p = _resolve_path(path, self.workspace_root)
        if not p.exists() or not p.is_file():
            return f"错误：文件不存在或不是文件 {p}"
        try:
            text = p.read_text(encoding="utf-8")
            if replace_all:
                if old_string not in text:
                    return f"错误：未找到匹配内容"
                new_text = text.replace(old_string, new_string)
            else:
                if old_string not in text:
                    return f"错误：未找到匹配内容"
                new_text = text.replace(old_string, new_string, 1)
            p.write_text(new_text, encoding="utf-8")
            return "替换完成"
        except OSError as e:
            return f"错误：{e}"

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
            target_notebook: 笔记本文件路径（相对或绝对）。
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

    @tool_method(
        name="delete_file",
        description="删除指定路径的文件；若文件不存在或无权限则报错。",
    )
    def delete_file(self, path: str) -> str:
        """
        删除指定路径的文件。

        Args:
            path: 要删除的文件绝对路径。
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
        name="grep",
        description="按精确文本或正则表达式在文件中搜索，支持上下文行数、文件类型、大小写等选项。",
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
        ignore_case: bool = False,
        file_type: Optional[str] = None,
        head_limit: Optional[int] = None,
        offset: Optional[int] = None,
        multiline: bool = False,
        **kwargs: Any,
    ) -> str:
        """
        在文件中按正则或字面量搜索。

        Args:
            pattern: 正则表达式或字面量搜索串。
            path: 要搜索的文件或目录路径，不传则默认为工作区根目录。
            glob: 文件名过滤，如 '*.js'、'**/*.ts'。
            output_mode: content=匹配行及上下文；files_with_matches=仅文件路径；count=匹配数量。
            context_before: 匹配行之前显示的上下文行数（-B）。
            context_after: 匹配行之后显示的上下文行数（-A）。
            context_both: 匹配行前后各显示的上下文行数（-C）。
            ignore_case: 是否忽略大小写。
            file_type: 按文件类型过滤，如 js, py, ts。
            head_limit: 结果数量上限。
            offset: 跳过前 N 条结果，用于分页。
            multiline: 是否启用跨行匹配。
        """
        # 兼容 schema 中的 -B, -A, -C, -i, type
        context_before = context_before or kwargs.get("-B")
        context_after = context_after or kwargs.get("-A")
        context_both = context_both or kwargs.get("-C")
        ignore_case = ignore_case or kwargs.get("-i", False)
        file_type = file_type or kwargs.get("type")
        output_mode = output_mode or "content"

        root = _resolve_path(path or ".", self.workspace_root)
        if not root.exists():
            return f"错误：路径不存在 {root}"

        flags = re.IGNORECASE if ignore_case else 0
        if multiline:
            flags |= re.DOTALL
        try:
            rx = re.compile(pattern, flags)
        except re.error:
            rx = re.compile(re.escape(pattern), re.IGNORECASE if ignore_case else 0)

        before = context_before if context_before is not None else (context_both or 0)
        after = context_after if context_after is not None else (context_both or 0)

        if root.is_file():
            files = [root]
        else:
            if glob:
                files = list(root.rglob(glob.lstrip("/")))
                files = [f for f in files if f.is_file()]
            else:
                files = [f for f in root.rglob("*") if f.is_file()]
            if file_type:
                suffix = "." + file_type.lstrip(".")
                files = [f for f in files if f.suffix.lower() == suffix.lower()]

        matches: list[tuple[Path, int, list[str]]] = []  # (file, line_idx, context_lines)
        file_set: set[Path] = set()

        for fp in sorted(files):
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            lines = text.splitlines()
            for i, line in enumerate(lines):
                if not rx.search(line):
                    continue
                if output_mode == "files_with_matches":
                    file_set.add(fp)
                    break
                start = max(0, i - before)
                end = min(len(lines), i + after + 1)
                ctx = [f"{j+1}|{lines[j]}" for j in range(start, end)]
                matches.append((fp, i, ctx))

        skip = offset or 0
        limit = head_limit

        if output_mode == "count":
            return str(len(matches))
        if output_mode == "files_with_matches":
            out = sorted(str(p) for p in file_set)
            if skip:
                out = out[skip:]
            if limit is not None:
                out = out[:limit]
            return "\n".join(out) if out else ""

        out_lines = []
        subset = matches[skip : skip + limit] if limit else matches[skip:]
        for fp, _i, ctx in subset:
            for line in ctx:
                out_lines.append(f"{fp}:{line}")
        return "\n".join(out_lines) if out_lines else "无匹配"

    def get_tools(self) -> List[BaseTool]:
        """返回本工具集中所有 @tool_method 对应的 BaseTool 列表，供 ToolManager.register_tools() 使用。"""
        tools: List[BaseTool] = []
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            attr = getattr(self, attr_name)
            tools.append(attr)
        return tools


