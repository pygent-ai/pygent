"""Narrow service boundaries used to compose :class:`FileTools`."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FileIOService:
    read_backend: Callable[[str, int | None, int | None, str | None], str]
    write_backend: Callable[[str, str], str]
    edit_backend: Callable[[str, str, str, bool], str]

    def read(
        self, file_path: str, limit: int | None, offset: int | None, pages: str | None
    ) -> str:
        return self.read_backend(file_path, limit, offset, pages)

    def write(self, file_path: str, content: str) -> str:
        return self.write_backend(file_path, content)

    def edit(
        self, file_path: str, old_string: str, new_string: str, replace_all: bool
    ) -> str:
        return self.edit_backend(file_path, old_string, new_string, replace_all)


@dataclass(frozen=True, slots=True)
class NotebookService:
    edit_backend: Callable[[str, int, bool, str, str, str], str]

    def edit(
        self,
        target_notebook: str,
        cell_idx: int,
        is_new_cell: bool,
        cell_language: str,
        old_string: str,
        new_string: str,
    ) -> str:
        return self.edit_backend(
            target_notebook,
            cell_idx,
            is_new_cell,
            cell_language,
            old_string,
            new_string,
        )


@dataclass(frozen=True, slots=True)
class FileDiagnosticsService:
    lint_backend: Callable[[list[str] | None], str]

    def read_lints(self, paths: list[str] | None) -> str:
        return self.lint_backend(paths)


@dataclass(frozen=True, slots=True)
class FileSearchService:
    glob_backend: Callable[[str, str | None, int], Awaitable[str]]
    grep_backend: Callable[
        [str, str | None, str | None, bool, bool, int, int], Awaitable[str]
    ]

    async def glob(self, pattern: str, path: str | None, limit: int) -> str:
        return await self.glob_backend(pattern, path, limit)

    async def grep(
        self,
        pattern: str,
        path: str | None,
        glob: str | None,
        ignore_case: bool,
        literal: bool,
        context: int,
        limit: int,
    ) -> str:
        return await self.grep_backend(
            pattern, path, glob, ignore_case, literal, context, limit
        )


__all__ = [
    "FileDiagnosticsService",
    "FileIOService",
    "FileSearchService",
    "NotebookService",
]
