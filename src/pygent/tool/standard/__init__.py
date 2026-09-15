"""Optional standard tools built from ordinary Pygent 0.2 function tools.

The classes in this package are deployment-local handler owners. They are not
portable Tool values and never enter Context, Message, ToolSpec, or an
ExecutionPlan. Use their bound methods with :class:`pygent.tool.ToolKit`, or use
``StandardTools.toolkit`` as an explicit assembly convenience.
"""

from __future__ import annotations

from pathlib import Path

from pygent.tool.executors import ToolTaskManager
from pygent.tool.functional import ToolKit

from ._bash import BashTools
from ._files import FileTools
from ._web import Fetcher, Resolver, Searcher, WebFetchTools, WebSearchTools


class StandardTools:
    """Assemble workspace, web and task-control tools with explicit ownership."""

    def __init__(
        self,
        *,
        workspace_root: str | Path,
        restrict_to_workspace: bool = True,
        bash_executable: str | None = None,
        bash_timeout: float = 600,
        task_manager: ToolTaskManager | None = None,
        web_searcher: Searcher | None = None,
        web_fetcher: Fetcher | None = None,
        web_resolver: Resolver | None = None,
    ) -> None:
        self.bash = BashTools(
            workspace_root=workspace_root,
            restrict_to_workspace=restrict_to_workspace,
            bash_executable=bash_executable,
            timeout=bash_timeout,
            task_manager=task_manager,
        )
        self.files = FileTools(
            workspace_root=workspace_root,
            restrict_to_workspace=restrict_to_workspace,
        )
        self.web_fetch = WebFetchTools(
            fetcher=web_fetcher,
            resolver=web_resolver,
        )
        self.web_search = WebSearchTools(searcher=web_searcher)
        self.toolkit = ToolKit(
            self.bash.bash,
            self.bash.tool_task_get,
            self.bash.tool_task_stop,
            self.files.edit,
            self.files.edit_notebook,
            self.files.glob,
            self.files.grep,
            self.files.read,
            self.files.read_lints,
            self.web_fetch.web_fetch,
            self.web_search.web_search,
            self.files.write,
            wait_timeouts={"bash": bash_timeout},
        )

    async def aclose(self) -> None:
        await self.bash.aclose()

    async def close(self) -> None:
        await self.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


__all__ = [
    "BashTools",
    "FileTools",
    "StandardTools",
    "WebFetchTools",
    "WebSearchTools",
]
