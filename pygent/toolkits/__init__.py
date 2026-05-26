from typing import Optional

from .file_operations import FileToolkits
from .run_terminal_cmd import TerminalToolkits
from .web_search import WebSearchToolkits
from .web_fetch import WebFetchToolkits


class RestrictedTerminal(TerminalToolkits):
    """Backward-compatible terminal toolkit name.

    Older examples constructed ``RestrictedTerminal(root_dir=".")``. Keep that
    shape while delegating to ``TerminalToolkits`` internally.
    """

    def __init__(self, root_dir: str = ".", session_id: str = "terminal", workspace_root: Optional[str] = None):
        super().__init__(session_id=session_id, workspace_root=workspace_root or root_dir)

    def get_tools(self):
        return self.get_all_tools()


__all__ = [
    "FileToolkits",
    "TerminalToolkits",
    "RestrictedTerminal",
    "WebSearchToolkits",
    "WebFetchToolkits",
]

