"""Backward-compatible public API entry points."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from pygent.llm import AsyncOpenAIClient, AsyncRequestsClient
from pygent.toolkits import RestrictedTerminal, TerminalToolkits


def test_async_openai_client_alias():
    assert AsyncOpenAIClient is AsyncRequestsClient


def test_restricted_terminal_compat_shape(tmp_path):
    terminal = RestrictedTerminal(root_dir=str(tmp_path))
    assert isinstance(terminal, TerminalToolkits)
    assert terminal.workspace_root == str(tmp_path)
    assert terminal.get_tools() == terminal.get_all_tools()
