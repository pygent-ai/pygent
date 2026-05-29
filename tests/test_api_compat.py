"""Public API entry points."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from pygent.llm import AsyncOpenAIClient, AsyncRequestsClient
from pygent.toolkits import BashToolkits, FileToolkits, WebFetchToolkits, WebSearchToolkits


def test_async_openai_client_alias():
    assert AsyncOpenAIClient is AsyncRequestsClient


def test_toolkit_exports_are_current_public_api():
    assert BashToolkits is not None
    assert FileToolkits is not None
    assert WebFetchToolkits is not None
    assert WebSearchToolkits is not None
