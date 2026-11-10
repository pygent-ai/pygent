"""Readiness-retrying MCP stdio transport for pygent's ``MCPClientPool``.

Pygent 0.3.29's ``MCPClientPool`` owns the MCP session lifecycle: one
persistent session per server held by a pool-side owner task (anyio cancel
scopes force same-task enter/exit), reconnect after a broken session, and
bounded shutdown. This module only keeps the server-specific workaround the
pool does not provide: some servers reject a call before dispatching any
action while their own bridge connection is still starting up — e.g.
``open-browser-control`` connects to its bridge only after startup, so an
early call reliably fails with "Browser extension not connected". The same
call is retried on the shared session until the backend is ready; failures
after an action was dispatched are never retried.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from pygent.tool.mcp import MCPStdioTransport


class _ReadinessRetryingSession:
    """Delegate to the shared session, retrying pre-action readiness rejections.

    The server rejects a call before dispatching any action while its own
    bridge connection is still starting up, so retrying the same call on the
    warm session converges without re-execution risk. Post-action timeouts
    are never retried here.
    """

    _RETRYABLE_MARKERS = (
        "Browser extension not connected",
        "Timeout waiting for session to start",
    )
    _RETRY_ATTEMPTS = 3
    _RETRY_DELAY_S = 1.5

    def __init__(self, session: Any) -> None:
        self._session = session

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)

    async def call_tool(self, name: str, arguments: Any = None, **kwargs: Any) -> Any:
        result = await self._session.call_tool(name, arguments=arguments, **kwargs)
        for attempt in range(self._RETRY_ATTEMPTS - 1):
            if not result.is_error:
                break
            text = "\n".join(
                item.text for item in result.content if getattr(item, "text", None)
            )
            if not any(marker in text for marker in self._RETRYABLE_MARKERS):
                break
            await asyncio.sleep(self._RETRY_DELAY_S * (attempt + 1))
            result = await self._session.call_tool(
                name, arguments=arguments, **kwargs
            )
        return result


class ReadinessRetryingTransport:
    """Cold stdio transport whose sessions retry pre-action readiness rejections.

    Passed to ``MCPClientPool`` as the server transport: the pool enters
    ``session()`` once per session generation in its own owner task and holds
    the yielded session for discovery and every tool call, so no
    per-operation reconnect and no extra lifecycle code is needed here.
    """

    def __init__(
        self,
        command: str,
        *,
        args: Sequence[str] = (),
        env: Mapping[str, str] | None = None,
        cwd: str | Path | None = None,
    ) -> None:
        self._cold = MCPStdioTransport(command, args=args, env=env, cwd=cwd)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[Any]:
        async with self._cold.session() as session:
            yield _ReadinessRetryingSession(session)


__all__ = ["ReadinessRetryingTransport"]
