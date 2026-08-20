"""MCP stdio adapter implementing the common ToolExecutor protocol."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Protocol, cast

from mcp import ClientSession, StdioServerParameters
from mcp import types as mcp_types
from mcp.client.stdio import stdio_client

from pygent.core import FrozenJsonObject, thaw_json

from ..executors import ExecutorRegistry, ToolExecutionContext, ToolExecutionError
from ..types import (
    IdempotencyPolicy,
    ToolCall,
    ToolDefinition,
    ToolSideEffect,
    ToolSpec,
)


class MCPTransport(Protocol):
    def session(self) -> Any: ...


class MCPStdioTransport:
    """Cold-connect stdio transport; subprocess lifetime is scoped to one operation."""

    def __init__(
        self,
        command: str,
        *,
        args: Sequence[str] = (),
        env: Mapping[str, str] | None = None,
        cwd: str | Path | None = None,
    ) -> None:
        if not command:
            raise ValueError("command must be non-empty")
        self._parameters = StdioServerParameters(
            command=command,
            args=list(args),
            env=None if env is None else dict(env),
            cwd=cwd,
        )

    @asynccontextmanager
    async def session(self) -> AsyncIterator[ClientSession]:
        async with (
            stdio_client(self._parameters) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()
            yield session


class MCPToolExecutor:
    """Execute a named MCP tool through stdio without a parallel API."""

    def __init__(
        self, transport: MCPTransport, *, remote_name: str | None = None
    ) -> None:
        self._transport = transport
        self._remote_name = remote_name

    async def execute(
        self, spec: ToolSpec, call: ToolCall, context: ToolExecutionContext
    ) -> object:
        name = self._remote_name or spec.definition.name
        try:
            async with self._transport.session() as session:
                result = await session.call_tool(
                    name,
                    arguments=thaw_json(cast(FrozenJsonObject, call.arguments)),
                )
        except ToolExecutionError:
            raise
        except Exception as exc:
            raise ToolExecutionError(
                "MCP tool transport failed",
                kind="transport_error",
                code="mcp_transport",
                retryable=True,
                side_effect_committed=None,
            ) from exc
        if result.is_error:
            raise ToolExecutionError(
                "MCP tool reported an error",
                kind="remote_error",
                code="mcp_tool_error",
                retryable=False,
                side_effect_committed=None,
            )
        if result.structured_content is not None:
            return result.structured_content
        return {
            "content": [
                item.model_dump(mode="json", by_alias=True, exclude_none=True)
                for item in result.content
            ]
        }


async def discover_mcp_tools(
    transport: MCPTransport,
    *,
    namespace: str = "mcp",
    version: str = "1",
    timeout: float | None = None,
) -> tuple[ToolSpec, ...]:
    """Discover MCP definitions and map them into portable ToolSpec values."""

    if not namespace or not version:
        raise ValueError("namespace and version must be non-empty")
    discovered: list[ToolSpec] = []
    cursor: str | None = None
    async with transport.session() as session:
        while True:
            page = await session.list_tools(
                params=mcp_types.PaginatedRequestParams(cursor=cursor)
            )
            for tool in page.tools:
                annotations = tool.annotations
                read_only = bool(
                    annotations is not None and annotations.read_only_hint is True
                )
                idempotent = bool(
                    annotations is not None and annotations.idempotent_hint is True
                )
                discovered.append(
                    ToolSpec(
                        tool_id=f"{namespace}.{tool.name}",
                        version=version,
                        definition=ToolDefinition(
                            name=tool.name,
                            description=tool.description or "",
                            parameters=tool.input_schema,
                            output_schema=tool.output_schema,
                        ),
                        side_effect=(
                            ToolSideEffect.READ
                            if read_only
                            else ToolSideEffect.EXTERNAL
                        ),
                        idempotency=(
                            IdempotencyPolicy.INHERENT
                            if read_only or idempotent
                            else IdempotencyPolicy.NOT_IDEMPOTENT
                        ),
                        timeout=timeout,
                    )
                )
            cursor = page.next_cursor
            if cursor is None:
                break
    return tuple(discovered)


def register_mcp_tools(
    registry: ExecutorRegistry,
    transport: MCPTransport,
    specs: Sequence[ToolSpec],
) -> None:
    """Register discovered MCP tools in the common executor registry."""

    for spec in specs:
        registry.register(
            spec.tool_id,
            spec.version,
            MCPToolExecutor(transport, remote_name=spec.definition.name),
        )


__all__ = [
    "MCPStdioTransport",
    "MCPToolExecutor",
    "MCPTransport",
    "discover_mcp_tools",
    "register_mcp_tools",
]
