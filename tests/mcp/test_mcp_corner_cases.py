from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from mcp import types as mcp_types

from pygent.tool import (
    ExecutorRegistry,
    IdempotencyPolicy,
    ToolCall,
    ToolExecutionContext,
    ToolExecutionError,
    ToolSideEffect,
)
from pygent.tool import mcp as mcp_module
from pygent.tool.mcp import (
    MCPToolExecutor,
    discover_mcp_tools,
    register_mcp_tools,
)


class FakeTransport:
    def __init__(self, session) -> None:
        self._session = session

    @asynccontextmanager
    async def session(self):
        yield self._session


def test_mcp_public_surface_has_no_legacy_sse_transport() -> None:
    assert not hasattr(mcp_module, "MCPSseTransport")


class PaginatedSession:
    def __init__(self) -> None:
        self.cursors: list[str | None] = []

    async def list_tools(self, *, params):
        self.cursors.append(params.cursor)
        if params.cursor is None:
            return mcp_types.ListToolsResult(
                tools=[
                    mcp_types.Tool(
                        name="read",
                        description=None,
                        input_schema={"type": "object"},
                        output_schema={"type": "string"},
                        annotations=mcp_types.ToolAnnotations(read_only_hint=True),
                    ),
                ],
                next_cursor="page-2",
            )
        return mcp_types.ListToolsResult(
            tools=[
                mcp_types.Tool(
                    name="write",
                    description="write data",
                    input_schema={"type": "object"},
                    output_schema=None,
                    annotations=mcp_types.ToolAnnotations(idempotent_hint=True),
                ),
                mcp_types.Tool(
                    name="notify",
                    description="notify once",
                    input_schema={"type": "object"},
                    output_schema=None,
                    annotations=None,
                ),
            ],
            next_cursor=None,
        )


@pytest.mark.asyncio
async def test_discovery_consumes_all_pages_and_maps_mcp_hints() -> None:
    session = PaginatedSession()
    specs = await discover_mcp_tools(
        FakeTransport(session), namespace="memory", version="3", timeout=2.5
    )

    assert session.cursors == [None, "page-2"]
    assert [item.tool_id for item in specs] == [
        "memory.read",
        "memory.write",
        "memory.notify",
    ]
    assert [item.definition.name for item in specs] == ["read", "write", "notify"]
    assert [item.side_effect for item in specs] == [
        ToolSideEffect.READ,
        ToolSideEffect.EXTERNAL,
        ToolSideEffect.EXTERNAL,
    ]
    assert [item.idempotency for item in specs] == [
        IdempotencyPolicy.INHERENT,
        IdempotencyPolicy.INHERENT,
        IdempotencyPolicy.NOT_IDEMPOTENT,
    ]
    assert all(item.timeout == 2.5 for item in specs)


@pytest.mark.asyncio
async def test_mcp_executor_prefers_structured_content_and_falls_back_to_blocks() -> (
    None
):
    class Session:
        def __init__(self) -> None:
            self.structured = True

        async def call_tool(self, name, *, arguments):
            assert name == "remote_read"
            assert arguments == {"path": "notes.txt"}
            if self.structured:
                return SimpleNamespace(
                    is_error=False,
                    structured_content={"text": "hello"},
                    content=(),
                )
            block = SimpleNamespace(
                model_dump=lambda **_kwargs: {"type": "text", "text": "hello"}
            )
            return SimpleNamespace(
                is_error=False,
                structured_content=None,
                content=(block,),
            )

    session = Session()
    transport = FakeTransport(session)
    spec = (
        await discover_mcp_tools(FakeTransport(PaginatedSession()), namespace="test")
    )[0]
    executor = MCPToolExecutor(transport, remote_name="remote_read")
    call = ToolCall("mcp", "read", {"path": "notes.txt"})

    assert await executor.execute(spec, call, ToolExecutionContext()) == {
        "text": "hello"
    }
    session.structured = False
    assert await executor.execute(spec, call, ToolExecutionContext()) == {
        "content": [{"type": "text", "text": "hello"}]
    }


@pytest.mark.asyncio
async def test_mcp_error_and_transport_failure_are_classified() -> None:
    spec = (
        await discover_mcp_tools(FakeTransport(PaginatedSession()), namespace="test")
    )[0]
    call = ToolCall("mcp", "read", {})

    class ErrorSession:
        async def call_tool(self, _name, *, arguments):
            return SimpleNamespace(is_error=True, content=())

    with pytest.raises(ToolExecutionError) as raised:
        await MCPToolExecutor(FakeTransport(ErrorSession())).execute(
            spec, call, ToolExecutionContext()
        )
    assert (raised.value.kind, raised.value.code, raised.value.retryable) == (
        "remote_error",
        "mcp_tool_error",
        False,
    )

    class BrokenTransport:
        @asynccontextmanager
        async def session(self):
            raise ConnectionError("private transport detail")
            yield

    with pytest.raises(ToolExecutionError) as raised:
        await MCPToolExecutor(BrokenTransport()).execute(
            spec, call, ToolExecutionContext()
        )
    assert (raised.value.kind, raised.value.code, raised.value.retryable) == (
        "transport_error",
        "mcp_transport",
        True,
    )
    assert "private transport detail" not in str(raised.value)


@pytest.mark.asyncio
async def test_registration_is_versioned_and_collision_safe() -> None:
    specs = await discover_mcp_tools(
        FakeTransport(PaginatedSession()), namespace="test"
    )
    registry = ExecutorRegistry()
    transport = FakeTransport(SimpleNamespace())
    register_mcp_tools(registry, transport, specs)
    assert registry.resolve("test.read", "1") is not None
    with pytest.raises(ValueError, match="already registered"):
        register_mcp_tools(registry, transport, specs)
