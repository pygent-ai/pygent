from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mcp
import pytest
from mcp import types as mcp_types

from pygent import AIMessage, Context
from pygent.runtime import LocalRuntime
from pygent.tool import (
    ToolAuthorizationDecision,
    ToolCall,
    ToolCallLayer,
    ToolExecutionContext,
    ToolExecutionError,
    ToolKit,
)
from pygent.tool.mcp import MCPClientPool, MCPStdioTransport


def allow(request, _context):
    return ToolAuthorizationDecision(
        call_id=request.call.call_id,
        allowed=True,
        reason_code="allowed",
    )


def mcp_tool(name: str, *, description: str = "fake tool") -> mcp_types.Tool:
    return mcp_types.Tool(
        name=name,
        description=description,
        input_schema={"type": "object", "properties": {}},
        output_schema=None,
        annotations=None,
    )


class FakePoolSession:
    def __init__(self, transport: FakePoolTransport) -> None:
        self._transport = transport

    async def initialize(self) -> None:
        self._transport.initialize_calls += 1
        if self._transport.initialize_error is not None:
            raise self._transport.initialize_error

    async def list_tools(self, *, params: Any) -> mcp_types.ListToolsResult:
        assert params.cursor is None
        return mcp_types.ListToolsResult(
            tools=list(self._transport.tools), next_cursor=None
        )

    async def call_tool(self, name: str, *, arguments: Any) -> SimpleNamespace:
        transport = self._transport
        transport.calls += 1
        transport.concurrent += 1
        transport.peak_concurrent = max(
            transport.peak_concurrent, transport.concurrent
        )
        try:
            if transport.broken:
                raise ConnectionError("simulated session death")
            if transport.gate is not None:
                await transport.gate.wait()
            return SimpleNamespace(
                is_error=False,
                structured_content={"name": name, "arguments": dict(arguments)},
                content=(),
            )
        finally:
            transport.concurrent -= 1


class FakePoolTransport:
    """In-memory MCPTransport counting session generations and calls."""

    def __init__(
        self,
        tools: list[mcp_types.Tool],
        *,
        open_error: Exception | None = None,
        initialize_error: Exception | None = None,
    ) -> None:
        self.tools = list(tools)
        self.open_error = open_error
        self.initialize_error = initialize_error
        self.openings = 0
        self.initialize_calls = 0
        self.live_sessions = 0
        self.calls = 0
        self.concurrent = 0
        self.peak_concurrent = 0
        self.broken = False
        self.gate: asyncio.Event | None = None

    @asynccontextmanager
    async def session(self) -> AsyncIterator[FakePoolSession]:
        self.openings += 1
        if self.open_error is not None:
            raise self.open_error
        self.live_sessions += 1
        try:
            yield FakePoolSession(self)
        finally:
            self.live_sessions -= 1


async def execute_once(tool: Any, call_id: str) -> object:
    return await tool.executor.execute(
        tool.spec,
        ToolCall(call_id, tool.name, {"value": 2}),
        ToolExecutionContext(),
    )


async def test_discovery_intersection_and_deterministic_naming() -> None:
    pool = MCPClientPool(
        servers={
            "github": FakePoolTransport([mcp_tool("get_file"), mcp_tool("search")]),
            "search": FakePoolTransport([mcp_tool("get_file"), mcp_tool("web")]),
        }
    )
    await pool.connect()

    everything = pool.tools()
    assert [(item.server, item.name) for item in everything] == [
        ("github", "get_file"),
        ("github", "search"),
        ("search", "get_file"),
        ("search", "web"),
    ]
    by_key = {(item.server, item.name): item for item in everything}
    assert by_key[("github", "search")].spec.definition.name == "search"
    assert by_key[("search", "web")].spec.definition.name == "web"
    # Same remote name on several servers is qualified deterministically.
    assert by_key[("github", "get_file")].spec.definition.name == "github.get_file"
    assert by_key[("search", "get_file")].spec.definition.name == "search.get_file"
    assert [item.spec.tool_id for item in everything] == [
        "mcp.github.get_file",
        "mcp.github.search",
        "mcp.search.get_file",
        "mcp.search.web",
    ]
    assert pool.tools() is pool.tools()

    intersected = pool.tools(servers=["github"], names=["get_file"])
    assert [(item.server, item.name) for item in intersected] == [
        ("github", "get_file")
    ]
    assert pool.tools(filter=lambda tool: tool.server == "search") == tuple(
        item for item in everything if item.server == "search"
    )
    assert pool.tools(servers=["missing"]) == ()
    # Empty sequences select nothing; None leaves the dimension unrestricted.
    assert pool.tools(servers=[]) == ()
    assert pool.tools(names=[]) == ()
    assert pool.tools(filter=lambda tool: False) == ()
    assert pool.tools(servers=["github"], names=["web"]) == ()
    await pool.aclose()


async def test_calls_reuse_one_live_session() -> None:
    transport = FakePoolTransport([mcp_tool("double")])
    pool = MCPClientPool(servers={"a": transport})
    await pool.connect()
    assert transport.openings == 1
    assert transport.initialize_calls == 1

    tool = pool.tools()[0]
    for index in range(5):
        result = await execute_once(tool, f"c{index}")
        assert result == {"name": "double", "arguments": {"value": 2}}
    assert transport.openings == 1
    assert transport.initialize_calls == 1
    assert transport.live_sessions == 1
    await pool.aclose()


async def test_shared_pool_serves_distinct_session_selections() -> None:
    pool = MCPClientPool(
        servers={
            "github": FakePoolTransport([mcp_tool("get_file")]),
            "files": FakePoolTransport([mcp_tool("read"), mcp_tool("write")]),
        }
    )
    await pool.connect()
    runtime = LocalRuntime()
    runtime.register_tools(pool.executors())
    with pytest.raises(ValueError, match="already registered"):
        runtime.register_tools(pool.executors())

    github_kit = ToolKit(*pool.tools(servers=["github"]))
    files_kit = ToolKit(*pool.tools(servers=["files"]))
    assert {item.name for item in github_kit.definitions} == {"get_file"}
    assert {item.name for item in files_kit.definitions} == {"read", "write"}

    bound_github = runtime.bind(
        ToolCallLayer(tools=github_kit.specs, authorization_adapter=allow)
    )
    bound_files = runtime.bind(
        ToolCallLayer(tools=files_kit.specs, authorization_adapter=allow)
    )
    github_message, _ = await bound_github.invoke(
        AIMessage(tool_calls=(ToolCall("g1", "get_file", {"path": "a.txt"}),)),
        github_kit.make_visible_in(Context()),
    )
    files_message, _ = await bound_files.invoke(
        AIMessage(tool_calls=(ToolCall("f1", "read", {"path": "b.txt"}),)),
        files_kit.make_visible_in(Context()),
    )
    assert github_message.results[0].status == "succeeded"
    assert github_message.results[0].output.to_dict() == {
        "name": "get_file",
        "arguments": {"path": "a.txt"},
    }
    assert files_message.results[0].status == "succeeded"
    await runtime.close()
    await pool.aclose()


async def test_broken_session_reconnects_lazily_with_initialize() -> None:
    transport = FakePoolTransport([mcp_tool("double")])
    pool = MCPClientPool(servers={"a": transport})
    await pool.connect()
    tool = pool.tools()[0]
    assert await execute_once(tool, "c1") is not None

    transport.broken = True
    with pytest.raises(ToolExecutionError) as raised:
        await execute_once(tool, "c2")
    assert (raised.value.kind, raised.value.code, raised.value.retryable) == (
        "transport_error",
        "mcp_transport",
        True,
    )
    assert transport.openings == 1

    transport.broken = False
    result = await execute_once(tool, "c3")
    assert result == {"name": "double", "arguments": {"value": 2}}
    assert transport.openings == 2
    assert transport.initialize_calls == 2
    assert transport.live_sessions == 1
    await pool.aclose()


async def test_reconnect_failure_returns_retryable_transport_error() -> None:
    transport = FakePoolTransport([mcp_tool("double")])
    pool = MCPClientPool(servers={"a": transport})
    await pool.connect()
    tool = pool.tools()[0]

    transport.broken = True
    with pytest.raises(ToolExecutionError):
        await execute_once(tool, "c1")
    transport.broken = False
    transport.open_error = RuntimeError("spawn refused")
    with pytest.raises(ToolExecutionError) as raised:
        await execute_once(tool, "c2")
    assert (raised.value.kind, raised.value.retryable) == ("transport_error", True)
    assert "spawn refused" not in str(raised.value)
    assert transport.live_sessions == 0
    await pool.aclose()


async def test_refresh_rebuilds_snapshot_and_keeps_assembled_references() -> None:
    transport = FakePoolTransport([mcp_tool("double")])
    pool = MCPClientPool(servers={"a": transport})
    await pool.connect()
    kit = ToolKit(*pool.tools())
    before = pool.tools()[0].spec

    transport.tools.append(mcp_tool("triple"))
    await pool.refresh()
    assert [item.name for item in pool.tools()] == ["double", "triple"]
    assert [item.name for item in kit.definitions] == ["double"]
    refreshed = pool.tools()[0].spec
    assert refreshed.tool_id == before.tool_id
    assert refreshed.version == before.version

    with pytest.raises(RuntimeError, match="not connected"):
        await MCPClientPool(servers={"a": FakePoolTransport([])}).refresh()
    await pool.aclose()


async def test_conflicting_refresh_keeps_previous_catalog_serving() -> None:
    first = FakePoolTransport([mcp_tool("b.get_file")])
    second = FakePoolTransport([])
    pool = MCPClientPool(servers={"a": first, "a.b": second})
    await pool.connect()
    before = pool.tools()[0]
    assert before.name == "b.get_file"

    second.tools.append(mcp_tool("get_file"))
    with pytest.raises(ValueError, match="identity conflict"):
        await pool.refresh()
    assert pool.tools() == (before,)
    assert await execute_once(before, "c1") == {
        "name": "b.get_file",
        "arguments": {"value": 2},
    }

    second.tools.clear()
    await pool.refresh()
    assert pool.tools()[0].spec == before.spec
    await pool.aclose()


async def test_definition_change_gets_a_new_version_without_silent_reuse() -> None:
    transport = FakePoolTransport([mcp_tool("double", description="v1 contract")])
    pool = MCPClientPool(servers={"a": transport})
    await pool.connect()
    before = pool.tools()[0].spec

    transport.tools[0] = mcp_tool("double", description="v2 contract")
    await pool.refresh()
    after = pool.tools()[0].spec
    assert after.tool_id == before.tool_id
    assert after.version != before.version
    assert after.version.startswith("1-")
    await pool.aclose()


async def test_visible_rename_gets_a_new_version_with_the_contract() -> None:
    first = FakePoolTransport([mcp_tool("get_file")])
    second = FakePoolTransport([])
    pool = MCPClientPool(servers={"a": first, "b": second})
    await pool.connect()
    before = pool.tools()[0]
    assert before.name == "get_file"

    second.tools.append(mcp_tool("get_file"))
    await pool.refresh()
    renamed = next(item for item in pool.tools() if item.server == "a")
    assert renamed.spec.definition.name == "a.get_file"
    assert renamed.spec.version != before.spec.version
    assert renamed.spec.tool_id == before.spec.tool_id

    second.tools.clear()
    await pool.refresh()
    assert pool.tools()[0].spec == before.spec
    await pool.aclose()


async def test_aclose_joins_in_flight_calls_then_rejects_new_calls() -> None:
    transport = FakePoolTransport([mcp_tool("slow")])
    pool = MCPClientPool(servers={"a": transport})
    await pool.connect()
    tool = pool.tools()[0]

    transport.gate = asyncio.Event()
    call_task = asyncio.create_task(execute_once(tool, "c1"))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert transport.calls == 1
    aclose_task = asyncio.create_task(pool.aclose())
    await asyncio.sleep(0)
    assert not aclose_task.done()
    transport.gate.set()
    await asyncio.gather(call_task, aclose_task)
    assert call_task.result() == {"name": "slow", "arguments": {"value": 2}}

    with pytest.raises(ToolExecutionError) as raised:
        await execute_once(tool, "c2")
    assert (raised.value.kind, raised.value.retryable) == ("transport_error", True)
    assert transport.live_sessions == 0


async def test_aclose_rejects_semaphore_waiters_without_deadlock() -> None:
    transport = FakePoolTransport([mcp_tool("work")])
    pool = MCPClientPool(servers={"a": transport}, max_concurrency=1)
    await pool.connect()
    tool = pool.tools()[0]

    transport.gate = asyncio.Event()
    first = asyncio.create_task(execute_once(tool, "c1"))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert transport.calls == 1
    waiter = asyncio.create_task(execute_once(tool, "c2"))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert transport.calls == 1

    aclose_task = asyncio.create_task(pool.aclose())
    await asyncio.sleep(0)
    assert not aclose_task.done()
    transport.gate.set()
    with pytest.raises(ToolExecutionError) as raised:
        await waiter
    assert (raised.value.kind, raised.value.retryable) == ("transport_error", True)
    await asyncio.gather(first, aclose_task)
    assert transport.openings == 1
    assert transport.live_sessions == 0


async def test_per_server_concurrency_cap_serializes_calls() -> None:
    transport = FakePoolTransport([mcp_tool("work")])
    pool = MCPClientPool(servers={"a": transport}, max_concurrency=1)
    await pool.connect()
    tool = pool.tools()[0]

    transport.gate = asyncio.Event()
    first = asyncio.create_task(execute_once(tool, "c1"))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert transport.calls == 1
    second = asyncio.create_task(execute_once(tool, "c2"))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert transport.calls == 1
    assert transport.concurrent == 1
    transport.gate.set()
    await asyncio.gather(first, second)
    assert transport.calls == 2
    assert transport.peak_concurrent == 1
    await pool.aclose()


async def test_unlimited_multiplexing_runs_parallel_calls_on_one_session() -> None:
    transport = FakePoolTransport([mcp_tool("work")])
    pool = MCPClientPool(servers={"a": transport})
    await pool.connect()
    tool = pool.tools()[0]

    transport.gate = asyncio.Event()
    first = asyncio.create_task(execute_once(tool, "c1"))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    second = asyncio.create_task(execute_once(tool, "c2"))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert transport.calls == 2
    assert transport.peak_concurrent == 2
    transport.gate.set()
    await asyncio.gather(first, second)
    assert transport.openings == 1
    await pool.aclose()


async def test_connect_failures_raise_transport_errors() -> None:
    pool = MCPClientPool(
        servers={
            "a": FakePoolTransport(
                [mcp_tool("double")], initialize_error=RuntimeError("handshake refused")
            )
        }
    )
    with pytest.raises(ToolExecutionError) as raised:
        await pool.connect()
    assert (raised.value.kind, raised.value.retryable) == ("transport_error", True)
    assert "handshake refused" not in str(raised.value)
    assert pool.tools() == ()
    await pool.aclose()


async def test_duplicate_remote_names_within_a_server_are_rejected() -> None:
    pool = MCPClientPool(
        servers={"a": FakePoolTransport([mcp_tool("dup"), mcp_tool("dup")])}
    )
    with pytest.raises(ValueError, match="duplicate tool names"):
        await pool.connect()
    await pool.aclose()


async def test_identity_conflicts_across_dotted_names_are_detected() -> None:
    pool = MCPClientPool(
        servers={
            "a": FakePoolTransport([mcp_tool("b.c")]),
            "a.b": FakePoolTransport([mcp_tool("c")]),
        }
    )
    try:
        with pytest.raises(ValueError, match="identity conflict"):
            await pool.connect()
    finally:
        await pool.aclose()


def test_pool_construction_validates_servers_and_limits() -> None:
    with pytest.raises(TypeError, match="mapping"):
        MCPClientPool(servers=[("a", FakePoolTransport([]))])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="session"):
        MCPClientPool(servers={"a": object()})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown MCP server concurrency"):
        MCPClientPool(
            servers={"a": FakePoolTransport([])},
            max_concurrency={"missing": 2},
        )
    with pytest.raises(ValueError, match="positive integer"):
        MCPClientPool(servers={"a": FakePoolTransport([])}, max_concurrency=0)


async def test_pool_runs_real_stdio_server_end_to_end() -> None:
    server = Path(__file__).with_name("stdio_server.py")
    env = dict(os.environ)
    site_packages = str(Path(mcp.__file__).parent.parent)
    env["PYTHONPATH"] = os.pathsep.join((site_packages, *sys.path))
    virtual_env = Path.cwd() / ".venv"
    python = virtual_env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    pool = MCPClientPool(
        servers={
            "test": MCPStdioTransport(str(python), args=(str(server),), env=env)
        }
    )
    await pool.connect()
    tool = pool.tools()[0]
    assert tool.server == "test"
    assert tool.name == "double"
    assert tool.spec.tool_id == "mcp.test.double"

    runtime = LocalRuntime()
    runtime.register_tools(pool.executors())
    bound = runtime.bind(
        ToolCallLayer(tools=(tool.spec,), authorization_adapter=allow)
    )
    message, _ = await bound.invoke(
        AIMessage(tool_calls=(ToolCall("pool-1", "double", {"value": 3}),)),
        Context(tools=(tool.spec.definition,)),
    )
    assert message.results[0].status == "succeeded", message.results[0].error
    assert message.results[0].output.to_dict() == {"result": 6}

    await runtime.close()
    await pool.aclose()
