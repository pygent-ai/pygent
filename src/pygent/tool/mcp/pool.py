"""Deployment-side MCP connection pool and unified tool assembly."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from typing import Any, cast

from pygent.core import FrozenJsonObject, thaw_json

from ..executors import ToolExecutionError, ToolExecutor
from ..types import ToolDefinition, ToolSpec
from .client import MCPToolExecutor, MCPTransport, discover_mcp_tools


@dataclass(frozen=True, slots=True)
class MCPTool:
    """Internal assembly binding for one discovered MCP tool.

    The pool hands these values to :class:`~pygent.tool.ToolKit`, which reads
    ``spec`` for the portable declaration and installs ``executor`` unchanged
    into Runtime registration. ``name`` is the remote MCP tool name and
    ``spec.definition.name`` is the model-visible name, which the pool
    qualifies with the server name only when several servers expose the same
    remote name. Live session objects and pool state never appear on this
    value; it is not a public export and must not be kept in Module state.
    """

    server: str
    name: str
    spec: ToolSpec
    executor: ToolExecutor


def _definition_fingerprint(definition: ToolDefinition) -> str:
    """Return a stable short hash of one assembled tool contract.

    Identity stays stable while the contract is unchanged, and a changed
    contract can never silently reuse a previously registered
    ``(tool_id, version)``: the fingerprint becomes part of the version. The
    model-visible name belongs to the contract, so cross-server naming
    changes produce a new identity too.
    """

    payload = {
        "name": definition.name,
        "description": definition.description,
        "parameters": thaw_json(cast(FrozenJsonObject, definition.parameters)),
        "output_schema": (
            None
            if definition.output_schema is None
            else thaw_json(cast(FrozenJsonObject, definition.output_schema))
        ),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:8]


def _build_catalog(
    entries: Mapping[str, tuple[ToolSpec, ...]],
    connections: Mapping[str, _PooledServerConnection],
    version: str,
) -> tuple[MCPTool, ...]:
    """Assemble the MCPTool catalog from one candidate discovery state.

    The function is pure: duplicate remote names, model-visible name
    conflicts and identity conflicts raise before any caller mutates pool
    state, so a discovery result and its catalog are always committed or
    rejected together.
    """

    counts = Counter(
        item.definition.name for specs in entries.values() for item in specs
    )
    catalog: list[MCPTool] = []
    visible_names: set[str] = set()
    tool_ids: set[str] = set()
    for server, specs in entries.items():
        connection = connections[server]
        for item in specs:
            remote_name = item.definition.name
            visible = (
                remote_name
                if counts[remote_name] == 1
                else f"{server}.{remote_name}"
            )
            if visible in visible_names:
                raise ValueError(
                    f"MCP tool model-visible name conflict: {visible!r}"
                )
            if item.tool_id in tool_ids:
                raise ValueError(f"MCP tool identity conflict: {item.tool_id}")
            visible_names.add(visible)
            tool_ids.add(item.tool_id)
            definition = replace(item.definition, name=visible)
            catalog.append(
                MCPTool(
                    server=server,
                    name=remote_name,
                    spec=replace(
                        item,
                        version=f"{version}-{_definition_fingerprint(definition)}",
                        definition=definition,
                    ),
                    executor=MCPToolExecutor(
                        connection, remote_name=remote_name
                    ),
                )
            )
    return tuple(catalog)


def _validated_limit(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("max_concurrency must be a positive integer or None")
    return value


def _marks_broken_session(exc: BaseException) -> bool:
    """Whether an exception leaving a session body marks the session broken.

    Cancellation of one caller does not prove session death, and a
    ``remote_error`` is a server-reported logical outcome on a usable
    session; every other failure is treated as transport death so the next
    call reconnects lazily.
    """

    if isinstance(exc, asyncio.CancelledError):
        return False
    if isinstance(exc, ToolExecutionError):
        return exc.kind != "remote_error"
    return True


class _LiveSession:
    """One transport context owned by a dedicated task, plus its user count."""

    __slots__ = ("close_event", "discarded", "owner", "session", "users")

    def __init__(self) -> None:
        self.close_event = asyncio.Event()
        self.discarded = False
        self.owner: asyncio.Task[None] | None = None
        self.session: Any = None
        self.users = 0


class _BoundSessionTransport:
    """Present one already-acquired session through the MCPTransport protocol."""

    def __init__(self, session: Any) -> None:
        self._session = session

    @asynccontextmanager
    async def session(self) -> AsyncIterator[Any]:
        yield self._session


class _PooledServerConnection:
    """Persistent per-server session source satisfying MCPTransport.

    Anyio cancel scopes must be exited by the task that entered them, so one
    owner task per session generation enters and later exits the transport
    contexts. Callers acquire users of the live session through
    :meth:`session`; a broken session is reconnected lazily by the next
    acquisition, which re-initializes and re-discovers the tool list.
    """

    def __init__(
        self,
        name: str,
        transport: MCPTransport,
        *,
        namespace: str,
        version: str,
        max_concurrency: int | None,
        on_rediscover: Callable[[str, tuple[ToolSpec, ...]], None],
    ) -> None:
        self.name = name
        self._transport = transport
        self._namespace = namespace
        self._version = version
        self._on_rediscover = on_rediscover
        self._semaphore = (
            None if max_concurrency is None else asyncio.Semaphore(max_concurrency)
        )
        self._lock = asyncio.Lock()
        self._current: _LiveSession | None = None
        self._active = 0
        self._idle = asyncio.Event()
        self._idle.set()
        self._closing = False
        self._closed = False
        self._owner_tasks: set[asyncio.Task[None]] = set()

    @asynccontextmanager
    async def session(self) -> AsyncIterator[Any]:
        """Yield the live session without cold-connecting per call."""

        if self._closing:
            raise self._unavailable_error("MCP connection pool is closing")
        if self._semaphore is not None:
            await self._semaphore.acquire()
        try:
            live = await self._acquire()
            try:
                yield live.session
            except BaseException as exc:
                if _marks_broken_session(exc):
                    live.discarded = True
                raise
            finally:
                await self._release(live)
        finally:
            if self._semaphore is not None:
                self._semaphore.release()

    async def rediscover(self) -> tuple[ToolSpec, ...]:
        """Re-run tools/list on the live session and record the result."""

        specs = await discover_mcp_tools(
            self, namespace=self._namespace, version=self._version
        )
        self._on_rediscover(self.name, specs)
        return specs

    def request_close(self) -> None:
        self._closing = True

    async def wait_idle(self) -> None:
        """Wait until no call or session open is in flight."""

        await self._idle.wait()

    async def close(self) -> None:
        """Close the current session after its in-flight users have drained."""

        async with self._lock:
            self._closing = True
            current = self._current
            self._current = None
            self._closed = True
            if current is not None:
                current.discarded = True
                current.close_event.set()
        await asyncio.gather(*self._owner_tasks, return_exceptions=True)

    async def _acquire(self) -> _LiveSession:
        if self._closing:
            raise self._unavailable_error("MCP connection pool is closing")
        async with self._lock:
            if self._closing:
                raise self._unavailable_error("MCP connection pool is closing")
            current = self._current
            if current is not None and not current.discarded:
                current.users += 1
                self._begin_active()
                return current
            if self._closed:
                raise self._unavailable_error("MCP connection pool is closed")
            self._begin_active()
            try:
                live = await self._open_session()
            except BaseException:
                self._end_active()
                raise
            self._current = live
            live.users = 1
            return live

    async def _open_session(self) -> _LiveSession:
        live = _LiveSession()
        entered: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        owner = asyncio.create_task(
            self._own_session(live, entered),
            name=f"pygent-mcp-{self.name}-session-owner",
        )
        live.owner = owner
        self._owner_tasks.add(owner)
        owner.add_done_callback(self._owner_finished)
        try:
            session = await entered
        except asyncio.CancelledError:
            live.close_event.set()
            raise
        except BaseException as exc:
            live.close_event.set()
            raise self._unavailable_error(
                f"MCP server {self.name!r} session could not be opened"
            ) from exc
        try:
            await session.initialize()
            specs = await discover_mcp_tools(
                _BoundSessionTransport(session),
                namespace=self._namespace,
                version=self._version,
            )
        except asyncio.CancelledError:
            live.discarded = True
            live.close_event.set()
            raise
        except BaseException as exc:
            live.discarded = True
            live.close_event.set()
            raise self._unavailable_error(
                f"MCP server {self.name!r} session is unavailable"
            ) from exc
        try:
            self._on_rediscover(self.name, specs)
        except BaseException:
            live.discarded = True
            live.close_event.set()
            raise
        return live

    async def _own_session(
        self, live: _LiveSession, entered: asyncio.Future[Any]
    ) -> None:
        """Enter and later exit the transport contexts for one generation.

        anyio cancel scopes must be exited by the task that entered them, so
        closing signals the owner instead of exiting from a caller's task.
        """

        try:
            async with self._transport.session() as session:
                live.session = session
                entered.set_result(session)
                await live.close_event.wait()
        except BaseException as exc:  # noqa: BLE001 - anyio scope teardown must be relayed by its entering task
            if not entered.done():
                entered.set_exception(exc)
                entered.exception()  # consumed; the opener re-raises its own error
            else:
                live.discarded = True
                live.close_event.set()

    def _owner_finished(self, task: asyncio.Task[None]) -> None:
        self._owner_tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            task.exception()  # consumed; session failures surface through calls

    async def _release(self, live: _LiveSession) -> None:
        live.users -= 1
        self._end_active()
        if live.discarded and live.users == 0:
            live.close_event.set()
            owner = live.owner
            if owner is not None:
                await asyncio.shield(asyncio.gather(owner, return_exceptions=True))

    def _begin_active(self) -> None:
        self._active += 1
        self._idle.clear()

    def _end_active(self) -> None:
        self._active -= 1
        if self._active == 0:
            self._idle.set()

    def _unavailable_error(self, message: str) -> ToolExecutionError:
        return ToolExecutionError(
            message,
            kind="transport_error",
            code="mcp_transport",
            retryable=True,
            side_effect_committed=None,
        )


class MCPClientPool:
    """Deployment-side pool of one persistent session per MCP server.

    The pool owns MCP server configuration, live sessions and the discovered
    tool catalog. It exposes portable tool selections for
    :class:`~pygent.tool.ToolKit` assembly and ``(ToolSpec, ToolExecutor)``
    pairs for the existing Runtime registration; it never binds a Runtime,
    manages Session state, or places live connections into portable execution
    state. Every executor call runs over the shared live session instead of
    cold-connecting per call.
    """

    def __init__(
        self,
        servers: Mapping[str, MCPTransport],
        *,
        namespace: str = "mcp",
        version: str = "1",
        max_concurrency: int | Mapping[str, int | None] | None = None,
    ) -> None:
        """Store server configuration without opening connections.

        Args:
            servers: Server name to transport mapping; insertion order breaks
                ties deterministically for catalog order and visible naming.
            namespace: Tool identity prefix; each tool becomes
                ``<namespace>.<server>.<remote tool name>``.
            version: Base version component; the discovered contract adds a
                stable fingerprint so a changed definition can never silently
                reuse a previously registered ``(tool_id, version)``.
            max_concurrency: Per-server call concurrency cap protecting
                servers that do not accept parallel requests; an int applies
                to every server, a mapping configures servers individually,
                and ``None`` (the default) means unlimited multiplexing.
        """

        if not isinstance(servers, Mapping):
            raise TypeError("servers must be a mapping of names to transports")
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("namespace must be non-empty")
        if not isinstance(version, str) or not version:
            raise ValueError("version must be non-empty")
        self._servers: dict[str, MCPTransport] = {}
        for name, transport in servers.items():
            if not isinstance(name, str) or not name:
                raise ValueError("server names must be non-empty strings")
            if not callable(getattr(transport, "session", None)):
                raise TypeError(
                    f"server {name!r} transport must expose a session() context manager"
                )
            self._servers[name] = transport
        if isinstance(max_concurrency, Mapping):
            unknown = set(max_concurrency) - set(self._servers)
            if unknown:
                raise ValueError(
                    f"unknown MCP server concurrency limit: {sorted(unknown)}"
                )
            limits = {
                name: _validated_limit(max_concurrency[name])
                for name in self._servers
            }
        else:
            shared = _validated_limit(max_concurrency)
            limits = {name: shared for name in self._servers}
        self._connections = {
            name: _PooledServerConnection(
                name,
                transport,
                namespace=f"{namespace}.{name}",
                version=version,
                max_concurrency=limits[name],
                on_rediscover=self._record_discovery,
            )
            for name, transport in self._servers.items()
        }
        self._entries: dict[str, tuple[ToolSpec, ...]] = {}
        self._catalog: tuple[MCPTool, ...] = ()
        self._namespace = namespace
        self._version = version
        self._connected = False
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None

    async def connect(self) -> None:
        """Open one persistent session per configured server and discover tools.

        Connecting an already connected server re-discovers its tools, so a
        retry after a partial failure only opens the missing sessions.
        """

        if self._closing:
            raise RuntimeError("MCP connection pool is closing")
        for name in self._servers:
            await self._connections[name].rediscover()
        self._connected = True

    async def refresh(self) -> None:
        """Re-run tools/list on every server and rebuild the catalog snapshot.

        Already assembled ToolKit values and running Executions keep their
        previous snapshot; re-assemble from :meth:`tools` to adopt the
        refreshed catalog.
        """

        if not self._connected:
            raise RuntimeError("MCP connection pool is not connected")
        if self._closing:
            raise RuntimeError("MCP connection pool is closing")
        for name in self._servers:
            await self._connections[name].rediscover()

    def tools(
        self,
        servers: Sequence[str] | None = None,
        names: Sequence[str] | None = None,
        filter: Callable[[MCPTool], bool] | None = None,
    ) -> tuple[MCPTool, ...]:
        """Return the current catalog snapshot narrowed by three filters.

        The arguments intersect: ``None`` leaves a dimension unrestricted and
        an empty sequence selects nothing for that dimension. ``names``
        matches the remote MCP tool name (``MCPTool.name``), not the
        model-visible name, and ``filter`` runs at assembly time over the
        fixed snapshot. The returned tuple is frozen until the next
        ``connect()`` discovery, reconnect or :meth:`refresh`.
        """

        selected = self._catalog
        if servers is not None:
            allowed = set(servers)
            selected = tuple(item for item in selected if item.server in allowed)
        if names is not None:
            allowed_names = set(names)
            selected = tuple(item for item in selected if item.name in allowed_names)
        if filter is not None:
            selected = tuple(item for item in selected if filter(item))
        return selected

    def executors(self) -> tuple[tuple[ToolSpec, ToolExecutor], ...]:
        """Return every discovered tool for deployment-time registration."""

        return tuple((item.spec, item.executor) for item in self._catalog)

    async def aclose(self) -> None:
        """Join in-flight calls, then close every pooled session.

        Aligns with ``DefaultModelInvoker.aclose()``: closing is idempotent
        and shielded, calls arriving while closing are rejected, in-flight
        calls are joined before sessions close, and calls after closing raise
        a retryable transport error.
        """

        self._closing = True
        for connection in self._connections.values():
            connection.request_close()
        if self._close_task is None:
            self._close_task = asyncio.create_task(
                self._close_all(), name="pygent-mcp-pool-close"
            )
        await asyncio.shield(self._close_task)

    async def _close_all(self) -> None:
        for connection in self._connections.values():
            await connection.wait_idle()
        for connection in self._connections.values():
            await connection.close()

    def _record_discovery(self, server: str, specs: tuple[ToolSpec, ...]) -> None:
        """Commit one server's discovery together with the whole catalog.

        The candidate catalog is built before any pool state changes, so a
        conflicting discovery leaves the previous consistent snapshot serving
        calls and later refreshes. Rediscovering an unchanged tool list is a
        no-op: catalog entries and their executors keep their identity.
        """

        if self._entries.get(server) == specs:
            return
        remote_names = [item.definition.name for item in specs]
        duplicates = sorted(
            {name for name in remote_names if remote_names.count(name) > 1}
        )
        if duplicates:
            raise ValueError(
                f"MCP server {server!r} reported duplicate tool names: {duplicates}"
            )
        entries = dict(self._entries)
        entries[server] = specs
        catalog = _build_catalog(entries, self._connections, self._version)
        self._entries = entries
        self._catalog = catalog


__all__ = ["MCPClientPool"]
