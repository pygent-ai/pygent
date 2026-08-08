"""Deployment-side tool executors and the stable executor registry."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from typing import Protocol, TypeAlias, cast, runtime_checkable
from uuid import uuid4

import httpx

from pygent.core import (
    Context,
    FrozenJsonObject,
    JsonValue,
    Message,
    ModuleDependency,
    active_infrastructure,
    freeze_json,
    thaw_json,
)

from .types import (
    ToolCall,
    ToolResult,
    ToolResultStatus,
    ToolSideEffect,
    ToolSpec,
    ToolTask,
    ToolTaskState,
)

ToolHandler: TypeAlias = Callable[[Mapping[str, JsonValue]], object | Awaitable[object]]
ToolTaskExecution: TypeAlias = Callable[
    ["ToolSpec", "ToolCall", "ToolExecutionContext"], Awaitable[object]
]
AgentToolRequestBuilder: TypeAlias = Callable[
    ["ToolSpec", "ToolCall"], tuple[Message, Context]
]
AgentToolResultBuilder: TypeAlias = Callable[
    [Message, Context], object | Awaitable[object]
]
ToolEventEmitter: TypeAlias = Callable[[str, Mapping[str, JsonValue]], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    """Effective budget and event channel supplied to every Tool executor."""

    deadline: float | None = None
    emit: ToolEventEmitter | None = None
    execution_id: str | None = None
    task_id: str | None = None
    recovery: bool = False

    def __post_init__(self) -> None:
        for name in ("execution_id", "task_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a non-empty string or None")
        if not isinstance(self.recovery, bool):
            raise TypeError("recovery must be a bool")


@dataclass(frozen=True, slots=True)
class SandboxExecutorSupport:
    """Deployment proof advertised by an external-sandbox ToolExecutor."""

    profiles: tuple[str, ...]
    durable_reconnect: bool = False
    deployment_fingerprint: str | None = None

    def __post_init__(self) -> None:
        profiles = tuple(self.profiles)
        if not profiles or any(not isinstance(item, str) or not item for item in profiles):
            raise ValueError("profiles must contain non-empty strings")
        if len(profiles) != len(set(profiles)):
            raise ValueError("profiles must be unique")
        if not isinstance(self.durable_reconnect, bool):
            raise TypeError("durable_reconnect must be a bool")
        if self.deployment_fingerprint is not None and (
            not isinstance(self.deployment_fingerprint, str)
            or not self.deployment_fingerprint
        ):
            raise ValueError("deployment_fingerprint must be non-empty or None")
        if self.durable_reconnect and self.deployment_fingerprint is None:
            raise ValueError(
                "durable_reconnect requires a stable deployment_fingerprint"
            )
        object.__setattr__(self, "profiles", profiles)

    def capability_for_fingerprint(self) -> str | None:
        if self.deployment_fingerprint is None:
            return None
        digest = hashlib.sha256(self.deployment_fingerprint.encode()).hexdigest()
        return f"tool.sandbox.deployment.{digest}"


@dataclass(frozen=True, slots=True)
class ToolTaskAdmission:
    """Structured outcome of managed detached ToolTask admission."""

    task: ToolTask | None = None
    error: str | None = None
    error_kind: str | None = None
    error_code: str | None = None
    missing_capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (self.task is None) == (self.error_code is None):
            raise ValueError("admission must contain exactly one of task or error_code")
        for name in ("error", "error_kind", "error_code"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a non-empty string or None")
        capabilities = tuple(self.missing_capabilities)
        if any(not isinstance(item, str) or not item for item in capabilities):
            raise ValueError("missing_capabilities must contain non-empty strings")
        object.__setattr__(self, "missing_capabilities", capabilities)


class ToolExecution:
    """Control plane for one ToolRunner-owned operation."""

    def __init__(self, operation: Awaitable[ToolResult]) -> None:
        self._operation = operation
        self._owner: asyncio.Task[object] | None = None
        self._future: asyncio.Future[ToolResult] | None = None

    async def result(self) -> ToolResult:
        if self._future is None:
            self._future = asyncio.get_running_loop().create_future()
            self._owner = asyncio.current_task()
            try:
                value = await self._operation
            except BaseException as exc:
                if not self._future.done():
                    self._future.set_exception(exc)
                    self._future.exception()
                raise
            else:
                self._future.set_result(value)
            finally:
                self._owner = None
        return await asyncio.shield(self._future)

    async def cancel(self) -> bool:
        if self._future is not None and self._future.done():
            return False
        if self._owner is not None:
            self._owner.cancel()
        else:
            close = getattr(self._operation, "close", None)
            if callable(close):
                close()
        return True


class ToolExecutionError(RuntimeError):
    """A classified executor failure safe to project into ToolResult."""

    def __init__(
        self,
        message: str,
        *,
        kind: str = "executor_error",
        code: str | None = None,
        retryable: bool = False,
        side_effect_committed: bool | None = None,
        missing_capabilities: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.code = code
        self.retryable = retryable
        self.side_effect_committed = side_effect_committed
        self.missing_capabilities = tuple(missing_capabilities)


def validate_executor_sandbox(
    spec: ToolSpec,
    executor: ToolExecutor,
    *,
    durable: bool = False,
    required_capabilities: tuple[str, ...] = (),
) -> SandboxExecutorSupport | None:
    """Validate the exact executor selected for a sandboxed managed call."""

    profile = spec.sandbox_profile
    if profile is None:
        return None
    capability = f"tool.sandbox.{profile}"
    support = getattr(executor, "sandbox_support", None)
    if not isinstance(support, SandboxExecutorSupport) or profile not in support.profiles:
        raise ToolExecutionError(
            f"{spec.tool_id}@{spec.version} requires {capability}",
            kind="capability_error",
            code="missing_sandbox_capability",
            retryable=False,
            side_effect_committed=False,
            missing_capabilities=(capability,),
        )
    if durable and not support.durable_reconnect:
        raise ToolExecutionError(
            f"{spec.tool_id}@{spec.version} sandbox does not support durable reconnect",
            kind="capability_error",
            code="sandbox_reconnect_unavailable",
            retryable=False,
            side_effect_committed=False,
            missing_capabilities=(f"{capability}.durable-reconnect",),
        )
    fingerprint_capability = support.capability_for_fingerprint()
    expected = next(
        (
            item
            for item in required_capabilities
            if item.startswith("tool.sandbox.deployment.")
        ),
        None,
    )
    if durable and expected is not None and fingerprint_capability != expected:
        raise ToolExecutionError(
            f"{spec.tool_id}@{spec.version} sandbox deployment fingerprint changed",
            kind="capability_error",
            code="sandbox_deployment_changed",
            retryable=False,
            side_effect_committed=False,
            missing_capabilities=(expected,),
        )
    return support


@runtime_checkable
class ToolExecutor(Protocol):
    """Trusted deployment adapter; executor instances are not portable values."""

    async def execute(
        self, spec: ToolSpec, call: ToolCall, context: ToolExecutionContext
    ) -> object: ...


class LocalToolExecutor:
    """Adapt a synchronous or asynchronous Python callable."""

    def __init__(self, handler: ToolHandler) -> None:
        if not callable(handler):
            raise TypeError("handler must be callable")
        self._handler = handler

    async def execute(
        self, spec: ToolSpec, call: ToolCall, context: ToolExecutionContext
    ) -> object:
        value = self._handler(cast(FrozenJsonObject, call.arguments))
        if inspect.isawaitable(value):
            value = await value
        return value


class HttpToolExecutor:
    """POST a portable tool invocation to a trusted HTTP endpoint."""

    def __init__(
        self,
        url: str,
        *,
        client: httpx.AsyncClient | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        if not url:
            raise ValueError("url must be non-empty")
        self._url = url
        self._client = client
        self._headers = dict(headers or {})

    async def execute(
        self, spec: ToolSpec, call: ToolCall, context: ToolExecutionContext
    ) -> object:
        client = self._client or httpx.AsyncClient()
        owns_client = self._client is None
        try:
            response = await client.post(
                self._url,
                headers=self._headers,
                json={
                    "tool_id": spec.tool_id,
                    "version": spec.version,
                    "call_id": call.call_id,
                    "name": call.name,
                    "arguments": thaw_json(cast(FrozenJsonObject, call.arguments)),
                    "idempotency_key": call.idempotency_key,
                },
                timeout=None,
            )
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as exc:
            raise ToolExecutionError(
                "remote tool timed out",
                kind="timeout",
                code="http_timeout",
                retryable=True,
                side_effect_committed=None,
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ToolExecutionError(
                f"remote tool returned HTTP {exc.response.status_code}",
                kind="remote_error",
                code=f"http_{exc.response.status_code}",
                retryable=exc.response.status_code in {408, 429}
                or exc.response.status_code >= 500,
                side_effect_committed=None,
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ToolExecutionError(
                "remote tool transport failed",
                kind="transport_error",
                code="http_transport",
                retryable=True,
                side_effect_committed=None,
            ) from exc
        finally:
            if owns_client:
                await client.aclose()


class AgentToolExecutor:
    """Adapt a Module/Agent through explicit request and response conversion."""

    def __init__(
        self,
        invoke: Callable[[ToolSpec, ToolCall], object | Awaitable[object]] | None = None,
        *,
        agent: ModuleDependency[Message, Message] | None = None,
        request_builder: AgentToolRequestBuilder | None = None,
        result_builder: AgentToolResultBuilder | None = None,
    ) -> None:
        if (invoke is None) == (agent is None):
            raise ValueError("configure exactly one of invoke or agent")
        if invoke is not None and not callable(invoke):
            raise TypeError("invoke must be callable")
        if agent is not None and not callable(agent):
            raise TypeError("agent must be a Module dependency")
        if agent is not None and (
            not callable(request_builder) or not callable(result_builder)
        ):
            raise TypeError(
                "agent execution requires request_builder and result_builder"
            )
        self._invoke = invoke
        self._agent = agent
        self._request_builder = request_builder
        self._result_builder = result_builder

    async def execute(
        self, spec: ToolSpec, call: ToolCall, context: ToolExecutionContext
    ) -> object:
        if self._agent is None:
            invoke = self._invoke
            if invoke is None:  # pragma: no cover - constructor invariant
                raise RuntimeError("AgentToolExecutor has no invocation adapter")
            value = invoke(spec, call)
            if inspect.isawaitable(value):
                value = await value
            return value

        request_builder = self._request_builder
        result_builder = self._result_builder
        if request_builder is None or result_builder is None:  # pragma: no cover
            raise RuntimeError("AgentToolExecutor has no conversion adapter")
        request = request_builder(spec, call)
        if (
            not isinstance(request, tuple)
            or len(request) != 2
            or not isinstance(request[0], Message)
            or not isinstance(request[1], Context)
        ):
            raise TypeError("request_builder must return (Message, Context)")

        message, agent_context = request
        if active_infrastructure() is None:
            # Detached ToolTasks are independent executions. A raw Module uses
            # direct Root execution; a pre-bound dependency uses its own
            # Runtime/Binding admission rather than rejoining the former Parent.
            root_invoke = getattr(self._agent, "invoke", None)
            if not callable(root_invoke):
                raise ToolExecutionError(
                    "detached Agent-backed tool requires an invokable Module binding",
                    kind="capability_error",
                    code="agent_root_unavailable",
                    retryable=False,
                    side_effect_committed=False,
                )
            output, next_context = await root_invoke(message, agent_context)
        else:
            # Within an active execution this direct dependency call is the
            # ordinary structured Child path, including lineage and cancellation.
            output, next_context = await self._agent(message, agent_context)
        value = result_builder(output, next_context)
        if inspect.isawaitable(value):
            value = await value
        return value


class ExecutorRegistry:
    """Resolve deployment adapters by stable ``(tool_id, version)`` identity."""

    def __init__(self) -> None:
        self._executors: dict[tuple[str, str], ToolExecutor] = {}

    def register(
        self,
        tool_id: str,
        version: str,
        executor: ToolExecutor,
        *,
        replace_existing: bool = False,
    ) -> None:
        if not tool_id or not version:
            raise ValueError("tool_id and version must be non-empty")
        if not isinstance(executor, ToolExecutor):
            raise TypeError("executor must implement ToolExecutor")
        key = (tool_id, version)
        if key in self._executors and not replace_existing:
            raise ValueError(f"executor already registered for {tool_id}@{version}")
        self._executors[key] = executor

    def unregister(self, tool_id: str, version: str) -> None:
        self._executors.pop((tool_id, version), None)

    def contains(self, tool_id: str, version: str) -> bool:
        """Return whether an exact Tool executor identity is registered."""

        return (tool_id, version) in self._executors

    def resolve(self, tool_id: str, version: str) -> ToolExecutor:
        try:
            return self._executors[(tool_id, version)]
        except KeyError as exc:
            raise LookupError(
                f"no executor registered for {tool_id}@{version}"
            ) from exc

    @property
    def sandbox_capabilities(self) -> frozenset[str]:
        capabilities: set[str] = set()
        for executor in self._executors.values():
            support = getattr(executor, "sandbox_support", None)
            if not isinstance(support, SandboxExecutorSupport):
                continue
            capabilities.update(f"tool.sandbox.{item}" for item in support.profiles)
            fingerprint = support.capability_for_fingerprint()
            if fingerprint is not None:
                capabilities.add(fingerprint)
        return frozenset(capabilities)

    async def execute(
        self,
        spec: ToolSpec,
        call: ToolCall,
        context: ToolExecutionContext,
    ) -> object:
        executor = self.resolve(spec.tool_id, spec.version)
        if context.execution_id is not None:
            validate_executor_sandbox(spec, executor)
        return await executor.execute(spec, call, context)


class ToolRunner:
    """Unique owner of Tool timeout, cancellation, error and result semantics."""

    def execute(
        self,
        spec: ToolSpec,
        call: ToolCall,
        registry: ExecutorRegistry,
        *,
        context: ToolExecutionContext | None = None,
        task: ToolTask | None = None,
        operation: ToolTaskExecution | None = None,
    ) -> ToolExecution:
        effective_context = context or ToolExecutionContext()
        return ToolExecution(
            self._run(
                spec,
                call,
                registry,
                effective_context,
                task=task,
                operation=operation,
            )
        )

    async def _run(
        self,
        spec: ToolSpec,
        call: ToolCall,
        registry: ExecutorRegistry,
        context: ToolExecutionContext,
        *,
        task: ToolTask | None,
        operation: ToolTaskExecution | None,
    ) -> ToolResult:
        async def emit(kind: str, data: Mapping[str, JsonValue]) -> None:
            if context.emit is not None:
                await context.emit(kind, data)

        await emit("tool.started", {"call_id": call.call_id, "tool_id": spec.tool_id})
        try:
            async def invoke() -> object:
                if operation is not None:
                    return await operation(spec, call, context)
                return await registry.execute(spec, call, context)

            deadlines = [value for value in (context.deadline,) if value is not None]
            if spec.timeout is not None:
                deadlines.append(time.monotonic() + spec.timeout)
            if deadlines:
                async with asyncio.timeout(max(0.0, min(deadlines) - time.monotonic())):
                    value = await invoke()
            else:
                value = await invoke()
        except asyncio.CancelledError:
            result = ToolResult(
                call_id=call.call_id,
                name=call.name,
                status="cancelled",
                task=task,
                error_kind="cancelled",
                retryable=False,
                side_effect_committed=None,
                tool_id=spec.tool_id,
                tool_version=spec.version,
            )
            await emit("tool.cancelled", {"call_id": call.call_id})
            raise
        except TimeoutError:
            result = result_from_exception(
                spec,
                call,
                ToolExecutionError(
                    "tool execution timed out",
                    kind="timeout",
                    code="tool_timeout",
                    retryable=True,
                    side_effect_committed=None,
                ),
                task=task,
            )
            await emit("tool.failed", {"call_id": call.call_id, "error_kind": "timeout"})
            return result
        except Exception as exc:  # noqa: BLE001 - executor boundary
            result = result_from_exception(spec, call, exc, task=task)
            await emit(
                "tool.failed",
                {"call_id": call.call_id, "error_kind": result.error_kind or "executor_error"},
            )
            return result
        result = ToolResult(
            call_id=call.call_id,
            name=call.name,
            status="succeeded",
            task=task,
            output=freeze_json(value),
            side_effect_committed=True,
            tool_id=spec.tool_id,
            tool_version=spec.version,
        )
        await emit("tool.completed", {"call_id": call.call_id})
        return result


class ToolTaskManager(Protocol):
    async def prepare(
        self,
        spec: ToolSpec,
        call: ToolCall,
        *,
        execution: ToolTaskExecution | None = None,
    ) -> ToolTask: ...
    async def start(self, task_id: str) -> None: ...
    async def submit(
        self,
        spec: ToolSpec,
        call: ToolCall,
        *,
        execution: ToolTaskExecution | None = None,
    ) -> ToolTask: ...
    async def get_task(self, task_id: str) -> ToolTask | None: ...
    async def cancel(self, task_id: str) -> bool: ...
    async def get_result(
        self, task_id: str, *, wait: bool = False
    ) -> ToolResult | None: ...
    async def close(self, *, cancel: bool = False) -> None: ...


class InMemoryToolTaskManager:
    """Explicit process-local detached-task facility for applications and tests.

    It is intentionally separate from ``ToolCallLayer`` and never claims durable
    recovery. Managed runtimes can provide their own implementation instead.
    """

    def __init__(self, registry: ExecutorRegistry) -> None:
        self._registry = registry
        self._snapshots: dict[str, ToolTask] = {}
        self._results: dict[str, ToolResult] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._invocations: dict[
            str, tuple[ToolSpec, ToolCall, ToolTaskExecution | None]
        ] = {}
        self._lock = asyncio.Lock()

    async def submit(
        self,
        spec: ToolSpec,
        call: ToolCall,
        *,
        execution: ToolTaskExecution | None = None,
    ) -> ToolTask:
        snapshot = await self.prepare(spec, call, execution=execution)
        await self.start(snapshot.task_id)
        return snapshot

    async def prepare(
        self,
        spec: ToolSpec,
        call: ToolCall,
        *,
        execution: ToolTaskExecution | None = None,
    ) -> ToolTask:
        snapshot = ToolTask(
            task_id=f"tool-{uuid4()}",
            call_id=call.call_id,
            tool_id=spec.tool_id,
            version=spec.version,
            state=ToolTaskState.PENDING,
        )
        async with self._lock:
            self._snapshots[snapshot.task_id] = snapshot
            self._invocations[snapshot.task_id] = (spec, call, execution)
        return snapshot

    async def start(self, task_id: str) -> None:
        async with self._lock:
            existing = self._tasks.get(task_id)
            if existing is not None and not existing.done():
                return
            try:
                spec, call, execution = self._invocations[task_id]
                snapshot = self._snapshots[task_id]
            except KeyError as exc:
                raise KeyError(f"unknown prepared ToolTask {task_id!r}") from exc
            self._tasks[snapshot.task_id] = asyncio.create_task(
                self._run(snapshot, spec, call, execution),
                name=f"pygent-{snapshot.task_id}",
            )

    async def _run(
        self,
        snapshot: ToolTask,
        spec: ToolSpec,
        call: ToolCall,
        execution: ToolTaskExecution | None,
    ) -> None:
        running = replace(snapshot, state=ToolTaskState.RUNNING)
        async with self._lock:
            self._snapshots[snapshot.task_id] = running
        try:
            value = await _execute_with_timeout(
                self._registry,
                spec,
                call,
                execution=execution,
                context=ToolExecutionContext(task_id=snapshot.task_id),
            )
            result = ToolResult(
                call_id=call.call_id,
                name=call.name,
                status="succeeded",
                task=replace(snapshot, state=ToolTaskState.SUCCEEDED),
                output=freeze_json(value),
                side_effect_committed=True,
                tool_id=spec.tool_id,
                tool_version=spec.version,
            )
            state = ToolTaskState.SUCCEEDED
        except asyncio.CancelledError:
            result = ToolResult(
                call_id=call.call_id,
                name=call.name,
                status="cancelled",
                task=replace(snapshot, state=ToolTaskState.CANCELLED),
                error_kind="cancelled",
                retryable=False,
                side_effect_committed=None,
                tool_id=spec.tool_id,
                tool_version=spec.version,
            )
            state = ToolTaskState.CANCELLED
        except Exception as exc:  # noqa: BLE001 - converted to a public result
            result = result_from_exception(spec, call, exc, task=snapshot)
            state = (
                ToolTaskState.UNKNOWN
                if result.status == "unknown"
                else ToolTaskState.FAILED
            )
            result = replace(result, task=replace(snapshot, state=state))
        async with self._lock:
            self._snapshots[snapshot.task_id] = replace(snapshot, state=state)
            self._results[snapshot.task_id] = result

    async def get_task(self, task_id: str) -> ToolTask | None:
        async with self._lock:
            return self._snapshots.get(task_id)

    async def cancel(self, task_id: str) -> bool:
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.done():
                return False
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        async with self._lock:
            if task_id not in self._results:
                snapshot = self._snapshots[task_id]
                spec, call, _ = self._invocations[task_id]
                cancelled = replace(snapshot, state=ToolTaskState.CANCELLED)
                self._snapshots[task_id] = cancelled
                self._results[task_id] = ToolResult(
                    call_id=call.call_id,
                    name=call.name,
                    status="cancelled",
                    task=cancelled,
                    error_kind="cancelled",
                    retryable=False,
                    side_effect_committed=None,
                    tool_id=spec.tool_id,
                    tool_version=spec.version,
                )
        return True

    async def get_result(
        self, task_id: str, *, wait: bool = False
    ) -> ToolResult | None:
        async with self._lock:
            task = self._tasks.get(task_id)
            result = self._results.get(task_id)
        if result is not None or not wait or task is None:
            return result
        await asyncio.gather(task, return_exceptions=True)
        async with self._lock:
            return self._results.get(task_id)

    async def close(self, *, cancel: bool = False) -> None:
        async with self._lock:
            tasks = tuple(self._tasks.values())
        if cancel:
            for task in tasks:
                if not task.done():
                    task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _execute_with_timeout(
    registry: ExecutorRegistry,
    spec: ToolSpec,
    call: ToolCall,
    *,
    execution: ToolTaskExecution | None = None,
    context: ToolExecutionContext | None = None,
) -> object:
    result = await ToolRunner().execute(
        spec,
        call,
        registry,
        context=context,
        operation=execution,
    ).result()
    if result.status == "succeeded":
        return thaw_json(result.output)
    raise ToolExecutionError(
        result.error or "tool executor failed",
        kind=result.error_kind or "executor_error",
        code=result.error_code,
        retryable=result.retryable,
        side_effect_committed=result.side_effect_committed,
        missing_capabilities=result.missing_capabilities,
    )


def result_from_exception(
    spec: ToolSpec,
    call: ToolCall,
    exc: BaseException,
    *,
    task: ToolTask | None = None,
) -> ToolResult:
    if isinstance(exc, ToolExecutionError):
        committed = exc.side_effect_committed
        status: ToolResultStatus = (
            "unknown"
            if committed is None and spec.side_effect.value in {"write", "external"}
            else "failed"
        )
        return ToolResult(
            call_id=call.call_id,
            name=call.name,
            status=status,
            task=task,
            error=str(exc),
            error_kind=exc.kind,
            error_code=exc.code,
            retryable=exc.retryable,
            side_effect_committed=committed,
            missing_capabilities=exc.missing_capabilities,
            tool_id=spec.tool_id,
            tool_version=spec.version,
        )
    unknown_committed: bool | None = None
    unknown_status: ToolResultStatus = (
        "unknown"
        if spec.side_effect in (ToolSideEffect.WRITE, ToolSideEffect.EXTERNAL)
        else "failed"
    )
    return ToolResult(
        call_id=call.call_id,
        name=call.name,
        status=unknown_status,
        task=task,
        error="tool executor failed",
        error_kind="executor_error",
        error_code=type(exc).__name__,
        retryable=False,
        side_effect_committed=unknown_committed,
        tool_id=spec.tool_id,
        tool_version=spec.version,
    )


__all__ = [
    "AgentToolExecutor",
    "AgentToolRequestBuilder",
    "AgentToolResultBuilder",
    "ExecutorRegistry",
    "HttpToolExecutor",
    "InMemoryToolTaskManager",
    "LocalToolExecutor",
    "SandboxExecutorSupport",
    "ToolExecution",
    "ToolExecutionContext",
    "ToolExecutionError",
    "ToolExecutor",
    "ToolHandler",
    "ToolRunner",
    "ToolTaskAdmission",
    "ToolTaskExecution",
    "ToolTaskManager",
    "result_from_exception",
    "validate_executor_sandbox",
]
