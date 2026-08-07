"""HTTP/SSE worker protocol and stable logical worker registry."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from contextvars import Context as ContextVarsContext
from dataclasses import dataclass, field
from time import monotonic, time
from typing import Any, Self, cast
from uuid import uuid4

import httpx
from httpx_sse import aconnect_sse
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from pygent.core import (
    Context,
    ExecutionEvent,
    ExecutionFailure,
    ExecutionFailureError,
    ExecutionStatus,
    FrozenJsonObject,
    JsonValue,
    Message,
    PlacementMode,
    PlacementPolicy,
    freeze_json,
    freeze_json_object,
    thaw_json,
)
from pygent.llm import ModelCallOptions

from .api import BoundModule, ExecutionOptions
from .codec import invocation_from_dict, invocation_to_dict
from .history import SQLiteHistoryStore, StoredTask
from .plan import CodeArtifactSpec


class WorkerProtocolError(RuntimeError):
    """Base class for remote Worker protocol failures."""


class WorkerUnavailableError(WorkerProtocolError):
    """Raised after every declared target is unavailable."""


class WorkerRemoteError(WorkerProtocolError):
    """Raised when a Worker reports a terminal execution failure."""

    def __init__(self, failure: ExecutionFailure) -> None:
        if not isinstance(failure, ExecutionFailure):
            raise TypeError("failure must be an ExecutionFailure")
        super().__init__(failure.message)
        self.failure = failure

    @property
    def kind(self) -> str:
        return self.failure.kind


class WorkerOutcomeUnknownError(WorkerRemoteError):
    """Remote owner may still commit and replay is not proven safe."""

    def __init__(self, message: str) -> None:
        super().__init__(
            ExecutionFailure(
                domain="worker",
                kind="outcome_unknown",
                message=message,
                retryable=False,
                outcome_unknown=True,
            )
        )


def _worker_failure(
    kind: str,
    message: str,
    *,
    retryable: bool = False,
    outcome_unknown: bool = False,
) -> ExecutionFailure:
    return ExecutionFailure(
        domain="worker",
        kind=kind,
        message=message,
        retryable=retryable,
        outcome_unknown=outcome_unknown,
    )


def _failure_from_exception(
    error: BaseException, *, persistence: bool = False
) -> ExecutionFailure:
    if isinstance(error, WorkerRemoteError):
        return error.failure
    if isinstance(error, ExecutionFailureError):
        return error.failure
    if isinstance(error, asyncio.TimeoutError):
        return _worker_failure("timeout", "Worker execution timed out", retryable=True)
    if persistence:
        return _worker_failure(
            "persistence_error", "Worker persistence operation failed"
        )
    return _worker_failure("worker_internal", "Worker execution failed")


def _validate_plan_identity(plan_id: str, graph_hash: str) -> None:
    if (
        not isinstance(graph_hash, str)
        or len(graph_hash) != 64
        or any(character not in "0123456789abcdef" for character in graph_hash)
    ):
        raise WorkerProtocolError("graph_hash must be a lowercase SHA-256 digest")
    if plan_id != f"sha256:{graph_hash}":
        raise WorkerProtocolError("plan_id does not identify graph_hash")


@dataclass(frozen=True, slots=True)
class WorkerTarget:
    target_id: str
    endpoint: str
    capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.target_id or not self.endpoint:
            raise ValueError("worker target_id and endpoint must be non-empty")
        capabilities = tuple(self.capabilities)
        if any(not isinstance(item, str) or not item for item in capabilities):
            raise ValueError("worker capabilities must be non-empty strings")
        if len(capabilities) != len(set(capabilities)):
            raise ValueError("worker capabilities must be unique")
        object.__setattr__(self, "capabilities", capabilities)


class WorkerRegistry:
    """Registry for already-declared logical Binding targets."""

    def __init__(self) -> None:
        self._targets: dict[str, tuple[WorkerTarget, ...]] = {}
        self._generation: dict[str, int] = {}

    def publish(self, binding_ref: str, targets: tuple[WorkerTarget, ...]) -> int:
        if not binding_ref or not targets:
            raise ValueError("binding_ref and at least one target are required")
        if len({target.target_id for target in targets}) != len(targets):
            raise ValueError("worker target ids must be unique")
        self._targets[binding_ref] = tuple(targets)
        generation = self._generation.get(binding_ref, 0) + 1
        self._generation[binding_ref] = generation
        return generation

    def resolve(
        self,
        binding_ref: str,
        *,
        required_capabilities: tuple[str, ...] = (),
    ) -> tuple[WorkerTarget, ...]:
        try:
            targets = self._targets[binding_ref]
        except KeyError as exc:
            raise KeyError(f"undeclared binding_ref {binding_ref!r}") from exc
        required = set(required_capabilities)
        eligible = tuple(
            target for target in targets if required.issubset(target.capabilities)
        )
        if not eligible:
            raise WorkerUnavailableError(
                "no Worker target satisfies the required capabilities"
            )
        return eligible

    def generation(self, binding_ref: str) -> int:
        return self._generation[binding_ref]


@dataclass(frozen=True, slots=True)
class WorkerInvocation:
    binding_ref: str
    request_id: str
    input: FrozenJsonObject
    plan_id: str
    graph_hash: str
    deadline: float | None
    required_capabilities: tuple[str, ...] = ()
    idempotency_key: str | None = None
    trace_id: str | None = None
    parent_execution_id: str | None = None
    parent_span_id: str | None = None
    attempt: int = 1
    expires_at: float | None = None
    model_calls: FrozenJsonObject = field(default_factory=FrozenJsonObject)
    model_admission_ref: str | None = None
    model_store_namespace: str | None = None

    def __post_init__(self) -> None:
        capabilities = tuple(self.required_capabilities)
        if any(not isinstance(item, str) or not item for item in capabilities):
            raise ValueError("required capabilities must be non-empty strings")
        if len(capabilities) != len(set(capabilities)):
            raise ValueError("required capabilities must be unique")
        object.__setattr__(self, "required_capabilities", capabilities)
        if not isinstance(self.model_calls, FrozenJsonObject):
            raise TypeError("model_calls must be a frozen JSON object")
        if self.model_admission_ref is not None and not self.model_admission_ref:
            raise ValueError("model_admission_ref must be non-empty")
        if self.model_store_namespace is not None and not self.model_store_namespace:
            raise ValueError("model_store_namespace must be non-empty")


WorkerEventSink = Callable[[ExecutionEvent], Awaitable[None]]
WorkerHandler = Callable[[WorkerInvocation, WorkerEventSink], Awaitable[JsonValue]]


@dataclass(frozen=True, slots=True)
class WorkerDeploymentManifest:
    """Resolver result binding loaded code and wire contracts to an artifact."""

    artifact: CodeArtifactSpec
    verified_digest: str
    entrypoint: Callable[[], object]
    input_schema: str
    output_schema: str
    serializer: str


WorkerArtifactResolver = Callable[[CodeArtifactSpec], WorkerDeploymentManifest]


@dataclass(slots=True)
class _ServerExecution:
    execution_id: str
    binding_ref: str
    request_id: str
    input: FrozenJsonObject
    task: asyncio.Task[JsonValue] | None
    events: list[dict[str, Any]]
    condition: asyncio.Condition
    invocation: WorkerInvocation
    root_span_id: str = field(default_factory=lambda: str(uuid4()))
    status: str = "pending"
    result: JsonValue | None = None
    error: JsonValue | None = None
    terminal: bool = False
    next_event_index: int = 0


@dataclass(frozen=True, slots=True)
class RemoteExecutionHandle:
    """Client-side control plane for one remotely owned execution."""

    client: HTTPWorkerClient = field(repr=False, compare=False)
    execution_id: str
    target: WorkerTarget
    binding_ref: str = ""
    input: FrozenJsonObject | None = None
    request_id: str = ""
    required_capabilities: tuple[str, ...] = ()
    idempotency_key: str | None = None
    trace_id: str | None = None
    parent_execution_id: str | None = None
    parent_span_id: str | None = None
    attempt: int = 1
    attempted_target_ids: tuple[str, ...] = ()
    plan_id: str = ""
    graph_hash: str = ""
    placement_mode: PlacementMode = PlacementMode.ADAPTIVE
    pinned_target_id: str | None = None
    model_calls: FrozenJsonObject = field(default_factory=FrozenJsonObject)
    model_admission_ref: str | None = None
    model_store_namespace: str | None = None

    @property
    def status(self) -> ExecutionStatus:
        return ExecutionStatus.RUNNING

    async def result(self, *, deadline: float | None = None) -> JsonValue:
        return await self.client.result(self, deadline=deadline)

    async def cancel(self) -> bool:
        return await self.client.cancel(self)

    def subscribe(self, *, after: int | None = None) -> _RemoteExecutionSubscription:
        return _RemoteExecutionSubscription(self, -1 if after is None else after)


class _RemoteExecutionSubscription:
    def __init__(self, handle: RemoteExecutionHandle, after: int) -> None:
        self._handle = handle
        self._after = after

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def __aiter__(self) -> AsyncIterator[ExecutionEvent]:
        return self._handle.client.events(
            self._handle.target,
            self._handle.execution_id,
            after=self._after,
        )


class HTTPWorkerApp:
    """Small ASGI Worker exposing invoke, cancel, health and cursor SSE."""

    def __init__(
        self,
        handler: WorkerHandler,
        *,
        capabilities: tuple[str, ...] = ("local", "sse"),
        max_retained_events: int = 1024,
        max_retained_executions: int = 1024,
        history: SQLiteHistoryStore | None = None,
        model_store_namespace: str | None = None,
    ) -> None:
        if max_retained_events <= 0:
            raise ValueError("max_retained_events must be positive")
        if max_retained_executions <= 0:
            raise ValueError("max_retained_executions must be positive")
        if any(not isinstance(item, str) or not item for item in capabilities):
            raise ValueError("Worker capabilities must be non-empty strings")
        if len(capabilities) != len(set(capabilities)):
            raise ValueError("Worker capabilities must be unique")
        self.handler = handler
        self.model_store_namespace = model_store_namespace or cast(
            str | None, getattr(handler, "_pygent_model_store_namespace", None)
        )
        if self.model_store_namespace is not None and not self.model_store_namespace:
            raise ValueError("model_store_namespace must be non-empty")
        self.capabilities = tuple(capabilities)
        self.max_retained_events = max_retained_events
        self.max_retained_executions = max_retained_executions
        self.history = history
        self.executions: dict[str, _ServerExecution] = {}
        self._requests: dict[tuple[str, str], str] = {}

        @asynccontextmanager
        async def lifespan(_app: Starlette) -> AsyncIterator[None]:
            yield
            await self.close()

        self.app = Starlette(
            routes=[
                Route("/health", self.health, methods=["GET"]),
                Route(
                    "/v1/bindings/{binding_ref:str}/invoke",
                    self.start,
                    methods=["POST"],
                ),
                Route(
                    "/v1/executions/{execution_id:str}", self.result, methods=["GET"]
                ),
                Route(
                    "/v1/executions/{execution_id:str}/cancel",
                    self.cancel,
                    methods=["POST"],
                ),
                Route(
                    "/v1/executions/{execution_id:str}/events",
                    self.events,
                    methods=["GET"],
                ),
            ],
            lifespan=lifespan,
        )

    async def close(self) -> None:
        """Cancel and join every active Worker-owned run."""

        tasks = tuple(
            run.task
            for run in self.executions.values()
            if run.task is not None and not run.task.done()
        )
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    def _job_request(invocation: WorkerInvocation) -> dict[str, object]:
        request: dict[str, object] = {
            "binding_ref": invocation.binding_ref,
            "request_id": invocation.request_id,
            "input": thaw_json(invocation.input),
            "plan_id": invocation.plan_id,
            "graph_hash": invocation.graph_hash,
            "idempotency_key": invocation.idempotency_key,
            "trace_id": invocation.trace_id,
            "parent_execution_id": invocation.parent_execution_id,
            "parent_span_id": invocation.parent_span_id,
            "attempt": invocation.attempt,
            "expires_at": invocation.expires_at,
        }
        if invocation.required_capabilities:
            request["required_capabilities"] = list(invocation.required_capabilities)
        if invocation.model_calls:
            request["model_calls"] = thaw_json(invocation.model_calls)
        if invocation.model_admission_ref is not None:
            request["model_admission_ref"] = invocation.model_admission_ref
        if invocation.model_store_namespace is not None:
            request["model_store_namespace"] = invocation.model_store_namespace
        return request

    @staticmethod
    def _logical_request_key(invocation: WorkerInvocation) -> tuple[str, str]:
        identity = (
            f"idempotency:{invocation.idempotency_key}"
            if invocation.idempotency_key is not None
            else f"request:{invocation.request_id}"
        )
        return invocation.binding_ref, identity

    @staticmethod
    def _job_invocation(request: JsonValue) -> WorkerInvocation:
        if not isinstance(request, FrozenJsonObject):
            raise WorkerProtocolError("durable Worker job request must be an object")
        payload = request.get("input")
        if not isinstance(payload, FrozenJsonObject):
            raise WorkerProtocolError("durable Worker job input must be an object")
        expires_at = cast(float | None, request.get("expires_at"))
        deadline = (
            None if expires_at is None else monotonic() + max(0.0, expires_at - time())
        )
        plan_id = cast(str, request["plan_id"])
        graph_hash = cast(str, request["graph_hash"])
        _validate_plan_identity(plan_id, graph_hash)
        return WorkerInvocation(
            binding_ref=cast(str, request["binding_ref"]),
            request_id=cast(str, request["request_id"]),
            input=payload,
            plan_id=plan_id,
            graph_hash=graph_hash,
            deadline=deadline,
            required_capabilities=tuple(
                cast(tuple[str, ...], request.get("required_capabilities", ()))
            ),
            idempotency_key=cast(str | None, request.get("idempotency_key")),
            trace_id=cast(str | None, request.get("trace_id")),
            parent_execution_id=cast(str | None, request.get("parent_execution_id")),
            parent_span_id=cast(str | None, request.get("parent_span_id")),
            attempt=cast(int, request.get("attempt", 1)),
            expires_at=expires_at,
            model_calls=freeze_json_object(
                cast(Mapping[str, object], thaw_json(request.get("model_calls", FrozenJsonObject())))
            ),
            model_admission_ref=cast(str | None, request.get("model_admission_ref")),
            model_store_namespace=cast(
                str | None, request.get("model_store_namespace")
            ),
        )

    async def recover(self) -> None:
        """Rebuild durable Worker jobs and restart unfinished ownership attempts."""

        if self.history is None:
            raise WorkerProtocolError("this HTTPWorkerApp has no SQLiteHistoryStore")
        for item in await self.history.list_tasks(kind="job"):
            if item.task_id in self.executions:
                continue
            run = await self._restore_job(item)
            if item.status in {"pending", "running"}:
                self._launch(run)

    async def _restore_job(self, item: StoredTask) -> _ServerExecution:
        invocation = self._job_invocation(item.request)
        durable_events: list[dict[str, Any]] = []
        if self.history is not None:
            for value in await self.history.events_tail(
                execution_id=item.task_id, limit=min(self.max_retained_events, 4096)
            ):
                thawed = thaw_json(value)
                if isinstance(thawed, dict):
                    durable_events.append(thawed)
        run = _ServerExecution(
            execution_id=item.task_id,
            binding_ref=invocation.binding_ref,
            request_id=invocation.request_id,
            input=invocation.input,
            task=None,
            events=durable_events,
            condition=asyncio.Condition(),
            invocation=invocation,
            status=item.status,
            result=item.result,
            error=item.error,
            terminal=item.status not in {"pending", "running"},
            next_event_index=(
                int(durable_events[-1]["sequence"]) + 1 if durable_events else 0
            ),
        )
        self.executions[run.execution_id] = run
        self._requests[self._logical_request_key(invocation)] = run.execution_id
        return run

    async def _persist_job(
        self, run: _ServerExecution, *, status: str | None = None
    ) -> None:
        if self.history is None:
            return
        await self.history.put_task(
            task_id=run.execution_id,
            kind="job",
            status=status or run.status,
            request=self._job_request(run.invocation),
            result=run.result,
            error=run.error,
        )

    async def health(self, request: Request) -> Response:
        payload: dict[str, object] = {
            "status": "ok",
            "capabilities": list(self.capabilities),
        }
        if self.model_store_namespace is not None:
            payload["model_store_namespace"] = self.model_store_namespace
        return JSONResponse(payload)

    async def start(self, request: Request) -> Response:
        try:
            body = await request.json()
            request_id = body["request_id"]
            payload = freeze_json(body["input"])
            plan_id = body["plan_id"]
            graph_hash = body["graph_hash"]
            required_capabilities = body.get("required_capabilities", [])
            deadline_seconds = body.get("deadline_seconds")
            idempotency_key = body.get("idempotency_key")
            trace_id = body.get("trace_id")
            parent_execution_id = body.get("parent_execution_id")
            parent_span_id = body.get("parent_span_id")
            attempt = body.get("attempt", 1)
            model_calls = freeze_json(body.get("model_calls", {}))
            model_admission_ref = body.get("model_admission_ref")
            model_store_namespace = body.get("model_store_namespace")
            if not isinstance(request_id, str) or not request_id:
                raise TypeError
            if not isinstance(plan_id, str) or not isinstance(graph_hash, str):
                raise TypeError
            _validate_plan_identity(plan_id, graph_hash)
            if not isinstance(payload, FrozenJsonObject):
                raise TypeError
            if (
                not isinstance(required_capabilities, list)
                or any(
                    not isinstance(value, str) or not value
                    for value in required_capabilities
                )
                or len(required_capabilities) != len(set(required_capabilities))
            ):
                raise TypeError
            if deadline_seconds is not None and (
                isinstance(deadline_seconds, bool)
                or not isinstance(deadline_seconds, (int, float))
                or not math.isfinite(deadline_seconds)
                or deadline_seconds <= 0
            ):
                raise ValueError
            if any(
                value is not None and (not isinstance(value, str) or not value)
                for value in (
                    idempotency_key,
                    trace_id,
                    parent_execution_id,
                    parent_span_id,
                )
            ):
                raise TypeError
            if type(attempt) is not int or attempt <= 0:
                raise ValueError
            if not isinstance(model_calls, FrozenJsonObject):
                raise TypeError
            if model_admission_ref is not None and (
                not isinstance(model_admission_ref, str) or not model_admission_ref
            ):
                raise TypeError
            if model_store_namespace is not None and (
                not isinstance(model_store_namespace, str)
                or not model_store_namespace
            ):
                raise TypeError
        except (
            KeyError,
            TypeError,
            ValueError,
            WorkerProtocolError,
            json.JSONDecodeError,
        ):
            return JSONResponse({"error": "invalid_request"}, status_code=400)
        binding_ref = request.path_params["binding_ref"]
        missing_capabilities = set(required_capabilities) - set(self.capabilities)
        if missing_capabilities:
            return JSONResponse(
                {
                    "error": "capability_mismatch",
                    "missing_capabilities": sorted(missing_capabilities),
                },
                status_code=412,
            )
        expires_at = (
            None if deadline_seconds is None else time() + float(deadline_seconds)
        )
        invocation = WorkerInvocation(
            binding_ref=binding_ref,
            request_id=request_id,
            input=payload,
            plan_id=plan_id,
            graph_hash=graph_hash,
            deadline=(
                None
                if deadline_seconds is None
                else monotonic() + float(deadline_seconds)
            ),
            required_capabilities=tuple(required_capabilities),
            idempotency_key=idempotency_key,
            trace_id=trace_id,
            parent_execution_id=parent_execution_id,
            parent_span_id=parent_span_id,
            attempt=attempt,
            expires_at=expires_at,
            model_calls=model_calls,
            model_admission_ref=model_admission_ref,
            model_store_namespace=model_store_namespace,
        )
        request_key = self._logical_request_key(invocation)
        existing_id = self._requests.get(request_key)
        if existing_id is None and self.history is not None:
            for item in await self.history.list_tasks(kind="job"):
                candidate = self._job_invocation(item.request)
                if self._logical_request_key(candidate) == request_key:
                    existing_id = (await self._restore_job(item)).execution_id
                    break
        if existing_id is not None:
            existing = self.executions[existing_id]
            if not self._same_invocation(existing.invocation, invocation):
                return JSONResponse(
                    {"error": "request_conflict", "execution_id": existing_id},
                    status_code=409,
                )
            if existing.status in {"pending", "running"} and existing.task is None:
                self._launch(existing)
            return JSONResponse(
                {"execution_id": existing_id, "status": self._status(existing)},
                status_code=202,
            )
        self._purge_terminal_executions()
        if len(self.executions) >= self.max_retained_executions:
            return JSONResponse({"error": "worker_capacity"}, status_code=429)
        execution_id = str(uuid4())
        condition = asyncio.Condition()
        server_run = _ServerExecution(
            execution_id=execution_id,
            binding_ref=binding_ref,
            request_id=request_id,
            input=payload,
            task=None,
            events=[],
            condition=condition,
            invocation=invocation,
        )
        self.executions[execution_id] = server_run
        self._requests[request_key] = execution_id
        await self._persist_job(server_run)
        self._launch(server_run)
        return JSONResponse(
            {"execution_id": execution_id, "status": "running"}, status_code=202
        )

    @staticmethod
    def _same_invocation(left: WorkerInvocation, right: WorkerInvocation) -> bool:
        return (
            left.binding_ref == right.binding_ref
            and left.input == right.input
            and left.plan_id == right.plan_id
            and left.graph_hash == right.graph_hash
            and left.required_capabilities == right.required_capabilities
            and left.idempotency_key == right.idempotency_key
            and (
                left.idempotency_key is not None or left.request_id == right.request_id
            )
            and left.trace_id == right.trace_id
            and left.parent_execution_id == right.parent_execution_id
            and left.parent_span_id == right.parent_span_id
            and left.attempt == right.attempt
            and left.model_calls == right.model_calls
            and left.model_admission_ref == right.model_admission_ref
            and left.model_store_namespace == right.model_store_namespace
        )

    def _launch(self, run: _ServerExecution) -> None:
        if run.task is not None and not run.task.done():
            return

        phase = "persistence"

        async def run_owned() -> JsonValue:
            nonlocal phase
            recovered = bool(run.events)
            run.status = "running"
            run.terminal = False
            await self._persist_job(run)
            await self._emit(
                run, "execution.recovered" if recovered else "execution.started", {}
            )
            phase = "execution"
            remaining = (
                None
                if run.invocation.deadline is None
                else max(0.0, run.invocation.deadline - monotonic())
            )
            call = self.handler(
                run.invocation,
                lambda event: self._relay_event(run, event),
            )
            result = (
                await call
                if remaining is None
                else await asyncio.wait_for(call, timeout=remaining)
            )
            frozen_result = freeze_json(result)
            phase = "persistence"
            run.result = frozen_result
            await self._emit(run, "execution.completed", {})
            await self._persist_job(run, status="succeeded")
            # Publish terminal success only after its durable record commits.
            run.status = "succeeded"
            return frozen_result

        async def execute() -> JsonValue:
            try:
                history = self.history
                if history is None:
                    return await run_owned()
                claim_id = f"worker-job:{run.execution_id}"
                owner_id = f"worker:{id(self)}"
                fencing_token: int | None = None
                while fencing_token is None:
                    fencing_token = await history.claim_execution(
                        execution_id=claim_id,
                        owner_id=owner_id,
                        lease_ttl=5.0,
                    )
                    if fencing_token is not None:
                        break
                    stored = await history.get_task(run.execution_id)
                    if stored is not None and stored.status not in {
                        "pending",
                        "running",
                    }:
                        run.status = stored.status
                        run.result = stored.result
                        run.error = stored.error
                        if stored.status == "succeeded" and stored.result is not None:
                            return stored.result
                        failure = ExecutionFailure.from_dict(thaw_json(stored.error))
                        raise WorkerRemoteError(failure)
                    await asyncio.sleep(0.05)

                owner_task = asyncio.current_task()

                async def renew() -> None:
                    assert fencing_token is not None
                    while True:
                        await asyncio.sleep(1.5)
                        if not await history.renew_execution_claim(
                            execution_id=claim_id,
                            owner_id=owner_id,
                            fencing_token=fencing_token,
                            lease_ttl=5.0,
                        ):
                            if owner_task is not None:
                                owner_task.cancel()
                            return

                heartbeat = asyncio.create_task(
                    renew(), name=f"pygent-worker-claim-{run.execution_id}"
                )
                try:
                    return await run_owned()
                finally:
                    heartbeat.cancel()
                    await asyncio.gather(heartbeat, return_exceptions=True)
                    assert fencing_token is not None
                    release_task = asyncio.create_task(
                        history.release_execution_claim(
                            execution_id=claim_id,
                            owner_id=owner_id,
                            fencing_token=fencing_token,
                        ),
                        name=f"pygent-worker-claim-release-{run.execution_id}",
                    )
                    try:
                        await asyncio.shield(release_task)
                    except asyncio.CancelledError:
                        # A second cancellation from Worker shutdown must not
                        # orphan release work past History Store closure.
                        await release_task
                        raise
            except asyncio.CancelledError:
                await self._finish_cancelled(run)
                raise
            except BaseException as exc:
                await self._finish_failed(
                    run,
                    _failure_from_exception(exc, persistence=phase == "persistence"),
                )
                raise
            finally:
                async with run.condition:
                    run.terminal = True
                    run.condition.notify_all()

        task = asyncio.create_task(
            execute(),
            name=f"pygent-worker-{run.execution_id}",
            context=ContextVarsContext(),
        )
        task.add_done_callback(self._observe_task)
        run.task = task

    async def _finish_cancelled(self, run: _ServerExecution) -> None:
        failure = _worker_failure("cancelled", "Worker execution was cancelled")
        run.error = freeze_json(failure.to_dict())
        run.status = "cancelled"
        await self._best_effort(
            self._emit(
                run,
                "execution.cancelled",
                freeze_json_object({"failure": failure.to_dict()}),
            )
        )
        await self._best_effort(self._persist_job(run, status="cancelled"))

    async def _finish_failed(
        self, run: _ServerExecution, failure: ExecutionFailure
    ) -> None:
        run.error = freeze_json(failure.to_dict())
        run.status = "failed"
        await self._best_effort(
            self._emit(
                run,
                "execution.failed",
                freeze_json_object({"failure": failure.to_dict()}),
            )
        )
        await self._best_effort(self._persist_job(run, status="failed"))

    @staticmethod
    async def _best_effort(operation: Awaitable[object]) -> bool:
        try:
            await operation
        except Exception:  # noqa: BLE001 - terminal state must remain observable
            return False
        return True

    @staticmethod
    def _status(run: _ServerExecution) -> str:
        return run.status

    @staticmethod
    def _observe_task(task: asyncio.Task[JsonValue]) -> None:
        if not task.cancelled():
            task.exception()

    def _purge_terminal_executions(self) -> None:
        while len(self.executions) >= self.max_retained_executions:
            terminal_id = next(
                (
                    execution_id
                    for execution_id, run in self.executions.items()
                    if run.terminal
                ),
                None,
            )
            if terminal_id is None:
                return
            run = self.executions.pop(terminal_id)
            if run.task is not None:
                self._observe_task(run.task)
            self._requests.pop(self._logical_request_key(run.invocation), None)

    async def result(self, request: Request) -> Response:
        run = self.executions.get(request.path_params["execution_id"])
        if run is None and self.history is not None:
            stored = await self.history.get_task(request.path_params["execution_id"])
            if stored is not None and stored.kind == "job":
                run = await self._restore_job(stored)
        if run is None:
            return JSONResponse({"error": "execution_not_found"}, status_code=404)
        status = self._status(run)
        if not run.terminal or status in {"pending", "running"}:
            return JSONResponse(
                {"execution_id": run.execution_id, "status": status}, status_code=202
            )
        if status == "cancelled":
            return JSONResponse(
                {
                    "execution_id": run.execution_id,
                    "status": "cancelled",
                    "error": thaw_json(run.error),
                },
                status_code=409,
            )
        if status != "succeeded":
            return JSONResponse(
                {
                    "execution_id": run.execution_id,
                    "status": status,
                    "error": thaw_json(run.error),
                },
                status_code=422,
            )
        if run.result is None:
            return JSONResponse({"error": "missing_result"}, status_code=500)
        return JSONResponse(
            {
                "execution_id": run.execution_id,
                "status": "succeeded",
                "result": thaw_json(run.result),
            }
        )

    async def _emit(
        self,
        run: _ServerExecution,
        kind: str,
        data: Mapping[str, JsonValue],
    ) -> None:
        async with run.condition:
            index = run.next_event_index
            run.next_event_index += 1
            event = {
                "schema_version": "0.2",
                "event_id": str(uuid4()),
                "sequence": index,
                "timestamp_unix_ns": int(time() * 1_000_000_000),
                "kind": kind,
                "data": thaw_json(cast(JsonValue, freeze_json_object(data))),
                "execution_id": run.execution_id,
                "trace_id": run.invocation.trace_id or run.execution_id,
                "span_id": run.root_span_id,
                "parent_span_id": run.invocation.parent_span_id,
                "module_path": "worker",
            }
            if self.history is not None:
                await self.history.append_event(
                    execution_id=run.execution_id, index=index, event=event
                )
            run.events.append(event)
            if len(run.events) > self.max_retained_events:
                del run.events[: len(run.events) - self.max_retained_events]
            run.condition.notify_all()

    async def _relay_event(self, run: _ServerExecution, origin: ExecutionEvent) -> None:
        async with run.condition:
            sequence = run.next_event_index
            run.next_event_index += 1
            data = freeze_json_object(origin.data).to_dict()
            data.update(
                {
                    "origin_execution_id": origin.execution_id,
                    "origin_sequence": origin.sequence,
                }
            )
            event = {
                "schema_version": "0.2",
                "event_id": origin.event_id,
                "sequence": sequence,
                "timestamp_unix_ns": origin.timestamp_unix_ns,
                "kind": origin.kind,
                "data": data,
                "execution_id": run.execution_id,
                "trace_id": origin.trace_id,
                "span_id": origin.span_id,
                "parent_span_id": origin.parent_span_id,
                "module_path": origin.module_path,
            }
            if self.history is not None:
                await self.history.append_event(
                    execution_id=run.execution_id, index=sequence, event=event
                )
            run.events.append(event)
            if len(run.events) > self.max_retained_events:
                del run.events[: len(run.events) - self.max_retained_events]
            run.condition.notify_all()

    async def cancel(self, request: Request) -> Response:
        run = self.executions.get(request.path_params["execution_id"])
        if run is None:
            return JSONResponse({"error": "execution_not_found"}, status_code=404)
        task = run.task
        if task is None or task.done():
            return JSONResponse({"cancelled": False})
        task.cancel()
        return JSONResponse({"cancelled": True})

    async def events(self, request: Request) -> Response:
        run = self.executions.get(request.path_params["execution_id"])
        if run is None and self.history is not None:
            stored = await self.history.get_task(request.path_params["execution_id"])
            if stored is not None and stored.kind == "job":
                run = await self._restore_job(stored)
        if run is None:
            return JSONResponse({"error": "execution_not_found"}, status_code=404)
        after_text = request.query_params.get(
            "after", request.headers.get("last-event-id", "-1")
        )
        try:
            after = int(after_text)
        except ValueError:
            return JSONResponse({"error": "invalid_cursor"}, status_code=400)
        if (
            self.history is None
            and run.events
            and after < int(run.events[0]["sequence"]) - 1
        ):
            return JSONResponse(
                {
                    "error": "cursor_expired",
                    "oldest_available": run.events[0]["sequence"],
                },
                status_code=409,
            )

        async def generate() -> AsyncIterator[bytes]:
            cursor = after
            while True:
                available: list[dict[str, Any]] = []
                if self.history is not None:
                    for value in await self.history.events_after(
                        execution_id=run.execution_id, after=cursor, limit=4096
                    ):
                        thawed = thaw_json(value)
                        if isinstance(thawed, dict):
                            available.append(thawed)
                async with run.condition:
                    if not available:
                        available = [
                            event for event in run.events if event["sequence"] > cursor
                        ]
                    drained = run.terminal and (
                        not run.events or cursor >= int(run.events[-1]["sequence"])
                    )
                    if not available and not drained:
                        await run.condition.wait()
                        continue
                for event in available:
                    cursor = int(event["sequence"])
                    data = json.dumps(event, separators=(",", ":"))
                    yield f"id: {cursor}\nevent: execution\ndata: {data}\n\n".encode()
                if drained:
                    return

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )


class HTTPWorkerClient:
    """Client that resolves stable Binding targets and fails over safely."""

    def __init__(
        self,
        registry: WorkerRegistry,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.registry = registry
        self._client = httpx.AsyncClient(transport=transport, timeout=timeout)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    async def health(self, target: WorkerTarget) -> tuple[str, ...]:
        capabilities, _ = await self._health_details(target)
        return capabilities

    async def _health_details(
        self, target: WorkerTarget
    ) -> tuple[tuple[str, ...], str | None]:
        response = await self._client.get(f"{target.endpoint.rstrip('/')}/health")
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "ok":
            raise WorkerUnavailableError(
                f"Worker {target.target_id} health status is not ok"
            )
        capabilities = payload.get("capabilities", ())
        if not isinstance(capabilities, list) or any(
            not isinstance(value, str) or not value for value in capabilities
        ):
            raise WorkerProtocolError("Worker health capabilities are invalid")
        namespace = payload.get("model_store_namespace")
        if namespace is not None and (
            not isinstance(namespace, str) or not namespace
        ):
            raise WorkerProtocolError("Worker model store namespace is invalid")
        return tuple(capabilities), namespace

    async def invoke(
        self,
        binding_ref: str,
        input: Mapping[str, Any],
        *,
        request_id: str,
        plan_id: str,
        graph_hash: str,
        placement: PlacementPolicy | None = None,
        deadline: float | None = None,
        required_capabilities: tuple[str, ...] = (),
        idempotency_key: str | None = None,
        trace_id: str | None = None,
        parent_execution_id: str | None = None,
        parent_span_id: str | None = None,
        attempt: int = 1,
        model_calls: Mapping[str, object] | None = None,
        model_admission_ref: str | None = None,
        model_store_namespace: str | None = None,
    ) -> tuple[str, JsonValue]:
        ref = await self.start(
            binding_ref,
            input,
            request_id=request_id,
            plan_id=plan_id,
            graph_hash=graph_hash,
            placement=placement or PlacementPolicy.adaptive(),
            deadline=deadline,
            required_capabilities=required_capabilities,
            idempotency_key=idempotency_key,
            trace_id=trace_id,
            parent_execution_id=parent_execution_id,
            parent_span_id=parent_span_id,
            attempt=attempt,
            model_calls=model_calls,
            model_admission_ref=model_admission_ref,
            model_store_namespace=model_store_namespace,
        )
        try:
            result = await self.result(ref, deadline=deadline)
        except (asyncio.CancelledError, TimeoutError):
            try:
                await asyncio.shield(self.cancel(ref))
            except (WorkerProtocolError, httpx.HTTPError):
                pass
            raise
        return ref.execution_id, result

    async def start(
        self,
        binding_ref: str,
        input: Mapping[str, Any],
        *,
        request_id: str,
        plan_id: str,
        graph_hash: str,
        placement: PlacementPolicy | None = None,
        deadline: float | None = None,
        required_capabilities: tuple[str, ...] = (),
        idempotency_key: str | None = None,
        trace_id: str | None = None,
        parent_execution_id: str | None = None,
        parent_span_id: str | None = None,
        attempt: int = 1,
        model_calls: Mapping[str, object] | None = None,
        model_admission_ref: str | None = None,
        model_store_namespace: str | None = None,
        _exclude_target_ids: frozenset[str] = frozenset(),
    ) -> RemoteExecutionHandle:
        """Start a remote run and return its identity before it completes."""

        _validate_plan_identity(plan_id, graph_hash)
        placement = placement or PlacementPolicy.adaptive()
        if not isinstance(placement, PlacementPolicy):
            raise TypeError("placement must be a PlacementPolicy")
        if placement.mode is PlacementMode.INHERIT:
            raise WorkerProtocolError(
                "HTTP Worker placement cannot inherit a local Runtime"
            )
        failures: list[str] = []
        targets = self.registry.resolve(
            binding_ref, required_capabilities=required_capabilities
        )
        if placement.mode is PlacementMode.PINNED:
            targets = tuple(
                target for target in targets if target.target_id == placement.target_id
            )
            if not targets:
                raise WorkerUnavailableError(
                    "pinned Worker target is not declared, capable, or healthy"
                )
        for target in targets:
            if target.target_id in _exclude_target_ids:
                continue
            remaining = None if deadline is None else deadline - monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError("remote Worker deadline expired")
            try:
                healthy_capabilities, healthy_namespace = await self._health_details(
                    target
                )
                if not set(required_capabilities) <= set(healthy_capabilities):
                    failures.append(f"{target.target_id}:capability-mismatch")
                    continue
                if "model.deferred.exact-pin.v1" in required_capabilities:
                    if healthy_namespace is None:
                        failures.append(f"{target.target_id}:missing-model-store")
                        continue
                    if (
                        model_store_namespace is not None
                        and healthy_namespace != model_store_namespace
                    ):
                        failures.append(f"{target.target_id}:model-store-mismatch")
                        continue
                    model_store_namespace = healthy_namespace
                wire_request: dict[str, object] = {
                    "request_id": request_id,
                    "input": dict(input),
                    "plan_id": plan_id,
                    "graph_hash": graph_hash,
                    "required_capabilities": list(required_capabilities),
                    "deadline_seconds": remaining,
                    "idempotency_key": idempotency_key,
                    "trace_id": trace_id,
                    "parent_execution_id": parent_execution_id,
                    "parent_span_id": parent_span_id,
                    "attempt": attempt,
                }
                if model_calls:
                    wire_request["model_calls"] = dict(model_calls)
                if model_admission_ref is not None:
                    wire_request["model_admission_ref"] = model_admission_ref
                if model_store_namespace is not None:
                    wire_request["model_store_namespace"] = model_store_namespace
                response = await self._client.post(
                    f"{target.endpoint.rstrip('/')}/v1/bindings/{binding_ref}/invoke",
                    json=wire_request,
                    timeout=remaining,
                )
            except (
                httpx.HTTPError,
                TimeoutError,
                WorkerProtocolError,
                WorkerUnavailableError,
            ) as exc:
                failures.append(f"{target.target_id}:{type(exc).__name__}")
                continue
            if response.status_code == 429 or response.status_code >= 500:
                failures.append(f"{target.target_id}:http-{response.status_code}")
                continue
            payload = response.json()
            if response.status_code >= 400:
                raise WorkerRemoteError(
                    _worker_failure(
                        "invocation_rejected",
                        f"Worker {target.target_id} rejected invocation",
                    )
                )
            execution_id = payload.get("execution_id")
            if not isinstance(execution_id, str) or not execution_id:
                raise WorkerProtocolError("Worker start response has no execution_id")
            frozen_input = freeze_json(input)
            if not isinstance(frozen_input, FrozenJsonObject):
                raise WorkerProtocolError("Worker invocation input must be an object")
            return RemoteExecutionHandle(
                client=self,
                execution_id=execution_id,
                target=target,
                binding_ref=binding_ref,
                input=frozen_input,
                request_id=request_id,
                required_capabilities=tuple(required_capabilities),
                idempotency_key=idempotency_key,
                trace_id=trace_id,
                parent_execution_id=parent_execution_id,
                parent_span_id=parent_span_id,
                attempt=attempt,
                attempted_target_ids=tuple(
                    sorted(_exclude_target_ids | {target.target_id})
                ),
                plan_id=plan_id,
                graph_hash=graph_hash,
                placement_mode=placement.mode,
                pinned_target_id=placement.target_id,
                model_calls=freeze_json_object(model_calls or {}),
                model_admission_ref=model_admission_ref,
                model_store_namespace=model_store_namespace,
            )
        raise WorkerUnavailableError(
            "all Worker targets unavailable: " + ", ".join(failures)
        )

    async def result(
        self,
        ref: RemoteExecutionHandle,
        *,
        deadline: float | None = None,
        poll_interval: float = 0.01,
    ) -> JsonValue:
        """Poll one remote run until its portable result becomes terminal."""

        if poll_interval < 0:
            raise ValueError("poll_interval cannot be negative")
        endpoint = ref.target.endpoint.rstrip("/")
        while True:
            remaining = None if deadline is None else deadline - monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError("remote Worker deadline expired")
            try:
                response = await self._client.get(
                    f"{endpoint}/v1/executions/{ref.execution_id}", timeout=remaining
                )
            except httpx.TransportError:
                ref = await self._failover_ref(
                    ref, deadline=deadline, failed=ref.target.target_id
                )
                endpoint = ref.target.endpoint.rstrip("/")
                continue
            payload = response.json()
            if response.status_code == 202:
                await asyncio.sleep(poll_interval)
                continue
            if response.status_code == 404 or response.status_code >= 500:
                ref = await self._failover_ref(
                    ref, deadline=deadline, failed=ref.target.target_id
                )
                endpoint = ref.target.endpoint.rstrip("/")
                continue
            if response.status_code >= 400:
                try:
                    failure = ExecutionFailure.from_dict(payload.get("error"))
                except (TypeError, ValueError, KeyError) as exc:
                    raise WorkerProtocolError(
                        "Worker result response has an invalid failure"
                    ) from exc
                raise WorkerRemoteError(failure)
            if "result" not in payload:
                raise WorkerProtocolError("Worker result response has no result")
            return freeze_json(payload["result"])

    async def _failover_ref(
        self, ref: RemoteExecutionHandle, *, deadline: float | None, failed: str
    ) -> RemoteExecutionHandle:
        if (
            ref.idempotency_key is None
            or "durability.shared-worker-jobs" not in ref.required_capabilities
        ):
            raise WorkerOutcomeUnknownError(
                "remote Worker outcome is unknown; failover requires a stable "
                "idempotency key and durability.shared-worker-jobs"
            )
        if ref.input is None or not ref.binding_ref or not ref.request_id:
            raise WorkerUnavailableError(
                "remote Worker failed and the logical invocation is unavailable"
            )
        return await self.start(
            ref.binding_ref,
            cast(Mapping[str, Any], thaw_json(ref.input)),
            request_id=ref.request_id,
            plan_id=ref.plan_id,
            graph_hash=ref.graph_hash,
            placement=PlacementPolicy(ref.placement_mode, ref.pinned_target_id),
            deadline=deadline,
            required_capabilities=ref.required_capabilities,
            idempotency_key=ref.idempotency_key,
            trace_id=ref.trace_id,
            parent_execution_id=ref.parent_execution_id,
            parent_span_id=ref.parent_span_id,
            attempt=ref.attempt,
            model_calls=cast(Mapping[str, object], thaw_json(ref.model_calls)),
            model_admission_ref=ref.model_admission_ref,
            model_store_namespace=ref.model_store_namespace,
            _exclude_target_ids=frozenset(ref.attempted_target_ids) | {failed},
        )

    async def cancel(self, ref: RemoteExecutionHandle) -> bool:
        """Request cancellation of a remotely owned run."""

        try:
            response = await self._client.post(
                f"{ref.target.endpoint.rstrip('/')}/v1/executions/{ref.execution_id}/cancel"
            )
        except httpx.TransportError as exc:
            raise WorkerUnavailableError(
                "remote Worker unavailable during cancellation"
            ) from exc
        response.raise_for_status()
        return bool(response.json().get("cancelled", False))

    async def events(
        self,
        target: WorkerTarget,
        execution_id: str,
        *,
        after: int = -1,
        reconnects: int = 3,
    ) -> AsyncIterator[ExecutionEvent]:
        cursor = after
        observed_event_ids: set[str] = set()
        for attempt in range(reconnects + 1):
            try:
                async with aconnect_sse(
                    self._client,
                    "GET",
                    f"{target.endpoint.rstrip('/')}/v1/executions/{execution_id}/events",
                    params={"after": cursor},
                    headers={"Last-Event-ID": str(cursor)},
                ) as source:
                    source.response.raise_for_status()
                    async for event in source.aiter_sse():
                        payload = freeze_json(json.loads(event.data))
                        if not isinstance(payload, FrozenJsonObject):
                            raise WorkerProtocolError("SSE event must be an object")
                        event_cursor = payload["sequence"]
                        if type(event_cursor) is not int or event_cursor <= cursor:
                            continue
                        cursor = event_cursor
                        event_id = payload.get("event_id")
                        if (
                            not isinstance(event_id, str)
                            or event_id in observed_event_ids
                        ):
                            continue
                        observed_event_ids.add(event_id)
                        data: object = payload.get("data", {})
                        if not isinstance(data, Mapping):
                            raise WorkerProtocolError(
                                "SSE event data must be an object"
                            )
                        yield ExecutionEvent(
                            schema_version=cast(str, payload.get("schema_version")),
                            event_id=event_id,
                            execution_id=cast(str, payload.get("execution_id")),
                            trace_id=cast(str, payload.get("trace_id")),
                            span_id=cast(str, payload.get("span_id")),
                            parent_span_id=cast(
                                str | None, payload.get("parent_span_id")
                            ),
                            sequence=event_cursor,
                            timestamp_unix_ns=cast(
                                int, payload.get("timestamp_unix_ns")
                            ),
                            module_path=cast(str, payload.get("module_path")),
                            kind=cast(str, payload.get("kind")),
                            data=cast(Mapping[str, object], data),
                        )
                    return
            except httpx.TransportError:
                if attempt == reconnects:
                    raise WorkerUnavailableError("SSE reconnect budget exhausted")
                await asyncio.sleep(0)


def bound_module_worker_handler(
    bindings: Mapping[str, BoundModule[Any, Any]],
    *,
    artifact_resolver: WorkerArtifactResolver | None = None,
) -> WorkerHandler:
    """Adapt declared BoundModules to the portable HTTP Worker payload."""

    declared = dict(bindings)
    for binding_ref, bound in declared.items():
        plan = getattr(bound, "plan", None)
        if plan is None or not plan.is_portable:
            raise WorkerProtocolError(
                "HTTP Worker binding "
                f"{binding_ref!r} requires a portable ExecutionPlan with "
                "artifact, entrypoint, schemas, and serializer"
            )
        if artifact_resolver is None or plan.artifact is None:
            raise WorkerProtocolError(
                "HTTP Worker requires a deployment artifact resolver"
            )
        manifest = artifact_resolver(plan.artifact)
        if not isinstance(manifest, WorkerDeploymentManifest):
            raise WorkerProtocolError("artifact resolver returned an invalid manifest")
        entrypoint_name = (
            f"{manifest.entrypoint.__module__}:{manifest.entrypoint.__qualname__}"
        )
        root_spec = next(spec for spec in plan.modules if spec.path == plan.root)
        if (
            manifest.artifact != plan.artifact
            or manifest.verified_digest != plan.artifact.digest
            or entrypoint_name != plan.artifact.entrypoint
            or manifest.input_schema != root_spec.input_schema
            or manifest.output_schema != root_spec.output_schema
            or manifest.serializer != root_spec.serializer
        ):
            raise WorkerProtocolError(
                f"Worker deployment manifest does not verify {binding_ref!r}"
            )

    async def invoke(
        request: WorkerInvocation, event_sink: WorkerEventSink
    ) -> JsonValue:
        try:
            bound = declared[request.binding_ref]
        except KeyError as exc:
            raise WorkerRemoteError(
                _worker_failure(
                    "binding_not_found",
                    f"undeclared Worker binding_ref {request.binding_ref!r}",
                )
            ) from exc
        plan = getattr(bound, "plan", None)
        if plan is None:
            raise WorkerRemoteError(
                _worker_failure(
                    "invalid_execution_plan",
                    f"Worker binding {request.binding_ref!r} has no ExecutionPlan",
                )
            )
        if request.plan_id != plan.plan_id or request.graph_hash != plan.graph_hash:
            raise WorkerRemoteError(
                _worker_failure(
                    "invalid_execution_plan",
                    "remote ExecutionPlan identity does not match the declared "
                    f"Worker binding {request.binding_ref!r}",
                )
            )
        has_deferred_models = any(spec.model_requirements for spec in plan.modules)
        if has_deferred_models:
            if "model.deferred.exact-pin.v1" not in request.required_capabilities:
                raise WorkerRemoteError(
                    _worker_failure(
                        "capability_mismatch",
                        "dynamic model execution requires model.deferred.exact-pin.v1",
                    )
                )
            if request.model_admission_ref != request.request_id:
                raise WorkerRemoteError(
                    _worker_failure(
                        "invalid_model_admission",
                        "dynamic model admission reference must equal the stable request identity",
                    )
                )
            runtime = getattr(bound, "runtime", None)
            scope_id = getattr(bound, "deployment_scope_id", None)
            store = getattr(runtime, "model_deployment_store", None)
            ensure_open = getattr(runtime, "_ensure_model_store_open", None)
            resolvers = getattr(runtime, "_model_resource_resolvers", None)
            if (
                not isinstance(scope_id, str)
                or store is None
                or not callable(ensure_open)
                or not isinstance(resolvers, dict)
            ):
                raise WorkerRemoteError(
                    _worker_failure(
                        "model_deployment_unavailable",
                        "Worker cannot validate an exact model deployment manifest",
                    )
                )
            store_namespace = getattr(store, "namespace_id", None)
            if (
                not isinstance(store_namespace, str)
                or request.model_store_namespace != store_namespace
            ):
                raise WorkerRemoteError(
                    _worker_failure(
                        "model_store_namespace_mismatch",
                        "Worker model deployment store namespace does not match the coordinator",
                    )
                )
            requirements = tuple(
                sorted(
                    {
                        requirement.group_name
                        for spec in plan.modules
                        for requirement in spec.model_requirements
                    }
                )
            )
            selections: dict[str, str | None] = {}
            for group_name in requirements:
                raw = request.model_calls.get(group_name)
                options = (
                    ModelCallOptions()
                    if raw is None
                    else ModelCallOptions.from_dict(cast(Mapping[str, object], raw))
                )
                selections[group_name] = options.profile
            await ensure_open()
            admission = await store.admit(
                scope_id,
                requirements,
                selections,
                admission_id=request.model_admission_ref,
            )
            for _, snapshot in admission.snapshots:
                resources = snapshot.resources
                if resources is None or resources.resolver_id not in resolvers:
                    raise WorkerRemoteError(
                        _worker_failure(
                            "model_deployment_unavailable",
                            "Worker cannot reconstruct the exact model deployment",
                        )
                    )
                resolver = resolvers[resources.resolver_id]
                resolver_domain = getattr(resolver, "coordinator_domain", None)
                if (
                    resolver_domain is not None
                    and resolver_domain != resources.coordinator_domain
                ):
                    raise WorkerRemoteError(
                        _worker_failure(
                            "model_capacity_domain_mismatch",
                            "Worker model resolver uses a different capacity domain",
                        )
                    )
        message, context = invocation_from_dict(request.input)
        handle = await bound.start(
            message,
            context,
            execution=ExecutionOptions(
                request_id=request.request_id,
                execution_id=request.request_id,
                trace_id=request.trace_id,
                parent_execution_id=request.parent_execution_id,
                parent_span_id=request.parent_span_id,
                idempotency_key=request.idempotency_key,
                context_ref=(
                    None
                    if request.parent_execution_id is None
                    else f"parent-run:{request.parent_execution_id}"
                ),
                deadline=request.deadline,
                model_calls=request.model_calls,
            ),
        )

        async def relay() -> None:
            async with handle.subscribe() as events:
                async for event in events:
                    await event_sink(event)

        relay_task = asyncio.create_task(relay(), name="pygent-worker-event-relay")
        try:
            output, next_context = await handle.result()
        except asyncio.CancelledError:
            await handle.cancel()
            relay_task.cancel()
            await asyncio.gather(relay_task, return_exceptions=True)
            raise
        except BaseException:
            relay_task.cancel()
            await asyncio.gather(relay_task, return_exceptions=True)
            raise
        else:
            await relay_task
        return invocation_to_dict(output, next_context)

    dynamic_namespaces = {
        namespace
        for bound in declared.values()
        if any(
            spec.model_requirements
            for spec in cast(Any, bound).plan.modules
        )
        for namespace in (
            getattr(
                getattr(getattr(bound, "runtime", None), "model_deployment_store", None),
                "namespace_id",
                None,
            ),
        )
        if isinstance(namespace, str) and namespace
    }
    if len(dynamic_namespaces) > 1:
        raise WorkerProtocolError(
            "dynamic Worker bindings must share one model store namespace"
        )
    if dynamic_namespaces:
        cast(Any, invoke)._pygent_model_store_namespace = next(iter(dynamic_namespaces))
    return invoke


@dataclass(frozen=True, slots=True)
class HTTPRemoteModuleTarget:
    """Managed RemoteModule target backed by the HTTP Worker protocol."""

    client: HTTPWorkerClient
    binding_ref: str
    plan_id: str
    graph_hash: str
    required_capabilities: tuple[str, ...] = ()
    placement: PlacementPolicy = field(default_factory=PlacementPolicy.adaptive)

    def __post_init__(self) -> None:
        if not self.binding_ref:
            raise ValueError("binding_ref must be non-empty")
        _validate_plan_identity(self.plan_id, self.graph_hash)
        capabilities = tuple(self.required_capabilities)
        if any(not isinstance(item, str) or not item for item in capabilities):
            raise ValueError("required capabilities must be non-empty strings")
        if len(capabilities) != len(set(capabilities)):
            raise ValueError("required capabilities must be unique")
        object.__setattr__(self, "required_capabilities", capabilities)
        if not isinstance(self.placement, PlacementPolicy):
            raise TypeError("placement must be a PlacementPolicy")
        if self.placement.mode is PlacementMode.INHERIT:
            raise ValueError("HTTP remote target placement cannot be inherit")

    async def invoke(
        self, message: Message, context: Context, *, deadline: float | None
    ) -> tuple[Message, Context]:
        return await self.invoke_remote_child(
            message,
            context,
            plan_id=self.plan_id,
            graph_hash=self.graph_hash,
            required_capabilities=self.required_capabilities,
            placement_mode=self.placement.mode.value,
            pinned_target_id=self.placement.target_id,
            deadline=deadline,
            trace_id=None,
            parent_execution_id=None,
            parent_span_id=None,
            event_sink=None,
            child_execution_id=None,
            attempt=1,
            idempotency_key=None,
        )

    async def invoke_remote_child(
        self,
        message: Message,
        context: Context,
        *,
        plan_id: str | None,
        graph_hash: str | None,
        required_capabilities: tuple[str, ...],
        placement_mode: str,
        pinned_target_id: str | None,
        deadline: float | None,
        trace_id: str | None,
        parent_execution_id: str | None,
        parent_span_id: str | None,
        event_sink: WorkerEventSink | None,
        child_execution_id: str | None,
        attempt: int,
        idempotency_key: str | None,
    ) -> tuple[Message, Context]:
        if plan_id is None or graph_hash is None:
            raise WorkerProtocolError(
                "distributed RemoteModule requires a stable ExecutionPlan identity"
            )
        _validate_plan_identity(plan_id, graph_hash)
        if (
            plan_id != self.plan_id
            or graph_hash != self.graph_hash
            or tuple(required_capabilities) != self.required_capabilities
        ):
            raise WorkerProtocolError(
                "registered remote target identity differs from the RemoteModule "
                "declaration; compile a new caller ExecutionPlan"
            )
        placement = PlacementPolicy(PlacementMode(placement_mode), pinned_target_id)
        if placement != self.placement:
            raise WorkerProtocolError(
                "registered remote target placement differs from the "
                "RemoteModule declaration; compile a new caller ExecutionPlan"
            )
        payload = invocation_to_dict(message, context)
        request_id = child_execution_id or str(uuid4())
        logical_key = idempotency_key or (
            None if trace_id is None else f"remote-child:{trace_id}:{self.binding_ref}"
        )
        ref = await self.client.start(
            self.binding_ref,
            cast(Mapping[str, Any], thaw_json(payload)),
            request_id=request_id,
            plan_id=plan_id,
            graph_hash=graph_hash,
            placement=placement,
            deadline=deadline,
            required_capabilities=required_capabilities,
            idempotency_key=logical_key,
            trace_id=trace_id,
            parent_execution_id=parent_execution_id,
            parent_span_id=parent_span_id,
            attempt=attempt,
            model_admission_ref=(
                request_id
                if "model.deferred.exact-pin.v1" in required_capabilities
                else None
            ),
        )
        relay_task: asyncio.Task[None] | None = None
        if event_sink is not None:

            async def relay() -> None:
                async with ref.subscribe() as events:
                    async for event in events:
                        await event_sink(event)

            relay_task = asyncio.create_task(relay(), name="pygent-remote-event-relay")
        try:
            result = await self.client.result(ref, deadline=deadline)
        except (asyncio.CancelledError, TimeoutError):
            try:
                await asyncio.shield(self.client.cancel(ref))
            except (WorkerProtocolError, httpx.HTTPError):
                pass
            if relay_task is not None:
                relay_task.cancel()
                await asyncio.gather(relay_task, return_exceptions=True)
            raise
        except BaseException:
            if relay_task is not None:
                relay_task.cancel()
                await asyncio.gather(relay_task, return_exceptions=True)
            raise
        else:
            if relay_task is not None:
                await relay_task
        return invocation_from_dict(result)


__all__ = [
    "HTTPRemoteModuleTarget",
    "HTTPWorkerApp",
    "HTTPWorkerClient",
    "RemoteExecutionHandle",
    "WorkerArtifactResolver",
    "WorkerDeploymentManifest",
    "WorkerInvocation",
    "WorkerOutcomeUnknownError",
    "WorkerProtocolError",
    "WorkerRegistry",
    "WorkerRemoteError",
    "WorkerTarget",
    "WorkerUnavailableError",
    "bound_module_worker_handler",
]
