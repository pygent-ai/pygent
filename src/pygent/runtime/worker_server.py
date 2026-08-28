"""HTTP/SSE Worker server and durable execution journal."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import AsyncIterator, Awaitable, Mapping
from contextlib import asynccontextmanager
from contextvars import Context as ContextVarsContext
from dataclasses import dataclass, field
from time import monotonic, time
from typing import Any, cast
from uuid import uuid4

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from pygent.core import (
    EXECUTION_EVENT_SCHEMA_VERSION,
    ExecutionEvent,
    ExecutionFailure,
    ExecutionInputDelivery,
    FrozenJsonObject,
    JsonValue,
    freeze_json,
    freeze_json_object,
    thaw_json,
)

from ._history_store import SQLiteHistoryStore
from ._history_types import StoredTask
from ._worker_protocol import (
    MODEL_ROUTE_PROVIDER_OPTIONS_CAPABILITY,
    WorkerHandler,
    WorkerInvocation,
    WorkerProtocolError,
    WorkerRemoteError,
    _failure_from_exception,
    _validate_plan_identity,
    _worker_failure,
)


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
    attempt_id: str = field(default_factory=lambda: str(uuid4()))
    status: str = "pending"
    result: JsonValue | None = None
    error: JsonValue | None = None
    terminal: bool = False
    next_event_index: int = 0


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
        advertised_capabilities = list(capabilities)
        if (
            self.model_store_namespace is not None
            and MODEL_ROUTE_PROVIDER_OPTIONS_CAPABILITY not in advertised_capabilities
        ):
            advertised_capabilities.append(MODEL_ROUTE_PROVIDER_OPTIONS_CAPABILITY)
        self.capabilities = tuple(advertised_capabilities)
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
                    "/v1/executions/{execution_id:str}/inputs",
                    self.input,
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
                {
                    "execution_id": existing_id,
                    "attempt_id": existing.attempt_id,
                    "status": self._status(existing),
                },
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
            {
                "execution_id": execution_id,
                "attempt_id": server_run.attempt_id,
                "status": "running",
            },
            status_code=202,
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
        metadata = {
            "execution_id": run.execution_id,
            "attempt_id": run.attempt_id,
            "trace_id": run.invocation.trace_id or run.execution_id,
            "last_sequence": run.next_event_index - 1,
            "terminal_sequence": run.next_event_index - 1 if run.terminal else None,
        }
        if not run.terminal or status in {"pending", "running"}:
            return JSONResponse(
                {**metadata, "status": status}, status_code=202
            )
        if status == "cancelled":
            return JSONResponse(
                {
                    **metadata,
                    "status": "cancelled",
                    "error": thaw_json(run.error),
                },
                status_code=409,
            )
        if status != "succeeded":
            return JSONResponse(
                {
                    **metadata,
                    "status": status,
                    "error": thaw_json(run.error),
                },
                status_code=422,
            )
        if run.result is None:
            return JSONResponse({"error": "missing_result"}, status_code=500)
        return JSONResponse(
            {
                **metadata,
                "status": "succeeded",
                "result": thaw_json(run.result),
            }
        )

    async def input(self, request: Request) -> Response:
        run = self.executions.get(request.path_params["execution_id"])
        if run is None:
            return JSONResponse({"error": "execution_not_found"}, status_code=404)
        try:
            body = await request.json()
            if set(body) != {"input_id", "kind", "value"}:
                raise ValueError
            input_id = body["input_id"]
            kind = body["kind"]
            value = freeze_json(body["value"])
            if not isinstance(input_id, str) or not input_id:
                raise ValueError
            if not isinstance(kind, str) or not kind:
                raise ValueError
        except (TypeError, ValueError, json.JSONDecodeError):
            return JSONResponse({"error": "invalid_request"}, status_code=400)
        sender = getattr(self.handler, "_pygent_send_input", None)
        if not callable(sender):
            return JSONResponse(
                {"error": "execution_input_unavailable"}, status_code=501
            )
        try:
            delivery = await sender(
                run.invocation, input_id=input_id, kind=kind, value=value
            )
        except ValueError:
            return JSONResponse({"error": "invalid_request"}, status_code=400)
        except OverflowError:
            return JSONResponse({"error": "worker_capacity"}, status_code=429)
        if not isinstance(delivery, ExecutionInputDelivery):
            return JSONResponse({"error": "invalid_delivery"}, status_code=500)
        return JSONResponse(delivery.to_dict())

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
                "schema_version": EXECUTION_EVENT_SCHEMA_VERSION,
                "event_id": str(uuid4()),
                "sequence": index,
                "timestamp_unix_ns": int(time() * 1_000_000_000),
                "kind": kind,
                "data": thaw_json(cast(JsonValue, freeze_json_object(data))),
                "execution_id": run.execution_id,
                "attempt_id": run.attempt_id,
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
                "schema_version": EXECUTION_EVENT_SCHEMA_VERSION,
                "event_id": origin.event_id,
                "sequence": sequence,
                "timestamp_unix_ns": origin.timestamp_unix_ns,
                "kind": origin.kind,
                "data": data,
                "execution_id": run.execution_id,
                "attempt_id": origin.attempt_id,
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
