"""HTTP/SSE Worker client, failover, and remote event decoding."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from time import monotonic
from typing import Any, Self, cast

import httpx
from httpx_sse import aconnect_sse

from pygent.core import (
    ExecutionEvent,
    ExecutionFailure,
    FrozenJsonObject,
    JsonValue,
    PlacementMode,
    PlacementPolicy,
    freeze_json,
    freeze_json_object,
    thaw_json,
)

from ._worker_handle import RemoteExecutionHandle
from ._worker_protocol import (
    WorkerOutcomeUnknownError,
    WorkerProtocolError,
    WorkerRegistry,
    WorkerRemoteError,
    WorkerTarget,
    WorkerUnavailableError,
    _validate_plan_identity,
    _worker_failure,
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
