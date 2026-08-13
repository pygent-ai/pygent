from __future__ import annotations

import asyncio
from dataclasses import dataclass
from time import monotonic, time_ns
from uuid import uuid4

import httpx
import pytest

from pygent import (
    Agent,
    ModelCallError,
    ModelErrorKind,
)
from pygent.core import (
    AIMessage,
    Context,
    Message,
    Module,
    PlacementPolicy,
    RemoteModule,
    UserMessage,
    thaw_json,
)
from pygent.llm import ModelAttempt
from pygent.runtime import (
    CapacityPolicy,
    CapacityScope,
    CodeArtifactSpec,
    ExecutionCapacityPolicy,
    ExecutionEvent,
    ExecutionOptions,
    ExecutionPhase,
    LocalRuntime,
    SQLiteHistoryStore,
)
from pygent.runtime._worker_handle import RemoteExecutionHandle
from pygent.runtime._worker_protocol import (
    WorkerDeploymentManifest,
    WorkerOutcomeUnknownError,
    WorkerProtocolError,
    WorkerRegistry,
    WorkerRemoteError,
    WorkerTarget,
    WorkerUnavailableError,
)
from pygent.runtime.context_codec import BASE_CONTEXT_CODEC
from pygent.runtime.worker_client import HTTPWorkerClient
from pygent.runtime.worker_server import HTTPWorkerApp
from pygent.runtime.worker_target import (
    HTTPRemoteModuleTarget,
    bound_module_worker_handler,
)

_GRAPH_HASH = "0" * 64
_PLAN_ID = f"sha256:{_GRAPH_HASH}"


def _identity() -> dict[str, str]:
    return {"plan_id": _PLAN_ID, "graph_hash": _GRAPH_HASH}


def _portable_runtime(**kwargs) -> LocalRuntime:
    return LocalRuntime(
        code_artifact=CodeArtifactSpec(
            package="pygent-runtime-tests",
            version="0.2.0",
            digest="sha256:portable-worker-fixture",
            entrypoint="tests.runtime.test_http_worker:build_worker",
        ),
        input_schema="schema://pygent/message-context-input@0.3",
        output_schema="schema://pygent/message-context-output@0.3",
        serializer="pygent-json-v2",
        **kwargs,
    )


def build_worker() -> object:
    return object()


def _artifact_resolver(artifact: CodeArtifactSpec) -> WorkerDeploymentManifest:
    return WorkerDeploymentManifest(
        artifact=artifact,
        verified_digest=artifact.digest,
        entrypoint=build_worker,
        input_schema="schema://pygent/message-context-input@0.3",
        output_schema="schema://pygent/message-context-output@0.3",
        serializer="pygent-json-v2",
        context_codecs=(BASE_CONTEXT_CODEC.identity,),
    )


def _bound_worker_handler(bindings):
    return bound_module_worker_handler(bindings, artifact_resolver=_artifact_resolver)


@pytest.mark.asyncio
async def test_http_worker_invoke_health_and_cursor_sse():
    async def handler(invocation, event_sink):
        return {
            "binding": invocation.binding_ref,
            "echo": invocation.input["message"],
        }

    worker = HTTPWorkerApp(handler)
    transport = httpx.ASGITransport(app=worker.app)
    registry = WorkerRegistry()
    target = WorkerTarget("worker-1", "http://worker", ("local", "sse"))
    registry.publish("echo", (target,))

    async with HTTPWorkerClient(registry, transport=transport) as client:
        assert await client.health(target) == ("local", "sse")
        execution_id, result = await client.invoke(
            "echo",
            {"message": "hello"},
            request_id="request-1",
            plan_id=_PLAN_ID,
            graph_hash=_GRAPH_HASH,
            deadline=monotonic() + 1,
        )
        events = [event async for event in client.events(target, execution_id, after=0)]

    assert thaw_json(result) == {"binding": "echo", "echo": "hello"}
    assert [event.kind for event in events] == ["execution.completed"]


@pytest.mark.asyncio
async def test_http_worker_client_fails_over_declared_targets():
    attempts: list[str] = []

    async def send(request: httpx.Request) -> httpx.Response:
        attempts.append(request.url.host)
        if request.url.host == "unhealthy":
            return httpx.Response(503, json={"error": "unavailable"})
        if request.url.path == "/health":
            return httpx.Response(
                200, json={"status": "ok", "capabilities": ["durable"]}
            )
        if request.method == "POST":
            return httpx.Response(
                202,
                json={
                    "execution_id": "remote-run",
                    "attempt_id": "attempt-1",
                    "status": "running",
                },
            )
        return httpx.Response(
            200,
            json={
                "execution_id": "remote-run",
                "status": "succeeded",
                "result": {"ok": True},
            },
        )

    registry = WorkerRegistry()
    registry.publish(
        "service",
        (
            WorkerTarget("bad", "http://unhealthy", ("durable",)),
            WorkerTarget("good", "http://healthy", ("durable",)),
        ),
    )
    async with HTTPWorkerClient(
        registry, transport=httpx.MockTransport(send)
    ) as client:
        execution_id, result = await client.invoke(
            "service",
            {"value": 1},
            request_id="request-1",
            plan_id=_PLAN_ID,
            graph_hash=_GRAPH_HASH,
            required_capabilities=("durable",),
        )

    assert attempts == ["unhealthy", "healthy", "healthy", "healthy"]
    assert execution_id == "remote-run"
    assert thaw_json(result) == {"ok": True}


@pytest.mark.asyncio
async def test_stale_registry_capabilities_cannot_bypass_live_health_check():
    requests: list[tuple[str, str]] = []

    async def send(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "capabilities": ["local"]})
        raise AssertionError("client must not invoke a Worker missing live capability")

    registry = WorkerRegistry()
    registry.publish(
        "echo",
        (WorkerTarget("stale", "http://stale", ("local", "durable")),),
    )
    async with HTTPWorkerClient(
        registry, transport=httpx.MockTransport(send)
    ) as client:
        with pytest.raises(WorkerUnavailableError, match="unavailable"):
            await client.start(
                "echo",
                {},
                request_id="request-1",
                plan_id=_PLAN_ID,
                graph_hash=_GRAPH_HASH,
                required_capabilities=("durable",),
            )

    assert requests == [("GET", "/health")]


@pytest.mark.asyncio
async def test_worker_revalidates_required_capabilities_on_start():
    called = False

    async def handler(invocation, event_sink):
        nonlocal called
        called = True
        return {"ok": True}

    worker = HTTPWorkerApp(handler, capabilities=("local",))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=worker.app), base_url="http://worker"
    ) as client:
        response = await client.post(
            "/v1/bindings/echo/invoke",
            json={
                "request_id": "request-1",
                "input": {},
                "required_capabilities": ["durable"],
                **_identity(),
            },
        )

    assert response.status_code == 412
    assert response.json()["error"] == "capability_mismatch"
    assert not called


@pytest.mark.asyncio
async def test_pinned_placement_never_fails_over_to_another_declared_target():
    attempts: list[str] = []

    async def send(request: httpx.Request) -> httpx.Response:
        attempts.append(request.url.host)
        return httpx.Response(503, json={"error": "unavailable"})

    registry = WorkerRegistry()
    registry.publish(
        "service",
        (
            WorkerTarget("pinned", "http://pinned", ("durable",)),
            WorkerTarget("other", "http://other", ("durable",)),
        ),
    )
    async with HTTPWorkerClient(
        registry, transport=httpx.MockTransport(send)
    ) as client:
        with pytest.raises(WorkerUnavailableError):
            await client.start(
                "service",
                {},
                request_id="request-1",
                plan_id=_PLAN_ID,
                graph_hash=_GRAPH_HASH,
                required_capabilities=("durable",),
                placement=PlacementPolicy.pinned("pinned"),
            )

    assert attempts == ["pinned"]


def test_worker_registry_rejects_undeclared_or_incapable_targets():
    registry = WorkerRegistry()
    registry.publish("service", (WorkerTarget("worker", "http://worker"),))

    with pytest.raises(KeyError):
        registry.resolve("user-controlled-name")
    with pytest.raises(WorkerUnavailableError):
        registry.resolve("service", required_capabilities=("durable",))


@pytest.mark.asyncio
async def test_http_worker_cancel_propagates_to_owner_task():
    entered = asyncio.Event()

    async def handler(invocation, event_sink):
        entered.set()
        await asyncio.Event().wait()

    worker = HTTPWorkerApp(handler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=worker.app), base_url="http://worker"
    ) as client:
        response = await client.post(
            "/v1/bindings/slow/invoke",
            json={"request_id": "request-1", "input": {}, **_identity()},
        )
        await entered.wait()
        execution_id = response.json()["execution_id"]
        cancelled = await client.post(f"/v1/executions/{execution_id}/cancel")
        await asyncio.sleep(0)
        result = await client.get(f"/v1/executions/{execution_id}")

    assert response.status_code == 202
    assert cancelled.json() == {"cancelled": True}
    assert result.status_code == 409


@pytest.mark.asyncio
async def test_http_worker_rejects_missing_or_inconsistent_plan_identity():
    async def handler(invocation, event_sink):
        return {"ok": True}

    worker = HTTPWorkerApp(handler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=worker.app), base_url="http://worker"
    ) as client:
        missing = await client.post(
            "/v1/bindings/service/invoke",
            json={"request_id": "missing", "input": {}},
        )
        inconsistent = await client.post(
            "/v1/bindings/service/invoke",
            json={
                "request_id": "inconsistent",
                "input": {},
                "plan_id": _PLAN_ID,
                "graph_hash": "1" * 64,
            },
        )

    assert missing.status_code == 400
    assert inconsistent.status_code == 400
    assert worker.executions == {}


@pytest.mark.asyncio
async def test_client_start_exposes_run_for_parallel_events_and_cancel():
    entered = asyncio.Event()

    async def handler(invocation, event_sink):
        entered.set()
        await asyncio.Event().wait()

    worker = HTTPWorkerApp(handler)
    registry = WorkerRegistry()
    target = WorkerTarget("worker", "http://worker", ("sse",))
    registry.publish("slow", (target,))
    async with HTTPWorkerClient(
        registry, transport=httpx.ASGITransport(app=worker.app)
    ) as client:
        ref = await client.start(
            "slow",
            {},
            request_id="request-1",
            plan_id=_PLAN_ID,
            graph_hash=_GRAPH_HASH,
        )
        await entered.wait()
        events_task = asyncio.create_task(
            _collect_events(client, target, ref.execution_id)
        )
        assert await client.cancel(ref) is True
        with pytest.raises(WorkerRemoteError, match="cancelled"):
            await client.result(ref)
        events = await events_task

    assert [event.kind for event in events] == [
        "execution.started",
        "execution.cancelled",
    ]


@pytest.mark.asyncio
async def test_worker_close_cancels_and_joins_active_executions():
    entered = asyncio.Event()

    async def handler(invocation, event_sink):
        entered.set()
        await asyncio.Event().wait()

    worker = HTTPWorkerApp(handler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=worker.app), base_url="http://worker"
    ) as client:
        started = await client.post(
            "/v1/bindings/slow/invoke",
            json={"request_id": "request-1", "input": {}, **_identity()},
        )
        await entered.wait()
        await worker.close()
        run = worker.executions[started.json()["execution_id"]]

    assert run.task is not None and run.task.cancelled()
    assert run.terminal is True


@pytest.mark.asyncio
async def test_durable_worker_accepts_concurrent_jobs_without_nested_transactions(
    tmp_path,
):
    async def handler(invocation, event_sink):
        for index in range(8):
            await event_sink(
                ExecutionEvent(
                    schema_version="0.2",
                    event_id=str(uuid4()),
                    execution_id=invocation.request_id,
                    attempt_id="attempt-1",
                    trace_id=invocation.request_id,
                    span_id=invocation.request_id,
                    parent_span_id=None,
                    sequence=index,
                    timestamp_unix_ns=time_ns(),
                    module_path="test",
                    kind="model.text.delta",
                    data={"text": "x"},
                )
            )
        await asyncio.sleep(0.01)
        return {"request_id": invocation.request_id}

    path = tmp_path / "concurrent-worker.sqlite3"
    graph_hash = "a" * 64
    async with SQLiteHistoryStore(path) as history:
        worker = HTTPWorkerApp(handler, history=history)
        registry = WorkerRegistry()
        registry.publish(
            "service",
            (WorkerTarget("worker", "http://worker", ("local",)),),
        )
        async with HTTPWorkerClient(
            registry, transport=httpx.ASGITransport(app=worker.app)
        ) as client:
            results = await asyncio.gather(
                *(
                    client.invoke(
                        "service",
                        {"index": index},
                        request_id=f"request-{index}",
                        plan_id=f"sha256:{graph_hash}",
                        graph_hash=graph_hash,
                        idempotency_key=f"operation-{index}",
                    )
                    for index in range(8)
                )
            )
        await worker.close()

    assert [thaw_json(result) for _, result in results] == [
        {"request_id": f"request-{index}"} for index in range(8)
    ]


@pytest.mark.asyncio
async def test_claim_failure_becomes_terminal_instead_of_polling_forever(
    tmp_path, monkeypatch
):
    async with SQLiteHistoryStore(tmp_path / "worker.sqlite3") as history:

        async def fail_claim(**kwargs):
            raise RuntimeError("injected claim failure")

        monkeypatch.setattr(history, "claim_execution", fail_claim)
        worker = HTTPWorkerApp(
            lambda invocation, event_sink: asyncio.sleep(0), history=history
        )
        registry = WorkerRegistry()
        target = WorkerTarget("worker", "http://worker")
        registry.publish("service", (target,))
        async with HTTPWorkerClient(
            registry, transport=httpx.ASGITransport(app=worker.app)
        ) as client:
            ref = await client.start(
                "service",
                {},
                request_id="request-claim-failure",
                plan_id=_PLAN_ID,
                graph_hash=_GRAPH_HASH,
                idempotency_key="claim-failure",
            )
            with pytest.raises(WorkerRemoteError) as raised:
                await client.result(ref, deadline=monotonic() + 1)

        run = worker.executions[ref.execution_id]
        assert raised.value.kind == "persistence_error"
        assert run.status == "failed"
        assert run.terminal is True
        assert run.task is not None and run.task.done()
        await worker.close()


@pytest.mark.asyncio
async def test_worker_close_joins_claim_release_before_history_close(
    tmp_path, monkeypatch
):
    release_started = asyncio.Event()
    allow_release = asyncio.Event()
    handler_started = asyncio.Event()
    async with SQLiteHistoryStore(tmp_path / "worker.sqlite3") as history:
        original_release = history.release_execution_claim

        async def delayed_release(**kwargs):
            release_started.set()
            await allow_release.wait()
            return await original_release(**kwargs)

        monkeypatch.setattr(history, "release_execution_claim", delayed_release)

        async def handler(invocation, event_sink):
            del invocation, event_sink
            handler_started.set()
            await asyncio.Event().wait()

        worker = HTTPWorkerApp(handler, history=history)
        registry = WorkerRegistry()
        registry.publish("service", (WorkerTarget("worker", "http://worker"),))
        async with HTTPWorkerClient(
            registry, transport=httpx.ASGITransport(app=worker.app)
        ) as client:
            await client.start(
                "service",
                {"value": 1},
                request_id="request-1",
                **_identity(),
            )
            await asyncio.wait_for(handler_started.wait(), timeout=1)
            close_task = asyncio.create_task(worker.close())
            await asyncio.wait_for(release_started.wait(), timeout=1)
            await asyncio.sleep(0)
            assert not close_task.done()
            allow_release.set()
            await asyncio.wait_for(close_task, timeout=1)


@pytest.mark.asyncio
async def test_worker_preserves_model_failure_kind_attempts_and_partial_output():
    error = ModelCallError(
        "model capacity exhausted",
        kind=ModelErrorKind.RATE_LIMIT,
        attempts=(
            ModelAttempt("glm", "failed", ModelErrorKind.RATE_LIMIT, 1),
            ModelAttempt("glm", "failed", ModelErrorKind.RATE_LIMIT, 2),
        ),
        partial_output=True,
    )

    async def handler(invocation, event_sink):
        raise error

    worker = HTTPWorkerApp(handler)
    registry = WorkerRegistry()
    target = WorkerTarget("worker", "http://worker", ("sse",))
    registry.publish("service", (target,))
    async with HTTPWorkerClient(
        registry, transport=httpx.ASGITransport(app=worker.app)
    ) as client:
        ref = await client.start(
            "service",
            {},
            request_id="request-model-failure",
            plan_id=_PLAN_ID,
            graph_hash=_GRAPH_HASH,
        )
        with pytest.raises(WorkerRemoteError) as raised:
            await client.result(ref)
        events = [event async for event in client.events(target, ref.execution_id)]

    remote = raised.value
    assert remote.failure.domain == "model"
    assert remote.kind == "rate_limit"
    assert remote.failure.retryable is True
    assert remote.failure.partial_output is True
    assert len(remote.failure.details["attempts"]) == 2
    assert events[-1].kind == "execution.failed"
    assert events[-1].data["failure"]["kind"] == "rate_limit"


async def _collect_events(client, target, execution_id):
    return [event async for event in client.events(target, execution_id)]


@pytest.mark.asyncio
async def test_worker_deduplicates_start_and_rejects_conflicting_retry():
    release = asyncio.Event()

    async def handler(invocation, event_sink):
        await release.wait()
        return {"ok": True}

    worker = HTTPWorkerApp(handler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=worker.app), base_url="http://worker"
    ) as client:
        first = await client.post(
            "/v1/bindings/service/invoke",
            json={
                "request_id": "same",
                "idempotency_key": "operation",
                "input": {"value": 1},
                **_identity(),
            },
        )
        duplicate = await client.post(
            "/v1/bindings/service/invoke",
            json={
                "request_id": "retry-request",
                "idempotency_key": "operation",
                "input": {"value": 1},
                **_identity(),
            },
        )
        conflict = await client.post(
            "/v1/bindings/service/invoke",
            json={
                "request_id": "conflict-request",
                "idempotency_key": "operation",
                "input": {"value": 2},
                **_identity(),
            },
        )
        release.set()

    assert first.json()["execution_id"] == duplicate.json()["execution_id"]
    assert conflict.status_code == 409


@pytest.mark.asyncio
async def test_sse_rejects_cursor_older_than_bounded_retention():
    async def handler(invocation, event_sink):
        return {"ok": True}

    worker = HTTPWorkerApp(handler, max_retained_events=1)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=worker.app), base_url="http://worker"
    ) as client:
        started = await client.post(
            "/v1/bindings/service/invoke",
            json={"request_id": "request-1", "input": {}, **_identity()},
        )
        execution_id = started.json()["execution_id"]
        while (await client.get(f"/v1/executions/{execution_id}")).status_code == 202:
            await asyncio.sleep(0)
        expired = await client.get(f"/v1/executions/{execution_id}/events?after=-1")
        current = await client.get(f"/v1/executions/{execution_id}/events?after=0")

    assert expired.status_code == 409
    assert expired.json()["oldest_available"] == 1
    assert "id: 1" in current.text


@pytest.mark.asyncio
async def test_sse_client_reconnects_from_last_observed_cursor():
    requests: list[str] = []

    async def send(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        after = request.url.params["after"]
        if after == "-1":
            stream = _InterruptedSSE(
                b'id: 0\nevent: execution\ndata: {"schema_version":"1","event_id":"event-0","execution_id":"run-1","attempt_id":"attempt-1","trace_id":"run-1","span_id":"span-1","parent_span_id":null,"sequence":0,"timestamp_unix_ns":1,"module_path":"worker","kind":"execution.started","data":{}}\n\n'
            )
        else:
            stream = _CompleteSSE(
                b'id: 1\nevent: execution\ndata: {"schema_version":"1","event_id":"event-1","execution_id":"run-1","attempt_id":"attempt-1","trace_id":"run-1","span_id":"span-1","parent_span_id":null,"sequence":1,"timestamp_unix_ns":2,"module_path":"worker","kind":"execution.completed","data":{}}\n\n'
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
        )

    registry = WorkerRegistry()
    target = WorkerTarget("worker", "http://worker", ("sse",))
    registry.publish("service", (target,))
    async with HTTPWorkerClient(
        registry, transport=httpx.MockTransport(send)
    ) as client:
        events = [event async for event in client.events(target, "run-1")]

    assert [event.sequence for event in events] == [0, 1]
    assert "after=-1" in requests[0]
    assert "after=0" in requests[1]


class _InterruptedSSE(httpx.AsyncByteStream):
    def __init__(self, chunk: bytes) -> None:
        self.chunk = chunk

    async def __aiter__(self):
        yield self.chunk
        raise httpx.ReadError("connection dropped")


class _CompleteSSE(httpx.AsyncByteStream):
    def __init__(self, chunk: bytes) -> None:
        self.chunk = chunk

    async def __aiter__(self):
        yield self.chunk


@pytest.mark.asyncio
async def test_remote_module_executes_through_http_worker_protocol():
    class Echo(Module[UserMessage, AIMessage]):
        async def forward(self, message, context):
            output = AIMessage(content=message.content.upper())
            return output, context + message + output

    worker_runtime = _portable_runtime()
    worker_bound = worker_runtime.bind(Echo())
    assert worker_bound.plan.is_portable
    assert worker_bound.plan.artifact is not None
    assert worker_bound.plan.artifact.entrypoint == (
        "tests.runtime.test_http_worker:build_worker"
    )

    class Caller(Module[UserMessage, AIMessage]):
        def __init__(self):
            super().__init__()
            self.remote = RemoteModule[UserMessage, AIMessage](
                binding_ref="echo-service",
                plan_id=worker_bound.plan.plan_id,
                graph_hash=worker_bound.plan.graph_hash,
                required_capabilities=("local",),
            )

        async def forward(self, message, context):
            return await self.remote(message, context)

    captured = []
    bound_handler = _bound_worker_handler({"echo-service": worker_bound})

    async def handler(invocation, event_sink):
        captured.append(invocation)
        return await bound_handler(invocation, event_sink)

    worker = HTTPWorkerApp(handler)
    registry = WorkerRegistry()
    registry.publish(
        "echo-service",
        (WorkerTarget("worker", "http://worker", ("local",)),),
    )
    caller_runtime = LocalRuntime()
    async with HTTPWorkerClient(
        registry, transport=httpx.ASGITransport(app=worker.app)
    ) as client:
        caller_runtime.register_remote_target(
            "echo-service",
            HTTPRemoteModuleTarget(
                client,
                "echo-service",
                worker_bound.plan.plan_id,
                worker_bound.plan.graph_hash,
                ("local",),
            ),
        )
        handle = await caller_runtime.bind(Caller()).start(
            UserMessage(content="remote"),
            Context(),
            execution=ExecutionOptions(idempotency_key="caller-operation"),
        )
        output, context = await handle.result()
        async with handle.subscribe() as subscription:
            parent_events = [event async for event in subscription]

    assert output.content == "REMOTE"
    assert [message.content for message in context.messages] == ["remote", "REMOTE"]
    assert captured[0].trace_id == handle.trace_id
    assert captured[0].parent_execution_id == handle.execution_id
    assert captured[0].attempt == 1
    assert captured[0].plan_id == worker_bound.plan.plan_id
    assert captured[0].graph_hash == worker_bound.plan.graph_hash
    assert captured[0].idempotency_key.startswith("caller-operation:remote:")
    remote_events = [
        event for event in parent_events if event.data.get("remote") is True
    ]
    assert [event.kind for event in remote_events] == [
        "span.started",
        "span.completed",
    ]
    assert {event.execution_id for event in remote_events} == {handle.execution_id}
    assert {event.span_id for event in remote_events} == {captured[0].parent_span_id}
    worker_events = next(iter(worker.executions.values())).events
    assert {event["trace_id"] for event in worker_events} == {handle.trace_id}
    assert {event["parent_span_id"] for event in worker_events} == {
        captured[0].parent_span_id
    }
    await caller_runtime.close()
    await worker_runtime.close()


@pytest.mark.asyncio
async def test_agent_context_crosses_http_worker_without_field_loss():
    @dataclass(frozen=True, slots=True)
    class RemoteContext(Context):
        context_schema = "tests.http-worker-context"
        context_schema_version = 1
        tenant: str = ""
        history: tuple[str, ...] = ()

    class Echo(Agent[UserMessage, AIMessage]):
        context_type = RemoteContext

        async def forward(self, message, context):
            return AIMessage(content=message.content.upper()), context + message

    worker_runtime = _portable_runtime()
    worker_bound = worker_runtime.bind(Echo())

    def resolver(artifact):
        manifest = _artifact_resolver(artifact)
        return WorkerDeploymentManifest(
            artifact=manifest.artifact,
            verified_digest=manifest.verified_digest,
            entrypoint=manifest.entrypoint,
            input_schema=manifest.input_schema,
            output_schema=manifest.output_schema,
            serializer=manifest.serializer,
            context_codecs=worker_bound.plan.context_codecs,
        )

    worker = HTTPWorkerApp(
        bound_module_worker_handler(
            {"agent-context": worker_bound}, artifact_resolver=resolver
        )
    )
    registry = WorkerRegistry()
    registry.publish(
        "agent-context", (WorkerTarget("worker", "http://worker", ("local",)),)
    )
    async with HTTPWorkerClient(
        registry, transport=httpx.ASGITransport(app=worker.app)
    ) as client:
        remote = HTTPRemoteModuleTarget(
            client,
            "agent-context",
            worker_bound.plan.plan_id,
            worker_bound.plan.graph_hash,
            ("local",),
        )
        output, context = await remote.invoke(
            UserMessage(content="remote"),
            RemoteContext(tenant="acme", history=("created",)),
            deadline=None,
        )

    assert output.content == "REMOTE"
    assert type(context) is RemoteContext
    assert context.tenant == "acme"
    assert context.history == ("created",)
    assert [message.content for message in context.messages] == ["remote"]


@pytest.mark.asyncio
async def test_remote_child_wait_hands_off_parent_runnable_lease():
    entered, release = asyncio.Event(), asyncio.Event()

    class Slow(Module[UserMessage, AIMessage]):
        async def forward(self, message, context):
            entered.set()
            await release.wait()
            return AIMessage(content="remote"), context

    class LocalEcho(Module[UserMessage, AIMessage]):
        async def forward(self, message, context):
            return AIMessage(content="local"), context

    worker_runtime = _portable_runtime()
    worker_bound = worker_runtime.bind(Slow())

    class Caller(Module[UserMessage, AIMessage]):
        def __init__(self):
            super().__init__()
            self.remote = RemoteModule[UserMessage, AIMessage](
                binding_ref="slow",
                plan_id=worker_bound.plan.plan_id,
                graph_hash=worker_bound.plan.graph_hash,
                required_capabilities=("local",),
            )

        async def forward(self, message, context):
            return await self.remote(message, context)

    worker = HTTPWorkerApp(_bound_worker_handler({"slow": worker_bound}))
    registry = WorkerRegistry()
    registry.publish("slow", (WorkerTarget("worker", "http://worker", ("local",)),))
    runtime = LocalRuntime()
    binding = runtime.create_binding(
        name="single-runnable",
        execution_capacity=ExecutionCapacityPolicy(
            scope=CapacityScope.RUNTIME_INSTANCE,
            max_live_executions=2,
            max_runnable_executions=1,
            max_queue_size=1,
            max_waiters=1,
            max_child_depth=4,
            max_children_per_execution=8,
        ),
        model_capacity=CapacityPolicy.passthrough(),
        tool_capacity=CapacityPolicy.passthrough(),
    )
    async with HTTPWorkerClient(
        registry, transport=httpx.ASGITransport(app=worker.app)
    ) as client:
        runtime.register_remote_target(
            "slow",
            HTTPRemoteModuleTarget(
                client,
                "slow",
                worker_bound.plan.plan_id,
                worker_bound.plan.graph_hash,
                ("local",),
            ),
        )
        remote = binding.bind(Caller())
        local = binding.bind(LocalEcho())
        handle = await remote.start(UserMessage(content="x"), Context())
        await entered.wait()
        assert (await handle.snapshot()).phase is ExecutionPhase.WAITING_CHILD
        assert (await local.invoke(UserMessage(content="y"), Context()))[
            0
        ].content == "local"
        release.set()
        assert (await handle.result())[0].content == "remote"

    await runtime.close()
    await worker_runtime.close()


def test_bound_worker_rejects_nonportable_execution_plan():
    class Echo(Module[UserMessage, AIMessage]):
        async def forward(self, message, context):
            return AIMessage(content=message.content), context

    runtime = LocalRuntime()
    bound = runtime.bind(Echo())
    with pytest.raises(WorkerProtocolError, match="portable ExecutionPlan"):
        bound_module_worker_handler({"echo": bound})


def test_bound_worker_rejects_unverified_artifact_manifest():
    class Echo(Module[UserMessage, AIMessage]):
        async def forward(self, message, context):
            return AIMessage(content=message.content), context

    runtime = _portable_runtime()
    bound = runtime.bind(Echo())

    def unverified(artifact):
        manifest = _artifact_resolver(artifact)
        return WorkerDeploymentManifest(
            artifact=manifest.artifact,
            verified_digest="sha256:not-the-loaded-artifact",
            entrypoint=manifest.entrypoint,
            input_schema=manifest.input_schema,
            output_schema=manifest.output_schema,
            serializer=manifest.serializer,
            context_codecs=manifest.context_codecs,
        )

    with pytest.raises(WorkerProtocolError, match="does not verify"):
        bound_module_worker_handler({"echo": bound}, artifact_resolver=unverified)


def test_bound_worker_rejects_manifest_without_context_codecs():
    class Echo(Module[UserMessage, AIMessage]):
        async def forward(self, message, context):
            return AIMessage(content=message.content), context

    bound = _portable_runtime().bind(Echo())

    def missing_codecs(artifact):
        manifest = _artifact_resolver(artifact)
        return WorkerDeploymentManifest(
            artifact=manifest.artifact,
            verified_digest=manifest.verified_digest,
            entrypoint=manifest.entrypoint,
            input_schema=manifest.input_schema,
            output_schema=manifest.output_schema,
            serializer=manifest.serializer,
        )

    with pytest.raises(WorkerProtocolError, match="does not verify"):
        bound_module_worker_handler({"echo": bound}, artifact_resolver=missing_codecs)


@pytest.mark.asyncio
async def test_domain_message_and_context_history_cross_http_worker_losslessly():
    class Handoff(Module[Message, Message]):
        async def forward(self, message, context):
            assert message.kind == "handoff.requested"
            output = Message(
                kind="handoff.completed",
                data={
                    "operation_id": message.data["operation_id"],
                    "worker": "review-service",
                },
                slot="handoff/current",
            )
            return output, context + message + output

    worker_runtime = _portable_runtime()
    worker_bound = worker_runtime.bind(Handoff())

    class Caller(Module[Message, Message]):
        def __init__(self):
            super().__init__()
            self.remote = RemoteModule[Message, Message](
                binding_ref="handoff-service",
                plan_id=worker_bound.plan.plan_id,
                graph_hash=worker_bound.plan.graph_hash,
                required_capabilities=("local",),
            )

        async def forward(self, message, context):
            return await self.remote(message, context)

    worker = HTTPWorkerApp(_bound_worker_handler({"handoff-service": worker_bound}))
    registry = WorkerRegistry()
    registry.publish(
        "handoff-service",
        (WorkerTarget("worker", "http://worker", ("local",)),),
    )
    caller_runtime = LocalRuntime()
    request = Message(
        kind="handoff.requested",
        data={"operation_id": "op-42", "reviewers": ["alice", "bob"]},
    )
    prior = Message(
        kind="approval.approved",
        data={"operation_id": "op-42"},
        slot="approval/current",
    )
    initial_context = Context(messages=(prior,), metadata={"tenant": "tenant-1"})

    async with HTTPWorkerClient(
        registry, transport=httpx.ASGITransport(app=worker.app)
    ) as client:
        caller_runtime.register_remote_target(
            "handoff-service",
            HTTPRemoteModuleTarget(
                client,
                "handoff-service",
                worker_bound.plan.plan_id,
                worker_bound.plan.graph_hash,
                ("local",),
            ),
        )
        output, context = await caller_runtime.bind(Caller()).invoke(
            request, initial_context
        )

    assert output == Message(
        kind="handoff.completed",
        data={"operation_id": "op-42", "worker": "review-service"},
        slot="handoff/current",
    )
    assert context.metadata == initial_context.metadata
    assert context.messages == (prior, request, output)
    await caller_runtime.close()
    await worker_runtime.close()


@pytest.mark.asyncio
async def test_bound_worker_fails_closed_on_execution_plan_mismatch():
    class Echo(Module[UserMessage, AIMessage]):
        async def forward(self, message, context):
            return AIMessage(content=message.content), context

    runtime = _portable_runtime()
    bound = runtime.bind(Echo())
    worker = HTTPWorkerApp(_bound_worker_handler({"echo": bound}))
    registry = WorkerRegistry()
    target = WorkerTarget("worker", "http://worker")
    registry.publish("echo", (target,))

    async with HTTPWorkerClient(
        registry, transport=httpx.ASGITransport(app=worker.app)
    ) as client:
        with pytest.raises(WorkerRemoteError, match="ExecutionPlan identity"):
            await client.invoke(
                "echo",
                {
                    "message": {
                        "role": "user",
                        "content": "hello",
                        "slot": None,
                        "kind": None,
                        "data": {},
                        "metadata": {},
                    },
                    "context": {"messages": [], "tools": [], "metadata": {}},
                },
                request_id="mismatch",
                plan_id=_PLAN_ID,
                graph_hash=_GRAPH_HASH,
            )

    assert bound.plan.plan_id != _PLAN_ID
    await runtime.close()


@pytest.mark.asyncio
async def test_registered_target_cannot_override_remote_declaration_identity():
    remote = RemoteModule[UserMessage, AIMessage](
        binding_ref="reviewer",
        plan_id=_PLAN_ID,
        graph_hash=_GRAPH_HASH,
        required_capabilities=("durable",),
    )

    class Caller(Module[UserMessage, AIMessage]):
        async def forward(self, message, context):
            return await remote(message, context)

    registry = WorkerRegistry()
    registry.publish(
        "reviewer",
        (WorkerTarget("worker", "http://worker", ("durable", "local")),),
    )
    runtime = LocalRuntime()
    async with HTTPWorkerClient(
        registry, transport=httpx.MockTransport(lambda _request: None)
    ) as client:
        runtime.register_remote_target(
            "reviewer",
            HTTPRemoteModuleTarget(
                client,
                "reviewer",
                _PLAN_ID,
                _GRAPH_HASH,
                ("local",),
            ),
        )
        with pytest.raises(WorkerProtocolError, match="differs.*declaration"):
            await runtime.bind(Caller()).invoke(UserMessage(content="x"), Context())

    await runtime.close()


@pytest.mark.asyncio
async def test_registered_target_cannot_override_remote_declaration_placement():
    remote = RemoteModule[UserMessage, AIMessage](
        binding_ref="reviewer",
        plan_id=_PLAN_ID,
        graph_hash=_GRAPH_HASH,
        placement=PlacementPolicy.pinned("worker"),
    )

    class Caller(Module[UserMessage, AIMessage]):
        async def forward(self, message, context):
            return await remote(message, context)

    registry = WorkerRegistry()
    registry.publish("reviewer", (WorkerTarget("worker", "http://worker"),))
    runtime = LocalRuntime()
    async with HTTPWorkerClient(
        registry, transport=httpx.MockTransport(lambda _request: None)
    ) as client:
        runtime.register_remote_target(
            "reviewer",
            HTTPRemoteModuleTarget(
                client,
                "reviewer",
                _PLAN_ID,
                _GRAPH_HASH,
                placement=PlacementPolicy.adaptive(),
            ),
        )
        with pytest.raises(WorkerProtocolError, match="placement differs"):
            await runtime.bind(Caller()).invoke(UserMessage(content="x"), Context())

    await runtime.close()


@pytest.mark.asyncio
async def test_remote_module_cancellation_reaches_worker_owner_task():
    entered = asyncio.Event()

    async def handler(invocation, event_sink):
        entered.set()
        await asyncio.Event().wait()

    worker = HTTPWorkerApp(handler)
    registry = WorkerRegistry()
    registry.publish("slow", (WorkerTarget("worker", "http://worker"),))
    async with HTTPWorkerClient(
        registry, transport=httpx.ASGITransport(app=worker.app)
    ) as client:
        remote = HTTPRemoteModuleTarget(client, "slow", _PLAN_ID, _GRAPH_HASH)
        invocation = asyncio.create_task(
            remote.invoke_remote_child(
                UserMessage(content="cancel"),
                Context(),
                plan_id=_PLAN_ID,
                graph_hash=_GRAPH_HASH,
                required_capabilities=(),
                placement_mode="adaptive",
                pinned_target_id=None,
                deadline=None,
                trace_id="trace-cancel",
                parent_execution_id="parent-cancel",
                parent_span_id="span-cancel",
                event_sink=lambda _event: asyncio.sleep(0),
                child_execution_id="child-cancel",
                attempt=1,
                idempotency_key="cancel-operation",
            )
        )
        await entered.wait()
        invocation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await invocation
        await asyncio.sleep(0)
        owner = next(iter(worker.executions.values())).task
        relay_tasks = [
            task
            for task in asyncio.all_tasks()
            if task.get_name() == "pygent-remote-event-relay" and not task.done()
        ]

    assert owner is not None and owner.cancelled()
    assert relay_tasks == []


@pytest.mark.asyncio
async def test_durable_worker_rebuild_restores_terminal_poll_and_sse(tmp_path):
    path = tmp_path / "worker.sqlite3"

    async def handler(invocation, event_sink):
        return {"value": invocation.input["value"]}

    async with SQLiteHistoryStore(path) as history:
        first = HTTPWorkerApp(handler, history=history)
        registry = WorkerRegistry()
        target = WorkerTarget("worker", "http://worker", ("durable", "sse"))
        registry.publish("service", (target,))
        async with HTTPWorkerClient(
            registry, transport=httpx.ASGITransport(app=first.app)
        ) as client:
            execution_id, result = await client.invoke(
                "service",
                {"value": 7},
                request_id="request-1",
                plan_id=_PLAN_ID,
                graph_hash=_GRAPH_HASH,
                idempotency_key="operation-1",
                trace_id="root-1",
                parent_execution_id="parent-1",
                parent_span_id="span-parent-1",
                attempt=2,
            )
        assert result["value"] == 7

    async with SQLiteHistoryStore(path) as history:
        rebuilt = HTTPWorkerApp(handler, history=history)
        await rebuilt.recover()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=rebuilt.app),
            base_url="http://worker",
        ) as client:
            restored = await client.get(f"/v1/executions/{execution_id}")
            events = await client.get(f"/v1/executions/{execution_id}/events?after=-1")

    assert restored.json()["result"] == {"value": 7}
    assert "execution.started" in events.text
    assert "execution.completed" in events.text
    assert '"trace_id":"root-1"' in events.text
    assert '"parent_span_id":"span-parent-1"' in events.text


@pytest.mark.asyncio
async def test_durable_worker_recover_restarts_running_job(tmp_path):
    async with SQLiteHistoryStore(tmp_path / "worker.sqlite3") as history:
        invocation = {
            "binding_ref": "service",
            "request_id": "request-1",
            "input": {"value": 9},
            "plan_id": _PLAN_ID,
            "graph_hash": _GRAPH_HASH,
            "idempotency_key": "operation-1",
            "trace_id": "root-1",
            "parent_execution_id": "parent-1",
            "parent_span_id": "span-parent-1",
            "attempt": 1,
            "expires_at": None,
        }
        await history.put_task(
            task_id="job-1",
            kind="job",
            status="running",
            request=invocation,
        )
        await history.append_event(
            execution_id="job-1",
            index=0,
            event={
                "schema_version": "0.2",
                "event_id": "event-0",
                "execution_id": "job-1",
                "trace_id": "root-1",
                "span_id": "span-job-1",
                "parent_span_id": "span-parent-1",
                "sequence": 0,
                "timestamp_unix_ns": 1,
                "module_path": "worker:service",
                "kind": "execution.started",
                "data": {},
            },
        )

        async def handler(worker_invocation, event_sink):
            return {"value": worker_invocation.input["value"]}

        rebuilt = HTTPWorkerApp(handler, history=history)
        await rebuilt.recover()
        registry = WorkerRegistry()
        target = WorkerTarget("worker", "http://worker", ("durable",))
        registry.publish("service", (target,))
        async with HTTPWorkerClient(
            registry, transport=httpx.ASGITransport(app=rebuilt.app)
        ) as client:
            result = await client.result(
                RemoteExecutionHandle(client, "job-1", target), poll_interval=0
            )
            events = [event async for event in client.events(target, "job-1", after=0)]

    assert result["value"] == 9
    assert [event.kind for event in events] == [
        "execution.recovered",
        "execution.completed",
    ]


@pytest.mark.asyncio
async def test_poll_failure_fails_closed_without_shared_durable_claim(tmp_path):
    first_entered = asyncio.Event()
    captured = []
    async with SQLiteHistoryStore(tmp_path / "worker.sqlite3") as history:

        async def first_handler(invocation, event_sink):
            captured.append(invocation)
            first_entered.set()
            await asyncio.Event().wait()

        async def second_handler(invocation, event_sink):
            captured.append(invocation)
            return {"owner": "second", "value": invocation.input["value"]}

        first = HTTPWorkerApp(first_handler, history=history)
        second = HTTPWorkerApp(second_handler, history=history)
        first_transport = httpx.ASGITransport(app=first.app)
        second_transport = httpx.ASGITransport(app=second.app)

        async def route(request: httpx.Request) -> httpx.Response:
            if (
                request.url.host == "first"
                and request.method == "GET"
                and request.url.path != "/health"
            ):
                return httpx.Response(503, json={"error": "node_lost"})
            transport = (
                first_transport if request.url.host == "first" else second_transport
            )
            return await transport.handle_async_request(request)

        registry = WorkerRegistry()
        registry.publish(
            "service",
            (
                WorkerTarget("first", "http://first", ("durable",)),
                WorkerTarget("second", "http://second", ("durable",)),
            ),
        )
        async with HTTPWorkerClient(
            registry, transport=httpx.MockTransport(route)
        ) as client:
            ref = await client.start(
                "service",
                {"value": 3},
                request_id="request-1",
                plan_id=_PLAN_ID,
                graph_hash=_GRAPH_HASH,
                idempotency_key="operation-1",
                trace_id="root-1",
                parent_execution_id="parent-1",
                attempt=4,
            )
            await first_entered.wait()
            with pytest.raises(WorkerOutcomeUnknownError, match="outcome is unknown"):
                await client.result(ref, poll_interval=0)

        first.history = None
        await first.close()
        await second.close()
        await first_transport.aclose()
        await second_transport.aclose()

    assert len(captured) == 1
    assert {item.request_id for item in captured} == {"request-1"}
    assert {item.idempotency_key for item in captured} == {"operation-1"}
    assert {item.trace_id for item in captured} == {"root-1"}
    assert {item.parent_execution_id for item in captured} == {"parent-1"}
    assert {item.attempt for item in captured} == {4}
