from __future__ import annotations

import json

import httpx
import pytest

from examples.live_agent import benchmark as benchmark_module
from examples.live_agent.agent import (
    INVALID_ROUTE_ID,
    VALID_ROUTE_ID,
    LiveAgentConfig,
    benchmark_context,
    benchmark_message,
    build_live_agent,
    build_live_resources,
)
from examples.live_agent.benchmark import execute_benchmark
from pygent import Context
from pygent.runtime import ExecutionEvent


def test_live_config_is_validated_and_never_repr_leaks_secrets():
    secret = "unit-test-secret"
    config = LiveAgentConfig.from_environment(
        {
            "GLM_API_BASE": "https://models.example/v1",
            "GLM_API_KEY": secret,
            "GLM_MODEL_NAME": "unit-model",
        }
    )

    assert config.safe_summary() == {
        "api_configured": True,
        "model_configured": True,
        "fallback_configured": True,
    }
    assert secret not in repr(config)
    assert "models.example" not in repr(config)
    assert "unit-model" not in repr(config)

    with pytest.raises(RuntimeError, match="GLM_API_KEY"):
        LiveAgentConfig.from_environment(
            {
                "GLM_API_BASE": "https://models.example/v1",
                "GLM_MODEL_NAME": "model",
            }
        )
    with pytest.raises(ValueError, match="API root"):
        LiveAgentConfig(
            "https://models.example/v1/chat/completions", secret, "model"
        )


@pytest.mark.asyncio
async def test_mock_transport_fallback_tools_context_isolation_and_metrics():
    valid_key = "unit-test-valid-key"
    calls: list[tuple[str, str]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        authorization = request.headers.get("authorization", "")
        body = json.loads(request.content)
        last_role = body["messages"][-1]["role"]
        calls.append((authorization, last_role))
        if authorization != f"Bearer {valid_key}":
            return httpx.Response(401, json={"error": {"message": "unauthorized"}})
        if last_role == "user":
            user_text = body["messages"][-1]["content"]
            index = int(user_text.split("a=", 1)[1].split(" ", 1)[0])
            return httpx.Response(
                200,
                json={
                    "id": f"tool-{index}",
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": f"call-{index}",
                                        "type": "function",
                                        "function": {
                                            "name": "benchmark_add",
                                            "arguments": {"a": index, "b": 7},
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 1,
                        "total_tokens": 4,
                    },
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "final",
                "choices": [{"message": {"content": "done"}}],
                "usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 1,
                    "total_tokens": 5,
                },
            },
        )

    config = LiveAgentConfig(
        "https://models.example/v1", valid_key, "unit-model"
    )
    resources = build_live_resources(config, transport=httpx.MockTransport(handler))
    report = await execute_benchmark(
        resources,
        requests=4,
        concurrency=4,
        model_concurrency=2,
        deadline_seconds=5.0,
    )

    assert report["requested"] == 4
    assert report["succeeded"] == 4, report
    assert report["failed"] == 0
    assert report["fallback_count"] == 4
    assert report["fallback_rate"] == 1.0
    assert report["attempt_failures_by_kind"] == {"authentication": 8}
    assert report["tool_statuses"] == {"succeeded": 4}
    assert report["context_isolation_failures"] == 0
    assert report["usage"] == {
        "completion_tokens": 8,
        "prompt_tokens": 28,
        "total_tokens": 36,
    }
    assert 1 <= report["peak_model_inflight"] <= 2
    assert report["provider_calls"] == 16
    assert report["throughput_rps"] > 0
    assert report["latency_ms"]["p99"] >= report["latency_ms"]["p50"]

    assert sum(route != f"Bearer {valid_key}" for route, _ in calls) == 8
    assert sum(route == f"Bearer {valid_key}" for route, _ in calls) == 8
    serialized = json.dumps(report, sort_keys=True)
    assert valid_key not in serialized
    assert "models.example" not in serialized
    assert "unit-model" not in serialized
    assert INVALID_ROUTE_ID not in serialized
    assert VALID_ROUTE_ID not in serialized

    direct_resources = build_live_resources(
        config, transport=httpx.MockTransport(handler)
    )
    direct_agent, direct_tool = build_live_agent(
        "unit-model",
        model_invoker=direct_resources.invoker,
        executor_registry=direct_resources.registry,
    )
    invoked = await direct_agent.invoke(
        benchmark_message(10), benchmark_context("direct-invoke", direct_tool)
    )
    async with direct_agent.stream(
        benchmark_message(11), benchmark_context("direct-stream", direct_tool)
    ) as stream:
        stream_events = [event async for event in stream]
        streamed = await stream.final_result()

    assert invoked[0].content == "done"
    assert streamed[0].content == "done"
    assert len(invoked[1].messages) == len(streamed[1].messages) == 4
    assert [item.role for item in invoked[1].messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert "model.attempt.succeeded" in [event.kind for event in stream_events]
    assert "model.usage" in [event.kind for event in stream_events]
    assert stream_events[-1].kind == "execution.completed"
    await direct_resources.aclose()


def test_failed_sample_context_identity_is_actually_checked():
    assert not benchmark_module._failure_context_isolated(
        Context(metadata={"benchmark_request_id": "other"}),
        "expected",
        (),
        None,
    )

    request_id = "expected"
    context = Context(metadata={"benchmark_request_id": request_id})
    events = (
        ExecutionEvent(
            schema_version="0.2",
            event_id="event-0",
            execution_id="root",
            attempt_id="attempt-1",
            trace_id="trace-root",
            span_id="span-root",
            parent_span_id=None,
            module_path="agent",
            sequence=0,
            timestamp_unix_ns=1,
            kind="execution.started",
            data={"request_id": request_id},
        ),
        ExecutionEvent(
            schema_version="0.2",
            event_id="event-1",
            execution_id="root",
            attempt_id="attempt-1",
            trace_id="trace-root",
            span_id="span-child",
            parent_span_id="span-root",
            module_path="agent.model",
            sequence=1,
            timestamp_unix_ns=2,
            kind="model.attempt.failed",
            data={"error_kind": "unavailable"},
        ),
    )
    assert benchmark_module._failure_context_isolated(
        context, request_id, events, "root"
    )
    unrelated = events + (
        ExecutionEvent(
            schema_version="0.2",
            event_id="event-2",
            execution_id="foreign",
            attempt_id="attempt-2",
            trace_id="trace-foreign",
            span_id="span-foreign",
            parent_span_id=None,
            module_path="other",
            sequence=2,
            timestamp_unix_ns=3,
            kind="execution.failed",
            data={},
        ),
    )
    assert not benchmark_module._failure_context_isolated(
        context, request_id, unrelated, "root"
    )


@pytest.mark.asyncio
async def test_live_wrapper_closes_resources_when_setup_fails(monkeypatch):
    class Resources:
        closed = 0

        async def aclose(self):
            self.closed += 1

    resources = Resources()
    config = LiveAgentConfig(
        "https://models.example/v1", "unit-key", "unit-model"
    )
    monkeypatch.setattr(
        benchmark_module, "build_live_resources", lambda _config: resources
    )

    async def fail(*args, **kwargs):
        raise RuntimeError("setup failed")

    monkeypatch.setattr(benchmark_module, "execute_benchmark", fail)
    with pytest.raises(RuntimeError, match="setup failed"):
        await benchmark_module.run_live_benchmark(
            config,
            requests=1,
            concurrency=1,
            model_concurrency=1,
            deadline_seconds=1.0,
        )
    assert resources.closed == 1
