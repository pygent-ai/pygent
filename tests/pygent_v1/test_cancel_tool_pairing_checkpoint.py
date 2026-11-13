"""Execution-level cancel must leave a paired tool message in the transcript.

Regression for the 0.6.0 defect: when ``handle.cancel()`` (or a deadline kill)
interrupted a running tool batch, ``ToolCallLayer`` rebuilt the paired
``ToolResult`` values into the journal (``tool.batch.cancelled``) only. The
conversation checkpoint module never received a ``ToolMessage``, so session
history kept the assistant ``tool_calls`` dangling, and every later turn
replayed the dangling tool_use to the provider -- whose strict
tool_use/tool_result validation answers HTTP 400, permanently wedging the
session.

The regression scenario is end to end: real ``LocalRuntime`` + real
``DurableToolTaskManager`` + the real pygent_v1 graph
(``ConversationCheckpointToolModule`` included), scripted model, session
history asserted directly, and a strict fake adapter that rejects any
unpaired tool_use with HTTP 400 for the follow-up turn.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pygent.agent.pygent_v1.graph.pipeline import (
    ConversationCheckpointToolModule,
    checkpoint_conversation_message,
)
from pygent.agent.pygent_v1.orchestration.execution_host import LocalExecutionHost
from pygent.agent.pygent_v1.runtime.context import (
    PYGENT_CONTEXT_CODEC,
    PygentContext,
)
from pygent.agent.pygent_v1.schema import build_run_config
from pygent.agent.pygent_v1.sessions import SessionManager
from pygent.core import (
    AIMessage,
    Context,
    ExecutionStatus,
    Message,
    Module,
    UserMessage,
)
from pygent.runtime import (
    CapacityPolicy,
    CapacityScope,
    ExecutionCapacityPolicy,
    ExecutionOptions,
    LocalRuntime,
)
from pygent.runtime.tasks import DurableToolTaskManager
from pygent.tool import (
    ExecutorRegistry,
    LocalToolExecutor,
    ToolAuthorizationDecision,
    ToolCall,
    ToolCallLayer,
    ToolDefinition,
    ToolSideEffect,
    ToolSpec,
)
from pygent.tool.executors import SandboxExecutorSupport

_BASH_TOOL_ID = "standard.shell.bash"
_BASH_VERSION = "3.1.0"

_SLEEPS_EXECUTION = {"command": "sleep 30"}


def _model_mapping(base_url: str) -> dict[str, Any]:
    return {
        "connections": {
            "facade": {
                "provider": "cancel-pairing-scripted",
                "credential": {"none": True},
                "protocols": {
                    "openai_chat_completions": {"base_url": base_url}
                },
            }
        },
        "models": {
            "scripted": {
                "connection": "facade",
                "model_id": "scripted",
                "protocol": "openai_chat_completions",
                "capabilities": {
                    "modalities": {"input": ["text"], "output": ["text"]},
                    "streaming": {"output": []},
                    "tools": {
                        "call": True,
                        "choice": ["auto", "named", "none", "required"],
                        "parallel": False,
                    },
                    "structured_output": {
                        "json_object": False,
                        "json_schema": False,
                    },
                    "reasoning": {"supported": False, "controllable": False},
                    "limits": {
                        "context_tokens": 100_000,
                        "max_output_tokens": 4_096,
                    },
                },
            }
        },
        "model_groups": {"default": {"models": ["scripted"]}},
    }


def _text_response(text: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-text",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "scripted",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": text},
            }
        ],
        "usage": {
            "prompt_tokens": 16,
            "completion_tokens": 8,
            "total_tokens": 24,
        },
    }


def _bash_call_response(call_id: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{call_id}",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "scripted",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": json.dumps(_SLEEPS_EXECUTION),
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {
            "prompt_tokens": 16,
            "completion_tokens": 16,
            "total_tokens": 32,
        },
    }


def _pairing_violation(messages: list[dict[str, Any]]) -> str | None:
    """Strict provider rule: every assistant tool_use needs a tool reply."""

    pending: dict[str, str] = {}
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            for call in message.get("tool_calls") or ():
                call_id = call.get("id") if isinstance(call, dict) else None
                if isinstance(call_id, str) and call_id:
                    pending[call_id] = str(call.get("function", {}).get("name"))
        elif role == "tool":
            pending.pop(str(message.get("tool_call_id")), None)
    if pending:
        return ", ".join(sorted(pending))
    return None


async def _read_http_request(
    reader: asyncio.StreamReader,
) -> tuple[str, str, bytes] | None:
    request_line = await reader.readline()
    if not request_line:
        return None
    method, path, _version = request_line.decode("latin-1").split(" ", 2)
    content_length = 0
    while True:
        header = await reader.readline()
        if header in (b"\r\n", b"\n", b""):
            break
        name, _, value = header.decode("latin-1").partition(":")
        if name.strip().lower() == "content-length":
            content_length = int(value.strip())
    body = await reader.readexactly(content_length) if content_length else b""
    return method, path, body


class _ScriptedModelServer:
    """In-process OpenAI-compatible endpoint with a scripted FIFO.

    Entries are consumed in request arrival order. An entry with
    ``strict_pairing`` first validates that the request history pairs every
    assistant tool_call with a tool reply and answers HTTP 400 otherwise --
    standing in for the provider that wedged the reported incident.
    """

    def __init__(self, script: list[dict[str, Any]]) -> None:
        self.script = script
        self._cursor = 0
        self.received: asyncio.Queue[int] = asyncio.Queue()
        self._server: asyncio.Server | None = None
        self.url = ""

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._serve_connection, "127.0.0.1", 0
        )
        port = self._server.sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}/v1"

    async def aclose(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def wait_received(self, count: int, timeout: float = 15.0) -> list[int]:
        return [
            await asyncio.wait_for(self.received.get(), timeout)
            for _ in range(count)
        ]

    async def _serve_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            while True:
                request = await _read_http_request(reader)
                if request is None:
                    return
                method, path, body = request
                status, payload = self._serve(method, path, body)
                data = json.dumps(payload).encode("utf-8")
                writer.write(
                    b"HTTP/1.1 "
                    + str(status).encode()
                    + b" OK\r\nContent-Type: application/json\r\nContent-Length: "
                    + str(len(data)).encode()
                    + b"\r\nConnection: keep-alive\r\n\r\n"
                    + data
                )
                await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            return
        finally:
            writer.close()

    def _serve(
        self, method: str, path: str, body: bytes
    ) -> tuple[int, dict[str, Any]]:
        if method == "POST" and path.endswith("/chat/completions"):
            index = min(self._cursor, len(self.script) - 1)
            self._cursor += 1
            self.received.put_nowait(index)
            entry = self.script[index]
            if entry.get("strict_pairing"):
                request = json.loads(body.decode("utf-8"))
                dangling = _pairing_violation(request.get("messages") or [])
                if dangling is not None:
                    return 400, {
                        "error": {
                            "message": (
                                "unpaired tool_use without tool_result: "
                                f"[{dangling}]"
                            ),
                            "type": "invalid_request_error",
                        }
                    }
            return 200, entry["response"]
        return 404, {"error": {"message": f"unexpected {method} {path}"}}


async def _harness(tmp_path: Path, script: list[dict[str, Any]]) -> SimpleNamespace:
    server = _ScriptedModelServer(script)
    await server.start()
    config = build_run_config(
        workspace_root=tmp_path / "workspace",
        model_config_mapping=_model_mapping(server.url),
        default_model_group="default",
        user_pygent_root=tmp_path / "home" / ".pygent_v1",
        system_prompt="You are a coding assistant.",
    )
    config.runtime_approvals.enabled = False
    manager = SessionManager(config)
    session = manager.create(
        "chat",
        mode="chat",
        model_group_name="default",
        selected_model_key="scripted",
    )
    return SimpleNamespace(
        server=server,
        config=config,
        manager=manager,
        session_id=session.session_id,
        session_dir=session.session_dir,
        host=LocalExecutionHost(credential_environ={}),
    )


async def _wait_for(
    predicate: Any, timeout: float = 15.0, interval: float = 0.05
) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if asyncio.iscoroutine(result):
            result = await result
        if result:
            return result
        await asyncio.sleep(interval)
    raise AssertionError("condition not met before timeout")


def _tail(message: Any) -> dict[str, Any] | None:
    history = message.manager.load(message.session_id).history
    return history[-1] if history else None


async def _deliver(host: Any, harness: SimpleNamespace, message: str) -> Any:
    delivery = await host.deliver_user_message(
        config=harness.config,
        manager=harness.manager,
        message=message,
        session_id=harness.session_id,
        interactive_approvals=False,
    )
    assert delivery.mode == "turn" and delivery.turn is not None
    return delivery.turn


def _assert_paired_tail(harness: SimpleNamespace, expected_status: str) -> str:
    """The transcript tail must be assistant(tool_calls) -> tool(paired)."""

    history = harness.manager.load(harness.session_id).history
    tail = history[-1]
    assert tail["role"] == "tool", (
        f"transcript tail is not a tool message (dangling tool_calls): "
        f"{history[-2:]}"
    )
    assistant = history[-2]
    assert assistant["role"] == "assistant", assistant
    call_ids = [call["call_id"] for call in assistant["tool_calls"]]
    assert call_ids, assistant
    results = tail["results"]
    assert [result["call_id"] for result in results] == call_ids, results
    assert results[0]["status"] == expected_status, results[0]
    assert (
        results[0]["error_code"] == "interrupted_by_execution_cancel"
    ), results[0]
    return call_ids[0]


def _replace_bash_executor(harness: SimpleNamespace, handler: Any) -> None:
    class _ScriptedBashExecutor(LocalToolExecutor):
        sandbox_support = SandboxExecutorSupport(profiles=("workspace",))

    service = harness.runtime_service
    service.executor_registry.register(
        _BASH_TOOL_ID,
        _BASH_VERSION,
        _ScriptedBashExecutor(handler),
        replace_existing=True,
    )


def _gate_task_launch(harness: SimpleNamespace, gate: asyncio.Event) -> None:
    """Park every durable launch after admission, before the job starts."""

    manager = harness.runtime_service.task_manager
    assert isinstance(manager, DurableToolTaskManager)
    original_start = manager.start

    async def gated_start(task_id: str) -> None:
        await gate.wait()
        await original_start(task_id)

    manager.start = gated_start  # type: ignore[method-assign]


async def _finish_turn(turn: Any, expected_status: str) -> None:
    await asyncio.wait_for(turn.result(), 60)
    assert turn.status == expected_status, turn.status


async def test_cancel_before_launch_leaves_paired_cancelled_tool_message(
    tmp_path: Path,
) -> None:
    """Round 2 cancelled with the job admitted but not launched.

    The transcript tail must pair the assistant tool_calls with a
    status=cancelled tool result, and the next turn must project a fully
    paired history that the strict fake adapter accepts.
    """

    harness = await _harness(
        tmp_path,
        script=[
            {"response": _text_response("第一轮完成")},
            {"response": _bash_call_response("call-round2")},
            {"response": _text_response("第三轮完成"), "strict_pairing": True},
        ],
    )
    launch_gate = asyncio.Event()
    try:
        lease = await harness.host.acquire_runtime(config=harness.config)
        harness.runtime_service = lease.runtime.runtime_service
        _replace_bash_executor(
            harness, lambda arguments: asyncio.sleep(30, "sleep done")
        )
        _gate_task_launch(harness, launch_gate)
        await lease.release()

        first = await _deliver(harness.host, harness, "第一轮")
        await _finish_turn(first, "passed")

        second = await _deliver(harness.host, harness, "第二轮")
        await second.wait_ready()
        await _wait_for(
            lambda: (
                (tail := _tail(harness)) is not None
                and tail.get("role") == "assistant"
                and bool(tail.get("tool_calls"))
            )
        )
        # The tool phase entered and the durable job is parked before launch.
        await asyncio.sleep(0.3)
        handle = second.execution_handle
        assert handle is not None
        assert await handle.cancel() is True
        outcome = await handle.outcome()
        assert outcome.status is ExecutionStatus.CANCELLED
        await _finish_turn(second, "skipped")

        _assert_paired_tail(harness, "cancelled")

        third = await _deliver(harness.host, harness, "第三轮")
        await _finish_turn(third, "passed")
    finally:
        await harness.host.aclose()
        await harness.server.aclose()


def _allow_decision(request: Any, context: Context) -> ToolAuthorizationDecision:
    return ToolAuthorizationDecision(
        call_id=request.call.call_id,
        allowed=True,
        reason_code="allowed",
        lifecycle="sync",
    )


class _DeadlineModel(Module[Message, AIMessage]):
    """Scripted model returning one tool-call answer."""

    trusted_live_resource_attributes = ("answers",)

    def __init__(self, answer: AIMessage) -> None:
        super().__init__()
        self.answers = (answer,)

    async def forward(
        self, message: Message, context: Context
    ) -> tuple[AIMessage, Context]:
        return self.answers[0], context


class _DeadlineTurnDriver(Module[UserMessage, AIMessage]):
    """Graph slice under test: model step, assistant checkpoint, tool step."""

    trusted_live_resource_attributes = ("config", "model", "tool_step")

    def __init__(
        self,
        config: Any,
        *,
        model: Module[Message, AIMessage],
        tool_step: ConversationCheckpointToolModule,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = model
        self.tool_step = tool_step

    async def forward(
        self, message: UserMessage, context: PygentContext
    ) -> tuple[AIMessage, PygentContext]:
        answer, model_context = await self.model(message, context)
        await checkpoint_conversation_message(
            self.config,
            model_context,
            answer,
            boundary="deadline-test-model",
        )
        _tool_message, tool_context = await self.tool_step(answer, model_context)
        return answer, tool_context


def _double_spec() -> ToolSpec:
    return ToolSpec(
        tool_id="math.double",
        version="1",
        definition=ToolDefinition(
            name="double",
            description="double an integer",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            output_schema={"type": "integer"},
        ),
        side_effect=ToolSideEffect.PURE,
    )


async def test_deadline_kill_leaves_paired_tool_message(tmp_path: Path) -> None:
    """A deadline kill unwinds as ``_ExecutionDeadlineExpired`` (TimeoutError).

    The conversation checkpoint module must treat it as the same
    execution-level cancel: the tool layer's rebuilt batch lands in the
    transcript paired, even though the exception that reaches the module is
    not a ``CancelledError``.
    """

    config = build_run_config(
        workspace_root=tmp_path / "workspace",
        model_config_mapping=_model_mapping("http://127.0.0.1:9/v1"),
        default_model_group="default",
        user_pygent_root=tmp_path / "home" / ".pygent_v1",
        system_prompt="You are a coding assistant.",
    )
    manager = SessionManager(config)
    session = manager.create(
        "chat",
        mode="chat",
        model_group_name="default",
        selected_model_key="scripted",
    )
    run_ref = manager.start_case_run(
        session.session_id, "chat", run_config=config
    )
    spec = _double_spec()
    registry = ExecutorRegistry()
    slow_entered = asyncio.Event()

    async def double(arguments: Mapping[str, Any]) -> int:
        if arguments["value"] == 1:
            return 2
        slow_entered.set()
        await asyncio.sleep(30)
        return 4  # pragma: no cover - cancelled before this

    registry.register(spec.tool_id, spec.version, LocalToolExecutor(double))
    tools = ConversationCheckpointToolModule(
        config,
        ToolCallLayer(
            tools=(spec,),
            executor_registry=registry,
            authorization_adapter=_allow_decision,
            max_concurrency=2,
        ),
    )
    driver = _DeadlineTurnDriver(
        config,
        model=_DeadlineModel(
            AIMessage(
                content="",
                tool_calls=(
                    ToolCall(call_id="fast", name="double", arguments={"value": 1}),
                    ToolCall(call_id="slow", name="double", arguments={"value": 2}),
                ),
            )
        ),
        tool_step=tools,
    )
    runtime = LocalRuntime(context_codecs=(PYGENT_CONTEXT_CODEC,))
    runtime.attach_executor_registry(registry)
    binding = runtime.create_binding(
        name="deadline-pairing-test",
        execution_capacity=ExecutionCapacityPolicy(
            scope=CapacityScope.RUNTIME_INSTANCE,
            max_live_executions=2,
            max_runnable_executions=2,
            max_queue_size=2,
            max_waiters=2,
            max_child_depth=8,
            max_children_per_execution=16,
        ),
        model_capacity=CapacityPolicy.passthrough(),
        tool_capacity=CapacityPolicy.passthrough(),
    )
    context = PygentContext(
        tools=(spec.definition,),
        metadata={"permissions": ["tool:use"]},
        session_id=run_ref.session_id,
        case_id="chat",
        case_run_id=run_ref.case_run_id,
        run_dir=run_ref.run_dir,
        turn_id="turn-1",
    )
    handle = await binding.bind(driver).start(
        UserMessage(content="start"),
        context,
        execution=ExecutionOptions(deadline=time.monotonic() + 0.5),
    )
    await slow_entered.wait()
    await asyncio.sleep(0.05)  # let the fast call finish deterministically
    outcome = await handle.outcome()
    assert outcome.status is ExecutionStatus.DEADLINE_EXCEEDED

    history = manager.load(session.session_id).history
    assistant = history[-2]
    tail = history[-1]
    assert assistant["role"] == "assistant" and assistant["tool_calls"]
    assert tail["role"] == "tool", (
        f"transcript tail is not a tool message (dangling tool_calls): "
        f"{history[-2:]}"
    )
    call_ids = [call["call_id"] for call in assistant["tool_calls"]]
    results = tail["results"]
    assert [result["call_id"] for result in results] == call_ids, results
    # The deadline race has two legitimate winners (see the journal-level
    # deadline test): the layer's cancelled mapping or the runner's timeout
    # envelope. Either way the transcript pair is complete.
    assert results[0]["status"] == "succeeded", results[0]
    assert results[1]["status"] in {"cancelled", "failed"}, results[1]
    await runtime.close()


async def test_cancel_mid_execution_leaves_paired_detached_tool_message(
    tmp_path: Path,
) -> None:
    """Round 2 cancelled while the bash job was already running.

    The started job keeps its identity, so the paired tool result reports
    detached with the task reference -- still a complete tool_use/tool_result
    pair for the provider, which the next turn proves.
    """

    bash_entered = asyncio.Event()

    async def sleeping_bash(arguments: Mapping[str, object]) -> str:
        bash_entered.set()
        await asyncio.sleep(30)
        return "sleep done"  # pragma: no cover - cancelled before this

    harness = await _harness(
        tmp_path,
        script=[
            {"response": _text_response("第一轮完成")},
            {"response": _bash_call_response("call-round2")},
            {"response": _text_response("第三轮完成"), "strict_pairing": True},
        ],
    )
    try:
        lease = await harness.host.acquire_runtime(config=harness.config)
        harness.runtime_service = lease.runtime.runtime_service
        _replace_bash_executor(harness, sleeping_bash)
        await lease.release()

        first = await _deliver(harness.host, harness, "第一轮")
        await _finish_turn(first, "passed")

        second = await _deliver(harness.host, harness, "第二轮")
        await second.wait_ready()
        await _wait_for(
            lambda: (
                (tail := _tail(harness)) is not None
                and tail.get("role") == "assistant"
                and bool(tail.get("tool_calls"))
            )
        )
        await asyncio.wait_for(bash_entered.wait(), 15)
        handle = second.execution_handle
        assert handle is not None
        assert await handle.cancel() is True
        outcome = await handle.outcome()
        assert outcome.status is ExecutionStatus.CANCELLED
        await _finish_turn(second, "skipped")

        call_id = _assert_paired_tail(harness, "detached")
        results = harness.manager.load(harness.session_id).history[-1]["results"]
        assert results[0]["task"] is not None, results[0]
        assert results[0]["task"]["call_id"] == call_id, results[0]

        third = await _deliver(harness.host, harness, "第三轮")
        await _finish_turn(third, "passed")
    finally:
        await harness.host.aclose()
        await harness.server.aclose()
