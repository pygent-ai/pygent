"""Full-chain realtime delivery tests against a scripted model endpoint.

Real ``LaraRuntimeService``, real agent graph, real fork reply, real
collaboration store and real session coordinator — the only stub is the
model wire: an in-process OpenAI-compatible endpoint serving scripted
completions. The chain under test is the one users experience: user input
rides the fork, lands as an exchange envelope, and must reach the main
agent's context through a continuation turn instead of stranding.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from collections.abc import Mapping
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pygent.lara.core.io import read_json
from pygent.lara.orchestration.execution_host import LocalExecutionHost
from pygent.lara.runtime.realtime.speak import SPEAK_TOOL_NAME
from pygent.lara.schema import build_run_config
from pygent.lara.schema.models import RealtimeInteractionConfig
from pygent.lara.sessions import SessionCollaborationStore
from pygent.lara.sessions.manager import SessionManager

_DRAFT = "重构完成,改动 A/B/C"
_FORK_REPLY = "收到,我看一下"
_ANSWER = "性能影响很小,已经做了缓存"


def _model_mapping(base_url: str) -> dict[str, Any]:
    return {
        "connections": {
            "facade": {
                "provider": "e2e-scripted",
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


def _speak_call_response(call_id: str, text: str) -> dict[str, Any]:
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
                            "id": f"call-{call_id}",
                            "type": "function",
                            "function": {
                                "name": SPEAK_TOOL_NAME,
                                "arguments": json.dumps({"text": text}),
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


def _tool_call_response(
    call_id: str, name: str, arguments: str
) -> dict[str, Any]:
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
                            "id": f"call-{call_id}",
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
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

    Entries are consumed in request arrival order; an entry with a gate holds
    its response until the test releases it, which is what makes "the fork
    finishes after the turn chain" reproducible deterministically.
    """

    def __init__(self, script: list[dict[str, Any]]) -> None:
        self.script = script
        self._cursor = 0
        self.received: asyncio.Queue[int] = asyncio.Queue()
        self.completed: asyncio.Queue[int] = asyncio.Queue()
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
                status, payload = await self._route(method, path, body)
                encoded = json.dumps(payload).encode("utf-8")
                writer.write(
                    (
                        f"HTTP/1.1 {status} OK\r\n"
                        "Content-Type: application/json\r\n"
                        f"Content-Length: {len(encoded)}\r\n"
                        "Connection: keep-alive\r\n\r\n"
                    ).encode("latin-1")
                    + encoded
                )
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionResetError):
            return
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionResetError, BrokenPipeError):
                pass

    async def _route(
        self, method: str, path: str, body: bytes
    ) -> tuple[int, dict[str, Any]]:
        if method == "GET" and path.endswith("/models"):
            return 200, {"object": "list", "data": []}
        if method == "POST" and path.endswith("/chat/completions"):
            index = self._cursor
            self._cursor += 1
            self.received.put_nowait(index)
            entry = (
                self.script[index]
                if index < len(self.script)
                else {"response": _text_response("done")}
            )
            gate = entry.get("gate")
            if isinstance(gate, asyncio.Event):
                await gate.wait()
            self.completed.put_nowait(index)
            return 200, entry["response"]
        return 404, {"error": {"message": f"unexpected {method} {path}"}}


async def _harness(tmp_path: Path, script: list[dict[str, Any]]) -> SimpleNamespace:
    server = _ScriptedModelServer(script)
    await server.start()
    config = build_run_config(
        workspace_root=tmp_path / "workspace",
        model_config_mapping=_model_mapping(server.url),
        default_model_group="default",
        user_lara_root=tmp_path / "home" / ".lara",
        system_prompt="You are a realtime coding assistant. The user sees only speak calls.",
    )
    config.realtime_interaction = RealtimeInteractionConfig(enabled=True)
    # The desktop preauthorizes speak for realtime sessions; the kernel ships
    # an empty preauthorization list by default.
    config.runtime_approvals.preauthorized_tools = ("speak",)
    manager = SessionManager(config)
    ref = manager.create(
        "chat",
        mode="lara-says",
        model_group_name="default",
        selected_model_key="scripted",
    )
    host = LocalExecutionHost(credential_environ={})
    store = SessionCollaborationStore(config.lara_root)
    return SimpleNamespace(
        server=server,
        config=config,
        manager=manager,
        session_id=ref.session_id,
        session_dir=ref.session_dir,
        host=host,
        store=store,
    )


async def _acquire_service(harness: SimpleNamespace) -> Any:
    lease = await harness.host.acquire_runtime(config=harness.config)
    service = lease.runtime.runtime_service
    await lease.release()
    return service


def _envelope_state(store_path: Path, submission_id: str) -> str:
    with closing(sqlite3.connect(store_path)) as connection:
        row = connection.execute(
            "SELECT state FROM agent_messages WHERE submission_id = ?",
            (submission_id,),
        ).fetchone()
    assert row is not None, f"no envelope {submission_id!r} in the store"
    return str(row[0])


def _speak_from_event(event: Any) -> str | None:
    def walk(node: Any) -> str | None:
        if isinstance(node, Mapping):
            if node.get("kind") == "lara.realtime.speak":
                data = node.get("data")
                if isinstance(data, Mapping) and "text" in data:
                    return str(data["text"])
            for value in node.values():
                found = walk(value)
                if found is not None:
                    return found
        elif isinstance(node, (list, tuple)):
            for value in node:
                found = walk(value)
                if found is not None:
                    return found
        return None

    return walk(event)


async def _speak_texts(service: Any, session_dir: str | Path) -> list[str]:
    """User-visible speech from the durable event journal, in emit order."""

    texts: list[str] = []
    for metadata_path in sorted(Path(session_dir).rglob("run_metadata.json")):
        execution_id = read_json(metadata_path, default={}).get(
            "runtime_execution_id"
        )
        if not execution_id:
            continue
        events = await service.history.events_after(
            execution_id=execution_id, after=-1, limit=4096
        )
        for event in events:
            text = _speak_from_event(event)
            if text is not None:
                texts.append(text)
    return texts


async def _answer_spoken(service: Any, session_dir: str | Path) -> bool:
    return _ANSWER in await _speak_texts(service, session_dir)


async def _eventually(predicate: Any, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if asyncio.iscoroutine(result):
            result = await result
        if result:
            return
        await asyncio.sleep(0.05)
    raise AssertionError("condition not met before timeout")


async def test_plain_turn_runs_to_completion(tmp_path: Path) -> None:
    """Control: the harness itself drives a plain realtime turn to completion."""

    harness = await _harness(
        tmp_path,
        script=[
            {"response": _speak_call_response("ack", "开始重构")},
            {"response": _text_response(_DRAFT)},
        ],
    )
    try:
        turn = await harness.host.turns.submit(
            config=harness.config,
            manager=harness.manager,
            message="帮我重构这个模块",
            session_id=harness.session_id,
            turn_id="turn-1",
            interactive_approvals=False,
            model_group_name="default",
            selected_model_key="scripted",
        )
        output, _ = await asyncio.wait_for(turn.result(), 30)
        assert turn.status == "passed"
        assert output is not None
        texts = await _speak_texts(
            await _acquire_service(harness), harness.session_dir
        )
        assert texts == ["开始重构"]
    finally:
        await harness.host.aclose()
        await harness.server.aclose()


async def test_fork_reply_in_idle_session_drains_immediately(tmp_path: Path) -> None:
    harness = await _harness(
        tmp_path,
        script=[
            {"response": _speak_call_response("fork", _FORK_REPLY)},
            {"response": _speak_call_response("cont", _ANSWER)},
            {"response": _text_response("done")},
        ],
    )
    try:
        service = await _acquire_service(harness)
        run_ref = harness.manager.start_case_run(harness.session_id, "chat")

        outcome = await service.send_interrupt(
            "execution-1",
            manager=harness.manager,
            run_ref=run_ref,
            session_id=harness.session_id,
            input_id="i1",
            message="顺便问下,影响性能吗?",
        )
        assert outcome.speak_args is not None
        assert outcome.speak_args.text == _FORK_REPLY

        # The delivery hook must have submitted the continuation turn: the
        # model round after the fork sees the envelope and answers through
        # speak, and the envelope ends delivered — nothing strands.
        await _eventually(
            lambda: _answer_spoken(service, harness.session_dir)
        )
        texts = await _speak_texts(service, harness.session_dir)
        assert texts == [_ANSWER]
        history = harness.manager.load(harness.session_id).history
        assert "顺便问下,影响性能吗?" in str(history)
        assert harness.store.pending_messages(harness.session_id) == []
    finally:
        await harness.host.aclose()
        await harness.server.aclose()


async def test_fork_reply_after_finalizer_chain_gets_delivered(
    tmp_path: Path,
) -> None:
    after_interrupt = asyncio.Event()
    fork_gate = asyncio.Event()
    harness = await _harness(
        tmp_path,
        script=[
            {"response": _speak_call_response("t1-ack", "开始重构")},
            {"response": _tool_call_response("t1-bash", "bash", '{"command": "echo hi"}')},
            # The draft is gated: while it is held, the test injects the
            # interrupt so the fork's model call is in flight when the turn
            # ends.
            {"response": _text_response(_DRAFT), "gate": after_interrupt},
            {"response": _speak_call_response("fork", _FORK_REPLY), "gate": fork_gate},
            {"response": _speak_call_response("final", _DRAFT)},
            {"response": _text_response("done")},
            {"response": _speak_call_response("cont", _ANSWER)},
            {"response": _text_response("done")},
        ],
    )
    try:
        service = await _acquire_service(harness)
        run_ref = harness.manager.start_case_run(harness.session_id, "chat")

        turn1 = await harness.host.turns.submit(
            config=harness.config,
            manager=harness.manager,
            message="帮我重构这个模块",
            session_id=harness.session_id,
            turn_id="turn-1",
            interactive_approvals=False,
            model_group_name="default",
            selected_model_key="scripted",
        )
        assert await harness.server.wait_received(3) == [0, 1, 2]

        interrupt = asyncio.create_task(
            service.send_interrupt(
                "execution-1",
                manager=harness.manager,
                run_ref=run_ref,
                session_id=harness.session_id,
                input_id="i1",
                message="顺便问下,影响性能吗?",
            )
        )
        assert await harness.server.wait_received(1) == [3]

        # Release the turn's final round: it ends pure-text with an invisible
        # draft, the fork still mid-flight, so the plain finalizer round runs.
        after_interrupt.set()
        assert await harness.server.wait_received(2) == [4, 5]
        await asyncio.sleep(0.2)

        # The fork lands only now — after the whole chain ended. The delivery
        # hook drains the idle session: a continuation turn claims the
        # envelope in its opening context and answers through speak.
        fork_gate.set()
        assert await harness.server.wait_received(2) == [6, 7]
        await _eventually(lambda: _answer_spoken(service, harness.session_dir))
        await asyncio.wait_for(interrupt, 10)
        await turn1.result()

        outcome = await asyncio.wait_for(interrupt, 0.1)
        assert outcome.speak_args is not None
        assert outcome.speak_args.text == _FORK_REPLY

        assert turn1.status == "passed"
        await _eventually(
            lambda: _envelope_state(harness.store.path, "realtime-exchange:i1")
            == "delivered"
        )
        texts = await _speak_texts(service, harness.session_dir)
        assert texts == ["开始重构", _DRAFT, _ANSWER]
        history = harness.manager.load(harness.session_id).history
        assert "顺便问下,影响性能吗?" in str(history)
        assert _DRAFT in str(history)
        assert harness.store.pending_messages(harness.session_id) == []
    finally:
        await harness.host.aclose()
        await harness.server.aclose()
