"""Runtime fork-context entry point tests.

The live window must preserve the fork-reply cache prefix, so its rendered
system prompt and message list are compared against the direct restore/render
calls. The frozen window is pure and must stay deterministic for the memory
extractor's fixed cursor range.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from pygent import AIMessage, UserMessage
from pygent.lara.agents import AgentRunner, ForkAgent, ForkSpec
from pygent.lara.core.io import write_json
from pygent.lara.runtime.agent.pipeline import render_model_request_text
from pygent.lara.runtime.fork_context import (
    SessionForkWindowSource,
    session_fork_window,
)
from pygent.lara.runtime.service import LaraRuntimeService
from pygent.lara.schema import AgentSession, CaseRunRef, RunConfig
from pygent.runtime.codec import message_to_dict


def _fixture(tmp_path: Path) -> tuple[RunConfig, AgentSession, CaseRunRef]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    user_root = tmp_path / "user-root"
    user_root.mkdir()
    config = RunConfig(
        workspace_root=str(workspace),
        user_lara_root=str(user_root),
        system_prompt="persisted session prompt",
    )
    session_dir = tmp_path / "sessions" / "session-1"
    (session_dir / "state").mkdir(parents=True)
    run_dir = session_dir / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    session = AgentSession(
        session_id="session-1",
        workspace_root=str(workspace),
        session_dir=str(session_dir),
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        system_prompt="persisted session prompt",
        history=[
            message_to_dict(UserMessage(content="hello")),
            message_to_dict(AIMessage(content="hi there")),
        ],
        metadata={"mode": "chat"},
    )
    write_json(session_dir / "session.json", session.to_dict())
    run_ref = CaseRunRef(
        session_id=session.session_id,
        case_id="case-1",
        case_run_id="run-1",
        run_dir=str(run_dir),
    )
    return config, session, run_ref


def test_live_window_matches_direct_restore_and_render(tmp_path: Path) -> None:
    config, session, run_ref = _fixture(tmp_path)
    # The service method is the pre-refactor live path; keep it as the oracle.
    service = object.__new__(LaraRuntimeService)
    direct = service._restored_session_context(
        session=session,
        run_ref=run_ref,
        turn_id="fork:test",
        config=config,
    )
    direct_prompt = render_model_request_text(config, direct, None)

    window = session_fork_window(
        mode="live",
        config=config,
        session=session,
        run_ref=run_ref,
        turn_id="fork:test",
    )

    assert window.system_prompt == direct_prompt
    assert [message_to_dict(message) for message in window.messages] == [
        message_to_dict(message) for message in direct.messages
    ]
    assert window.metadata["session_id"] == session.session_id


def test_frozen_window_is_deterministic_and_auditable() -> None:
    history = (
        message_to_dict(UserMessage(content="first")),
        message_to_dict(AIMessage(content="second")),
        message_to_dict(UserMessage(content="third")),
    )

    first = session_fork_window(
        mode="frozen",
        system_prompt="frozen session prompt",
        history_range=history,
    )
    second = session_fork_window(
        mode="frozen",
        system_prompt="frozen session prompt",
        history_range=history,
    )

    assert first == second
    assert first.metadata == {}
    assert [message.content for message in first.messages] == [
        "first",
        "second",
        "third",
    ]
    source = SessionForkWindowSource(first)
    assert source.resolve() == first
    assert source.describe() == "session-frozen:3"
    window, snapshot = ForkAgent(
        cast(AgentRunner, None), ForkSpec(source=source)
    ).materialize()
    assert window == first
    assert snapshot.source == "session-frozen:3"
    assert snapshot.message_count == 3
    assert snapshot.source_hash == snapshot.effective_context_hash
