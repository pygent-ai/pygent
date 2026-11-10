"""Byte-level golden tests for every XML envelope rendering site.

These tests intentionally call the pre-Prompt-API implementation. They are
committed before the Prompt migration and must stay unchanged while each site
is migrated.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import pygent.agent.lara.graph.compressor as compressor_module
import pygent.agent.lara.runtime.reminders.bootstrap as bootstrap_module
import pygent.agent.lara.runtime.reminders.skills_context as skills_module
from pygent import AIMessage, UserMessage
from pygent.agent.lara.core.io import write_json
from pygent.agent.lara.core.prompt import system_reminder
from pygent.agent.lara.graph.compressor import LaraCompressorModule
from pygent.agent.lara.runtime.attachments import (
    ResolvedAttachment,
    render_attachments_xml,
)
from pygent.agent.lara.runtime.context import LaraContext
from pygent.agent.lara.runtime.context_compression import (
    FileReadRecord,
    render_file_read_block,
)
from pygent.agent.lara.runtime.eternal_conversation import (
    render_memory_access_instruction,
)
from pygent.agent.lara.runtime.realtime.envelope import render_exchange_envelope
from pygent.agent.lara.runtime.realtime.speak import render_reply_instruction
from pygent.agent.lara.runtime.reminders.agent_messages import render_agent_message
from pygent.agent.lara.runtime.reminders.bootstrap import build_initial_snapshot
from pygent.agent.lara.runtime.reminders.cli_context import collect_initial_cli
from pygent.agent.lara.runtime.reminders.git_context import render_git_section
from pygent.agent.lara.runtime.reminders.models import ReminderScope, ReminderSection
from pygent.agent.lara.runtime.reminders.rendering import render_context_body
from pygent.agent.lara.runtime.reminders.skills_context import collect_initial_skills
from pygent.agent.lara.runtime.service import LaraRuntimeService
from pygent.agent.lara.schema import AgentSession, BashCliPreset, CaseRunRef, RunConfig
from pygent.agent.lara.sessions import AgentMessage
from pygent.agent.lara.sessions.collaboration import AgentMessageState
from pygent.agent.lara.sessions.manager import LARA_SAYS_SESSION_MODE
from pygent.runtime.codec import message_to_dict


def _assert_exact(actual: str, expected: str) -> None:
    assert actual == expected


def test_system_reminder_golden() -> None:
    _assert_exact(
        system_reminder("  do the thing  "),
        """<system-reminder>
do the thing
</system-reminder>""",
    )


def test_exchange_envelope_golden() -> None:
    _assert_exact(
        render_exchange_envelope(
            user_asked=" what changed? \n", response=" I fixed it. "
        ),
        """<system-reminder>
<user-asked>
what changed?
</user-asked>
<your-response-in-another-session>
I fixed it.
</your-response-in-another-session>
</system-reminder>""",
    )


def test_reply_instruction_golden() -> None:
    _assert_exact(
        render_reply_instruction(),
        """  <reply-instruction>
    Open this turn by calling speak with one short reassurance line to the
    user before any tool call or long work. The acknowledgment is in addition to
    the work, never a replacement: after speaking, carry the task through. Even
    if you end up needing to do nothing, still speak before finishing.
  </reply-instruction>""",
    )


def test_agent_message_golden() -> None:
    message = AgentMessage(
        message_id="msg-1",
        submission_id="sub-1",
        source_session_id="source-1",
        source_agent_alias="lara-child",
        target_session_id="target-1",
        content="line <one>\nline & two",
        state=AgentMessageState.QUEUED,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )
    _assert_exact(
        render_agent_message(message),
        """<runtime-context>
  <agent-message message-id="msg-1" source-session-id="source-1" source-agent-alias="lara-child">
    line &lt;one&gt;
    line &amp; two
  </agent-message>
</runtime-context>""",
    )


def test_git_context_baseline_golden() -> None:
    snapshot = {
        "repository_root": "C:/repo & co",
        "captured_at": "2026-01-02T03:04:05+00:00",
        "branch_line": "## main",
        "entries": {
            "a.py": {"status": " M", "signature": "1:2"},
            "b <x>.py": {"status": "??", "signature": "3:4"},
        },
        "status_lines": ["## main", " M a.py", "?? b <x>.py"],
        "truncated": False,
    }
    section = render_git_section(snapshot, reason="session-baseline")
    _assert_exact(
        "\n".join(section.lines),
        """<git-context repository-root="C:/repo &amp; co" captured-at="2026-01-02T03:04:05+00:00" reason="session-baseline" mode="baseline" truncated="false">
  <status format="porcelain-v1">
    ## main
     M a.py
    ?? b &lt;x&gt;.py
  </status>
  <notice>This is a point-in-time snapshot. Run git status before staging, committing, or reporting the current repository state.</notice>
</git-context>""",
    )


def test_git_context_delta_golden() -> None:
    snapshot = {
        "repository_root": "C:/repo & co",
        "captured_at": "2026-01-02T03:04:05+00:00",
        "branch_line": "## main",
        "entries": {
            "a.py": {"status": " M", "signature": "1:2"},
            "b <x>.py": {"status": "??", "signature": "3:4"},
        },
        "status_lines": ["## main", " M a.py", "?? b <x>.py"],
        "truncated": False,
    }
    previous = {
        "branch_line": "## old",
        "entries": {
            "a.py": {"status": " M", "signature": "0:0"},
            "gone.py": {"status": "??", "signature": "5:6"},
        },
    }
    section = render_git_section(
        snapshot, reason="tool-change", previous_snapshot=previous
    )
    _assert_exact(
        "\n".join(section.lines),
        """<git-context repository-root="C:/repo &amp; co" captured-at="2026-01-02T03:04:05+00:00" reason="tool-change" mode="delta" truncated="false">
  <changes>
    <branch-change before="## old" after="## main" />
    <change kind="content-changed" status=" M" path="a.py" />
    <change kind="added" status="??" path="b &lt;x&gt;.py" />
    <change kind="removed" previous-status="??" path="gone.py" />
  </changes>
  <summary dirty-total="2" change-count="4" />
  <notice>This is a point-in-time snapshot. Run git status before staging, committing, or reporting the current repository state.</notice>
</git-context>""",
    )


def test_skills_context_golden(monkeypatch: pytest.MonkeyPatch) -> None:
    skills = [
        {
            "name": "alpha-skill",
            "description": "Does <alpha> & stuff",
            "path": "C:/skills/alpha/SKILL.md",
            "source": "user",
            "scope": "user",
            "uri": "user://skills/alpha-skill/SKILL.md",
            "content_hash": "hash",
            "shadowed": [],
        },
        {
            "name": "beta-skill",
            "description": "Beta description",
            "path": "C:/skills/beta/SKILL.md",
            "source": "project",
            "scope": "project",
            "uri": "project://skills/beta-skill/SKILL.md",
            "content_hash": "hash2",
            "shadowed": [],
        },
    ]
    monkeypatch.setattr(skills_module, "scan_multilevel_skills", lambda **_: skills)
    monkeypatch.setattr(skills_module, "skills_fingerprint", lambda *_: "fingerprint")
    scope = ReminderScope(
        "session-1",
        Path("C:/session"),
        Path("C:/workspace"),
        (),
        Path("C:/user-skills"),
        Path("C:/project-skills"),
    )
    section, _ = collect_initial_skills(scope)
    assert section is not None
    _assert_exact(
        "\n".join(section.lines),
        """<skills-context>
  <available-skills>
    <skill name="alpha-skill" source="user">
      <description>Does &lt;alpha&gt; &amp; stuff</description>
      <path>C:/skills/alpha/SKILL.md</path>
    </skill>
    <skill name="beta-skill" source="project">
      <description>Beta description</description>
      <path>C:/skills/beta/SKILL.md</path>
    </skill>
  </available-skills>
</skills-context>""",
    )


def test_cli_context_golden() -> None:
    presets = (
        BashCliPreset(
            name="alpha-cli",
            command="alpha-cli --help",
            description="Alpha <cli> & tools",
        ),
        BashCliPreset(
            name="beta-cli",
            command="uv run beta-cli --help",
            description="Beta tools",
        ),
    )
    section, _ = collect_initial_cli(presets)
    assert section is not None
    _assert_exact(
        "\n".join(section.lines),
        """<cli-context>
  <available-bash-cli>
    <alpha-cli>
      Alpha &lt;cli&gt; &amp; tools
      Command: alpha-cli --help
      Status: not installed.
    </alpha-cli>
    <beta-cli>
      Beta tools
      Command: uv run beta-cli --help
      Available via uv run in this workspace.
    </beta-cli>
  </available-bash-cli>
</cli-context>""",
    )


def test_rendering_time_golden() -> None:
    sections = [
        ReminderSection(
            "a",
            0,
            ("<runtime-context>", "  body", "</runtime-context>"),
            True,
        )
    ]
    _assert_exact(
        render_context_body(
            sections,
            rendered_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        )
        or "",
        """<time>
  Current system time: 2026-01-02 03:04:05 UTC
</time>

<runtime-context>
  body
</runtime-context>""",
    )


def test_bootstrap_runtime_context_golden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        bootstrap_module, "_now", lambda: "2026-01-02T03:04:05+00:00"
    )
    monkeypatch.setattr(
        bootstrap_module, "capture_git_snapshot", lambda *_, **__: None
    )
    scope = ReminderScope(
        "session-1",
        Path("C:/session"),
        Path("C:/workspace"),
        (),
        Path("C:/user-skills"),
        Path("C:/project-skills"),
    )
    snapshot, _ = build_initial_snapshot(scope)
    _assert_exact(
        snapshot.content.replace(os.sep, "/"),
        r"""<time>
  Current system time: 2026-01-02 03:04:05 UTC
</time>

<runtime-context>
  <session id="session-1" />
  <workspace-root>C:/workspace</workspace-root>
</runtime-context>""",
    )


def test_file_read_golden() -> None:
    records = [
        FileReadRecord(path="src/a.py", mode="full", content="print('<hi>')\n"),
        FileReadRecord(
            path="src/b <x>.py",
            mode="partial",
            range="10-20",
            content="x" * 20,
        ),
    ]
    text, metadata = render_file_read_block(records, max_chars=15)
    _assert_exact(
        text,
        """<file-read>
<file path="src/a.py" mode="full" truncated="false" char_count="14">
print('<hi>')

</file>
<file path="src/b &lt;x&gt;.py" mode="partial" range="10-20" truncated="true" char_count="20">
The previous file read result is too large to include in this compacted context, so its details are not shown here. Path: src/b <x>.py. Read scope: lines 10-20. Returned content length: 20 characters. Re-read this exact file or range if you need the content.
</file>
</file-read>""",
    )
    assert metadata == [
        {
            "path": "src/a.py",
            "mode": "full",
            "included": True,
            "char_count": 14,
            "truncated": False,
        },
        {
            "path": "src/b <x>.py",
            "mode": "partial",
            "included": False,
            "char_count": 20,
            "truncated": True,
            "range": "10-20",
        },
    ]


def test_memory_access_instruction_golden() -> None:
    _assert_exact(
        render_memory_access_instruction(
            Path("C:/golden/session"),
            {"memory_cli_command": "uv run memory-cli --root C:/mem"},
        ).replace(os.sep, "/"),
        r"""<memory-access-instruction>
Use the mounted dynamic-memory-cli search interface when the Snapshot is insufficient or the task depends on prior decisions, constraints, commitments, preferences, or detailed history.
Published memory records prior decisions, not immutable authority.
Memory publication is automatic and asynchronous: this conversation is durably journaled and distilled into long-term memory without action from you, so you may promise the user that something will be remembered. If a search for something the user asked to remember misses, do not claim loss — the entry is recorded and its long-term publication is pending.
The Snapshot is a lossy compression: its entries record current facts without the underlying reasons, decision rationale, or history. Treat every Snapshot entry as a pointer, not the full truth. Coverage-gap rule: if keywords, nouns, or component names in the user's input are not fully explained by the current context — their background, rationale, or why-it-is-so is missing — first run the search command with those exact keywords or nouns to recover the stored rationale and prior context, then complete the task. When a recovered record explains why the current state exists, your final answer must explicitly state that reason.
Conflict protocol (apply before acting or changing files): an override is acknowledged only when the current user message refers to the earlier constraint or decision and communicates an intent to replace it. A message that states only the new, contradictory behavior is always unacknowledged; never infer acknowledgment merely because the requested behavior is clearly opposite.
Mandatory conflict evidence (before any clarification question): when the current request may contradict, remove, or weaken any prior decision, constraint, commitment, or named convention, search dynamic memory at least twice before responding — once for the exact proper noun, codename, alias, or named constraint, and once for the affected component or the contradictory behavior. Do this even when the current context already states the conflict: the context helps you notice it, the published records supply the auditable decision evidence. If the search command is not available yet, perform the same two lookups directly against Raw History.
Action gate: while a conflict is suspected under the protocol above, the only permitted tool calls are the required memory searches; do not inspect, modify, or test project files until the user resolves the conflict.
Literal-term rule: if a search result or Raw History match contains a user-defined proper noun, codename, alias, or environment name, repeat that exact original term in both your conflict statement and your final clarification question; never replace it with a generic label. If the request removes a promise that carries a literal version/date boundary, repeat that boundary verbatim as well.
For an unacknowledged conflict, pause before acting. Stop further exploration immediately. Do not accept the new request, reject it, or choose a workaround on the user's behalf. In at most 120 words, state the conflict briefly, then end the response with exactly one direct clarification question that names the prior option and the new option; the final character of your response MUST be `?` or `？`. Wait for the user's answer.
Search command: uv run memory-cli --root C:/mem search <keyword-or-key-phrase>
Complete observable Raw History: C:/golden/session/raw-history/events.jsonl
Use read/grep tools directly on Raw History for evidence. Before changing existing behavior, search both dynamic memory and Raw History for prior user constraints, commitments, preferences, and environment assumptions. For compatibility removal, alias removal, deprecation, or version migration, explicitly search Raw History for the affected feature name together with support-window, version, release, compatibility, and deprecation terms before editing. Preserve the earliest relevant direct-user matches (for example, use a bounded first-match search); a tail-only view can hide the original constraint and is not sufficient evidence. If memory search returns only implementation facts rather than direct user intent, Raw History is the required fallback, not an optional extra. Never invent missing history.
</memory-access-instruction>""",
    )


def test_attachments_golden(tmp_path: Path) -> None:
    note = tmp_path / "notes.txt"
    note.write_text("hello", encoding="utf-8")
    items = (
        ResolvedAttachment(
            path=note,
            relative="notes.txt",
            name="notes.txt",
            size=5,
            mime="text/plain",
            delivery="inline",
        ),
        ResolvedAttachment(
            path=tmp_path / "photo.png",
            relative="photo.png",
            name="photo.png",
            size=10,
            mime="image/png",
            delivery="media",
            media_type="image",
        ),
        ResolvedAttachment(
            path=tmp_path / "big.bin",
            relative="big.bin",
            name="big.bin",
            size=100,
            mime="application/octet-stream",
            delivery="reference",
        ),
    )
    _assert_exact(
        render_attachments_xml(items),
        """  <attachments>
    <attachment delivery="inline" name="notes.txt" path="notes.txt" size="5" mime="text/plain" truncated="false">hello</attachment>
    <attachment delivery="media" name="photo.png" path="photo.png" size="10" mime="image/png" note="media attached natively to this user message"/>
    <attachment delivery="reference" name="big.bin" path="big.bin" size="100" mime="application/octet-stream" note="content not embedded; inspect with read/grep tools"/>
  </attachments>""",
    )


async def test_user_context_golden(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_dir = tmp_path / "sessions" / "session-1"
    (session_dir / "state").mkdir(parents=True)
    run_dir = session_dir / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    note = workspace / "notes.txt"
    note.write_text("hello", encoding="utf-8")
    config = RunConfig(
        workspace_root=str(workspace),
        user_lara_root=str(tmp_path / "user"),
    )
    session = AgentSession(
        session_id="session-1",
        workspace_root=str(workspace),
        session_dir=str(session_dir),
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        system_prompt="persisted prompt",
        history=[
            message_to_dict(UserMessage(content="earlier")),
            message_to_dict(AIMessage(content="reply")),
        ],
        metadata={"mode": "chat"},
    )
    write_json(session_dir / "session.json", session.to_dict())
    run_ref = CaseRunRef(
        session_id="session-1",
        case_id="case-1",
        case_run_id="run-1",
        run_dir=str(run_dir),
    )
    service = object.__new__(LaraRuntimeService)
    service.reminders = _NoReminders()  # type: ignore[assignment]
    message, _ = await service._prepare_turn(
        manager=None,  # type: ignore[arg-type]
        run_ref=run_ref,
        message="please <inspect> & fix",
        config=config,
        turn_id="turn-1",
        session=session,
        attachments=["notes.txt"],
    )
    _assert_exact(
        message.content,
        """<user-context>
  <user-identity>default</user-identity>
  <user-message>please &lt;inspect&gt; &amp; fix</user-message>
  <attachments>
    <attachment delivery="inline" name="notes.txt" path="notes.txt" size="5" mime="text/plain" truncated="false">hello</attachment>
  </attachments>
</user-context>""",
    )


async def test_user_context_realtime_reply_instruction_golden(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_dir = tmp_path / "sessions" / "session-1"
    (session_dir / "state").mkdir(parents=True)
    run_dir = session_dir / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    config = RunConfig(
        workspace_root=str(workspace),
        user_lara_root=str(tmp_path / "user"),
    )
    session = AgentSession(
        session_id="session-1",
        workspace_root=str(workspace),
        session_dir=str(session_dir),
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        system_prompt="persisted prompt",
        history=[
            message_to_dict(UserMessage(content="earlier")),
            message_to_dict(AIMessage(content="reply")),
        ],
        metadata={"mode": LARA_SAYS_SESSION_MODE},
    )
    write_json(session_dir / "session.json", session.to_dict())
    run_ref = CaseRunRef(
        session_id="session-1",
        case_id="case-1",
        case_run_id="run-1",
        run_dir=str(run_dir),
    )
    service = object.__new__(LaraRuntimeService)
    service.reminders = _NoReminders()  # type: ignore[assignment]
    message, _ = await service._prepare_turn(
        manager=None,  # type: ignore[arg-type]
        run_ref=run_ref,
        message="please <inspect> & fix",
        config=config,
        turn_id="turn-1",
        session=session,
    )
    _assert_exact(
        message.content,
        """<user-context>
  <user-identity>default</user-identity>
  <user-message>please &lt;inspect&gt; &amp; fix</user-message>
  <reply-instruction>
    Open this turn by calling speak with one short reassurance line to the
    user before any tool call or long work. The acknowledgment is in addition to
    the work, never a replacement: after speaking, carry the task through. Even
    if you end up needing to do nothing, still speak before finishing.
  </reply-instruction>
</user-context>""",
    )


class _NoReminders:
    async def claim_initial(self, *_args: Any, **_kwargs: Any) -> str:
        return ""

    async def collect_pending(self, *_args: Any, **_kwargs: Any) -> str:
        return ""


class _FakeCompressorModel:
    async def __call__(
        self, _message: Any, context: LaraContext
    ) -> tuple[AIMessage, LaraContext]:
        return AIMessage(content="<summary>Compact summary & details</summary>"), context


async def test_compressor_session_context_golden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        compressor_module,
        "_session_dir_for_run",
        lambda _: Path("C:/golden/session"),
    )
    monkeypatch.setattr(
        compressor_module,
        "collect_recent_file_reads",
        lambda *_, **__: [],
    )
    monkeypatch.setattr(
        compressor_module,
        "render_file_read_block",
        lambda *_, **__: (
            (
                '<file-read>\n<file path="a.py" mode="full" truncated="false" '
                'char_count="2">ok</file>\n</file-read>'
            ),
            [],
        ),
    )
    config = RunConfig(
        workspace_root="C:/golden/workspace",
        user_lara_root="C:/golden/user",
    )
    module = LaraCompressorModule(config, _FakeCompressorModel())  # type: ignore[arg-type]

    async def _emit(**_kwargs: Any) -> None:
        return None

    module.emit = _emit  # type: ignore[method-assign]
    context = LaraContext(
        session_id="s",
        case_id="c",
        case_run_id="r",
        run_dir="C:/golden/session",
        turn_id="t",
    )
    answer, _ = await module.forward(AIMessage(content="request"), context)
    _assert_exact(
        answer.content.replace(os.sep, "/"),
        r"""<session-context>
Compact summary & details

<file-read>
<file path="a.py" mode="full" truncated="false" char_count="2">ok</file>
</file-read>

If you need specific details from before compaction (like exact code snippets, error messages, or content you generated), read the full transcript at:
C:/golden/session/session.json (the authoritative transcript is in its history field)
</session-context>""",
    )
