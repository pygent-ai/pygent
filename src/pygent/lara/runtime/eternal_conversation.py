from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import warnings
from collections.abc import Awaitable, Callable
from contextvars import Context
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pygent.lara.core.io import append_jsonl, read_json, utc_now, write_json
from pygent.lara.core.prompt import Prompt
from pygent.lara.core.redaction import redact_secrets
from pygent.lara.schema import AgentSession, EternalConversationConfig, RunConfig
from pygent.lara.clis.provision import ensure_dynamic_memory_cli

MAX_MEMORY_JOB_ATTEMPTS = 3


def eternal_enabled_for(session: AgentSession, config: RunConfig) -> bool:
    """The deployment master switch gates the feature; a session opts in.

    Normal sessions stay off the eternal conversation unless their metadata
    carries ``eternal_memory``; forks inherit the flag with the copied
    ``session.json``, so a fork of the memory session stays a memory session.
    """
    return bool(config.eternal_conversation.enabled) and bool(
        session.metadata.get("eternal_memory")
    )


@dataclass(frozen=True)
class MemoryJobContext:
    """Everything a memory job needs to reach the memory CLI and evidence."""

    session_id: str
    memory_script: Path
    memory_root: Path
    evidence_ref: str
    covered_through: int
    target_cursor: int
    build_batch_path: Path | None = None


@dataclass(frozen=True)
class MemoryJobRequest:
    """One background memory job, fork-shaped for the extractor.

    The extractor is forked from the foreground conversation: block 1 is the
    foreground system prompt plus the frozen working-memory history range
    (verbatim, uncompacted); block 2 is the directive. The builder is an
    independent agent with its own system prompt and an empty history range.
    Tools come from the runner's registered surface, aligned with the
    foreground StandardTools composition (bash, files, task control) per the
    owner's decision; there are no dedicated memory tools.
    """

    role: str  # "extractor" | "builder"
    alias: str
    label: str
    system_prompt: str
    history_range: tuple[dict[str, Any], ...]
    directive: str
    context: MemoryJobContext

    def __post_init__(self) -> None:
        if self.role not in {"extractor", "builder"}:
            raise ValueError(f"unsupported memory job role {self.role!r}")
        if not self.system_prompt.strip():
            # The harness applies the legacy-session fallback; an empty value
            # reaching this point means the fallback itself was skipped.
            raise ValueError("MemoryJobRequest.system_prompt must be non-empty")
        if not self.directive.strip():
            raise ValueError("MemoryJobRequest.directive must be non-empty")


MemoryJobRunner = Callable[[MemoryJobRequest], Awaitable[str]]


EXTRACTOR_GUIDANCE = """You are the same agent, forked from the foreground conversation to
maintain its memory. The conversation above is the frozen Working Memory; use it plus the
old Snapshot from this reminder as the continuous history input. Compare proposed memories
with the published UTs and resolve conflicts by updating stable UT IDs. Preserve exact
answer-bearing facts, decisions, constraints, commitments, active work, and likely future
retrieval phrasings. Version support windows, deprecation deadlines, compatibility
promises, and constraints the user asks not to publish in project documentation are still
durable memory and MUST be retained with their exact version/date boundary. A durable user
decision or constraint MUST have its own stable UT and MUST NOT share a UT with an
evolving implementation summary; this separation prevents later code updates from
overwriting it. Treat user-defined proper nouns, internal codenames, aliases, environment names,
and their referents as durable retrieval keys whenever they may affect future behavior. Preserve the exact user-authored name in the UT content, at least one query, and must_include;
never leave it only in the Snapshot. A request to keep a name out of source
code, repository files, or project documentation means memory-only visibility, not
permission to omit it. Before publishing, self-audit every future-relevant named entity and
alias in the continuous history and ensure an existing or changed stable UT carries
both the exact name and its meaning. When updating an existing UT, carry forward every still-effective decision,
constraint, commitment, and exact boundary from the published UT; absence from the frozen Working Memory is not evidence that an older constraint became
stale. Do not decide cursor or publication legality. You MUST publish the proposal file
with the memory CLI publish-pending command. Do not finish until publish-pending succeeds.
If the command reports an over-limit field, rewrite only as needed to make the proposal
shorter while preserving its exact durable facts, then run publish-pending again in this
same ReAct execution.

The proposal file is: {"base_memory_revision": <int>, "base_snapshot_revision": <int>,
"from_cursor": <int>, "to_cursor": <int>, "snapshot": {"resident_memory":[],
"recent_context":[],"current_state":[],"completed":[],"next_actions":[],"constraints":[]},
"changed_uts": [UT changes], "evidence_refs": [...], "semantic_statement":"..."}. Each
upsert UT needs action,id,memory_id,priority,content,queries,must_include,evidence_refs,
source,tags. priority MUST be an integer from 0 through 100. Every must_include item MUST
be an exact substring of content. Use action=retire with id only when evidence makes a UT
stale. An empty changed_uts list is valid. The Snapshot must carry everything the
foreground must know without retrieval and must treat later Working Memory as newer than
the Snapshot. Keep the result compact: at most 8 changed UTs; merge updates into stable
component-level UTs; each UT content at most 700 characters, at most 4 queries, and at
most 3 must_include phrases. Split larger bodies across multiple focused UTs instead of
overloading one UT. Snapshot limits are: resident_memory 4 items, recent_context 4,
current_state 6, completed 4, next_actions 4, constraints 6; each item at most 500
characters. Prefer exact dense facts over narration. Never copy old Snapshot items
unchanged when a shorter merged item preserves them."""

BUILDER_GUIDANCE = """You are the memory build Agent. Review the frozen Pending UT batch for
internal consistency and build readiness without changing memory semantics. When it is
ready, run the memory CLI build-pending command over the frozen batch file. The command
performs deterministic construction, Built-only tests, exact-content comparison, and
atomic migration from Pending to Built. Do not claim the batch is Built unless build-pending
reports success. If the command fails, use its result to decide whether a corrected retry
is possible in this same ReAct execution."""


class DynamicMemoryCli:
    def __init__(self, script: Path, root: Path) -> None:
        self.script = script
        self.root = root

    async def call(self, *args: str) -> dict[str, Any]:
        env = dict(os.environ)
        env.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(self.script),
            *args,
            cwd=str(self.root),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        text = stdout.decode("utf-8", errors="replace")
        try:
            result = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"dynamic memory CLI returned invalid JSON: {text or stderr.decode(errors='replace')}"
            ) from exc
        if process.returncode != 0:
            raise RuntimeError(f"dynamic memory CLI failed: {result}")
        return result

    async def init(self) -> dict[str, Any]:
        self.root.mkdir(parents=True, exist_ok=True)
        return await self.call("init", "--path", str(self.root))


def load_projection(session_dir: str | Path) -> dict[str, Any]:
    return read_json(
        Path(session_dir) / "state" / "eternal-conversation.json", default={}
    )


def render_memory_access_instruction(
    session_dir: str | Path, projection: dict[str, Any]
) -> str:
    raw = Path(session_dir) / "raw-history" / "events.jsonl"
    command = str(projection.get("memory_cli_command") or "").strip()
    if command:
        access = "Use the mounted dynamic-memory-cli search interface when the Snapshot is insufficient or the task depends on prior decisions, constraints, commitments, preferences, or detailed history."
        search = f"Search command: {command} search <keyword-or-key-phrase>"
        evidence_sources = "both dynamic memory and Raw History"
    else:
        access = "Memory search is not available for this session yet. Do not invoke or invent a memory CLI command. Use the current conversation and any available Snapshot; consult Raw History if the file exists when earlier context is needed."
        search = "The memory search command will be provided after the first memory publication. An unavailable search interface does not imply that prior history is empty."
        evidence_sources = "available Raw History"
    return str(
        Prompt(
            "\n".join(
                (
                    access,
                    "Published memory records prior decisions, not immutable authority.",
                    "Memory publication is automatic and asynchronous: this conversation is durably journaled and distilled into long-term memory without action from you, so you may promise the user that something will be remembered. If a search for something the user asked to remember misses, do not claim loss — the entry is recorded and its long-term publication is pending.",
                    "The Snapshot is a lossy compression: its entries record current facts without the underlying reasons, decision rationale, or history. Treat every Snapshot entry as a pointer, not the full truth. Coverage-gap rule: if keywords, nouns, or component names in the user's input are not fully explained by the current context — their background, rationale, or why-it-is-so is missing — first run the search command with those exact keywords or nouns to recover the stored rationale and prior context, then complete the task. When a recovered record explains why the current state exists, your final answer must explicitly state that reason.",
                    "Conflict protocol (apply before acting or changing files): an override is acknowledged only when the current user message refers to the earlier constraint or decision and communicates an intent to replace it. A message that states only the new, contradictory behavior is always unacknowledged; never infer acknowledgment merely because the requested behavior is clearly opposite.",
                    "Mandatory conflict evidence (before any clarification question): when the current request may contradict, remove, or weaken any prior decision, constraint, commitment, or named convention, search dynamic memory at least twice before responding — once for the exact proper noun, codename, alias, or named constraint, and once for the affected component or the contradictory behavior. Do this even when the current context already states the conflict: the context helps you notice it, the published records supply the auditable decision evidence. If the search command is not available yet, perform the same two lookups directly against Raw History.",
                    "Action gate: while a conflict is suspected under the protocol above, the only permitted tool calls are the required memory searches; do not inspect, modify, or test project files until the user resolves the conflict.",
                    "Literal-term rule: if a search result or Raw History match contains a user-defined proper noun, codename, alias, or environment name, repeat that exact original term in both your conflict statement and your final clarification question; never replace it with a generic label. If the request removes a promise that carries a literal version/date boundary, repeat that boundary verbatim as well.",
                    "For an unacknowledged conflict, pause before acting. Stop further exploration immediately. Do not accept the new request, reject it, or choose a workaround on the user's behalf. In at most 120 words, state the conflict briefly, then end the response with exactly one direct clarification question that names the prior option and the new option; the final character of your response MUST be `?` or `？`. Wait for the user's answer.",
                    search,
                    f"Complete observable Raw History: {raw}",
                    f"Use read/grep tools directly on Raw History for evidence. Before changing existing behavior, search {evidence_sources} for prior user constraints, commitments, preferences, and environment assumptions. For compatibility removal, alias removal, deprecation, or version migration, explicitly search Raw History for the affected feature name together with support-window, version, release, compatibility, and deprecation terms before editing. Preserve the earliest relevant direct-user matches (for example, use a bounded first-match search); a tail-only view can hide the original constraint and is not sufficient evidence. If memory search returns only implementation facts rather than direct user intent, Raw History is the required fallback, not an optional extra. Never invent missing history.",
                )
            )
        ).wrap("memory-access-instruction")
    )


def render_memory_snapshot(projection: dict[str, Any]) -> str:
    snapshot = projection.get("snapshot") or {}
    revision = int(projection.get("snapshot_revision") or 0)
    covered = int(projection.get("covered_through") or 0)
    return "\n".join(
        (
            f'<memory-snapshot revision="{revision}" covered-through="{covered}">',
            json.dumps(snapshot, ensure_ascii=False, indent=2),
            "</memory-snapshot>",
        )
    )


def estimate_context_tokens(
    history: list[dict[str, Any]], projection: dict[str, Any]
) -> int:
    """Estimate tokens of the model-visible history: snapshot + uncovered tail.

    Uses the same lexical approximation as the Pygent request estimator so the
    value is comparable against a RunConfig.context_window budget.
    """

    covered = int(projection.get("covered_through") or 0)
    visible: list[Any] = []
    if covered > 0:
        visible.append(render_memory_snapshot(projection))
    visible.extend(history[max(covered, 0) :])
    canonical = json.dumps(visible, ensure_ascii=False, default=str)
    ascii_bytes = len(canonical.encode("ascii", errors="ignore"))
    non_ascii_codepoints = len(canonical) - ascii_bytes
    lexical = (ascii_bytes + 2) // 3 + (non_ascii_codepoints * 3 + 1) // 2
    structural = 8 * (len(visible) + 1)
    return lexical + structural


class EternalConversationHarness:
    def __init__(
        self,
        config: EternalConversationConfig,
        *,
        run_memory_job: MemoryJobRunner,
        user_clis_root: Path | None = None,
        foreground_system_prompt: str | None = None,
    ) -> None:
        self.config = config
        self.run_memory_job = run_memory_job
        self._foreground_system_prompt = (foreground_system_prompt or "").strip()
        self._user_clis_root = (
            (user_clis_root or Path.home() / ".lara" / "clis")
            .expanduser()
            .resolve()
        )
        self._workers: dict[str, asyncio.Task[None]] = {}
        self._builders: dict[str, asyncio.Task[None]] = {}
        self._closed = False
        self._accepting = True

    def _script(self, session_dir: Path) -> Path:
        if self.config.dynamic_memory_cli_path:
            path = Path(self.config.dynamic_memory_cli_path)
            if path.is_file():
                return path
        try:
            ensure_dynamic_memory_cli(self._user_clis_root)
        except OSError as exc:
            warnings.warn(
                f"dynamic-memory-cli provisioning failed: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
        for root in (
            session_dir.parents[1] / "clis",
            self._user_clis_root,
        ):
            path = root / "dynamic-memory-cli" / "dynamic_memory_cli.py"
            if path.is_file():
                return path
        env = os.environ.get("LARA_DYNAMIC_MEMORY_CLI")
        if env and Path(env).is_file():
            return Path(env).resolve()
        raise FileNotFoundError(
            "dynamic-memory-cli script is not configured or installed"
        )

    async def record_and_trigger(
        self,
        session: AgentSession,
        *,
        model_envelope: dict[str, Any] | None = None,
        context_window_tokens: int | None = None,
    ) -> None:
        if not self.config.enabled or self._closed or not self._accepting:
            return
        session_dir = Path(session.session_dir)
        state_path = session_dir / "state" / "eternal-harness.json"
        state = read_json(state_path, default={})
        recorded = int(state.get("recorded_messages") or 0)
        raw_path = session_dir / "raw-history" / "events.jsonl"
        if model_envelope is not None:
            append_jsonl(
                raw_path,
                {
                    "cursor": f"envelope-{len(session.history)}",
                    "type": "model-visible-envelope",
                    "session_id": session.session_id,
                    "created_at": utc_now(),
                    "payload": redact_secrets(model_envelope),
                },
            )
        for index, message in enumerate(session.history[recorded:], start=recorded + 1):
            event = {
                "cursor": index,
                "type": "message",
                "session_id": session.session_id,
                "created_at": utc_now(),
                "payload": redact_secrets(message),
            }
            append_jsonl(raw_path, event)
            append_jsonl(
                session_dir / "agent-history" / "foreground" / "conversation.jsonl",
                event,
            )
        state.update({"recorded_messages": len(session.history)})
        if not self._should_extract(session, session_dir, context_window_tokens):
            # Observability recording continues every turn; extraction waits for
            # the configured trigger so Working Memory freezes in larger batches.
            write_json(state_path, state)
            return
        state.update({"requested_cursor": len(session.history)})
        _update_job_state(state, "extractor", status="queued", error=None)
        write_json(state_path, state)
        task = self._workers.get(session.session_id)
        if task is None or task.done():
            self._workers[session.session_id] = asyncio.create_task(
                self._guarded_extract_loop(session.session_id, session_dir),
                name=f"eternal-extractor:{session.session_id}",
                context=Context(),
            )

    def _should_extract(
        self,
        session: AgentSession,
        session_dir: Path,
        context_window_tokens: int | None,
    ) -> bool:
        if self.config.extraction_trigger != "context_ratio":
            return True
        if not isinstance(context_window_tokens, int) or context_window_tokens <= 0:
            # Without a known window budget the ratio cannot be evaluated;
            # deferring extraction could let visible history grow unbounded.
            return True
        projection = load_projection(session_dir)
        estimate = estimate_context_tokens(list(session.history), projection)
        return estimate >= int(
            context_window_tokens * self.config.extraction_trigger_ratio
        )

    async def _guarded_extract_loop(self, session_id: str, session_dir: Path) -> None:
        await self._guarded_job(
            "extractor",
            session_dir,
            lambda failure: self._extract_loop(
                session_id, session_dir, previous_failure=failure
            ),
        )

    async def _extract_loop(
        self,
        session_id: str,
        session_dir: Path,
        *,
        previous_failure: str | None = None,
    ) -> None:
        cli = DynamicMemoryCli(self._script(session_dir), session_dir / "memory")
        await cli.init()
        published_during_run = False
        while not self._closed:
            harness_state = read_json(
                session_dir / "state" / "eternal-harness.json", default={}
            )
            formal = await cli.call("get-state")
            target = int(harness_state.get("requested_cursor") or 0)
            covered = int(formal.get("covered_through") or 0)
            if target <= covered:
                self._set_job_state(session_dir, "extractor", status="idle", error=None)
                if not published_during_run:
                    self._schedule_builder(session_id, session_dir, cli)
                return
            session = read_json(session_dir / "session.json")
            history = list(session.get("history") or [])[covered:target]
            published = await cli.call("list", "--full")
            published_memories = list(published.get("memories") or [])
            query = _memory_query(history)
            related = (
                await cli.call("search", query)
                if query and published_memories
                else {"matches": []}
            )
            related_ids = [
                str(item.get("id")) for item in related.get("matches") or []
            ][:32]
            by_id = {str(item.get("id")): item for item in published_memories}
            high_priority = sorted(
                published_memories,
                key=lambda item: int(item.get("priority") or 0),
                reverse=True,
            )[:12]
            conflict_reference = []
            seen: set[str] = set()
            for item in [
                *(by_id[ut_id] for ut_id in related_ids if ut_id in by_id),
                *high_priority,
            ]:
                ut_id = str(item.get("id"))
                if ut_id not in seen:
                    seen.add(ut_id)
                    conflict_reference.append(item)
            evidence_ref = f"raw-history:range-{covered + 1}-{target}"
            job_context = MemoryJobContext(
                session_id=session_id,
                memory_script=cli.script,
                memory_root=cli.root,
                evidence_ref=evidence_ref,
                covered_through=covered,
                target_cursor=target,
            )
            directive = _extractor_directive(
                context=job_context,
                snapshot=formal.get("snapshot") or {},
                published_uts=conflict_reference,
                previous_failure=previous_failure,
            )
            # Binding order: the session's persisted prompt governed the
            # frozen turns (historically accurate); the composition root's
            # configured prompt covers sessions without one. No kernel
            # fallback content — an empty result is a configuration error.
            bound_system_prompt = (
                str(session.get("system_prompt") or "").strip()
                or self._foreground_system_prompt
            )
            if not bound_system_prompt:
                raise RuntimeError(
                    "no system prompt available for the memory extractor: "
                    "the session has no persisted prompt and none was "
                    "configured by the composition root"
                )
            request = MemoryJobRequest(
                role="extractor",
                alias=self.config.extractor_agent_alias or "default",
                label=f"lara-fork.extractor:{covered + 1}-{target}",
                system_prompt=bound_system_prompt,
                history_range=tuple(history),
                directive=directive,
                context=job_context,
            )
            self._append_job_history(session_dir, request)
            # The fork agent publishes through the memory CLI itself (bash);
            # the harness only verifies the formal state afterwards.
            await self.run_memory_job(request)
            formal_after = await cli.call("get-state")
            if int(formal_after.get("covered_through") or 0) != target:
                raise RuntimeError(
                    "extractor fork finished without publishing coverage: "
                    f"covered_through={formal_after.get('covered_through')} target={target}"
                )
            result = dict(formal_after)
            result["build_state"] = "pending"
            result["memory_cli_command"] = (
                f'"{sys.executable}" "{cli.script}" --root "{cli.root}"'
            )
            write_json(session_dir / "state" / "eternal-conversation.json", result)
            self._schedule_builder(session_id, session_dir, cli)
            published_during_run = True

    def _schedule_builder(
        self, session_id: str, session_dir: Path, cli: DynamicMemoryCli
    ) -> None:
        task = self._builders.get(session_id)
        if task is None or task.done():
            self._set_job_state(session_dir, "builder", status="queued", error=None)
            self._builders[session_id] = asyncio.create_task(
                self._guarded_build_loop(session_id, session_dir, cli),
                name=f"eternal-builder:{session_id}",
                context=Context(),
            )

    async def _guarded_build_loop(
        self, session_id: str, session_dir: Path, cli: DynamicMemoryCli
    ) -> None:
        await self._guarded_job(
            "builder",
            session_dir,
            lambda _: self._build_loop(session_id, session_dir, cli),
        )

    async def _guarded_job(
        self,
        role: str,
        session_dir: Path,
        operation: Callable[[str | None], Awaitable[None]],
    ) -> None:
        state = read_json(session_dir / "state" / "eternal-harness.json", default={})
        failure = ((state.get("memory_jobs") or {}).get(role) or {}).get("error")
        failure = str(failure) if failure else None
        last_error: Exception | None = None
        for job_attempt in range(1, MAX_MEMORY_JOB_ATTEMPTS + 1):
            self._set_job_state(session_dir, role, status="running", error=None)
            try:
                await operation(failure)
                return
            except Exception as exc:
                last_error = exc
                failure = f"{type(exc).__name__}: {exc}"
                append_jsonl(
                    session_dir / "agent-history" / role / "errors.jsonl",
                    {
                        "created_at": utc_now(),
                        "job_attempt": job_attempt,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                self._set_job_state(session_dir, role, status="failed", error=failure)
                if self._closed:
                    break
        assert last_error is not None
        raise last_error

    async def _build_loop(
        self, session_id: str, session_dir: Path, cli: DynamicMemoryCli
    ) -> None:
        while not self._closed:
            batch_path = (
                session_dir
                / "memory"
                / "jobs"
                / f"build-{utc_now().replace(':', '-')}.json"
            )
            frozen = await cli.call("freeze-pending", "--output", str(batch_path))
            if int(frozen.get("count") or 0) == 0:
                self._set_job_state(session_dir, "builder", status="idle", error=None)
                return
            batch = read_json(batch_path)
            job_context = MemoryJobContext(
                session_id=session_id,
                memory_script=cli.script,
                memory_root=cli.root,
                evidence_ref=f"build-batch:{batch_path.name}",
                covered_through=0,
                target_cursor=0,
                build_batch_path=batch_path,
            )
            directive = _builder_directive(context=job_context, batch=batch)
            request = MemoryJobRequest(
                role="builder",
                alias=self.config.builder_agent_alias or "default",
                label=f"lara-child.builder:{batch_path.stem}",
                system_prompt=BUILDER_GUIDANCE,
                history_range=(),
                directive=directive,
                context=job_context,
            )
            self._append_job_history(session_dir, request)
            # The child agent runs build-pending through the memory CLI
            # (bash); the harness verifies the pending queue afterwards.
            await self.run_memory_job(request)
            pending_check = await cli.call(
                "freeze-pending",
                "--output",
                str(
                    session_dir
                    / "memory"
                    / "jobs"
                    / f"pending-check-{utc_now().replace(':', '-')}.json"
                ),
            )
            if int(pending_check.get("count") or 0) > 0:
                raise RuntimeError(
                    "builder fork finished with Pending UTs still unbuilt: "
                    f"{pending_check.get('count')}"
                )

    @staticmethod
    def _append_agent_history(
        session_dir: Path, role: str, value: dict[str, Any]
    ) -> None:
        append_jsonl(
            session_dir / "agent-history" / role / "conversation.jsonl",
            {"created_at": utc_now(), **value},
        )

    def _append_job_history(self, session_dir: Path, request: MemoryJobRequest) -> None:
        # Content-addressed logging: the directive embeds snapshots, conflict
        # UTs, and failure text; only its hash and shape are persisted here so
        # agent history cannot become a second copy of sensitive content.
        self._append_agent_history(
            session_dir,
            "extractor" if request.role == "extractor" else "builder",
            {
                "agent_alias": request.alias,
                "kind": "memory-job",
                "role": request.role,
                "label": request.label,
                "directive_hash": hashlib.sha256(
                    request.directive.encode("utf-8")
                ).hexdigest(),
                "directive_chars": len(request.directive),
                "history_message_count": len(request.history_range),
                "evidence_ref": request.context.evidence_ref,
                "covered_through": request.context.covered_through,
                "target_cursor": request.context.target_cursor,
            },
        )

    async def close(self, *, timeout: float = 30.0) -> None:
        """Stop accepting work and cancel in-flight jobs within ``timeout``.

        Cancellation is state-safe: publish/coverage state lives in the
        memory CLI's transactional store, harness files are written
        synchronously at await boundaries, and the next foreground turn
        re-triggers extraction from the persisted cursor — an interrupted
        job loses only its in-flight model/bash call. Service shutdown must
        not block for hours on a stuck background job.
        """

        self._accepting = False
        # Set before cancelling so worker loops observe the shutdown flag at
        # their next check instead of racing the cancellation.
        self._closed = True
        tasks = [
            task
            for task in (*self._workers.values(), *self._builders.values())
            if not task.done()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(tasks, timeout=timeout)
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.wait(tasks, timeout=5)
        for task in (*self._workers.values(), *self._builders.values()):
            if task.done() and not task.cancelled():
                task.exception()  # Consume isolated background failures.
        self._workers.clear()
        self._builders.clear()

    @staticmethod
    def _set_job_state(
        session_dir: Path, role: str, *, status: str, error: str | None
    ) -> None:
        path = session_dir / "state" / "eternal-harness.json"
        state = read_json(path, default={})
        _update_job_state(state, role, status=status, error=error)
        write_json(path, state)


def _update_job_state(
    state: dict[str, Any], role: str, *, status: str, error: str | None
) -> None:
    jobs = dict(state.get("memory_jobs") or {})
    job = dict(jobs.get(role) or {})
    job.update({"status": status, "updated_at": utc_now()})
    job.pop("error", None)
    if error is not None:
        job["error"] = error
    jobs[role] = job
    state["memory_jobs"] = jobs


def _extractor_directive(
    *,
    context: MemoryJobContext,
    snapshot: dict[str, Any],
    published_uts: list[dict[str, Any]],
    previous_failure: str | None,
) -> str:
    proposal_path = (
        context.memory_root
        / "jobs"
        / f"proposal-{context.covered_through + 1}-{context.target_cursor}.json"
    )
    cli_command = (
        f'"{sys.executable}" "{context.memory_script}" --root "{context.memory_root}"'
    )
    lines = [
        "You are now maintaining this conversation's memory (background memory job).",
        "The model-visible history above is the frozen Working Memory for "
        f"{context.evidence_ref} ({context.covered_through + 1}..{context.target_cursor}).",
        "",
        "Old Snapshot (JSON):",
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        "",
        "Conflict-reference UTs from the published memory (JSON; may be empty):",
        json.dumps(published_uts, ensure_ascii=False, indent=2),
        "",
        f"Session id: {context.session_id}",
        f"Coverage: from_cursor={context.covered_through + 1}, to_cursor={context.target_cursor}.",
        f'evidence_refs MUST be ["{context.evidence_ref}"].',
        "",
        "Memory CLI (run through bash):",
        f'  "{sys.executable}" "{context.memory_script}" --root "{context.memory_root}" get-state',
        f'  "{sys.executable}" "{context.memory_script}" --root "{context.memory_root}" publish-pending --file "{proposal_path}"',
        "",
        "Steps: run get-state to read base_memory_revision and base_snapshot_revision; "
        f'write the proposal JSON to "{proposal_path}"; run publish-pending; on validation '
        "errors, fix the proposal file and run publish-pending again.",
        "",
        EXTRACTOR_GUIDANCE,
    ]
    if previous_failure:
        lines.append(
            "The prior memory job published nothing. Correct this exact failure before "
            f"submitting: {previous_failure}"
        )
    return str(Prompt("\n".join(lines)).wrap("memory-extract-instruction"))


def _builder_directive(*, context: MemoryJobContext, batch: dict[str, Any]) -> str:
    assert context.build_batch_path is not None
    cli_command = (
        f'"{sys.executable}" "{context.memory_script}" --root "{context.memory_root}"'
    )
    lines = [
        "Frozen Pending UT batch (JSON):",
        json.dumps(batch, ensure_ascii=False, indent=2),
        "",
        "Memory CLI (run through bash):",
        f'  "{sys.executable}" "{context.memory_script}" --root "{context.memory_root}" build-pending --file "{context.build_batch_path}"',
        "",
        BUILDER_GUIDANCE,
    ]
    return str(Prompt("\n".join(lines)).wrap("memory-build-instruction"))


def _memory_query(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in reversed(messages):
        if message.get("role") not in {"user", "assistant"}:
            continue
        data = message.get("data")
        raw = data.get("raw_content") if isinstance(data, dict) else None
        content = str(raw or message.get("content") or "").strip()
        if content:
            parts.append(content[:800])
        if len(parts) >= 4:
            break
    return " ".join(reversed(parts))[:2400] or "recent conversation"


__all__ = [
    "EternalConversationHarness",
    "load_projection",
    "render_memory_access_instruction",
    "render_memory_snapshot",
]
