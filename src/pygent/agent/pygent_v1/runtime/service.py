from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import time
import uuid
import warnings
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from html import escape
from pathlib import Path
from typing import Any, Protocol, TypeAlias, cast

from pygent.agent.pygent_v1.agents import (
    AgentLimits,
    AgentRunner,
    ChildAgent,
    ChildSpec,
    ForkAgent,
    ForkSpec,
    ToolkitToolSurface,
    ToolProfile,
    ToolRegistry,
    build_model_layer_factory,
)
from pygent.agent.pygent_v1.core.io import (
    aclose_if_supported,
    plain_object,
    read_json,
    write_json,
)
from pygent.agent.pygent_v1.core.prompt import Prompt
from pygent.agent.pygent_v1.runtime.attachments import (
    ResolvedAttachment,
    build_media_blocks,
    render_attachments_xml,
    resolve_attachments,
)
from pygent.agent.pygent_v1.runtime.bash_tasks import BashTaskObservations
from pygent.agent.pygent_v1.runtime.mcp import ReadinessRetryingTransport
from pygent.agent.pygent_v1.runtime.reminders import ReminderService
from pygent.agent.pygent_v1.scheduler.tools import (
    SCHEDULER_TOOL_SPECS,
    SchedulerToolExecutor,
)
from pygent.agent.pygent_v1.schema import AgentSession, CaseRunRef, RunConfig
from pygent.agent.pygent_v1.sessions import SessionManager, realtime_enabled_for
from pygent.agent.pygent_v1.tracing import DIFF_TOOL_SPEC, DiffTool
from pygent.agent.react_projection_operations import (
    REACT_PROJECTION_OPERATION_KIND,
    ReplaceMessageProjection,
    encode_react_projection_operation,
)
from pygent.core import AIMessage, JsonValue, UserMessage, freeze_json_object
from pygent.llm import ModelGroup, ModelResourceRef
from pygent.runtime import (
    CapacityPolicy,
    CapacityScope,
    DurabilityMode,
    DurabilityPolicy,
    DurableToolTaskManager,
    ExecutionAdmissionError,
    ExecutionCapacityPolicy,
    ExecutionOptions,
    LocalRuntime,
    SQLiteCapacityCoordinator,
    SQLiteHistoryStore,
    SQLiteModelDeploymentStore,
)
from pygent.runtime.codec import invocation_from_dict, message_to_dict
from pygent.tool import (
    ExecutorRegistry,
    LocalToolExecutor,
    StandardTools,
    ToolKit,
    ToolSpec,
)
from pygent.tool.executors import ToolExecutionContext, ToolExecutor
from pygent.tool.mcp import MCPClientPool

from ..graph.common import DEFAULT_REACT_MAX_STEPS
from ..graph.core import (
    PYGENT_PROJECTION_READY_KIND,
    PygentAgent,
    model_retry_policy,
)
from ..graph.prompts import PromptRegistry
from .context import PYGENT_CONTEXT_CODEC, PygentContext
from .delivery_store import DeliveryStore, session_delivery_path
from .deployment import (
    PygentModelResourceResolver,
    WorkspaceToolExecutor,
)
from .eternal_conversation import (
    EternalConversationHarness,
    MemoryJobRequest,
    eternal_enabled_for,
)
from .file_effects import FILE_EFFECT_TOOL_SPEC, FileEffectToolExecutor
from .fork_context import (
    SessionForkWindowSource,
    eternal_conversation_messages,
    restore_session_context,
    session_fork_window,
)
from .model_configuration import preferred_models, preferred_profile_name
from .realtime.continuation import CONTINUATION_MESSAGE_KIND
from .realtime.envelope import (
    ExchangeEnvelope,
    build_exchange_envelope,
    format_user_asked,
    render_interrupt_attachments,
)
from .realtime.fork_agent import (
    FORK_MAX_MODEL_ATTEMPTS,
    ForkReplyModule,
    outcome_from_result,
)
from .realtime.fork_reasoning import ForkModelProfile, fork_model_profiles
from .realtime.fork_reply import build_fork_context
from .realtime.fork_state import SessionForkState
from .realtime.speak import (
    SILENT_ACKNOWLEDGEMENT,
    SPEAK_TOOL_SPEC,
    SpeakChannel,
    parse_speak_args,
    render_reply_instruction,
)
from .realtime.turn_finalizer import FINALIZER_MESSAGE_KIND

RECOVERY_CLAIM_WAIT_SECONDS = 31.0
# Pygent agent graphs declare requires_finite_deadline=True, so an unlimited
# deadline is rejected at admission; this ceiling is the practical maximum.
TURN_DEADLINE_SECONDS = 24 * 60 * 60
JOURNAL_PRUNE_INTERVAL_SECONDS = 60 * 60

# Tool names the kernel registers itself: the standard workspace tools plus
# every spec the service installs in __init__. Host tools and MCP servers
# must not shadow any of them; both registration paths check this one set.
RESERVED_TOOL_NAMES = frozenset(
    {
        "bash",
        "read",
        "write",
        "edit",
        "glob",
        "grep",
        DIFF_TOOL_SPEC.definition.name,
        FILE_EFFECT_TOOL_SPEC.definition.name,
        SPEAK_TOOL_SPEC.definition.name,
        *(spec.definition.name for spec in SCHEDULER_TOOL_SPECS),
    }
)

AgentDefinitionKey: TypeAlias = tuple[
    int, bool, str, str | None, bool, bool
]


class ModelBoundAgent(Protocol):
    """The model-facing surface shared by PygentAgent and ForkReplyModule."""

    config: RunConfig
    model_group_name: str
    llm: Any | None


# Non-durable (interruptible) turns resolve their context from this registry;
# durable turns keep reading the SQLite history record.
MAX_TRACKED_TURN_CONTEXTS = 256
PROJECTION_OPERATION_METADATA_KEY = "projection_replacement_operation"
PROJECTION_INPUT_ID_METADATA_KEY = "projection_replacement_input_id"
PROJECTION_READY_INPUT_ID_METADATA_KEY = "projection_replacement_ready_input_id"


def _config_fingerprint(config: RunConfig) -> str:
    # Imported lazily: pygent_v1.orchestration imports runtime.service at package load.
    from pygent.agent.pygent_v1.orchestration.runtime_keys import RuntimeGenerationKey

    return RuntimeGenerationKey.from_config(config).config_fingerprint


class _SpeakExecutor:
    """Emit foreground speak events and queue replies for configured channels."""

    DRAIN_TIMEOUT_SECONDS = 5.0

    def __init__(
        self, service: PygentRuntimeService, channels: Sequence[SpeakChannel] = ()
    ) -> None:
        self.service = service
        self.channels = tuple(channels)
        self._owner = uuid.uuid4().hex
        self._store = (
            DeliveryStore(session_delivery_path(SessionManager(service.config).sessions_root))
            if channels
            else None
        )
        self._recipients = {
            f"speak:{getattr(channel, 'recipient_id', f'{type(channel).__module__}.{type(channel).__qualname__}:{index}')}": channel
            for index, channel in enumerate(channels)
        }
        self._delivery_tasks: set[asyncio.Task[None]] = set()
        self._sending: set[tuple[str, str]] = set()
        self._poller: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._store is not None:
            self._poller = asyncio.create_task(self._poll(), name="pygent-speak-outbox")

    async def _poll(self) -> None:
        while True:
            try:
                assert self._store is not None
                for lane in self._recipients:
                    for row in self._store.pending(lane):
                        self._schedule(lane, row["key"])
            except Exception as exc:  # noqa: BLE001 - keep the outbox polling
                self._warn(f"speak outbox scan failed: {exc!r}")
            await asyncio.sleep(1)

    async def drain(self) -> None:
        if self._poller is not None:
            self._poller.cancel()
            await asyncio.gather(self._poller, return_exceptions=True)
        pending = set(self._delivery_tasks)
        if not pending:
            return
        _, still_pending = await asyncio.wait(
            pending, timeout=self.DRAIN_TIMEOUT_SECONDS
        )
        if still_pending:
            self._warn(
                f"{len(still_pending)} speak channel deliveries still pending at "
                "runtime close"
            )

    def publish(
        self, reply_id: str, text: str, attachments: Sequence[Path],
        *, exchange: ExchangeEnvelope | None = None,
    ) -> None:
        store = self._store
        if store is None and exchange is not None:
            store = DeliveryStore(
                session_delivery_path(SessionManager(self.service.config).sessions_root)
            )
        if store is None:
            return
        recipients = tuple(self._recipients) if text or attachments else ()
        payload = {"text": text, "attachments": [str(path) for path in attachments]}
        with store.transaction() as db:
            if exchange is not None:
                store.record_exchange(db, exchange)
            for lane in recipients:
                store.put(db, lane, reply_id, payload)
        for lane in recipients:
            self._schedule(lane, reply_id)

    def _schedule(self, lane: str, reply_id: str) -> None:
        key = (lane, reply_id)
        if key in self._sending:
            return
        self._sending.add(key)
        task = asyncio.create_task(self._send(lane, reply_id))
        self._delivery_tasks.add(task)
        task.add_done_callback(self._delivery_tasks.discard)

    async def _send(self, lane: str, reply_id: str) -> None:
        store = self._store
        assert store is not None
        channel = self._recipients[lane]
        name = type(channel).__name__
        try:
            row = store.claim(lane, reply_id, self._owner, 60)
            if row is None:
                return
            payload = json.loads(row["payload"])
            paths = tuple(Path(path) for path in payload["attachments"])
            send_with_id = getattr(channel, "deliver_with_id", None)
            send = (
                asyncio.to_thread(channel.deliver, payload["text"], paths)
                if send_with_id is None
                else asyncio.to_thread(send_with_id, reply_id, payload["text"], paths)
            )
            attempt = asyncio.create_task(send)
            while not attempt.done():
                done, _ = await asyncio.wait({attempt}, timeout=15)
                if not done:
                    store.renew(lane, reply_id, self._owner, 60)
            if attempt.result() is True:
                store.finish(lane, reply_id, {"delivered": True})
            else:
                store.retry(lane, reply_id, self._owner, delay=5)
                self._warn(f"speak channel {name} declined the message")
        except Exception as exc:  # noqa: BLE001 - host channel failure
            store.retry(lane, reply_id, self._owner, delay=5)
            self._warn(f"speak channel {name} failed: {exc!r}")
        finally:
            self._sending.discard((lane, reply_id))

    def _warn(self, message: str) -> None:
        self.service.warnings.append(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    async def execute(
        self, spec: ToolSpec, call: Any, context: ToolExecutionContext
    ) -> object:
        del spec
        args = parse_speak_args(plain_object(call.arguments))
        if args.say_nothing:
            return SILENT_ACKNOWLEDGEMENT
        if context.emit is None:  # pragma: no cover - managed executions carry emit
            raise RuntimeError("speak delivery requires an execution event emitter")
        text = args.text or ""
        attachments: tuple[ResolvedAttachment, ...] = ()
        if args.attachments:
            attachments = resolve_attachments(
                args.attachments, self.service.config.workspace_root
            )
        speak_event: Mapping[str, Any] = {
            "text": text,
            "attachments": [item.relative for item in attachments],
        }
        call_id = getattr(call, "call_id", None)
        reply_id = (
            f"speak:{context.execution_id}:{call_id}"
            if context.execution_id and call_id
            else uuid.uuid4().hex
        )
        self.publish(reply_id, text, tuple(item.path for item in attachments))
        await context.emit("pygent.v1.realtime.speak", speak_event)
        return "delivered"


class _DiffExecutor:
    def __init__(self, service: PygentRuntimeService) -> None:
        self.service = service

    async def execute(
        self, spec: ToolSpec, call: Any, context: ToolExecutionContext
    ) -> object:
        del spec
        if context.execution_id is None:
            raise RuntimeError("managed diff execution requires an execution id")
        execution_context = await self.service._turn_context(context.execution_id)
        diff = DiffTool(
            case_run_ref=execution_context.case_run_ref,
            workspace_root=self.service.config.workspace_root,
            turn_id=execution_context.turn_id,
        )
        return await diff.forward(**plain_object(call.arguments))


MEMORY_JOB_LIMITS = AgentLimits(
    max_steps=DEFAULT_REACT_MAX_STEPS,
    max_model_calls=DEFAULT_REACT_MAX_STEPS,
    max_tool_calls=DEFAULT_REACT_MAX_STEPS,
)


class PygentRuntimeService:
    """Workspace-scoped owner of Pygent execution and deployment resources."""

    def __init__(
        self,
        config: RunConfig,
        *,
        max_live_executions: int = 32,
        max_runnable_executions: int = 4,
        max_queue_size: int = 64,
        model_max_concurrency: int = 4,
        tool_max_concurrency: int = 8,
        reminders: ReminderService | None = None,
        own_reminders: bool | None = None,
        credential_environ: Mapping[str, str] | None = None,
        run_config_loader: Callable[..., RunConfig] | None = None,
        prompt_registry: PromptRegistry | None = None,
        on_exchange_ready: Callable[[ExchangeEnvelope], Awaitable[None]] | None = None,
        speak_channels: Sequence[SpeakChannel] = (),
        external_tools: Iterable[tuple[ToolSpec, ToolExecutor]] = (),
    ) -> None:
        """Compose the runtime for one workspace.

        ``speak_channels`` receive every delivered speak text in addition to
        the stream event; ``external_tools`` are host-provided tool/executor
        pairs that join the scheduler tools in the model-visible tool set from
        the first turn. The kernel ships neither by default.
        """
        self.config = config
        self._credential_environ = credential_environ
        self._run_config_loader = run_config_loader
        self._prompt_registry = prompt_registry
        # Receives each fork reply's exchange envelope. The host owns the
        # session's turn chain, so only it can inject into the running turn or
        # hold the envelope for the next turn's opening context.
        self._on_exchange_ready = on_exchange_ready
        self.reminders = reminders or ReminderService(config)
        self._owns_reminders = (
            reminders is None if own_reminders is None else own_reminders
        )
        history_path = Path(config.runtime_durability.history_path)
        self.history_path = history_path
        model_path = history_path.with_name(
            f"model-deployments-v2-{_config_fingerprint(config)[:16]}.sqlite3"
        )
        self.model_path = model_path
        self.history = SQLiteHistoryStore(history_path)
        self.model_store = SQLiteModelDeploymentStore(model_path)
        self.capacity = (
            SQLiteCapacityCoordinator(config.runtime_capacity.coordinator_path)
            if config.runtime_capacity.scope == "deployment"
            else None
        )
        self.executor_registry = ExecutorRegistry()
        self.runtime = LocalRuntime(
            history=self.history,
            capacity_coordinator=self.capacity,
            model_deployment_store=self.model_store,
            deployment_namespace=str(Path(config.workspace_root).resolve()),
            journal_retention_seconds=config.runtime_durability.retention_seconds,
            context_codecs=(PYGENT_CONTEXT_CODEC,),
        )
        self.runtime.attach_executor_registry(self.executor_registry)
        self._diff_executor = _DiffExecutor(self)
        standard = StandardTools(
            workspace_root=config.workspace_root, restrict_to_workspace=False
        )
        self._standard_tools = standard
        # Memory jobs (fork extractor / child builder) share a surface with the
        # exact same tool composition as the foreground, per the owner's decision
        # to open them up before the production tool pipeline lands.
        self._memory_job_tools = StandardTools(
            workspace_root=config.workspace_root,
            restrict_to_workspace=False,
        )
        ToolKit(
            standard.bash.bash,
            standard.bash.tool_task_get,
            standard.bash.tool_task_stop,
            standard.files.read,
            standard.files.write,
            standard.files.edit,
            standard.files.glob,
            standard.files.grep,
        ).register_into_runtime(
            self.runtime,
            executor_factory=lambda spec, handler: (
                WorkspaceToolExecutor(handler)
                if spec.sandbox_profile == "workspace"
                else LocalToolExecutor(handler)
            ),
        )
        self.runtime.register_tool(FILE_EFFECT_TOOL_SPEC, FileEffectToolExecutor())
        self.runtime.register_tool(DIFF_TOOL_SPEC, self._diff_executor)
        self._speak_executor = _SpeakExecutor(
            self,
            # The speak channel slot stays generic; delivery channels are
            # host-provided integrations (e.g. a standalone Feishu
            # package) and the kernel ships none by default.
            channels=speak_channels,
        )
        self.runtime.register_tool(SPEAK_TOOL_SPEC, self._speak_executor)
        scheduler_executor = SchedulerToolExecutor(config)
        for spec in SCHEDULER_TOOL_SPECS:
            self.runtime.register_tool(spec, scheduler_executor)
        host_tools = tuple(external_tools)
        visible_names = set(RESERVED_TOOL_NAMES)
        for spec, _executor in host_tools:
            name = spec.definition.name
            if name in visible_names:
                raise ValueError(f"duplicate host model-visible tool name: {name}")
            visible_names.add(name)
        self.runtime.register_tools(host_tools)
        self.bash_tasks = BashTaskObservations(
            config,
            self.runtime,
            history_path.with_suffix(".bash-observations"),
        )
        self.task_manager = DurableToolTaskManager(
            self.history,
            self.executor_registry,
            emit=self._on_tool_task_event,
        )
        self.runtime.attach_tool_task_manager(self.task_manager)
        self.model_resolver = PygentModelResourceResolver()
        self.runtime.register_model_resource_resolver(self.model_resolver)
        scope = (
            CapacityScope.DEPLOYMENT
            if config.runtime_capacity.scope == "deployment"
            else CapacityScope.RUNTIME_INSTANCE
        )
        self.binding = self.runtime.create_binding(
            name="pygent.v1-agent",
            execution_capacity=ExecutionCapacityPolicy(
                scope=scope,
                max_live_executions=max_live_executions,
                max_runnable_executions=max_runnable_executions,
                max_queue_size=max_queue_size,
                max_waiters=max_live_executions + max_queue_size,
                max_child_depth=max(8, config.delegation.max_depth + 2),
                max_children_per_execution=max(
                    4096, (config.max_steps if config.max_steps > 0 else 128) * 64
                ),
                max_external_wait_seconds=config.runtime_approvals.timeout_seconds,
            ),
            model_capacity=CapacityPolicy.limited(
                max_concurrency=model_max_concurrency,
                max_queue_size=max_queue_size,
                capacity_key="pygent.v1-api-model",
                scope=scope,
            ),
            tool_capacity=CapacityPolicy.limited(
                max_concurrency=tool_max_concurrency,
                max_queue_size=max_queue_size,
                capacity_key="pygent.v1-api-tool",
                scope=scope,
            ),
            durability=DurabilityPolicy(DurabilityMode(config.runtime_durability.mode)),
        )
        self.external_tools: tuple[ToolSpec, ...] = (
            *SCHEDULER_TOOL_SPECS,
            *(spec for spec, _executor in host_tools),
        )
        self._turn_contexts: dict[str, PygentContext] = {}
        self._fork_agents: dict[tuple, ForkReplyModule] = {}
        self._fork_states: dict[str, SessionForkState] = {}
        self.warnings: list[str] = []
        self._initialized = False
        self._initializing_task: asyncio.Task[None] | None = None
        self._journal_prune_task: asyncio.Task[None] | None = None
        self._closed = False
        self._mcp_pools: list[MCPClientPool] = []
        self.memory_harness = EternalConversationHarness(
            config.eternal_conversation,
            run_memory_job=self._run_memory_job,
            user_clis_root=Path(config.user_pygent_root) / "clis",
            foreground_system_prompt=config.system_prompt,
        )
        template = (
            PygentAgent(
                config,
                prompt_registry=self._prompt_registry,
                external_tools=self.external_tools,
                managed_model=True,
                model_group_name=config.resolved_agent.default_model_group,
                memory_harness=self.memory_harness,
                reminders=self.reminders,
                bash_tasks=self.bash_tasks,
                credential_environ=self._credential_environ,
            )
            if config.resolved_agent is not None and config.model_config is not None
            else None
        )
        self._agent_definitions: dict[AgentDefinitionKey, PygentAgent] = (
            {
                (
                    id(config),
                    False,
                    config.resolved_agent.default_model_group,
                    None,
                    config.eternal_conversation.enabled,
                    config.realtime_interaction.enabled,
                ): template
            }
            if template is not None and config.resolved_agent is not None
            else {}
        )
        self._model_invokers: dict[str, Any] = (
            {config.resolved_agent.alias: template.llm}
            if template is not None and config.resolved_agent is not None
            else {}
        )

    def _track_turn_context(self, execution_id: str, context: PygentContext) -> None:
        """Remember one execution context for non-durable (interruptible) turns.

        Durable turns resolve their context from the SQLite history record;
        interruptible turns have no history row, so tool executors read the
        context captured at start here.
        """

        self._turn_contexts[execution_id] = context
        while len(self._turn_contexts) > MAX_TRACKED_TURN_CONTEXTS:
            self._turn_contexts.pop(next(iter(self._turn_contexts)), None)

    async def _turn_context(self, execution_id: str) -> PygentContext:
        tracked = self._turn_contexts.get(execution_id)
        if tracked is not None:
            return tracked
        record = await self.history.get_execution(execution_id)
        if record is None:
            raise RuntimeError(f"execution {execution_id!r} has no resolvable context")
        _, decoded = invocation_from_dict(
            record.input,
            registry=self.runtime.context_codec_registry,
        )
        if not isinstance(decoded, PygentContext):
            raise TypeError("execution context is not a PygentContext")
        return decoded

    async def _on_tool_task_event(self, kind: str, data: Mapping[str, Any]) -> None:
        # Delegate through self.bash_tasks so a replaced observation instance
        # (restore path) is still the event target.
        await self.bash_tasks.on_tool_task_event(kind, data)

    async def initialize(self) -> None:
        if self._closed:
            raise RuntimeError("PygentRuntimeService is closed")
        if self._initialized:
            return
        if self._initializing_task is None:
            self._initializing_task = asyncio.create_task(self._initialize_once())
        try:
            await asyncio.shield(self._initializing_task)
        except BaseException:
            if self._initializing_task.done():
                self._initializing_task = None
            raise

    async def _initialize_once(self) -> None:
        for path in (
            self.history_path,
            Path(self.config.runtime_capacity.coordinator_path),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
        await self.history.open()
        self._speak_executor.start()
        self._journal_prune_task = asyncio.create_task(
            self._prune_journal_periodically(), name="pygent-journal-prune"
        )
        discovered = list(self.external_tools)
        visible_names: set[str] = {
            *RESERVED_TOOL_NAMES,
            *(spec.definition.name for spec in discovered),
        }
        for server in self.config.mcp_servers:
            try:
                pool = await self._mcp_pool(server)
                entries = pool.executors()
                # The pool discovers without a call timeout; the server config
                # timeout belongs on every portable spec for Runtime dispatch.
                specs = tuple(
                    replace(spec, timeout=server.timeout) for spec, _executor in entries
                )
                for spec in specs:
                    name = spec.definition.name
                    if name in visible_names:
                        raise ValueError(
                            f"duplicate MCP model-visible tool name: {name}"
                        )
                    visible_names.add(name)
                for (_discovered_spec, executor), spec in zip(entries, specs):
                    self.executor_registry.register(
                        spec.tool_id, spec.version, executor
                    )
                discovered.extend(specs)
            except Exception as exc:
                if server.required:
                    raise RuntimeError(
                        f"required MCP server {server.name!r} failed: {exc}"
                    ) from exc
                message = f"optional MCP server {server.name!r} skipped: {exc}"
                self.warnings.append(message)
                warnings.warn(message, RuntimeWarning, stacklevel=2)
        self.external_tools = tuple(discovered)
        await self.bash_tasks.restore()
        self._initialized = True

    async def _prune_journal_periodically(self) -> None:
        while True:
            try:
                await self.runtime.prune()
            except Exception as exc:  # noqa: BLE001 - pruning never takes a turn down
                message = f"execution journal prune failed: {exc}"
                self.warnings.append(message)
                warnings.warn(message, RuntimeWarning, stacklevel=2)
            await asyncio.sleep(JOURNAL_PRUNE_INTERVAL_SECONDS)

    async def _mcp_pool(self, server: Any) -> MCPClientPool:
        if server.transport != "stdio":
            raise ValueError(
                f"MCP transport {server.transport!r} is unsupported by the installed Pygent; "
                "use stdio"
            )
        env = {name: os.environ[name] for name in server.env_from if name in os.environ}
        # A cold transport re-spawns the server per call and loses the readiness
        # race against servers that connect to their backends asynchronously.
        # The pool holds one session per server; the retrying transport keeps
        # the pre-action readiness workaround on top of that shared session.
        transport = ReadinessRetryingTransport(
            server.command,
            args=server.args,
            env=env,
            cwd=server.cwd,
        )
        pool = MCPClientPool({server.name: transport}, version="1")
        await pool.connect()
        self._mcp_pools.append(pool)
        return pool

    def new_agent(
        self,
        *,
        interactive_approvals: bool,
        model_group_name: str | None = None,
        config: RunConfig | None = None,
        selected_model_key: str | None = None,
        eternal_memory_enabled: bool | None = None,
        realtime_interaction_enabled: bool | None = None,
    ) -> PygentAgent:
        effective_config = config or self.config
        resolved = effective_config.resolved_agent
        if resolved is None:
            raise ValueError("resolved agent configuration is required")
        selected_group = model_group_name or resolved.default_model_group
        # Eternal sessions disable context compression at agent construction,
        # so the flag must split the shared agent templates per session opt-in.
        eternal_memory = (
            effective_config.eternal_conversation.enabled
            if eternal_memory_enabled is None
            else eternal_memory_enabled
        )
        realtime_interaction = (
            effective_config.realtime_interaction.enabled
            if realtime_interaction_enabled is None
            else realtime_interaction_enabled
        )
        # The selected model decides the layer's media output modalities, so
        # agents must not be shared across different session selections.
        key = (
            id(effective_config),
            interactive_approvals,
            selected_group,
            selected_model_key,
            eternal_memory,
            realtime_interaction,
        )
        existing = self._agent_definitions.get(key)
        if existing is not None:
            return existing
        agent = PygentAgent(
            effective_config,
            resolved_agent=effective_config.resolved_agent,
            prompt_registry=self._prompt_registry,
            external_tools=self.external_tools,
            managed_model=True,
            model_group_name=selected_group,
            interactive_approvals=interactive_approvals,
            model_invoker=self._model_invokers.get(resolved.alias),
            memory_harness=self.memory_harness,
            reminders=self.reminders,
            bash_tasks=self.bash_tasks,
            selected_model_key=selected_model_key,
            eternal_memory_enabled=eternal_memory,
            credential_environ=self._credential_environ,
            realtime_interaction_enabled=realtime_interaction,
        )
        if agent.llm is not None:
            self._model_invokers.setdefault(resolved.alias, agent.llm)
        self._agent_definitions[key] = agent
        return agent

    def _child_agent_config(self, alias: str) -> RunConfig:
        if alias == self.config.agent_alias:
            # Round-trip through the host's own RunConfig type: a subclass may
            # carry extension fields the base deserializer would reject.
            return type(self.config).from_dict(self.config.to_dict())
        if self._run_config_loader is None:
            raise RuntimeError(
                "child agent configuration requires an injected run-config loader"
            )
        return self._run_config_loader(
            workspace_root=self.config.workspace_root,
            agent_alias=alias,
            max_steps=self.config.max_steps,
            context_window=self.config.context_window,
        )

    async def bind(
        self,
        module: Any,
        agent: ModelBoundAgent,
        *,
        fork_profiles: Mapping[str, ForkModelProfile] | None = None,
    ) -> Any:
        await self.initialize()
        bound = self.binding.bind(module)
        if agent.llm is not None and (
            module is agent or getattr(module, "agent", None) is agent
        ):
            requirement = ModelGroup.deferred(name=f"pygent.v1:{agent.model_group_name}")
            handle = bound.model_groups.get(requirement)
            native = agent.config.model_config
            if native is None:
                raise RuntimeError("model_configuration_required")
            group = native.model_groups[agent.model_group_name]
            revision = hashlib.sha256(
                repr(
                    (
                        agent.model_group_name,
                        group.models,
                        tuple(native.connections.items()),
                    )
                ).encode("utf-8")
            ).hexdigest()
            self.model_resolver.register(revision, agent.llm)
            for index, entry in enumerate(group.models):
                fork_profile = (
                    None if fork_profiles is None else fork_profiles.get(entry.key)
                )
                await handle.ensure_profile(
                    profile=(
                        fork_profile.profile
                        if fork_profile is not None
                        else preferred_profile_name(entry.key)
                    ),
                    models=(
                        fork_profile.models
                        if fork_profile is not None
                        else preferred_models(
                            native, agent.model_group_name, entry.key
                        )
                    ),
                    invoker=agent.llm,
                    resource_ref=ModelResourceRef(
                        resolver_id=self.model_resolver.resolver_id,
                        resource_id=f"pygent.v1:{agent.model_group_name}",
                        revision=revision,
                        capacity_owner_id=f"pygent.v1:{Path(self.config.workspace_root).resolve()}",
                        coordinator_domain=str(
                            Path(self.config.workspace_root).resolve()
                        ),
                    ),
                    # Fork profiles are selected per execution; publishing one
                    # as the group default would change the main turn's fallback.
                    make_default=fork_profile is None and index == 0,
                    deadline=time.monotonic() + 60,
                )
        return bound

    @staticmethod
    async def _deliver_projection_replacement(
        handle: Any, context: PygentContext
    ) -> None:
        operation = context.metadata.get(PROJECTION_OPERATION_METADATA_KEY)
        if operation is None:
            return
        for key, kind, value in (
            (
                PROJECTION_INPUT_ID_METADATA_KEY,
                REACT_PROJECTION_OPERATION_KIND,
                operation,
            ),
            (PROJECTION_READY_INPUT_ID_METADATA_KEY, PYGENT_PROJECTION_READY_KIND, True),
        ):
            result = await handle.send_input(
                input_id=str(context.metadata[key]), kind=kind, value=value
            )
            if result.status not in {"accepted", "duplicate"}:
                raise RuntimeError(
                    f"native projection input was not delivered: {result.status}"
                )

    async def _start_agent_execution(
        self,
        bound: Any,
        message: UserMessage,
        context: PygentContext,
        *,
        execution: Any,
        wait_for_invocation_commit: bool = False,
    ) -> Any:
        handle = await bound.start(message, context, execution=execution)
        try:
            await self._deliver_projection_replacement(handle, context)
            if wait_for_invocation_commit:
                await handle.wait_for_invocation_commit()
        except BaseException:
            await handle.cancel()
            raise
        return handle

    def _model_calls_option(
        self,
        group_name: str,
        selected_model_key: str | None,
        *,
        profile_name: str | None = None,
    ) -> dict[str, dict[str, str]]:
        """The ExecutionOptions ``model_calls`` entry for one group/profile."""

        if profile_name is None:
            if selected_model_key is None:
                raise ValueError("model_key must be a non-empty string")
            profile_name = preferred_profile_name(selected_model_key)
        return {
            f"pygent.v1:{group_name}": {
                "profile": profile_name,
            }
        }

    async def start_turn(
        self,
        *,
        manager: SessionManager,
        message: str,
        run_ref: CaseRunRef,
        turn_id: str,
        interactive_approvals: bool,
        message_kind: str = "pygent.v1.chat.turn",
        message_data: dict[str, object] | None = None,
        attachments: Sequence[str] = (),
        opening_exchanges: Sequence[ExchangeEnvelope] = (),
        deadline: float | None = None,
    ) -> Any:
        await self.initialize()
        group_name, selected_model_key = manager.model_selection(run_ref.session_id)
        session = manager.load(run_ref.session_id)
        realtime_enabled = realtime_enabled_for(session)
        agent = self.new_agent(
            interactive_approvals=interactive_approvals,
            model_group_name=group_name,
            selected_model_key=selected_model_key,
            eternal_memory_enabled=eternal_enabled_for(session, self.config),
            realtime_interaction_enabled=realtime_enabled,
        )
        if opening_exchanges:
            message_data = {
                **(message_data or {}),
                "opening_exchange_ids": [item.exchange_id for item in opening_exchanges],
            }
        try:
            turn_message, turn_context = await self._prepare_turn(
                manager=manager,
                run_ref=run_ref,
                message=message,
                config=self.config,
                turn_id=turn_id,
                message_kind=message_kind,
                message_data=message_data,
                attachments=attachments,
                exchange_context=(
                    "\n".join(item.content for item in opening_exchanges) or None
                ),
            )
            bound = await self.bind(agent, agent)
            handle = await self._start_agent_execution(
                bound,
                turn_message,
                turn_context,
                execution=ExecutionOptions(
                    execution_id=cast(str | None, (message_data or {}).get("delivery_execution_id")),
                    request_id=run_ref.case_run_id,
                    idempotency_key=run_ref.case_run_id,
                    identity=run_ref.session_id,
                    model_calls=(
                        self._model_calls_option(group_name, selected_model_key)
                        if agent.llm is not None
                        else {}
                    ),
                    deadline=deadline
                    if deadline is not None
                    else time.monotonic() + TURN_DEADLINE_SECONDS,
                ),
                wait_for_invocation_commit=bool(
                    opening_exchanges or (message_data or {}).get("delivery_execution_id")
                ),
            )
        except BaseException:
            await self.reminders.release_initial(run_ref.session_id, turn_id)
            raise
        await self.reminders.acknowledge_initial(
            run_ref.session_id, turn_id, handle.execution_id
        )
        self._record_execution_id(run_ref, handle.execution_id)
        self._track_turn_context(handle.execution_id, turn_context)
        return handle

    def owns_interrupt(self, session_id: str, input_id: str) -> bool:
        state = self._fork_states.get(session_id)
        return state is not None and input_id in state.input_ids

    async def send_interrupt(
        self,
        *,
        manager: SessionManager,
        session_id: str,
        input_id: str,
        message: str,
        case_id: str = "chat",
        attachments: Sequence[str] = (),
    ) -> Any:
        """Fork-reply path for a user message arriving mid-turn (D-RT2/3).

        A session-level :class:`SessionForkState` supersedes an in-flight
        fork when another interrupt arrives: the old handle is cancelled, the
        messages are merged into a numbered batch, and a new generation starts
        with a fresh deadline. A late result from an older generation is
        discarded before it can queue an exchange envelope.

        Attachments resolve under the same rules as a turn's: the fork sees
        images and videos natively and other files as references, and the
        exchange envelope lists them so the main agent can inspect them.
        """

        await self.initialize()
        config = self.config
        session = manager.load(session_id)
        if not realtime_enabled_for(session):
            raise RuntimeError("realtime interaction is not enabled for this session")
        if self._on_exchange_ready is None:
            raise RuntimeError(
                "realtime interrupts require an exchange delivery hook; "
                "run the service through an execution host"
            )
        resolved_attachments: tuple[ResolvedAttachment, ...] = ()
        try:
            if attachments:
                resolved_attachments = resolve_attachments(
                    attachments, config.workspace_root
                )
        except BaseException:
            # The fork never started; the bare question still reaches the
            # main agent, so an accepted interrupt is never dropped silently.
            await self._deliver_unanswered_exchange(
                session_id=session_id,
                input_id=input_id,
                user_asked=(message,),
                source_input_ids=(input_id,),
            )
            raise
        run_ref = manager.start_case_run(session_id, case_id)
        state = self._fork_states.setdefault(session_id, SessionForkState())
        async with state.lock:
            if state.handle is not None and not state.finished:
                state.pending.append(message)
                state.input_ids.append(input_id)
                state.merge_attachments(resolved_attachments)
                state.generation += 1
                generation = state.generation
                fork_messages = list(state.pending)
                await state.handle.cancel()
                fork_input_id = f"{input_id}:g{generation}"
                request_message = format_user_asked(fork_messages)
            else:
                was_idle = state.handle is None
                state.generation += 1
                generation = state.generation
                state.pending = [message]
                state.input_ids = [input_id]
                state.attachments = []
                state.merge_attachments(resolved_attachments)
                fork_messages = [message]
                fork_input_id = (
                    input_id
                    if was_idle and generation == 1
                    else f"{input_id}:g{generation}"
                )
                request_message = message
            source_input_ids = tuple(state.input_ids)
            fork_attachments = tuple(state.attachments)
            state.finished = False
            try:
                handle = await self._start_fork_reply(
                    session=session,
                    run_ref=run_ref,
                    config=config,
                    input_id=fork_input_id,
                    message=request_message,
                    attachments=fork_attachments,
                )
            except BaseException:
                # The fork never started: the queued generation drops and the
                # bare question still reaches the main agent.
                state.reset()
                await self._deliver_unanswered_exchange(
                    session_id=session_id,
                    input_id=fork_input_id,
                    user_asked=fork_messages,
                    source_input_ids=source_input_ids,
                    attachments=fork_attachments,
                )
                raise
            state.handle = handle

        try:
            output, _ = await handle.result()
        except asyncio.CancelledError:
            output = None
        except BaseException:
            async with state.lock:
                if state.generation == generation:
                    # Total fork failure: the bare question still reaches the
                    # main agent. A superseded generation stays silent — its
                    # messages ride the replacing generation's envelope.
                    state.reset()
                    await self._deliver_unanswered_exchange(
                        session_id=session_id,
                        input_id=fork_input_id,
                        user_asked=fork_messages,
                        source_input_ids=source_input_ids,
                        attachments=fork_attachments,
                    )
            raise
        if output is None:
            return None
        assert isinstance(output, AIMessage)
        outcome = outcome_from_result(output)
        speak = outcome.speak_args
        silent = speak is not None and speak.say_nothing
        text = "" if silent else (speak.text if speak is not None else outcome.text) or ""
        response = "(decided not to speak)" if silent else text or "(no verbal reply)"
        async with state.lock:
            if state.generation != generation:
                return None
            state.finished = True
            envelope = build_exchange_envelope(
                session_id=session_id,
                input_id=fork_input_id,
                user_asked=tuple(fork_messages),
                response=response,
                attachments=fork_attachments,
            )
            envelope = replace(
                envelope,
                source_input_ids=source_input_ids,
                reply=cast(JsonValue, message_to_dict(output)),
            )
            state.reset()
        paths: list[Path] = []
        for attachment in speak.attachments if speak is not None else ():
            try:
                paths.extend(
                    item.path
                    for item in resolve_attachments((attachment,), config.workspace_root)
                )
            except (OSError, ValueError):
                self._speak_executor._warn("fork speak attachment could not be resolved")
        self._speak_executor.publish(
            f"{session_id}:{envelope.exchange_id}", text, paths, exchange=envelope,
        )
        await self._on_exchange_ready(envelope)
        return outcome

    async def _deliver_unanswered_exchange(
        self,
        *,
        session_id: str,
        input_id: str,
        user_asked: Sequence[str],
        source_input_ids: tuple[str, ...] = (),
        attachments: Sequence[ResolvedAttachment] = (),
    ) -> None:
        """Forward the unanswered question without masking the fork failure."""

        hook = self._on_exchange_ready
        if hook is None:  # unreachable behind send_interrupt's guard
            return
        envelope = build_exchange_envelope(
            session_id=session_id,
            input_id=input_id,
            user_asked=user_asked,
            response=None,
            attachments=attachments,
        )
        # Delivery failure must not mask the fork failure — best effort.
        with contextlib.suppress(Exception):
            await hook(replace(envelope, source_input_ids=source_input_ids))
    async def _start_fork_reply(
        self,
        *,
        session: AgentSession,
        run_ref: CaseRunRef,
        config: RunConfig,
        input_id: str,
        message: str,
        attachments: Sequence[ResolvedAttachment] = (),
    ) -> Any:
        main_agent = self.new_agent(
            interactive_approvals=False,
            model_group_name=session.model_group_name,
            selected_model_key=session.selected_model_key,
            eternal_memory_enabled=eternal_enabled_for(session, config),
            realtime_interaction_enabled=True,
        )
        fork_window = session_fork_window(
            mode="live",
            config=config,
            session=session,
            run_ref=run_ref,
            turn_id=f"fork:{input_id}",
        )
        request = build_fork_context(
            system_prompt=fork_window.system_prompt,
            messages=fork_window.messages,
            tools=main_agent.tool_definitions,
        )
        fork_agent = self._fork_agent(
            model_group_name=session.model_group_name,
            selected_model_key=session.selected_model_key,
            tools=main_agent.tool_definitions,
        )
        resolved_agent = config.resolved_agent
        if resolved_agent is None:
            raise ValueError("resolved agent configuration is required")
        group_name = (
            session.model_group_name or resolved_agent.default_model_group
        )
        fork_profiles = (
            fork_model_profiles(
                config.model_config,
                group_name,
                config.realtime_interaction.fork_reasoning_effort,
            )
            if config.model_config is not None
            else {}
        )
        bound = await self.bind(
            fork_agent, fork_agent, fork_profiles=fork_profiles
        )
        selected_fork_profile = (
            fork_profiles.get(session.selected_model_key)
            if session.selected_model_key is not None
            else None
        )
        model_calls = (
            self._model_calls_option(
                group_name,
                session.selected_model_key,
                profile_name=(
                    None
                    if selected_fork_profile is None
                    else selected_fork_profile.profile
                ),
            )
            if main_agent.llm is not None
            else {}
        )
        fork_content = message
        fork_data: dict[str, object] = {}
        if attachments:
            fork_content = "\n".join(
                (message, render_interrupt_attachments(attachments))
            )
            fork_data["attachments"] = [item.relative for item in attachments]
        return await bound.start(
            UserMessage(
                content=fork_content,
                kind="pygent.v1.user.interrupt",
                data=freeze_json_object(fork_data),
                media=build_media_blocks(attachments),
            ),
            request,
            execution=ExecutionOptions(
                request_id=f"fork:{input_id}",
                idempotency_key=f"fork:{input_id}",
                identity=run_ref.session_id,
                model_calls=model_calls,
                deadline=time.monotonic()
                + config.realtime_interaction.fork_deadline_seconds
                * FORK_MAX_MODEL_ATTEMPTS,
            ),
        )

    def _fork_agent(
        self,
        *,
        model_group_name: str | None,
        selected_model_key: str | None,
        tools: tuple,
    ) -> ForkReplyModule:
        """One cached fork-reply graph per configuration shape."""

        resolved_agent = self.config.resolved_agent
        if resolved_agent is None:
            raise ValueError("resolved agent configuration is required")
        group_name = model_group_name or resolved_agent.default_model_group
        key = (id(self.config), group_name, selected_model_key, tools)
        existing = self._fork_agents.get(key)
        if existing is not None:
            return existing
        invoker = self._model_invokers.get(resolved_agent.alias)
        retry = self.config.model_retry
        agent = ForkReplyModule(
            config=self.config,
            model_group_name=group_name,
            llm=invoker,
            tools=tools,
            retry_policy=model_retry_policy(retry),
            model_call_timeout=self.config.realtime_interaction.fork_deadline_seconds,
        )
        self._fork_agents[key] = agent
        return agent

    async def recovery_case_run(self, execution_id: str) -> CaseRunRef:
        stored = await self.history.get_execution(execution_id)
        if stored is None:
            raise KeyError(execution_id)
        _, context = invocation_from_dict(
            stored.input, registry=self.runtime.context_codec_registry
        )
        if not isinstance(context, PygentContext):
            raise TypeError("durable execution context is not a PygentContext")
        return context.case_run_ref

    async def recover_turn(
        self,
        execution_id: str,
        *,
        deadline: float | None = None,
    ) -> Any:
        """Claim and resume one non-terminal durable Pygent execution."""

        await self.initialize()
        if deadline is None:
            deadline = time.monotonic() + TURN_DEADLINE_SECONDS
        stored = await self.history.get_execution(execution_id)
        if stored is None:
            raise KeyError(f"unknown durable execution {execution_id!r}")
        recovered_input, recovered_context = invocation_from_dict(
            stored.input,
            registry=self.runtime.context_codec_registry,
        )
        if not isinstance(recovered_context, PygentContext):
            raise TypeError("durable execution context is not a PygentContext")
        session_manager = SessionManager(self.config)
        group_name, selected_model_key = session_manager.model_selection(
            recovered_context.session_id
        )
        recovered_session = session_manager.load(recovered_context.session_id)
        agent = self.new_agent(
            interactive_approvals=bool(plain_object(recovered_input.data).get("interactive_approvals", True)),
            model_group_name=group_name,
            selected_model_key=selected_model_key,
            eternal_memory_enabled=eternal_enabled_for(
                recovered_session, self.config
            ),
            realtime_interaction_enabled=realtime_enabled_for(recovered_session),
        )
        bound = await self.bind(agent, agent)
        await self.runtime.recover_tool_jobs(bound)
        lease_wait_deadline = min(
            deadline,
            time.monotonic() + RECOVERY_CLAIM_WAIT_SECONDS,
        )
        while True:
            try:
                handle = await self.runtime.recover(
                    bound,
                    execution_id,
                    deadline=deadline,
                )
                stored = await self.history.get_execution(execution_id)
                if stored is None:
                    raise KeyError(f"unknown durable execution {execution_id!r}")
                _, decoded_context = invocation_from_dict(
                    stored.input,
                    registry=self.runtime.context_codec_registry,
                )
                context = cast(PygentContext, decoded_context)
                self._track_turn_context(execution_id, context)
                await self._deliver_projection_replacement(handle, context)
                return handle
            except ExecutionAdmissionError as exc:
                if (
                    "owned by another recovery attempt" not in str(exc)
                    or time.monotonic() >= lease_wait_deadline
                ):
                    raise
                await asyncio.sleep(0.1)

    def _restored_session_context(
        self,
        *,
        session: AgentSession,
        run_ref: CaseRunRef,
        turn_id: str,
        config: RunConfig,
        carry_context: bool = True,
    ) -> PygentContext:
        """Rebuild one session projection for main turns and compression."""

        return restore_session_context(
            session=session,
            run_ref=run_ref,
            turn_id=turn_id,
            config=config,
            carry_context=carry_context,
        )

    async def _prepare_turn(
        self,
        *,
        manager: SessionManager,
        run_ref: CaseRunRef,
        message: str,
        config: RunConfig,
        turn_id: str,
        session: AgentSession | None = None,
        carry_context: bool = True,
        message_kind: str = "pygent.v1.chat.turn",
        message_data: dict[str, object] | None = None,
        attachments: Sequence[str] = (),
        exchange_context: str | None = None,
    ) -> tuple[UserMessage, PygentContext]:
        if session is None:
            session = manager.load(run_ref.session_id)
        execution_context = self._restored_session_context(
            session=session,
            run_ref=run_ref,
            turn_id=turn_id,
            config=config,
            carry_context=carry_context,
        )
        covered_through = execution_context.memory_covered_through
        resolved_attachments: tuple[ResolvedAttachment, ...] = ()
        if message_kind != "pygent.v1.automation.trigger" and attachments:
            resolved_attachments = resolve_attachments(
                attachments, config.workspace_root
            )
        if message_kind in (
            "pygent.v1.automation.trigger",
            FINALIZER_MESSAGE_KIND,
            CONTINUATION_MESSAGE_KIND,
        ):
            # Harness-triggered injections carry their own framing; wrapping
            # them in <user-context> would misrepresent their origin.
            wrapped = message
        else:
            body_lines = [
                f"  <user-identity>{escape(config.user_identity or 'default', quote=False)}</user-identity>",
                f"  <user-message>{escape(message, quote=False)}</user-message>",
            ]
            if resolved_attachments:
                body_lines.append(render_attachments_xml(resolved_attachments))
            # Realtime sessions must open with a speak acknowledgment before
            # any tool work (D-RT5); plain-text sessions answer directly.
            if realtime_enabled_for(session):
                body_lines.append(render_reply_instruction())
            wrapped = str(Prompt("\n".join(body_lines)).wrap("user-context"))
        reminder = await self.reminders.claim_initial(session.session_id, turn_id)
        dynamic_reminder = await self.reminders.collect_pending(session.session_id)
        if dynamic_reminder:
            reminder = (
                f"{reminder}\n\n{dynamic_reminder}" if reminder else dynamic_reminder
            )
        if exchange_context:
            reminder = (
                f"{reminder}\n\n{exchange_context}" if reminder else exchange_context
            )
        if reminder:
            wrapped = f"{wrapped}\n\n{reminder}"
        current = UserMessage(
            content=wrapped,
            kind=message_kind,
            data=freeze_json_object(
                {
                    "raw_content": message,
                    **(
                        {
                            "attachments": [
                                item.relative for item in resolved_attachments
                            ]
                        }
                        if resolved_attachments
                        else {}
                    ),
                    **dict(message_data or {}),
                }
            ),
            media=build_media_blocks(resolved_attachments),
        )
        metadata = dict(execution_context.metadata)
        memory_messages = eternal_conversation_messages(session, execution_context)
        if memory_messages is not None:
            replacement = ReplaceMessageProjection(
                messages=(*memory_messages, current),
                expected_revision=execution_context.projection_revision + 1,
            )
            snapshot_revision = execution_context.memory_projection.get(
                "snapshot_revision", 0
            )
            identity = (
                f"{run_ref.case_run_id}:memory-projection:"
                f"{snapshot_revision}:{covered_through}"
            )
            metadata.update(
                {
                    "projection_replacement_pending": True,
                    "projection_replacement_message_count": len(replacement.messages),
                    PROJECTION_OPERATION_METADATA_KEY: encode_react_projection_operation(
                        replacement
                    ),
                    PROJECTION_INPUT_ID_METADATA_KEY: identity,
                    PROJECTION_READY_INPUT_ID_METADATA_KEY: f"{identity}:ready",
                }
            )
        execution_context = replace(execution_context, metadata=metadata)
        return (
            current,
            execution_context,
        )

    @staticmethod
    def _record_execution_id(run_ref: CaseRunRef, execution_id: str) -> None:
        path = Path(run_ref.run_dir) / "run_metadata.json"
        metadata = read_json(path, default={})
        metadata["runtime_execution_id"] = execution_id
        write_json(path, metadata)

    async def deliver_approval(
        self, approval_id: str, *, approved: bool, comment: str = ""
    ) -> bool:
        return await self.runtime.deliver_external(
            kind="tool-approval",
            key=approval_id,
            value={"approved": approved, "comment": comment},
        )

    async def _run_memory_job(self, request: MemoryJobRequest) -> str:
        """Run one background memory job through the pygent_v1 agent scaffold.

        The extractor is a fork agent: block 1 is the foreground system
        prompt plus the frozen working-memory history (verbatim), block 2 is
        the directive, block 3 is a multi-turn ReAct execution whose tools
        come from a registered surface (bash reaches the memory CLI; there
        are no dedicated memory tools). The builder is an independent child
        agent. Both run in pygent direct-execution mode, like the previous
        memory agent, but through the single AgentRunner seam.
        """

        alias = request.alias
        child_config = self._child_agent_config(alias)
        if child_config.resolved_agent is None:
            raise RuntimeError(f"background memory Agent {alias!r} is unconfigured")
        agent = PygentAgent(
            child_config,
            managed_model=False,
            model_group_name=child_config.resolved_agent.default_model_group,
            credential_environ=self._credential_environ,
        )
        try:
            if agent.llm is None:
                raise RuntimeError(
                    f"background memory Agent {alias!r} has no configured model"
                )
            registry = ToolRegistry()
            # Aligned with the foreground: the memory-job surface exposes the
            # same StandardTools composition (bash, files, task control). The
            # audit/EventStore/effects pipeline remains a §6.8 item; opening
            # the tools before that is the owner's explicit decision (D15).
            registry.register(
                "memory-job",
                ToolkitToolSurface(self._memory_job_tools.toolkit),
            )
            runner = AgentRunner(
                model_factory=build_model_layer_factory(
                    agent.llm,
                    model_group_name=(
                        f"pygent.v1:{child_config.resolved_agent.default_model_group}"
                    ),
                    models=agent._model_entries(),
                ),
                registry=registry,
                default_tools=ToolProfile("memory-job"),
                limits=MEMORY_JOB_LIMITS,
                label=request.label,
            )
            job: ForkAgent | ChildAgent
            if request.role == "extractor":
                window = session_fork_window(
                    mode="frozen",
                    system_prompt=request.system_prompt,
                    history_range=request.history_range,
                )
                job = ForkAgent(
                    runner, ForkSpec(source=SessionForkWindowSource(window)), label=request.label
                )
            else:
                job = ChildAgent(
                    runner, ChildSpec(name="memory-builder", system_prompt=request.system_prompt), label=request.label
                )
            return (await job.run(request.directive)).text
        finally:
            await agent.aclose()

    async def close(self, *, cancel: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        await self.memory_harness.close()
        prune_task, self._journal_prune_task = self._journal_prune_task, None
        if prune_task is not None:
            prune_task.cancel()
            await asyncio.gather(prune_task, return_exceptions=True)
        await self.runtime.close(cancel=cancel)
        await self._speak_executor.drain()
        await self.bash_tasks.close()
        await self._standard_tools.aclose()
        await self._memory_job_tools.aclose()
        for pool in self._mcp_pools:
            try:
                await asyncio.wait_for(asyncio.shield(pool.aclose()), 10)
            except TimeoutError:
                # A hung in-flight call must not block shutdown; the shielded
                # close task keeps draining in the background.
                pass
        for agent in self._agent_definitions.values():
            await aclose_if_supported(agent._standard_tools)
        for invoker in {
            id(value): value for value in self._model_invokers.values()
        }.values():
            await aclose_if_supported(invoker)
        if self.capacity is not None:
            await self.capacity.close(release_leases=True)
        await self.history.close()
        if self._owns_reminders:
            await self.reminders.close()


__all__ = ["PygentRuntimeService"]
