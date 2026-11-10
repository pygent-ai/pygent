from __future__ import annotations

import asyncio
import hashlib
import os
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, cast

from pygent import (
    AIMessage,
    ModelGroup,
    ToolKit,
    UserMessage,
    freeze_json_object,
)
from pygent.agent import (
    REACT_PROJECTION_OPERATION_KIND,
    ReplaceMessageProjection,
    encode_react_projection_operation,
)
from pygent.lara.agents import (
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
from pygent.lara.core.io import aclose_if_supported, plain_object, read_json, write_json
from pygent.lara.core.paths import user_lara_root_for
from pygent.lara.core.prompt import Prompt
from pygent.lara.dispatch import DispatchOperationView, DispatchService, DispatchStore
from pygent.lara.runtime.attachments import (
    ResolvedAttachment,
    build_media_blocks,
    render_attachments_xml,
    resolve_attachments,
)
from pygent.lara.runtime.bash_tasks import BashTaskObservations
from pygent.lara.runtime.mcp import ReadinessRetryingTransport
from pygent.lara.runtime.reminders import ReminderService
from pygent.lara.runtime.reminders.agent_messages import render_agent_message
from pygent.lara.scheduler.tools import SCHEDULER_TOOL_SPECS, SchedulerToolExecutor
from pygent.lara.schema import AgentSession, CaseRunRef, RunConfig
from pygent.lara.sessions import (
    AgentMessage,
    CollaborationOperation,
    SessionManager,
    realtime_enabled_for,
)
from pygent.lara.tracing import DIFF_TOOL_SPEC, DiffTool
from pygent.llm import ModelResourceRef
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
from pygent.runtime.codec import invocation_from_dict
from pygent.tool import (
    AgentToolExecutor,
    ExecutorRegistry,
    LocalToolExecutor,
    SandboxExecutorSupport,
    StandardTools,
    ToolSpec,
)
from pygent.tool.executors import ToolExecutionContext
from pygent.tool.mcp import MCPClientPool

from .agent.common import (
    DEFAULT_REACT_MAX_STEPS,
    _to_pygent_message,
)
from .agent.core import (
    LARA_PROJECTION_READY_KIND,
    LaraAgent,
    model_retry_policy,
)
from .agent.prompts import PromptRegistry
from .agent_collaboration import (
    AGENT_COLLABORATION_TOOL_SPECS,
    visible_agent_collaboration_specs,
)
from .context import LARA_CONTEXT_CODEC, LaraContext
from .deployment import (
    LaraModelResourceResolver,
    WorkspaceToolExecutor,
)
from .eternal_conversation import (
    EternalConversationHarness,
    MemoryJobRequest,
    eternal_enabled_for,
    load_projection,
    render_memory_snapshot,
)
from .file_effects import FILE_EFFECT_TOOL_SPEC, FileEffectToolExecutor
from .fork_context import (
    SessionForkWindowSource,
    restore_session_context,
    session_fork_window,
)
from .model_configuration import preferred_models, preferred_profile_name
from .realtime.continuation import CONTINUATION_MESSAGE_KIND
from .realtime.envelope import format_user_asked
from .realtime.fork_agent import ForkReplyModule, outcome_from_result
from .realtime.fork_reasoning import ForkModelProfile, fork_model_profiles
from .realtime.fork_reply import build_fork_context
from .realtime.fork_state import SessionForkState
from .realtime.posting import queue_exchange_envelope
from .realtime.speak import (
    SILENT_ACKNOWLEDGEMENT,
    SPEAK_TOOL_SPEC,
    parse_speak_args,
    render_reply_instruction,
)
from .realtime.turn_finalizer import FINALIZER_MESSAGE_KIND

if TYPE_CHECKING:
    from pygent.lara.orchestration.session_collaboration import (
        SessionCollaborationService,
    )

RECOVERY_CLAIM_WAIT_SECONDS = 31.0
# Pygent agent graphs declare requires_finite_deadline=True, so an unlimited
# deadline is rejected at admission; this ceiling is the practical maximum.
TURN_DEADLINE_SECONDS = 24 * 60 * 60

AgentDefinitionKey: TypeAlias = tuple[
    int, bool, str, str | None, bool, bool
]


class ModelBoundAgent(Protocol):
    """The model-facing surface shared by LaraAgent and ForkReplyModule."""

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
    # Imported lazily: lara.orchestration imports runtime.service at package load.
    from pygent.lara.orchestration.runtime_keys import RuntimeGenerationKey

    return RuntimeGenerationKey.from_config(config).config_fingerprint


def _uncovered_conversation_messages(
    history: list[dict[str, Any]], covered_through: int
) -> tuple[Any, ...]:
    """Keep the uncovered suffix, including its complete opening user turn.

    When the snapshot covers everything, retain the latest exchange so short
    follow-ups still have their immediate conversational referent.
    """
    if not 0 <= covered_through <= len(history):
        raise ValueError("memory coverage is outside the persisted conversation")
    start = covered_through
    if start == len(history) and start:
        start -= 1
    while start > 0 and history[start].get("role") != "user":
        start -= 1
    return tuple(
        message
        for item in history[start:]
        if (message := _to_pygent_message(item)) is not None
    )


class _SpeakExecutor:
    """Delivers the model's speak calls as user-facing stream events (D-RT10).

    The event (``lara.realtime.speak``) is the bubble the desktop renders;
    ``say_nothing`` emits nothing and acknowledges silently. Attachments are
    validated against the workspace and reported as workspace-relative paths.
    Configured external channels receive the same text fire-and-forget —
    channel failures never affect the tool result. Channels are supplied by
    the host application through the speak channel slot; the kernel ships
    none by default.
    """

    def __init__(self, service: LaraRuntimeService, channels: Sequence[Any] = ()) -> None:
        self.service = service
        self.channels = tuple(channels)
        self._delivery_tasks: set[asyncio.Task[bool]] = set()

    async def execute(
        self, spec: ToolSpec, call: Any, context: ToolExecutionContext
    ) -> object:
        del spec
        args = parse_speak_args(plain_object(call.arguments))
        if args.say_nothing:
            return SILENT_ACKNOWLEDGEMENT
        if context.emit is None:  # pragma: no cover - managed executions carry emit
            raise RuntimeError("speak delivery requires an execution event emitter")
        attachments: list[str] = []
        if args.attachments:
            resolved = resolve_attachments(
                args.attachments, self.service.config.workspace_root
            )
            attachments = [item.relative for item in resolved]
        speak_event: Mapping[str, Any] = {
            "text": args.text or "",
            "attachments": attachments,
        }
        await context.emit("lara.realtime.speak", speak_event)
        if self.channels:
            text = args.text or ""
            if attachments:
                text = "\n".join([text, *attachments]) if text else "\n".join(attachments)
            for channel in self.channels:
                task = asyncio.create_task(
                    asyncio.to_thread(channel.deliver, text)
                )
                self._delivery_tasks.add(task)
                task.add_done_callback(self._delivery_tasks.discard)
        return "delivered"


class _DiffExecutor:
    def __init__(self, service: LaraRuntimeService) -> None:
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


class LaraRuntimeService:
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
        collaboration: SessionCollaborationService | None = None,
        dispatch: DispatchService | None = None,
        credential_environ: Mapping[str, str] | None = None,
        run_config_loader: Callable[..., RunConfig] | None = None,
        prompt_registry: PromptRegistry | None = None,
    ) -> None:
        self.config = config
        self.collaboration = collaboration
        self._dispatch = dispatch
        self._credential_environ = credential_environ
        self._run_config_loader = run_config_loader
        self._prompt_registry = prompt_registry
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
            context_codecs=(LARA_CONTEXT_CODEC,),
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
        self.runtime.register_tool(
            SPEAK_TOOL_SPEC,
            _SpeakExecutor(
                self,
                # The speak channel slot stays generic; delivery channels are
                # host-provided integrations (e.g. a standalone Feishu
                # package) and the kernel ships none by default.
                channels=(),
            ),
        )
        collaboration_executor = AgentToolExecutor(
            invoke=self._execute_agent_collaboration
        )
        # The framework reads executor sandbox support via getattr(); the attribute is
        # intentionally attached to the shared pygent executor for this deployment.
        collaboration_executor.sandbox_support = SandboxExecutorSupport(profiles=("agent",), durable_reconnect=True, deployment_fingerprint="lara:agent-collaboration:v1")  # type: ignore[attr-defined]
        for spec in AGENT_COLLABORATION_TOOL_SPECS:
            self.runtime.register_tool(spec, collaboration_executor)
        scheduler_executor = SchedulerToolExecutor(config)
        for spec in SCHEDULER_TOOL_SPECS:
            self.runtime.register_tool(spec, scheduler_executor)
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
        self.model_resolver = LaraModelResourceResolver()
        self.runtime.register_model_resource_resolver(self.model_resolver)
        scope = (
            CapacityScope.DEPLOYMENT
            if config.runtime_capacity.scope == "deployment"
            else CapacityScope.RUNTIME_INSTANCE
        )
        self.binding = self.runtime.create_binding(
            name="lara-agent",
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
                capacity_key="lara-api-model",
                scope=scope,
            ),
            tool_capacity=CapacityPolicy.limited(
                max_concurrency=tool_max_concurrency,
                max_queue_size=max_queue_size,
                capacity_key="lara-api-tool",
                scope=scope,
            ),
            durability=DurabilityPolicy(DurabilityMode(config.runtime_durability.mode)),
        )
        self.external_tools: tuple[ToolSpec, ...] = (
            *visible_agent_collaboration_specs(
                config,
                collaboration_available=collaboration is not None,
            ),
            *SCHEDULER_TOOL_SPECS,
        )
        self._turn_contexts: dict[str, LaraContext] = {}
        self._fork_agents: dict[tuple, ForkReplyModule] = {}
        self._fork_states: dict[str, SessionForkState] = {}
        self.warnings: list[str] = []
        self._initialized = False
        self._initializing_task: asyncio.Task[None] | None = None
        self._closed = False
        self._mcp_pools: list[MCPClientPool] = []
        self.memory_harness = EternalConversationHarness(
            config.eternal_conversation,
            run_memory_job=self._run_memory_job,
            user_clis_root=Path(config.user_lara_root) / "clis",
            foreground_system_prompt=config.system_prompt,
        )
        template = (
            LaraAgent(
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
        self._agent_definitions: dict[AgentDefinitionKey, LaraAgent] = (
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

    async def _execute_agent_collaboration(
        self,
        spec: ToolSpec,
        call: Any,
        context: ToolExecutionContext,
    ) -> object:
        collaboration = self.collaboration
        if collaboration is None:
            raise RuntimeError("Agent collaboration is unavailable in this runtime")
        actor = await self._agent_tool_context(context)
        arguments = plain_object(call.arguments)
        name = spec.definition.name
        submission_id = str(call.idempotency_key or call.call_id)
        if name == "agent_start":
            agent_alias = str(arguments["agent"])
            if agent_alias not in self.config.delegation.allowed_agents:
                raise PermissionError(
                    f"collaboration with agent {agent_alias!r} is not allowed"
                )
            model_group = arguments.get("model_group")
            child_config = self._child_agent_config(agent_alias)
            if model_group is not None:
                self._require_model_group(child_config, str(model_group))
            # Background starts are dispatched user-globally; the dispatch
            # scheduler claims and drives them outside this process.
            item = await asyncio.to_thread(
                self._dispatch_service().submit_start,
                config=child_config,
                source_session_id=actor.session_id,
                message=str(arguments["task"]),
                submission_id=submission_id,
                model_group=None if model_group is None else str(model_group),
            )
            return DispatchOperationView.from_item(item).to_dict()
        if name == "agent_send":
            # Addressing policy: sessions resolve through their own project
            # store; unknown sessions fail closed inside collaboration.send.
            target_session_id = str(arguments["session_id"])
            message = await collaboration.send(
                config=self.config,
                manager=SessionManager(self.config),
                source_session_id=actor.session_id,
                source_agent_alias=self.config.agent_alias,
                target_session_id=target_session_id,
                message=str(arguments["message"]),
                submission_id=submission_id,
            )
            return message.to_dict()
        if name == "agent_list":
            operations = await collaboration.list_operations(
                config=self.config,
                session_id=actor.session_id,
            )
            return {"operations": [operation.to_dict() for operation in operations]}
        if name == "agent_status":
            collaboration_id = str(arguments["collaboration_id"])
            return await self._agent_collaboration_status(
                collaboration, collaboration_id, actor.session_id
            )
        if name == "agent_wait":
            timeout = float(arguments.get("timeout_seconds", 30))
            if not 0 < timeout <= 120:
                raise ValueError(
                    "timeout_seconds must be greater than 0 and at most 120"
                )
            has_single = "collaboration_id" in arguments
            multiple = "collaboration_ids" in arguments
            if has_single == multiple:
                raise ValueError(
                    "provide exactly one of collaboration_id or collaboration_ids"
                )
            collaboration_ids = (
                tuple(str(item) for item in arguments["collaboration_ids"])
                if multiple
                else (str(arguments["collaboration_id"]),)
            )
            if not 1 <= len(collaboration_ids) <= 32:
                raise ValueError(
                    "collaboration_ids must contain between 1 and 32 items"
                )
            if len(set(collaboration_ids)) != len(collaboration_ids):
                raise ValueError("collaboration_ids must be unique")
            return await self._wait_for_agent_collaborations(
                collaboration,
                collaboration_ids,
                actor.session_id,
                timeout=timeout,
                multiple=multiple,
            )
        raise ValueError(f"unsupported Agent collaboration tool {name!r}")

    def _track_turn_context(self, execution_id: str, context: LaraContext) -> None:
        """Remember one execution context for non-durable (interruptible) turns.

        Durable turns resolve their context from the SQLite history record;
        interruptible turns have no history row, so tool executors read the
        context captured at start here.
        """

        self._turn_contexts[execution_id] = context
        while len(self._turn_contexts) > MAX_TRACKED_TURN_CONTEXTS:
            self._turn_contexts.pop(next(iter(self._turn_contexts)), None)

    async def _turn_context(self, execution_id: str) -> LaraContext:
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
        if not isinstance(decoded, LaraContext):
            raise TypeError("execution context is not a LaraContext")
        return decoded

    async def _agent_tool_context(self, context: ToolExecutionContext) -> LaraContext:
        if context.execution_id is None:
            raise RuntimeError("Agent collaboration requires an execution id")
        return await self._turn_context(context.execution_id)

    async def _agent_collaboration_status(
        self,
        collaboration: SessionCollaborationService,
        collaboration_id: str,
        actor_session_id: str,
    ) -> dict[str, Any]:
        item = await collaboration.item_status(
            config=self.config,
            collaboration_id=collaboration_id,
        )
        self._require_collaboration_access(item, actor_session_id)
        return item.to_dict()

    async def _wait_for_agent_collaborations(
        self,
        collaboration: SessionCollaborationService,
        collaboration_ids: tuple[str, ...],
        actor_session_id: str,
        *,
        timeout: float,
        multiple: bool,
    ) -> dict[str, Any]:
        items = tuple(
            await asyncio.gather(
                *(
                    collaboration.item_status(
                        config=self.config,
                        collaboration_id=collaboration_id,
                    )
                    for collaboration_id in collaboration_ids
                )
            )
        )
        for item in items:
            self._require_collaboration_access(item, actor_session_id)
        if any(item.done for item in items):
            completed = items
            timed_out = False
        else:
            # Dispatched items are driven by the dispatch scheduler; waiting is
            # pure polling, there is no in-process resume anymore.
            completed, timed_out = await collaboration.wait_any(
                config=self.config,
                collaboration_ids=collaboration_ids,
                timeout=timeout,
            )
            for item in completed:
                self._require_collaboration_access(item, actor_session_id)
        if not multiple:
            return {**completed[0].to_dict(), "timed_out": timed_out}
        return {
            "ready": [item.to_dict() for item in completed if item.done],
            "pending": [item.to_dict() for item in completed if not item.done],
            "timed_out": timed_out,
        }

    @classmethod
    def _require_collaboration_access(
        cls,
        item: AgentMessage | CollaborationOperation | DispatchOperationView,
        actor_session_id: str,
    ) -> None:
        if isinstance(item, AgentMessage):
            cls._require_message_access(item, actor_session_id)
        else:
            cls._require_operation_access(item, actor_session_id)

    @staticmethod
    def _require_operation_access(
        operation: CollaborationOperation | DispatchOperationView,
        actor_session_id: str,
    ) -> None:
        if actor_session_id not in {
            operation.source_session_id,
            operation.target_session_id,
        }:
            raise PermissionError("Agent operation is outside this collaboration")

    @staticmethod
    def _require_message_access(message: AgentMessage, actor_session_id: str) -> None:
        if actor_session_id not in {
            message.source_session_id,
            message.target_session_id,
        }:
            raise PermissionError("Agent message is outside this collaboration")

    async def _on_tool_task_event(self, kind: str, data: Mapping[str, Any]) -> None:
        # Delegate through self.bash_tasks so a replaced observation instance
        # (restore path) is still the event target.
        await self.bash_tasks.on_tool_task_event(kind, data)

    async def initialize(self) -> None:
        if self._closed:
            raise RuntimeError("LaraRuntimeService is closed")
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
        discovered = list(self.external_tools)
        visible_names: set[str] = {
            "bash",
            "read",
            "write",
            "edit",
            "glob",
            "grep",
            "diff",
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
    ) -> LaraAgent:
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
        invoker = (
            self._model_invokers.get(resolved.alias) if resolved is not None else None
        )
        agent = LaraAgent(
            effective_config,
            resolved_agent=effective_config.resolved_agent,
            prompt_registry=self._prompt_registry,
            external_tools=self.external_tools,
            managed_model=True,
            model_group_name=selected_group,
            interactive_approvals=interactive_approvals,
            model_invoker=invoker,
            memory_harness=self.memory_harness,
            reminders=self.reminders,
            bash_tasks=self.bash_tasks,
            selected_model_key=selected_model_key,
            eternal_memory_enabled=eternal_memory,
            credential_environ=self._credential_environ,
            realtime_interaction_enabled=realtime_interaction,
        )
        if resolved is not None and agent.llm is not None:
            self._model_invokers.setdefault(resolved.alias, agent.llm)
        self._agent_definitions[key] = agent
        return agent

    def _dispatch_service(self) -> DispatchService:
        """User-level dispatch service; overridable via the constructor."""
        if self._dispatch is None:
            root = user_lara_root_for(self.config.lara_root)
            self._dispatch = DispatchService(
                DispatchStore(Path(root) / "dispatch-v1.sqlite3")
            )
        return self._dispatch

    def _child_agent_config(self, alias: str) -> RunConfig:
        if alias == self.config.agent_alias:
            return RunConfig.from_dict(self.config.to_dict())
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

    @staticmethod
    def _require_model_group(child_config: RunConfig, model_group: str) -> None:
        native = child_config.model_config
        if native is None or model_group not in native.model_groups:
            raise ValueError(
                f"unknown model group {model_group!r} for agent "
                f"{child_config.agent_alias!r}"
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
            requirement = ModelGroup.deferred(name=f"lara:{agent.model_group_name}")
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
                        resource_id=f"lara:{agent.model_group_name}",
                        revision=revision,
                        capacity_owner_id=f"lara:{Path(self.config.workspace_root).resolve()}",
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
        handle: Any, context: LaraContext
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
            (PROJECTION_READY_INPUT_ID_METADATA_KEY, LARA_PROJECTION_READY_KIND, True),
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
        context: LaraContext,
        *,
        execution: Any,
    ) -> Any:
        handle = await bound.start(message, context, execution=execution)
        try:
            await self._deliver_projection_replacement(handle, context)
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
            f"lara:{group_name}": {
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
        message_kind: str = "lara.chat.turn",
        message_data: dict[str, object] | None = None,
        attachments: Sequence[str] = (),
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
        initial_claim_id = self.reminders.turn_opening_claim_id(
            run_ref.session_id, turn_id
        )
        initial_agent_messages = await self.reminders.claim_turn_opening(
            run_ref.session_id,
            turn_id=turn_id,
            message_kind=message_kind,
            realtime_enabled=realtime_enabled,
        )
        initial_agent_context = (
            "\n".join(
                render_agent_message(item) for item in initial_agent_messages
            )
            if initial_agent_messages
            else None
        )
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
                initial_agent_context=initial_agent_context,
            )
            bound = await self.bind(agent, agent)
            handle = await self._start_agent_execution(
                bound,
                turn_message,
                turn_context,
                execution=ExecutionOptions(
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
            )
        except BaseException:
            await self.reminders.release_initial(run_ref.session_id, turn_id)
            for item in initial_agent_messages:
                await self.reminders.release_agent_message(
                    item.message_id, claim_id=initial_claim_id
                )
            raise
        await self.reminders.acknowledge_initial(
            run_ref.session_id, turn_id, handle.execution_id
        )
        for item in initial_agent_messages:
            await self.reminders.acknowledge_agent_message(
                item.message_id,
                claim_id=initial_claim_id,
                execution_id=handle.execution_id,
            )
        self._record_execution_id(run_ref, handle.execution_id)
        self._track_turn_context(handle.execution_id, turn_context)
        return handle

    async def pending_agent_messages(
        self, session_id: str
    ) -> tuple[AgentMessage, ...]:
        """Peek the session's queued agent messages without claiming them."""

        return await self.reminders.pending_agent_messages(session_id)

    async def send_interrupt(
        self,
        execution_id: str,
        *,
        manager: SessionManager,
        run_ref: CaseRunRef,
        session_id: str,
        input_id: str,
        message: str,
    ) -> Any:
        """Fork-reply path for a user message arriving mid-turn (D-RT2/3).

        A session-level :class:`SessionForkState` supersedes an in-flight
        fork when another interrupt arrives: the old handle is cancelled, the
        messages are merged into a numbered batch, and a new generation starts
        with a fresh deadline. A late result from an older generation is
        discarded before it can queue an exchange envelope.
        """

        await self.initialize()
        config = self.config
        session = manager.load(session_id)
        if not realtime_enabled_for(session):
            raise RuntimeError("realtime interaction is not enabled for this session")
        state = self._fork_states.setdefault(session_id, SessionForkState())
        async with state.lock:
            if state.handle is not None and not state.finished:
                state.pending.append(message)
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
                fork_messages = [message]
                fork_input_id = (
                    input_id
                    if was_idle and generation == 1
                    else f"{input_id}:g{generation}"
                )
                request_message = message
            state.finished = False
            handle = await self._start_fork_reply(
                session=session,
                run_ref=run_ref,
                config=config,
                input_id=fork_input_id,
                message=request_message,
            )
            state.handle = handle

        try:
            output, _ = await handle.result()
        except asyncio.CancelledError:
            output = None
        except BaseException:
            async with state.lock:
                if state.generation == generation:
                    state.handle = None
                    state.finished = False
                    state.pending = []
            raise
        if output is None:
            return None
        assert isinstance(output, AIMessage)
        outcome = outcome_from_result(output)
        async with state.lock:
            if state.generation != generation:
                return None
            state.finished = True
            if outcome.speak_args is not None and outcome.speak_args.say_nothing:
                response_text = "(decided not to speak)"
            elif outcome.speak_args is not None:
                response_text = outcome.speak_args.text or ""
            else:
                response_text = outcome.text or ""
            queue_exchange_envelope(
                self.reminders.message_store,
                session_id=session_id,
                submission_id=f"realtime-exchange:{fork_input_id}",
                user_asked=tuple(fork_messages),
                response=response_text or "(no verbal reply)",
            )
            state.handle = None
            state.finished = False
            state.pending = []
        return outcome

    async def _start_fork_reply(
        self,
        *,
        session: AgentSession,
        run_ref: CaseRunRef,
        config: RunConfig,
        input_id: str,
        message: str,
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
        return await bound.start(
            UserMessage(content=message, kind="lara.user.interrupt"),
            request,
            execution=ExecutionOptions(
                request_id=f"fork:{input_id}",
                idempotency_key=f"fork:{input_id}",
                identity=run_ref.session_id,
                model_calls=model_calls,
                deadline=time.monotonic()
                + config.realtime_interaction.fork_deadline_seconds,
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
        )
        self._fork_agents[key] = agent
        return agent

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
        _, recovered_context = invocation_from_dict(
            stored.input,
            registry=self.runtime.context_codec_registry,
        )
        if not isinstance(recovered_context, LaraContext):
            raise TypeError("durable execution context is not a LaraContext")
        session_manager = SessionManager(self.config)
        group_name, selected_model_key = session_manager.model_selection(
            recovered_context.session_id
        )
        recovered_session = session_manager.load(recovered_context.session_id)
        agent = self.new_agent(
            interactive_approvals=True,
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
                context = cast(LaraContext, decoded_context)
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
    ) -> LaraContext:
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
        message_kind: str = "lara.chat.turn",
        message_data: dict[str, object] | None = None,
        attachments: Sequence[str] = (),
        initial_agent_context: str | None = None,
    ) -> tuple[UserMessage, LaraContext]:
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
        if message_kind != "lara.automation.trigger" and attachments:
            resolved_attachments = resolve_attachments(
                attachments, config.workspace_root
            )
        if message_kind in (
            "lara.automation.trigger",
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
        if initial_agent_context:
            reminder = (
                f"{reminder}\n\n{initial_agent_context}"
                if reminder
                else initial_agent_context
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
        if (
            eternal_enabled_for(session, config)
            and carry_context
            and covered_through > 0
        ):
            memory_projection = load_projection(session.session_dir)
            snapshot = UserMessage(
                content=render_memory_snapshot(memory_projection),
                kind="lara.memory.snapshot",
                slot="lara.memory.snapshot",
            )
            replacement = ReplaceMessageProjection(
                messages=(
                    snapshot,
                    *_uncovered_conversation_messages(session.history, covered_through),
                    current,
                ),
                expected_revision=execution_context.projection_revision + 1,
            )
            identity = (
                f"{run_ref.case_run_id}:memory-projection:"
                f"{memory_projection.get('snapshot_revision', 0)}:{covered_through}"
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
        """Run one background memory job through the LARA agent scaffold.

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
        agent = LaraAgent(
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
                        f"lara:{child_config.resolved_agent.default_model_group}"
                    ),
                    models=agent._model_entries(),
                ),
                registry=registry,
                default_tools=ToolProfile("memory-job"),
                limits=MEMORY_JOB_LIMITS,
                label=request.label,
            )
            if request.role == "extractor":
                window = session_fork_window(
                    mode="frozen",
                    system_prompt=request.system_prompt,
                    history_range=request.history_range,
                )
                result = await ForkAgent(
                    runner,
                    ForkSpec(source=SessionForkWindowSource(window)),
                    label=request.label,
                ).run(request.directive)
                return result.text
            child = ChildAgent(
                runner,
                ChildSpec(name="memory-builder", system_prompt=request.system_prompt),
                label=request.label,
            )
            child_result = await child.run(request.directive)
            return child_result.text
        finally:
            await agent.aclose()

    async def close(self, *, cancel: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        await self.memory_harness.close()
        await self.runtime.close(cancel=cancel)
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


__all__ = ["LaraRuntimeService"]
