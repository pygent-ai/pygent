from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from pygent import (
    Agent,
    AIMessage,
    ExponentialBackoff,
    GenerationConfig,
    ModelCallLayer,
    ModelErrorKind,
    ModelGroup,
    PygentAgent,
    RetryPolicy,
    ToolCallLayer,
    ToolDefinition,
    ToolKit,
    UserMessage,
)
from pygent.core import EffectSafety, ExecutionRequirements, RecoverySafety
from pygent.llm import ModelCallError, ModelCallPolicy
from pygent.runtime.codec import context_to_dict
from pygent.tool import StandardTools, ToolSpec

from pygent.lara.core.io import aclose_if_supported, plain_data
from pygent.lara.core.redaction import redact_secrets
from pygent.lara.runtime.context import LARA_CONTEXT_CODECS, LaraContext
from pygent.lara.runtime.bash_tasks import BashTaskObservations
from pygent.lara.runtime.context_compression import COMPRESSION_REQUEST_PROMPT
from pygent.lara.runtime.eternal_conversation import (
    EternalConversationHarness,
    eternal_enabled_for,
)
from pygent.lara.runtime.file_effects import FILE_EFFECT_TOOL_SPEC
from pygent.lara.runtime.reminders import ReminderService
from pygent.lara.runtime.realtime.injection import RealtimeNudgeModule
from pygent.lara.runtime.realtime.speak import SPEAK_TOOL_SPEC
from pygent.lara.runtime.model_configuration import (
    CredentialEnvironment,
    build_model_invoker,
    workspace_media_resolver,
)
from pygent.lara.schema import ResolvedAgentConfig, RunConfig
from pygent.lara.sessions import SessionManager
from pygent.lara.tracing import DIFF_TOOL_SPEC, EventStore

from .common import DEFAULT_REACT_MAX_STEPS
from .compressor import LaraCompressorModule
from .pipeline import (
    ConversationCheckpointModelModule,
    ConversationCheckpointToolModule,
    DynamicPromptModule,
    ForegroundModelModule,
    LaraToolAuthorization,
    PersistedDiffModule,
    PreparedToolModule,
    RuntimeReminderModule,
    ToolAuditModule,
    _model_tool_definition,
    checkpoint_conversation_message,
)
from .prompts import PromptRegistry

DISABLED_COMPRESSION_WINDOW_TOKENS = 1 << 60
MAX_CONTEXT_COMPRESSIONS = 128
MODEL_RETRYABLE_ERROR_KINDS = (
    ModelErrorKind.TIMEOUT,
    ModelErrorKind.RATE_LIMIT,
    ModelErrorKind.UNAVAILABLE,
    ModelErrorKind.INVALID_RESPONSE,
)
LARA_PROJECTION_READY_KIND = "lara.react.projection.ready.v1"

_MEDIA_OUTPUT_MODALITIES = frozenset({"image", "audio", "video"})


def model_retry_policy(retry) -> RetryPolicy:
    """The shared model retry posture for foreground and fork calls."""

    return RetryPolicy(
        max_attempts_per_model=retry.max_attempts_per_model,
        retry_on=MODEL_RETRYABLE_ERROR_KINDS,
        attempt_idle_timeout_seconds=retry.attempt_idle_timeout_seconds,
        backoff=ExponentialBackoff(
            initial=retry.backoff_initial,
            maximum=retry.backoff_maximum,
            multiplier=retry.backoff_multiplier,
        ),
    )


def _model_trace_payload(
    config: RunConfig,
    *,
    group_name: str,
    model_key: str | None,
) -> dict[str, str | None]:
    native = config.model_config
    entry = native.models.get(model_key) if native is not None and model_key else None
    if entry is None:
        return {
            "model_group": group_name,
            "model_key": model_key,
            "provider": None,
            "model_id": None,
        }
    return {
        "model_group": group_name,
        "model_key": entry.key,
        "provider": entry.spec.provider,
        "model_id": entry.spec.model_id,
    }


def _actual_model_key(answer: AIMessage | None) -> str | None:
    if answer is None:
        return None
    metadata = dict(answer.metadata)
    model_key = metadata.get("model_key")
    return model_key if isinstance(model_key, str) else None


def _error_trace_payload(error: BaseException) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "error": str(error),
        "error_type": type(error).__name__,
    }
    if isinstance(error, sqlite3.Error):
        payload["sqlite_errorcode"] = getattr(error, "sqlite_errorcode", None)
        payload["sqlite_errorname"] = getattr(error, "sqlite_errorname", None)
    if isinstance(error, ModelCallError):
        payload["model_failure"] = _sanitized_model_failure(error.failure)
    return payload


def _sanitized_model_failure(failure: Any) -> dict[str, Any]:
    """Strip credentials from provider diagnostics before they reach traces.

    pygent forwards provider error text verbatim; the trace payload promises
    sanitized diagnostics, so the boundary redacts it here.
    """
    payload = failure.to_dict()
    attempts = payload.get("details", {}).get("attempts")
    if isinstance(attempts, list):
        for attempt in attempts:
            message = attempt.get("provider_message")
            if isinstance(message, str):
                attempt["provider_message"] = redact_secrets(message)
    return payload


class LaraForegroundAgent(PygentAgent):
    """Pygent foreground Agent with Lara's verified recovery declaration."""

    execution_requirements = ExecutionRequirements(
        recovery_safety=RecoverySafety.MODULE_BOUNDARY_RETRY,
        effect_safety=EffectSafety.MANAGED_EFFECTS,
    )


class LaraAgent(Agent[UserMessage, AIMessage]):
    execution_requirements = ExecutionRequirements(
        recovery_safety=RecoverySafety.MODULE_BOUNDARY_RETRY,
        effect_safety=EffectSafety.MANAGED_EFFECTS,
    )
    trusted_live_resource_attributes = (
        "config",
        "resolved_agent",
        "model_group_name",
        "prompt_registry",
        "workspace_root",
        "llm",
        "_standard_tools",
        "_toolkit",
        "_external_tools",
        "memory_harness",
        "reminders",
        "bash_tasks",
    )

    def __init__(
        self,
        config: RunConfig,
        resolved_agent: ResolvedAgentConfig | None = None,
        prompt_registry: PromptRegistry | None = None,
        external_tools: tuple[ToolSpec, ...] = (),
        managed_model: bool = False,
        model_group_name: str | None = None,
        interactive_approvals: bool = False,
        model_invoker: Any | None = None,
        memory_harness: EternalConversationHarness | None = None,
        reminders: ReminderService | None = None,
        bash_tasks: BashTaskObservations | None = None,
        selected_model_key: str | None = None,
        eternal_memory_enabled: bool | None = None,
        realtime_interaction_enabled: bool = False,
        credential_environ: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        resolved = resolved_agent or config.resolved_agent
        if resolved is None:
            raise ValueError(
                "LaraAgent requires a configured native model configuration"
            )
        self.resolved_agent: ResolvedAgentConfig = resolved
        native = config.model_config
        if native is None:
            raise ValueError("LaraAgent requires a configured native model configuration")
        self.model_group_name = model_group_name or resolved.default_model_group
        if self.model_group_name not in native.model_groups:
            raise ValueError(f"unknown model group {self.model_group_name!r}")
        self.selected_model_key = selected_model_key
        self.prompt_registry = prompt_registry
        self.managed_model = managed_model
        self.interactive_approvals = interactive_approvals
        # Eternal sessions must not compress: the memory projection replaces
        # the compressed tail, so both layers never manage one session.
        self.eternal_memory_enabled = (
            config.eternal_conversation.enabled
            if eternal_memory_enabled is None
            else eternal_memory_enabled
        )
        # Realtime silent-work interaction (speak nudges, fork replies) is
        # wired but inert until a session turns it on; the speak tool surface
        # itself lands with the routing seam.
        self.realtime_interaction_enabled = realtime_interaction_enabled
        self.workspace_root = Path(config.workspace_root)
        if model_invoker is not None:
            self.llm = model_invoker
        else:
            try:
                environment = credential_environ or CredentialEnvironment(
                    config.user_lara_root
                )
                self.llm = build_model_invoker(
                    native,
                    credential_environ=environment,
                    media_resolver=workspace_media_resolver(self.workspace_root),
                )
            except LookupError:
                self.llm = None
        self._standard_tools: StandardTools | None = None
        self._toolkit: ToolKit | None = None
        self._external_tools = external_tools
        self.memory_harness = memory_harness
        self.bash_tasks = bash_tasks
        self.reminders = reminders or ReminderService(config)
        self._owns_reminders = reminders is None
        self._register_default_tools()
        self._assemble_definition()

    @property
    def tool_definitions(self) -> tuple[ToolDefinition, ...]:
        local = self._toolkit.definitions if self._toolkit is not None else ()
        realtime_defs = (
            (SPEAK_TOOL_SPEC.definition,) if self.realtime_interaction_enabled else ()
        )
        return (
            *(_model_tool_definition(definition) for definition in local),
            DIFF_TOOL_SPEC.definition,
            *realtime_defs,
            *(spec.definition for spec in self._external_tools),
        )

    def _model_entries(self) -> tuple[Any, ...]:
        assert self.config.model_config is not None
        return self.config.model_config.model_groups[self.model_group_name].models

    def _media_output_modalities(self) -> tuple[str, ...]:
        """Non-text output modalities the selected main model can produce.

        Only the foreground managed path resolves a session-selected model;
        internal ReAct jobs (memory, evaluation) stay text-only. Pygent's
        invoker validates these against the candidate model's declared
        capabilities, so a wrong capability only routes away from the model.
        """

        if not self.managed_model or not self.selected_model_key:
            return ()
        for entry in self._model_entries():
            if entry.key == self.selected_model_key:
                return tuple(
                    modality
                    for modality in entry.spec.capabilities.modalities.output
                    if modality in _MEDIA_OUTPUT_MODALITIES
                )
        return ()

    def new_model_layer(self) -> ModelCallLayer:
        if self.llm is None:
            raise RuntimeError("model invoker is not configured")
        group = (
            ModelGroup.deferred(name=f"lara:{self.model_group_name}")
            if self.managed_model
            else ModelGroup(
                name=f"lara:{self.model_group_name}",
                models=self._model_entries(),
            )
        )
        retry = self.config.model_retry
        return ModelCallLayer(
            model_group=group,
            policy=ModelCallPolicy(allow_profile_override=self.managed_model),
            retry_policy=model_retry_policy(retry),
            generation=GenerationConfig(
                output_modalities=self._media_output_modalities()
            ),
            tools=self.tool_definitions,
            invoker=None if self.managed_model else self.llm,
        )

    def new_tool_layer(self, *, max_concurrency: int = 8) -> ToolCallLayer:
        if self._toolkit is None:
            raise RuntimeError("LaraAgent run graph has not been assembled")
        realtime_specs = (SPEAK_TOOL_SPEC,) if self.realtime_interaction_enabled else ()
        return ToolCallLayer(
            tools=(
                *self._toolkit.specs,
                DIFF_TOOL_SPEC,
                *realtime_specs,
                *self._external_tools,
            ),
            authorization=LaraToolAuthorization(
                enabled=self.config.runtime_approvals.enabled,
                timeout_seconds=self.config.runtime_approvals.timeout_seconds,
                preauthorized_tools=self.config.runtime_approvals.preauthorized_tools,
                interactive=self.interactive_approvals,
                scope_key="lara",
                detached_tools=("bash",),
            ),
            max_concurrency=max_concurrency,
        )

    def _assemble_definition(self) -> None:
        self.prompt = DynamicPromptModule(self.config, self.prompt_registry)
        if self.llm is not None:
            diff_tasks = ToolCallLayer(
                tools=(FILE_EFFECT_TOOL_SPEC,),
                authorization=LaraToolAuthorization(
                    enabled=False,
                    timeout_seconds=self.config.runtime_approvals.timeout_seconds,
                    preauthorized_tools=(),
                    interactive=False,
                    scope_key="lara-file-effects",
                ),
            )
            model = self.new_model_layer()
            compression_model = self.new_model_layer()
            prepared_model = ConversationCheckpointModelModule(
                self.config,
                ForegroundModelModule(model=model),
            )
            prepared_tools = ConversationCheckpointToolModule(
                self.config,
                PreparedToolModule(
                    tools=self.new_tool_layer(),
                    audit=ToolAuditModule(self.config, bash_tasks=self.bash_tasks),
                    reminders=RuntimeReminderModule(self.reminders),
                    persisted_diff=PersistedDiffModule(self.workspace_root, diff_tasks),
                    realtime_nudge=(
                        RealtimeNudgeModule(
                            interval=self.config.realtime_interaction.nudge_interval
                        )
                        if self.realtime_interaction_enabled
                        else None
                    ),
                ),
            )
            context_window = self.config.context_window
            compression_enabled = (
                self.config.context_compression_enabled
                and not self.eternal_memory_enabled
                and isinstance(context_window, int)
            )
            context_window_tokens = (
                context_window
                if isinstance(context_window, int) and compression_enabled
                else DISABLED_COMPRESSION_WINDOW_TOKENS
            )
            max_steps = (
                self.config.max_steps
                if self.config.max_steps > 0
                else DEFAULT_REACT_MAX_STEPS
            )
            self.foreground = LaraForegroundAgent(
                system_prompt="Lara supplies the execution system prompt in LaraContext.",
                compression_prompt=COMPRESSION_REQUEST_PROMPT,
                model=prepared_model,
                compressor=LaraCompressorModule(self.config, compression_model),
                tools=prepared_tools,
                context_window_tokens=context_window_tokens,
                compression_trigger_ratio=min(
                    self.config.context_compression_trigger_ratio, 0.999_999
                ),
                compression_context_window_tokens=context_window_tokens,
                max_compressions=MAX_CONTEXT_COMPRESSIONS,
                max_steps=max_steps,
                max_model_calls=max_steps,
                max_tool_calls=max(32, max_steps * 4),
            )
            self.foreground.react.model.execution_requirements = (
                LaraForegroundAgent.execution_requirements
            )

    async def forward(
        self,
        message: UserMessage,
        context: LaraContext,
    ) -> tuple[AIMessage, LaraContext]:
        if not isinstance(context, LaraContext):
            raise TypeError("LaraAgent requires LaraContext")
        manager = SessionManager(self.config)
        store = EventStore(context.case_run_ref)
        turn_id = context.turn_id
        message_data = plain_data(message.data)
        raw_content = str(
            (
                message_data.get("raw_content")
                if isinstance(message_data, dict)
                else None
            )
            or message.content
        )
        await checkpoint_conversation_message(
            self.config,
            context,
            message,
            boundary="user-input",
        )
        projection_pending = (
            context.metadata.get("projection_replacement_pending") is True
        )
        if projection_pending:
            while not await self.receive_execution_inputs(
                kinds=(LARA_PROJECTION_READY_KIND,),
                limit=1,
                seal_if_empty=False,
            ):
                await asyncio.sleep(0)
        execution_context = context
        await self.emit(kind="lara.chat.started", data=context.case_run_ref.to_dict())
        replacement_count = context.metadata.get(
            "projection_replacement_message_count",
            len(context.messages) + 1,
        )
        if not isinstance(replacement_count, int):
            replacement_count = len(context.messages) + 1
        selected_model_key = context.metadata.get("selected_model_key")
        if not isinstance(selected_model_key, str):
            selected_model_key = self._model_entries()[0].key
        store.append(
            "model.request",
            actor="system",
            payload={
                "agent": type(self).__name__,
                "agent_alias": self.resolved_agent.alias,
                **_model_trace_payload(
                    self.config,
                    group_name=self.model_group_name,
                    model_key=selected_model_key,
                ),
                "model_fallback": [entry.key for entry in self._model_entries()],
                "max_steps": self.config.max_steps,
                "history_message_count": replacement_count,
                "latest_user_input": raw_content,
                "input_origin": (
                    message_data.get("origin", "user")
                    if isinstance(message_data, dict)
                    else "user"
                ),
            },
            turn_id=turn_id,
        )
        status = "passed"
        error: str | None = None
        error_payload: dict[str, Any] | None = None
        answer: AIMessage | None = None
        try:
            visible = replace(execution_context, tools=self.tool_definitions)
            if self.llm is None:
                message, visible = await self.prompt(message, visible)
                answer = AIMessage(
                    content=(
                        "Lara agent is wired into chat, but API key is not configured "
                        f"for agent alias {self.resolved_agent.alias!r}."
                    )
                )
                next_context = visible + message + answer
                await checkpoint_conversation_message(
                    self.config,
                    next_context,
                    answer,
                    boundary="local-answer",
                )
            else:
                message, visible = await self.prompt(message, visible)
                answer, next_context = await self.foreground(message, visible)
            assert answer is not None
            committed_messages = next_context.committed_messages
            execution_context = next_context
            result = {
                "session_id": context.session_id,
                "case_id": context.case_id,
                "case_run_id": context.case_run_id,
                "turn_id": turn_id,
                "status": status,
                "final_answer": answer.content,
                "error": None,
                "message_count": len(committed_messages),
            }
            return replace(
                answer, kind="lara.chat.result", data={"result": result}
            ), next_context
        except asyncio.CancelledError:
            status, error = "skipped", "cancelled"
            store.append(
                "runtime.cancelled",
                actor="system",
                payload={"status": status, "reason": error},
                turn_id=turn_id,
            )
            raise
        except Exception as exc:
            status, error = "error", str(exc)
            error_payload = _error_trace_payload(exc)
            store.append(
                "runtime.error",
                actor="system",
                payload=error_payload,
                turn_id=turn_id,
            )
            raise
        finally:
            session = manager.load(context.session_id)
            # Stable user/model/tool boundaries are already persisted as
            # idempotent conversation checkpoints. Loading the session merges
            # any raw checkpoint written before the latest session snapshot.
            session.system_prompt = execution_context.system_prompt
            session.metadata.update(
                {
                    "active_case_id": context.case_id,
                    "last_case_run_id": context.case_run_id,
                    "last_case_run_status": status,
                }
            )
            if status == "passed":
                session.metadata["agent_context"] = context_to_dict(
                    execution_context,
                    registry=LARA_CONTEXT_CODECS,
                )
                # Record how much of the durable transcript this projection
                # covers; a later restore replays boundaries recorded after
                # it (written by turns that failed or were cancelled).
                session.metadata["agent_context_history_count"] = len(
                    session.history
                )
            manager.save(session)
            if (
                self.memory_harness is not None
                and status == "passed"
                and eternal_enabled_for(session, self.config)
            ):
                await self.memory_harness.record_and_trigger(
                    session,
                    model_envelope={
                        "system_prompt": execution_context.system_prompt,
                        "tools": [
                            {
                                "name": definition.name,
                                "description": definition.description,
                                "parameters": plain_data(definition.parameters),
                            }
                            for definition in execution_context.tools
                        ],
                    },
                    context_window_tokens=self.config.context_window,
                )
            store.append(
                "model.response",
                actor="system",
                payload={
                    "agent": type(self).__name__,
                    "agent_alias": self.resolved_agent.alias,
                    **_model_trace_payload(
                        self.config,
                        group_name=self.model_group_name,
                        model_key=_actual_model_key(answer),
                    ),
                    "status": status,
                    "error": error,
                    **(error_payload or {}),
                },
                turn_id=turn_id,
            )
            store.append(
                "context.checkpoint",
                actor="system",
                payload={
                    "status": status,
                    "history_message_count": len(session.history),
                    "case_run_id": context.case_run_id,
                },
                turn_id=turn_id,
            )

    async def aclose(self) -> None:
        await aclose_if_supported(self._standard_tools)
        if self._owns_reminders:
            await self.reminders.close()
        await aclose_if_supported(self.llm)

    def _register_default_tools(self) -> None:
        self._standard_tools = StandardTools(
            workspace_root=self.workspace_root, restrict_to_workspace=False
        )
        self._toolkit = ToolKit(
            self._standard_tools.bash.bash,
            self._standard_tools.bash.tool_task_get,
            self._standard_tools.bash.tool_task_stop,
            self._standard_tools.files.read,
            self._standard_tools.files.write,
            self._standard_tools.files.edit,
            self._standard_tools.files.glob,
            self._standard_tools.files.grep,
        )
        visible = {
            definition.name
            for definition in (*self._toolkit.definitions, DIFF_TOOL_SPEC.definition)
        }
        for spec in self._external_tools:
            if spec.definition.name in visible:
                raise ValueError(
                    f"duplicate model-visible tool name: {spec.definition.name}"
                )
            visible.add(spec.definition.name)
