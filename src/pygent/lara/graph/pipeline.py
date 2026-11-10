from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from pygent import (
    AIMessage,
    MediaBlock,
    ModelCallLayer,
    Module,
    ToolAuthorizationDecision,
    ToolAuthorizationRequest,
    ToolCall,
    ToolCallLayer,
    ToolDefinition,
    ToolMessage,
    freeze_json_object,
    thaw_json,
)
from pygent import (
    Context as PygentContext,
)
from pygent import (
    Message as PygentMessage,
)
from pygent import (
    ToolResult as PygentToolResult,
)
from pygent.core import (
    EffectIdempotency,
    EffectRetryPolicy,
    EffectSafety,
    EffectSideEffect,
    EffectSpec,
    ExecutionRequirements,
    RecoverySafety,
    active_infrastructure,
    freeze_json,
)
from pygent.lara.core.io import plain_data, plain_object
from pygent.lara.runtime.bash_tasks import BashTaskObservations
from pygent.lara.runtime.context import LaraContext
from pygent.lara.runtime.eternal_conversation import render_memory_access_instruction
from pygent.lara.runtime.file_effect_models import DeferredFileEffectJob
from pygent.lara.runtime.file_effects import (
    FILE_EFFECT_TOOL_SPEC,
    DeferredFileEffectBatch,
)
from pygent.lara.runtime.realtime.injection import RealtimeNudgeModule
from pygent.lara.runtime.reminders import ReminderService
from pygent.lara.runtime.reminders.agent_messages import render_agent_message
from pygent.lara.runtime.tools import ToolObserver
from pygent.lara.schema import RunConfig
from pygent.lara.sessions import AgentMessage, AgentMessageState, SessionManager
from pygent.lara.tracing import EventStore
from pygent.runtime.codec import message_from_dict, message_to_dict
from pygent.tool import ToolSideEffect

from .common import (
    _serialize_tool_payload_for_model,
    _session_dir_for_run,
)
from .media_offload import (
    default_remote_fetcher,
    offload_inline_media,
    offload_remote_media,
)
from .prompts import AgentContextManager, PromptRegistry

MAX_EMPTY_TOOL_FOLLOWUP_RETRIES = 5

EFFECT_FREE_RECOVERY = ExecutionRequirements(
    recovery_safety=RecoverySafety.MODULE_BOUNDARY_RETRY,
    effect_safety=EffectSafety.EFFECT_FREE,
)
MANAGED_EFFECT_RECOVERY = ExecutionRequirements(
    recovery_safety=RecoverySafety.MODULE_BOUNDARY_RETRY,
    effect_safety=EffectSafety.MANAGED_EFFECTS,
)


async def checkpoint_conversation_message(
    config: RunConfig,
    context: LaraContext,
    message: PygentMessage,
    *,
    boundary: str,
) -> bool:
    """Persist one conversation boundary; report whether this call inserted it."""
    checkpoint_id = f"{context.case_run_id}:{context.turn_id}:{boundary}"
    payload = message_to_dict(message)
    # Generated images arrive inline as megabytes of base64 or as remote
    # provider links; persisting them verbatim makes every session open and
    # journal replay reload them all, and remote links keep the image
    # dependent on the host staying reachable. The checkpoint copy references
    # workspace files instead while the in-memory message keeps its original
    # bytes for this run.
    payload = offload_inline_media(payload, workspace_root=config.workspace_root)
    payload = await offload_remote_media(
        payload,
        workspace_root=config.workspace_root,
        fetch_url=default_remote_fetcher,
    )
    metadata = plain_data(context.metadata)
    include_in_session = not (
        isinstance(metadata, dict)
        and metadata.get("persist_conversation_history") is False
    )

    def persist() -> bool:
        manager = SessionManager(config)
        inserted = manager.append_history_checkpoint(
            context.case_run_ref,
            turn_id=context.turn_id,
            checkpoint_id=checkpoint_id,
            message=payload,
            include_in_session=include_in_session,
        )
        if inserted:
            store = EventStore(context.case_run_ref)
            event_type = {
                "assistant": "conversation.assistant_message",
                "tool": "conversation.tool_message",
            }.get(message.role, "conversation.user_message")
            event_actor = message.role
            if message.kind == "lara.automation.trigger":
                event_type = "conversation.automation_trigger"
                event_actor = "system"
            event_payload = {
                **payload,
                "checkpoint_id": checkpoint_id,
            }
            if message.role == "user":
                message_data = plain_data(message.data)
                raw_content = (
                    message_data.get("raw_content")
                    if isinstance(message_data, dict)
                    else None
                )
                event_payload.update(
                    {
                        "raw_content": (
                            raw_content
                            if isinstance(raw_content, str)
                            else message.content
                        ),
                        "user_identity": config.user_identity,
                        "wrapped": message.kind != "lara.automation.trigger",
                        "origin": (
                            "automation"
                            if message.kind == "lara.automation.trigger"
                            else "user"
                        ),
                    }
                )
            store.append(
                event_type,
                actor=event_actor,
                payload=event_payload,
                turn_id=context.turn_id,
            )
        return inserted

    infrastructure = active_infrastructure()
    if infrastructure is None:
        return persist()

    async def operation():
        return freeze_json({"inserted": persist()})

    effect = await infrastructure.execute_effect(
        spec=EffectSpec(
            effect_type="lara.conversation.checkpoint",
            side_effect=EffectSideEffect.WRITE,
            idempotency=EffectIdempotency.INHERENT,
            retry_policy=EffectRetryPolicy.REPLAY_SAFE,
        ),
        request=freeze_json_object(
            {
                "checkpoint_id": checkpoint_id,
                "message": payload,
                "include_in_session": include_in_session,
            }
        ),
        operation=operation,
    )
    value = thaw_json(effect.value)
    return bool(value.get("inserted")) if isinstance(value, dict) else False


def _message_boundary(
    context: PygentContext,
    *,
    phase: str,
    current: PygentMessage,
    output: PygentMessage,
) -> str:
    material = {
        "phase": phase,
        "history": [message_to_dict(item) for item in context.messages],
        "current": message_to_dict(current),
        "output": message_to_dict(output),
    }
    digest = hashlib.sha256(
        json.dumps(material, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]
    return f"{phase}-{digest}"


_READ_PARAMETER_DESCRIPTIONS = {
    "limit": "Maximum number of text lines to return from the starting offset.",
    "offset": "One-based text line number at which to start reading.",
    "pages": (
        "PDF-only one-based page number or inclusive range such as '3' or '1-5'; "
        "at most 20 pages."
    ),
}


def _model_tool_definition(definition: ToolDefinition) -> ToolDefinition:
    if definition.name != "read":
        return definition
    parameters = plain_data(definition.parameters)
    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        return definition
    for name, description in _READ_PARAMETER_DESCRIPTIONS.items():
        parameter = properties.get(name)
        if isinstance(parameter, dict):
            parameter["description"] = description
    return replace(definition, parameters=parameters)


class LaraToolAuthorization(
    Module[ToolAuthorizationRequest, ToolAuthorizationDecision]
):
    execution_requirements = ExecutionRequirements(
        recovery_safety=RecoverySafety.MODULE_BOUNDARY_RETRY,
        effect_safety=EffectSafety.MANAGED_EFFECTS,
    )

    def __init__(
        self,
        *,
        enabled: bool,
        timeout_seconds: float,
        preauthorized_tools: tuple[str, ...],
        interactive: bool,
        scope_key: str,
        detached_tools: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.timeout_seconds = timeout_seconds
        self.preauthorized_tools = preauthorized_tools
        self.interactive = interactive
        self.scope_key = scope_key
        self.detached_tools = detached_tools

    async def forward(
        self,
        request: ToolAuthorizationRequest,
        context: PygentContext,
    ) -> tuple[ToolAuthorizationDecision, PygentContext]:
        name = request.spec.definition.name
        lifecycle: Literal["sync", "detach"] = "detach" if name in self.detached_tools else "sync"
        high_risk = request.spec.side_effect in {
            ToolSideEffect.WRITE,
            ToolSideEffect.EXTERNAL,
        }
        if not self.enabled or not high_risk or name in self.preauthorized_tools:
            return (
                ToolAuthorizationDecision(
                    call_id=request.call.call_id,
                    allowed=True,
                    reason_code="lara_policy_allowed",
                    lifecycle=lifecycle,
                ),
                context,
            )
        scope_key = getattr(context, "case_run_id", None) or self.scope_key
        approval_id = f"{scope_key}:{request.call.call_id}"
        if not self.interactive:
            return (
                ToolAuthorizationDecision(
                    call_id=request.call.call_id,
                    allowed=False,
                    reason_code="approval_required_noninteractive",
                    lifecycle=lifecycle,
                ),
                context,
            )
        decision, _ = await self.gather(
            self.wait_external(
                kind="tool-approval",
                key=approval_id,
                request={"tool_name": name, "call_id": request.call.call_id},
                timeout=self.timeout_seconds,
            ),
            self.emit(
                kind="lara.approval.requested",
                data={
                    "approval_id": approval_id,
                    "call_id": request.call.call_id,
                    "tool_name": name,
                    "tool_id": request.spec.tool_id,
                    "side_effect": request.spec.side_effect.value,
                    "arguments": plain_data(request.call.arguments),
                },
            ),
        )
        approved = bool(decision.get("approved"))
        return (
            ToolAuthorizationDecision(
                call_id=request.call.call_id,
                allowed=approved,
                reason_code="user_approved" if approved else "user_rejected",
                lifecycle=lifecycle,
            ),
            context,
        )


def _context_manager(
    config: RunConfig,
    context: LaraContext,
    prompt_registry: PromptRegistry | None,
) -> AgentContextManager:
    return AgentContextManager(
        session_dir=_session_dir_for_run(Path(context.run_dir)),
        workspace_root=Path(config.workspace_root),
        store=EventStore(context.case_run_ref),
        prompt_registry=prompt_registry,
        system_prompt=config.system_prompt,
        project_lara_root=Path(config.lara_root),
        user_lara_root=Path(config.user_lara_root or Path.home() / ".lara"),
    )


def render_model_request_text(
    config: Any, context: LaraContext, prompt_registry: Any
) -> str:
    """Render the model-request system prompt for one context.

    Shared by :class:`DynamicPromptModule` (managed turns) and the fork-reply
    path so both produce byte-identical prompt text for the same state.
    """

    tool_names = [definition.name for definition in context.tools]
    prompt = _context_manager(config, context, prompt_registry).build_model_request_prompt(
        context=context,
        tool_names=tool_names,
    )
    text = prompt.text
    if context.eternal_memory_enabled:
        text = (
            f"{text}\n\n"
            f"{render_memory_access_instruction(_session_dir_for_run(Path(context.run_dir)), plain_object(context.memory_projection))}"
        )
    return text


class DynamicPromptModule(Module[PygentMessage, PygentMessage]):
    execution_requirements = MANAGED_EFFECT_RECOVERY
    trusted_live_resource_attributes = ("config", "prompt_registry")

    def __init__(
        self, config: RunConfig, prompt_registry: PromptRegistry | None
    ) -> None:
        super().__init__()
        self.config = config
        self.prompt_registry = prompt_registry

    async def forward(
        self, message: PygentMessage, context: LaraContext
    ) -> tuple[PygentMessage, LaraContext]:
        tool_names = [definition.name for definition in context.tools]
        infrastructure = active_infrastructure()
        if infrastructure is None:  # pragma: no cover - managed graph invariant
            raise RuntimeError("dynamic prompt requires managed execution")

        async def operation():
            text = render_model_request_text(self.config, context, self.prompt_registry)
            return freeze_json({"text": text})

        effect = await infrastructure.execute_effect(
            spec=EffectSpec(
                effect_type="lara.prompt.render",
                side_effect=EffectSideEffect.WRITE,
                idempotency=EffectIdempotency.INHERENT,
                retry_policy=EffectRetryPolicy.REPLAY_SAFE,
            ),
            request=freeze_json_object(
                {
                    "message": message_to_dict(message),
                    "history": context.history,
                    "tool_names": tool_names,
                    "memory_projection": plain_data(context.memory_projection),
                }
            ),
            operation=operation,
        )
        result = thaw_json(effect.value)
        if not isinstance(result, dict) or not isinstance(result.get("text"), str):
            raise TypeError("replayed dynamic prompt is invalid")
        text = result["text"]
        return message, replace(context, system_prompt=text)


class ForegroundModelModule(Module[PygentMessage, AIMessage]):
    """Foreground provider call with bounded empty tool-followup recovery."""

    execution_requirements = EFFECT_FREE_RECOVERY

    def __init__(
        self,
        *,
        model: ModelCallLayer,
    ) -> None:
        super().__init__()
        self.model = model

    async def forward(
        self, message: PygentMessage, context: PygentContext
    ) -> tuple[AIMessage, PygentContext]:
        current = message
        history = context
        for retry in range(MAX_EMPTY_TOOL_FOLLOWUP_RETRIES + 1):
            answer, model_context = await self.model(current, history)
            if (
                not isinstance(message, ToolMessage)
                or answer.content
                or answer.tool_calls
            ):
                return answer, model_context
            if retry == MAX_EMPTY_TOOL_FOLLOWUP_RETRIES:
                raise RuntimeError(
                    "empty assistant response after tool result "
                    f"after {MAX_EMPTY_TOOL_FOLLOWUP_RETRIES} retries"
                )
        raise AssertionError("unreachable")


class ConversationCheckpointModelModule(Module[PygentMessage, AIMessage]):
    execution_requirements = MANAGED_EFFECT_RECOVERY
    trusted_live_resource_attributes = ("config", "inner")

    def __init__(
        self, config: RunConfig, inner: Module[PygentMessage, AIMessage]
    ) -> None:
        super().__init__()
        self.config = config
        self.inner = inner

    async def forward(
        self, message: PygentMessage, context: LaraContext
    ) -> tuple[AIMessage, LaraContext]:
        # Pygent can consume several queued user inputs at one ReAct boundary.
        # Persist each consumed input, including those now in context.messages.
        for candidate in (*context.messages, message):
            if candidate.kind != "lara.user.steering":
                continue
            data = plain_data(candidate.data)
            if data.get("case_run_id") != context.case_run_id:
                continue
            inserted = await checkpoint_conversation_message(
                self.config, context, candidate,
                boundary=f"steering-{data['input_id']}",
            )
            if inserted:
                # ReAct drains steering inputs at a step boundary, so this is the
                # boundary the transcript must split on. Announce it once, with
                # the same event family the tool audit uses.
                await self.emit(
                    kind="lara.runtime.message",
                    data={
                        "role": "user",
                        "kind": "lara.user.steering",
                        "content": candidate.content,
                        "message_type": "conversation.user_message",
                        "data": {
                            "input_id": data["input_id"],
                            **(
                                {"attachments": data["attachments"]}
                                if data.get("attachments")
                                else {}
                            ),
                        },
                        "is_delta": False,
                    },
                )
        answer, next_context = await self.inner(message, context)
        await checkpoint_conversation_message(
            self.config,
            next_context,
            answer,
            boundary=_message_boundary(
                context,
                phase="model",
                current=message,
                output=answer,
            ),
        )
        # The composer's context gauge reads the live context size from the
        # stream while the turn runs. The provider input count is the same
        # number pygent records as the context's last_input_tokens, so the
        # live ring and the persisted post-run value agree.
        input_tokens = answer.usage.get("input_tokens")
        if isinstance(input_tokens, int) and not isinstance(input_tokens, bool) and input_tokens > 0:
            await self.emit(
                kind="lara.context.usage",
                data={
                    "session_id": next_context.session_id,
                    "case_run_id": next_context.case_run_id,
                    "turn_id": next_context.turn_id,
                    "last_input_tokens": input_tokens,
                    "compression_count": next_context.compression_count,
                },
            )
        return answer, next_context


class ToolAuditModule(Module[ToolMessage, ToolMessage]):
    execution_requirements = MANAGED_EFFECT_RECOVERY
    trusted_live_resource_attributes = ("config", "bash_tasks")

    def __init__(self, config: RunConfig, *, bash_tasks: BashTaskObservations | None = None) -> None:
        super().__init__()
        self.config = config
        self.bash_tasks = bash_tasks

    async def forward(
        self, message: ToolMessage, context: LaraContext
    ) -> tuple[ToolMessage, LaraContext]:
        infrastructure = active_infrastructure()
        if infrastructure is None:  # pragma: no cover - managed graph invariant
            raise RuntimeError("tool audit requires managed execution")
        assistant = context.messages[-1] if context.messages else None
        calls = (
            {call.call_id: call for call in assistant.tool_calls}
            if isinstance(assistant, AIMessage)
            else {}
        )

        async def operation():
            projected: list[PygentToolResult] = []
            next_context = context
            observer = ToolObserver(
                EventStore(context.case_run_ref),
                workspace_root=self.config.workspace_root,
                track_file_effects=True,
                defer_file_effects=True,
                allow_read_outside_workspace=self.config.allow_read_outside_workspace,
                bash_full_output_allowlist=self.config.bash_full_output_allowlist,
            )
            for result in message.results:
                call = calls.get(result.call_id)
                arguments = plain_data(call.arguments) if call is not None else {}
                if not isinstance(arguments, dict):
                    arguments = {}
                payload, deferred_job = observer.record_framework_result(
                    result.name,
                    arguments,
                    context.turn_id,
                    result,
                    available_tools=tuple(
                        definition.name for definition in context.tools
                    ),
                )
                if deferred_job is not None:
                    next_context = next_context.append_file_effects(deferred_job)
                if result.name == "bash" and result.status == "detached" and self.bash_tasks is not None:
                    self.bash_tasks.track(context, result, payload, arguments)
                await self.emit(
                    kind="lara.runtime.message",
                    data={
                        "role": "tool",
                        "content": _serialize_tool_payload_for_model(payload),
                        "message_type": "conversation.tool_message",
                        "payload": {
                            "role": "tool",
                            "tool_call_id": result.call_id,
                            "name": result.name,
                        },
                        "is_delta": False,
                    },
                )
                projected.append(
                    replace(
                        result,
                        # Keep media blocks native; text reads stay on the
                        # provider's plain output path instead of requiring
                        # structured tool-result content support.
                        content=(
                            ()
                            if result.name == "read"
                            and not any(isinstance(block, MediaBlock) for block in result.content)
                            else result.content
                        ),
                    )
                )
            return freeze_json(
                {
                    "message": message_to_dict(ToolMessage(results=tuple(projected))),
                    "pending_file_effects": [
                        plain_data(item) for item in next_context.pending_file_effects
                    ],
                }
            )

        effect = await infrastructure.execute_effect(
            spec=EffectSpec(
                effect_type="lara.tool.audit",
                side_effect=EffectSideEffect.WRITE,
                idempotency=EffectIdempotency.INHERENT,
                retry_policy=EffectRetryPolicy.REPLAY_SAFE,
            ),
            request=freeze_json_object(
                {
                    "message": message_to_dict(message),
                    "assistant": (
                        message_to_dict(assistant)
                        if isinstance(assistant, AIMessage)
                        else None
                    ),
                    "turn_id": context.turn_id,
                    "available_tools": [
                        definition.name for definition in context.tools
                    ],
                }
            ),
            operation=operation,
        )
        result = thaw_json(effect.value)
        if not isinstance(result, dict):
            raise TypeError("replayed tool audit is invalid")
        projected_message = message_from_dict(result.get("message"))
        if not isinstance(projected_message, ToolMessage):
            raise TypeError("replayed tool audit did not return a ToolMessage")
        raw_jobs = result.get("pending_file_effects", [])
        if not isinstance(raw_jobs, list):
            raise TypeError("replayed tool audit pending effects are invalid")
        jobs = tuple(DeferredFileEffectJob.from_dict(item) for item in raw_jobs)
        next_context = replace(context, pending_file_effects=()).append_file_effects(
            *jobs
        )
        return projected_message, next_context


class RuntimeReminderModule(Module[ToolMessage, ToolMessage]):
    execution_requirements = MANAGED_EFFECT_RECOVERY
    trusted_live_resource_attributes = ("reminders",)

    def __init__(self, reminders: ReminderService) -> None:
        super().__init__()
        self.reminders = reminders

    async def forward(
        self, message: ToolMessage, context: LaraContext
    ) -> tuple[ToolMessage, LaraContext]:
        infrastructure = active_infrastructure()
        if infrastructure is None:  # pragma: no cover - managed graph invariant
            raise RuntimeError("runtime context requires managed execution")
        execution_id = infrastructure.managed_execution_id
        if execution_id is None:
            raise RuntimeError("runtime context requires an execution id")
        identity = hashlib.sha256(
            json.dumps(
                [context.turn_id, [result.call_id for result in message.results]]
            ).encode()
        ).hexdigest()
        claim_id = f"agent-messages:{execution_id}:{identity}"
        assistant = context.messages[-1] if context.messages else None
        calls = (
            {call.call_id: call for call in assistant.tool_calls}
            if isinstance(assistant, AIMessage)
            else {}
        )

        async def operation():
            mutation_indexes = [
                index
                for index, result in enumerate(message.results)
                if result.status == "succeeded"
                and result.name in {"bash", "write", "edit"}
            ]
            bash_commands: list[str] = []
            for index in mutation_indexes:
                result = message.results[index]
                if result.name != "bash":
                    continue
                call = calls.get(result.call_id)
                arguments = plain_data(call.arguments) if call is not None else {}
                if not isinstance(arguments, dict):
                    arguments = {}
                bash_commands.append(str(arguments.get("command") or ""))
            reminder = await self.reminders.observe_after_tools(
                context.session_id,
                bash_commands=tuple(bash_commands),
                file_mutation=bool(mutation_indexes),
                has_results=bool(message.results),
            )
            agent_messages = await self.reminders.claim_agent_messages(
                context.session_id,
                claim_id=claim_id,
            )
            return freeze_json(
                {
                    "content": reminder,
                    "agent_messages": [
                        item.to_dict(include_content=True) for item in agent_messages
                    ],
                }
            )

        effect = await infrastructure.execute_effect(
            spec=EffectSpec(
                effect_type="lara.tool.reminder",
                side_effect=EffectSideEffect.WRITE,
                idempotency=EffectIdempotency.INHERENT,
                retry_policy=EffectRetryPolicy.REPLAY_SAFE,
            ),
            request=freeze_json_object(
                {
                    "message": message_to_dict(message),
                    "assistant": (
                        message_to_dict(assistant)
                        if isinstance(assistant, AIMessage)
                        else None
                    ),
                    "turn_id": context.turn_id,
                    "agent_message_claim_id": claim_id,
                }
            ),
            operation=operation,
        )
        replayed = thaw_json(effect.value)
        if not isinstance(replayed, dict):
            raise TypeError("replayed runtime context is invalid")
        content = replayed.get("content")
        if content is not None and not isinstance(content, str):
            raise TypeError("replayed runtime context content is invalid")
        raw_agent_messages = replayed.get("agent_messages", [])
        if not isinstance(raw_agent_messages, list):
            raise TypeError("replayed agent messages are invalid")
        delivered: list[AgentMessage] = []
        for raw in raw_agent_messages:
            if not isinstance(raw, dict):
                raise TypeError("replayed agent message is invalid")
            candidate = AgentMessage.from_dict(raw)
            claimed = await self.reminders.claimed_agent_message(
                candidate.message_id,
                claim_id=claim_id,
            )
            if claimed is not None:
                delivered.append(claimed)
                continue
            current = await self.reminders.agent_message_status(candidate.message_id)
            if (
                current.state is AgentMessageState.DELIVERED
                and current.delivered_execution_id == execution_id
            ):
                delivered.append(candidate)
        if delivered:
            agent_context = "\n".join(render_agent_message(item) for item in delivered)
            message = replace(
                message,
                content=(
                    agent_context
                    if not message.content
                    else f"{message.content}\n{agent_context}"
                ),
            )
            try:
                for item in delivered:
                    current = await self.reminders.agent_message_status(item.message_id)
                    if current.state is AgentMessageState.DELIVERED:
                        continue
                    await self.reminders.acknowledge_agent_message(
                        item.message_id,
                        claim_id=claim_id,
                        execution_id=execution_id,
                    )
            except BaseException:
                for item in delivered:
                    await self.reminders.release_agent_message(
                        item.message_id,
                        claim_id=claim_id,
                    )
                raise
        if content:
            # Pygent's runtime-context envelope escapes its payload, which would
            # flatten this nested XML body into inert text. Append it like the
            # agent messages so the model sees the rendered markup itself.
            message = replace(
                message,
                content=(
                    content if not message.content else f"{message.content}\n{content}"
                ),
            )
        return message, context


class PersistedDiffModule(Module[ToolMessage, ToolMessage]):
    execution_requirements = EFFECT_FREE_RECOVERY
    trusted_live_resource_attributes = ("workspace_root",)

    def __init__(self, workspace_root: Path, tasks: ToolCallLayer) -> None:
        super().__init__()
        self.workspace_root = workspace_root
        self.tasks = tasks

    async def forward(
        self, message: ToolMessage, context: LaraContext
    ) -> tuple[ToolMessage, LaraContext]:
        jobs, next_context = context.drain_file_effects()
        if jobs:
            batch = DeferredFileEffectBatch.create(
                case_run_ref=context.case_run_ref,
                workspace_root=self.workspace_root,
                jobs=list(jobs),
            )
            await self.tasks(
                AIMessage(
                    tool_calls=(
                        ToolCall(
                            call_id=batch.batch_id,
                            name=FILE_EFFECT_TOOL_SPEC.definition.name,
                            arguments={"batch": batch.to_dict()},
                            tool_id=FILE_EFFECT_TOOL_SPEC.tool_id,
                            tool_version=FILE_EFFECT_TOOL_SPEC.version,
                            idempotency_key=batch.batch_id,
                        ),
                    )
                ),
                replace(context, tools=(FILE_EFFECT_TOOL_SPEC.definition,)),
            )
        return message, next_context


class PreparedToolModule(Module[AIMessage, ToolMessage]):
    execution_requirements = EFFECT_FREE_RECOVERY

    def __init__(
        self,
        *,
        tools: ToolCallLayer,
        audit: ToolAuditModule,
        reminders: RuntimeReminderModule,
        persisted_diff: PersistedDiffModule,
        realtime_nudge: RealtimeNudgeModule | None = None,
    ) -> None:
        super().__init__()
        self.tools = tools
        self.audit = audit
        self.reminders = reminders
        self.persisted_diff = persisted_diff
        self.realtime_nudge = realtime_nudge

    async def forward(
        self, message: AIMessage, context: LaraContext
    ) -> tuple[ToolMessage, LaraContext]:
        if self.audit.bash_tasks is not None and any(call.name == "bash" for call in message.tool_calls):
            await self.audit.bash_tasks.prepare(context)
        tool_message, tool_context = await self.tools(message, context)
        projection_context = tool_context + message
        tool_message, projection_context = await self.audit(
            tool_message, projection_context
        )
        tool_message, projection_context = await self.reminders(
            tool_message, projection_context
        )
        tool_message, projection_context = await self.persisted_diff(
            tool_message, projection_context
        )
        if self.realtime_nudge is not None:
            tool_message, projection_context = await self.realtime_nudge(
                tool_message, projection_context
            )
        return tool_message, replace(
            tool_context,
            pending_file_effects=projection_context.pending_file_effects,
        )


class ConversationCheckpointToolModule(Module[AIMessage, ToolMessage]):
    execution_requirements = MANAGED_EFFECT_RECOVERY
    trusted_live_resource_attributes = ("config", "inner")

    def __init__(
        self, config: RunConfig, inner: Module[AIMessage, ToolMessage]
    ) -> None:
        super().__init__()
        self.config = config
        self.inner = inner

    async def forward(
        self, message: AIMessage, context: LaraContext
    ) -> tuple[ToolMessage, LaraContext]:
        tool_message, next_context = await self.inner(message, context)
        await checkpoint_conversation_message(
            self.config,
            next_context,
            tool_message,
            boundary=_message_boundary(
                context,
                phase="tool",
                current=message,
                output=tool_message,
            ),
        )
        return tool_message, next_context
