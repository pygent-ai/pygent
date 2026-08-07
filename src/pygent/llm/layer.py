"""Provider-neutral model call Module."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

from pygent.core import (
    AIMessage,
    Context,
    EffectIdempotency,
    EffectRetryPolicy,
    EffectSafety,
    EffectSideEffect,
    EffectSpec,
    ExecutionFailure,
    ExecutionRequirements,
    FrozenJsonObject,
    JsonValue,
    Message,
    Module,
    RecoverySafety,
    ToolMessage,
    current_infrastructure,
    freeze_json,
    freeze_json_object,
    thaw_json,
)
from pygent.tool import ToolCall, ToolDefinition

from .adapter import ModelInvoker
from .types import (
    GenerationConfig,
    ModelCallError,
    ModelCallOptions,
    ModelCallPolicy,
    ModelGroupConfig,
    RetryPolicy,
)

_DEFAULT_MODEL_CALL_POLICY = ModelCallPolicy()


class ModelCallLayer(Module[Message, AIMessage]):
    """Immutable model-call declaration executed by a configured invoker.

    The layer projects tool definitions only. It deliberately has no tool
    executor or authorization path.
    """

    execution_requirements = ExecutionRequirements(
        requires_finite_deadline=True,
        recovery_safety=RecoverySafety.MODULE_BOUNDARY_RETRY,
        effect_safety=EffectSafety.MANAGED_EFFECTS,
    )
    trusted_live_resource_attributes = ("invoker",)

    def __init__(
        self,
        *,
        model_group: ModelGroupConfig,
        retry_policy: RetryPolicy,
        generation: GenerationConfig,
        policy: ModelCallPolicy = _DEFAULT_MODEL_CALL_POLICY,
        tools: tuple[ToolDefinition, ...] = (),
        invoker: ModelInvoker | None = None,
    ) -> None:
        super().__init__()
        declared_tools = tuple(tools)
        if any(type(tool) is not ToolDefinition for tool in declared_tools):
            raise TypeError("tools must contain ToolDefinition values")
        names = tuple(tool.name for tool in declared_tools)
        if len(names) != len(set(names)):
            raise ValueError("declared model tools must have unique names")
        if not isinstance(policy, ModelCallPolicy):
            raise TypeError("policy must be a ModelCallPolicy")
        if model_group.is_deferred and invoker is not None:
            raise ValueError("deferred ModelGroup cannot contain a local invoker")
        self.model_group = model_group
        self.retry_policy = retry_policy
        self.generation = generation
        self.policy = policy
        self.tools = declared_tools
        self.invoker = invoker

    def effective_tools(self, context: Context) -> tuple[ToolDefinition, ...]:
        """Return the layer-ordered intersection with context-visible tools."""

        visible_names = {tool.name for tool in context.tools}
        return tuple(tool for tool in self.tools if tool.name in visible_names)

    async def forward(
        self, message: Message, context: Context
    ) -> tuple[AIMessage, Context]:
        async def emit(kind: str, data: FrozenJsonObject) -> None:
            await self.emit(kind=kind, data=data)

        infrastructure = current_infrastructure()
        options_resolver = getattr(infrastructure, "model_call_options", None)
        raw_options = (
            options_resolver(self.model_group.name)
            if callable(options_resolver)
            else {}
        )
        options = ModelCallOptions.from_dict(raw_options)
        if options.profile is not None:
            if not self.model_group.is_deferred:
                raise ValueError(
                    f"fixed model group {self.model_group.name!r} cannot select a profile"
                )
            if not self.policy.allow_profile_override:
                raise ValueError(
                    f"model group {self.model_group.name!r} does not allow profile override"
                )
        for field_name in ("temperature", "max_output_tokens"):
            if (
                getattr(options, field_name) is not None
                and field_name not in self.policy.overridable_generation
            ):
                raise ValueError(
                    f"model group {self.model_group.name!r} does not allow "
                    f"{field_name} override"
                )
        generation = replace(
            self.generation,
            temperature=(
                self.generation.temperature
                if options.temperature is None
                else options.temperature
            ),
            max_output_tokens=(
                self.generation.max_output_tokens
                if options.max_output_tokens is None
                else options.max_output_tokens
            ),
        )
        deployment = None
        model_group = self.model_group
        if self.model_group.is_deferred:
            deployment = infrastructure.resolve_model_deployment(self.model_group.name)
            model_group = cast(Any, deployment).model_group
        effective_tools = self.effective_tools(context)

        async def execute_with(invoker: ModelInvoker) -> JsonValue:
            try:
                model_execution = invoker.execute(
                    model_group=model_group,
                    retry_policy=self.retry_policy,
                    generation=generation,
                    message=message,
                    context=context,
                    tools=effective_tools,
                    deadline=infrastructure.deadline,
                )

                async def relay_events() -> None:
                    async with model_execution.subscribe() as events:
                        async for event in events:
                            await emit(event.kind, freeze_json_object(event.data))

                relay = asyncio.create_task(
                    relay_events(), name="pygent-model-event-relay"
                )
                try:
                    response = await model_execution.result()
                except asyncio.CancelledError:
                    await model_execution.cancel()
                    relay.cancel()
                    await asyncio.gather(relay, return_exceptions=True)
                    raise
                except BaseException:
                    relay.cancel()
                    await asyncio.gather(relay, return_exceptions=True)
                    raise
                else:
                    await relay
            except ModelCallError as exc:
                return freeze_json(
                    {
                        "outcome": "failed",
                        "failure": exc.failure.to_dict(),
                    }
                )
            return freeze_json(
                {
                    "outcome": "succeeded",
                    "message": _message_effect_value(response.message),
                    "usage": thaw_json(cast(JsonValue, response.usage)),
                    "provider_request_id": response.provider_request_id,
                }
            )

        async def invoke() -> JsonValue:
            async with infrastructure.model_permit(
                model_group.capacity_key or model_group.name,
                max_concurrency=model_group.max_concurrency,
            ):
                if deployment is not None:
                    async with infrastructure.model_deployment_lease(deployment) as item:
                        return await execute_with(cast(ModelInvoker, item))
                invoker = self.invoker
                if invoker is None:
                    invoker = cast(
                        ModelInvoker,
                        infrastructure.resolve_model_invoker(model_group.name),
                    )
                return await execute_with(invoker)

        effect = await infrastructure.execute_effect(
            spec=EffectSpec(
                effect_type="model.invoke",
                side_effect=EffectSideEffect.EXTERNAL,
                idempotency=EffectIdempotency.NOT_IDEMPOTENT,
                retry_policy=EffectRetryPolicy.FAIL_CLOSED,
            ),
            request=_model_effect_request(
                self,
                message,
                context,
                effective_tools,
                model_group=model_group,
                generation=generation,
                deployment=deployment,
            ),
            operation=invoke,
        )
        return _message_from_effect(effect.value), context


def _message_effect_value(message: Message) -> dict[str, object]:
    value: dict[str, object] = {
        "role": message.role,
        "content": message.content,
        "slot": message.slot,
        "metadata": thaw_json(cast(JsonValue, message.metadata)),
    }
    if isinstance(message, AIMessage):
        value["tool_calls"] = [
            {
                "call_id": call.call_id,
                "name": call.name,
                "arguments": thaw_json(cast(JsonValue, call.arguments)),
                "tool_id": call.tool_id,
                "tool_version": call.tool_version,
                "idempotency_key": call.idempotency_key,
            }
            for call in message.tool_calls
        ]
    elif isinstance(message, ToolMessage):
        value["results"] = [
            {
                "call_id": result.call_id,
                "name": result.name,
                "status": result.status,
                "output": thaw_json(result.output),
                "error": result.error,
                "error_kind": result.error_kind,
                "error_code": result.error_code,
                "retryable": result.retryable,
                "side_effect_committed": result.side_effect_committed,
                "tool_id": result.tool_id,
                "tool_version": result.tool_version,
                "task": (
                    None
                    if result.task is None
                    else {
                        "task_id": result.task.task_id,
                        "call_id": result.task.call_id,
                        "tool_id": result.task.tool_id,
                        "version": result.task.version,
                        "state": result.task.state.value,
                        "metadata": thaw_json(cast(JsonValue, result.task.metadata)),
                    }
                ),
            }
            for result in message.results
        ]
    return value


def _model_effect_request(
    layer: ModelCallLayer,
    message: Message,
    context: Context,
    tools: tuple[ToolDefinition, ...],
    *,
    model_group: ModelGroupConfig | None = None,
    generation: GenerationConfig | None = None,
    deployment: object | None = None,
) -> FrozenJsonObject:
    generation = generation or layer.generation
    model_group = model_group or layer.model_group
    retry = layer.retry_policy
    return cast(
        FrozenJsonObject,
        freeze_json(
            {
                "model_group": {
                    "name": model_group.name,
                    "routes": [
                        {
                            "route_id": route.route_id,
                            "provider": route.provider,
                            "model": route.model,
                        }
                        for route in model_group.routes
                    ],
                    "fallback": list(model_group.fallback.order),
                    "capacity_key": model_group.capacity_key,
                    "profile": getattr(deployment, "profile", None),
                    "snapshot_id": getattr(deployment, "snapshot_id", None),
                    "deployment_digest": getattr(deployment, "digest", None),
                    "resource_bundle_digest": getattr(
                        deployment, "resource_bundle_digest", None
                    ),
                },
                "retry": {
                    "max_attempts_per_route": retry.max_attempts_per_route,
                    "retry_on": [kind.value for kind in retry.retry_on],
                    "backoff": {
                        "initial": retry.backoff.initial,
                        "maximum": retry.backoff.maximum,
                        "multiplier": retry.backoff.multiplier,
                    },
                },
                "generation": {
                    "temperature": generation.temperature,
                    "max_output_tokens": generation.max_output_tokens,
                    "response_schema": (
                        None
                        if generation.response_schema is None
                        else thaw_json(cast(JsonValue, generation.response_schema))
                    ),
                    "response_schema_name": generation.response_schema_name,
                    "tool_choice": generation.tool_choice,
                },
                "message": _message_effect_value(message),
                "context": {
                    "system_prompt": context.system_prompt,
                    "messages": [
                        _message_effect_value(item) for item in context.messages
                    ],
                    "tools": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": thaw_json(cast(JsonValue, tool.parameters)),
                            "output_schema": (
                                None
                                if tool.output_schema is None
                                else thaw_json(cast(JsonValue, tool.output_schema))
                            ),
                        }
                        for tool in tools
                    ],
                    "metadata": thaw_json(cast(JsonValue, context.metadata)),
                },
            }
        ),
    )


def _message_from_effect(value: JsonValue) -> AIMessage:
    decoded = thaw_json(value)
    if not isinstance(decoded, Mapping):
        raise TypeError("replayed model effect must be a JSON object")
    if decoded.get("outcome") == "failed":
        raise ModelCallError.from_failure(
            ExecutionFailure.from_dict(decoded.get("failure"))
        )
    if decoded.get("outcome") != "succeeded":
        raise TypeError("replayed model effect has an invalid outcome")
    raw_message = decoded.get("message")
    if not isinstance(raw_message, Mapping):
        raise TypeError("replayed model effect message must be a JSON object")
    raw_calls = raw_message.get("tool_calls", [])
    if not isinstance(raw_calls, list):
        raise TypeError("replayed model tool_calls must be an array")
    calls: list[ToolCall] = []
    for raw_call in raw_calls:
        if not isinstance(raw_call, Mapping):
            raise TypeError("replayed model tool call must be a JSON object")
        arguments = raw_call.get("arguments", {})
        if not isinstance(arguments, Mapping):
            raise TypeError("replayed tool arguments must be a JSON object")
        calls.append(
            ToolCall(
                call_id=cast(str, raw_call.get("call_id")),
                name=cast(str, raw_call.get("name")),
                arguments=cast(Mapping[str, Any], arguments),
                tool_id=cast(str | None, raw_call.get("tool_id")),
                tool_version=cast(str | None, raw_call.get("tool_version")),
                idempotency_key=cast(str | None, raw_call.get("idempotency_key")),
            )
        )
    metadata = raw_message.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError("replayed model metadata must be a JSON object")
    return AIMessage(
        content=cast(str, raw_message.get("content", "")),
        slot=cast(str | None, raw_message.get("slot")),
        metadata=cast(Mapping[str, object], metadata),
        tool_calls=tuple(calls),
    )


__all__ = ["ModelCallLayer"]
