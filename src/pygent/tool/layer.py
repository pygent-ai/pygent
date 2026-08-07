"""ToolCallLayer admission and ordered execution orchestration."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Mapping
from dataclasses import replace
from typing import Any, Protocol, cast
from uuid import uuid4

from jsonschema import (  # type: ignore[import-untyped]
    Draft202012Validator,
    ValidationError,
)

from pygent.core import (
    AIMessage,
    Context,
    EffectIdempotency,
    EffectRetryPolicy,
    EffectSafety,
    EffectSideEffect,
    EffectSpec,
    ExecutionRequirements,
    FrozenJsonObject,
    JsonValue,
    Module,
    RecoverySafety,
    ToolMessage,
    current_infrastructure,
    freeze_json,
    independent_execution,
    thaw_json,
)

from .executors import (
    ExecutorRegistry,
    ToolExecutionContext,
    ToolRunner,
    ToolTaskManager,
    result_from_exception,
)
from .types import (
    IdempotencyPolicy,
    ToolAuthorizationDecision,
    ToolAuthorizationRequest,
    ToolCall,
    ToolDefinition,
    ToolResult,
    ToolSideEffect,
    ToolSpec,
    ToolTask,
    ToolTaskState,
)


class TrustedAuthorizationAdapter(Protocol):
    """Deployment-trusted alternative to an application authorization Module."""

    def __call__(
        self, request: ToolAuthorizationRequest, context: Context
    ) -> ToolAuthorizationDecision | Awaitable[ToolAuthorizationDecision]: ...


class ToolCallLayer(Module[AIMessage, ToolMessage]):
    """Validate, authorize, admit, and execute one ordered batch of tool calls."""

    execution_requirements = ExecutionRequirements(
        recovery_safety=RecoverySafety.MODULE_BOUNDARY_RETRY,
        effect_safety=EffectSafety.MANAGED_EFFECTS,
    )
    trusted_live_resource_attributes = (
        "authorization_adapter",
        "executor_registry",
        "task_manager",
    )

    def __init__(
        self,
        *,
        tools: tuple[ToolSpec, ...],
        authorization: Module[ToolAuthorizationRequest, ToolAuthorizationDecision]
        | None = None,
        authorization_adapter: TrustedAuthorizationAdapter | None = None,
        executor_registry: ExecutorRegistry | None = None,
        task_manager: ToolTaskManager | None = None,
        max_concurrency: int | None = None,
    ) -> None:
        super().__init__()
        specs = tuple(tools)
        if any(type(spec) is not ToolSpec for spec in specs):
            raise TypeError("tools must contain only ToolSpec values")
        identities = [(spec.tool_id, spec.version) for spec in specs]
        names = [spec.definition.name for spec in specs]
        if len(identities) != len(set(identities)):
            raise ValueError("tools contains duplicate (tool_id, version) identities")
        if len(names) != len(set(names)):
            raise ValueError("tools contains duplicate model-visible names")
        if authorization is not None and authorization_adapter is not None:
            raise ValueError(
                "configure either authorization or authorization_adapter, not both"
            )
        if max_concurrency is not None and (
            isinstance(max_concurrency, bool)
            or not isinstance(max_concurrency, int)
            or max_concurrency <= 0
        ):
            raise ValueError("layer max_concurrency must be greater than zero")
        self.tools = specs
        self.authorization = authorization
        self.authorization_adapter = authorization_adapter
        self.executor_registry = executor_registry
        self.task_manager = task_manager
        self.max_concurrency = max_concurrency
        if authorization_adapter is not None:
            # An arbitrary trusted callback is useful for deployment-local
            # authorization, but Runtime cannot verify that it is deterministic
            # or effect-free across process recovery.
            self.execution_requirements = ExecutionRequirements(
                recovery_safety=RecoverySafety.MODULE_BOUNDARY_RETRY,
                effect_safety=EffectSafety.UNDECLARED,
            )

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        """Model-visible definitions without execution or authorization policy."""

        return tuple(spec.definition for spec in self.tools)

    async def forward(
        self, message: AIMessage, context: Context
    ) -> tuple[ToolMessage, Context]:
        if not isinstance(message, AIMessage):
            raise TypeError("ToolCallLayer expects an AIMessage")
        if not isinstance(context, Context):
            raise TypeError("ToolCallLayer expects a Context")

        visible_names = {definition.name for definition in context.tools}
        specs_by_name = {
            spec.definition.name: spec
            for spec in self.tools
            if spec.definition.name in visible_names
        }
        call_id_counts: dict[str, int] = {}
        for call in message.tool_calls:
            call_id_counts[call.call_id] = call_id_counts.get(call.call_id, 0) + 1
        duplicate_call_ids = {
            call_id for call_id, count in call_id_counts.items() if count > 1
        }
        semaphore = (
            asyncio.Semaphore(self.max_concurrency)
            if self.max_concurrency is not None
            else None
        )

        async def run(call: ToolCall) -> ToolResult:
            await self.emit(
                kind="tool.requested",
                data={"call_id": call.call_id, "name": call.name, "call_span_id": str(uuid4())},
            )
            if call.call_id in duplicate_call_ids:
                result = self._invalid_call(
                    call, "duplicate_call_id", specs_by_name.get(call.name)
                )
                await self.emit(
                    kind="tool.rejected",
                    data={"call_id": call.call_id, "reason_code": "duplicate_call_id"},
                )
                return result
            if semaphore is None:
                return await self._handle_call(call, context, specs_by_name)
            async with semaphore:
                return await self._handle_call(call, context, specs_by_name)

        # The scope registers a structured parallel group and preserves input order.
        infrastructure = current_infrastructure()
        operations = tuple(
            (lambda call=call: run(call)) for call in message.tool_calls
        )
        gather = getattr(infrastructure, "gather", None)
        results = (
            await gather(operations)
            if callable(gather)
            else tuple(await asyncio.gather(*(operation() for operation in operations)))
        )
        return ToolMessage(results=cast(tuple[ToolResult, ...], results)), context

    async def _handle_call(
        self,
        call: ToolCall,
        context: Context,
        specs_by_name: Mapping[str, ToolSpec],
    ) -> ToolResult:
        spec = specs_by_name.get(call.name)
        if spec is None:
            return await self._reject_event(call, "tool_not_visible")
        if call.tool_id is not None and call.tool_id != spec.tool_id:
            return await self._reject_event(call, "tool_identity_mismatch", spec)
        if call.tool_version is not None and call.tool_version != spec.version:
            return await self._reject_event(call, "tool_version_mismatch", spec)

        try:
            Draft202012Validator(
                cast(
                    dict[str, Any],
                    thaw_json(cast(FrozenJsonObject, spec.definition.parameters)),
                )
            ).validate(thaw_json(cast(FrozenJsonObject, call.arguments)))
        except ValidationError as exc:
            await self.emit(
                kind="tool.rejected",
                data={"call_id": call.call_id, "reason_code": "invalid_arguments"},
            )
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                status="rejected",
                error=exc.message,
                error_kind="validation_error",
                error_code="invalid_arguments",
                retryable=False,
                side_effect_committed=False,
                tool_id=spec.tool_id,
                tool_version=spec.version,
            )

        try:
            decision = await self._authorize(call, spec, context)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - trusted authorization boundary
            return await self._reject_event(call, "authorization_failed", spec)
        if decision.call_id != call.call_id:
            return await self._reject_event(call, "authorization_call_id_mismatch", spec)
        if not decision.allowed:
            return await self._reject_event(call, decision.reason_code, spec)

        await self.emit(
            kind="tool.authorized",
            data={"call_id": call.call_id, "lifecycle": decision.lifecycle},
        )

        admitted_call = call
        if spec.idempotency is IdempotencyPolicy.REQUIRES_KEY:
            infrastructure = current_infrastructure()
            key = call.idempotency_key
            if key is None:
                key = infrastructure.tool_idempotency_key(call.call_id)
            if key is None:
                return await self._reject_event(call, "idempotency_key_required", spec)
            admitted_call = replace(call, idempotency_key=key)

        if decision.lifecycle == "detach":
            result = await self._detach(admitted_call, spec)
            await self.emit(kind="tool.detached", data={"call_id": call.call_id})
            return result
        return await self._execute_sync(admitted_call, spec)

    async def _reject_event(
        self, call: ToolCall, reason: str, spec: ToolSpec | None = None
    ) -> ToolResult:
        await self.emit(
            kind="tool.rejected",
            data={"call_id": call.call_id, "reason_code": reason},
        )
        return self._rejected(call, reason, spec)

    async def _authorize(
        self, call: ToolCall, spec: ToolSpec, context: Context
    ) -> ToolAuthorizationDecision:
        permissions_value = cast(FrozenJsonObject, context.metadata).get(
            "permissions", ()
        )
        permissions = (
            tuple(value for value in permissions_value if isinstance(value, str))
            if isinstance(permissions_value, tuple)
            else ()
        )
        request = ToolAuthorizationRequest(
            call=call,
            spec=spec,
            permissions=permissions,
        )
        if self.authorization is not None:
            decision, _ = await self.authorization(request, context)
        elif self.authorization_adapter is not None:
            decision_value = self.authorization_adapter(request, context)
            decision = (
                await decision_value
                if inspect.isawaitable(decision_value)
                else decision_value
            )
        else:
            return ToolAuthorizationDecision(
                call_id=call.call_id,
                allowed=False,
                reason_code="authorization_not_configured",
            )
        if not isinstance(decision, ToolAuthorizationDecision):
            raise TypeError("authorization must return ToolAuthorizationDecision")
        return decision

    async def _execute_sync(self, call: ToolCall, spec: ToolSpec) -> ToolResult:
        infrastructure = current_infrastructure()
        task = ToolTask(
            task_id=f"tool-{uuid4()}",
            call_id=call.call_id,
            tool_id=spec.tool_id,
            version=spec.version,
            state=ToolTaskState.RUNNING,
        )
        registry = self.executor_registry
        if registry is None:
            registry = cast(
                ExecutorRegistry | None,
                infrastructure.resolve_tool_registry(),
            )
        if registry is None:
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                status="failed",
                task=replace(task, state=ToolTaskState.FAILED),
                error="no direct executor registry is configured",
                error_kind="executor_unavailable",
                error_code="executor_not_configured",
                retryable=False,
                side_effect_committed=False,
                tool_id=spec.tool_id,
                tool_version=spec.version,
            )
        try:

            async def execute() -> JsonValue:
                try:
                    async with infrastructure.tool_permit(spec.resource_key):
                        async def emit_tool_event(
                            kind: str, data: Mapping[str, JsonValue]
                        ) -> None:
                            await self.emit(kind=kind, data=data)

                        tool_result = await ToolRunner().execute(
                            spec,
                            call,
                            registry,
                            context=ToolExecutionContext(
                                deadline=infrastructure.deadline,
                                emit=emit_tool_event,
                            ),
                            task=task,
                        ).result()
                    if tool_result.status != "succeeded":
                        return freeze_json(
                            {
                                "outcome": "failed",
                                "task_id": task.task_id,
                                "result": _result_effect_value(tool_result),
                            }
                        )
                    return freeze_json(
                        {
                            "outcome": "succeeded",
                            "task_id": task.task_id,
                            "output": tool_result.output,
                        }
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - durable error envelope
                    return freeze_json(
                        {
                            "outcome": "failed",
                            "task_id": task.task_id,
                            "result": _result_effect_value(
                                result_from_exception(spec, call, exc, task=task)
                            ),
                        }
                    )

            async def execute_effect() -> JsonValue:
                outcome = await infrastructure.execute_effect(
                    spec=EffectSpec(
                        effect_type="tool.execute",
                        side_effect=EffectSideEffect(spec.side_effect.value),
                        idempotency=EffectIdempotency(spec.idempotency.value),
                        retry_policy=(
                            EffectRetryPolicy.REPLAY_SAFE
                            if (
                                spec.side_effect in (
                                    ToolSideEffect.PURE,
                                    ToolSideEffect.READ,
                                )
                                or spec.idempotency
                                is not IdempotencyPolicy.NOT_IDEMPOTENT
                            )
                            else EffectRetryPolicy.FAIL_CLOSED
                        ),
                        idempotency_key=call.idempotency_key,
                    ),
                    request={
                        "tool_id": spec.tool_id,
                        "version": spec.version,
                        "call_id": call.call_id,
                        "name": call.name,
                        "arguments": cast(JsonValue, call.arguments),
                        "idempotency_key": call.idempotency_key,
                    },
                    operation=execute,
                )
                return outcome.value

            effect = thaw_json(await execute_effect())
            if not isinstance(effect, Mapping):
                raise TypeError("replayed tool effect must be a JSON object")
            task_id = effect.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                raise TypeError("replayed tool effect task_id must be non-empty")
            task = replace(task, task_id=task_id)
            if effect.get("outcome") == "failed":
                raw_result = effect.get("result")
                if not isinstance(raw_result, Mapping):
                    raise TypeError("replayed tool failure must be a JSON object")
                return _result_from_effect(raw_result, spec, call, task)
            if effect.get("outcome") != "succeeded":
                raise TypeError("replayed tool effect has an invalid outcome")
            output = freeze_json(effect.get("output"))
            if spec.definition.output_schema is not None:
                Draft202012Validator(
                    cast(
                        dict[str, Any],
                        thaw_json(
                            cast(FrozenJsonObject, spec.definition.output_schema)
                        ),
                    )
                ).validate(thaw_json(output))
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                status="succeeded",
                task=replace(task, state=ToolTaskState.SUCCEEDED),
                output=freeze_json(output),
                side_effect_committed=True,
                tool_id=spec.tool_id,
                tool_version=spec.version,
            )
        except ValidationError as exc:
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                status="failed",
                task=replace(task, state=ToolTaskState.FAILED),
                error=exc.message,
                error_kind="validation_error",
                error_code="invalid_output",
                retryable=False,
                side_effect_committed=True,
                tool_id=spec.tool_id,
                tool_version=spec.version,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - converted to a public result
            result = result_from_exception(spec, call, exc, task=task)
            state = (
                ToolTaskState.UNKNOWN
                if result.status == "unknown"
                else ToolTaskState.FAILED
            )
            return replace(result, task=replace(task, state=state))

    async def _detach(self, call: ToolCall, spec: ToolSpec) -> ToolResult:
        infrastructure = current_infrastructure()
        runtime_submit = getattr(infrastructure, "submit_tool_task", None)
        task = None
        # asyncio Tasks inherit ContextVars. Clear the Parent execution scope
        # while task admission creates its owner Task so a detached Agent-backed
        # executor cannot accidentally re-enter the former structured tree.
        with independent_execution():
            if callable(runtime_submit):
                task = cast(ToolTask | None, await runtime_submit(spec, call))
            if task is None and self.task_manager is not None:
                task = await self.task_manager.submit(spec, call)
        if task is None:
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                status="rejected",
                error="detach requires a managed executiontime or explicit task facility",
                error_kind="capability_error",
                error_code="detach_unavailable",
                retryable=False,
                side_effect_committed=False,
                tool_id=spec.tool_id,
                tool_version=spec.version,
            )
        return ToolResult(
            call_id=call.call_id,
            name=call.name,
            status="detached",
            task=task,
            side_effect_committed=False,
            tool_id=spec.tool_id,
            tool_version=spec.version,
        )

    @staticmethod
    def _invalid_call(
        call: ToolCall, reason_code: str, spec: ToolSpec | None = None
    ) -> ToolResult:
        return ToolResult(
            call_id=call.call_id,
            name=call.name,
            status="rejected",
            error_kind="validation_error",
            error_code=reason_code,
            retryable=False,
            side_effect_committed=False,
            tool_id=None if spec is None else spec.tool_id,
            tool_version=None if spec is None else spec.version,
        )

    @staticmethod
    def _rejected(
        call: ToolCall, reason_code: str, spec: ToolSpec | None = None
    ) -> ToolResult:
        return ToolResult(
            call_id=call.call_id,
            name=call.name,
            status="rejected",
            error_kind="authorization_error",
            error_code=reason_code,
            retryable=False,
            side_effect_committed=False,
            tool_id=None if spec is None else spec.tool_id,
            tool_version=None if spec is None else spec.version,
        )


def _result_effect_value(result: ToolResult) -> dict[str, object]:
    return {
        "status": result.status,
        "error": result.error,
        "error_kind": result.error_kind,
        "error_code": result.error_code,
        "retryable": result.retryable,
        "side_effect_committed": result.side_effect_committed,
    }


def _result_from_effect(
    value: Mapping[str, object],
    spec: ToolSpec,
    call: ToolCall,
    task: ToolTask,
) -> ToolResult:
    status = value.get("status")
    if status not in ("failed", "unknown"):
        raise TypeError("replayed tool failure has an invalid status")
    error = value.get("error")
    error_kind = value.get("error_kind")
    error_code = value.get("error_code")
    retryable = value.get("retryable")
    committed = value.get("side_effect_committed")
    if error is not None and not isinstance(error, str):
        raise TypeError("replayed tool error must be a string or null")
    if error_kind is not None and not isinstance(error_kind, str):
        raise TypeError("replayed tool error_kind must be a string or null")
    if error_code is not None and not isinstance(error_code, str):
        raise TypeError("replayed tool error_code must be a string or null")
    if not isinstance(retryable, bool):
        raise TypeError("replayed tool retryable must be a bool")
    if committed is not None and not isinstance(committed, bool):
        raise TypeError("replayed side_effect_committed must be a bool or null")
    state = ToolTaskState.UNKNOWN if status == "unknown" else ToolTaskState.FAILED
    return ToolResult(
        call_id=call.call_id,
        name=call.name,
        status=cast(Any, status),
        task=replace(task, state=state),
        error=error,
        error_kind=error_kind,
        error_code=error_code,
        retryable=retryable,
        side_effect_committed=committed,
        tool_id=spec.tool_id,
        tool_version=spec.version,
    )


__all__ = ["ToolCallLayer", "TrustedAuthorizationAdapter"]
