"""Retry, fallback, cancellation, and lifecycle model invoker."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any, cast

from pygent.core import (
    Context,
    FrozenJsonObject,
    Message,
    freeze_json_object,
)
from pygent.tool import ToolDefinition

from ._adapter_contracts import (
    EventSink,
    ModelEventKind,
    ModelProviderAdapter,
    ModelProviderClient,
    ModelProviderRequest,
    ModelProviderResponse,
    ModelProviderSpecValidator,
    ModelProviderStreamKind,
    ModelProviderStreamPart,
    _attempt_failed_payload,
    _emit,
    _usage_event_payload,
    _validated_canonical_usage,
)
from ._model_execution import ModelExecution, _ProviderStreamOwner
from ._request_snapshot import prepared_request_event
from ._stream_accumulator import ModelStreamAccumulator
from .configuration import ModelEntry, ModelGroup, ModelSpec
from .types import (
    GenerationConfig,
    ModelAttempt,
    ModelCallError,
    ModelErrorKind,
    ModelFailureReason,
    ModelGroupConfigurationError,
    ModelProviderError,
    RetryPolicy,
)

_CANCELLATION_CLEANUP_GRACE_SECONDS = 1.0


class DefaultModelInvoker:
    """Bounded model executor with deterministic route/retry/fallback order."""

    def __init__(
        self,
        *,
        adapters: Mapping[str, ModelProviderAdapter],
        clients: Mapping[str, ModelProviderClient],
    ) -> None:
        self._adapters = dict(adapters)
        self._clients = dict(clients)
        self._quarantined_tasks: dict[int, set[asyncio.Future[Any]]] = {}
        self._active_executions: set[asyncio.Task[Any]] = set()
        self._stream_owner_tasks: set[asyncio.Task[None]] = set()
        self._close_task: asyncio.Task[None] | None = None
        self._closing = False

    def validate_model(self, entry: ModelEntry) -> None:
        """Validate one named model during deployment preparation."""

        model = entry.spec
        adapter = self._adapters.get(model.protocol)
        client = self._clients.get(entry.name)
        if adapter is None or client is None:
            raise ModelGroupConfigurationError(
                f"model {entry.name!r} has no local protocol/client binding"
            )
        if not model.provider_options:
            return
        if not isinstance(adapter, ModelProviderSpecValidator):
            raise ModelGroupConfigurationError(
                f"protocol adapter for model {entry.name!r} does not validate provider options"
            )
        try:
            adapter.validate_model(model)
        except (TypeError, ValueError, ModelProviderError) as exc:
            raise ModelGroupConfigurationError(str(exc)) from None

    def execute(
        self,
        *,
        model_group: ModelGroup,
        retry_policy: RetryPolicy,
        generation: GenerationConfig,
        message: Message,
        context: Context,
        tools: tuple[ToolDefinition, ...] = (),
        deadline: float | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> ModelExecution:
        return ModelExecution._from_trusted_operation(
            lambda event_sink: self._execute_registered(
                model_group=model_group,
                retry_policy=retry_policy,
                generation=generation,
                message=message,
                context=context,
                tools=tools,
                deadline=deadline,
                cancel_event=cancel_event,
                event_sink=event_sink,
            )
        )

    async def _execute_registered(self, **kwargs: Any) -> ModelProviderResponse:
        if self._closing:
            raise ModelCallError(
                "model invoker is closing", kind=ModelErrorKind.OUTCOME_UNKNOWN
            )
        task = asyncio.current_task()
        if task is None:
            return await self._execute_with_lifecycle(**kwargs)
        self._active_executions.add(task)
        try:
            return await self._execute_with_lifecycle(**kwargs)
        finally:
            self._active_executions.discard(task)

    async def _execute_with_lifecycle(
        self,
        *,
        model_group: ModelGroup,
        retry_policy: RetryPolicy,
        generation: GenerationConfig,
        message: Message,
        context: Context,
        tools: tuple[ToolDefinition, ...],
        deadline: float | None,
        cancel_event: asyncio.Event | None,
        event_sink: EventSink,
    ) -> ModelProviderResponse:
        await _emit(
            event_sink,
            ModelEventKind.STARTED,
            {"model_group": model_group.name},
        )
        try:
            return await self._reduce_stream(
                model_group=model_group,
                retry_policy=retry_policy,
                generation=generation,
                message=message,
                context=context,
                tools=tools,
                deadline=deadline,
                cancel_event=cancel_event,
                event_sink=event_sink,
            )
        except asyncio.CancelledError:
            await _emit(event_sink, ModelEventKind.CANCELLED, {})
            raise
        except ModelCallError as exc:
            await _emit(
                event_sink,
                ModelEventKind.FAILED,
                {
                    "error_kind": exc.kind.value,
                    "partial_output": exc.partial_output,
                },
            )
            raise
        except Exception as exc:  # noqa: BLE001 - provider SPI boundary
            kind = (
                exc.kind
                if isinstance(exc, ModelProviderError)
                else ModelErrorKind.UNKNOWN
            )
            error = ModelCallError("model call failed", kind=kind)
            await _emit(
                event_sink,
                ModelEventKind.FAILED,
                {"error_kind": kind.value, "partial_output": False},
            )
            raise error from None

    async def aclose(self) -> None:
        """Strictly join execution and stream owners before closing clients."""

        self._closing = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(
                self._close_all(), name="pygent-model-invoker-close"
            )
        await asyncio.shield(self._close_task)

    async def _close_all(self) -> None:
        active = tuple(task for task in self._active_executions if not task.done())
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)
        while True:
            owners = tuple(task for task in self._stream_owner_tasks if not task.done())
            quarantined = tuple(
                task
                for tasks in self._quarantined_tasks.values()
                for task in tasks
                if not task.done()
            )
            pending = tuple(dict.fromkeys((*owners, *quarantined)))
            if not pending:
                break
            for pending_task in pending:
                pending_task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        clients = {id(client): client for client in self._clients.values()}
        await asyncio.gather(*(client.aclose() for client in clients.values()))

    def _quarantine(
        self, client: ModelProviderClient, task: asyncio.Future[Any]
    ) -> None:
        key = id(client)
        tasks = self._quarantined_tasks.setdefault(key, set())
        tasks.add(task)

        def release(completed: asyncio.Future[Any]) -> None:
            _consume_task_result(completed)
            current = self._quarantined_tasks.get(key)
            if current is None:
                return
            current.discard(completed)
            if not current:
                self._quarantined_tasks.pop(key, None)

        task.add_done_callback(release)

    def _ensure_client_available(self, client: ModelProviderClient) -> None:
        if self._closing:
            raise ModelProviderError(
                ModelErrorKind.OUTCOME_UNKNOWN,
                "model invoker is closing",
            )
        if self._quarantined_tasks.get(id(client)):
            raise ModelProviderError(
                ModelErrorKind.OUTCOME_UNKNOWN,
                "model provider client is awaiting cancellation cleanup",
            )

    def _open_stream_owner(
        self,
        client: ModelProviderClient,
        model: ModelSpec,
        payload: FrozenJsonObject,
    ) -> _ProviderStreamOwner:
        owner = _ProviderStreamOwner(client, model, payload)
        self._stream_owner_tasks.add(owner.task)

        def release(completed: asyncio.Task[None]) -> None:
            _consume_task_result(completed)
            self._stream_owner_tasks.discard(completed)

        owner.task.add_done_callback(release)
        return owner

    async def _stream_events(
        self,
        *,
        model_group: ModelGroup,
        retry_policy: RetryPolicy,
        generation: GenerationConfig,
        message: Message,
        context: Context,
        tools: tuple[ToolDefinition, ...] = (),
        deadline: float | None = None,
        cancel_event: asyncio.Event | None = None,
        event_sink: EventSink | None = None,
    ) -> AsyncIterator[ModelProviderStreamPart]:
        _validate_deadline(deadline)
        attempts: list[ModelAttempt] = []
        last_kind = ModelErrorKind.UNKNOWN
        for model_index, entry in enumerate(model_group.models):
            model_key = entry.name
            model = entry.spec
            adapter, client = self._resolve(entry)
            _validate_model_for_request(adapter, model)
            missing_capabilities = _missing_capabilities(
                model,
                generation=generation,
                tools=tools,
            )
            if missing_capabilities:
                await _emit(
                    event_sink,
                    ModelEventKind.CAPABILITY_WARNING,
                    {
                        "model_key": model_key,
                        "provider": model.provider,
                        "model_id": model.model_id,
                        "missing_capabilities": missing_capabilities,
                    },
                )
            request = ModelProviderRequest(
                model_key=model_key,
                model=model,
                message=message,
                context=context,
                generation=generation,
                tools=tuple(tools),
            )
            payload = adapter.build_request(request)
            for number in range(1, retry_policy.max_attempts_per_route + 1):
                prepared_event = prepared_request_event(request, attempt=number)
                emitted = False
                completed = False
                attempt_usage = freeze_json_object()
                await _emit(
                    event_sink,
                    ModelEventKind.ATTEMPT_STARTED,
                    {"model_key": model_key, "attempt": number},
                )
                await _emit(
                    event_sink,
                    ModelEventKind.REQUEST_PREPARED,
                    prepared_event,
                )
                try:
                    self._ensure_client_available(client)
                    async for part in self._transport_events(
                        model=model,
                        adapter=adapter,
                        client=client,
                        request=request,
                        payload=payload,
                        deadline=deadline,
                        idle_timeout_seconds=(
                            retry_policy.attempt_idle_timeout_seconds
                        ),
                        cancel_event=cancel_event,
                    ):
                        if part.kind == ModelProviderStreamKind.FINISH:
                            _raise_for_unsuccessful_finish(
                                cast(FrozenJsonObject, part.data).get("finish_reason")
                            )
                        if part.kind != ModelProviderStreamKind.CONTINUATION:
                            part_payload = cast(FrozenJsonObject, part.data).to_dict()
                            part_payload.update(
                                {"model_key": model_key, "attempt": number}
                            )
                            part = ModelProviderStreamPart(part.kind, part_payload)
                        if part.kind == ModelProviderStreamKind.USAGE:
                            raw_usage = cast(FrozenJsonObject, part.data).to_dict()
                            raw_usage.pop("model_key", None)
                            raw_usage.pop("attempt", None)
                            attempt_usage = _validated_canonical_usage(raw_usage)
                        emitted = emitted or part.kind in {
                            ModelProviderStreamKind.REASONING,
                            ModelProviderStreamKind.TEXT,
                            ModelProviderStreamKind.TOOL_CALL,
                        }
                        completed = (
                            completed or part.kind == ModelProviderStreamKind.FINISH
                        )
                        yield part
                    if not completed:
                        raise ModelProviderError(
                            ModelErrorKind.INVALID_RESPONSE,
                            "model stream ended before a completion marker",
                            reason_code=ModelFailureReason.STREAM_INCOMPLETE,
                        )
                    return
                except asyncio.CancelledError:
                    attempts.append(ModelAttempt(model_key, "cancelled", attempt=number))
                    raise
                except Exception as exc:  # noqa: BLE001 - provider SPI boundary
                    kind = adapter.normalize_error(exc)
                    last_kind = kind
                    reason_code, http_status = _safe_failure_diagnostics(exc, kind)
                    attempts.append(
                        ModelAttempt(
                            model_key,
                            "failed",
                            kind,
                            attempt=number,
                            reason_code=reason_code,
                            http_status=http_status,
                        )
                    )
                    await _emit(
                        event_sink,
                        ModelEventKind.USAGE,
                        _usage_event_payload(
                            attempt_usage,
                            model_key=model_key,
                            attempt=number,
                            final=True,
                        ),
                    )
                    await _emit(
                        event_sink,
                        ModelEventKind.ATTEMPT_FAILED,
                        _attempt_failed_payload(
                            model_key=model_key,
                            attempt=number,
                            kind=kind,
                        ),
                    )
                    if kind is ModelErrorKind.OUTCOME_UNKNOWN:
                        raise ModelCallError(
                            "model provider outcome is unknown after cancellation",
                            kind=kind,
                            attempts=tuple(attempts),
                        ) from None
                    has_budget = deadline is None or time.monotonic() < deadline
                    can_retry = (
                        has_budget
                        and kind in retry_policy.retry_on
                        and number < retry_policy.max_attempts_per_route
                    )
                    can_fallback = has_budget and model_index + 1 < len(model_group.models)
                    retryable_partial = (
                        emitted
                        and reason_code
                        is ModelFailureReason.PROVIDER_IDLE_TIMEOUT
                        and (can_retry or can_fallback)
                    )
                    if emitted and not retryable_partial:
                        raise ModelCallError(
                            "model stream failed after output was emitted",
                            kind=kind,
                            attempts=tuple(attempts),
                            partial_output=True,
                        ) from None
                    if can_retry:
                        try:
                            await _sleep_budget(
                                retry_policy.backoff.delay(number - 1),
                                deadline=deadline,
                                cancel_event=cancel_event,
                            )
                        except ModelProviderError:
                            if emitted:
                                raise ModelCallError(
                                    "model retry budget expired after partial output",
                                    kind=ModelErrorKind.TIMEOUT,
                                    attempts=tuple(attempts),
                                    partial_output=True,
                                ) from None
                            raise
                    if can_retry or can_fallback:
                        yield ModelProviderStreamPart(
                            ModelProviderStreamKind.RESET,
                            {
                                "model_key": model_key,
                                "attempt": number,
                                "public_output": emitted,
                            },
                        )
                    if not can_retry:
                        break
        raise ModelCallError(
            _terminal_failure_message(attempts),
            kind=last_kind,
            attempts=tuple(attempts),
        )

    async def _transport_events(
        self,
        *,
        model: ModelSpec,
        adapter: ModelProviderAdapter,
        client: ModelProviderClient,
        request: ModelProviderRequest,
        payload: FrozenJsonObject,
        deadline: float | None,
        idle_timeout_seconds: float | None,
        cancel_event: asyncio.Event | None,
    ) -> AsyncIterator[ModelProviderStreamPart]:
        if "text" in model.capabilities.streaming.output:
            decoder = adapter.create_stream_decoder(request)
            owner = self._open_stream_owner(client, model, payload)
            try:
                while True:
                    try:
                        raw = await _await_stream_owner(
                            owner,
                            client=client,
                            on_cleanup_stuck=self._quarantine,
                            deadline=deadline,
                            idle_timeout_seconds=idle_timeout_seconds,
                            cancel_event=cancel_event,
                        )
                    except StopAsyncIteration:
                        break
                    for part in decoder.feed(raw):
                        yield part
            finally:
                if not owner.done and not self._is_quarantined(client, owner.task):
                    owner.cancel()
                    cleaned = await _await_cancellation_cleanup(owner.task)
                    if not cleaned:
                        self._quarantine(client, owner.task)
            for part in decoder.finish():
                yield part
        else:
            raw = await _await_budget(
                client.invoke(model, payload),
                deadline=_earliest_deadline(deadline, idle_timeout_seconds),
                cancel_event=cancel_event,
                on_cleanup_stuck=lambda task: self._quarantine(client, task),
            )
            response = adapter.parse_response(request, raw)
            if response.usage:
                yield ModelProviderStreamPart("usage", response.usage)
            _raise_for_unsuccessful_finish(response.finish_reason)
            if response.message.continuation is not None:
                continuation = response.message.continuation
                yield ModelProviderStreamPart(
                    "continuation",
                    {
                        "provider": continuation.provider,
                        "protocol": continuation.protocol,
                        "data": continuation.data,
                    },
                )
            if response.message.content:
                yield ModelProviderStreamPart(
                    "text", {"text": response.message.content}
                )
            for index, call in enumerate(response.message.tool_calls):
                yield ModelProviderStreamPart(
                    "tool_call",
                    {
                        "index": index,
                        "call_id_delta": call.call_id,
                        "name_delta": call.name,
                        "arguments_delta": json.dumps(
                            freeze_json_object(call.arguments).to_dict(),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                )
            yield ModelProviderStreamPart(
                "finish",
                {
                    "finish_reason": response.finish_reason,
                    "provider_request_id": response.provider_request_id,
                },
            )

    def _is_quarantined(
        self, client: ModelProviderClient, task: asyncio.Future[Any]
    ) -> bool:
        return task in self._quarantined_tasks.get(id(client), ())

    async def _reduce_stream(
        self,
        *,
        model_group: ModelGroup,
        retry_policy: RetryPolicy,
        generation: GenerationConfig,
        message: Message,
        context: Context,
        tools: tuple[ToolDefinition, ...],
        deadline: float | None,
        cancel_event: asyncio.Event | None,
        event_sink: EventSink | None,
    ) -> ModelProviderResponse:
        accumulator = ModelStreamAccumulator(generation=generation, tools=tools)
        async for part in self._stream_events(
            model_group=model_group,
            retry_policy=retry_policy,
            generation=generation,
            message=message,
            context=context,
            tools=tools,
            deadline=deadline,
            cancel_event=cancel_event,
            event_sink=event_sink,
        ):
            await accumulator.consume(part, event_sink)
        return await accumulator.finish(event_sink)

    def _resolve(
        self, entry: ModelEntry
    ) -> tuple[ModelProviderAdapter, ModelProviderClient]:
        adapter = self._adapters.get(entry.spec.protocol)
        client = self._clients.get(entry.name)
        if adapter is None or client is None:
            raise ModelCallError(
                f"model {entry.name!r} has no local protocol/client binding",
                kind=ModelErrorKind.INVALID_REQUEST,
            )
        return adapter, client


def _missing_capabilities(
    model: ModelSpec,
    *,
    generation: GenerationConfig,
    tools: tuple[ToolDefinition, ...],
) -> tuple[str, ...]:
    capabilities = model.capabilities
    missing: list[str] = []
    if "text" not in capabilities.modalities.input:
        missing.append("modalities.input.text")
    if "text" not in capabilities.modalities.output:
        missing.append("modalities.output.text")
    if tools and not capabilities.tools.call:
        missing.append("tools.call")
    choice = generation.tool_choice
    if tools and choice is not None:
        required_choice = "named" if choice not in {"none", "auto", "required"} else choice
        if required_choice not in capabilities.tools.choice:
            missing.append(f"tools.choice.{required_choice}")
    if generation.response_schema is not None and not capabilities.structured_output.json_schema:
        missing.append("structured_output.json_schema")
    if (
        generation.max_output_tokens is not None
        and capabilities.limits.max_output_tokens is not None
        and generation.max_output_tokens > capabilities.limits.max_output_tokens
    ):
        missing.append("limits.max_output_tokens")
    return tuple(missing)


def _validate_model_for_request(
    adapter: ModelProviderAdapter, model: ModelSpec
) -> None:
    if not model.provider_options:
        return
    if not isinstance(adapter, ModelProviderSpecValidator):
        raise ModelProviderError(
            ModelErrorKind.INVALID_REQUEST,
            "protocol adapter does not support provider options",
        )
    try:
        adapter.validate_model(model)
    except ModelProviderError:
        raise
    except (TypeError, ValueError) as exc:
        raise ModelProviderError(ModelErrorKind.INVALID_REQUEST, str(exc)) from None


def _safe_failure_diagnostics(
    error: BaseException, kind: ModelErrorKind
) -> tuple[ModelFailureReason | None, int | None]:
    if isinstance(error, ModelProviderError) and error.kind is kind:
        return error.reason_code, error.http_status
    return None, None


def _terminal_failure_message(attempts: list[ModelAttempt]) -> str:
    message = "model stream failed after retry and fallback"
    if not attempts:
        return message
    last = attempts[-1]
    if last.reason_code is None:
        return message
    diagnostic = last.reason_code.value
    if last.http_status is not None:
        diagnostic += f" (HTTP {last.http_status})"
    return f"{message}: {diagnostic}"


def _raise_for_unsuccessful_finish(reason: object) -> None:
    if reason == "length":
        raise ModelProviderError(
            ModelErrorKind.INCOMPLETE_RESPONSE,
            "model output stopped at the output-token limit",
            reason_code=ModelFailureReason.OUTPUT_LIMIT_REACHED,
        )
    if reason == "content_filter":
        raise ModelProviderError(
            ModelErrorKind.INVALID_RESPONSE,
            "model output was rejected by the provider content policy",
            reason_code=ModelFailureReason.CONTENT_POLICY_REJECTED,
        )


def _earliest_deadline(
    absolute_deadline: float | None, timeout_seconds: float | None
) -> float | None:
    if timeout_seconds is None:
        return absolute_deadline
    timeout_deadline = time.monotonic() + timeout_seconds
    if absolute_deadline is None:
        return timeout_deadline
    return min(absolute_deadline, timeout_deadline)


def _validate_deadline(deadline: float | None) -> None:
    if deadline is not None and (
        not isinstance(deadline, (int, float))
        or not deadline > 0
        or deadline == float("inf")
    ):
        raise ValueError("deadline must be a finite absolute monotonic time")


def _check_budget(deadline: float | None, cancel_event: asyncio.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise asyncio.CancelledError
    if deadline is not None and time.monotonic() >= deadline:
        raise ModelProviderError(ModelErrorKind.TIMEOUT, "model deadline exceeded")


async def _await_budget(
    awaitable: Awaitable[Any],
    *,
    deadline: float | None,
    cancel_event: asyncio.Event | None,
    on_cleanup_stuck: Callable[[asyncio.Future[Any]], None] | None = None,
) -> Any:
    try:
        _check_budget(deadline, cancel_event)
    except BaseException:
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()
        raise
    task = asyncio.ensure_future(awaitable)
    cancel_task = (
        asyncio.create_task(cancel_event.wait()) if cancel_event is not None else None
    )
    timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
    waiters = {task}
    if cancel_task is not None:
        waiters.add(cancel_task)
    try:
        done, _ = await asyncio.wait(
            waiters, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
        )
        if task in done:
            return await task
        task.cancel()
        cleaned = await _await_cancellation_cleanup(task)
        if cancel_task is not None and cancel_task in done:
            if not cleaned and on_cleanup_stuck is not None:
                on_cleanup_stuck(task)
            raise asyncio.CancelledError
        if not cleaned:
            if on_cleanup_stuck is not None:
                on_cleanup_stuck(task)
            raise ModelProviderError(
                ModelErrorKind.OUTCOME_UNKNOWN,
                "model provider outcome is unknown after cancellation",
            )
        raise ModelProviderError(ModelErrorKind.TIMEOUT, "model deadline exceeded")
    except asyncio.CancelledError:
        task.cancel()
        cleaned = await _await_cancellation_cleanup(task)
        if not cleaned and on_cleanup_stuck is not None:
            on_cleanup_stuck(task)
        raise
    finally:
        if cancel_task is not None:
            cancel_task.cancel()
            await asyncio.gather(cancel_task, return_exceptions=True)


async def _await_stream_owner(
    owner: _ProviderStreamOwner,
    *,
    client: ModelProviderClient,
    on_cleanup_stuck: Callable[[ModelProviderClient, asyncio.Future[Any]], None],
    deadline: float | None,
    idle_timeout_seconds: float | None,
    cancel_event: asyncio.Event | None,
) -> FrozenJsonObject:
    try:
        _check_budget(deadline, cancel_event)
    except asyncio.CancelledError:
        owner.cancel()
        cleaned = await _await_cancellation_cleanup(owner.task)
        if not cleaned:
            on_cleanup_stuck(client, owner.task)
        raise
    except ModelProviderError:
        owner.cancel()
        cleaned = await _await_cancellation_cleanup(owner.task)
        if not cleaned:
            on_cleanup_stuck(client, owner.task)
            raise ModelProviderError(
                ModelErrorKind.OUTCOME_UNKNOWN,
                "model provider outcome is unknown after cancellation",
            ) from None
        raise
    idle_deadline = (
        None
        if idle_timeout_seconds is None
        else time.monotonic() + idle_timeout_seconds
    )
    idle_limited = idle_deadline is not None and (
        deadline is None or idle_deadline <= deadline
    )
    wait_deadline = (
        idle_deadline
        if deadline is None
        else deadline
        if idle_deadline is None
        else min(deadline, idle_deadline)
    )
    if cancel_event is None:
        try:
            if wait_deadline is None:
                return await owner.next()
            async with asyncio.timeout(
                max(0.0, wait_deadline - time.monotonic())
            ):
                return await owner.next()
        except TimeoutError:
            owner.cancel()
            cleaned = await _await_cancellation_cleanup(owner.task)
            if not cleaned:
                on_cleanup_stuck(client, owner.task)
                raise ModelProviderError(
                    ModelErrorKind.OUTCOME_UNKNOWN,
                    "model provider outcome is unknown after cancellation",
                ) from None
            raise _stream_timeout_error(idle_limited) from None
        except asyncio.CancelledError:
            owner.cancel()
            cleaned = await _await_cancellation_cleanup(owner.task)
            if not cleaned:
                on_cleanup_stuck(client, owner.task)
            raise
    next_task = asyncio.create_task(owner.next(), name="pygent-model-stream-next")
    cancel_task = (
        asyncio.create_task(cancel_event.wait()) if cancel_event is not None else None
    )
    waiters: set[asyncio.Future[Any]] = {next_task}
    if cancel_task is not None:
        waiters.add(cancel_task)
    timeout = (
        None
        if wait_deadline is None
        else max(0.0, wait_deadline - time.monotonic())
    )
    try:
        done, _ = await asyncio.wait(
            waiters, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
        )
        if next_task in done:
            return await next_task
        owner.cancel()
        cleaned = await _await_cancellation_cleanup(owner.task)
        if not cleaned:
            on_cleanup_stuck(client, owner.task)
        if cancel_task is not None and cancel_task in done:
            raise asyncio.CancelledError
        if not cleaned:
            raise ModelProviderError(
                ModelErrorKind.OUTCOME_UNKNOWN,
                "model provider outcome is unknown after cancellation",
            )
        raise _stream_timeout_error(idle_limited)
    except asyncio.CancelledError:
        owner.cancel()
        cleaned = await _await_cancellation_cleanup(owner.task)
        if not cleaned:
            on_cleanup_stuck(client, owner.task)
        raise
    finally:
        if not next_task.done():
            next_task.cancel()
        if cancel_task is not None and not cancel_task.done():
            cancel_task.cancel()
        await asyncio.gather(
            next_task,
            *((cancel_task,) if cancel_task is not None else ()),
            return_exceptions=True,
        )


def _stream_timeout_error(idle_limited: bool) -> ModelProviderError:
    if idle_limited:
        return ModelProviderError(
            ModelErrorKind.TIMEOUT,
            "model provider stream idle timeout exceeded",
            reason_code=ModelFailureReason.PROVIDER_IDLE_TIMEOUT,
        )
    return ModelProviderError(ModelErrorKind.TIMEOUT, "model deadline exceeded")


async def _await_cancellation_cleanup(task: asyncio.Future[Any]) -> bool:
    """Give cancellation its own bounded acknowledgement window.

    The operation deadline has already expired when this helper runs. Reusing
    it would collapse cooperative cleanup to a zero-length poll and make the
    error classification depend on event-loop scheduling.
    """

    grace = _CANCELLATION_CLEANUP_GRACE_SECONDS
    if task.done():
        _consume_task_result(task)
        return True
    if grace > 0:
        done, _ = await asyncio.wait({task}, timeout=grace)
        if task in done:
            _consume_task_result(task)
            return True
    task.add_done_callback(_consume_task_result)
    return False


def _consume_task_result(task: asyncio.Future[Any]) -> None:
    if not task.done():
        return
    try:
        task.exception()
    except asyncio.CancelledError:
        return


async def _sleep_budget(
    delay: float,
    *,
    deadline: float | None,
    cancel_event: asyncio.Event | None,
) -> None:
    if delay <= 0:
        _check_budget(deadline, cancel_event)
        return
    await _await_budget(
        asyncio.sleep(delay), deadline=deadline, cancel_event=cancel_event
    )
