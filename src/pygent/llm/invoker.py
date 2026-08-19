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
    ModelProviderCapabilities,
    ModelProviderClient,
    ModelProviderRequest,
    ModelProviderResponse,
    ModelProviderRouteValidator,
    ModelProviderStreamKind,
    ModelProviderStreamPart,
    _attempt_failed_payload,
    _emit,
    _usage_event_payload,
    _validated_canonical_usage,
)
from ._model_execution import ModelExecution, _ProviderStreamOwner
from ._stream_accumulator import ModelStreamAccumulator
from .types import (
    GenerationConfig,
    ModelAttempt,
    ModelCallError,
    ModelErrorKind,
    ModelFailureReason,
    ModelGroupConfig,
    ModelGroupConfigurationError,
    ModelProviderError,
    ModelRoute,
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
        capabilities: Mapping[str, ModelProviderCapabilities] | None = None,
    ) -> None:
        self._adapters = dict(adapters)
        self._clients = dict(clients)
        self._capabilities = dict(capabilities or {})
        self._quarantined_tasks: dict[int, set[asyncio.Future[Any]]] = {}
        self._active_executions: set[asyncio.Task[Any]] = set()
        self._stream_owner_tasks: set[asyncio.Task[None]] = set()
        self._close_task: asyncio.Task[None] | None = None
        self._closing = False

    def validate_route(self, route: ModelRoute) -> None:
        """Validate one route during LLM/application deployment preparation."""

        adapter = self._adapters.get(route.provider)
        client = self._clients.get(route.route_id, self._clients.get(route.provider))
        if adapter is None or client is None:
            raise ModelGroupConfigurationError(
                f"model route {route.route_id!r} has no local provider binding"
            )
        if not route.provider_options:
            return
        if not isinstance(adapter, ModelProviderRouteValidator):
            raise ModelGroupConfigurationError(
                f"provider adapter for route {route.route_id!r} does not validate provider options"
            )
        try:
            adapter.validate_route(route)
        except (TypeError, ValueError, ModelProviderError) as exc:
            raise ModelGroupConfigurationError(str(exc)) from None

    def execute(
        self,
        *,
        model_group: ModelGroupConfig,
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
        model_group: ModelGroupConfig,
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
        route: ModelRoute,
        payload: FrozenJsonObject,
    ) -> _ProviderStreamOwner:
        owner = _ProviderStreamOwner(client, route, payload)
        self._stream_owner_tasks.add(owner.task)

        def release(completed: asyncio.Task[None]) -> None:
            _consume_task_result(completed)
            self._stream_owner_tasks.discard(completed)

        owner.task.add_done_callback(release)
        return owner

    async def _stream_events(
        self,
        *,
        model_group: ModelGroupConfig,
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
        routes = {route.route_id: route for route in model_group.routes}
        order = model_group.fallback.order or tuple(routes)
        attempts: list[ModelAttempt] = []
        last_kind = ModelErrorKind.UNKNOWN
        for route_id in order:
            route = routes[route_id]
            adapter, client = self._resolve(route)
            _validate_route_for_request(adapter, route)
            request = ModelProviderRequest(
                route=route,
                message=message,
                context=context,
                generation=generation,
                tools=tuple(tools),
            )
            payload = adapter.build_request(request)
            for number in range(1, retry_policy.max_attempts_per_route + 1):
                attempt_deadline = _earliest_deadline(
                    deadline, retry_policy.attempt_timeout_seconds
                )
                emitted = False
                completed = False
                attempt_usage = freeze_json_object()
                await _emit(
                    event_sink,
                    ModelEventKind.ATTEMPT_STARTED,
                    {"route_id": route_id, "attempt": number},
                )
                try:
                    self._ensure_client_available(client)
                    async for part in self._transport_events(
                        route=route,
                        adapter=adapter,
                        client=client,
                        request=request,
                        payload=payload,
                        deadline=attempt_deadline,
                        cancel_event=cancel_event,
                    ):
                        part_payload = cast(FrozenJsonObject, part.data).to_dict()
                        part_payload.update({"route_id": route_id, "attempt": number})
                        part = ModelProviderStreamPart(part.kind, part_payload)
                        if part.kind == ModelProviderStreamKind.USAGE:
                            raw_usage = cast(FrozenJsonObject, part.data).to_dict()
                            raw_usage.pop("route_id", None)
                            raw_usage.pop("attempt", None)
                            attempt_usage = _validated_canonical_usage(raw_usage)
                        emitted = emitted or part.kind != ModelProviderStreamKind.FINISH
                        completed = (
                            completed or part.kind == ModelProviderStreamKind.FINISH
                        )
                        yield part
                    if not completed:
                        raise ModelProviderError(
                            ModelErrorKind.INVALID_RESPONSE,
                            "model stream ended before a completion marker",
                        )
                    return
                except asyncio.CancelledError:
                    attempts.append(ModelAttempt(route_id, "cancelled", attempt=number))
                    raise
                except Exception as exc:  # noqa: BLE001 - provider SPI boundary
                    kind = adapter.normalize_error(exc)
                    last_kind = kind
                    reason_code, http_status = _safe_failure_diagnostics(exc, kind)
                    attempts.append(
                        ModelAttempt(
                            route_id,
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
                            route_id=route_id,
                            attempt=number,
                            final=True,
                        ),
                    )
                    await _emit(
                        event_sink,
                        ModelEventKind.ATTEMPT_FAILED,
                        _attempt_failed_payload(
                            route_id=route_id,
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
                    if emitted:
                        raise ModelCallError(
                            "model stream failed after output was emitted",
                            kind=kind,
                            attempts=tuple(attempts),
                            partial_output=True,
                        ) from None
                    if (
                        kind not in retry_policy.retry_on
                        or number >= retry_policy.max_attempts_per_route
                    ):
                        break
                    await _sleep_budget(
                        retry_policy.backoff.delay(number - 1),
                        deadline=deadline,
                        cancel_event=cancel_event,
                    )
        raise ModelCallError(
            _terminal_failure_message(attempts),
            kind=last_kind,
            attempts=tuple(attempts),
        )

    async def _transport_events(
        self,
        *,
        route: ModelRoute,
        adapter: ModelProviderAdapter,
        client: ModelProviderClient,
        request: ModelProviderRequest,
        payload: FrozenJsonObject,
        deadline: float | None,
        cancel_event: asyncio.Event | None,
    ) -> AsyncIterator[ModelProviderStreamPart]:
        capabilities = self._capabilities.get(
            route.route_id,
            self._capabilities.get(route.provider, ModelProviderCapabilities()),
        )
        if capabilities.streaming:
            owner = self._open_stream_owner(client, route, payload)
            try:
                while True:
                    try:
                        raw = await _await_stream_owner(
                            owner,
                            client=client,
                            on_cleanup_stuck=self._quarantine,
                            deadline=deadline,
                            cancel_event=cancel_event,
                        )
                    except StopAsyncIteration:
                        return
                    for part in adapter.parse_stream_events(request, raw):
                        yield part
            finally:
                if not owner.done and not self._is_quarantined(client, owner.task):
                    owner.cancel()
                    cleaned = await _await_cancellation_cleanup(owner.task)
                    if not cleaned:
                        self._quarantine(client, owner.task)
        else:
            raw = await _await_budget(
                client.invoke(route, payload),
                deadline=deadline,
                cancel_event=cancel_event,
                on_cleanup_stuck=lambda task: self._quarantine(client, task),
            )
            response = adapter.parse_response(request, raw)
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
            if response.usage:
                yield ModelProviderStreamPart("usage", response.usage)
            yield ModelProviderStreamPart(
                "finish",
                {
                    "finish_reason": (
                        "tool_calls" if response.message.tool_calls else "stop"
                    ),
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
        model_group: ModelGroupConfig,
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
        self, route: ModelRoute
    ) -> tuple[ModelProviderAdapter, ModelProviderClient]:
        adapter = self._adapters.get(route.provider)
        client = self._clients.get(route.route_id, self._clients.get(route.provider))
        if adapter is None or client is None:
            raise ModelCallError(
                f"model route {route.route_id!r} has no local provider binding",
                kind=ModelErrorKind.INVALID_REQUEST,
            )
        return adapter, client


def _validate_route_for_request(
    adapter: ModelProviderAdapter, route: ModelRoute
) -> None:
    if not route.provider_options:
        return
    if not isinstance(adapter, ModelProviderRouteValidator):
        raise ModelProviderError(
            ModelErrorKind.INVALID_REQUEST,
            "provider adapter does not support route options",
        )
    try:
        adapter.validate_route(route)
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


def _earliest_deadline(
    request_deadline: float | None, attempt_timeout_seconds: float | None
) -> float | None:
    if attempt_timeout_seconds is None:
        return request_deadline
    attempt_deadline = time.monotonic() + attempt_timeout_seconds
    if request_deadline is None:
        return attempt_deadline
    return min(request_deadline, attempt_deadline)


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
    if cancel_event is None:
        try:
            if deadline is None:
                return await owner.next()
            async with asyncio.timeout(max(0.0, deadline - time.monotonic())):
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
            raise ModelProviderError(
                ModelErrorKind.TIMEOUT, "model deadline exceeded"
            ) from None
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
    timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
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
        raise ModelProviderError(ModelErrorKind.TIMEOUT, "model deadline exceeded")
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
