from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from typing import cast

from pygent import (
    Context,
    GenerationConfig,
    ToolDefinition,
    ToolMessage,
    ToolResult,
    UserMessage,
)
from pygent.core import FrozenJsonObject, freeze_json_object
from pygent.llm import (
    CapabilityPresetCatalog,
    ModelErrorKind,
    ModelFailureReason,
    ModelProviderClient,
    ModelProviderError,
    ModelProviderRequest,
    ModelSpec,
    OpenAICompatibleAdapter,
    OpenAIResponsesAdapter,
)
from tests.live.az_conformance.results import ErrorKind, ProbeResult
from tests.live.az_conformance.runner import ProbeContext
from tests.live.az_conformance.schemas import AzRoute

_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1Pe"
    "AAAADElEQVR4nGNgYPgPAAEDAQAIicLsAAAAAElFTkSuQmCC"
)
_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}
_TOOL = ToolDefinition(
    name="lookup",
    description="Return the supplied probe value.",
    parameters={
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    },
)


def _spec(
    route: AzRoute, *, provider_options: dict[str, object] | None = None
) -> ModelSpec:
    capabilities = CapabilityPresetCatalog.builtin().presets[
        "text_tools_structured_reasoning"
    ].materialize(context_tokens=1_000_000, max_output_tokens=4096)
    return ModelSpec(
        provider=cast(str, route.canonical_provider),
        model_id=cast(str, route.canonical_model_id),
        protocol="openai_chat_completions",
        provider_options=provider_options or {},
        capabilities=capabilities,
    )


def _request(
    route: AzRoute,
    *,
    message=None,
    context: Context | None = None,
    generation: GenerationConfig | None = None,
    tools: tuple[ToolDefinition, ...] = (),
    provider_options: dict[str, object] | None = None,
) -> ModelProviderRequest:
    return ModelProviderRequest(
        model_key=route.route_id,
        model=_spec(route, provider_options=provider_options),
        message=message or UserMessage(content="Reply with a short answer."),
        context=context or Context(),
        generation=generation or GenerationConfig(max_output_tokens=32),
        tools=tools,
    )


def _wire_payload(
    adapter: OpenAICompatibleAdapter,
    request: ModelProviderRequest,
    route: AzRoute,
) -> FrozenJsonObject:
    body = adapter.build_request(request).to_dict()
    body["model"] = route.route_id
    return freeze_json_object(body)


def _client(context: ProbeContext) -> ModelProviderClient:
    if context.client is None:
        raise ValueError("probe client is not configured")
    return cast(ModelProviderClient, context.client)


def _passed(context: ProbeContext, route: AzRoute) -> ProbeResult:
    return ProbeResult(
        snapshot_sha256=context.snapshot_sha256,
        source_revision=context.source_revision,
        route_id=route.route_id,
        canonical_provider=route.canonical_provider,
        canonical_model_id=route.canonical_model_id,
        protocol=context.protocol,
        scenario=context.scenario,
        status="passed",
        attempts=context.attempt,
        error_kind=None,
    )


def _failed(
    context: ProbeContext, route: AzRoute, error: BaseException
) -> ProbeResult:
    return ProbeResult(
        snapshot_sha256=context.snapshot_sha256,
        source_revision=context.source_revision,
        route_id=route.route_id,
        canonical_provider=route.canonical_provider,
        canonical_model_id=route.canonical_model_id,
        protocol=context.protocol,
        scenario=context.scenario,
        status="failed",
        attempts=context.attempt,
        error_kind=_error_kind(error),
        private_detail=error,
    )


def _error_kind(error: BaseException) -> ErrorKind:
    if isinstance(error, ModelProviderError):
        if error.reason_code is ModelFailureReason.PERMISSION_DENIED:
            return ErrorKind.PERMISSION
        return {
            ModelErrorKind.AUTHENTICATION: ErrorKind.AUTHENTICATION,
            ModelErrorKind.RATE_LIMIT: ErrorKind.RATE_LIMIT,
            ModelErrorKind.TIMEOUT: ErrorKind.TIMEOUT,
            ModelErrorKind.UNAVAILABLE: ErrorKind.UPSTREAM_UNAVAILABLE,
            ModelErrorKind.INVALID_REQUEST: ErrorKind.INVALID_REQUEST,
            ModelErrorKind.INVALID_RESPONSE: ErrorKind.INVALID_RESPONSE,
            ModelErrorKind.INCOMPLETE_RESPONSE: ErrorKind.INVALID_RESPONSE,
            ModelErrorKind.OUTCOME_UNKNOWN: ErrorKind.UNKNOWN,
            ModelErrorKind.UNKNOWN: ErrorKind.UNKNOWN,
        }[error.kind]
    if isinstance(error, (TimeoutError, asyncio.TimeoutError)):
        return ErrorKind.TIMEOUT
    if isinstance(error, (TypeError, ValueError)):
        return ErrorKind.INVALID_RESPONSE
    return ErrorKind.UNKNOWN


async def _non_stream(
    context: ProbeContext,
    route: AzRoute,
    request: ModelProviderRequest,
    *,
    mutate: Callable[[dict[str, object]], None] | None = None,
):
    adapter = OpenAICompatibleAdapter()
    payload = _wire_payload(adapter, request, route).to_dict()
    if mutate is not None:
        mutate(payload)
    raw = await _client(context).invoke(request.model, freeze_json_object(payload))
    return adapter.parse_response(request, raw)


async def openai_text_probe(context: ProbeContext, route: AzRoute) -> ProbeResult:
    try:
        response = await _non_stream(context, route, _request(route))
        if not response.message.content.strip():
            raise ValueError("text response is empty")
        return _passed(context, route)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - probe records classified failure
        return _failed(context, route, exc)


async def openai_stream_probe(context: ProbeContext, route: AzRoute) -> ProbeResult:
    request = _request(route)
    adapter = OpenAICompatibleAdapter()
    try:
        decoder = adapter.create_stream_decoder(request)
        saw_text = False
        saw_finish = False
        payload = _wire_payload(adapter, request, route)
        stream: AsyncIterator[FrozenJsonObject] = _client(context).stream(
            request.model, payload
        )
        async for item in stream:
            for part in decoder.feed(item):
                saw_text = saw_text or part.kind == "text"
                saw_finish = saw_finish or part.kind == "finish"
        decoder.finish()
        if not saw_text or not saw_finish:
            raise ValueError("stream lacks text or completion evidence")
        return _passed(context, route)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - probe records classified failure
        return _failed(context, route, exc)


async def openai_tool_probe(context: ProbeContext, route: AzRoute) -> ProbeResult:
    adapter = OpenAICompatibleAdapter()
    first_request = _request(route, tools=(_TOOL,))
    try:
        first_raw = await _client(context).invoke(
            first_request.model, _wire_payload(adapter, first_request, route)
        )
        first = adapter.parse_response(first_request, first_raw)
        if len(first.message.tool_calls) != 1:
            raise ValueError("model did not return exactly one tool call")
        call = first.message.tool_calls[0]
        if call.name != _TOOL.name:
            raise ValueError("model returned the wrong tool")
        second_request = _request(
            route,
            message=ToolMessage(
                results=(
                    ToolResult(
                        call_id=call.call_id,
                        name=call.name,
                        status="succeeded",
                        output="probe",
                    ),
                )
            ),
            context=Context(messages=(first.message,)),
            tools=(_TOOL,),
        )
        second_raw = await _client(context).invoke(
            second_request.model, _wire_payload(adapter, second_request, route)
        )
        second = adapter.parse_response(second_request, second_raw)
        if not second.message.content.strip():
            raise ValueError("tool continuation response is empty")
        return _passed(context, route)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - probe records classified failure
        return _failed(context, route, exc)


async def openai_tool_choice_probe(
    context: ProbeContext, route: AzRoute
) -> ProbeResult:
    request = _request(
        route,
        tools=(_TOOL,),
        generation=GenerationConfig(max_output_tokens=32, tool_choice="lookup"),
    )
    try:
        response = await _non_stream(context, route, request)
        if len(response.message.tool_calls) != 1:
            raise ValueError("forced tool choice did not produce one tool call")
        return _passed(context, route)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - probe records classified failure
        return _failed(context, route, exc)


async def openai_json_object_probe(
    context: ProbeContext, route: AzRoute
) -> ProbeResult:
    def add_json_object(body: dict[str, object]) -> None:
        body["response_format"] = {"type": "json_object"}

    try:
        response = await _non_stream(
            context, route, _request(route), mutate=add_json_object
        )
        value = json.loads(response.message.content)
        if not isinstance(value, dict):
            raise TypeError("JSON object response is not an object")
        return _passed(context, route)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - probe records classified failure
        return _failed(context, route, exc)


async def openai_json_schema_probe(
    context: ProbeContext, route: AzRoute
) -> ProbeResult:
    request = _request(
        route,
        generation=GenerationConfig(
            max_output_tokens=32,
            response_schema=_SCHEMA,
            response_schema_name="probe_response",
        ),
    )
    try:
        response = await _non_stream(context, route, request)
        if not response.message.content:
            raise ValueError("JSON schema response is empty")
        return _passed(context, route)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - probe records classified failure
        return _failed(context, route, exc)


async def openai_reasoning_probe(
    context: ProbeContext, route: AzRoute
) -> ProbeResult:
    request = _request(route, provider_options={"reasoning_effort": "low"})
    try:
        response = await _non_stream(context, route, request)
        continuation = response.message.continuation
        if continuation is None or not continuation.data.get("reasoning_content"):
            raise ValueError("reasoning response has no protocol evidence")
        return _passed(context, route)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - probe records classified failure
        return _failed(context, route, exc)


async def openai_image_input_probe(
    context: ProbeContext, route: AzRoute
) -> ProbeResult:
    def add_image(body: dict[str, object]) -> None:
        messages = cast(list[dict[str, object]], body["messages"])
        messages[-1]["content"] = [
            {"type": "text", "text": "Name the color of the square."},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_PNG}"},
            },
        ]

    try:
        response = await _non_stream(
            context, route, _request(route), mutate=add_image
        )
        if not response.message.content.strip():
            raise ValueError("vision response is empty")
        return _passed(context, route)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - probe records classified failure
        return _failed(context, route, exc)


def _responses_spec(
    route: AzRoute, *, provider_options: dict[str, object] | None = None
) -> ModelSpec:
    return ModelSpec(
        provider=cast(str, route.canonical_provider),
        model_id=cast(str, route.canonical_model_id),
        protocol="openai_responses",
        provider_options=provider_options or {},
        capabilities=CapabilityPresetCatalog.builtin()
        .presets["text_tools_structured_reasoning"]
        .materialize(context_tokens=1_000_000, max_output_tokens=4096),
    )


def _responses_request(
    route: AzRoute,
    *,
    message=None,
    context: Context | None = None,
    generation: GenerationConfig | None = None,
    tools: tuple[ToolDefinition, ...] = (),
    provider_options: dict[str, object] | None = None,
) -> ModelProviderRequest:
    return ModelProviderRequest(
        model_key=route.route_id,
        model=_responses_spec(route, provider_options=provider_options),
        message=message or UserMessage(content="Reply with a short answer."),
        context=context or Context(),
        generation=generation or GenerationConfig(max_output_tokens=32),
        tools=tools,
    )


def _responses_payload(
    adapter: OpenAIResponsesAdapter,
    request: ModelProviderRequest,
    route: AzRoute,
) -> FrozenJsonObject:
    body = adapter.build_request(request).to_dict()
    body["model"] = route.route_id
    return freeze_json_object(body)


async def _responses_non_stream(
    context: ProbeContext,
    route: AzRoute,
    request: ModelProviderRequest,
    *,
    mutate: Callable[[dict[str, object]], None] | None = None,
):
    adapter = OpenAIResponsesAdapter()
    body = _responses_payload(adapter, request, route).to_dict()
    if mutate is not None:
        mutate(body)
    raw = await _client(context).invoke(request.model, freeze_json_object(body))
    return adapter.parse_response(request, raw)


async def openai_responses_text_probe(
    context: ProbeContext, route: AzRoute
) -> ProbeResult:
    try:
        response = await _responses_non_stream(
            context, route, _responses_request(route)
        )
        if not response.message.content.strip():
            raise ValueError("text response is empty")
        return _passed(context, route)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - probe records classified failure
        return _failed(context, route, exc)


async def openai_responses_stream_probe(
    context: ProbeContext, route: AzRoute
) -> ProbeResult:
    request = _responses_request(route)
    adapter = OpenAIResponsesAdapter()
    try:
        decoder = adapter.create_stream_decoder(request)
        saw_text = False
        saw_finish = False
        stream: AsyncIterator[FrozenJsonObject] = _client(context).stream(
            request.model, _responses_payload(adapter, request, route)
        )
        async for item in stream:
            for part in decoder.feed(item):
                saw_text = saw_text or part.kind == "text"
                saw_finish = saw_finish or part.kind == "finish"
        decoder.finish()
        if not saw_text or not saw_finish:
            raise ValueError("stream lacks text or completion evidence")
        return _passed(context, route)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - probe records classified failure
        return _failed(context, route, exc)


async def openai_responses_tool_probe(
    context: ProbeContext, route: AzRoute
) -> ProbeResult:
    adapter = OpenAIResponsesAdapter()
    first_request = _responses_request(route, tools=(_TOOL,))
    try:
        first_raw = await _client(context).invoke(
            first_request.model,
            _responses_payload(adapter, first_request, route),
        )
        first = adapter.parse_response(first_request, first_raw)
        if len(first.message.tool_calls) != 1:
            raise ValueError("model did not return exactly one function call")
        call = first.message.tool_calls[0]
        second_request = _responses_request(
            route,
            message=ToolMessage(
                results=(
                    ToolResult(
                        call_id=call.call_id,
                        name=call.name,
                        status="succeeded",
                        output="probe",
                    ),
                )
            ),
            context=Context(messages=(first.message,)),
            tools=(_TOOL,),
        )
        second_raw = await _client(context).invoke(
            second_request.model,
            _responses_payload(adapter, second_request, route),
        )
        second = adapter.parse_response(second_request, second_raw)
        if not second.message.content.strip():
            raise ValueError("function continuation response is empty")
        return _passed(context, route)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - probe records classified failure
        return _failed(context, route, exc)


async def openai_responses_tool_choice_probe(
    context: ProbeContext, route: AzRoute
) -> ProbeResult:
    request = _responses_request(
        route,
        tools=(_TOOL,),
        generation=GenerationConfig(max_output_tokens=32, tool_choice="lookup"),
    )
    try:
        response = await _responses_non_stream(context, route, request)
        if len(response.message.tool_calls) != 1:
            raise ValueError("forced function choice did not produce one call")
        return _passed(context, route)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - probe records classified failure
        return _failed(context, route, exc)


async def openai_responses_json_object_probe(
    context: ProbeContext, route: AzRoute
) -> ProbeResult:
    def json_mode(body: dict[str, object]) -> None:
        body["text"] = {"format": {"type": "json_object"}}

    try:
        response = await _responses_non_stream(
            context, route, _responses_request(route), mutate=json_mode
        )
        if not isinstance(json.loads(response.message.content), dict):
            raise TypeError("JSON response is not an object")
        return _passed(context, route)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - probe records classified failure
        return _failed(context, route, exc)


async def openai_responses_json_schema_probe(
    context: ProbeContext, route: AzRoute
) -> ProbeResult:
    request = _responses_request(
        route,
        generation=GenerationConfig(
            max_output_tokens=32,
            response_schema=_SCHEMA,
            response_schema_name="probe_response",
        ),
    )
    try:
        response = await _responses_non_stream(context, route, request)
        value = json.loads(response.message.content)
        if not isinstance(value, dict) or not isinstance(value.get("answer"), str):
            raise TypeError("JSON schema response does not match the probe schema")
        return _passed(context, route)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - probe records classified failure
        return _failed(context, route, exc)


async def openai_responses_reasoning_probe(
    context: ProbeContext, route: AzRoute
) -> ProbeResult:
    request = _responses_request(
        route,
        provider_options={"reasoning": {"effort": "low", "summary": "auto"}},
    )
    try:
        response = await _responses_non_stream(context, route, request)
        if response.message.continuation is None:
            raise ValueError("reasoning response has no Responses reasoning item")
        return _passed(context, route)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - probe records classified failure
        return _failed(context, route, exc)


async def openai_responses_image_input_probe(
    context: ProbeContext, route: AzRoute
) -> ProbeResult:
    def add_image(body: dict[str, object]) -> None:
        items = cast(list[dict[str, object]], body["input"])
        items[-1]["content"] = [
            {"type": "input_text", "text": "Name the color of the square."},
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{_PNG}",
            },
        ]

    try:
        response = await _responses_non_stream(
            context, route, _responses_request(route), mutate=add_image
        )
        if not response.message.content.strip():
            raise ValueError("vision response is empty")
        return _passed(context, route)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - probe records classified failure
        return _failed(context, route, exc)


__all__ = [
    "openai_image_input_probe",
    "openai_json_object_probe",
    "openai_json_schema_probe",
    "openai_reasoning_probe",
    "openai_responses_image_input_probe",
    "openai_responses_json_object_probe",
    "openai_responses_json_schema_probe",
    "openai_responses_reasoning_probe",
    "openai_responses_stream_probe",
    "openai_responses_text_probe",
    "openai_responses_tool_choice_probe",
    "openai_responses_tool_probe",
    "openai_stream_probe",
    "openai_text_probe",
    "openai_tool_choice_probe",
    "openai_tool_probe",
]
