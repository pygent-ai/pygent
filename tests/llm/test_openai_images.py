"""Tests for the OpenAI Images wire codec and its invoker delivery."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import cast

import httpx
import pytest

from pygent.core import Context, UserMessage, freeze_json_object
from pygent.llm import (
    DefaultModelInvoker,
    ModelEntry,
    ModelGroup,
    ModelSpec,
    OpenAIImagesAdapter,
    OpenAIImagesClient,
    openai_images_adapters,
)
from pygent.llm._adapter_contracts import ModelProviderRequest
from pygent.llm.catalogs import ModelCapabilityCatalog
from pygent.llm.types import (
    GenerationConfig,
    ModelErrorKind,
    ModelProviderError,
    RetryPolicy,
)
from pygent.tool import MediaBlock, MediaSource, ToolDefinition

_PNG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\nfixture").decode("ascii")


def _spec(provider: str = "volcengine", model_id: str = "doubao-seedream-5-0-260128"):
    capabilities = ModelCapabilityCatalog.builtin().models[
        (provider, model_id, "openai_images")
    ]
    return ModelSpec(
        provider=provider,
        model_id=model_id,
        protocol="openai_images",
        capabilities=capabilities,
    )


def _request(
    spec=None,
    *,
    content: str = "A single blue square.",
    tools=(),
    generation: GenerationConfig | None = None,
) -> ModelProviderRequest:
    return ModelProviderRequest(
        model_key="primary",
        model=spec or _spec(),
        message=UserMessage(content=content),
        context=Context(),
        generation=generation or GenerationConfig(max_output_tokens=None),
        tools=tools,
    )


def test_openai_images_builds_b64_json_generation_request() -> None:
    payload = OpenAIImagesAdapter().build_request(_request()).to_dict()

    assert payload["model"] == "doubao-seedream-5-0-260128"
    assert payload["prompt"] == "A single blue square."
    assert payload["response_format"] == "b64_json"
    assert "messages" not in payload


def test_openai_images_merges_provider_options_into_the_body() -> None:
    spec = ModelSpec(
        provider="volcengine",
        model_id="doubao-seedream-5-0-260128",
        protocol="openai_images",
        provider_options={"size": "1024x1024"},
        capabilities=_spec().capabilities,
    )
    payload = OpenAIImagesAdapter().build_request(_request(spec)).to_dict()

    assert payload["size"] == "1024x1024"
    assert payload["response_format"] == "b64_json"


def test_openai_images_rejects_reserved_provider_options() -> None:
    spec = ModelSpec(
        provider="volcengine",
        model_id="doubao-seedream-5-0-260128",
        protocol="openai_images",
        provider_options={"prompt": "override"},
        capabilities=_spec().capabilities,
    )
    with pytest.raises(ValueError, match="reserved fields"):
        OpenAIImagesAdapter().build_request(_request(spec))


def test_openai_images_rejects_tools_and_media_input() -> None:
    adapter = OpenAIImagesAdapter()
    tool = ToolDefinition(
        name="lookup",
        description="Return the supplied probe value.",
        parameters={"type": "object", "properties": {}},
    )
    with pytest.raises(Exception, match="does not accept tools"):
        adapter.build_request(_request(tools=(tool,)))
    media_message = UserMessage(
        content="Make it blue.",
        media=(
            MediaBlock(
                media_type="image",
                mime_type="image/png",
                source=MediaSource.inline(b"\x89PNG\r\n\x1a\nfixture"),
            ),
        ),
    )
    request = ModelProviderRequest(
        model_key="primary",
        model=_spec(),
        message=media_message,
        context=Context(),
        generation=GenerationConfig(),
        tools=(),
    )
    with pytest.raises(Exception, match="image input"):
        adapter.build_request(request)


def test_openai_images_parses_inline_image_data() -> None:
    response = OpenAIImagesAdapter().parse_response(
        _request(),
        freeze_json_object(
            {
                "created": 1_700_000_000,
                "data": [{"b64_json": _PNG_B64}],
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 256,
                    "total_tokens": 268,
                },
            }
        ),
    )

    assert response.finish_reason == "stop"
    assert response.message.content == ""
    assert [block.media_type for block in response.message.media] == ["image"]
    assert response.message.media[0].mime_type == "image/png"
    assert response.message.media[0].source.base64_data == _PNG_B64
    assert response.message.media[0].source.sha256
    usage = cast(Mapping[str, int], response.usage)
    assert usage["input_tokens"] == 12
    assert usage["output_tokens"] == 256


def test_openai_images_sniffs_jpeg_magic_over_defaults() -> None:
    response = OpenAIImagesAdapter().parse_response(
        _request(),
        freeze_json_object(
            {"data": [{"b64_json": base64.b64encode(b"\xff\xd8\xff\xe0JFIF").decode()}]}
        ),
    )
    assert response.message.media[0].mime_type == "image/jpeg"


def test_openai_images_maps_declared_output_format() -> None:
    response = OpenAIImagesAdapter().parse_response(
        _request(),
        freeze_json_object(
            {"data": [{"b64_json": _PNG_B64, "output_format": "jpeg"}]}
        ),
    )
    assert response.message.media[0].mime_type == "image/png"


def test_openai_images_fails_closed_on_url_only_response() -> None:
    from pygent.llm.types import ModelErrorKind, ModelFailureReason, ModelProviderError

    with pytest.raises(ModelProviderError) as raised:
        OpenAIImagesAdapter().parse_response(
            _request(),
            freeze_json_object({"data": [{"url": "https://example.test/image.png"}]}),
        )
    assert raised.value.kind is ModelErrorKind.INVALID_RESPONSE
    assert raised.value.reason_code is ModelFailureReason.MEDIA_CONTENT_INVALID


def test_openai_images_fails_closed_on_empty_data() -> None:
    with pytest.raises(Exception, match="no data array"):
        OpenAIImagesAdapter().parse_response(
            _request(), freeze_json_object({"data": []})
        )


def test_openai_images_streaming_is_rejected() -> None:
    adapter = OpenAIImagesAdapter()
    decoder = adapter.create_stream_decoder(_request())
    with pytest.raises(Exception, match="does not support streaming"):
        decoder.feed(freeze_json_object({}))
    assert decoder.finish() == ()

    async def _stream() -> None:
        client = OpenAIImagesClient(
            base_url="https://ark.cn-beijing.volces.com/api/v3", api_key="k"
        )
        async for _ in client.stream(_spec(), freeze_json_object({})):  # pragma: no cover
            raise AssertionError("stream must not yield")

    import asyncio

    with pytest.raises(Exception, match="does not support streaming"):
        asyncio.run(_stream())


def test_openai_images_adapter_registry() -> None:
    assert set(openai_images_adapters()) == {"openai_images"}


@pytest.mark.parametrize(
    ("status", "expected_kind"),
    [(401, ModelErrorKind.AUTHENTICATION), (429, ModelErrorKind.RATE_LIMIT)],
)
@pytest.mark.asyncio
async def test_images_status_errors_carry_redacted_provider_message(
    status: int, expected_kind: ModelErrorKind
) -> None:
    """The client must hand its secrets to the shared status-error builder,
    or provider bodies echoing the bearer token would leak into run events."""

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            json={
                "error": {
                    "message": (
                        "request rejected for bearer images-key-canary-2468 at"
                        " https://synthetic-endpoint.invalid/private"
                    )
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = OpenAIImagesClient(
            base_url="https://synthetic-endpoint.invalid/private",
            api_key="images-key-canary-2468",
            client=http,
        )
        with pytest.raises(ModelProviderError) as raised:
            await client.invoke(_spec(), freeze_json_object({}))

    assert raised.value.kind is expected_kind
    message = raised.value.provider_message
    assert message is not None
    assert "request rejected for bearer" in message
    assert "images-key-canary-2468" not in message
    assert "https://synthetic-endpoint.invalid/private" not in message
    assert "[REDACTED]" in message


async def test_openai_images_delivers_media_through_the_invoker() -> None:
    """Regression: the non-stream invoker path must forward parse_response media."""

    class FakeClient:
        async def invoke(self, model, payload):
            return freeze_json_object({"data": [{"b64_json": _PNG_B64}]})

        async def stream(self, model, payload):  # pragma: no cover - must not be called
            raise AssertionError("stream must not be called")
            yield freeze_json_object()

        async def aclose(self) -> None:
            return None

    invoker = DefaultModelInvoker(
        adapters={"openai_images": OpenAIImagesAdapter()},
        clients={"primary": FakeClient()},
    )
    execution = invoker.execute(
        model_group=ModelGroup("images", (ModelEntry("primary", _spec()),)),
        retry_policy=RetryPolicy(),
        generation=GenerationConfig(output_modalities=("image",)),
        message=UserMessage(content="A single blue square."),
        context=Context(),
    )
    result = await execution.result()

    assert [block.media_type for block in result.message.media] == ["image"]
    media_data = result.message.media[0].source.base64_data
    assert media_data
    decoded = base64.b64decode(media_data)
    assert decoded.startswith(b"\x89PNG\r\n\x1a\n")
