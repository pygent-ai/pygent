from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx

from pygent import MediaSource
from pygent.core import FrozenJsonObject
from pygent.llm import (
    AnthropicMessagesAdapter,
    AnthropicMessagesClient,
    GeminiGenerateContentAdapter,
    GeminiGenerateContentClient,
    MediaTransportCapabilities,
    ModelConfig,
    ModelEntry,
    ModelProviderClient,
    ModelSpec,
    OpenAICompatibleAdapter,
    OpenAICompatibleClient,
    OpenAIResponsesAdapter,
    OpenAIResponsesClient,
    ResolvedModelConnection,
)

from ..graph.model_invoker import LaraModelInvoker
from .attachments import MEDIA_MAX_BYTES

# The kernel owns the lookup *protocol*; the shell supplies the store-backed
# implementation at the process composition root.
CredentialLookup = Callable[[str, Path], tuple[str | None, str]]
CredentialNameLister = Callable[[Path], Sequence[str]]


def environment_credential_lookup(
    env_name: str, user_lara_root: Path
) -> tuple[str | None, str]:
    """Kernel fallback: resolve credentials from the process environment only."""

    del user_lara_root
    value = os.environ.get(env_name)
    if value:
        return value, f"env:{env_name}"
    return None, "missing"


def environment_credential_names(user_lara_root: Path) -> Sequence[str]:
    """Kernel fallback: the process environment is the only credential source."""

    del user_lara_root
    return tuple(os.environ)


CLIENT_FACTORIES: dict[str, Any] = {
    "openai_chat_completions": OpenAICompatibleClient,
    "openai_responses": OpenAIResponsesClient,
    "anthropic_messages": AnthropicMessagesClient,
    "gemini_generate_content": GeminiGenerateContentClient,
}


class CredentialEnvironment(Mapping[str, str]):
    """Resolve native environment references through an injected lookup.

    ``lookup`` and ``name_lister`` default to process-environment-only
    resolution so kernel-only callers stay self-contained. The shell injects
    its store-backed implementation (user file + environment + keyring) when
    composing the process.
    """

    def __init__(
        self,
        user_lara_root: str | Path,
        *,
        transient: Mapping[str, str] | None = None,
        lookup: CredentialLookup = environment_credential_lookup,
        name_lister: CredentialNameLister = environment_credential_names,
    ) -> None:
        self.user_lara_root = Path(user_lara_root).expanduser().resolve()
        self.transient = {
            key: value for key, value in dict(transient or {}).items() if value
        }
        self._lookup = lookup
        self._name_lister = name_lister

    def __getitem__(self, key: str) -> str:
        if key in self.transient:
            return self.transient[key]
        value, _ = self._lookup(key, self.user_lara_root)
        if value is None:
            raise KeyError(key)
        return value

    def __iter__(self) -> Iterator[str]:
        return iter(
            set(self.transient)
            | set(os.environ)
            | set(self._name_lister(self.user_lara_root))
        )

    def __len__(self) -> int:
        return len(set(iter(self)))


class _OwnedProviderClient:
    """Close an injected HTTP client that Pygent intentionally treats as borrowed."""

    def __init__(self, inner: ModelProviderClient, http_client: httpx.AsyncClient) -> None:
        self.inner = inner
        self.http_client = http_client

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    async def invoke(
        self, model: ModelSpec, payload: FrozenJsonObject
    ) -> FrozenJsonObject:
        return await self.inner.invoke(model, payload)

    def stream(
        self, model: ModelSpec, payload: FrozenJsonObject
    ) -> AsyncIterator[FrozenJsonObject]:
        return self.inner.stream(model, payload)

    async def aclose(self) -> None:
        try:
            await self.inner.aclose()
        finally:
            await self.http_client.aclose()


def preferred_profile_name(model_key: str) -> str:
    if not isinstance(model_key, str) or not model_key:
        raise ValueError("model_key must be a non-empty string")
    return f"preferred:{model_key}"


def preferred_models(
    config: ModelConfig, group_name: str, preferred_model_key: str
) -> tuple[ModelEntry, ...]:
    try:
        group = config.model_groups[group_name]
    except KeyError:
        raise ValueError(f"unknown model group {group_name!r}") from None
    by_name = {entry.key: entry for entry in group.models}
    if preferred_model_key not in by_name:
        raise ValueError(
            f"model {preferred_model_key!r} is not in group {group_name!r}"
        )
    return (
        by_name[preferred_model_key],
        *(entry for entry in group.models if entry.key != preferred_model_key),
    )


def workspace_media_resolver(
    workspace_root: str | Path,
) -> Callable[[MediaSource], bytes]:
    """Resolve resource media sources to workspace-local file bytes."""

    root = Path(workspace_root).resolve()

    def resolve(source: MediaSource) -> bytes:
        uri = source.uri
        if not uri:
            raise ValueError("resource media source requires a uri")
        path = Path(uri)
        if not path.is_absolute():
            path = root / path
        resolved = path.resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError(f"media resource escapes the workspace: {uri}")
        return resolved.read_bytes()

    return resolve


def build_model_invoker(
    config: ModelConfig,
    *,
    credential_environ: Mapping[str, str],
    media_resolver: Callable[[MediaSource], bytes] | None = None,
) -> LaraModelInvoker:
    protocols = {entry.spec.protocol for entry in config.models.values()}
    unsupported = protocols - set(CLIENT_FACTORIES)
    if unsupported:
        raise ValueError("unsupported model protocols: " + ", ".join(sorted(unsupported)))
    adapters: dict[str, Any] = {
        "openai_chat_completions": OpenAICompatibleAdapter(
            media_transport=MediaTransportCapabilities(
                enabled=True,
                modalities=("image", "video"),
                source_kinds=(
                    ("inline", "resource") if media_resolver else ("inline",)
                ),
                max_media_bytes=MEDIA_MAX_BYTES,
            ),
            media_resolver=media_resolver,
        ),
        # Every multimodal protocol gets the workspace resolver so user
        # attachments carried as resource sources reach models on that
        # protocol; without it the adapters only accept inline data and
        # workspace media fails delivery for every turn.
        "anthropic_messages": AnthropicMessagesAdapter(
            media_resolver=media_resolver,
        ),
        "gemini_generate_content": GeminiGenerateContentAdapter(
            media_resolver=media_resolver,
        ),
        "openai_responses": OpenAIResponsesAdapter(
            media_resolver=media_resolver,
        ),
    }
    clients = {
        key: _create_client(
            connection=config.connection_for(key),
            credential_environ=credential_environ,
        )
        for key in config.models
    }
    return LaraModelInvoker(
        adapters={protocol: adapters[protocol] for protocol in protocols},
        clients=clients,
        media_resolver=media_resolver,
    )


def _create_client(
    *,
    connection: ResolvedModelConnection,
    credential_environ: Mapping[str, str],
) -> ModelProviderClient:
    try:
        factory = CLIENT_FACTORIES[connection.protocol]
    except KeyError:
        raise ValueError(
            f"unsupported model protocol {connection.protocol!r}"
        ) from None
    api_key = connection.credential.resolve(credential_environ)
    if connection.proxy is None:
        return factory(
            base_url=connection.base_url,
            api_key=api_key,
            verify_ssl=connection.verify_ssl,
        )
    http_client = httpx.AsyncClient(
        proxy=connection.proxy,
        verify=connection.verify_ssl,
    )
    inner = factory(
        base_url=connection.base_url,
        api_key=api_key,
        client=http_client,
    )
    return _OwnedProviderClient(inner, http_client)


__all__ = [
    "CLIENT_FACTORIES",
    "CredentialEnvironment",
    "CredentialLookup",
    "CredentialNameLister",
    "build_model_invoker",
    "environment_credential_lookup",
    "environment_credential_names",
    "preferred_models",
    "preferred_profile_name",
]
