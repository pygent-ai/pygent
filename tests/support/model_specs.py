"""Concise model values used by execution-focused tests."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from pygent.llm import (
    CapabilityPresetCatalog,
    DefaultModelInvoker,
    ModelEntry,
    ModelGroup,
    ModelSpec,
    ModelStreamingCapabilities,
)

_UNSET = object()


def model_entry(
    name: str,
    provider: str,
    model_id: str,
    *,
    provider_options: object = _UNSET,
    streaming: bool = True,
) -> ModelEntry:
    capabilities = CapabilityPresetCatalog.builtin().presets[
        "text_tools_structured_reasoning"
    ].materialize(context_tokens=1_000_000, max_output_tokens=384_000)
    capabilities = replace(
        capabilities,
        streaming=ModelStreamingCapabilities(output=("text",) if streaming else ()),
    )
    return ModelEntry(
        name,
        ModelSpec(
            provider=provider,
            model_id=model_id,
            protocol="openai_chat_completions",
            provider_options={} if provider_options is _UNSET else provider_options,  # type: ignore[arg-type]
            capabilities=capabilities,
        ),
    )


def model_group(
    name: str,
    models: tuple[ModelEntry, ...],
    order: tuple[str, ...] | None = None,
) -> ModelGroup:
    if order is not None:
        by_name = {model.name: model for model in models}
        models = tuple(by_name[key] for key in order)
    return ModelGroup(name, models)


@dataclass(frozen=True)
class TransportMode:
    streaming: bool


def transport_mode(*, streaming: bool) -> TransportMode:
    return TransportMode(streaming)


class _ConfiguredInvoker:
    def __init__(
        self,
        adapters: dict[str, object],
        clients: dict[str, object],
        capabilities: dict[str, TransportMode],
    ) -> None:
        adapter = adapters.get("openai_chat_completions") or next(iter(adapters.values()))
        self._invoker = DefaultModelInvoker(
            adapters={"openai_chat_completions": adapter},  # type: ignore[dict-item]
            clients=clients,  # type: ignore[arg-type]
        )
        self._clients = clients
        self._capabilities = capabilities

    def execute(self, **kwargs: Any):
        group = kwargs["model_group"]
        models = []
        for entry in group.models:
            client = self._clients.get(entry.name) or self._clients.get(
                entry.spec.provider
            )
            if client is not None:
                self._invoker._clients[entry.name] = client
            mode = self._capabilities.get(
                entry.name, self._capabilities.get(entry.spec.provider)
            )
            if mode is not None:
                capabilities = replace(
                    entry.spec.capabilities,
                    streaming=ModelStreamingCapabilities(
                        output=("text",) if mode.streaming else ()
                    ),
                )
                entry = replace(entry, spec=replace(entry.spec, capabilities=capabilities))
            models.append(entry)
        kwargs["model_group"] = ModelGroup(group.name, tuple(models))
        return self._invoker.execute(**kwargs)

    def validate_model(self, entry: ModelEntry) -> None:
        self._invoker.validate_model(entry)

    async def aclose(self) -> None:
        await self._invoker.aclose()


def configured_invoker(
    *,
    adapters: dict[str, object],
    clients: dict[str, object],
    capabilities: dict[str, TransportMode] | None = None,
) -> _ConfiguredInvoker:
    return _ConfiguredInvoker(adapters, clients, capabilities or {})


__all__ = [
    "configured_invoker",
    "model_entry",
    "model_group",
    "transport_mode",
]
