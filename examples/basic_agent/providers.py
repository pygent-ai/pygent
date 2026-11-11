"""OpenAI-compatible model configuration for the basic agent example."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from pygent.llm import (
    DefaultModelInvoker,
    OpenAICompatibleAdapter,
    OpenAICompatibleClient,
)
from pygent.llm.spi import ModelInvoker

from .agent import _MODEL_ENTRY_KEY


@dataclass(frozen=True, slots=True, repr=False)
class LiveModelConfig:
    """OpenAI-compatible model settings without a printable secret."""

    api_base: str
    api_key: str
    model_name: str
    verify_ssl: bool = True

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> LiveModelConfig:
        values = os.environ if environment is None else environment

        def required(name: str) -> str:
            value = values.get(name, "").strip()
            if not value:
                raise RuntimeError(f"missing required environment variable {name}")
            return value

        def optional_bool(name: str, *, default: bool) -> bool:
            value = values.get(name)
            if value is None or not value.strip():
                return default
            normalized = value.strip().lower()
            if normalized == "true":
                return True
            if normalized == "false":
                return False
            raise RuntimeError(f"{name} must be 'true' or 'false'")

        return cls(
            api_base=required("PYGENT_API_BASE"),
            api_key=required("PYGENT_API_KEY"),
            model_name=required("PYGENT_MODEL_NAME"),
            verify_ssl=optional_bool("PYGENT_VERIFY_SSL", default=True),
        )

    def __repr__(self) -> str:
        return (
            "LiveModelConfig(api_configured=True, model_configured=True, "
            f"verify_ssl={self.verify_ssl!r})"
        )


def build_live_invoker(config: LiveModelConfig) -> ModelInvoker:
    """Create one caller-owned live resource; the runner closes it."""

    client = OpenAICompatibleClient(
        base_url=config.api_base,
        api_key=config.api_key,
        verify_ssl=config.verify_ssl,
    )
    return DefaultModelInvoker(
        adapters={"openai_chat_completions": OpenAICompatibleAdapter()},
        clients={_MODEL_ENTRY_KEY: client},
    )


__all__ = ["LiveModelConfig", "build_live_invoker"]
