"""Offline and opt-in live model resources for the tutorial."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from pygent import (
    AIMessage,
    FrozenJsonObject,
    ModelGroupConfig,
    ToolCall,
    ToolMessage,
    freeze_json_object,
)
from pygent.llm import (
    DefaultModelInvoker,
    ModelEventKind,
    ModelExecution,
    OpenAICompatibleAdapter,
    OpenAICompatibleClient,
)
from pygent.llm.spi import (
    ModelInvoker,
    ModelProviderResponse,
)


class OfflineModelInvoker:
    """A deterministic provider boundary that still exercises a real tool loop."""

    def __init__(self, label: str = "offline") -> None:
        self.label = label
        self.calls = 0
        self.closed = False

    def execute(self, **kwargs: object) -> ModelExecution:
        if self.closed:
            raise RuntimeError("offline model invoker is closed")

        async def operation(emit: object) -> ModelProviderResponse:
            self.calls += 1
            model_group = cast(ModelGroupConfig, kwargs["model_group"])
            route_id = model_group.routes[0].route_id
            attempt = 1

            async def publish(kind: str, data: dict[str, object]) -> None:
                await emit(kind, freeze_json_object(data))  # type: ignore[operator]

            await publish(
                ModelEventKind.STARTED.value,
                {"model_group": model_group.name},
            )
            await publish(
                ModelEventKind.ATTEMPT_STARTED.value,
                {"route_id": route_id, "attempt": attempt},
            )

            message = kwargs["message"]
            if isinstance(message, ToolMessage):
                result = message.results[0]
                output = result.output
                total = (
                    output.get("sum") if isinstance(output, FrozenJsonObject) else None
                )
                content = f"{self.label}: 2 + 3 = {total}"
                response = AIMessage(content=content)
                await publish(
                    ModelEventKind.TEXT_DELTA.value,
                    {"route_id": route_id, "attempt": attempt, "text": content},
                )
            else:
                response = AIMessage(
                    tool_calls=(
                        ToolCall(
                            call_id=f"tutorial-add-{self.calls}",
                            name="add",
                            arguments={"a": 2, "b": 3},
                        ),
                    )
                )

            usage = {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
            }
            await publish(
                ModelEventKind.USAGE.value,
                {
                    "route_id": route_id,
                    "attempt": attempt,
                    "mode": "cumulative",
                    "final": True,
                    "available": True,
                    **usage,
                    "cached_input_tokens": None,
                    "reasoning_tokens": None,
                },
            )
            await publish(
                ModelEventKind.ATTEMPT_SUCCEEDED.value,
                {"route_id": route_id, "attempt": attempt},
            )
            await publish(
                ModelEventKind.COMPLETED.value,
                {
                    "route_id": route_id,
                    "attempt": attempt,
                    "finish_reason": "tool_calls" if response.tool_calls else "stop",
                    "provider_request_id": f"tutorial-{self.label}-{self.calls}",
                },
            )
            return ModelProviderResponse(
                response,
                usage=usage,
                provider_request_id=f"tutorial-{self.label}-{self.calls}",
            )

        return ModelExecution(operation)

    async def aclose(self) -> None:
        self.closed = True


@dataclass(frozen=True, slots=True, repr=False)
class LiveModelConfig:
    """OpenAI-compatible tutorial settings without a printable secret."""

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
    """Create one caller-owned live resource closed by the tutorial runner."""

    client = OpenAICompatibleClient(
        base_url=config.api_base,
        api_key=config.api_key,
        verify_ssl=config.verify_ssl,
    )
    return DefaultModelInvoker(
        adapters={"openai": OpenAICompatibleAdapter()},
        clients={"primary": client},
    )


__all__ = ["LiveModelConfig", "OfflineModelInvoker", "build_live_invoker"]
