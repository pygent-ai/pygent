from __future__ import annotations

import pytest

from examples.tutorial import (
    LiveModelConfig,
    OfflineModelInvoker,
    run_direct_demo,
    run_managed_demo,
)
from examples.tutorial.runner import close_invoker
from pygent import AIMessage, FrozenJsonObject, ToolMessage, UserMessage


@pytest.mark.asyncio
async def test_offline_tutorial_runs_a_real_tool_loop_and_commits_context() -> None:
    invoker = OfflineModelInvoker()
    try:
        result = await run_direct_demo(invoker)
    finally:
        await close_invoker(invoker)

    assert result.answer == "offline: 2 + 3 = 5"
    assert [type(message) for message in result.context.messages] == [
        UserMessage,
        AIMessage,
        ToolMessage,
        AIMessage,
    ]
    tool_message = result.context.messages[2]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.results[0].status == "succeeded"
    output = tool_message.results[0].output
    assert isinstance(output, FrozenJsonObject)
    assert output.to_dict() == {"sum": 5}
    assert invoker.calls == 2
    assert invoker.closed


@pytest.mark.asyncio
async def test_stream_tutorial_observes_events_then_returns_the_same_result() -> None:
    invoker = OfflineModelInvoker()
    try:
        result = await run_direct_demo(invoker, stream=True)
    finally:
        await close_invoker(invoker)

    assert result.answer == "offline: 2 + 3 = 5"
    assert result.event_kinds[0] == "execution.started"
    assert "model.text.delta" in result.event_kinds
    assert result.event_kinds[-1] == "execution.completed"
    assert invoker.closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("selected_profile", "expected"),
    [(None, "quick: 2 + 3 = 5"), ("quality", "quality: 2 + 3 = 5")],
)
async def test_managed_tutorial_uses_default_or_explicit_profile_and_closes_resources(
    selected_profile: str | None,
    expected: str,
) -> None:
    result, quick, quality = await run_managed_demo(
        selected_profile=selected_profile
    )

    assert result.answer == expected
    assert quick.closed
    assert quality.closed


def test_live_config_is_explicit_and_does_not_repr_the_secret() -> None:
    config = LiveModelConfig.from_environment(
        {
            "PYGENT_API_BASE": "https://models.example/v1",
            "PYGENT_API_KEY": "very-secret",
            "PYGENT_MODEL_NAME": "example-model",
        }
    )

    assert config.model_name == "example-model"
    assert "very-secret" not in repr(config)

    with pytest.raises(RuntimeError, match="PYGENT_API_KEY"):
        LiveModelConfig.from_environment(
            {
                "PYGENT_API_BASE": "https://models.example/v1",
                "PYGENT_MODEL_NAME": "example-model",
            }
        )
