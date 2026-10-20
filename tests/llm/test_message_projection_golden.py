"""Golden pin for the message projection unification refactor.

WP0 recorded the pre-refactor outputs of the four projection consumers over
``tests/support/message_projection_corpus.py`` into
``tests/support/message_projection_golden.json``. Every migration must keep
this test green byte-for-byte: any difference means the refactor changed
observable behavior and is rejected.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pygent.agent.pygent_agent import _message_projection as agent_projection
from pygent.llm._request_snapshot import _message_projection as snapshot_projection
from pygent.llm.layer import _message_effect_value as layer_effect
from pygent.runtime.codec import message_from_dict, message_to_dict
from tests.support.message_projection_corpus import CORPUS

GOLDEN = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "support"
        / "message_projection_golden.json"
    ).read_text(encoding="utf-8")
)

CONSUMERS = {
    "codec_wire": message_to_dict,
    "layer_effect": layer_effect,
    "request_snapshot": snapshot_projection,
    "agent_estimate": agent_projection,
}


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


@pytest.mark.parametrize("sample_name", sorted(CORPUS))
@pytest.mark.parametrize("consumer_name", sorted(CONSUMERS))
def test_projection_matches_golden_baseline(
    consumer_name: str, sample_name: str
) -> None:
    message = CORPUS[sample_name]
    assert _canonical(CONSUMERS[consumer_name](message)) == GOLDEN[consumer_name][
        sample_name
    ]


@pytest.mark.parametrize("sample_name", sorted(CORPUS))
def test_wire_decode_roundtrip_matches_golden(sample_name: str) -> None:
    value = GOLDEN["codec_wire"][sample_name]
    restored = message_from_dict(json.loads(value))
    assert _canonical(message_to_dict(restored)) == value


@pytest.mark.parametrize(
    "sample_name",
    # The effect decoder is the model-response boundary by design: it only
    # reconstructs AIMessage samples.
    ["ai_plain", "ai_tool_calls_full", "ai_media", "ai_continuation"],
)
def test_layer_effect_decode_roundtrip_matches_golden(sample_name: str) -> None:
    from pygent.llm.layer import _message_from_effect

    value = json.loads(GOLDEN["layer_effect"][sample_name])
    wrapper = {
        "outcome": "succeeded",
        "message": value,
        "usage": {},
        "provider_request_id": None,
    }
    restored = _message_from_effect(wrapper)
    assert _canonical(layer_effect(restored)) == GOLDEN["layer_effect"][sample_name]
