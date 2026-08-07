"""Strict immutable JSON value contract."""

import math

import pytest

from pygent import (
    ExecutionEvent,
    FrozenJsonObject,
    JsonValueError,
    UserMessage,
    freeze_json,
    thaw_json,
)
from pygent.core import json_values


def test_nested_json_values_are_recursively_frozen_and_thawed():
    source = {"name": "weather", "values": [1, {"ok": True}]}

    frozen = freeze_json(source)
    source["name"] = "changed"

    assert isinstance(frozen, FrozenJsonObject)
    assert frozen["name"] == "weather"
    assert thaw_json(frozen) == {
        "name": "weather",
        "values": [1, {"ok": True}],
    }


@pytest.mark.parametrize(
    "value",
    [object(), b"bytes", math.inf, -math.inf, math.nan, {1: "bad key"}],
)
def test_non_json_public_values_are_rejected(value):
    with pytest.raises(JsonValueError):
        freeze_json(value)


def test_message_and_execution_event_freeze_public_metadata():
    metadata = {"tags": ["a", "b"]}
    message = UserMessage(content="hello", metadata=metadata)
    event = ExecutionEvent(
        schema_version="0.2",
        event_id="event-1",
        execution_id="run-1",
        trace_id="trace-1",
        span_id="span-1",
        parent_span_id=None,
        module_path="root",
        sequence=0,
        timestamp_unix_ns=1,
        kind="demo.progress",
        data={"percent": 50},
    )

    metadata["tags"].append("changed")

    assert thaw_json(message.metadata) == {"tags": ["a", "b"]}
    assert event.data["percent"] == 50


@pytest.mark.parametrize("container_kind", ["list", "object"])
def test_cyclic_json_values_are_rejected(container_kind):
    if container_kind == "list":
        value = []
        value.append(value)
    else:
        value = {}
        value["self"] = value

    with pytest.raises(JsonValueError, match="cyclic"):
        freeze_json(value)


def test_json_depth_and_aggregate_size_are_bounded(monkeypatch):
    monkeypatch.setattr(json_values, "MAX_JSON_DEPTH", 2)
    with pytest.raises(JsonValueError, match="maximum depth"):
        freeze_json([[[None]]])

    monkeypatch.setattr(json_values, "MAX_JSON_NODES", 3)
    with pytest.raises(JsonValueError, match="maximum size"):
        freeze_json([1, 2, 3])


def test_shared_non_cyclic_container_is_copied_for_each_reference():
    shared = [1, 2]

    frozen = freeze_json({"left": shared, "right": shared})
    shared.append(3)

    assert thaw_json(frozen) == {"left": [1, 2], "right": [1, 2]}


def test_pre_frozen_values_still_count_toward_aggregate_depth(monkeypatch):
    monkeypatch.setattr(json_values, "MAX_JSON_DEPTH", 3)
    value = json_values.freeze_json_object({"leaf": True})
    value = json_values.freeze_json_object({"next": value})
    value = json_values.freeze_json_object({"next": value})

    with pytest.raises(JsonValueError, match="maximum depth"):
        json_values.freeze_json_object({"next": value})
