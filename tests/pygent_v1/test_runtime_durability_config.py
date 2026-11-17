from __future__ import annotations

import pytest

from pygent.agent.pygent_v1.config.validation import validate_config_mapping
from pygent.agent.pygent_v1.schema import RuntimeDurabilityConfig


def test_durability_retention_defaults_to_seven_days() -> None:
    assert RuntimeDurabilityConfig().retention_seconds == 7 * 24 * 3600.0


def test_durability_retention_accepts_null_and_positive_numbers() -> None:
    assert RuntimeDurabilityConfig(retention_seconds=None).retention_seconds is None
    assert RuntimeDurabilityConfig(retention_seconds=3600).retention_seconds == 3600.0


@pytest.mark.parametrize("value", [0, -1, float("inf"), True, "7d"])
def test_durability_retention_rejects_non_positive_or_non_numeric(value: object) -> None:
    with pytest.raises(ValueError, match="retention_seconds"):
        RuntimeDurabilityConfig(retention_seconds=value)  # type: ignore[arg-type]


def test_config_mapping_accepts_retention_seconds_key() -> None:
    validate_config_mapping({"runtime": {"durability": {"retention_seconds": 86400}}})
