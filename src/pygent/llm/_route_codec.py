"""Canonical portable projection for model routes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from pygent.core import FrozenJsonObject, thaw_json

from .types import ModelRoute

_ROUTE_FIELDS = frozenset({"route_id", "provider", "model", "provider_options"})


def model_route_value(route: ModelRoute) -> dict[str, object]:
    """Return the one compatibility-preserving portable route projection."""

    value: dict[str, object] = {
        "route_id": route.route_id,
        "provider": route.provider,
        "model": route.model,
    }
    options = cast(FrozenJsonObject, route.provider_options)
    if options:
        value["provider_options"] = thaw_json(options)
    return value


def model_route_from_value(value: Mapping[str, object]) -> ModelRoute:
    """Decode a portable route, treating the legacy missing field as empty."""

    unknown = set(value) - _ROUTE_FIELDS
    if unknown:
        raise ValueError("unknown stored model route fields: " + ", ".join(sorted(unknown)))
    route_id = value.get("route_id")
    provider = value.get("provider")
    model = value.get("model")
    if not isinstance(route_id, str) or not isinstance(provider, str) or not isinstance(model, str):
        raise TypeError("stored model route identity fields must be strings")
    options = value.get("provider_options", {})
    if not isinstance(options, Mapping):
        raise TypeError("stored provider_options must be an object")
    return ModelRoute(
        route_id,
        provider=provider,
        model=model,
        provider_options=cast(Mapping[str, object], options),
    )


__all__ = ["model_route_from_value", "model_route_value"]
