from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit


class RouteKind(StrEnum):
    OFFICIAL_MODEL = "official_model"
    GATEWAY_ALIAS = "gateway_alias"
    EXTERNAL_SERVICE = "external_service"


class Scenario(StrEnum):
    TEXT = "text"
    TEXT_STREAM = "text_stream"
    TOOLS = "tools"
    TOOL_CHOICE = "tool_choice"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"
    REASONING = "reasoning"
    IMAGE_INPUT = "image_input"
    IMAGE_OUTPUT = "image_output"
    IMAGE_EDIT = "image_edit"
    VIDEO_INPUT = "video_input"
    VIDEO_OUTPUT = "video_output"
    AUDIO_OUTPUT = "audio_output"
    AUDIO_INPUT = "audio_input"
    REALTIME = "realtime"
    EMBEDDING = "embedding"
    SEARCH = "search"


@dataclass(frozen=True, slots=True)
class ProtocolRequirements:
    protocol: str
    required_scenarios: tuple[Scenario, ...]


@dataclass(frozen=True, slots=True)
class AzRoute:
    route_id: str
    kind: RouteKind
    canonical_provider: str | None
    canonical_model_id: str | None
    protocols: tuple[ProtocolRequirements, ...]
    catalog_eligible: bool


@dataclass(frozen=True, slots=True)
class InventorySnapshot:
    captured_at: date
    count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class AzManifest:
    snapshot: InventorySnapshot
    routes: tuple[AzRoute, ...]


@dataclass(frozen=True, slots=True)
class SourceRecord:
    provider: str
    model_id: str
    protocol: str
    url: str
    checked_at: date


@dataclass(frozen=True, slots=True)
class SourceIndex:
    sources: Mapping[tuple[str, str, str], SourceRecord]

    def __init__(self, sources: Mapping[tuple[str, str, str], SourceRecord]) -> None:
        object.__setattr__(self, "sources", MappingProxyType(dict(sources)))


_ROUTE_FIELDS = frozenset(
    {
        "route_id",
        "kind",
        "canonical_provider",
        "canonical_model_id",
        "protocols",
        "catalog_eligible",
    }
)
_PROTOCOL_FIELDS = frozenset({"protocol", "required_scenarios"})
_SOURCE_FIELDS = frozenset(
    {"provider", "model_id", "protocol", "url", "checked_at"}
)
_DATA_DIR = Path(__file__).parent


def _object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise TypeError(f"{label} must be an object with string keys")
    return value


def _exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    actual = frozenset(value)
    unknown = actual - expected
    missing = expected - actual
    if unknown:
        raise ValueError(f"{label} has unknown fields: {sorted(unknown)!r}")
    if missing:
        raise ValueError(f"{label} is missing fields: {sorted(missing)!r}")


def _non_empty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _optional_non_empty(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _non_empty(value, label)


def _strings(value: object, label: str, duplicate_label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list")
    if not value:
        raise ValueError(f"{label} must not be empty")
    result = tuple(_non_empty(item, label) for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{label} contains a duplicate {duplicate_label}")
    return result


def _parse_date(value: object, label: str) -> date:
    text = _non_empty(value, label)
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO date") from exc


def _https_url(value: object) -> str:
    url = _non_empty(value, "source url")
    parts = urlsplit(url)
    if (
        parts.scheme != "https"
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
    ):
        raise ValueError("source url must be an HTTPS URL without credentials")
    return url


def route_from_mapping(value: object) -> AzRoute:
    raw = _object(value, "route")
    _exact_fields(raw, _ROUTE_FIELDS, "route")
    route_id = _non_empty(raw["route_id"], "route_id")
    try:
        kind = RouteKind(raw["kind"])
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid route kind") from exc
    canonical_provider = _optional_non_empty(
        raw["canonical_provider"], "canonical_provider"
    )
    canonical_model_id = _optional_non_empty(
        raw["canonical_model_id"], "canonical_model_id"
    )
    protocol_values = raw["protocols"]
    if not isinstance(protocol_values, list):
        raise TypeError("protocols must be a list")
    if not protocol_values:
        raise ValueError("protocols must not be empty")
    protocols: list[ProtocolRequirements] = []
    for item in protocol_values:
        protocol_raw = _object(item, "protocol requirements")
        _exact_fields(protocol_raw, _PROTOCOL_FIELDS, "protocol requirements")
        protocol = _non_empty(protocol_raw["protocol"], "protocol")
        scenario_values = _strings(
            protocol_raw["required_scenarios"],
            "required_scenarios",
            "scenario",
        )
        try:
            scenarios = tuple(Scenario(value) for value in scenario_values)
        except ValueError as exc:
            raise ValueError("invalid scenario") from exc
        protocols.append(ProtocolRequirements(protocol, scenarios))
    protocol_names = [item.protocol for item in protocols]
    if len(protocol_names) != len(set(protocol_names)):
        raise ValueError("protocols contains a duplicate protocol")
    catalog_eligible = raw["catalog_eligible"]
    if not isinstance(catalog_eligible, bool):
        raise TypeError("catalog_eligible must be a boolean")

    has_identity = canonical_provider is not None and canonical_model_id is not None
    has_partial_identity = (canonical_provider is None) != (canonical_model_id is None)
    if kind is RouteKind.OFFICIAL_MODEL:
        if not has_identity or has_partial_identity:
            raise ValueError("official model requires canonical identity")
        if route_id != canonical_model_id:
            raise ValueError("official model route_id must equal canonical_model_id")
        if not catalog_eligible:
            raise ValueError("official model must be catalog eligible")
    elif kind is RouteKind.GATEWAY_ALIAS:
        if not has_identity or has_partial_identity:
            raise ValueError("gateway alias requires canonical identity")
        if catalog_eligible:
            raise ValueError("gateway alias cannot be catalog eligible")
    else:
        if canonical_provider is not None or canonical_model_id is not None:
            raise ValueError("external service must not have canonical identity")
        if catalog_eligible:
            raise ValueError("external service cannot be catalog eligible")

    return AzRoute(
        route_id=route_id,
        kind=kind,
        canonical_provider=canonical_provider,
        canonical_model_id=canonical_model_id,
        protocols=tuple(protocols),
        catalog_eligible=catalog_eligible,
    )


def manifest_from_mapping(value: object) -> AzManifest:
    raw = _object(value, "manifest")
    _exact_fields(raw, frozenset({"schema_version", "snapshot", "routes"}), "manifest")
    if raw["schema_version"] != 2:
        raise ValueError("unsupported manifest schema_version")

    snapshot_raw = _object(raw["snapshot"], "snapshot")
    _exact_fields(
        snapshot_raw, frozenset({"captured_at", "count", "sha256"}), "snapshot"
    )
    count = snapshot_raw["count"]
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("snapshot count must be a non-negative integer")
    digest = _non_empty(snapshot_raw["sha256"], "snapshot sha256")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("snapshot sha256 must be lowercase hexadecimal")
    snapshot = InventorySnapshot(
        captured_at=_parse_date(snapshot_raw["captured_at"], "captured_at"),
        count=count,
        sha256=digest,
    )

    route_values = raw["routes"]
    if not isinstance(route_values, list):
        raise TypeError("routes must be a list")
    routes = tuple(route_from_mapping(item) for item in route_values)
    route_ids = [route.route_id for route in routes]
    if len(route_ids) != len(set(route_ids)):
        raise ValueError("manifest contains a duplicate route")
    if route_ids != sorted(route_ids):
        raise ValueError("manifest routes must be in ordinal order")
    if snapshot.count != len(routes):
        raise ValueError("snapshot count does not match routes")
    actual_digest = sha256(("\n".join(route_ids) + "\n").encode()).hexdigest()
    if snapshot.sha256 != actual_digest:
        raise ValueError("snapshot sha256 does not match routes")
    return AzManifest(snapshot=snapshot, routes=routes)


def source_index_from_mapping(value: object) -> SourceIndex:
    raw = _object(value, "source index")
    _exact_fields(raw, frozenset({"schema_version", "sources"}), "source index")
    if raw["schema_version"] != 1:
        raise ValueError("unsupported source index schema_version")
    items = raw["sources"]
    if not isinstance(items, list):
        raise TypeError("sources must be a list")

    sources: dict[tuple[str, str, str], SourceRecord] = {}
    for item in items:
        record_raw = _object(item, "source")
        _exact_fields(record_raw, _SOURCE_FIELDS, "source")
        provider = _non_empty(record_raw["provider"], "provider")
        model_id = _non_empty(record_raw["model_id"], "model_id")
        protocol = _non_empty(record_raw["protocol"], "protocol")
        key = (provider, model_id, protocol)
        if key in sources:
            raise ValueError("source index contains a duplicate source")
        sources[key] = SourceRecord(
            provider=provider,
            model_id=model_id,
            protocol=protocol,
            url=_https_url(record_raw["url"]),
            checked_at=_parse_date(record_raw["checked_at"], "checked_at"),
        )
    return SourceIndex(sources)


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(path: Path | None = None) -> AzManifest:
    return manifest_from_mapping(_load_json(path or _DATA_DIR / "manifest.json"))


def load_sources(path: Path | None = None) -> SourceIndex:
    return source_index_from_mapping(_load_json(path or _DATA_DIR / "sources.json"))


__all__ = [
    "AzManifest",
    "AzRoute",
    "InventorySnapshot",
    "ProtocolRequirements",
    "RouteKind",
    "Scenario",
    "SourceIndex",
    "SourceRecord",
    "load_manifest",
    "load_sources",
    "manifest_from_mapping",
    "route_from_mapping",
    "source_index_from_mapping",
]
