from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from tests.live.az_conformance.schemas import AzManifest


@dataclass(frozen=True, slots=True)
class InventoryDiff:
    expected_count: int
    actual_count: int
    expected_sha256: str
    actual_sha256: str
    added: tuple[str, ...]
    removed: tuple[str, ...]

    @property
    def matches(self) -> bool:
        return not self.added and not self.removed

    @property
    def covers_snapshot(self) -> bool:
        return not self.removed


class InventoryDriftError(RuntimeError):
    def __init__(self, diff: InventoryDiff) -> None:
        self.diff = diff
        super().__init__(
            "model inventory drift: "
            f"added={len(diff.added)} removed={len(diff.removed)}"
        )


def inventory_digest(route_ids: Iterable[str]) -> str:
    ordered = sorted(route_ids)
    return sha256(("\n".join(ordered) + "\n").encode()).hexdigest()


def _models_url(base_url: str) -> str:
    parts = urlsplit(base_url)
    if not parts.scheme or not parts.netloc or parts.query or parts.fragment:
        raise ValueError("base_url must be an absolute URL without query or fragment")
    if parts.username is not None or parts.password is not None:
        raise ValueError("base_url must not contain credentials")
    path = parts.path.rstrip("/")
    if path not in {"", "/v1"}:
        raise ValueError("base_url path must be empty or /v1")
    return urlunsplit((parts.scheme, parts.netloc, "/v1/models", "", ""))


def _inventory_ids(value: object) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        raise TypeError("model inventory must be an object")
    data = value.get("data")
    if not isinstance(data, list):
        raise TypeError("model inventory data must be a list")
    ids: list[str] = []
    for item in data:
        if not isinstance(item, Mapping):
            raise TypeError("model inventory entries must be objects")
        model_id = item.get("id")
        if not isinstance(model_id, str):
            raise TypeError("model inventory IDs must be strings")
        if not model_id.strip():
            raise ValueError("model inventory IDs must be non-empty strings")
        ids.append(model_id)
    if len(ids) != len(set(ids)):
        raise ValueError("model inventory contains duplicate IDs")
    return tuple(sorted(ids))


async def fetch_inventory_ids(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    headers: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    response = await client.get(_models_url(base_url), headers=headers)
    response.raise_for_status()
    try:
        payload: Any = response.json()
    except ValueError as exc:
        raise ValueError("model inventory response must be JSON") from exc
    return _inventory_ids(payload)


async def compare_inventory(
    client: httpx.AsyncClient,
    manifest: AzManifest,
    *,
    base_url: str,
    headers: Mapping[str, str] | None = None,
) -> InventoryDiff:
    actual_ids = await fetch_inventory_ids(
        client, base_url=base_url, headers=headers
    )
    expected_ids = tuple(route.route_id for route in manifest.routes)
    expected_set = frozenset(expected_ids)
    actual_set = frozenset(actual_ids)
    return InventoryDiff(
        expected_count=len(expected_ids),
        actual_count=len(actual_ids),
        expected_sha256=manifest.snapshot.sha256,
        actual_sha256=inventory_digest(actual_ids),
        added=tuple(sorted(actual_set - expected_set)),
        removed=tuple(sorted(expected_set - actual_set)),
    )


def require_matching_inventory(diff: InventoryDiff) -> None:
    if not diff.matches:
        raise InventoryDriftError(diff)


def require_snapshot_coverage(diff: InventoryDiff) -> None:
    if not diff.covers_snapshot:
        raise InventoryDriftError(diff)


def require_probe_scope(diff: InventoryDiff, *, route_id: str | None) -> None:
    if route_id is None:
        require_snapshot_coverage(diff)
    elif route_id in diff.removed:
        raise InventoryDriftError(diff)


__all__ = [
    "InventoryDiff",
    "InventoryDriftError",
    "compare_inventory",
    "fetch_inventory_ids",
    "inventory_digest",
    "require_matching_inventory",
    "require_probe_scope",
    "require_snapshot_coverage",
]
