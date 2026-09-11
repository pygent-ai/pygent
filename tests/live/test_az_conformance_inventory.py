from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256

import httpx
import pytest

from tests.live.az_conformance.inventory import (
    InventoryDiff,
    InventoryDriftError,
    compare_inventory,
    inventory_digest,
    require_matching_inventory,
    require_probe_scope,
    require_snapshot_coverage,
)
from tests.live.az_conformance.schemas import manifest_from_mapping


def _manifest(route_ids: list[str]):
    routes = [
        {
            "route_id": route_id,
            "kind": "gateway_alias",
            "canonical_provider": "openai",
            "canonical_model_id": "canonical-model",
            "protocols": [
                {
                    "protocol": "openai_chat_completions",
                    "required_scenarios": ["text"],
                }
            ],
            "catalog_eligible": False,
        }
        for route_id in route_ids
    ]
    digest = inventory_digest(route_ids)
    return manifest_from_mapping(
        {
            "schema_version": 2,
            "snapshot": {
                "captured_at": "2026-09-11",
                "count": len(routes),
                "sha256": digest,
            },
            "routes": routes,
        }
    )


@pytest.mark.asyncio
async def test_inventory_mismatch_reports_sorted_drift() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("https://gateway.example/v1/models")
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"id": "new-route"}, {"id": "chatgpt-4o-latest"}],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await compare_inventory(
            client,
            _manifest(["chatgpt-4o-latest"]),
            base_url="https://gateway.example",
        )

    assert result.added == ("new-route",)
    assert result.removed == ()
    assert result.matches is False
    assert result.covers_snapshot is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "base_url",
    ["https://gateway.example", "https://gateway.example/v1/"],
)
async def test_models_endpoint_accepts_host_or_v1_base(base_url: str) -> None:
    seen: list[httpx.URL] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return httpx.Response(200, json={"data": [{"id": "model-a"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await compare_inventory(
            client,
            _manifest(["model-a"]),
            base_url=base_url,
            headers={"Authorization": "Bearer very-secret-token"},
        )

    assert seen == [httpx.URL("https://gateway.example/v1/models")]
    assert result.matches is True
    assert "very-secret-token" not in repr(result)
    assert not hasattr(result, "headers")


def test_snapshot_digest_uses_sorted_ids_and_trailing_newline() -> None:
    assert inventory_digest(("b", "a")) == sha256(b"a\nb\n").hexdigest()


def test_inventory_drift_stops_before_probe_dispatch() -> None:
    dispatched: list[str] = []
    dispatch: Callable[[], None] = lambda: dispatched.append("called")
    diff = InventoryDiff(
        expected_count=1,
        actual_count=1,
        expected_sha256=inventory_digest(("expected",)),
        actual_sha256=inventory_digest(("actual",)),
        added=("actual",),
        removed=("expected",),
    )

    with pytest.raises(InventoryDriftError, match="inventory drift"):
        require_matching_inventory(diff)
        dispatch()

    assert dispatched == []


def test_snapshot_coverage_allows_additions_but_rejects_removals() -> None:
    addition = InventoryDiff(1, 2, "a", "b", ("new",), ())
    require_snapshot_coverage(addition)

    removal = InventoryDiff(2, 1, "a", "b", (), ("required",))
    with pytest.raises(InventoryDriftError, match="inventory drift"):
        require_snapshot_coverage(removal)


def test_probe_scope_ignores_unrelated_missing_routes() -> None:
    diff = InventoryDiff(3, 2, "a", "b", (), ("offline",))

    require_probe_scope(diff, route_id="online")
    with pytest.raises(InventoryDriftError, match="inventory drift"):
        require_probe_scope(diff, route_id="offline")
    with pytest.raises(InventoryDriftError, match="inventory drift"):
        require_probe_scope(diff, route_id=None)


@pytest.mark.asyncio
async def test_inventory_response_rejects_duplicate_or_malformed_ids() -> None:
    responses = [
        {"data": [{"id": "same"}, {"id": "same"}]},
        {"data": [{"id": ""}]},
        {"data": "not-a-list"},
    ]
    for payload in responses:
        async def handler(
            request: httpx.Request, payload: object = payload
        ) -> httpx.Response:
            return httpx.Response(200, json=payload)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises((TypeError, ValueError), match="model inventory"):
                await compare_inventory(
                    client,
                    _manifest(["expected"]),
                    base_url="https://gateway.example/v1",
                )
