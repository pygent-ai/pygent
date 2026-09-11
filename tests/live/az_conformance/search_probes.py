from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import httpx

from tests.live.az_conformance.media_probes import RawProbeClient
from tests.live.az_conformance.results import ErrorKind, ProbeResult
from tests.live.az_conformance.runner import ProbeContext
from tests.live.az_conformance.schemas import AzRoute


def _result(
    context: ProbeContext,
    route: AzRoute,
    *,
    error_kind: ErrorKind | None = None,
) -> ProbeResult:
    return ProbeResult(
        snapshot_sha256=context.snapshot_sha256,
        source_revision=context.source_revision,
        route_id=route.route_id,
        canonical_provider=route.canonical_provider,
        canonical_model_id=route.canonical_model_id,
        protocol=context.protocol,
        scenario=context.scenario,
        status="passed" if error_kind is None else "failed",
        attempts=context.attempt,
        error_kind=error_kind,
    )


async def serpapi_search_probe(
    context: ProbeContext, route: AzRoute
) -> ProbeResult:
    if context.client is None:
        raise ValueError("probe client is not configured")
    client = cast(RawProbeClient, context.client)
    try:
        response = await client.request(
            "POST",
            "/search",
            json={"model": route.route_id, "q": "Pygent agent framework"},
        )
        if response.status_code == 401:
            return _result(context, route, error_kind=ErrorKind.AUTHENTICATION)
        if response.status_code == 403:
            return _result(context, route, error_kind=ErrorKind.PERMISSION)
        if response.status_code == 429:
            return _result(context, route, error_kind=ErrorKind.RATE_LIMIT)
        if response.status_code >= 500:
            return _result(context, route, error_kind=ErrorKind.GATEWAY_UNAVAILABLE)
        if response.status_code >= 400:
            return _result(context, route, error_kind=ErrorKind.INVALID_REQUEST)
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise TypeError("search response must be an object")
        results = payload.get("organic_results", payload.get("results"))
        if (
            not isinstance(results, list)
            or not results
            or not isinstance(results[0], Mapping)
        ):
            raise ValueError("search response has no results")
    except (httpx.HTTPError, ValueError, TypeError):
        return _result(context, route, error_kind=ErrorKind.INVALID_RESPONSE)
    return _result(context, route)


__all__ = ["serpapi_search_probe"]
