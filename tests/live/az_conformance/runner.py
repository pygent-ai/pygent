from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias

from tests.live.az_conformance.inventory import (
    InventoryDiff,
    require_matching_inventory,
)
from tests.live.az_conformance.results import (
    ErrorKind,
    ProbeResult,
    ResultLedger,
)
from tests.live.az_conformance.schemas import AzManifest, AzRoute, Scenario


@dataclass(frozen=True, slots=True)
class ProbeContext:
    snapshot_sha256: str
    source_revision: str
    protocol: str
    scenario: Scenario
    attempt: int


Probe: TypeAlias = Callable[[ProbeContext, AzRoute], Awaitable[ProbeResult]]
ProbeKey: TypeAlias = tuple[str, Scenario]
ProbeCaseKey: TypeAlias = tuple[str, str, Scenario]


@dataclass(frozen=True, slots=True)
class RetryHint:
    retry_after_seconds: float

    def __post_init__(self) -> None:
        if self.retry_after_seconds < 0:
            raise ValueError("retry_after_seconds must not be negative")


@dataclass(frozen=True, slots=True)
class ProbeCase:
    route: AzRoute
    protocol: str
    scenario: Scenario

    @property
    def short_key(self) -> ProbeCaseKey:
        return (self.route.route_id, self.protocol, self.scenario)


@dataclass(frozen=True, slots=True)
class ProbeRegistry:
    probes: Mapping[ProbeKey, Probe]

    def __init__(self, probes: Mapping[ProbeKey, Probe]) -> None:
        object.__setattr__(self, "probes", MappingProxyType(dict(probes)))

    def require(self, protocol: str, scenario: Scenario) -> Probe:
        try:
            return self.probes[(protocol, scenario)]
        except KeyError as exc:
            raise ValueError(f"no probe for {protocol}/{scenario.value}") from exc


@dataclass(frozen=True, slots=True)
class ConformanceReport:
    passed_routes: tuple[str, ...]
    failed_keys: tuple[ProbeCaseKey, ...]
    missing_keys: tuple[ProbeCaseKey, ...]
    unexpected_keys: tuple[ProbeCaseKey, ...]
    complete: bool


_MEDIA_SCENARIOS = frozenset(
    {
        Scenario.IMAGE_OUTPUT,
        Scenario.IMAGE_EDIT,
        Scenario.VIDEO_OUTPUT,
        Scenario.AUDIO_OUTPUT,
        Scenario.AUDIO_INPUT,
        Scenario.REALTIME,
    }
)
_RETRYABLE = frozenset(
    {
        ErrorKind.RATE_LIMIT,
        ErrorKind.TIMEOUT,
        ErrorKind.GATEWAY_UNAVAILABLE,
        ErrorKind.UPSTREAM_UNAVAILABLE,
    }
)


def build_probe_queue(
    manifest: AzManifest,
    *,
    route_id: str | None = None,
    scenario: Scenario | None = None,
) -> tuple[ProbeCase, ...]:
    return tuple(
        ProbeCase(route, protocol, required_scenario)
        for route in manifest.routes
        if route_id is None or route.route_id == route_id
        for protocol in route.protocols
        for required_scenario in route.required_scenarios
        if scenario is None or required_scenario is scenario
    )


def build_report(
    *,
    manifest: AzManifest,
    results: Iterable[ProbeResult],
    source_revision: str,
) -> ConformanceReport:
    expected_cases = build_probe_queue(manifest)
    expected = {case.short_key for case in expected_cases}
    active: dict[ProbeCaseKey, ProbeResult] = {}
    for result in results:
        if (
            result.snapshot_sha256 == manifest.snapshot.sha256
            and result.source_revision == source_revision
        ):
            active[(result.route_id, result.protocol, result.scenario)] = result

    passed = {key for key, result in active.items() if result.status == "passed"}
    failed = {key for key, result in active.items() if result.status == "failed"}
    missing = expected - passed - failed
    unexpected = set(active) - expected
    passed_routes = tuple(
        route.route_id
        for route in manifest.routes
        if {
            case.short_key
            for case in expected_cases
            if case.route.route_id == route.route_id
        }
        <= passed
    )
    complete = (
        len(passed_routes) == len(manifest.routes)
        and not failed
        and not missing
        and not unexpected
    )
    return ConformanceReport(
        passed_routes=passed_routes,
        failed_keys=tuple(sorted(failed)),
        missing_keys=tuple(sorted(missing)),
        unexpected_keys=tuple(sorted(unexpected)),
        complete=complete,
    )


class ConformanceRunner:
    def __init__(
        self,
        manifest: AzManifest,
        registry: ProbeRegistry,
        ledger: ResultLedger,
        source_revision: str,
        *,
        text_concurrency: int = 4,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if text_concurrency < 1:
            raise ValueError("text_concurrency must be at least 1")
        if not source_revision.strip():
            raise ValueError("source_revision must not be empty")
        self.manifest = manifest
        self.registry = registry
        self.ledger = ledger
        self.source_revision = source_revision
        self._text_semaphore = asyncio.Semaphore(text_concurrency)
        self._media_semaphore = asyncio.Semaphore(1)
        self._sleep = sleep

    async def run(
        self,
        inventory: InventoryDiff,
        *,
        route_id: str | None = None,
        scenario: Scenario | None = None,
    ) -> ConformanceReport:
        require_matching_inventory(inventory)
        queue = build_probe_queue(
            self.manifest, route_id=route_id, scenario=scenario
        )
        probes = {
            case.short_key: self.registry.require(case.protocol, case.scenario)
            for case in queue
        }
        first_phase = [case for case in queue if case.scenario is not Scenario.AUDIO_INPUT]
        second_phase = [case for case in queue if case.scenario is Scenario.AUDIO_INPUT]
        await self._run_phase(first_phase, probes)
        await self._run_phase(second_phase, probes)
        return build_report(
            manifest=self.manifest,
            results=self.ledger.results,
            source_revision=self.source_revision,
        )

    async def _run_phase(
        self, cases: list[ProbeCase], probes: Mapping[ProbeCaseKey, Probe]
    ) -> None:
        if cases:
            await asyncio.gather(
                *(self._run_case(case, probes[case.short_key]) for case in cases)
            )

    async def _run_case(self, case: ProbeCase, probe: Probe) -> None:
        result_key = (
            self.manifest.snapshot.sha256,
            self.source_revision,
            case.route.route_id,
            case.protocol,
            case.scenario,
        )
        if self.ledger.reusable_pass(result_key) is not None:
            return
        semaphore = (
            self._media_semaphore
            if case.scenario in _MEDIA_SCENARIOS
            else self._text_semaphore
        )
        for attempt in range(1, 4):
            context = ProbeContext(
                snapshot_sha256=self.manifest.snapshot.sha256,
                source_revision=self.source_revision,
                protocol=case.protocol,
                scenario=case.scenario,
                attempt=attempt,
            )
            async with semaphore:
                result = await probe(context, case.route)
            self._validate_result(case, result, attempt)
            self.ledger.record(result)
            if result.status == "passed" or result.error_kind not in _RETRYABLE:
                return
            if attempt < 3:
                hint = result.private_detail
                delay = (
                    hint.retry_after_seconds
                    if isinstance(hint, RetryHint)
                    else float(2 ** (attempt - 1))
                )
                await self._sleep(delay)

    def _validate_result(
        self, case: ProbeCase, result: ProbeResult, attempt: int
    ) -> None:
        expected = (
            self.manifest.snapshot.sha256,
            self.source_revision,
            case.route.route_id,
            case.protocol,
            case.scenario,
        )
        if result.key != expected:
            raise ValueError("probe returned a result for a different key")
        if result.attempts != attempt:
            raise ValueError("probe result attempts does not match the active attempt")
        if (
            result.canonical_provider != case.route.canonical_provider
            or result.canonical_model_id != case.route.canonical_model_id
        ):
            raise ValueError("probe result canonical identity does not match the route")


__all__ = [
    "ConformanceReport",
    "ConformanceRunner",
    "Probe",
    "ProbeCase",
    "ProbeCaseKey",
    "ProbeContext",
    "ProbeKey",
    "ProbeRegistry",
    "RetryHint",
    "build_probe_queue",
    "build_report",
]
