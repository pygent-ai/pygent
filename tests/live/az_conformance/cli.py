from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, cast
from urllib.parse import urlsplit

import httpx

from pygent.core import FrozenJsonObject
from pygent.llm import (
    AnthropicMessagesClient,
    GeminiGenerateContentClient,
    ModelSpec,
    OpenAICompatibleClient,
)
from tests.live.az_conformance.inventory import (
    InventoryDriftError,
    compare_inventory,
    require_matching_inventory,
    require_snapshot_coverage,
)
from tests.live.az_conformance.probe_registry import builtin_probe_registry
from tests.live.az_conformance.results import ResultLedger
from tests.live.az_conformance.runner import (
    ConformanceRunner,
    ProbeCase,
    build_probe_queue,
    build_report,
)
from tests.live.az_conformance.schemas import (
    AzManifest,
    RouteKind,
    Scenario,
    load_manifest,
    load_sources,
)


@dataclass(frozen=True, slots=True)
class LiveConnection:
    base_url: str
    api_key: str = field(repr=False)


class _Delegate(Protocol):
    async def invoke(
        self, model: ModelSpec, payload: FrozenJsonObject
    ) -> FrozenJsonObject: ...

    def stream(
        self, model: ModelSpec, payload: FrozenJsonObject
    ) -> AsyncIterator[FrozenJsonObject]: ...

    async def aclose(self) -> None: ...


class LiveProbeClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        delegate: _Delegate | None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._delegate = delegate
        self._http = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {api_key}",
                "x-api-key": api_key,
                "x-goog-api-key": api_key,
            },
            timeout=httpx.Timeout(120.0),
            trust_env=False,
        )
        self.audio_fixture: bytes | None = None

    async def invoke(
        self, model: ModelSpec, payload: FrozenJsonObject
    ) -> FrozenJsonObject:
        if self._delegate is None:
            raise TypeError("this protocol has no model provider client")
        return await self._delegate.invoke(model, payload)

    def stream(
        self, model: ModelSpec, payload: FrozenJsonObject
    ) -> AsyncIterator[FrozenJsonObject]:
        if self._delegate is None:
            raise TypeError("this protocol has no model provider client")
        return self._delegate.stream(model, payload)

    async def request(
        self, method: str, path: str, **kwargs: object
    ) -> httpx.Response:
        url = path if path.startswith(("https://", "http://")) else self._base_url + path
        return await self._http.request(method, url, **kwargs)

    async def websocket_exchange(
        self, path: str, event: dict[str, object]
    ) -> object:
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError(
                "install the az-conformance extra or run with --with websockets"
            ) from exc
        scheme = "wss" if self._base_url.startswith("https://") else "ws"
        host_and_path = self._base_url.split("://", 1)[1]
        url = f"{scheme}://{host_and_path}{path}"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "OpenAI-Beta": "realtime=v1",
        }
        async with websockets.connect(url, additional_headers=headers) as socket:
            await socket.send(json.dumps(event, separators=(",", ":")))
            return json.loads(await socket.recv())

    async def aclose(self) -> None:
        await self._http.aclose()
        if self._delegate is not None:
            await self._delegate.aclose()


def _connection_from_environment(
    environ: Mapping[str, str],
) -> LiveConnection:
    base_url = environ.get("AZ_BASE_URL", "").strip().rstrip("/")
    api_key = environ.get("AZ_API_KEY", "").strip()
    parts = urlsplit(base_url)
    if (
        parts.scheme != "https"
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
    ):
        raise ValueError(
            "AZ_BASE_URL must be an HTTPS URL without credentials, query, or fragment"
        )
    if not api_key:
        raise ValueError("AZ_API_KEY must be a non-empty environment variable")
    return LiveConnection(base_url=base_url, api_key=api_key)


def _api_root(base_url: str) -> str:
    return _gateway_root(base_url) + "/v1"


def _gateway_root(base_url: str) -> str:
    return base_url.removesuffix("/v1")


def _source_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip()
    if not revision:
        raise RuntimeError("git revision is empty")
    return revision


def _validate(manifest: AzManifest) -> tuple[int, int]:
    registry = builtin_probe_registry()
    required_probes = {
        (requirements.protocol, scenario)
        for route in manifest.routes
        for requirements in route.protocols
        for scenario in requirements.required_scenarios
    }
    missing_probes = len(required_probes - set(registry.probes))
    source_keys = set(load_sources().sources)
    source_identities = {(provider, model_id) for provider, model_id, _ in source_keys}
    missing_sources = 0
    for route in manifest.routes:
        if route.catalog_eligible:
            missing_sources += sum(
                (
                    route.canonical_provider,
                    route.canonical_model_id,
                    requirements.protocol,
                )
                not in source_keys
                for requirements in route.protocols
            )
        elif route.kind is RouteKind.GATEWAY_ALIAS and (
            route.canonical_provider,
            route.canonical_model_id,
        ) not in source_identities:
            missing_sources += 1
    return missing_sources, missing_probes


async def _inventory(
    connection: LiveConnection, manifest: AzManifest
):
    async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
        return await compare_inventory(
            client,
            manifest,
            base_url=connection.base_url,
            headers={"Authorization": f"Bearer {connection.api_key}"},
        )


def _clients(connection: LiveConnection) -> Mapping[str, LiveProbeClient]:
    root = _gateway_root(connection.base_url)
    api_root = _api_root(root)
    openai = LiveProbeClient(
        base_url=api_root,
        api_key=connection.api_key,
        delegate=OpenAICompatibleClient(
            base_url=api_root,
            api_key=connection.api_key,
        ),
    )
    anthropic = LiveProbeClient(
        base_url=api_root,
        api_key=connection.api_key,
        delegate=AnthropicMessagesClient(
            base_url=root,
            api_key=connection.api_key,
        ),
    )
    gemini = LiveProbeClient(
        base_url=api_root,
        api_key=connection.api_key,
        delegate=GeminiGenerateContentClient(
            base_url=root + "/v1beta",
            api_key=connection.api_key,
        ),
    )
    search = LiveProbeClient(
        base_url=api_root,
        api_key=connection.api_key,
        delegate=None,
    )
    return MappingProxyType(
        {
            "openai_chat_completions": openai,
            "anthropic_messages": anthropic,
            "gemini_generate_content": gemini,
            "serpapi_search": search,
        }
    )


async def _close_clients(clients: Mapping[str, LiveProbeClient]) -> None:
    await asyncio.gather(*(client.aclose() for client in set(clients.values())))


def _pending(
    cases: Sequence[ProbeCase],
    ledger: ResultLedger,
    *,
    snapshot_sha256: str,
    source_revision: str,
) -> int:
    return sum(
        ledger.reusable_pass(
            (
                snapshot_sha256,
                source_revision,
                case.route.route_id,
                case.protocol,
                case.scenario,
            )
        )
        is None
        for case in cases
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pygent-az-conformance")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "inventory"):
        command = subparsers.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
    for name in ("run", "report"):
        command = subparsers.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        if name == "run":
            command.add_argument("--route")
            command.add_argument("--scenario", choices=[item.value for item in Scenario])
    return parser


async def _async_main(
    args: argparse.Namespace,
    *,
    environ: Mapping[str, str],
) -> int:
    manifest = load_manifest(args.manifest)
    missing_sources, missing_probes = _validate(manifest)
    if args.command == "validate":
        print(
            f"routes={len(manifest.routes)} classified={len(manifest.routes)} "
            f"unclassified=0 missing_sources={missing_sources} "
            f"missing_probes={missing_probes}"
        )
        print(f"snapshot_sha256={manifest.snapshot.sha256}")
        return 0 if not missing_sources and not missing_probes else 1

    revision = _source_revision()
    if args.command == "report":
        ledger = ResultLedger.load(args.output_dir / "results.jsonl")
        report = build_report(
            manifest=manifest,
            results=ledger.results,
            source_revision=revision,
        )
        print(
            f"routes={len(manifest.routes)} passed={len(report.passed_routes)} "
            f"failed={len(report.failed_keys)} missing={len(report.missing_keys)} "
            f"unexpected={len(report.unexpected_keys)} complete={str(report.complete).lower()}"
        )
        return 0 if report.complete else 1

    connection = _connection_from_environment(environ)
    inventory = await _inventory(connection, manifest)
    print(
        f"inventory_match={str(inventory.matches).lower()} "
        f"snapshot_covered={str(inventory.covers_snapshot).lower()} "
        f"expected={inventory.expected_count} actual={inventory.actual_count} "
        f"added={len(inventory.added)} removed={len(inventory.removed)}"
    )
    if args.command == "inventory":
        try:
            require_matching_inventory(inventory)
        except InventoryDriftError:
            return 1
        return 0

    if missing_sources or missing_probes:
        print(
            f"validation_failed missing_sources={missing_sources} "
            f"missing_probes={missing_probes}"
        )
        return 1
    try:
        require_snapshot_coverage(inventory)
    except InventoryDriftError:
        return 1

    scenario = None if args.scenario is None else Scenario(args.scenario)
    cases = build_probe_queue(manifest, route_id=args.route, scenario=scenario)
    if args.route is not None and not any(
        route.route_id == args.route for route in manifest.routes
    ):
        print("configuration_error unknown_route")
        return 1
    ledger = ResultLedger.load(args.output_dir / "results.jsonl")
    pending = _pending(
        cases,
        ledger,
        snapshot_sha256=manifest.snapshot.sha256,
        source_revision=revision,
    )
    print(f"pending={pending} selected={len(cases)}")
    clients = _clients(connection)
    try:
        runner = ConformanceRunner(
            manifest,
            builtin_probe_registry(),
            ledger,
            revision,
            clients=cast(Mapping[str, object], clients),
        )
        report = await runner.run(inventory, route_id=args.route, scenario=scenario)
    finally:
        await _close_clients(clients)
    print(
        f"passed_routes={len(report.passed_routes)} failed={len(report.failed_keys)} "
        f"missing={len(report.missing_keys)} complete={str(report.complete).lower()}"
    )
    return 0 if report.complete else 1


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> int:
    args = _parser().parse_args(argv)
    try:
        return asyncio.run(_async_main(args, environ=os.environ if environ is None else environ))
    except (httpx.HTTPError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error={type(exc).__name__}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
