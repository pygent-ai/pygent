from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.live.az_conformance import cli
from tests.live.az_conformance.inventory import InventoryDiff, inventory_digest
from tests.live.az_conformance.results import ErrorKind, ProbeResult, ResultLedger
from tests.live.az_conformance.runner import (
    ConformanceRunner,
    ProbeContext,
    ProbeRegistry,
    RetryHint,
    build_probe_queue,
    build_report,
)
from tests.live.az_conformance.schemas import Scenario, manifest_from_mapping

_SHA = "7" * 64
_REVISION = "abc1234"


def _manifest(
    routes: list[tuple[str, list[str], list[str]]],
):
    values = [
        {
            "route_id": route_id,
            "kind": "gateway_alias",
            "canonical_provider": "openai",
            "canonical_model_id": "canonical-model",
            "protocols": [
                {"protocol": protocol, "required_scenarios": scenarios}
                for protocol in protocols
            ],
            "catalog_eligible": False,
        }
        for route_id, protocols, scenarios in routes
    ]
    route_ids = [value["route_id"] for value in values]
    return manifest_from_mapping(
        {
            "schema_version": 2,
            "snapshot": {
                "captured_at": "2026-09-11",
                "count": len(values),
                "sha256": inventory_digest(route_ids),
            },
            "routes": values,
        }
    )


def _matching(manifest) -> InventoryDiff:
    return InventoryDiff(
        expected_count=len(manifest.routes),
        actual_count=len(manifest.routes),
        expected_sha256=manifest.snapshot.sha256,
        actual_sha256=manifest.snapshot.sha256,
        added=(),
        removed=(),
    )


def _result(
    context: ProbeContext,
    route_id: str,
    scenario: Scenario,
    *,
    status: str = "passed",
    error_kind: ErrorKind | None = None,
    detail: object = None,
) -> ProbeResult:
    return ProbeResult(
        snapshot_sha256=context.snapshot_sha256,
        source_revision=context.source_revision,
        route_id=route_id,
        canonical_provider="openai",
        canonical_model_id="canonical-model",
        protocol=context.protocol,
        scenario=scenario,
        status=status,  # type: ignore[arg-type]
        attempts=context.attempt,
        error_kind=error_kind,
        private_detail=detail,
    )


def test_probe_registry_requires_exact_protocol_and_scenario() -> None:
    async def probe(context, route):
        return _result(context, route.route_id, Scenario.TEXT)

    registry = ProbeRegistry({("p1", Scenario.TEXT): probe})
    assert registry.require("p1", Scenario.TEXT) is probe
    with pytest.raises(ValueError, match="no probe for p1/tools"):
        registry.require("p1", Scenario.TOOLS)
    with pytest.raises(ValueError, match="no probe for p2/text"):
        registry.require("p2", Scenario.TEXT)


def test_probe_queue_is_exact_and_deterministic() -> None:
    manifest = _manifest(
        [
            ("a", ["p1", "p2"], ["text", "tools"]),
            ("b", ["p1"], ["text"]),
        ]
    )
    assert [case.short_key for case in build_probe_queue(manifest)] == [
        ("a", "p1", Scenario.TEXT),
        ("a", "p1", Scenario.TOOLS),
        ("a", "p2", Scenario.TEXT),
        ("a", "p2", Scenario.TOOLS),
        ("b", "p1", Scenario.TEXT),
    ]


def test_probe_queue_keeps_scenarios_scoped_to_their_protocol() -> None:
    manifest = manifest_from_mapping(
        {
            "schema_version": 2,
            "snapshot": {
                "captured_at": "2026-09-11",
                "count": 1,
                "sha256": inventory_digest(["a"]),
            },
            "routes": [
                {
                    "route_id": "a",
                    "kind": "gateway_alias",
                    "canonical_provider": "openai",
                    "canonical_model_id": "canonical-model",
                    "protocols": [
                        {"protocol": "p1", "required_scenarios": ["text"]},
                        {"protocol": "p2", "required_scenarios": ["tools"]},
                    ],
                    "catalog_eligible": False,
                }
            ],
        }
    )

    assert [case.short_key for case in build_probe_queue(manifest)] == [
        ("a", "p1", Scenario.TEXT),
        ("a", "p2", Scenario.TOOLS),
    ]


@pytest.mark.asyncio
async def test_runner_stops_on_drift_before_probe_lookup(tmp_path: Path) -> None:
    manifest = _manifest([("a", ["p1"], ["text"])])
    registry = ProbeRegistry({})
    runner = ConformanceRunner(
        manifest=manifest,
        registry=registry,
        ledger=ResultLedger(tmp_path / "results.jsonl"),
        source_revision=_REVISION,
    )
    drift = InventoryDiff(1, 1, _SHA, _SHA, ("new",), ("a",))
    with pytest.raises(RuntimeError, match="inventory drift"):
        await runner.run(drift)
    assert not runner.ledger.path.exists()


@pytest.mark.asyncio
async def test_runner_reuses_only_exact_passed_checkpoint(tmp_path: Path) -> None:
    manifest = _manifest([("a", ["p1"], ["text"])])
    calls = 0

    async def probe(context, route):
        nonlocal calls
        calls += 1
        return _result(context, route.route_id, Scenario.TEXT)

    ledger = ResultLedger(tmp_path / "results.jsonl")
    runner = ConformanceRunner(manifest, ProbeRegistry({("p1", Scenario.TEXT): probe}), ledger, _REVISION)
    first = await runner.run(_matching(manifest))
    second = await runner.run(_matching(manifest))

    assert first.complete and second.complete
    assert calls == 1


@pytest.mark.asyncio
async def test_runner_injects_the_client_for_the_exact_protocol(tmp_path: Path) -> None:
    manifest = _manifest([("a", ["p1"], ["text"])])
    client = object()

    async def probe(context, route):
        assert context.client is client
        return _result(context, route.route_id, Scenario.TEXT)

    runner = ConformanceRunner(
        manifest,
        ProbeRegistry({("p1", Scenario.TEXT): probe}),
        ResultLedger(tmp_path / "results.jsonl"),
        _REVISION,
        clients={"p1": client},
    )
    assert (await runner.run(_matching(manifest))).complete


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_kind",
    [
        ErrorKind.RATE_LIMIT,
        ErrorKind.TIMEOUT,
        ErrorKind.GATEWAY_UNAVAILABLE,
        ErrorKind.UPSTREAM_UNAVAILABLE,
    ],
)
async def test_runner_retries_only_transient_failures(
    tmp_path: Path, error_kind: ErrorKind
) -> None:
    manifest = _manifest([("a", ["p1"], ["text"])])
    attempts: list[int] = []
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    async def probe(context, route):
        attempts.append(context.attempt)
        if context.attempt < 3:
            return _result(
                context,
                route.route_id,
                Scenario.TEXT,
                status="failed",
                error_kind=error_kind,
                detail=RetryHint(4.5) if context.attempt == 1 else None,
            )
        return _result(context, route.route_id, Scenario.TEXT)

    runner = ConformanceRunner(
        manifest,
        ProbeRegistry({("p1", Scenario.TEXT): probe}),
        ResultLedger(tmp_path / "results.jsonl"),
        _REVISION,
        sleep=sleep,
    )
    report = await runner.run(_matching(manifest))
    assert report.complete
    assert attempts == [1, 2, 3]
    assert delays == (
        [4.5, 30.0]
        if error_kind is ErrorKind.RATE_LIMIT
        else [4.5, 2.0]
    )


@pytest.mark.asyncio
async def test_runner_does_not_retry_permanent_failure(tmp_path: Path) -> None:
    manifest = _manifest([("a", ["p1"], ["text"])])
    calls = 0

    async def probe(context, route):
        nonlocal calls
        calls += 1
        return _result(
            context,
            route.route_id,
            Scenario.TEXT,
            status="failed",
            error_kind=ErrorKind.INVALID_REQUEST,
        )

    runner = ConformanceRunner(
        manifest,
        ProbeRegistry({("p1", Scenario.TEXT): probe}),
        ResultLedger(tmp_path / "results.jsonl"),
        _REVISION,
    )
    report = await runner.run(_matching(manifest))
    assert not report.complete
    assert calls == 1


@pytest.mark.asyncio
async def test_runner_enforces_a_wall_clock_timeout_per_attempt(
    tmp_path: Path,
) -> None:
    manifest = _manifest([("a", ["p1"], ["text"])])
    calls = 0

    async def probe(context, route):
        nonlocal calls
        calls += 1
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    runner = ConformanceRunner(
        manifest,
        ProbeRegistry({("p1", Scenario.TEXT): probe}),
        ResultLedger(tmp_path / "results.jsonl"),
        _REVISION,
        attempt_timeout_seconds=0.001,
        sleep=lambda _: asyncio.sleep(0),
    )

    report = await runner.run(_matching(manifest))

    assert not report.complete
    assert calls == 3
    result = runner.ledger.results[-1]
    assert result.error_kind is ErrorKind.TIMEOUT
    assert result.attempts == 3


@pytest.mark.asyncio
async def test_media_is_serialized_and_text_concurrency_is_bounded(
    tmp_path: Path,
) -> None:
    manifest = _manifest(
        [
            ("a", ["p1"], ["text", "image_output"]),
            ("b", ["p1"], ["text", "image_output"]),
        ]
    )
    active = {"text": 0, "media": 0}
    peaks = {"text": 0, "media": 0}

    async def probe(context, route):
        bucket = "media" if context.scenario is Scenario.IMAGE_OUTPUT else "text"
        active[bucket] += 1
        peaks[bucket] = max(peaks[bucket], active[bucket])
        await asyncio.sleep(0)
        active[bucket] -= 1
        return _result(context, route.route_id, context.scenario)

    registry = ProbeRegistry(
        {
            ("p1", Scenario.TEXT): probe,
            ("p1", Scenario.IMAGE_OUTPUT): probe,
        }
    )
    runner = ConformanceRunner(
        manifest,
        registry,
        ResultLedger(tmp_path / "results.jsonl"),
        _REVISION,
        text_concurrency=2,
    )
    assert (await runner.run(_matching(manifest))).complete
    assert peaks == {"text": 2, "media": 1}


@pytest.mark.asyncio
async def test_audio_output_completes_before_audio_input(tmp_path: Path) -> None:
    manifest = _manifest(
        [("a", ["p1"], ["audio_output", "audio_input"])]
    )
    events: list[str] = []

    async def probe(context, route):
        events.append(context.scenario.value)
        return _result(context, route.route_id, context.scenario)

    registry = ProbeRegistry(
        {
            ("p1", Scenario.AUDIO_OUTPUT): probe,
            ("p1", Scenario.AUDIO_INPUT): probe,
        }
    )
    runner = ConformanceRunner(
        manifest,
        registry,
        ResultLedger(tmp_path / "results.jsonl"),
        _REVISION,
    )
    await runner.run(_matching(manifest))
    assert events == ["audio_output", "audio_input"]


@pytest.mark.asyncio
async def test_selected_audio_input_rebuilds_an_in_memory_fixture(
    tmp_path: Path,
) -> None:
    manifest = _manifest(
        [
            ("speaker", ["p1"], ["audio_output"]),
            ("transcriber", ["p1"], ["audio_input"]),
        ]
    )
    events: list[str] = []

    class AudioClient:
        audio_fixture: bytes | None = None

    client = AudioClient()

    async def output_probe(context, route):
        events.append(f"{route.route_id}:{context.scenario.value}")
        client.audio_fixture = b"audio"
        return _result(context, route.route_id, context.scenario)

    async def input_probe(context, route):
        events.append(f"{route.route_id}:{context.scenario.value}")
        assert client.audio_fixture == b"audio"
        return _result(context, route.route_id, context.scenario)

    ledger = ResultLedger(tmp_path / "results.jsonl")
    output_context = ProbeContext(
        snapshot_sha256=manifest.snapshot.sha256,
        source_revision=_REVISION,
        protocol="p1",
        scenario=Scenario.AUDIO_OUTPUT,
        attempt=1,
    )
    ledger.record(_result(output_context, "speaker", Scenario.AUDIO_OUTPUT))
    runner = ConformanceRunner(
        manifest,
        ProbeRegistry(
            {
                ("p1", Scenario.AUDIO_OUTPUT): output_probe,
                ("p1", Scenario.AUDIO_INPUT): input_probe,
            }
        ),
        ledger,
        _REVISION,
        clients={"p1": client},
    )

    await runner.run(_matching(manifest), scenario=Scenario.AUDIO_INPUT)

    assert events == ["speaker:audio_output", "transcriber:audio_input"]


@pytest.mark.asyncio
async def test_runner_propagates_cancellation(tmp_path: Path) -> None:
    manifest = _manifest([("a", ["p1"], ["text"])])

    async def probe(context, route):
        raise asyncio.CancelledError

    runner = ConformanceRunner(
        manifest,
        ProbeRegistry({("p1", Scenario.TEXT): probe}),
        ResultLedger(tmp_path / "results.jsonl"),
        _REVISION,
    )
    with pytest.raises(asyncio.CancelledError):
        await runner.run(_matching(manifest))


def test_completion_requires_every_required_key_to_pass() -> None:
    manifest = _manifest(
        [("a", ["p1"], ["text", "tools"]), ("b", ["p1"], ["text"])]
    )
    context = ProbeContext(
        snapshot_sha256=manifest.snapshot.sha256,
        source_revision=_REVISION,
        protocol="p1",
        scenario=Scenario.TEXT,
        attempt=1,
    )
    report = build_report(
        manifest=manifest,
        results=[_result(context, "a", Scenario.TEXT)],
        source_revision=_REVISION,
    )
    assert report.passed_routes == ()
    assert report.missing_keys
    assert report.complete is False


def test_cli_validate_needs_no_credentials_and_reports_sanitized_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = Path(__file__).with_name("az_conformance") / "manifest.json"

    assert cli.main(["validate", "--manifest", str(manifest)], environ={}) == 0

    output = capsys.readouterr().out
    assert "routes=211 classified=211" in output
    assert "missing_sources=0 missing_probes=0" in output


def test_cli_connection_is_strict_and_hides_credentials() -> None:
    connection = cli._connection_from_environment(
        {"AZ_BASE_URL": "https://gateway.example/v1", "AZ_API_KEY": "secret-key"}
    )
    assert "secret-key" not in repr(connection)

    with pytest.raises(ValueError, match="HTTPS"):
        cli._connection_from_environment(
            {"AZ_BASE_URL": "http://gateway.example", "AZ_API_KEY": "secret-key"}
        )
    with pytest.raises(ValueError, match="credentials"):
        cli._connection_from_environment(
            {
                "AZ_BASE_URL": "https://user:password@gateway.example",
                "AZ_API_KEY": "secret-key",
            }
        )


@pytest.mark.parametrize(
    "base_url", ["https://gateway.example", "https://gateway.example/v1"]
)
def test_cli_normalizes_gateway_root_before_protocol_paths(base_url: str) -> None:
    assert cli._gateway_root(base_url) == "https://gateway.example"
    assert cli._api_root(base_url) == "https://gateway.example/v1"


def test_cli_inventory_drift_stops_before_client_construction(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path = Path(__file__).with_name("az_conformance") / "manifest.json"
    manifest = cli.load_manifest(manifest_path)
    drift = InventoryDiff(
        expected_count=211,
        actual_count=210,
        expected_sha256=manifest.snapshot.sha256,
        actual_sha256="8" * 64,
        added=(),
        removed=(manifest.routes[0].route_id,),
    )

    async def inventory(*args, **kwargs):
        return drift

    monkeypatch.setattr(cli, "_inventory", inventory)
    monkeypatch.setattr(cli, "_source_revision", lambda: "revision")
    monkeypatch.setattr(
        cli,
        "_clients",
        lambda connection: pytest.fail("clients constructed before drift gate"),
    )

    result = cli.main(
        [
            "run",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(manifest_path.parent / "unused-output"),
            "--scenario",
            "text",
        ],
        environ={
            "AZ_BASE_URL": "https://gateway.example",
            "AZ_API_KEY": "secret-key",
        },
    )

    assert result == 1
    output = capsys.readouterr().out
    assert "inventory_match=false" in output
    assert "secret-key" not in output
