from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from tests.live.az_conformance.schemas import (
    RouteKind,
    Scenario,
    load_manifest,
    load_sources,
    manifest_from_mapping,
    route_from_mapping,
    source_index_from_mapping,
)

_ROUTE_IDS = Path(__file__).with_name("az_conformance") / "route_ids.txt"


def test_frozen_route_id_snapshot_is_exact() -> None:
    assert _ROUTE_IDS.is_file(), "frozen AZ route ID snapshot is missing"
    text = _ROUTE_IDS.read_text(encoding="utf-8")
    ids = text.splitlines()
    payload = ("\n".join(ids) + "\n").encode()

    assert text.endswith("\n")
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids)) == 211
    assert sha256(payload).hexdigest() == (
        "7398801a0d480abc1b45d64d87e9c8eac508404f53dffc37d202c222f264330e"
    )


def _route(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "route_id": "gpt-5.4-urg",
        "kind": "gateway_alias",
        "canonical_provider": "openai",
        "canonical_model_id": "gpt-5.4",
        "protocols": [
            {
                "protocol": "openai_chat_completions",
                "required_scenarios": ["text", "text_stream"],
            }
        ],
        "catalog_eligible": False,
    }
    value.update(changes)
    return value


def _manifest(routes: list[dict[str, object]]) -> dict[str, object]:
    ids = [str(route["route_id"]) for route in routes]
    digest = sha256(("\n".join(ids) + "\n").encode()).hexdigest()
    return {
        "schema_version": 2,
        "snapshot": {
            "captured_at": "2026-09-11",
            "count": len(ids),
            "sha256": digest,
        },
        "routes": routes,
    }


def test_gateway_alias_requires_canonical_model_and_is_not_catalog_eligible() -> None:
    route = route_from_mapping(_route())

    assert route.kind is RouteKind.GATEWAY_ALIAS
    assert route.canonical_model_id == "gpt-5.4"
    assert route.protocols[0].protocol == "openai_chat_completions"
    assert route.protocols[0].required_scenarios == (
        Scenario.TEXT,
        Scenario.TEXT_STREAM,
    )


@pytest.mark.parametrize("kind", ["gateway_alias", "external_service"])
def test_nonofficial_route_cannot_enter_catalog(kind: str) -> None:
    value = _route(kind=kind, catalog_eligible=True)
    if kind == "external_service":
        value.update(
            canonical_provider=None,
            canonical_model_id=None,
            protocols=[
                {"protocol": "serpapi_search", "required_scenarios": ["search"]}
            ],
        )

    with pytest.raises(ValueError, match="cannot be catalog eligible"):
        route_from_mapping(value)


def test_official_model_requires_canonical_identity_and_catalog_eligibility() -> None:
    value = _route(
        route_id="gpt-5.4",
        kind="official_model",
        canonical_model_id=None,
        catalog_eligible=True,
    )
    with pytest.raises(ValueError, match="canonical identity"):
        route_from_mapping(value)

    value["canonical_model_id"] = "gpt-5.4"
    value["catalog_eligible"] = False
    with pytest.raises(ValueError, match="must be catalog eligible"):
        route_from_mapping(value)


def test_external_service_has_no_canonical_model() -> None:
    route = route_from_mapping(
        _route(
            route_id="serpapi-google",
            kind="external_service",
            canonical_provider=None,
            canonical_model_id=None,
            protocols=[
                {"protocol": "serpapi_search", "required_scenarios": ["search"]}
            ],
        )
    )
    assert route.kind is RouteKind.EXTERNAL_SERVICE
    assert route.canonical_provider is None

    with pytest.raises(ValueError, match="must not have canonical identity"):
        route_from_mapping(
            _route(
                kind="external_service",
                protocols=[
                    {
                        "protocol": "serpapi_search",
                        "required_scenarios": ["search"],
                    }
                ],
            )
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"extra": True}, "unknown fields"),
        ({"protocols": []}, "protocols must not be empty"),
        (
            {
                "protocols": [
                    {"protocol": "p1", "required_scenarios": ["text"]},
                    {"protocol": "p1", "required_scenarios": ["tools"]},
                ]
            },
            "duplicate protocol",
        ),
        (
            {"protocols": [{"protocol": "p1", "required_scenarios": []}]},
            "required_scenarios must not be empty",
        ),
        (
            {
                "protocols": [
                    {"protocol": "p1", "required_scenarios": ["text", "text"]}
                ]
            },
            "duplicate scenario",
        ),
        (
            {
                "protocols": [
                    {"protocol": "p1", "required_scenarios": ["made_up"]}
                ]
            },
            "invalid scenario",
        ),
    ],
)
def test_route_mapping_is_strict(change: dict[str, object], message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        route_from_mapping(_route(**change))


def test_manifest_requires_sorted_unique_routes_and_matching_snapshot() -> None:
    first = _route(route_id="a-route")
    second = _route(route_id="b-route")
    manifest = manifest_from_mapping(_manifest([first, second]))
    assert tuple(route.route_id for route in manifest.routes) == ("a-route", "b-route")

    with pytest.raises(ValueError, match="ordinal order"):
        manifest_from_mapping(_manifest([second, first]))
    with pytest.raises(ValueError, match="duplicate route"):
        manifest_from_mapping(_manifest([first, first]))

    bad_snapshot = _manifest([first])
    assert isinstance(bad_snapshot["snapshot"], dict)
    bad_snapshot["snapshot"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="snapshot sha256"):
        manifest_from_mapping(bad_snapshot)


def test_parsed_values_are_immutable() -> None:
    manifest = manifest_from_mapping(_manifest([_route()]))
    with pytest.raises(FrozenInstanceError):
        manifest.routes[0].route_id = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        manifest.routes[0].protocols[0] = "changed"  # type: ignore[index]


def test_source_index_is_strict_https_and_unique() -> None:
    value = {
        "schema_version": 1,
        "sources": [
            {
                "provider": "openai",
                "model_id": "gpt-5.4",
                "protocol": "openai_chat_completions",
                "url": "https://developers.openai.com/api/docs/models",
                "checked_at": "2026-09-11",
            }
        ],
    }
    index = source_index_from_mapping(value)
    key = ("openai", "gpt-5.4", "openai_chat_completions")
    assert index.sources[key].checked_at.isoformat() == "2026-09-11"
    with pytest.raises(TypeError):
        index.sources[key] = index.sources[key]  # type: ignore[index]

    duplicate = json.loads(json.dumps(value))
    duplicate["sources"].append(duplicate["sources"][0])
    with pytest.raises(ValueError, match="duplicate source"):
        source_index_from_mapping(duplicate)

    insecure = json.loads(json.dumps(value))
    insecure["sources"][0]["url"] = "http://example.test/model"
    with pytest.raises(ValueError, match="HTTPS"):
        source_index_from_mapping(insecure)


def test_builtin_source_index_is_populated() -> None:
    index = load_sources()
    assert len(index.sources) == 215


def test_builtin_manifest_classifies_the_exact_frozen_inventory() -> None:
    manifest = load_manifest()
    frozen_ids = tuple(_ROUTE_IDS.read_text(encoding="utf-8").splitlines())

    assert tuple(route.route_id for route in manifest.routes) == frozen_ids
    assert len(manifest.routes) == 211
    assert all(route.protocols for route in manifest.routes)
    assert all(
        requirements.required_scenarios
        for route in manifest.routes
        for requirements in route.protocols
    )


def test_builtin_manifest_preserves_advertised_protocol_counts() -> None:
    manifest = load_manifest()

    protocol_names = [
        requirements.protocol
        for route in manifest.routes
        for requirements in route.protocols
    ]
    assert protocol_names.count("openai_chat_completions") == 209
    assert protocol_names.count("anthropic_messages") == 46
    assert protocol_names.count("gemini_generate_content") == 28
    assert protocol_names.count("serpapi_search") == 7


def test_builtin_manifest_scopes_capability_probes_to_their_wire_protocol() -> None:
    routes = {route.route_id: route for route in load_manifest().routes}
    image_protocols = {
        item.protocol: {scenario.value for scenario in item.required_scenarios}
        for item in routes["gemini-2.5-flash-image"].protocols
    }
    assert image_protocols["gemini_generate_content"] == {
        "image_input",
        "image_output",
        "image_edit",
    }
    assert image_protocols["openai_chat_completions"] == {"image_input"}
    assert [
        item.protocol for item in routes["gemini-2.5-flash-preview-tts"].protocols
    ] == ["gemini_generate_content"]
    assert all(
        Scenario.JSON_OBJECT not in item.required_scenarios
        for route in routes.values()
        for item in route.protocols
        if item.protocol == "anthropic_messages"
    )


def test_builtin_manifest_exercises_every_catalogued_video_input() -> None:
    routes = {route.route_id: route for route in load_manifest().routes}
    for model_id in (
        "qwen-omni-turbo",
        "qwen3.5-omni-flash",
        "qwen3.5-omni-plus",
    ):
        protocol = next(
            item
            for item in routes[model_id].protocols
            if item.protocol == "openai_chat_completions"
        )
        assert Scenario.VIDEO_INPUT in protocol.required_scenarios


def test_every_official_catalog_triple_and_alias_target_has_a_source() -> None:
    manifest = load_manifest()
    source_keys = set(load_sources().sources)
    sourced_identities = {(provider, model_id) for provider, model_id, _ in source_keys}

    for route in manifest.routes:
        if route.catalog_eligible:
            assert route.canonical_provider is not None
            assert route.canonical_model_id is not None
            for requirements in route.protocols:
                assert (
                    route.canonical_provider,
                    route.canonical_model_id,
                    requirements.protocol,
                ) in source_keys
        elif route.kind is RouteKind.GATEWAY_ALIAS:
            assert (route.canonical_provider, route.canonical_model_id) in sourced_identities


def test_builtin_sources_only_use_reviewed_primary_domains() -> None:
    allowed_hosts = {
        "api-docs.deepseek.com",
        "ai.google.dev",
        "developers.openai.com",
        "docs.anthropic.com",
        "docs.bigmodel.cn",
        "docs.x.ai",
        "help.aliyun.com",
        "ir.kuaishou.com",
        "platform.minimax.io",
        "platform.moonshot.cn",
        "www.volcengine.com",
    }

    assert {
        urlsplit(source.url).hostname for source in load_sources().sources.values()
    } <= allowed_hosts
