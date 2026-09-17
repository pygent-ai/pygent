"""Opt-in live probe for image/video content returned by Pygent tools.

The probe prints only a sanitized result matrix. It loads credentials from the
process environment and then from the repository .env for local development.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from pygent import (
    AIMessage,
    Context,
    GenerationConfig,
    ToolAuthorizationDecision,
    ToolCall,
    ToolKit,
    UserMessage,
)
from pygent.llm import (
    CapabilityPresetCatalog,
    ModelEntry,
    ModelModalities,
    ModelProviderError,
    ModelProviderRequest,
    ModelSpec,
    OpenAICompatibleAdapter,
    OpenAICompatibleClient,
    ToolResultContentCapabilities,
)
from pygent.tool import FileTools
from tests.live.az_conformance.media_fixtures import (
    PNG_BYTES,
    SEQUENCE_MP4_BYTES,
)


@dataclass(frozen=True, slots=True)
class Target:
    name: str
    base_url: str
    api_key: str
    model_id: str


def _load_dotenv() -> None:
    path = Path(__file__).resolve().parents[2] / ".env"
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if name and name not in os.environ:
            os.environ[name] = value.strip().strip("\"'")


def configured_targets(
    az_models: tuple[str, ...], *, include_configured: bool = True
) -> tuple[Target, ...]:
    _load_dotenv()
    candidates = (
        ("configured-glm", "GLM_API_BASE", "GLM_API_KEY", "GLM_MODEL_NAME"),
        ("configured-generic", "API_BASE", "API_KEY", "MODEL_NAME"),
    )
    targets: list[Target] = []
    if include_configured:
        for name, base_name, key_name, model_name in candidates:
            values = tuple(
                os.environ.get(item, "").strip()
                for item in (base_name, key_name, model_name)
            )
            if all(values):
                targets.append(Target(name, values[0], values[1], values[2]))
    az_base = os.environ.get("AZ_BASE_URL", "").strip()
    az_key = os.environ.get("AZ_API_KEY", "").strip()
    if az_base and az_key:
        targets.extend(
            Target(f"az:{model_id}", az_base, az_key, model_id)
            for model_id in az_models
        )
    unique: dict[tuple[str, str], Target] = {}
    for target in targets:
        unique[(target.base_url, target.model_id)] = target
    return tuple(unique.values())


def az_catalog_media_cases() -> tuple[tuple[str, str], ...]:
    """Return AZ OpenAI-chat routes and each catalog-declared media input."""

    root = Path(__file__).resolve().parents[2]
    capabilities = json.loads(
        (root / "src/pygent/llm/data/model_capabilities.json").read_text(
            encoding="utf-8"
        )
    )["models"]
    routes = json.loads(
        (root / "tests/live/az_conformance/manifest.json").read_text(
            encoding="utf-8"
        )
    )["routes"]
    openai_routes = {
        route["route_id"]
        for route in routes
        if any(
            protocol["protocol"] == "openai_chat_completions"
            for protocol in route["protocols"]
        )
    }
    declared: dict[str, set[str]] = {}
    for item in capabilities:
        model_id = item["model_id"]
        if (
            item["protocol"] != "openai_chat_completions"
            or model_id not in openai_routes
        ):
            continue
        inputs = item["capabilities"]["modalities"]["input"]
        modalities = {value for value in inputs if value in ("image", "video")}
        if modalities:
            declared.setdefault(model_id, set()).update(modalities)
    return tuple(
        (model_id, modality)
        for model_id in sorted(declared, key=str.casefold)
        for modality in ("image", "video")
        if modality in declared[model_id]
    )


def _entry(target: Target, modality: str) -> ModelEntry:
    capabilities = CapabilityPresetCatalog.builtin().presets[
        "text_tools_structured_reasoning"
    ].materialize(context_tokens=131_072, max_output_tokens=4096)
    capabilities = replace(
        capabilities,
        modalities=ModelModalities(
            input=("text", modality),
            output=("text",),
        ),
    )
    return ModelEntry(
        target.name,
        ModelSpec(
            provider="live_probe",
            model_id=target.model_id,
            protocol="openai_chat_completions",
            capabilities=capabilities,
        ),
    )


def _authorize(request, _context):
    return ToolAuthorizationDecision(
        call_id=request.call.call_id,
        allowed=True,
        reason_code="live_probe",
    )


async def _tool_message(modality: str):
    with tempfile.TemporaryDirectory(prefix="pygent-media-probe-") as directory:
        root = Path(directory)
        if modality == "image":
            filename = "blue-square.png"
            data = PNG_BYTES
        else:
            filename = "red-then-blue.mp4"
            override = os.environ.get("PYGENT_LIVE_VIDEO_PATH", "").strip()
            data = Path(override).read_bytes() if override else SEQUENCE_MP4_BYTES
        (root / filename).write_bytes(data)
        files = FileTools(workspace_root=root, max_media_bytes=4 * 1024 * 1024)
        toolkit = ToolKit(files.read)
        call = ToolCall(
            call_id=f"call-read-{modality}",
            name="read",
            arguments={"file_path": filename},
        )
        layer = toolkit.local_layer(authorization_adapter=_authorize)
        message, _ = await layer.invoke(
            AIMessage(tool_calls=(call,)),
            toolkit.make_visible_in(Context()),
        )
    assert message.results[0].status == "succeeded"
    return call, message


def _prompt(modality: str) -> str:
    if modality == "image":
        return (
            "Use the read tool result. Identify the central shape and its "
            "color. Reply with exactly BLUE SQUARE if that is what you see."
        )
    return (
        "Use the read tool result. Identify the order of the two solid "
        "colors. Reply with exactly RED THEN BLUE if red appears before blue."
    )


def _passed(modality: str, answer: str) -> bool:
    normalized = " ".join(answer.upper().split())
    expected = "BLUE SQUARE" if modality == "image" else "RED THEN BLUE"
    return expected in normalized


async def probe(target: Target, modality: str) -> dict[str, object]:
    call, tool_message = await _tool_message(modality)
    entry = _entry(target, modality)
    adapter = OpenAICompatibleAdapter(
        tool_result_content=ToolResultContentCapabilities(
            enabled=True,
            modalities=(modality,),
            source_kinds=("inline",),
            max_media_bytes=4 * 1024 * 1024,
        )
    )
    client = OpenAICompatibleClient(
        base_url=target.base_url,
        api_key=target.api_key,
    )
    request = ModelProviderRequest(
        model_key=entry.key,
        model=entry.spec,
        message=tool_message,
        context=Context(
            messages=(
                UserMessage(content=_prompt(modality)),
                AIMessage(tool_calls=(call,)),
            )
        ),
        generation=GenerationConfig(max_output_tokens=512, temperature=0),
    )
    try:
        payload = adapter.build_request(request)
        wire_message = payload["messages"][-1]
        if wire_message["role"] != "tool" or wire_message["tool_call_id"] != call.call_id:
            raise AssertionError("tool message lost its call association")
        response = await client.invoke(entry.spec, payload)
        answer = adapter.parse_response(request, response).message.content.strip()
        return {
            "target": target.name,
            "model": target.model_id,
            "modality": modality,
            "result": "passed" if _passed(modality, answer) else "unexpected_answer",
            "answer": answer[:200],
        }
    except ModelProviderError as exc:
        return {
            "target": target.name,
            "model": target.model_id,
            "modality": modality,
            "result": "provider_error",
            "error_kind": exc.kind.value,
            "reason_code": exc.reason_code.value if exc.reason_code else None,
            "http_status": exc.http_status,
        }
    except Exception as exc:  # noqa: BLE001 - sanitized live-probe boundary
        return {
            "target": target.name,
            "model": target.model_id,
            "modality": modality,
            "result": "probe_error",
            "error_type": type(exc).__name__,
        }
    finally:
        await client.aclose()


async def run(
    az_models: tuple[str, ...], *, include_configured: bool = True
) -> tuple[dict[str, object], ...]:
    targets = configured_targets(az_models, include_configured=include_configured)
    results = []
    for target in targets:
        for modality in ("image", "video"):
            results.append(await probe(target, modality))
    return tuple(results)


async def run_az_catalog_media(
    *, concurrency: int = 4
) -> tuple[dict[str, object], ...]:
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    targets = configured_targets((), include_configured=False)
    if targets:
        raise AssertionError("empty AZ model list must not create targets")
    _load_dotenv()
    base_url = os.environ.get("AZ_BASE_URL", "").strip()
    api_key = os.environ.get("AZ_API_KEY", "").strip()
    if not base_url or not api_key:
        return ()
    cases = az_catalog_media_cases()
    semaphore = asyncio.Semaphore(concurrency)

    async def run_case(model_id: str, modality: str) -> dict[str, object]:
        async with semaphore:
            result = await probe(
                Target(f"az:{model_id}", base_url, api_key, model_id),
                modality,
            )
        progress = {
            key: result[key]
            for key in ("model", "modality", "result")
        }
        print(json.dumps(progress, ensure_ascii=False), file=sys.stderr, flush=True)
        return result

    return tuple(
        await asyncio.gather(
            *(run_case(model_id, modality) for model_id, modality in cases)
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--az-model",
        action="append",
        default=[],
        help="AZ model ID to test; defaults to glm-5.3 and glm-5.3-flash",
    )
    parser.add_argument(
        "--all-az-media-models",
        action="store_true",
        help="test every AZ OpenAI-chat route with catalog-declared image/video input",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="maximum live requests during an all-model sweep",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return nonzero unless every scenario passes",
    )
    parser.add_argument(
        "--az-only",
        action="store_true",
        help="skip GLM_/API_ configured targets and test only AZ models",
    )
    args = parser.parse_args()
    az_models = tuple(args.az_model) or ("glm-5.3", "glm-5.3-flash")
    if args.all_az_media_models:
        results = asyncio.run(run_az_catalog_media(concurrency=args.concurrency))
    else:
        results = asyncio.run(run(az_models, include_configured=not args.az_only))
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if not results:
        return 2
    return (
        1
        if args.strict and any(item["result"] != "passed" for item in results)
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Target",
    "az_catalog_media_cases",
    "configured_targets",
    "main",
    "probe",
    "run",
    "run_az_catalog_media",
]
