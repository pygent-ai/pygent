"""Explicit, sanitized live verification for Volcengine Ark video generation models.

This script intentionally uses the raw Ark contents/generations/tasks protocol for
catalog-only modalities. It is not a Pygent Adapter and is not collected as a
pytest test module.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx

_ARK_ROOT = "https://ark.cn-beijing.volces.com/api/v3"
_VIDEO_MODELS = (
    "doubao-seedance-2-0-260128",
    "doubao-seedance-2-5-260628",
)
_PROMPT = "A single blue square slowly moves across a plain white background."
_MAX_VIDEO_BYTES = 100 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ProbeResult:
    model_id: str
    protocol: str
    status: Literal["passed", "failed"]
    error_kind: str | None
    private_detail: object = field(default=None, repr=False, compare=False)


class _ProbeFailure(Exception):
    def __init__(self, error_kind: str) -> None:
        super().__init__(error_kind)
        self.error_kind = error_kind


def summarize_result(result: ProbeResult) -> dict[str, str | None]:
    return {
        "model_id": result.model_id,
        "protocol": result.protocol,
        "status": result.status,
        "error_kind": result.error_kind,
    }


def _passed(model_id: str, protocol: str, detail: object = None) -> ProbeResult:
    return ProbeResult(model_id, protocol, "passed", None, detail)


def _failed(model_id: str, protocol: str, exc: BaseException) -> ProbeResult:
    if isinstance(exc, _ProbeFailure):
        kind = exc.error_kind
    elif isinstance(exc, (httpx.TimeoutException, TimeoutError, asyncio.TimeoutError)):
        kind = "timeout"
    elif isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in (401, 403):
            kind = "authentication"
        elif status == 429:
            kind = "rate_limit"
        elif 400 <= status < 500:
            kind = "invalid_request"
        elif status >= 500:
            kind = "unavailable"
        else:
            kind = "http_error"
    elif isinstance(exc, httpx.TransportError):
        kind = "unavailable"
    else:
        kind = "unknown"
    return ProbeResult(model_id, protocol, "failed", kind, type(exc).__name__)


def _json(response: httpx.Response) -> Mapping[str, Any]:
    response.raise_for_status()
    try:
        value = response.json()
    except ValueError as exc:
        raise _ProbeFailure("invalid_response") from exc
    if not isinstance(value, dict):
        raise _ProbeFailure("invalid_response")
    return value


def _is_video(value: bytes) -> bool:
    return len(value) >= 12 and value[4:8] == b"ftyp"


class ArkVideoProbeRunner:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        api_key: str,
        base_url: str = _ARK_ROOT,
        poll_interval: float = 10,
        poll_timeout: float = 600,
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._root = base_url.rstrip("/")
        self._poll_interval = poll_interval
        self._poll_timeout = poll_timeout

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    async def probe_video(self, model_id: str) -> ProbeResult:
        protocol = "ark_video_generation"
        try:
            response = await self._client.post(
                f"{self._root}/contents/generations/tasks",
                headers=self._headers,
                json={
                    "model": model_id,
                    "content": [{"type": "text", "text": _PROMPT}],
                },
                timeout=60,
            )
            body = _json(response)
            task_id = body.get("id")
            if not isinstance(task_id, str) or not task_id:
                raise _ProbeFailure("invalid_response")
            deadline = asyncio.get_running_loop().time() + self._poll_timeout
            while True:
                if asyncio.get_running_loop().time() >= deadline:
                    raise _ProbeFailure("timeout")
                if self._poll_interval:
                    await asyncio.sleep(self._poll_interval)
                polled = _json(
                    await self._client.get(
                        f"{self._root}/contents/generations/tasks/{task_id}",
                        headers=self._headers,
                        timeout=60,
                    )
                )
                status = polled.get("status")
                if status == "succeeded":
                    return await self._download_result_video(model_id, protocol, polled)
                if status in {"failed", "cancelled"}:
                    raise _ProbeFailure("provider_rejected")
                if status not in {"queued", "running"}:
                    raise _ProbeFailure("invalid_response")
        except (_ProbeFailure, httpx.HTTPError, TimeoutError) as exc:
            return _failed(model_id, protocol, exc)

    async def _download_result_video(
        self, model_id: str, protocol: str, polled: Mapping[str, Any]
    ) -> ProbeResult:
        content = polled.get("content")
        if not isinstance(content, dict):
            raise _ProbeFailure("invalid_response")
        video_url = content.get("video_url")
        if not isinstance(video_url, str):
            raise _ProbeFailure("invalid_response")
        parts = urlsplit(video_url)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise _ProbeFailure("invalid_response")
        media = await self._client.get(video_url, timeout=300)
        media.raise_for_status()
        if not media.content or len(media.content) > _MAX_VIDEO_BYTES:
            raise _ProbeFailure("invalid_response")
        if not _is_video(media.content):
            raise _ProbeFailure("invalid_response")
        return _passed(model_id, protocol)


def _load_env_file(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(name, value)


async def _main(env_file: Path, models: tuple[str, ...]) -> int:
    _load_env_file(env_file)
    if not os.environ.get("ARK_API_KEY"):
        result = ProbeResult("configuration", "environment", "failed", "missing_credential")
        print(json.dumps(summarize_result(result), separators=(",", ":")))
        return 1
    async with httpx.AsyncClient() as client:
        runner = ArkVideoProbeRunner(
            client=client,
            api_key=os.environ["ARK_API_KEY"],
            base_url=os.environ.get("ARK_API_BASE", _ARK_ROOT),
        )
        results = [await runner.probe_video(model_id) for model_id in models]
    for result in results:
        print(
            json.dumps(
                summarize_result(result),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    return (
        0
        if len(results) == len(models) and all(
            result.status == "passed" for result in results
        )
        else 1
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument(
        "--model",
        action="append",
        choices=_VIDEO_MODELS,
        help="restrict the probe to one catalog model; repeatable",
    )
    arguments = parser.parse_args()
    selected = tuple(arguments.model) if arguments.model else _VIDEO_MODELS
    sys.exit(asyncio.run(_main(arguments.env_file, selected)))
