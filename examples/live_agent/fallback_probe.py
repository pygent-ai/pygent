"""Real fallback probe through a local authentication-enforcing HTTP boundary."""

from __future__ import annotations

import argparse
import asyncio
import json
import secrets
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from .agent import LiveAgentConfig
from .benchmark import run_live_benchmark


class AuthenticationProxy:
    """Reject invalid credentials locally and forward valid calls upstream."""

    def __init__(
        self,
        upstream_base_url: str,
        expected_api_key: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._upstream = httpx.URL(upstream_base_url)
        self._expected_authorization = f"Bearer {expected_api_key}"
        self._client = httpx.AsyncClient(transport=transport, timeout=None)
        self.rejected = 0
        self.forwarded = 0
        self.app = Starlette(
            routes=[Route("/{path:path}", self._handle, methods=["POST"])]
        )

    async def _handle(self, request: Request) -> Response:
        authorization = request.headers.get("authorization", "")
        if not secrets.compare_digest(
            authorization, self._expected_authorization
        ):
            self.rejected += 1
            return JSONResponse(
                {"error": {"message": "authentication rejected"}},
                status_code=401,
            )

        self.forwarded += 1
        upstream_url = self._upstream.copy_with(
            path=request.url.path,
            query=request.url.query.encode(),
        )
        headers = {
            "authorization": self._expected_authorization,
            "content-type": request.headers.get(
                "content-type", "application/json"
            ),
        }
        response = await self._client.post(
            upstream_url,
            content=await request.body(),
            headers=headers,
        )
        response_headers = {}
        content_type = response.headers.get("content-type")
        if content_type:
            response_headers["content-type"] = content_type
        return Response(
            response.content,
            status_code=response.status_code,
            headers=response_headers,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


@dataclass(frozen=True, slots=True)
class ProxyEndpoint:
    base_url: str
    proxy: AuthenticationProxy


@asynccontextmanager
async def serve_authentication_proxy(
    config: LiveAgentConfig,
) -> AsyncIterator[ProxyEndpoint]:
    proxy = AuthenticationProxy(config.api_base, config.api_key)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.setblocking(False)
    port = int(listener.getsockname()[1])
    base_path = httpx.URL(config.api_base).path.rstrip("/")
    endpoint = ProxyEndpoint(f"http://127.0.0.1:{port}{base_path}", proxy)
    server = uvicorn.Server(
        uvicorn.Config(
            proxy.app,
            log_level="error",
            access_log=False,
            lifespan="off",
        )
    )
    server_task = asyncio.create_task(
        server.serve(sockets=[listener]), name="pygent-auth-proxy"
    )
    try:
        for _ in range(500):
            if server.started:
                break
            if server_task.done():
                await server_task
                raise RuntimeError("authentication proxy stopped during startup")
            await asyncio.sleep(0.01)
        else:
            raise TimeoutError("authentication proxy did not start")
        yield endpoint
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(server_task, timeout=5.0)
        finally:
            listener.close()
            await proxy.aclose()


async def run_fallback_probe(
    config: LiveAgentConfig,
    *,
    deadline_seconds: float,
) -> dict[str, object]:
    async with serve_authentication_proxy(config) as endpoint:
        proxy_config = LiveAgentConfig(
            api_base=endpoint.base_url,
            api_key=config.api_key,
            model_name=config.model_name,
        )
        report = await run_live_benchmark(
            proxy_config,
            requests=1,
            concurrency=1,
            model_concurrency=1,
            deadline_seconds=deadline_seconds,
        )
        metrics = report["metrics"]
        if not isinstance(metrics, dict):
            raise TypeError("benchmark returned invalid metrics")
        if metrics.get("succeeded") != 1:
            raise RuntimeError("fallback probe did not complete successfully")
        if metrics.get("fallback_count") != 1:
            raise RuntimeError("fallback probe did not observe fallback")
        failures = metrics.get("attempt_failures_by_kind")
        if not isinstance(failures, dict) or failures.get("authentication") != 2:
            raise RuntimeError("fallback probe did not observe authentication failures")
        if endpoint.proxy.rejected != 2 or endpoint.proxy.forwarded != 2:
            raise RuntimeError("authentication proxy observed an invalid call sequence")
        return {
            "configuration": report["configuration"],
            "metrics": metrics,
            "authentication_proxy": {
                "rejected": endpoint.proxy.rejected,
                "forwarded": endpoint.proxy.forwarded,
            },
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify invalid-key fallback against the configured real model."
    )
    parser.add_argument("--deadline-seconds", type=float, default=60.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = LiveAgentConfig.from_environment()
    report = asyncio.run(
        run_fallback_probe(config, deadline_seconds=args.deadline_seconds)
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
