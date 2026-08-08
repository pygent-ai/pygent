from __future__ import annotations

import http.server
import socket
import threading
from types import SimpleNamespace

import pytest

from pygent import IdempotencyPolicy, ToolKit, ToolSideEffect
from pygent.tool.standard._web import (
    WebFetchTools,
    WebSearchTools,
    _DuckDuckGoHTMLParser,
    _SimpleHTMLToMarkdown,
)

from ._helpers import invoke_tool, succeeded


class _FakeResponse:
    def __init__(self, body: bytes, content_type: str = "text/html; charset=utf-8"):
        self._body = body
        self.headers = {"Content-Type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, limit: int = -1):
        if limit < 0:
            return self._body
        return self._body[:limit]


def _public_resolver(_host: str, _port: int):
    return [(None, None, None, None, ("93.184.216.34", 0))]


def test_simple_html_to_markdown_preserves_headings_links_and_lists():
    parser = _SimpleHTMLToMarkdown()
    parser.feed(
        "<h1>Title</h1><p>Hello <strong>world</strong></p>"
        "<ul><li><a href='https://e.test'>link</a></li></ul>"
    )

    markdown = parser.get_markdown()

    assert "# Title" in markdown
    assert "Hello **world **" in markdown
    assert "[link ](https://e.test)" in markdown


@pytest.mark.asyncio
async def test_web_fetch_rejects_empty_non_http_and_private_urls():
    tools = WebFetchTools(resolver=_public_resolver)

    empty = await invoke_tool(tools.web_fetch, {"url": ""})
    non_http = await invoke_tool(tools.web_fetch, {"url": "ftp://example.com"})
    loopback = await invoke_tool(tools.web_fetch, {"url": "http://127.0.0.1:8000"})
    private = await invoke_tool(tools.web_fetch, {"url": "http://10.0.0.1"})

    assert empty.error_code == "invalid_url"
    assert non_http.error_code == "unsupported_scheme"
    assert loopback.error_code == "private_address"
    assert private.error_code == "private_address"
    assert all(
        item.side_effect_committed is False
        for item in (empty, non_http, loopback, private)
    )


@pytest.mark.asyncio
async def test_web_fetch_rejects_private_dns_resolution():
    tools = WebFetchTools(
        resolver=lambda _host, _port: [(None, None, None, None, ("192.168.1.10", 0))]
    )

    result = await invoke_tool(tools.web_fetch, {"url": "https://internal.example"})

    assert result.status == "failed"
    assert result.error_code == "private_address"


@pytest.mark.asyncio
async def test_web_fetch_connects_to_the_validated_address(monkeypatch):
    validated_ip = "93.184.216.34"
    attempted_addresses = []

    def fake_create_connection(address, *args, **kwargs):
        attempted_addresses.append(address)
        raise OSError("connection intentionally stopped")

    monkeypatch.setattr(socket, "create_connection", fake_create_connection)
    tools = WebFetchTools(
        resolver=lambda _host, port: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (validated_ip, port))
        ]
    )

    result = await invoke_tool(
        tools.web_fetch, {"url": "http://rebinding.example:8765/secret"}
    )

    assert result.status == "failed"
    assert result.error_code == "fetch_failed"
    assert attempted_addresses
    assert attempted_addresses[0][0] == validated_ip


@pytest.mark.asyncio
async def test_default_web_fetch_reads_through_the_validated_address(monkeypatch):
    class PageHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = b"<html><body><h1>Pinned</h1><p>safe page</p></body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), PageHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    original_create_connection = socket.create_connection

    def route_validated_connection(address, *args, **kwargs):
        assert address[0] == "93.184.216.34"
        return original_create_connection(
            ("127.0.0.1", server.server_address[1]), *args, **kwargs
        )

    monkeypatch.setattr(socket, "create_connection", route_validated_connection)
    tools = WebFetchTools(
        resolver=lambda _host, port: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))
        ]
    )
    try:
        result = await invoke_tool(
            tools.web_fetch,
            {"url": f"http://pinned.example:{server.server_address[1]}/page"},
        )
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join()

    assert result.status == "succeeded"
    assert "# Pinned" in (result.output or "")
    assert "safe page" in (result.output or "")


@pytest.mark.asyncio
async def test_web_fetch_revalidates_redirect_before_connecting(monkeypatch):
    class RedirectHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(302)
            self.send_header(
                "Location",
                f"http://127.0.0.1:{self.server.server_address[1]}/secret",
            )
            self.end_headers()

        def log_message(self, *_args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    original_create_connection = socket.create_connection
    attempted_addresses = []

    def route_validated_connection(address, *args, **kwargs):
        attempted_addresses.append(address)
        assert address[0] == "93.184.216.34"
        return original_create_connection(
            ("127.0.0.1", server.server_address[1]), *args, **kwargs
        )

    monkeypatch.setattr(socket, "create_connection", route_validated_connection)
    tools = WebFetchTools(
        resolver=lambda _host, port: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))
        ]
    )
    try:
        result = await invoke_tool(
            tools.web_fetch,
            {"url": f"http://redirect.example:{server.server_address[1]}/start"},
        )
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join()

    assert result.status == "failed"
    assert result.error_code == "private_address"
    assert len(attempted_addresses) == 1


@pytest.mark.asyncio
async def test_web_fetch_converts_public_html():
    def fake_fetch(request, timeout):
        assert timeout == 15
        assert request.headers["User-agent"] == "pygent-standard-web-fetch/2.0"
        return _FakeResponse(b"<html><body><h2>News</h2><p>Body</p></body></html>")

    tools = WebFetchTools(fetcher=fake_fetch, resolver=_public_resolver)
    output = await succeeded(tools.web_fetch, url="https://example.com/page")

    assert "## News" in output
    assert "Body" in output


@pytest.mark.asyncio
async def test_web_fetch_rejects_unsupported_content_type():
    tools = WebFetchTools(
        fetcher=lambda request, timeout: _FakeResponse(
            b"\x00\x01", "application/octet-stream"
        ),
        resolver=_public_resolver,
    )

    result = await invoke_tool(tools.web_fetch, {"url": "https://example.com/file"})

    assert result.status == "failed"
    assert result.error_code == "unsupported_content_type"
    assert "application/octet-stream" in (result.error or "")


def test_duckduckgo_parser_extracts_real_result_urls_and_skips_ads():
    parser = _DuckDuckGoHTMLParser()
    parser.feed(
        """
        <a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Fone">First</a>
        <a class="result__snippet">First snippet</a>
        <a class="result__a" href="/y.js?ad_domain=ads.test&uddg=https%3A%2F%2Fad.test">Ad</a>
        """
    )

    assert parser.results == [("First", "https://example.com/one", "")]


@pytest.mark.asyncio
async def test_web_search_formats_results_and_fallback():
    tools = WebSearchTools(
        searcher=lambda query: [("Title", "https://example.com", "Snippet")]
    )
    output = await succeeded(
        tools.web_search, search_term="pygent", description="release check"
    )
    assert "pygent" in output
    assert "Title" in output
    assert "https://example.com" in output

    fallback_tools = WebSearchTools(searcher=lambda query: [])
    fallback = await succeeded(
        fallback_tools.web_search, search_term="pygent ai", description=""
    )
    assert "duckduckgo.com" in fallback


@pytest.mark.asyncio
async def test_web_search_handles_empty_and_search_errors():
    empty = await invoke_tool(
        WebSearchTools(searcher=lambda query: []).web_search,
        {"search_term": "", "description": ""},
    )

    def fail(_query):
        raise RuntimeError("network down")

    failed = await invoke_tool(
        WebSearchTools(searcher=fail).web_search,
        {"search_term": "pygent", "description": ""},
    )

    assert empty.error_code == "invalid_query"
    assert failed.error_code == "search_failed"
    assert "network down" not in (failed.error or "")


def test_web_tools_publish_portable_02_specs_without_clients():
    search = ToolKit(WebSearchTools(searcher=lambda query: []).web_search).specs[0]
    fetch = ToolKit(
        WebFetchTools(
            fetcher=lambda request, timeout: SimpleNamespace(),
            resolver=_public_resolver,
        ).web_fetch
    ).specs[0]

    assert (search.tool_id, search.version) == ("standard.web.search", "2.0.0")
    assert (fetch.tool_id, fetch.version) == ("standard.web.fetch", "2.0.0")
    assert search.side_effect is ToolSideEffect.READ
    assert fetch.side_effect is ToolSideEffect.READ
    assert search.idempotency is IdempotencyPolicy.INHERENT
    assert fetch.idempotency is IdempotencyPolicy.INHERENT
    assert search.required_permissions == ("web:search",)
    assert fetch.required_permissions == ("web:fetch",)
    assert "searcher" not in search.definition.parameters
    assert "fetcher" not in fetch.definition.parameters
