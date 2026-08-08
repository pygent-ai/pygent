"""Deployment-local web search and fetch adapters expressed as 0.2 tools."""

from __future__ import annotations

import asyncio
import http.client
import ipaddress
import re
import socket
import ssl
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from html.parser import HTMLParser
from typing import Any, Self

from pygent.tool.executors import ToolExecutionError
from pygent.tool.functional import tool
from pygent.tool.types import ToolSideEffect

SearchResult = tuple[str, str, str]
Searcher = Callable[[str], list[SearchResult]]
Fetcher = Callable[[urllib.request.Request, float], Any]
Resolver = Callable[[str, int], Iterable[tuple[Any, Any, Any, Any, tuple[Any, ...]]]]

_MAX_FETCH_BYTES = 2 * 1024 * 1024
_MAX_REDIRECTS = 5
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}


class _SimpleHTMLToMarkdown(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._href_stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag in ("h1", "h2", "h3", "h4"):
            self._parts.append("\n" + "#" * int(tag[1]) + " ")
        elif tag == "p":
            self._parts.append("\n\n")
        elif tag == "br":
            self._parts.append("  \n")
        elif tag in ("ul", "ol"):
            self._parts.append("\n")
        elif tag == "li":
            self._parts.append("\n- ")
        elif tag == "a":
            self._href_stack.append(attrs_dict.get("href") or "")
            self._parts.append("[")
        elif tag in ("strong", "b"):
            self._parts.append("**")
        elif tag in ("em", "i"):
            self._parts.append("*")

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            href = self._href_stack.pop() if self._href_stack else ""
            self._parts.append("]")
            if href:
                self._parts.append(f"({href})")
        elif tag in ("strong", "b"):
            self._parts.append("**")
        elif tag in ("em", "i"):
            self._parts.append("*")

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self._parts.append(text + " ")

    def get_markdown(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", "".join(self._parts)).strip()


class _DuckDuckGoHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[SearchResult] = []
        self._in_result_a = False
        self._in_result_snippet = False
        self._current_href = ""
        self._current_title = ""
        self._current_snippet = ""
        self._max_results = 8

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        css_class = attrs_dict.get("class") or ""
        href = attrs_dict.get("href") or ""
        if tag != "a":
            return
        if "result__a" in css_class and href and "uddg=" in href:
            self._in_result_a = True
            self._current_href = href
            self._current_title = ""
        elif "result__snippet" in css_class:
            self._in_result_snippet = True
            self._current_snippet = ""

    def handle_endtag(self, tag: str) -> None:
        if tag != "a":
            return
        if self._in_result_a:
            self._in_result_a = False
            if (
                self._current_title
                and self._current_href
                and "uddg=" in self._current_href
                and "y.js" not in self._current_href
                and "ad_domain" not in self._current_href
                and len(self.results) < self._max_results
            ):
                real_url = self._extract_real_url(self._current_href)
                if real_url.startswith("http"):
                    self.results.append(
                        (
                            self._current_title.strip(),
                            real_url,
                            self._current_snippet.strip(),
                        )
                    )
            self._current_href = ""
        elif self._in_result_snippet:
            self._in_result_snippet = False

    def handle_data(self, data: str) -> None:
        if self._in_result_a:
            self._current_title += data
        elif self._in_result_snippet:
            self._current_snippet += data

    @staticmethod
    def _extract_real_url(href: str) -> str:
        match = re.search(r"uddg=([^&]+)", href)
        return urllib.parse.unquote(match.group(1)) if match else href


def _search_via_html(query: str) -> list[SearchResult]:
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        html = response.read(_MAX_FETCH_BYTES + 1)
    if len(html) > _MAX_FETCH_BYTES:
        raise ValueError("search response exceeds the configured size limit")
    parser = _DuckDuckGoHTMLParser()
    parser.feed(html.decode("utf-8", errors="replace"))
    return parser.results


def _address_is_private(value: str) -> bool:
    address = ipaddress.ip_address(value)
    return not address.is_global


def _resolve_public_url(
    url: str, resolver: Resolver
) -> tuple[urllib.parse.ParseResult, tuple[str, ...]]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ToolExecutionError(
            "web_fetch supports only http and https URLs",
            kind="validation_error",
            code="unsupported_scheme",
            side_effect_committed=False,
        )
    if (
        not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ToolExecutionError(
            "web_fetch requires a public URL without embedded credentials",
            kind="validation_error",
            code="invalid_url",
            side_effect_committed=False,
        )
    host = parsed.hostname
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise ToolExecutionError(
            "web_fetch URL contains an invalid port",
            kind="validation_error",
            code="invalid_url",
            side_effect_committed=False,
        ) from exc
    try:
        addresses = [str(ipaddress.ip_address(host))]
    except ValueError:
        try:
            addresses = [str(item[4][0]) for item in resolver(host, port)]
        except OSError as exc:
            raise ToolExecutionError(
                "web_fetch could not resolve the target host",
                kind="transport_error",
                code="dns_failed",
                retryable=True,
                side_effect_committed=False,
            ) from exc
    if not addresses or any(_address_is_private(item) for item in addresses):
        raise ToolExecutionError(
            "web_fetch refuses local, private, reserved, or otherwise non-public addresses",
            kind="authorization_error",
            code="private_address",
            side_effect_committed=False,
        )
    return parsed, tuple(dict.fromkeys(addresses))


def _validate_public_url(url: str, resolver: Resolver) -> urllib.parse.ParseResult:
    parsed, _addresses = _resolve_public_url(url, resolver)
    return parsed


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, host: str, port: int, address: str, timeout: float) -> None:
        super().__init__(host, port, timeout=timeout)
        self._validated_address = address

    def connect(self) -> None:
        self.sock = socket.create_connection(
            (self._validated_address, self.port),
            self.timeout,
        )


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host: str, port: int, address: str, timeout: float) -> None:
        self._ssl_context = ssl.create_default_context()
        super().__init__(host, port, timeout=timeout, context=self._ssl_context)
        self._validated_address = address

    def connect(self) -> None:
        self.sock = socket.create_connection(
            (self._validated_address, self.port),
            self.timeout,
        )
        self.sock = self._ssl_context.wrap_socket(self.sock, server_hostname=self.host)


class _PinnedResponse:
    def __init__(
        self,
        response: http.client.HTTPResponse,
        connection: http.client.HTTPConnection,
    ) -> None:
        self._response = response
        self._connection = connection
        self.headers = response.headers

    def read(self, amount: int = -1) -> bytes:
        return self._response.read(amount)

    def close(self) -> None:
        try:
            self._response.close()
        finally:
            self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class WebSearchTools:
    """Deployment-local search adapter; only its ToolSpec is portable."""

    def __init__(self, *, searcher: Searcher | None = None) -> None:
        self._searcher = searcher or _search_via_html

    @tool(
        tool_id="standard.web.search",
        version="2.0.0",
        side_effect=ToolSideEffect.READ,
        timeout=20,
        resource_key="web",
        required_permissions=("web:search",),
    )
    async def web_search(self, search_term: str, description: str = "") -> str:
        """Search public web results and return titles, URLs, and short snippets.

        Args:
            search_term: Search keywords or a complete query.
            description: Human-readable reason for the search; not sent as a query.
        """

        query = (search_term or "").strip()
        if not query:
            raise ToolExecutionError(
                "search_term must not be empty",
                kind="validation_error",
                code="invalid_query",
                side_effect_committed=False,
            )
        try:
            results = await asyncio.to_thread(self._searcher, query)
        except Exception as exc:
            raise ToolExecutionError(
                "public web search failed",
                kind="transport_error",
                code="search_failed",
                retryable=True,
                side_effect_committed=False,
            ) from exc
        if not results:
            return "No structured results were found. Search URL:\n" + (
                "https://duckduckgo.com/?" + urllib.parse.urlencode({"q": query})
            )
        lines = [
            f"Search: {query}",
            f"Description: {(description or '').strip()}",
            "",
            f"Top {len(results)} results:",
        ]
        for index, (title, url, snippet) in enumerate(results, 1):
            lines.extend((f"\n{index}. {title}", f"   {url}"))
            if snippet:
                lines.append(f"   {snippet[:200]}{'...' if len(snippet) > 200 else ''}")
        return "\n".join(lines).strip()


class WebFetchTools:
    """Deployment-local public HTTP fetch adapter with SSRF checks."""

    def __init__(
        self,
        *,
        fetcher: Fetcher | None = None,
        resolver: Resolver | None = None,
    ) -> None:
        self._resolver = resolver or socket.getaddrinfo
        self._fetcher = fetcher or self._default_fetch

    def _default_fetch(self, request: urllib.request.Request, timeout: float):
        current = request
        for redirect_count in range(_MAX_REDIRECTS + 1):
            url = current.full_url
            parsed, addresses = _resolve_public_url(url, self._resolver)
            host = parsed.hostname
            if host is None:  # pragma: no cover - guarded by _resolve_public_url
                raise ToolExecutionError(
                    "web_fetch URL has no host",
                    kind="validation_error",
                    code="invalid_url",
                    side_effect_committed=False,
                )
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            default_port = 443 if parsed.scheme == "https" else 80
            host_value = f"[{host}]" if ":" in host else host
            if port != default_port:
                host_value = f"{host_value}:{port}"
            target = parsed.path or "/"
            if parsed.query:
                target += "?" + parsed.query
            headers = dict(current.header_items())
            headers["Host"] = host_value
            response_owner: _PinnedResponse | None = None
            last_error: OSError | None = None
            for address in addresses:
                connection_type = (
                    _PinnedHTTPSConnection
                    if parsed.scheme == "https"
                    else _PinnedHTTPConnection
                )
                connection = connection_type(host, port, address, timeout)
                try:
                    connection.request(
                        current.get_method(),
                        target,
                        body=current.data,
                        headers=headers,
                    )
                    response_owner = _PinnedResponse(
                        connection.getresponse(), connection
                    )
                    break
                except OSError as exc:
                    last_error = exc
                    connection.close()
            if response_owner is None:
                raise last_error or OSError("no validated address was connectable")

            status = response_owner._response.status
            location = response_owner.headers.get("Location")
            if status not in _REDIRECT_STATUSES or not location:
                if status >= 400:
                    response_owner.close()
                    raise OSError(f"HTTP request failed with status {status}")
                return response_owner

            response_owner.close()
            if redirect_count >= _MAX_REDIRECTS:
                raise ToolExecutionError(
                    "web_fetch exceeded the redirect limit",
                    kind="transport_error",
                    code="too_many_redirects",
                    retryable=False,
                    side_effect_committed=False,
                )
            next_url = urllib.parse.urljoin(url, location)
            method = current.get_method()
            data = current.data
            if status == 303 or (status in {301, 302} and method == "POST"):
                method = "GET"
                data = None
            current = urllib.request.Request(
                next_url,
                data=data,
                headers=dict(current.header_items()),
                method=method,
            )
        raise AssertionError("redirect loop must terminate")

    @tool(
        tool_id="standard.web.fetch",
        version="2.0.0",
        side_effect=ToolSideEffect.READ,
        timeout=20,
        resource_key="web",
        required_permissions=("web:fetch",),
    )
    async def web_fetch(self, url: str) -> str:
        """Fetch a public HTTP(S) page and convert basic HTML structure to Markdown.

        Args:
            url: Complete public HTTP or HTTPS URL without embedded credentials.
        """

        value = (url or "").strip()
        if not value:
            raise ToolExecutionError(
                "url must not be empty",
                kind="validation_error",
                code="invalid_url",
                side_effect_committed=False,
            )
        return await asyncio.to_thread(self._fetch_sync, value)

    def _fetch_sync(self, url: str) -> str:
        _validate_public_url(url, self._resolver)
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "pygent-standard-web-fetch/2.0",
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9",
            },
        )
        try:
            with self._fetcher(request, 15) as response:
                content_type = response.headers.get("Content-Type", "")
                raw = response.read(_MAX_FETCH_BYTES + 1)
        except ToolExecutionError:
            raise
        except Exception as exc:
            raise ToolExecutionError(
                "public web fetch failed",
                kind="transport_error",
                code="fetch_failed",
                retryable=True,
                side_effect_committed=False,
            ) from exc
        if len(raw) > _MAX_FETCH_BYTES:
            raise ToolExecutionError(
                "web response exceeds the configured size limit",
                kind="validation_error",
                code="response_too_large",
                side_effect_committed=False,
            )
        if not content_type.startswith("text/") and "html" not in content_type:
            raise ToolExecutionError(
                f"unsupported content type: {content_type or 'unknown'}",
                kind="validation_error",
                code="unsupported_content_type",
                side_effect_committed=False,
            )
        charset_match = re.search(r"charset=([^\s;]+)", content_type, re.IGNORECASE)
        charset = charset_match.group(1).strip() if charset_match else "utf-8"
        try:
            text = raw.decode(charset, errors="replace")
        except LookupError:
            text = raw.decode("utf-8", errors="replace")
        if "html" not in content_type:
            return text
        parser = _SimpleHTMLToMarkdown()
        parser.feed(text)
        markdown = parser.get_markdown()
        return markdown or "The page was fetched but contained no readable text."


__all__ = ["WebFetchTools", "WebSearchTools"]
