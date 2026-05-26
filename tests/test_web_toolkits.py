from types import SimpleNamespace

from pygent.toolkits.web_fetch import WebFetchToolkits, _SimpleHTMLToMarkdown
from pygent.toolkits.web_search import WebSearchToolkits, _DuckDuckGoHTMLParser


class _FakeResponse:
    def __init__(self, body: bytes, content_type: str = "text/html; charset=utf-8"):
        self._body = body
        self.headers = {"Content-Type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._body


def test_simple_html_to_markdown_preserves_headings_links_and_lists():
    parser = _SimpleHTMLToMarkdown()
    parser.feed("<h1>Title</h1><p>Hello <strong>world</strong></p><ul><li><a href='https://e.test'>link</a></li></ul>")

    markdown = parser.get_markdown()

    assert "# Title" in markdown
    assert "Hello **world **" in markdown
    assert "[link ](https://e.test)" in markdown


def test_web_fetch_rejects_empty_non_http_and_private_urls():
    tools = WebFetchToolkits(session_id="s")

    assert "url" in tools.mcp_web_fetch("")
    assert "http/https" in tools.mcp_web_fetch("ftp://example.com")
    assert "127.0.0.1" not in tools.mcp_web_fetch("http://127.0.0.1:8000")
    assert "10.0.0.1" not in tools.mcp_web_fetch("http://10.0.0.1")


def test_web_fetch_converts_public_html(monkeypatch):
    def fake_urlopen(request, timeout):
        assert timeout == 15
        assert request.headers["User-agent"] == "pygent-web-fetch/1.0"
        return _FakeResponse("<html><body><h2>News</h2><p>Body</p></body></html>".encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    output = WebFetchToolkits(session_id="s").mcp_web_fetch("https://example.com/page")

    assert "## News" in output
    assert "Body" in output


def test_web_fetch_rejects_unsupported_content_type(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: _FakeResponse(b"\x00\x01", "application/octet-stream"),
    )

    assert "application/octet-stream" in WebFetchToolkits(session_id="s").mcp_web_fetch("https://example.com/file")


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


def test_web_search_formats_results_and_fallback(monkeypatch):
    monkeypatch.setattr(
        "pygent.toolkits.web_search._search_via_html",
        lambda query: [("Title", "https://example.com", "Snippet")],
    )
    output = WebSearchToolkits(session_id="s").web_search("pygent", "release check")
    assert "pygent" in output
    assert "Title" in output
    assert "https://example.com" in output

    monkeypatch.setattr("pygent.toolkits.web_search._search_via_html", lambda query: [])
    fallback = WebSearchToolkits(session_id="s").web_search("pygent ai", "")
    assert "duckduckgo.com" in fallback


def test_web_search_handles_empty_and_search_errors(monkeypatch):
    tools = WebSearchToolkits(session_id="s")
    assert "search_term" in tools.web_search("", "")

    def fail(_query):
        raise RuntimeError("network down")

    monkeypatch.setattr("pygent.toolkits.web_search._search_via_html", fail)
    assert "network down" in tools.web_search("pygent", "")
