from __future__ import annotations

import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Optional, List, Tuple

from pygent.common import PygentString
from pygent.module.tool.utils import ToolClassBase, tool_method, tool_class


class _DuckDuckGoHTMLParser(HTMLParser):
    """解析 DuckDuckGo HTML 搜索结果页，提取标题、链接和摘要。"""

    def __init__(self) -> None:
        super().__init__()
        self.results: List[Tuple[str, str, str]] = []  # (title, url, snippet)
        self._in_result_a = False
        self._in_result_snippet = False
        self._current_href = ""
        self._current_title = ""
        self._current_snippet = ""
        self._max_results = 8

    def handle_starttag(self, tag: str, attrs: list) -> None:
        attrs_d = dict(attrs)
        cls = attrs_d.get("class", "")
        href = attrs_d.get("href", "")
        if tag == "a":
            if "result__a" in cls and href and "uddg=" in href:
                self._in_result_a = True
                self._current_href = href
                self._current_title = ""
            elif "result__snippet" in cls:
                self._in_result_snippet = True
                self._current_snippet = ""

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            if self._in_result_a:
                self._in_result_a = False
                # 过滤广告链接（y.js、ad_domain、ad_provider 等）
                if (
                    self._current_title
                    and self._current_href
                    and "uddg=" in self._current_href
                    and "y.js" not in self._current_href
                    and "ad_domain" not in self._current_href
                    and len(self.results) < self._max_results
                ):
                    real_url = self._extract_real_url(self._current_href)
                    if real_url and real_url.startswith("http"):
                        self.results.append((self._current_title.strip(), real_url, self._current_snippet.strip()))
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
        """从 DuckDuckGo 重定向链接中提取真实 URL。"""
        if "uddg=" in href:
            match = re.search(r"uddg=([^&]+)", href)
            if match:
                return urllib.parse.unquote(match.group(1))
        return href


def _search_via_html(query: str) -> List[Tuple[str, str, str]]:
    """通过 DuckDuckGo HTML 接口搜索，返回 [(title, url, snippet), ...]"""
    params = urllib.parse.urlencode({"q": query})
    url = f"https://html.duckduckgo.com/html/?{params}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode("utf-8", errors="replace")

    parser = _DuckDuckGoHTMLParser()
    parser.feed(html)
    return parser.results


@tool_class(description="网络搜索工具集：通过公共搜索引擎检索实时信息。")
class WebSearchToolkits(ToolClassBase):
    """网络搜索工具集：通过公共搜索引擎检索实时信息。"""

    def __init__(self, session_id: str, workspace_root: Optional[str] = None):
        self.session_id = PygentString(session_id)
        self.workspace_root = PygentString(workspace_root) or PygentString("")

    @tool_method(
        name="web_search",
        description="在网络上搜索实时信息，用于获取最新资讯或验证事实。",
    )
    def web_search(self, search_term: str, explanation: str) -> str:
        """
        使用公共搜索引擎查询给定关键词，并返回若干条结果摘要。
        仅使用 Python 标准库，不依赖第三方包。

        Args:
            search_term: 搜索关键词或完整查询。
            explanation: 使用该搜索的一句说明（仅用于可读性，不参与搜索逻辑）。
        """
        query = (search_term or "").strip()
        if not query:
            return "错误：search_term 不能为空"

        try:
            results = _search_via_html(query)
            if not results:
                fallback_url = "https://duckduckgo.com/?" + urllib.parse.urlencode({"q": query})
                return (
                    "未能解析结构化搜索结果。你可以手动访问以下搜索链接：\n"
                    f"{fallback_url}"
                )

            lines: List[str] = [
                f"搜索词：{query}",
                f"说明：{(explanation or '').strip()}",
                "",
                f"前 {len(results)} 条结果：",
            ]
            for i, (title, url, snippet) in enumerate(results, 1):
                lines.append(f"\n{i}. {title}")
                lines.append(f"   {url}")
                if snippet:
                    snip = snippet[:200] + "..." if len(snippet) > 200 else snippet
                    lines.append(f"   {snip}")
            return "\n".join(lines).strip()
        except Exception as e:
            return f"错误：网络搜索失败（{e}）"
