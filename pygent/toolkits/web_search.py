from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Optional, List

from pygent.common import PygentString
from pygent.module.tool.utils import ToolClassBase, tool_method, tool_class


@tool_class(description="网络搜索工具集：通过公共搜索引擎检索实时信息。")
class WebSearchToolkits(ToolClassBase):
    """网络搜索工具集：通过公共搜索引擎检索实时信息。"""

    def __init__(self, session_id: str, workspace_root: Optional[str] = None):
        self.session_id = PygentString(session_id)
        # workspace_root 当前未使用，但保持与其它 Toolkits 一致的签名，便于扩展与注入
        self.workspace_root = PygentString(workspace_root) or PygentString("")

    @tool_method(
        name="web_search",
        description="在网络上搜索实时信息，用于获取最新资讯或验证事实。",
    )
    def web_search(self, search_term: str, explanation: str) -> str:
        """
        使用公共搜索引擎查询给定关键词，并返回若干条结果摘要。

        Args:
            search_term: 搜索关键词或完整查询。
            explanation: 使用该搜索的一句说明（仅用于可读性，不参与搜索逻辑）。
        """
        query = (search_term or "").strip()
        if not query:
            return "错误：search_term 不能为空"

        try:
            # 使用 DuckDuckGo 的 Instant Answer JSON 接口获取基础搜索结果
            params = urllib.parse.urlencode(
                {
                    "q": query,
                    "format": "json",
                    "no_redirect": "1",
                    "no_html": "1",
                }
            )
            url = f"https://duckduckgo.com/?{params}"
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "pygent-web-search/1.0",
                    "Accept": "application/json,text/plain;q=0.8,*/*;q=0.5",
                },
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()

            text = data.decode("utf-8", errors="replace")
            results: List[str] = []

            try:
                payload = json.loads(text)
                related = payload.get("RelatedTopics") or []
                for item in related:
                    # RelatedTopics 可能是扁平项，也可能嵌套 Topics
                    if "Text" in item and "FirstURL" in item:
                        title = item.get("Text") or ""
                        url_item = item.get("FirstURL") or ""
                        if title or url_item:
                            results.append(f"- {title}\n  {url_item}")
                    elif "Topics" in item:
                        for sub in item.get("Topics") or []:
                            title = sub.get("Text") or ""
                            url_item = sub.get("FirstURL") or ""
                            if title or url_item:
                                results.append(f"- {title}\n  {url_item}")
                    if len(results) >= 5:
                        break
            except json.JSONDecodeError:
                results = []

            if not results:
                # 若无法解析结构化结果，则返回可点击的搜索 URL 供人工查看
                fallback_url = "https://duckduckgo.com/?" + urllib.parse.urlencode({"q": query})
                return (
                    "未能解析结构化搜索结果。你可以手动访问以下搜索链接：\n"
                    f"{fallback_url}"
                )

            header = [
                f"搜索词：{query}",
                f"说明：{(explanation or '').strip()}",
                "",
                f"前 {len(results)} 条结果：",
            ]
            return "\n".join(header + results)
        except Exception as e:
            return f"错误：网络搜索失败（{e}）"

