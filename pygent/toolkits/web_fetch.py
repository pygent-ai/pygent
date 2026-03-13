from __future__ import annotations

import ipaddress
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Optional

from pygent.common import PygentString
from pygent.module.tool.utils import ToolClassBase, tool_method, tool_class


class _SimpleHTMLToMarkdown(HTMLParser):
    """极简 HTML -> Markdown 转换器，仅保留基础结构信息。"""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._href_stack: list[str] = []

    def handle_starttag(self, tag: str, attrs):
        attrs_dict = dict(attrs)
        if tag in ("h1", "h2", "h3", "h4"):
            level = int(tag[1])
            self._parts.append("\n" + "#" * level + " ")
        elif tag == "p":
            self._parts.append("\n\n")
        elif tag == "br":
            self._parts.append("  \n")
        elif tag in ("ul", "ol"):
            self._parts.append("\n")
        elif tag == "li":
            self._parts.append("\n- ")
        elif tag == "a":
            href = attrs_dict.get("href", "")
            self._href_stack.append(href)
            self._parts.append("[")
        elif tag in ("strong", "b"):
            self._parts.append("**")
        elif tag in ("em", "i"):
            self._parts.append("*")

    def handle_endtag(self, tag: str):
        if tag == "a":
            href = self._href_stack.pop() if self._href_stack else ""
            self._parts.append("]")
            if href:
                self._parts.append(f"({href})")
        elif tag in ("strong", "b"):
            self._parts.append("**")
        elif tag in ("em", "i"):
            self._parts.append("*")

    def handle_data(self, data: str):
        text = data.strip()
        if text:
            # 在末尾加空格避免单词黏连
            self._parts.append(text + " ")

    def get_markdown(self) -> str:
        out = "".join(self._parts)
        # 压缩多余空行
        out = re.sub(r"\n{3,}", "\n\n", out)
        return out.strip()


@tool_class(description="网页抓取工具集：根据 URL 获取公开网页并转换为 Markdown。")
class WebFetchToolkits(ToolClassBase):
    """网页抓取工具集：根据 URL 获取公开网页并转换为 Markdown。"""

    def __init__(self, session_id: str, workspace_root: Optional[str] = None):
        self.session_id = PygentString(session_id)
        self.workspace_root = PygentString(workspace_root) or PygentString("")

    @tool_method(
        name="mcp_web_fetch",
        description="根据 URL 抓取网页内容并转换为可读的 Markdown；不支持需认证或本地/私有地址。",
    )
    def mcp_web_fetch(self, url: str) -> str:
        """
        抓取网页并转为 Markdown 文本。

        Args:
            url: 要抓取的完整有效 URL，仅支持 http/https 公网地址。
        """
        url = (url or "").strip()
        if not url:
            return "错误：url 不能为空"

        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return "错误：仅支持 http/https URL"

        host = parsed.hostname or ""

        # 阻止明显的本地/私有地址，避免安全风险
        try:
            ip = (
                ipaddress.ip_address(host)
                if re.match(r"^\d+\.\d+\.\d+\.\d+$", host)
                else None
            )
        except ValueError:
            ip = None

        if host in {"localhost", "127.0.0.1"} or (
            ip and (ip.is_private or ip.is_loopback or ip.is_link_local)
        ):
            return "错误：出于安全原因，不支持抓取本地或私有地址"

        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "pygent-web-fetch/1.0",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                },
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                content_type = resp.headers.get("Content-Type", "")
                raw = resp.read()

            if not content_type.startswith("text/") and "html" not in content_type:
                ct = content_type or "未知"
                return f"错误：不支持的内容类型 {ct}"

            # 尝试根据响应头中的 charset 解码
            charset = "utf-8"
            m = re.search(r"charset=([^\s;]+)", content_type, re.IGNORECASE)
            if m:
                charset = m.group(1).strip()
            try:
                text = raw.decode(charset, errors="replace")
            except LookupError:
                text = raw.decode("utf-8", errors="replace")

            parser = _SimpleHTMLToMarkdown()
            parser.feed(text)
            md = parser.get_markdown()
            if not md:
                return "警告：成功抓取网页，但未能提取到可读文本内容。"
            return md
        except Exception as e:
            return f"错误：抓取网页失败（{e}）"

