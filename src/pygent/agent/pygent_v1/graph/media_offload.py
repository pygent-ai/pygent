"""Offload large generated images out of persisted conversation checkpoints.

Generated images reach the conversation as multi-megabyte payloads: markdown
``data:image/...;base64,`` URIs inside assistant text, inline media blocks in
tool results, and markdown links to remote hosts. Storing them verbatim makes
every session open transfer and decode the full bytes again, and remote links
keep the image hostage to the host staying reachable and CORS-open. The
checkpoint boundary rewrites all three shapes into workspace files under
``.pygent_v1/uploads`` and keeps only resource references, which the uploads
endpoint and the pygent workspace media resolver both resolve back to bytes.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import copy
import hashlib
import os
import re
import urllib.request
import uuid
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any

from ..runtime.attachments import MEDIA_MAX_BYTES

UPLOADS_SUBDIR = Path(".pygent_v1") / "uploads"

# Only rendered markdown images are rewritten; a bare data URI inside prose
# or a code fence never becomes an <img> node on the client.
_MARKDOWN_DATA_IMAGE = re.compile(
    r"(!\[[^\]]*\]\()data:image/([a-z0-9.+-]+);base64,([A-Za-z0-9+/=]+)(\))"
)

# Same rule for remote hosts: only markdown image targets are fetched, so
# plain links in prose or code fences survive untouched.
_MARKDOWN_REMOTE_IMAGE = re.compile(r"(!\[[^\]]*\]\()(https?://[^\s)]+)(\))")

# Tool-result and message media blocks round-trip through strict field
# validation, so the replacement carries the full canonical source shape.
_RESOURCE_SOURCE_KEYS = ("url", "base64_data")

# Thumbnail-scale images stay inline: reloading them costs nothing and
# offloading would litter the uploads directory.
_INLINE_OFFLOAD_MIN_BYTES = 64 * 1024

_REMOTE_FETCH_TIMEOUT_SECONDS = 30.0

_MIME_EXTENSIONS = {
    "png": ".png",
    "jpeg": ".jpg",
    "jpg": ".jpg",
    "gif": ".gif",
    "webp": ".webp",
    "bmp": ".bmp",
    "avif": ".avif",
    "svg+xml": ".svg",
}


def offload_inline_media(
    payload: dict[str, Any], *, workspace_root: str | Path
) -> dict[str, Any]:
    """Return `payload` with large inline images replaced by file references.

    The input is never mutated. Any storage failure returns the original
    payload, so history persistence degrades to inline bytes instead of
    failing the turn that produced the image.
    """

    if not str(workspace_root or "").strip():
        return payload
    root = Path(workspace_root)
    try:
        return _walk_media(
            copy.deepcopy(payload),
            lambda node: _offload_inline_node(node, root),
        )
    except OSError:
        return payload


# Downloads one image URL and returns its bytes; transport errors surface as
# exceptions and are converted to "keep the remote link" by the caller.
RemoteMediaFetcher = Callable[[str], Awaitable[bytes]]


async def offload_remote_media(
    payload: dict[str, Any],
    *,
    workspace_root: str | Path,
    fetch_url: RemoteMediaFetcher | None,
) -> dict[str, Any]:
    """Return `payload` with remote image URLs replaced by workspace files.

    Markdown image links and ``kind: "url"`` media blocks pointing at
    ``http(s)`` hosts are downloaded once and rewritten to the same stored
    ``.pygent_v1/uploads`` references the inline path produces, so copies and
    replays stop depending on the remote host. Fetch failures, sub-thumbnail
    images, and oversized payloads keep their remote link: a dead host
    degrades to the previous behavior instead of failing the turn. Storage
    failures return the original payload, matching `offload_inline_media`.
    Without a `fetch_url` the payload passes through unchanged.
    """

    if fetch_url is None or not str(workspace_root or "").strip():
        return payload
    root = Path(workspace_root)
    fetcher = fetch_url

    def handle(node: dict[str, Any]) -> Awaitable[None]:
        return _offload_remote_node(node, root, fetcher, {})

    try:
        return await _walk_media_async(copy.deepcopy(payload), handle)
    except OSError:
        return payload


def _iter_media_nodes(node: Any) -> Iterator[dict[str, Any]]:
    """Yield every dict node of a media tree in document order."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _iter_media_nodes(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_media_nodes(item)


def _walk_media(
    node: Any, handler: Callable[[dict[str, Any]], None]
) -> Any:
    for item in _iter_media_nodes(node):
        handler(item)
    return node


async def _walk_media_async(
    node: Any, handler: Callable[[dict[str, Any]], Awaitable[None]]
) -> Any:
    for item in _iter_media_nodes(node):
        await handler(item)
    return node


def _offload_inline_node(node: dict[str, Any], workspace_root: Path) -> None:
    source = node.get("source")
    if (
        node.get("type") == "media"
        and isinstance(source, dict)
        and source.get("kind") == "inline"
    ):
        replacement = _resource_source(
            source.get("base64_data"),
            node.get("mime_type"),
            workspace_root,
        )
        if replacement is not None:
            node["source"] = replacement
    content = node.get("content")
    if isinstance(content, str):
        node["content"] = _offload_content_data_uris(content, workspace_root)


async def _offload_remote_node(
    node: dict[str, Any],
    workspace_root: Path,
    fetch_url: RemoteMediaFetcher,
    fetched: dict[str, bytes | None],
) -> None:
    source = node.get("source")
    if (
        node.get("type") == "media"
        and isinstance(source, dict)
        and source.get("kind") == "url"
    ):
        replacement = await _remote_resource_source(
            source.get("url"),
            node.get("mime_type"),
            workspace_root,
            fetch_url,
            fetched,
        )
        if replacement is not None:
            node["source"] = replacement
    content = node.get("content")
    if isinstance(content, str):
        node["content"] = await _offload_remote_content(
            content, workspace_root, fetch_url, fetched
        )


def _offload_content_data_uris(content: str, workspace_root: Path) -> str:
    def replace(match: re.Match[str]) -> str:
        data = _decode_inline(match.group(3))
        if data is None or len(data) < _INLINE_OFFLOAD_MIN_BYTES:
            return match.group(0)
        relative = _store_media_bytes(
            data,
            workspace_root,
            _MIME_EXTENSIONS.get(match.group(2).lower(), ".png"),
        )
        return f"{match.group(1)}{relative}{match.group(4)}"

    return _MARKDOWN_DATA_IMAGE.sub(replace, content)


async def _offload_remote_content(
    content: str,
    workspace_root: Path,
    fetch_url: RemoteMediaFetcher,
    fetched: dict[str, bytes | None],
) -> str:
    async def replace(match: re.Match[str]) -> str:
        source = await _remote_resource_source(
            match.group(2), None, workspace_root, fetch_url, fetched
        )
        if source is None:
            return match.group(0)
        return f"{match.group(1)}{source['uri']}{match.group(3)}"

    return await _sub_async(_MARKDOWN_REMOTE_IMAGE, replace, content)


async def _sub_async(
    pattern: re.Pattern[str],
    replace: Callable[[re.Match[str]], Awaitable[str]],
    text: str,
) -> str:
    parts: list[str] = []
    last = 0
    for match in pattern.finditer(text):
        parts.append(text[last : match.start()])
        parts.append(await replace(match))
        last = match.end()
    parts.append(text[last:])
    return "".join(parts)


async def _remote_resource_source(
    url: Any,
    mime: Any,
    workspace_root: Path,
    fetch_url: RemoteMediaFetcher,
    fetched: dict[str, bytes | None],
) -> dict[str, Any] | None:
    data = await _fetch_remote_media(url, fetch_url, fetched)
    if data is None:
        return None
    if not (isinstance(mime, str) and mime.startswith("image/")):
        mime = _sniff_image_mime(data)
    return _stored_resource_source(data, mime, workspace_root)


def _resource_source(
    encoded: Any, mime: Any, workspace_root: Path
) -> dict[str, Any] | None:
    data = _decode_inline(encoded)
    if data is None or len(data) < _INLINE_OFFLOAD_MIN_BYTES:
        return None
    return _stored_resource_source(data, mime, workspace_root)


def _stored_resource_source(
    data: bytes, mime: Any, workspace_root: Path
) -> dict[str, Any]:
    relative = _store_media_bytes(data, workspace_root, _mime_extension(mime))
    return {
        "kind": "resource",
        "uri": relative,
        **{key: None for key in _RESOURCE_SOURCE_KEYS},
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


async def _fetch_remote_media(
    url: Any,
    fetch_url: RemoteMediaFetcher,
    fetched: dict[str, bytes | None],
) -> bytes | None:
    """Return one URL's bytes, or None when the link must stay remote.

    Per-URL failures cache as None so every later reference to the same URL
    keeps its link without re-attempting the download. Any transport failure
    degrades to the remote link; storage errors still propagate to the
    payload-level fallback in `offload_remote_media`.
    """

    if not isinstance(url, str) or not url.lower().startswith(("http://", "https://")):
        return None
    if url in fetched:
        return fetched[url]
    try:
        data = await fetch_url(url)
    except Exception:  # noqa: BLE001 - transport failures degrade per URL
        fetched[url] = None
        return None
    # Thumbnail-scale images stay remote for the same reason inline ones stay
    # inline: offloading them would litter the uploads directory. Oversized
    # payloads stay remote instead of filling the workspace either.
    if len(data) < _INLINE_OFFLOAD_MIN_BYTES or len(data) > MEDIA_MAX_BYTES:
        fetched[url] = None
        return None
    fetched[url] = data
    return data


def _sniff_image_mime(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def default_remote_fetcher(url: str) -> Awaitable[bytes]:
    """Fetch one image over HTTP(S) via stdlib urllib, off the event loop.

    Reads at most `MEDIA_MAX_BYTES + 1` bytes; the size check lives in the
    caller so oversized payloads degrade to the remote link there.
    """

    async def fetch() -> bytes:
        request = urllib.request.Request(
            url, headers={"User-Agent": "pygent.v1-media-offload"}
        )

        def download() -> bytes:
            with urllib.request.urlopen(
                request, timeout=_REMOTE_FETCH_TIMEOUT_SECONDS
            ) as response:
                return response.read(MEDIA_MAX_BYTES + 1)

        return await asyncio.to_thread(download)

    return fetch()


def _decode_inline(value: Any) -> bytes | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return None


def _mime_extension(mime: Any) -> str:
    subtype = str(mime or "").rpartition("/")[2].lower()
    return _MIME_EXTENSIONS.get(subtype, ".png")


def _store_media_bytes(
    data: bytes, workspace_root: Path, extension: str
) -> str:
    digest = hashlib.sha256(data).hexdigest()
    relative = (
        UPLOADS_SUBDIR / f"media-{digest[:16]}{extension}"
    ).as_posix()
    target = workspace_root / relative
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        # Content-addressed names make retries idempotent; the temp-file
        # rename keeps a crash mid-write from leaving a truncated image
        # behind under its final name.
        tmp = target.with_name(f"{target.name}.{uuid.uuid4().hex[:8]}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, target)
    return relative
