"""User-attachment handling: workspace paths -> native media or inline XML."""

from __future__ import annotations

import mimetypes
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path

from pygent import MediaBlock, MediaSource
from pygent.agent.lara.core.prompt import Prompt

MAX_ATTACHMENTS = 10
MAX_INLINE_BYTES = 64 * 1024
MEDIA_MAX_BYTES = 20 * 1024 * 1024

IMAGE_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}
VIDEO_MIME_BY_SUFFIX = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
}
TEXT_SUFFIXES = frozenset(
    {
        ".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv",
        ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env",
        ".xml", ".html", ".htm", ".css", ".scss", ".less",
        ".py", ".pyi", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx",
        ".java", ".kt", ".go", ".rs", ".rb", ".php", ".swift", ".c", ".h",
        ".cpp", ".hpp", ".cc", ".cs", ".sh", ".bash", ".bat", ".ps1",
        ".sql", ".graphql", ".proto", ".vue", ".svelte", ".dockerfile",
        ".gitignore", ".editorconfig", ".lock", ".sum", ".mod", ".gradle",
    }
)


@dataclass(frozen=True, slots=True)
class ResolvedAttachment:
    """One user attachment classified for delivery."""

    path: Path
    relative: str
    name: str
    size: int
    mime: str
    delivery: str  # "media" | "inline" | "reference"
    media_type: str | None = None  # "image" | "video" when delivery == "media"

    @property
    def is_media(self) -> bool:
        return self.delivery == "media"


def resolve_attachments(
    paths: Sequence[str], workspace_root: str | Path
) -> tuple[ResolvedAttachment, ...]:
    """Validate and classify attachment paths; raise ValueError on bad input."""

    root = Path(workspace_root).resolve()
    unique: list[str] = []
    for raw in paths:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("attachment paths must be non-empty strings")
        if raw not in unique:
            unique.append(raw)
    if len(unique) > MAX_ATTACHMENTS:
        raise ValueError(f"too many attachments; at most {MAX_ATTACHMENTS} allowed")

    resolved: list[ResolvedAttachment] = []
    for raw in unique:
        path = _workspace_path(root, raw)
        if not path.is_file():
            raise ValueError(f"attachment file not found: {raw}")
        stat = path.stat()
        relative = path.relative_to(root).as_posix()
        mime, media_type, delivery = _classify(path)
        if delivery == "media" and stat.st_size > MEDIA_MAX_BYTES:
            delivery = "reference"
        resolved.append(
            ResolvedAttachment(
                path=path,
                relative=relative,
                name=path.name,
                size=stat.st_size,
                mime=mime,
                delivery=delivery,
                media_type=media_type if delivery == "media" else None,
            )
        )
    return tuple(resolved)


def build_media_blocks(
    attachments: Sequence[ResolvedAttachment],
) -> tuple[MediaBlock, ...]:
    """Native pygent media blocks for image/video attachments."""

    return tuple(
        MediaBlock(
            media_type=item.media_type,  # type: ignore[arg-type]
            mime_type=item.mime,
            source=MediaSource.resource(str(item.path), size_bytes=item.size),
        )
        for item in attachments
        if item.is_media
    )


def render_attachments_xml(attachments: Sequence[ResolvedAttachment]) -> str:
    """Render the <attachments> block placed inside <user-context>."""

    item_lines: list[str] = []
    for item in attachments:
        attrs = [
            f'delivery="{item.delivery}"',
            f'name="{escape(item.name)}"',
            f'path="{escape(item.relative)}"',
            f'size="{item.size}"',
        ]
        if item.mime:
            attrs.append(f'mime="{escape(item.mime)}"')
        if item.delivery == "media":
            attrs.append(
                'note="media attached natively to this user message"'
            )
            item_lines.append(f"<attachment {' '.join(attrs)}/>")
        elif item.delivery == "inline":
            content = _read_inline(item.path)
            truncated = content is None
            body = (
                content
                if content is not None
                else item.path.read_bytes()[:MAX_INLINE_BYTES].decode(
                    "utf-8", errors="replace"
                )
            )
            attrs.append(f'truncated="{str(truncated).lower()}"')
            if truncated:
                attrs.append('note="content truncated; use read tool for the full file"')
            item_lines.append(
                f"<attachment {' '.join(attrs)}>{escape(body, quote=False)}</attachment>"
            )
        else:
            attrs.append(
                'note="content not embedded; inspect with read/grep tools"'
            )
            item_lines.append(f"<attachment {' '.join(attrs)}/>")
    if not item_lines:
        return "  <attachments>\n  </attachments>"
    block = Prompt("\n".join(item_lines)).wrap("attachments", indent="  ")
    return "\n".join(f"  {line}" for line in str(block).splitlines())


def _workspace_path(root: Path, raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"attachment path escapes the workspace: {raw}")
    return resolved


def _classify(path: Path) -> tuple[str, str | None, str]:
    suffix = path.suffix.lower()
    if suffix in IMAGE_MIME_BY_SUFFIX:
        return IMAGE_MIME_BY_SUFFIX[suffix], "image", "media"
    if suffix in VIDEO_MIME_BY_SUFFIX:
        return VIDEO_MIME_BY_SUFFIX[suffix], "video", "media"
    if suffix in TEXT_SUFFIXES:
        guessed, _ = mimetypes.guess_type(path.name)
        if guessed is None or not guessed.startswith("text/"):
            guessed = "text/plain"
        return guessed, None, "inline"
    return "", None, "reference"


def _read_inline(path: Path) -> str | None:
    """Read full text content; return None when the file exceeds the limit."""

    if path.stat().st_size > MAX_INLINE_BYTES:
        return None
    return path.read_bytes().decode("utf-8", errors="replace")


__all__ = [
    "MAX_ATTACHMENTS",
    "MAX_INLINE_BYTES",
    "MEDIA_MAX_BYTES",
    "ResolvedAttachment",
    "build_media_blocks",
    "render_attachments_xml",
    "resolve_attachments",
]
