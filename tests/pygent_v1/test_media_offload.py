"""Checkpoint media offload: shared download memo and host uploads directory."""

from __future__ import annotations

import asyncio
import base64
import os
import subprocess
from pathlib import Path

import pytest

from pygent.agent.pygent_v1.graph.media_offload import (
    UPLOADS_SUBDIR,
    offload_inline_media,
    offload_remote_media,
)

_IMAGE = b"\x89PNG\r\n\x1a\n" + b"0" * (128 * 1024)
_URL = "https://images.example/picture.png"


def _payload_with_repeated_url() -> dict:
    return {
        "content": f"![a]({_URL}) and again ![b]({_URL})",
        "blocks": [
            {
                "type": "media",
                "mime_type": "image/png",
                "source": {"kind": "url", "url": _URL},
            }
        ],
    }


def test_same_remote_url_downloaded_once(tmp_path: Path) -> None:
    calls: list[str] = []

    async def fetch(url: str) -> bytes:
        calls.append(url)
        return _IMAGE

    result = asyncio.run(
        offload_remote_media(
            _payload_with_repeated_url(), workspace_root=tmp_path, fetch_url=fetch
        )
    )

    assert calls == [_URL]
    stored = result["blocks"][0]["source"]["uri"]
    assert stored.startswith(UPLOADS_SUBDIR.as_posix())
    assert result["content"].count(stored) == 2
    assert (tmp_path / stored).read_bytes() == _IMAGE


def test_failed_remote_url_not_retried_within_payload(tmp_path: Path) -> None:
    calls: list[str] = []

    async def fetch(url: str) -> bytes:
        calls.append(url)
        raise OSError("host unreachable")

    payload = _payload_with_repeated_url()
    result = asyncio.run(
        offload_remote_media(payload, workspace_root=tmp_path, fetch_url=fetch)
    )

    assert calls == [_URL]
    assert result == payload


def test_uploads_dir_overrides_default_location(tmp_path: Path) -> None:
    async def fetch(url: str) -> bytes:
        return _IMAGE

    remote = asyncio.run(
        offload_remote_media(
            _payload_with_repeated_url(),
            workspace_root=tmp_path,
            fetch_url=fetch,
            uploads_dir=Path(".lara") / "uploads",
        )
    )
    inline = offload_inline_media(
        {
            "type": "media",
            "mime_type": "image/png",
            "source": {
                "kind": "inline",
                "base64_data": base64.b64encode(_IMAGE).decode(),
            },
        },
        workspace_root=tmp_path,
        uploads_dir=".lara/uploads",
    )

    remote_uri = remote["blocks"][0]["source"]["uri"]
    assert remote_uri.startswith(".lara/uploads/media-")
    assert inline["source"]["uri"] == remote_uri
    assert (tmp_path / remote_uri).exists()
    assert not (tmp_path / UPLOADS_SUBDIR).exists()


@pytest.mark.parametrize("uploads_dir", ["../outside", "ABSOLUTE"])
def test_uploads_dir_must_stay_inside_workspace(
    tmp_path: Path, uploads_dir: str
) -> None:
    if uploads_dir == "ABSOLUTE":
        uploads_dir = str(tmp_path / "uploads")
    with pytest.raises(ValueError, match="uploads_dir"):
        offload_inline_media({}, workspace_root=tmp_path, uploads_dir=uploads_dir)


def test_symlinked_uploads_dir_cannot_escape_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    (workspace / ".lara").mkdir(parents=True)
    outside.mkdir()
    link = workspace / ".lara" / "uploads"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        # Unprivileged Windows accounts cannot create symlinks; a junction
        # redirects the same way and needs no privilege.
        if os.name != "nt":
            pytest.skip(f"symlink creation unavailable: {exc}")
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(outside)],
            check=True,
            capture_output=True,
        )

    payload = {
        "type": "media",
        "mime_type": "image/png",
        "source": {"kind": "inline", "base64_data": base64.b64encode(_IMAGE).decode()},
    }
    result = offload_inline_media(
        payload, workspace_root=workspace, uploads_dir=".lara/uploads"
    )

    # Storage is refused and the payload degrades to inline bytes.
    assert result == payload
    assert list(outside.iterdir()) == []
