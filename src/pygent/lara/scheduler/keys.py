"""Per-source HMAC keys for scheduler submissions.

HMAC-SHA256 prevents zero-dependency local forgery; the two-key layout is
ready to be replaced by Ed25519 later without changing the store schema or
submission envelope. Secrets live in ``<user-root>/scheduler/keys``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any

SOURCES = ("user", "agent")


class SchedulerKeysMissing(RuntimeError):
    """Raised when a scheduler operation needs keys that were not initialized."""


def key_path(root: str | Path, source: str) -> Path:
    if source not in SOURCES:
        raise ValueError(f"unknown scheduler source: {source!r}")
    return Path(root) / "scheduler" / "keys" / f"{source}.key"


def init_keys(root: str | Path, *, force: bool = False) -> dict[str, Path]:
    created: dict[str, Path] = {}
    for source in SOURCES:
        path = key_path(root, source)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or force:
            path.write_text(os.urandom(32).hex() + "\n", encoding="utf-8")
            try:
                path.chmod(0o600)
            except OSError:
                pass
        created[source] = path
    return created


def load_secret(root: str | Path, source: str) -> bytes:
    path = key_path(root, source)
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise SchedulerKeysMissing(
            "scheduler keys are not initialized; run `lara scheduler init-keys`"
        ) from exc
    try:
        return bytes.fromhex(value)
    except ValueError as exc:
        raise SchedulerKeysMissing(f"invalid scheduler key: {path}") from exc


def canonical_payload(obj: Any) -> bytes:
    return json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sign(payload: Any, secret: bytes) -> str:
    return hmac.new(secret, canonical_payload(payload), hashlib.sha256).hexdigest()


def verify(payload: Any, source: str, signature: str, root: str | Path) -> bool:
    return hmac.compare_digest(sign(payload, load_secret(root, source)), signature)


__all__ = [
    "SOURCES",
    "SchedulerKeysMissing",
    "canonical_payload",
    "init_keys",
    "key_path",
    "load_secret",
    "sign",
    "verify",
]
