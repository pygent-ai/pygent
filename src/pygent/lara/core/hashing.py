"""Stable canonical-JSON fingerprints shared by durable stores."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def fingerprint(**values: Any) -> str:
    payload = json.dumps(
        values, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
