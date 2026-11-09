from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any


REDACTED = "[REDACTED]"

_SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "credential",
    "password",
    "private_key",
    "secret",
)
_SENSITIVE_KEY_NAMES = (
    "access_token",
    "authorization",
    "auth_token",
    "bearer_token",
    "id_token",
    "proxy_authorization",
    "refresh_token",
    "session_token",
    "token",
)

_DOTENV_SECRET_RE = re.compile(
    r"(?im)^([ \t]*(?:[A-Z0-9_]*)(?:API[_-]?KEY|AUTH[_-]?TOKEN|PASSWORD|PRIVATE[_-]?KEY|SECRET|TOKEN|CREDENTIAL)(?:[A-Z0-9_]*)[ \t]*=[ \t]*)([^\r\n#]+)"
)
_ASSIGNMENT_SECRET_RE = re.compile(
    r"(?i)\b((?:api[_-]?key|auth[_-]?token|password|private[_-]?key|secret|token|credential)[\"']?[ \t]*[:=][ \t]*[\"']?)([^\"'\s,;}]+)"
)
_OPENAI_STYLE_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")


def redact_secrets(value: Any) -> Any:
    """Return a copy of value with common credentials removed before persistence."""
    return _redact(value, ())


def _redact(value: Any, path: tuple[str, ...]) -> Any:
    if isinstance(value, dict):
        redacted: dict[Any, Any] = {}
        for key, child in value.items():
            key_text = str(key)
            child_path = (*path, key_text)
            redacted[key] = REDACTED if _is_sensitive_key(key_text) else _redact(child, child_path)
        return redacted
    if isinstance(value, list):
        return [_redact(item, path) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact(item, path) for item in value)
    if isinstance(value, str):
        return _redact_string(value)
    return deepcopy(value)


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    if normalized.endswith("_source"):
        return False
    return normalized in _SENSITIVE_KEY_NAMES or any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def _redact_string(value: str) -> str:
    redacted = value
    redacted = _DOTENV_SECRET_RE.sub(lambda match: f"{match.group(1)}{REDACTED}", redacted)
    redacted = _ASSIGNMENT_SECRET_RE.sub(lambda match: f"{match.group(1)}{REDACTED}", redacted)
    redacted = _OPENAI_STYLE_KEY_RE.sub(REDACTED, redacted)
    redacted = _JWT_RE.sub(REDACTED, redacted)
    if redacted == value:
        parsed = _try_parse_json_string(value)
        if parsed is not None:
            parsed_redacted = redact_secrets(parsed)
            return json.dumps(parsed_redacted, ensure_ascii=False)
    return redacted


def _try_parse_json_string(value: str) -> Any | None:
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None
