from __future__ import annotations

import inspect
import json
import os
import threading
import uuid
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_JSONL_LOCKS: dict[str, threading.RLock] = {}
_JSONL_LOCKS_GUARD = threading.Lock()


def _jsonl_lock(path: str | Path) -> threading.RLock:
    key = os.path.normcase(str(Path(path).resolve()))
    with _JSONL_LOCKS_GUARD:
        return _JSONL_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def jsonl_path_lock(path: str | Path):
    """Serialize in-process readers and appenders for one JSONL path."""

    with _jsonl_lock(path):
        yield


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: str | Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    json_path = Path(path)
    if default is not None and not json_path.exists():
        return dict(default)
    return json.loads(json_path.read_text(encoding="utf-8"))


def _json_text(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(_json_text(data), encoding="utf-8")


def write_text(path: str | Path, content: str) -> None:
    text_path = Path(path)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(content, encoding="utf-8")


def write_atomic(
    path: str | Path,
    data: Any,
    *,
    serialize: Callable[[Any], str],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(serialize(data))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(path: str | Path, data: dict[str, Any]) -> None:
    write_atomic(path, data, serialize=_json_text)


def append_jsonl(path: str | Path, data: dict[str, Any], *, durable: bool = False) -> None:
    jsonl_path = Path(path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    row = json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n"
    with jsonl_path_lock(jsonl_path), jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(row)
            if durable:
                handle.flush()
                os.fsync(handle.fileno())


def validate_path_id(value: str, field_name: str) -> None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or "/" in value or "\\" in value:
        raise ValueError(f"{field_name} must not contain path traversal")


def parse_env_entries(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        return []
    entries: list[tuple[str, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        entries.append((key.strip(), value.strip().strip('"').strip("'")))
    return entries


def load_env_file(path: Path) -> None:
    for key, value in parse_env_entries(path):
        os.environ.setdefault(key, value)


def non_empty_string(value: object | None) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def plain_data(value: Any) -> Any:
    if hasattr(value, "data") and not isinstance(value, type):
        return plain_data(value.data)
    if isinstance(value, Mapping):
        return {key: plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_data(item) for item in value]
    return value


def plain_object(value: Any) -> dict[str, Any]:
    data = plain_data(value)
    return data if isinstance(data, dict) else {}


async def aclose_if_supported(value: Any) -> None:
    close = getattr(value, "aclose", None)
    if callable(close) and inspect.isawaitable(result := close()):
        await result
