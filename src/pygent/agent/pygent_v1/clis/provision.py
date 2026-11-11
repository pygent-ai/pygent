"""Provision bundled CLI tools into the user CLIs directory.

The bundled tool lives next to this module at
``dynamic-memory-cli/dynamic_memory_cli.py``; its canonical source is
pygent-ai/memory-cli branch ``codex/dynamic-memory-cli`` at commit
``c2b500673b0c9da45bc343f7c3ff252ef7e035ec`` — the same revision the removed
git dependency was locked to in ``uv.lock``.

:func:`ensure_dynamic_memory_cli` copies the tool to
``<user_clis_root>/dynamic-memory-cli/dynamic_memory_cli.py`` so that the
eternal-conversation harness resolves a stable execution path. A
``.provisioned.json`` marker records the mounted file's hash: an existing
copy is upgraded to the bundled version only when its content still matches
the marker (i.e. the user has not touched it); a user-modified copy is left
untouched and reported through ``RuntimeWarning``.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

from pygent.agent.pygent_v1.core.io import utc_now

CLI_NAME = "dynamic-memory-cli"
MARKER_NAME = ".provisioned.json"
_SCRIPT_FILENAME = "dynamic_memory_cli.py"


@dataclass(frozen=True, slots=True)
class _BundledCli:
    script: bytes
    script_sha256: str


def bundled_cli_dir() -> Path:
    return Path(__file__).resolve().parent / CLI_NAME


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_bundle() -> _BundledCli:
    script = (bundled_cli_dir() / _SCRIPT_FILENAME).read_bytes()
    return _BundledCli(script=script, script_sha256=_hash_bytes(script))


def _read_marker(path: Path) -> dict[str, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _atomic_write(path: Path, data: bytes) -> None:
    # Concurrent first-runs share the target directory; a torn script or
    # marker must never become visible, so replace atomically instead of
    # writing in place.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _write_marker(path: Path, bundle: _BundledCli) -> None:
    payload = {
        "script_sha256": bundle.script_sha256,
        "provisioned_at": utc_now(),
    }
    _atomic_write(path, (json.dumps(payload, indent=2) + "\n").encode("utf-8"))


def _install(target: Path, bundle: _BundledCli) -> None:
    # Marker last: it is the commit point that marks the copy as owned.
    _atomic_write(target / _SCRIPT_FILENAME, bundle.script)
    _write_marker(target / MARKER_NAME, bundle)


def ensure_dynamic_memory_cli(user_clis_root: Path) -> None:
    """Mount the bundled CLI under ``user_clis_root``; idempotent.

    First mount copies the script wholesale. Later calls no-op while the
    mounted copy equals the bundled version, upgrade it when it still matches
    the ``.provisioned.json`` marker, and warn without touching it when the
    user has modified it. A bundle-owned partial install (deleted script
    alongside a current marker) is repaired silently.
    """
    bundle = _load_bundle()
    target = Path(user_clis_root).expanduser() / CLI_NAME
    script_path = target / _SCRIPT_FILENAME

    current_script = script_path.read_bytes() if script_path.is_file() else None
    if current_script is None:
        _install(target, bundle)
        return

    script_is_bundled = _hash_bytes(current_script) == bundle.script_sha256
    if script_is_bundled:
        marker = _read_marker(target / MARKER_NAME)
        if marker.get("script_sha256") != bundle.script_sha256:
            _install(target, bundle)
        return

    marker = _read_marker(target / MARKER_NAME)
    owned = marker.get("script_sha256") == _hash_bytes(current_script)
    if not owned:
        warnings.warn(
            f"user-modified {CLI_NAME} CLI at {target} was left untouched; "
            "the bundled version was not upgraded",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    _install(target, bundle)
