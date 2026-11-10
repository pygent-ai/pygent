"""Provisioning of the bundled dynamic-memory-cli tool into the user CLIs dir.

The tool ships with the package (``lara.clis``) and is mounted on first use
of the eternal-conversation harness, so the harness's script resolution
works without the removed git dependency. The mounted directory carries no
SKILL.md and lives outside the agent-facing skills directories.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from pygent.agent.lara.clis.provision import (
    MARKER_NAME,
    bundled_cli_dir,
    ensure_dynamic_memory_cli,
)
from pygent.agent.lara.graph.skill_catalog import scan_multilevel_skills
from pygent.agent.lara.runtime.eternal_conversation import (
    EternalConversationHarness,
    MemoryJobRequest,
)
from pygent.agent.lara.schema import EternalConversationConfig


async def _fail_runner(request: MemoryJobRequest) -> str:
    raise AssertionError("memory job must not run during script resolution")


def _harness(clis_root: Path) -> EternalConversationHarness:
    return EternalConversationHarness(
        EternalConversationConfig(enabled=True),
        run_memory_job=_fail_runner,
        user_clis_root=clis_root,
    )


def _session_dir(tmp_path: Path) -> Path:
    return tmp_path / "lara" / "sessions" / "s1"


def _mounted(clis_root: Path) -> Path:
    return clis_root / "dynamic-memory-cli" / "dynamic_memory_cli.py"


def test_provision_mounts_bundled_cli_and_script_resolves(tmp_path: Path) -> None:
    clis_root = tmp_path / "home" / ".lara" / "clis"
    harness = _harness(clis_root)

    script = harness._script(_session_dir(tmp_path))

    mounted = _mounted(clis_root)
    assert script == mounted
    assert mounted.read_bytes() == (
        bundled_cli_dir() / "dynamic_memory_cli.py"
    ).read_bytes()
    assert (mounted.parent / MARKER_NAME).is_file()

    # The mounted script is runnable standalone.
    completed = subprocess.run(
        [sys.executable, str(mounted), "--help"],
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0


def test_provision_is_idempotent(tmp_path: Path) -> None:
    clis_root = tmp_path / "clis"
    ensure_dynamic_memory_cli(clis_root)
    mounted = _mounted(clis_root)
    first = mounted.read_bytes()

    ensure_dynamic_memory_cli(clis_root)

    assert mounted.read_bytes() == first


def test_provision_upgrades_owned_copy(tmp_path: Path) -> None:
    clis_root = tmp_path / "clis"
    ensure_dynamic_memory_cli(clis_root)
    mounted = _mounted(clis_root)
    script_bytes = b"outdated script"
    mounted.write_bytes(script_bytes)
    # Marker as a previous provision of this older bundle would have left it.
    marker = {
        "script_sha256": hashlib.sha256(script_bytes).hexdigest(),
        "provisioned_at": "2026-01-01T00:00:00+00:00",
    }
    (mounted.parent / MARKER_NAME).write_text(json.dumps(marker), encoding="utf-8")

    ensure_dynamic_memory_cli(clis_root)

    assert mounted.read_bytes() == (
        bundled_cli_dir() / "dynamic_memory_cli.py"
    ).read_bytes()


def test_provision_preserves_user_modified_copy(tmp_path: Path) -> None:
    clis_root = tmp_path / "clis"
    ensure_dynamic_memory_cli(clis_root)
    mounted = _mounted(clis_root)
    mounted.write_bytes(b"user customization")

    with pytest.warns(RuntimeWarning, match="user-modified"):
        ensure_dynamic_memory_cli(clis_root)

    assert mounted.read_bytes() == b"user customization"


def test_provision_self_heals_missing_script(tmp_path: Path) -> None:
    clis_root = tmp_path / "clis"
    ensure_dynamic_memory_cli(clis_root)
    mounted = _mounted(clis_root)
    mounted.unlink()

    # A bundle-owned partial install must be repaired silently, not reported
    # as user-modified.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ensure_dynamic_memory_cli(clis_root)

    assert mounted.read_bytes() == (
        bundled_cli_dir() / "dynamic_memory_cli.py"
    ).read_bytes()


def test_explicit_config_path_beats_provisioned_cli(tmp_path: Path) -> None:
    clis_root = tmp_path / "clis"
    explicit = tmp_path / "bin" / "dynamic_memory_cli.py"
    explicit.parent.mkdir(parents=True)
    explicit.write_bytes(b"# explicit cli path\n")
    harness = EternalConversationHarness(
        EternalConversationConfig(
            enabled=True, dynamic_memory_cli_path=str(explicit)
        ),
        run_memory_job=_fail_runner,
        user_clis_root=clis_root,
    )

    assert harness._script(_session_dir(tmp_path)) == explicit
    # Resolution never reached the CLIs directories, so nothing was mounted.
    assert not (clis_root / "dynamic-memory-cli").exists()


def test_project_clis_dir_still_resolves_first(tmp_path: Path) -> None:
    clis_root = tmp_path / "home" / ".lara" / "clis"
    session_dir = _session_dir(tmp_path)
    project_script = (
        session_dir.parents[1]
        / "clis"
        / "dynamic-memory-cli"
        / "dynamic_memory_cli.py"
    )
    project_script.parent.mkdir(parents=True)
    project_script.write_bytes(b"# project-local copy\n")
    harness = _harness(clis_root)

    assert harness._script(session_dir) == project_script


def test_provisioned_cli_is_not_a_skill(tmp_path: Path) -> None:
    clis_root = tmp_path / "clis"
    ensure_dynamic_memory_cli(clis_root)

    mounted = _mounted(clis_root)
    assert not (mounted.parent / "SKILL.md").exists()

    # Even pointed directly at the CLIs root, the catalog finds nothing:
    # the mounted directory carries no SKILL.md and is not a skill.
    assert (
        scan_multilevel_skills(
            user_skills_dir=clis_root,
            project_skills_dir=tmp_path / "project-skills",
        )
        == []
    )
