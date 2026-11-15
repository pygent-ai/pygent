"""File-effect tracking and git observation skip the host's in-workspace runtime directory."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from pygent.agent.pygent_v1.runtime.file_effect_models import (
    DeferredFileEffectBatch,
    DeferredFileEffectJob,
)
from pygent.agent.pygent_v1.runtime.reminders import ReminderService
from pygent.agent.pygent_v1.runtime.reminders.git_context import capture_git_snapshot
from pygent.agent.pygent_v1.runtime.tools import FileEffectTracker
from pygent.agent.pygent_v1.schema import CaseRunRef, RunConfig
from pygent.agent.pygent_v1.tracing.events import EventStore


def _store(tmp_path: Path) -> tuple[EventStore, CaseRunRef]:
    run_dir = tmp_path / "sessions" / "s1" / "runs" / "r1"
    run_dir.mkdir(parents=True)
    ref = CaseRunRef(
        session_id="s1", case_id="chat", case_run_id="r1", run_dir=str(run_dir)
    )
    return EventStore(ref), ref


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    (workspace / ".lara" / "uploads").mkdir(parents=True)
    (workspace / ".lara" / "uploads" / "media.png").write_bytes(b"x")
    (workspace / ".pygent_v1").mkdir()
    (workspace / ".pygent_v1" / "state.json").write_text("{}")
    (workspace / "src").mkdir()
    (workspace / "src" / "app.py").write_text("print()")
    return workspace


def test_snapshot_skips_configured_runtime_dir_and_default(tmp_path: Path) -> None:
    store, _ref = _store(tmp_path)
    workspace = _workspace(tmp_path)

    default_tracker = FileEffectTracker(workspace_root=workspace, store=store)
    host_tracker = FileEffectTracker(
        workspace_root=workspace, store=store, ignored_dirs=(".lara",)
    )

    default_paths = {Path(p).as_posix() for p in default_tracker.snapshot_workspace()}
    host_paths = {Path(p).as_posix() for p in host_tracker.snapshot_workspace()}

    assert any(p.endswith("src/app.py") for p in default_paths)
    assert any(".lara/" in p for p in default_paths)
    assert not any(".pygent_v1/" in p for p in default_paths)
    assert any(p.endswith("src/app.py") for p in host_paths)
    assert not any(".lara/" in p for p in host_paths)
    assert not any(".pygent_v1/" in p for p in host_paths)


def test_deferred_batch_round_trips_ignored_dirs(tmp_path: Path) -> None:
    _store_obj, ref = _store(tmp_path)
    job = DeferredFileEffectJob(
        tool_call_id="call-1", tool_name="bash", args={"command": "ls"},
        turn_id="turn-1", declared=[], include_declared=False,
    )
    batch = DeferredFileEffectBatch.create(
        case_run_ref=ref, workspace_root=tmp_path, jobs=[job], ignored_dirs=(".lara",)
    )

    restored = DeferredFileEffectBatch.from_dict(batch.to_dict())

    assert restored.ignored_dirs == (".lara",)
    # Batches persisted before the field existed still load.
    legacy = batch.to_dict()
    del legacy["ignored_dirs"]
    assert DeferredFileEffectBatch.from_dict(legacy).ignored_dirs == ()


def test_git_snapshot_excludes_configured_runtime_dir(tmp_path: Path) -> None:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not installed")
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run([git, "init", "-q"], cwd=repo, check=True)
    (repo / ".lara").mkdir()
    (repo / ".lara" / "state.json").write_text("{}")
    (repo / ".pygent_v1").mkdir()
    (repo / ".pygent_v1" / "state.json").write_text("{}")
    (repo / "notes.md").write_text("hello")

    plain = capture_git_snapshot(repo)
    host = capture_git_snapshot(repo, exclude_dirs=(".lara",))
    config = RunConfig(
        workspace_root=str(repo),
        user_pygent_root=str(tmp_path / "u"),
        workspace_runtime_dir=".lara",
    )
    scope = ReminderService(config).scope("session-1")
    via_scope = capture_git_snapshot(
        scope.workspace_root, exclude_dirs=(scope.workspace_runtime_dir,)
    )

    assert plain is not None and host is not None and via_scope is not None
    assert {".lara/", ".pygent_v1/", "notes.md"} <= set(plain["entries"])
    assert set(host["entries"]) == {".pygent_v1/", "notes.md"}
    assert via_scope["entries"] == host["entries"]


def test_run_config_workspace_runtime_dir(tmp_path: Path) -> None:
    def build(runtime_dir: str = ".pygent_v1") -> RunConfig:
        return RunConfig(
            workspace_root=str(tmp_path),
            user_pygent_root=str(tmp_path / "u"),
            workspace_runtime_dir=runtime_dir,
        )

    default = RunConfig(workspace_root=str(tmp_path), user_pygent_root=str(tmp_path / "u"))
    assert default.workspace_runtime_dir == ".pygent_v1"
    assert build(".lara").workspace_runtime_dir == ".lara"
    restored = RunConfig.from_dict(build(".lara").to_dict())
    assert restored.workspace_runtime_dir == ".lara"
    for bad in ("", "nested/dir", "..", "."):
        with pytest.raises(ValueError, match="workspace_runtime_dir"):
            build(bad)
