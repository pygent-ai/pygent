"""Role-scoped memory job directive envelopes.

The extractor and builder directives are injected as the fork/child's
block-2 user message wrapped in their own registered envelopes
(``memory-extract-instruction`` / ``memory-build-instruction``), each
carrying only that role's command surface.
"""

from __future__ import annotations

from pathlib import Path

from pygent.lara.runtime.eternal_conversation import (
    MemoryJobContext,
    _builder_directive,
    _extractor_directive,
)


def _context(tmp_path: Path, *, build_batch_path: Path | None = None) -> MemoryJobContext:
    return MemoryJobContext(
        session_id="s1",
        memory_script=tmp_path / "skills" / "dynamic_memory_cli.py",
        memory_root=tmp_path / "memory",
        evidence_ref="raw-history.jsonl:1-40",
        covered_through=12,
        target_cursor=40,
        build_batch_path=build_batch_path,
    )


def test_extractor_directive_wraps_extract_envelope(tmp_path: Path) -> None:
    directive = _extractor_directive(
        context=_context(tmp_path),
        snapshot={"resident_memory": [], "recent_context": []},
        published_uts=[],
        previous_failure=None,
    )

    assert directive.startswith("<memory-extract-instruction>")
    assert directive.endswith("</memory-extract-instruction>")
    assert "get-state" in directive
    assert "publish-pending" in directive
    assert "build-pending" not in directive


def test_builder_directive_wraps_build_envelope(tmp_path: Path) -> None:
    batch_path = tmp_path / "memory" / "jobs" / "batch.json"
    directive = _builder_directive(
        context=_context(tmp_path, build_batch_path=batch_path),
        batch={"uts": []},
    )

    assert directive.startswith("<memory-build-instruction>")
    assert directive.endswith("</memory-build-instruction>")
    assert "build-pending" in directive
    assert "publish-pending" not in directive
