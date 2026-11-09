"""Meta-test: every prompt envelope tag must be registered.

The scanner catches two escape hatches:
- raw string/f-string literals that start an opening tag, which would bypass
  the Prompt API; and
- literal tags passed to ``Prompt.wrap``/``wrap_escaped``/``self_closing``.

Inner body tags that are not envelope classes are listed explicitly; adding a
new top-level envelope tag without registering it fails the test.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from pygent.lara.core.prompt import KNOWN_ENVELOPES

SOURCE_ROOTS = (
    Path(__file__).resolve().parents[2] / "src" / "pygent" / "lara",
)
# The bundled dynamic-memory-cli tool is vendored third-party data; its
# output tags are its own wire format, not the prompt envelope layer.
_VENDORED_DATA_ROOTS = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "pygent"
    / "lara"
    / "clis"
    / "dynamic-memory-cli",
)
_TAG_START = re.compile(r"(?m)^\s*(</?)([a-z][a-z-]*)")
_PROMPT_METHODS = {"wrap", "wrap_escaped", "self_closing"}

# Tags rendered inside another envelope. They are not envelope classes and do
# not own a renderer, so they are exempt from the KNOWN_ENVELOPES catalog.
_INNER_TAGS = {
    "automation-trigger": "dispatch control message, outside the prompt envelope layer",
    "task-name": "dispatch control message field",
    "scheduled-for": "dispatch control message field",
    "timezone": "dispatch control message field",
    "mode": "dispatch control message field",
    "unattended": "dispatch control message field",
    "instructions": "dispatch control message field",
    "attachment": "entry inside the attachments envelope",
    "example": "compression prompt example",
    "analysis": "model-generated compression analysis block",
    "memory-snapshot": "snapshot body inside memory-access-instruction",
    "user-asked": "inner exchange frame",
    "your-response-in-another-session": "inner exchange frame",
    "session": "runtime-context field",
    "workspace-root": "runtime-context field",
    "notice": "git-context notice",
    "changes": "git-context delta wrapper",
    "status": "git-context status wrapper",
    "branch-change": "git-context delta entry",
    "change": "git-context delta entry",
    "skill": "skills-context entry",
    "description": "skills-context entry field",
    "path": "skills-context entry field",
    "user-identity": "user-context field",
    "user-message": "user-context field",
    "task": "task-context metadata tag",
}


def _iter_literal_tags(tree: ast.AST) -> list[tuple[str, int]]:
    tags: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        values: list[str] = []
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            values = [node.value]
        elif isinstance(node, ast.JoinedStr):
            values = [
                value.value
                for value in node.values
                if isinstance(value, ast.Constant) and isinstance(value.value, str)
            ]
        for value in values:
            for match in _TAG_START.finditer(value):
                # Closing tags are represented by their opening counterpart.
                if match.group(1) == "</":
                    continue
                tags.append((match.group(2), getattr(node, "lineno", 0)))
    return tags


def _iter_prompt_tags(tree: ast.AST) -> list[tuple[str, int]]:
    tags: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in _PROMPT_METHODS or not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            tags.append((first.value, node.lineno))
    return tags


def test_every_envelope_tag_is_registered() -> None:
    failures: list[str] = []
    scanned = 0
    for root in SOURCE_ROOTS:
        for path in sorted(root.rglob("*.py")):
            if any(
                path.is_relative_to(vendored) for vendored in _VENDORED_DATA_ROOTS
            ):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for tag, line in [
                *_iter_literal_tags(tree),
                *_iter_prompt_tags(tree),
            ]:
                scanned += 1
                if tag in KNOWN_ENVELOPES or tag in _INNER_TAGS:
                    continue
                failures.append(f"{path}:{line}: unregistered envelope tag {tag!r}")
    assert scanned > 0, "envelope scanner found no tags"
    assert not failures, "\n".join(failures)


def test_inner_tag_exemptions_are_not_envelope_classes() -> None:
    assert set(_INNER_TAGS).isdisjoint(KNOWN_ENVELOPES)
