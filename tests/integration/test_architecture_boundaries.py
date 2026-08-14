"""Static dependency rules that keep the refactored domain layering stable."""

from __future__ import annotations

import ast
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "pygent"


def _runtime_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()

    def visit(nodes: list[ast.stmt], *, type_checking: bool = False) -> None:
        for node in nodes:
            guarded = type_checking or (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Name)
                and node.test.id == "TYPE_CHECKING"
            )
            if isinstance(node, ast.Import) and not guarded:
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and not guarded and node.module:
                imports.add(node.module)
            for field in ("body", "orelse"):
                children = getattr(node, field, None)
                if isinstance(children, list):
                    visit(children, type_checking=guarded)

    visit(tree.body)
    return imports


def test_core_has_no_runtime_dependency_on_tool_or_runtime() -> None:
    violations: list[str] = []
    for path in (SOURCE_ROOT / "core").glob("*.py"):
        for imported in _runtime_imports(path):
            if imported == "pygent.tool" or imported.startswith("pygent.tool."):
                violations.append(f"{path.name}: {imported}")
            if imported == "pygent.runtime" or imported.startswith("pygent.runtime."):
                violations.append(f"{path.name}: {imported}")
    assert not violations


def test_production_does_not_import_removed_tool_value_definitions() -> None:
    core_values = {"ToolCall", "ToolDefinition", "ToolResult", "ToolTask"}
    violations: list[str] = []
    for path in SOURCE_ROOT.rglob("*.py"):
        if path == SOURCE_ROOT / "tool" / "types.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "pygent.tool.types"
                and core_values.intersection(alias.name for alias in node.names)
            ):
                violations.append(str(path.relative_to(SOURCE_ROOT)))
    assert not violations
