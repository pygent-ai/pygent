"""Import boundary regression: dependency direction and public SDK loading.

The internal implementation must not pull base capabilities through the
`pygent` SDK facade, the base packages must not reach into the pygent_v1
individual, and every documented import order must resolve to the same
class objects. Import-order checks run in fresh subprocesses so the test
process's `sys.modules` cannot mask initialization problems.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[2] / "src" / "pygent"

SDK_IMPORT_SCRIPT = """
from pygent import Agent, ReActAgent
from pygent.agent import PygentAgent
from pygent.agent.pygent_v1 import PygentAgent as LaraPygentAgent
from pygent.agent.react_agent import ReActAgent as BuiltinReActAgent

assert PygentAgent is LaraPygentAgent
assert ReActAgent is BuiltinReActAgent
print("ok")
"""

PYGENT_V1_FIRST_SCRIPT = """
from pygent.agent.pygent_v1 import PygentAgent as LaraPygentAgent
from pygent.agent.react_agent import ReActAgent as BuiltinReActAgent
from pygent import Agent, ReActAgent, ModelCallLayer
from pygent.agent import PygentAgent

assert PygentAgent is LaraPygentAgent
assert ReActAgent is BuiltinReActAgent
print("ok")
"""

LAZY_LOADING_SCRIPT = """
import sys

import pygent.agent

assert "pygent.agent.pygent_v1" not in sys.modules
from pygent.agent import PygentAgent

assert "pygent.agent.pygent_v1" in sys.modules
print("ok")
"""


def _run_fresh(script: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, f"import order failed:\n{result.stderr}"
    assert "ok" in result.stdout


def _iter_python_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.py"))


def _absolute_facade_imports(tree: ast.Module) -> list[ast.ImportFrom]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == "pygent":
            found.append(node)
    return found


def _imports_forbidden_target(tree: ast.Module, target: str) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == target or node.module.startswith(f"{target}."):
                return True
        elif isinstance(node, ast.Import) and any(
            alias.name == target or alias.name.startswith(f"{target}.")
            for alias in node.names
        ):
            return True
    return False


def test_internal_code_has_no_facade_imports() -> None:
    """Internal modules import owning packages, never the `pygent` facade."""
    offenders = []
    for path in _iter_python_files(REPO_SRC):
        if path.as_posix() == (REPO_SRC / "__init__.py").as_posix():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in _absolute_facade_imports(tree):
            offenders.append(f"{path.as_posix()}:{node.lineno}")
    assert offenders == []


def test_base_packages_never_import_pygent_v1() -> None:
    """core/llm/tool/runtime and the base agent modules stay pygent_v1-free."""
    guarded_roots = [
        REPO_SRC / "core",
        REPO_SRC / "llm",
        REPO_SRC / "tool",
        REPO_SRC / "runtime",
        REPO_SRC / "agent" / "react_agent",
        REPO_SRC / "agent" / "react.py",
        REPO_SRC / "agent" / "react_projection_operations.py",
        REPO_SRC / "agent" / "reminder.py",
    ]
    offenders = []
    for root in guarded_roots:
        paths = _iter_python_files(root) if root.is_dir() else [root]
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            if _imports_forbidden_target(tree, "pygent.agent.pygent_v1"):
                offenders.append(path.as_posix())
    assert offenders == []


def test_sdk_import_and_identity() -> None:
    _run_fresh(SDK_IMPORT_SCRIPT)


def test_pygent_v1_first_import_order() -> None:
    _run_fresh(PYGENT_V1_FIRST_SCRIPT)


def test_lazy_pygent_v1_export_defers_loading() -> None:
    _run_fresh(LAZY_LOADING_SCRIPT)


def test_heavy_lara_module_imports_standalone() -> None:
    _run_fresh("import pygent.agent.pygent_v1.graph.core\nprint('ok')")
    _run_fresh("import pygent.llm._json_sse_transport\nprint('ok')")
