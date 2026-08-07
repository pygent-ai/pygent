"""Governance tests for the Pygent 0.2 documentation and test layout."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = REPO_ROOT / "tests"
DOCS_ROOT = REPO_ROOT / "docs"
SOURCE_ROOT = REPO_ROOT / "src" / "pygent"
ROOT_DOCUMENTS = {"README.md", "FEATURES.md", "EXECUTION.md"}
MODULES = {"module", "context", "runtime", "agent", "llm", "tool"}
MODULE_DOCUMENTS = {"README.md", "FEATURES.md", "SDK.md"}
DATA_DIRECTORIES = {"fixtures"}


def test_02_documentation_has_one_directory_per_public_module():
    assert {path.name for path in DOCS_ROOT.iterdir() if path.is_file()} == ROOT_DOCUMENTS
    assert {path.name for path in DOCS_ROOT.iterdir() if path.is_dir()} == MODULES

    for module in MODULES:
        assert MODULE_DOCUMENTS <= {
            path.name for path in (DOCS_ROOT / module).iterdir() if path.is_file()
        }


def test_each_module_has_first_principles_then_sdk_contract():
    for module in MODULES:
        module_root = DOCS_ROOT / module
        principles = (module_root / "FEATURES.md").read_text(encoding="utf-8")
        sdk = (module_root / "SDK.md").read_text(encoding="utf-8")
        principle_title = principles.splitlines()[0].removeprefix("# ")

        assert "[Pygent 0.2 第一原则](../FEATURES.md)" in principles
        assert "第二级契约" in sdk
        assert f"[{principle_title}](FEATURES.md)" in sdk
        assert "```python" in sdk


def test_progressive_tutorial_is_linked_from_public_navigation() -> None:
    tutorial = DOCS_ROOT / "agent" / "TUTORIAL.md"
    assert tutorial.is_file()
    content = tutorial.read_text(encoding="utf-8")
    assert "python -m examples.tutorial" in content
    assert "python -m examples.tutorial managed" in content

    expected_links = {
        REPO_ROOT / "README.md": "docs/agent/TUTORIAL.md",
        DOCS_ROOT / "README.md": "agent/TUTORIAL.md",
        DOCS_ROOT / "agent" / "README.md": "TUTORIAL.md",
    }
    for document, link in expected_links.items():
        assert link in document.read_text(encoding="utf-8")


def test_source_layout_has_core_runtime_and_three_public_domains():
    assert {
        path.name
        for path in SOURCE_ROOT.iterdir()
        if path.is_dir() and not path.name.startswith("__")
    } == {"core", "runtime", "agent", "llm", "tool"}


def test_nonempty_test_directories_are_packages():
    package_directories = [
        path
        for path in TESTS_ROOT.iterdir()
        if path.is_dir()
        and any(path.glob("*.py"))
        and not path.name.startswith("__")
        and path.name not in DATA_DIRECTORIES
    ]

    assert package_directories
    assert not [
        path.name
        for path in package_directories
        if not (path / "__init__.py").is_file()
    ]


def test_test_modules_are_not_flat_or_hidden_in_support():
    assert not tuple(TESTS_ROOT.glob("test_*.py"))
    assert not tuple((TESTS_ROOT / "support").glob("test_*.py"))
