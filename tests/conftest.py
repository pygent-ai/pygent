"""Treat every collected test as part of the breaking Pygent 0.2.x contract."""

from __future__ import annotations

import sys
from importlib.abc import MetaPathFinder
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGE = REPO_ROOT / "src" / "pygent" / "__init__.py"


class _BlockExternalPygent(MetaPathFinder):
    """Prevent a site-packages build from masquerading as this workspace."""

    def find_spec(self, fullname, path=None, target=None):
        del path, target
        if fullname == "pygent" or fullname.startswith("pygent."):
            raise ModuleNotFoundError(
                "the local src/pygent package is absent; "
                "external installations are blocked during contract tests"
            )


if not LOCAL_PACKAGE.is_file():
    sys.meta_path.insert(0, _BlockExternalPygent())


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Apply one version contract without retaining a legacy test channel."""

    for item in items:
        item.add_marker(pytest.mark.contract_02)
