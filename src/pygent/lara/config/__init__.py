"""Kernel configuration surface: schema-level validation only.

Configuration *mechanism* (reading ``config.yaml``, the YAML subset, and file
editing) belongs to :mod:`lara_shell.config`. The kernel keeps the pure shape
validation used by that loader together with the typed models in
:mod:`lara.schema`.
"""

from __future__ import annotations

from .validation import validate_config_mapping

__all__ = ["validate_config_mapping"]
