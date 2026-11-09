from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from pygent.lara.schema import RunConfig


@dataclass(frozen=True, slots=True)
class RuntimeScopeKey:
    workspace_root: str
    lara_root: str

    @classmethod
    def from_config(cls, config: RunConfig) -> "RuntimeScopeKey":
        return cls(
            workspace_root=str(Path(config.workspace_root).resolve()),
            lara_root=str(Path(config.lara_root).resolve()),
        )


@dataclass(frozen=True, slots=True)
class RuntimeGenerationKey:
    scope: RuntimeScopeKey
    config_fingerprint: str

    @classmethod
    def from_config(cls, config: RunConfig) -> "RuntimeGenerationKey":
        payload = config.to_dict()
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return cls(
            scope=RuntimeScopeKey.from_config(config),
            config_fingerprint=hashlib.sha256(encoded).hexdigest(),
        )


__all__ = ["RuntimeGenerationKey", "RuntimeScopeKey"]
