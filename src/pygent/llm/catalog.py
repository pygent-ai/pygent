"""Provider-neutral model catalog values and optional discovery protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """One model visible to the current provider credential."""

    id: str
    created: int | None = None
    owned_by: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("model id must be a non-empty string")
        if self.created is not None and (
            not isinstance(self.created, int)
            or isinstance(self.created, bool)
            or self.created < 0
        ):
            raise ValueError("model created must be a non-negative integer")
        if self.owned_by is not None and (
            not isinstance(self.owned_by, str) or not self.owned_by
        ):
            raise ValueError("model owned_by must be non-empty when provided")


class ModelCatalog(Protocol):
    """Optional deployment capability for listing credential-visible models."""

    async def list(
        self, *, timeout: float | None = 10.0
    ) -> tuple[ModelInfo, ...]: ...


__all__ = ["ModelCatalog", "ModelInfo"]
