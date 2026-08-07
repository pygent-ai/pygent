"""Stable provider extension boundary.

Provider packages implement these protocols; Runtime may own their client and
capacity lifecycles but must not interpret provider wire payloads.
"""

from .adapter import (
    EventSink,
    ModelEventKind,
    ModelInvoker,
    ModelProviderAdapter,
    ModelProviderClient,
    ModelProviderRequest,
    ModelProviderResponse,
    ModelProviderStreamKind,
    ModelProviderStreamPart,
    ModelStreamEvent,
)
from .catalog import ModelCatalog
from .types import ModelErrorKind, ModelProviderError

__all__ = [
    "EventSink",
    "ModelCatalog",
    "ModelErrorKind",
    "ModelEventKind",
    "ModelInvoker",
    "ModelProviderAdapter",
    "ModelProviderClient",
    "ModelProviderError",
    "ModelProviderRequest",
    "ModelProviderResponse",
    "ModelProviderStreamKind",
    "ModelProviderStreamPart",
    "ModelStreamEvent",
]
