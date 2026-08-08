"""Stable provider extension boundary.

Provider packages implement these protocols; Runtime may own their client and
capacity lifecycles but must not interpret provider wire payloads.
"""

from ._adapter_contracts import (
    EventSink,
    ModelEventKind,
    ModelInvoker,
    ModelProviderAdapter,
    ModelProviderCapabilities,
    ModelProviderClient,
    ModelProviderRequest,
    ModelProviderResponse,
    ModelProviderStreamKind,
    ModelProviderStreamPart,
    ModelStreamEvent,
)
from ._model_execution import ModelExecution
from .catalog import ModelCatalog
from .types import ModelErrorKind, ModelProviderError

__all__ = [
    "EventSink",
    "ModelCatalog",
    "ModelErrorKind",
    "ModelEventKind",
    "ModelExecution",
    "ModelInvoker",
    "ModelProviderAdapter",
    "ModelProviderCapabilities",
    "ModelProviderClient",
    "ModelProviderError",
    "ModelProviderRequest",
    "ModelProviderResponse",
    "ModelProviderStreamKind",
    "ModelProviderStreamPart",
    "ModelStreamEvent",
]
