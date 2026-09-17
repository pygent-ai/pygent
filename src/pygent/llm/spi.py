"""Stable provider extension boundary.

Provider packages implement these protocols; Runtime may own their client and
capacity lifecycles but must not interpret provider wire payloads.
"""

from ._adapter_contracts import (
    EventSink,
    MediaResolver,
    ModelEventKind,
    ModelInvoker,
    ModelProviderAdapter,
    ModelProviderClient,
    ModelProviderRequest,
    ModelProviderResponse,
    ModelProviderSpecValidator,
    ModelProviderStreamKind,
    ModelProviderStreamPart,
    ModelStreamEvent,
    ToolResultContentCapabilities,
)
from ._model_execution import ModelExecution
from .catalog import ModelCatalog
from .types import ModelErrorKind, ModelProviderError

__all__ = [
    "EventSink",
    "MediaResolver",
    "ModelCatalog",
    "ModelErrorKind",
    "ModelEventKind",
    "ModelExecution",
    "ModelInvoker",
    "ModelProviderAdapter",
    "ModelProviderClient",
    "ModelProviderError",
    "ModelProviderRequest",
    "ModelProviderResponse",
    "ModelProviderSpecValidator",
    "ModelProviderStreamKind",
    "ModelProviderStreamPart",
    "ModelStreamEvent",
    "ToolResultContentCapabilities",
]
