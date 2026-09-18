"""Stable provider extension boundary.

Provider packages implement these protocols; Runtime may own their client and
capacity lifecycles but must not interpret provider wire payloads.
"""

from ._adapter_contracts import (
    EventSink,
    MediaProjectionPlan,
    MediaProjectionTrace,
    MediaProjector,
    MediaResolver,
    MediaTransportCapabilities,
    ModelEventKind,
    ModelInvoker,
    ModelProviderAdapter,
    ModelProviderClient,
    ModelProviderMediaDeliveryValidator,
    ModelProviderMediaTokenEstimator,
    ModelProviderRequest,
    ModelProviderResponse,
    ModelProviderSpecValidator,
    ModelProviderStreamKind,
    ModelProviderStreamPart,
    ModelRequestMediaTokenEstimator,
    ModelStreamEvent,
    ProjectedMedia,
)
from ._model_execution import ModelExecution
from .catalog import ModelCatalog
from .types import ModelErrorKind, ModelProviderError

__all__ = [
    "EventSink",
    "MediaProjectionPlan",
    "MediaProjectionTrace",
    "MediaProjector",
    "MediaResolver",
    "MediaTransportCapabilities",
    "ModelCatalog",
    "ModelErrorKind",
    "ModelEventKind",
    "ModelExecution",
    "ModelInvoker",
    "ModelProviderAdapter",
    "ModelProviderClient",
    "ModelProviderError",
    "ModelProviderMediaDeliveryValidator",
    "ModelProviderMediaTokenEstimator",
    "ModelProviderRequest",
    "ModelProviderResponse",
    "ModelProviderSpecValidator",
    "ModelProviderStreamKind",
    "ModelProviderStreamPart",
    "ModelRequestMediaTokenEstimator",
    "ModelStreamEvent",
    "ProjectedMedia",
]
