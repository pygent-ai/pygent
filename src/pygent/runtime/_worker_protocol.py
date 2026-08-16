"""Stable Worker values, errors, registry, and handler contracts."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from pygent.core import (
    ExecutionEvent,
    ExecutionFailure,
    ExecutionFailureError,
    FrozenJsonObject,
    JsonValue,
)

from .api import BoundModule
from .plan import CodeArtifactSpec

MODEL_ROUTE_PROVIDER_OPTIONS_CAPABILITY = "model.route.provider-options.v1"


class WorkerProtocolError(RuntimeError):
    """Base class for remote Worker protocol failures."""


class WorkerUnavailableError(WorkerProtocolError):
    """Raised after every declared target is unavailable."""


class WorkerRemoteError(WorkerProtocolError):
    """Raised when a Worker reports a terminal execution failure."""

    def __init__(self, failure: ExecutionFailure) -> None:
        if not isinstance(failure, ExecutionFailure):
            raise TypeError("failure must be an ExecutionFailure")
        super().__init__(failure.message)
        self.failure = failure

    @property
    def kind(self) -> str:
        return self.failure.kind


class WorkerOutcomeUnknownError(WorkerRemoteError):
    """Remote owner may still commit and replay is not proven safe."""

    def __init__(self, message: str) -> None:
        super().__init__(
            ExecutionFailure(
                domain="worker",
                kind="outcome_unknown",
                message=message,
                retryable=False,
                outcome_unknown=True,
            )
        )


def _worker_failure(
    kind: str,
    message: str,
    *,
    retryable: bool = False,
    outcome_unknown: bool = False,
) -> ExecutionFailure:
    return ExecutionFailure(
        domain="worker",
        kind=kind,
        message=message,
        retryable=retryable,
        outcome_unknown=outcome_unknown,
    )


def _failure_from_exception(
    error: BaseException, *, persistence: bool = False
) -> ExecutionFailure:
    if isinstance(error, WorkerRemoteError):
        return error.failure
    if isinstance(error, ExecutionFailureError):
        return error.failure
    if isinstance(error, asyncio.TimeoutError):
        return _worker_failure("timeout", "Worker execution timed out", retryable=True)
    if persistence:
        return _worker_failure(
            "persistence_error", "Worker persistence operation failed"
        )
    return _worker_failure("worker_internal", "Worker execution failed")


def _validate_plan_identity(plan_id: str, graph_hash: str) -> None:
    if (
        not isinstance(graph_hash, str)
        or len(graph_hash) != 64
        or any(character not in "0123456789abcdef" for character in graph_hash)
    ):
        raise WorkerProtocolError("graph_hash must be a lowercase SHA-256 digest")
    if plan_id != f"sha256:{graph_hash}":
        raise WorkerProtocolError("plan_id does not identify graph_hash")


@dataclass(frozen=True, slots=True)
class WorkerTarget:
    target_id: str
    endpoint: str
    capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.target_id or not self.endpoint:
            raise ValueError("worker target_id and endpoint must be non-empty")
        capabilities = tuple(self.capabilities)
        if any(not isinstance(item, str) or not item for item in capabilities):
            raise ValueError("worker capabilities must be non-empty strings")
        if len(capabilities) != len(set(capabilities)):
            raise ValueError("worker capabilities must be unique")
        object.__setattr__(self, "capabilities", capabilities)


class WorkerRegistry:
    """Registry for already-declared logical Binding targets."""

    def __init__(self) -> None:
        self._targets: dict[str, tuple[WorkerTarget, ...]] = {}
        self._generation: dict[str, int] = {}

    def publish(self, binding_ref: str, targets: tuple[WorkerTarget, ...]) -> int:
        if not binding_ref or not targets:
            raise ValueError("binding_ref and at least one target are required")
        if len({target.target_id for target in targets}) != len(targets):
            raise ValueError("worker target ids must be unique")
        self._targets[binding_ref] = tuple(targets)
        generation = self._generation.get(binding_ref, 0) + 1
        self._generation[binding_ref] = generation
        return generation

    def resolve(
        self,
        binding_ref: str,
        *,
        required_capabilities: tuple[str, ...] = (),
    ) -> tuple[WorkerTarget, ...]:
        try:
            targets = self._targets[binding_ref]
        except KeyError as exc:
            raise KeyError(f"undeclared binding_ref {binding_ref!r}") from exc
        required = set(required_capabilities)
        eligible = tuple(
            target for target in targets if required.issubset(target.capabilities)
        )
        if not eligible:
            raise WorkerUnavailableError(
                "no Worker target satisfies the required capabilities"
            )
        return eligible

    def generation(self, binding_ref: str) -> int:
        return self._generation[binding_ref]


@dataclass(frozen=True, slots=True)
class WorkerInvocation:
    binding_ref: str
    request_id: str
    input: FrozenJsonObject
    plan_id: str
    graph_hash: str
    deadline: float | None
    required_capabilities: tuple[str, ...] = ()
    idempotency_key: str | None = None
    trace_id: str | None = None
    parent_execution_id: str | None = None
    parent_span_id: str | None = None
    attempt: int = 1
    expires_at: float | None = None
    model_calls: FrozenJsonObject = field(default_factory=FrozenJsonObject)
    model_admission_ref: str | None = None
    model_store_namespace: str | None = None

    def __post_init__(self) -> None:
        capabilities = tuple(self.required_capabilities)
        if any(not isinstance(item, str) or not item for item in capabilities):
            raise ValueError("required capabilities must be non-empty strings")
        if len(capabilities) != len(set(capabilities)):
            raise ValueError("required capabilities must be unique")
        object.__setattr__(self, "required_capabilities", capabilities)
        if not isinstance(self.model_calls, FrozenJsonObject):
            raise TypeError("model_calls must be a frozen JSON object")
        if self.model_admission_ref is not None and not self.model_admission_ref:
            raise ValueError("model_admission_ref must be non-empty")
        if self.model_store_namespace is not None and not self.model_store_namespace:
            raise ValueError("model_store_namespace must be non-empty")


@dataclass(frozen=True, slots=True)
class WorkerDeploymentManifest:
    """Resolver result binding loaded code and wire contracts to an artifact."""

    artifact: CodeArtifactSpec
    verified_digest: str
    entrypoint: Callable[[], object]
    input_schema: str
    output_schema: str
    serializer: str
    context_codecs: tuple[tuple[str, int, str, str], ...] = ()


WorkerEventSink = Callable[[ExecutionEvent], Awaitable[None]]
WorkerHandler = Callable[[WorkerInvocation, WorkerEventSink], Awaitable[JsonValue]]
WorkerArtifactResolver = Callable[[CodeArtifactSpec], BoundModule]
