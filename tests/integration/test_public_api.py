"""Minimal Pygent 0.2 public surface used by the examples."""

import re

import pygent
from pygent import (
    AIMessage,
    Binding,
    BoundModule,
    CapacityPolicy,
    CapacityScope,
    CodeArtifactSpec,
    Context,
    ExecutionCapacityPolicy,
    ExecutionOptions,
    ExecutionPlan,
    ExponentialBackoff,
    FallbackPolicy,
    FrozenJsonObject,
    GenerationConfig,
    IdempotencyPolicy,
    JsonValueError,
    Message,
    ModelCallLayer,
    ModelErrorKind,
    ModelGroupConfig,
    ModelInvoker,
    ModelProviderAdapter,
    ModelRoute,
    Module,
    ModuleSpec,
    ReActLayer,
    RetryPolicy,
    Runtime,
    ToolAuthorizationDecision,
    ToolAuthorizationRequest,
    ToolCallLayer,
    ToolKit,
    ToolMessage,
    ToolSideEffect,
    ToolSpec,
    UserMessage,
    tool,
)


def test_minimal_example_surface_is_exported():
    assert all(
        symbol is not None
        for symbol in (
            AIMessage,
            Binding,
            BoundModule,
            CapacityPolicy,
            CapacityScope,
            CodeArtifactSpec,
            Context,
            ExecutionPlan,
            ExponentialBackoff,
            FallbackPolicy,
            GenerationConfig,
            FrozenJsonObject,
            IdempotencyPolicy,
            JsonValueError,
            Message,
            ModelCallLayer,
            ModelErrorKind,
            ModelGroupConfig,
            ModelInvoker,
            ModelProviderAdapter,
            ModelRoute,
            Module,
            ModuleSpec,
            ReActLayer,
            ExecutionOptions,
            ExecutionCapacityPolicy,
            Runtime,
            RetryPolicy,
            ToolCallLayer,
            ToolKit,
            ToolAuthorizationDecision,
            ToolAuthorizationRequest,
            ToolSideEffect,
            ToolSpec,
            ToolMessage,
            UserMessage,
            tool,
        )
    )


def test_execution_options_separate_transport_and_business_identities():
    options = ExecutionOptions(
        request_id="http-attempt-2",
        idempotency_key="submit-message-17",
    )

    assert options.request_id == "http-attempt-2"
    assert options.idempotency_key == "submit-message-17"


def test_binding_declares_execution_and_resource_capacity():
    binding = Binding(
        name="agent-service",
        execution_capacity=ExecutionCapacityPolicy(
            scope=CapacityScope.DEPLOYMENT,
            max_live_executions=16,
            max_runnable_executions=4,
            max_queue_size=8,
            max_waiters=16,
            max_child_depth=4,
            max_children_per_execution=8,
        ),
        model_capacity=CapacityPolicy.passthrough(),
        tool_capacity=CapacityPolicy.limited(
            max_concurrency=8,
            max_queue_size=16,
        ),
    )

    assert binding.tool_capacity.max_concurrency == 8
    assert binding.execution_capacity.scope is CapacityScope.DEPLOYMENT


def test_public_surface_has_no_legacy_run_names() -> None:
    assert not [name for name in dir(pygent) if re.match(r"^Run[A-Z]", name)]
    assert not hasattr(pygent, "DirectRunEvent")
