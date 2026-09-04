"""Minimal Pygent 0.2 public surface used by the examples."""

import importlib
import re
import subprocess
import sys

import pytest

import pygent
from pygent import (
    AIMessage,
    Context,
    ContextCodec,
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
    ModelRoute,
    Module,
    ReActLayer,
    RecurrentModule,
    RetryPolicy,
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
from pygent.llm.spi import ModelInvoker, ModelProviderAdapter
from pygent.runtime import (
    Binding,
    BoundModule,
    CapacityPolicy,
    CapacityScope,
    ExecutionCapacityPolicy,
    ExecutionOptions,
    Runtime,
)
from pygent.runtime.plan import CodeArtifactSpec, ExecutionPlan, ModuleSpec

EXPECTED_TOP_LEVEL_API = {
    "AIMessage",
    "Agent",
    "Context",
    "ContextCodec",
    "ExponentialBackoff",
    "FallbackPolicy",
    "FrozenJsonObject",
    "GenerationConfig",
    "IdempotencyPolicy",
    "InjectionKind",
    "JsonValueError",
    "Message",
    "ModelCallError",
    "ModelCallLayer",
    "ModelCallOptions",
    "ModelCallPolicy",
    "ModelErrorKind",
    "ModelGroupConfig",
    "ModelRoute",
    "Module",
    "PygentAgent",
    "PygentAgentContext",
    "RecurrentModule",
    "Reminder",
    "ReActBudgetExceeded",
    "ReActLayer",
    "RetryPolicy",
    "ToolAuthorizationDecision",
    "ToolAuthorizationRequest",
    "ToolCall",
    "ToolCallLayer",
    "ToolDefinition",
    "ToolKit",
    "ToolMessage",
    "ToolResult",
    "ToolSideEffect",
    "ToolSpec",
    "UserMessage",
    "freeze_json",
    "freeze_json_object",
    "thaw_json",
    "tool",
}


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
            ContextCodec,
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
            RecurrentModule,
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


def test_top_level_api_is_an_exact_application_allowlist() -> None:
    assert set(pygent.__all__) == EXPECTED_TOP_LEVEL_API
    assert set(dir(pygent)) == EXPECTED_TOP_LEVEL_API
    assert "__getattr__" not in pygent.__dict__


@pytest.mark.parametrize(
    "name",
    [
        "LocalRuntime",
        "ModelInvoker",
        "SQLiteHistoryStore",
        "WorkerInvocation",
    ],
)
def test_legacy_infrastructure_exports_are_unavailable(name: str) -> None:
    assert name not in pygent.__dict__
    assert not hasattr(pygent, name)
    completed = subprocess.run(
        [sys.executable, "-c", f"from pygent import {name}"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "ImportError" in completed.stderr


@pytest.mark.parametrize(
    "module_path",
    [
        "pygent.core.module",
        "pygent.llm.adapter",
        "pygent.runtime.history",
        "pygent.runtime.worker",
    ],
)
def test_legacy_aggregation_modules_are_removed(module_path: str) -> None:
    with pytest.raises(ModuleNotFoundError, match=module_path):
        importlib.import_module(module_path)


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
