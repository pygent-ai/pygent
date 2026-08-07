"""Contract checks for the service example and framework skeleton."""

from __future__ import annotations

import inspect
from pathlib import Path

from examples.service import CoordinatorAgent, build_agent
from examples.service.agents import ReviewAgent
from examples.service.models import build_assistant_model
from examples.service.tools import WeatherAuthorization, build_tool_layer
from pygent import (
    ModelCallLayer,
    ModelErrorKind,
    Module,
    ReActLayer,
    Runtime,
    ToolCallLayer,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_example_builds_a_user_authored_module_graph():
    agent = build_agent()

    assert isinstance(agent, CoordinatorAgent)
    assert isinstance(agent, Module)
    assert [name for name, _ in agent.named_children()] == ["react", "reviewer"]
    assert isinstance(agent.react, ReActLayer)
    assert isinstance(agent.react.model, ModelCallLayer)
    assert isinstance(agent.react.tools, ToolCallLayer)
    assert agent.react.model.model_group.name == "assistant"
    assert agent.react.model.model_group.fallback.order == (
        "assistant-primary",
        "assistant-fallback",
    )
    assert agent.react.model.retry_policy.retry_on == (
        ModelErrorKind.TIMEOUT,
        ModelErrorKind.RATE_LIMIT,
        ModelErrorKind.UNAVAILABLE,
    )
    assert agent.react.model.generation.max_output_tokens == 2048
    assert agent.react.max_steps == 4
    assert agent.react.max_model_calls == 4
    assert agent.react.max_tool_calls == 16
    assert isinstance(agent.react.tools.authorization, WeatherAuthorization)
    assert agent.reviewer.model.model_group.name == "reviewer"


def test_user_agent_defines_only_the_forward_business_entrypoint():
    methods = {
        name
        for name, value in CoordinatorAgent.__dict__.items()
        if inspect.isfunction(value)
    }
    assert methods == {"__init__", "forward"}


def test_example_modules_use_keyword_only_initialization():
    tool_parameters = inspect.signature(ToolCallLayer).parameters
    react_parameters = inspect.signature(ReActLayer).parameters
    review_parameters = inspect.signature(ReviewAgent).parameters
    coordinator_parameters = inspect.signature(CoordinatorAgent).parameters

    assert tool_parameters["tools"].kind is inspect.Parameter.KEYWORD_ONLY
    assert tool_parameters["max_concurrency"].kind is (inspect.Parameter.KEYWORD_ONLY)
    assert react_parameters["model"].kind is inspect.Parameter.KEYWORD_ONLY
    assert react_parameters["tools"].kind is inspect.Parameter.KEYWORD_ONLY
    assert react_parameters["max_steps"].kind is inspect.Parameter.KEYWORD_ONLY
    assert react_parameters["max_model_calls"].kind is (inspect.Parameter.KEYWORD_ONLY)
    assert react_parameters["max_tool_calls"].kind is (inspect.Parameter.KEYWORD_ONLY)
    assert review_parameters["model"].kind is inspect.Parameter.KEYWORD_ONLY
    assert coordinator_parameters["react"].kind is (inspect.Parameter.KEYWORD_ONLY)
    assert coordinator_parameters["reviewer"].kind is (inspect.Parameter.KEYWORD_ONLY)


def test_react_preserves_a_shared_model_definition_instance():
    model = build_assistant_model()
    react = ReActLayer(model=model, tools=build_tool_layer())
    root = Module()
    root.model = model
    root.react = react

    assert react.model is model
    assert dict(root.named_children())["model"] is model
    assert dict(root.named_children())["react"] is react
    assert dict(react.named_children())["model"] is model


def test_example_uses_inherited_events_and_current_sdk_names():
    service_root = REPO_ROOT / "examples" / "service"
    agent_source = (service_root / "agents.py").read_text("utf-8")
    model_source = (service_root / "models.py").read_text("utf-8")
    app_source = (service_root / "app.py").read_text("utf-8")
    tool_source = (service_root / "tools.py").read_text("utf-8")

    assert "await self.emit(" in agent_source
    assert "ModelGroupConfig(" not in agent_source
    assert "RetryPolicy(" not in agent_source
    assert "GenerationConfig(" not in agent_source
    assert "ModelGroupConfig(" in model_source
    assert "RetryPolicy(" in model_source
    assert "GenerationConfig(" in model_source
    assert "ModelConfig" not in model_source
    assert "deadline=monotonic() + 60.0" in app_source
    assert "ExecutionCapacityPolicy(" in app_source
    assert "authorization=WeatherAuthorization()" in tool_source
    assert "max_concurrency=8" not in tool_source


def test_example_separates_declarations_from_agent_dataflow():
    service_root = REPO_ROOT / "examples" / "service"

    assert (service_root / "models.py").is_file()
    assert (service_root / "tools.py").is_file()
    assert "ToolDefinition(" not in (service_root / "agents.py").read_text("utf-8")
    assert "ToolDefinition(" in (service_root / "tools.py").read_text("utf-8")


def test_runtime_interface_has_a_public_local_reference_implementation():
    from pygent import LocalRuntime

    assert inspect.isclass(Runtime)
    assert inspect.isclass(LocalRuntime)
    assert hasattr(LocalRuntime, "bind")
    assert hasattr(LocalRuntime, "create_binding")


def test_example_imports_production_package_not_a_second_framework():
    service_root = REPO_ROOT / "examples" / "service"
    source_root = REPO_ROOT / "src" / "pygent"

    assert not tuple((REPO_ROOT / "examples" / "framework").glob("*.py"))
    assert not [
        path
        for path in service_root.glob("*.py")
        if "examples.framework" in path.read_text("utf-8")
    ]
    assert not [
        path
        for path in source_root.rglob("*.py")
        if "examples.service" in path.read_text("utf-8")
    ]
