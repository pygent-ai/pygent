"""Executable checks for the external sandbox Tool SDK contract."""

from pathlib import Path

DOCS_ROOT = Path(__file__).resolve().parents[2] / "docs" / "tool"


def test_tool_principles_separate_sandbox_requirement_from_deployment() -> None:
    principles = (DOCS_ROOT / "FEATURES.md").read_text(encoding="utf-8")

    assert "**沙箱需求与部署能力分离**" in principles
    assert "不得自动授予 `tool.sandbox.<profile>` capability" in principles
    assert "必须通过同一 `ToolExecutor` 扩展缝包装" in principles
    assert "缺失时必须报告具体工具与 `tool.sandbox.<profile>`" in principles


def test_tool_sdk_pins_the_minimal_external_sandbox_adapter() -> None:
    sdk = (DOCS_ROOT / "SDK.md").read_text(encoding="utf-8")

    assert "class E2BWorkspaceExecutor" in sdk
    assert "SandboxExecutorSupport(" in sdk
    assert "runtime.register_tool(spec, E2BWorkspaceExecutor(e2b_sandboxes))" in sdk
    assert "参考实现已经提供 `SandboxExecutorSupport`" in sdk
    assert "只有真正申请 durable detach 时才要求" in sdk
    assert "ToolSpec 不得为 Runtime 自动授予 `tool.sandbox.<profile>`" in sdk
    assert "error_code: missing_sandbox_capability" in sdk
    assert "并发调用不依赖全局“当前沙箱”" in sdk
