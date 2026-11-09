"""Pure validation of the raw configuration mapping.

The loader in :mod:`lara_shell.config` merges defaults and file content before
calling :func:`validate_config_mapping`; this module only inspects data and
raises ``ValueError`` for unknown or malformed sections.
"""

from __future__ import annotations

from typing import Any


def validate_config_mapping(data: dict[str, Any]) -> None:
    _require_known_keys(
        data,
        {
            "max_steps",
            "session_id",
            "allow_read_outside_workspace",
            "context_window",
            "agent",
            "agents",
            "user",
            "cli",
            "context_compression",
            "runtime",
            "mcp",
            "delegation",
            "eternal_conversation",
            "scheduler",
            "connections",
            "models",
            "model_groups",
        },
        "config",
    )
    _validate_mapping(data.get("agent"), {"default_alias"}, "agent")
    _validate_mapping(data.get("user"), {"identity"}, "user")
    _validate_mapping(
        data.get("context_compression"),
        {"enabled", "trigger_ratio", "file_read_count", "file_read_max_chars"},
        "context_compression",
    )
    _validate_mapping(data.get("cli"), {"bash"}, "cli")
    cli = data.get("cli")
    if isinstance(cli, dict):
        _validate_mapping(
            cli.get("bash"), {"presets", "full_output_allowlist"}, "cli.bash"
        )
        bash = cli.get("bash")
        if isinstance(bash, dict) and isinstance(bash.get("presets"), list):
            for index, preset in enumerate(bash["presets"]):
                if isinstance(preset, dict):
                    _require_known_keys(
                        preset,
                        {"name", "command", "description"},
                        f"cli.bash.presets[{index}]",
                    )
    _validate_mapping(
        data.get("runtime"),
        {"durability", "steering", "capacity", "approvals"},
        "runtime",
    )
    runtime = data.get("runtime")
    if isinstance(runtime, dict):
        _validate_mapping(
            runtime.get("durability"), {"mode", "history_path"}, "runtime.durability"
        )
        _validate_mapping(runtime.get("steering"), {"mode"}, "runtime.steering")
        _validate_mapping(
            runtime.get("capacity"), {"scope", "coordinator_path"}, "runtime.capacity"
        )
        _validate_mapping(
            runtime.get("approvals"),
            {"enabled", "timeout_seconds", "preauthorized_tools"},
            "runtime.approvals",
        )
    _validate_mapping(data.get("mcp"), {"servers"}, "mcp")
    mcp = data.get("mcp")
    if isinstance(mcp, dict) and isinstance(mcp.get("servers"), list):
        for index, server in enumerate(mcp["servers"]):
            if isinstance(server, dict):
                _require_known_keys(
                    server,
                    {
                        "name",
                        "transport",
                        "command",
                        "args",
                        "cwd",
                        "env_from",
                        "url",
                        "headers_env",
                        "timeout",
                        "required",
                    },
                    f"mcp.servers[{index}]",
                )
    _validate_mapping(
        data.get("delegation"),
        {"allowed_agents", "max_depth", "max_parallel", "background_enabled"},
        "delegation",
    )
    _validate_mapping(
        data.get("eternal_conversation"),
        {
            "enabled",
            "extractor_agent_alias",
            "builder_agent_alias",
            "dynamic_memory_cli_path",
            "extraction_trigger",
            "extraction_trigger_ratio",
        },
        "eternal_conversation",
    )
    _validate_mapping(
        data.get("scheduler"),
        {"agent_create_enabled", "session", "max_concurrent", "poll_seconds"},
        "scheduler",
    )
    agents = data.get("agents")
    if agents is not None and not isinstance(agents, list):
        raise ValueError("agents must be a list")
    for index, agent in enumerate(agents or []):
        if not isinstance(agent, dict):
            raise ValueError(f"agents[{index}] must be a mapping")
        _require_known_keys(agent, {"alias", "model_request"}, f"agents[{index}]")
        request = agent.get("model_request")
        if not isinstance(request, dict):
            raise ValueError(f"agents[{index}].model_request must be a mapping")
        allowed_request_fields = {"default_model_group", "retry", "context_window"}
        if {"profile", "routes", "fallback"} & set(request):
            allowed_request_fields.update({"profile", "routes", "fallback"})
        _require_known_keys(
            request, allowed_request_fields, f"agents[{index}].model_request"
        )
        retry_fields = {
            "max_attempts_per_model",
            "attempt_idle_timeout_seconds",
            "backoff_initial",
            "backoff_maximum",
            "backoff_multiplier",
        }
        if {"profile", "routes", "fallback"} & set(request):
            retry_fields.add("max_attempts_per_route")
        _validate_mapping(
            request.get("retry"),
            retry_fields,
            f"agents[{index}].model_request.retry",
        )


def _validate_mapping(value: Any, allowed: set[str], path: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping")
    _require_known_keys(value, allowed, path)


def _require_known_keys(value: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{path} contains unknown fields: {', '.join(unknown)}")


__all__ = ["validate_config_mapping"]
