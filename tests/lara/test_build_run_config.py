"""Programmatic RunConfig construction (``lara.schema.build_run_config``).

The file-free path: a complete native model mapping binds through the same
``RunConfig`` post-init rules as the shell loader, with agent resolution
validated against the bound model groups.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pygent.lara.schema import build_run_config
from pygent.lara.core.paths import project_lara_root

MODEL_CONFIG_MAPPING: dict = {
    "connections": {
        "facade": {
            "provider": "builder-scripted",
            "credential": {"none": True},
            "protocols": {
                "openai_chat_completions": {"base_url": "http://127.0.0.1:1"}
            },
        }
    },
    "models": {
        "scripted": {
            "connection": "facade",
            "model_id": "scripted",
            "protocol": "openai_chat_completions",
            "capabilities": {
                "modalities": {"input": ["text"], "output": ["text"]},
                "streaming": {"output": []},
                "tools": {"call": True, "choice": ["auto"], "parallel": False},
                "structured_output": {"json_object": False, "json_schema": False},
                "reasoning": {"supported": False, "controllable": False},
                "limits": {"context_tokens": 100_000, "max_output_tokens": 4_096},
            },
        }
    },
    "model_groups": {"default": {"models": ["scripted"]}},
}


def test_build_run_config_binds_models_and_resolves_roots(tmp_path: Path) -> None:
    user_root = tmp_path / "home" / ".lara"
    workspace = tmp_path / "workspace"

    config = build_run_config(
        workspace_root=workspace,
        model_config_mapping=MODEL_CONFIG_MAPPING,
        default_model_group="default",
        user_lara_root=user_root,
    )

    assert config.model_config is not None
    assert config.model_configuration_status == "configured"
    assert config.model_configuration_error is None
    assert config.resolved_agent is not None
    assert config.resolved_agent.alias == "default"
    assert config.resolved_agent.default_model_group == "default"
    assert config.lara_root == str(
        project_lara_root(workspace, user_root)
    )


def test_build_run_config_rejects_incomplete_mapping(tmp_path: Path) -> None:
    incomplete = {
        "connections": MODEL_CONFIG_MAPPING["connections"],
        "models": MODEL_CONFIG_MAPPING["models"],
    }

    with pytest.raises(ValueError, match="model configuration incomplete"):
        build_run_config(
            workspace_root=tmp_path,
            model_config_mapping=incomplete,
            default_model_group="default",
        )


def test_build_run_config_rejects_unknown_model_group(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown model group"):
        build_run_config(
            workspace_root=tmp_path,
            model_config_mapping=MODEL_CONFIG_MAPPING,
            default_model_group="nope",
        )


def test_build_run_config_validates_max_steps(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_steps"):
        build_run_config(
            workspace_root=tmp_path,
            model_config_mapping=MODEL_CONFIG_MAPPING,
            default_model_group="default",
            max_steps=0,
        )
