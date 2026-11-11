from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .common import _hash_json


def _always_enabled(ctx: PromptRenderContext) -> bool:
    return True


@dataclass(slots=True)
class PromptModule:
    id: str
    phase: Literal["static", "request_system"]
    type: Literal["system", "project", "runtime", "tool", "memory", "policy"]
    cache_scope: Literal["session", "request", "turn", "none"]
    order: int
    render: Callable[[PromptRenderContext], str | None]
    required: bool = False
    depends_on: tuple[str, ...] = ()
    enabled: Callable[[PromptRenderContext], bool] = _always_enabled

    @property
    def version_hash(self) -> str:
        return _hash_json(
            {
                "id": self.id,
                "phase": self.phase,
                "type": self.type,
                "cache_scope": self.cache_scope,
                "order": self.order,
                "depends_on": list(self.depends_on),
            }
        )


@dataclass(slots=True)
class RenderedPromptModule:
    id: str
    phase: Literal["static", "request_system"]
    type: str
    order: int
    cache_scope: str
    version_hash: str
    input_hash: str
    content_hash: str
    rendered_at: str
    skipped_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "phase": self.phase,
            "type": self.type,
            "order": self.order,
            "cache_scope": self.cache_scope,
            "version_hash": self.version_hash,
            "input_hash": self.input_hash,
            "content_hash": self.content_hash,
            "rendered_at": self.rendered_at,
            "skipped_reason": self.skipped_reason,
        }


@dataclass(slots=True)
class PromptRenderContext:
    session_id: str
    workspace_root: Path
    session_dir: Path
    turn_id: str | None
    projection: dict[str, Any]
    tool_names: list[str]
    request_id: str | None = None
    request_type: str = "agent_turn"
    user_pygent_root: Path | None = None
    project_pygent_root: Path | None = None
    user_skills_dir: Path | None = None
    project_skills_dir: Path | None = None


@dataclass(slots=True)
class StaticPromptResult:
    text: str
    prompt_hash: str
    modules: list[dict[str, Any]]
    created: bool = False


@dataclass(slots=True)
class PromptRequestContext:
    request_stage: Literal["before_model_request", "other"]
    request_type: Literal["agent_turn", "case_run", "summary", "evaluation"]


@dataclass(slots=True)
class PromptInjectionDecision:
    inject_dynamic: bool
    module_ids: list[str]
    reason: str
    skipped_module_ids: list[str] = field(default_factory=list)
    request_stage: Literal["before_model_request", "other"] = "before_model_request"
    decision_hash: str = ""

    def __post_init__(self) -> None:
        if not self.decision_hash:
            self.decision_hash = _hash_json(
                {
                    "inject_dynamic": self.inject_dynamic,
                    "module_ids": self.module_ids,
                    "reason": self.reason,
                    "skipped_module_ids": self.skipped_module_ids,
                    "request_stage": self.request_stage,
                }
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "inject_dynamic": self.inject_dynamic,
            "module_ids": self.module_ids,
            "reason": self.reason,
            "skipped_module_ids": self.skipped_module_ids,
            "request_stage": self.request_stage,
            "decision_hash": self.decision_hash,
        }


@dataclass(slots=True)
class ModelRequestPrompt:
    """The rendered model-request prompt; consumers read only ``text``."""

    text: str
