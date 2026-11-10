from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pygent.agent.lara.core.paths import project_lara_root as resolve_project_lara_root
from pygent.agent.lara.runtime.context import LaraContext
from pygent.agent.lara.tracing import EventStore

from .common import (
    _file_lock,
    _hash_json,
    _hash_text,
    _now,
    _write_json_atomic,
    _write_text_atomic,
)
from .prompt_models import (
    ModelRequestPrompt,
    PromptInjectionDecision,
    PromptModule,
    PromptRenderContext,
    PromptRequestContext,
    RenderedPromptModule,
    StaticPromptResult,
)
from .prompt_sources import _prompt_render_context_payload


class PromptRegistry:
    def __init__(self) -> None:
        # Mechanism only: the registry ships empty. The composition root
        # injects prompt modules on top of RunConfig.system_prompt; the
        # kernel carries no built-in prompt content.
        self._modules: list[PromptModule] = []

    def register(self, module: PromptModule) -> None:
        """Composition-root injection: add one prompt module by stable id."""

        if any(existing.id == module.id for existing in self._modules):
            raise ValueError(f"prompt module {module.id!r} is already registered")
        self._modules.append(module)

    def resolve(
        self,
        *,
        phase: Literal["static", "request_system"] | None = None,
        include: list[str] | None = None,
        render_context: PromptRenderContext | None = None,
    ) -> list[PromptModule]:
        include_set = set(include) if include is not None else None
        modules = []
        for module in self._modules:
            if phase is not None and module.phase != phase:
                continue
            if include_set is not None and module.id not in include_set:
                continue
            if render_context is not None and not module.enabled(render_context):
                continue
            modules.append(module)
        return sorted(modules, key=lambda module: (module.order, module.id))


class PromptComposer:
    def __init__(self, registry: PromptRegistry | None = None) -> None:
        self.registry = registry or PromptRegistry()

    def resolve_request_system_modules(
        self,
        ctx: PromptRenderContext,
    ) -> list[PromptModule]:
        return self.registry.resolve(phase="request_system", render_context=ctx)

    def compose_static(
        self, ctx: PromptRenderContext
    ) -> tuple[str, list[dict[str, Any]]]:
        modules = self.registry.resolve(phase="static", render_context=ctx)
        return self._compose_modules(ctx, modules)

    def compose_request_system(
        self,
        ctx: PromptRenderContext,
        *,
        module_ids: list[str] | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        modules = self.registry.resolve(
            phase="request_system", include=module_ids, render_context=ctx
        )
        return self._compose_modules(ctx, modules)

    def _compose_modules(
        self,
        ctx: PromptRenderContext,
        modules: list[PromptModule],
    ) -> tuple[str, list[dict[str, Any]]]:
        parts: list[str] = []
        rendered_modules: list[dict[str, Any]] = []
        render_input_hash = _hash_json(_prompt_render_context_payload(ctx))
        for module in modules:
            text = module.render(ctx)
            if not text:
                if module.required:
                    raise ValueError(
                        f"Required prompt module {module.id!r} rendered empty content"
                    )
                continue
            parts.append(text)
            rendered_modules.append(
                RenderedPromptModule(
                    id=module.id,
                    phase=module.phase,
                    type=module.type,
                    order=module.order,
                    cache_scope=module.cache_scope,
                    version_hash=module.version_hash,
                    input_hash=render_input_hash,
                    content_hash=_hash_text(text),
                    rendered_at=_now(),
                ).to_dict()
            )
        return "\n\n".join(parts), rendered_modules


class StaticPromptSessionCache:
    def __init__(self, session_dir: Path, composer: PromptComposer) -> None:
        self.session_dir = session_dir
        self.composer = composer
        self.prompt_dir = session_dir / "context" / "prompts"
        self.text_path = self.prompt_dir / "static_prompt.txt"
        self.metadata_path = self.prompt_dir / "static_prompt.json"
        self.lock_path = session_dir / "state" / "prompt_cache.lock"

    def get_or_create(self, ctx: PromptRenderContext) -> StaticPromptResult:
        cached = self._read_cached()
        if cached is not None:
            return cached

        with _file_lock(self.lock_path):
            cached = self._read_cached()
            if cached is not None:
                return cached

            static_ctx = PromptRenderContext(
                session_id=ctx.session_id,
                workspace_root=ctx.workspace_root,
                session_dir=ctx.session_dir,
                turn_id=None,
                projection={},
                tool_names=[],
                request_id=None,
                request_type=ctx.request_type,
                user_lara_root=ctx.user_lara_root,
                project_lara_root=ctx.project_lara_root,
                user_skills_dir=ctx.user_skills_dir,
                project_skills_dir=ctx.project_skills_dir,
            )
            text, modules = self.composer.compose_static(static_ctx)
            prompt_hash = _hash_text(text)
            metadata = {
                "session_id": ctx.session_id,
                "created_at": _now(),
                "prompt_hash": prompt_hash,
                "module_ids": [module["id"] for module in modules],
                "modules": modules,
                "registry_version": _hash_json(
                    [
                        {"id": module["id"], "version_hash": module["version_hash"]}
                        for module in modules
                    ]
                ),
                "cache_status": "ready",
            }
            _write_text_atomic(self.text_path, text)
            _write_json_atomic(self.metadata_path, metadata)
            return StaticPromptResult(
                text=text,
                prompt_hash=prompt_hash,
                modules=modules,
                created=True,
            )

    def _read_cached(self) -> StaticPromptResult | None:
        text_exists = self.text_path.exists()
        metadata_exists = self.metadata_path.exists()
        if not text_exists and not metadata_exists:
            return None
        if text_exists != metadata_exists:
            raise RuntimeError(
                f"Incomplete static prompt cache under {self.prompt_dir}"
            )

        text = self.text_path.read_text(encoding="utf-8")
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if metadata.get("cache_status") != "ready":
            raise RuntimeError(
                f"Static prompt cache is not ready: {self.metadata_path}"
            )
        prompt_hash = _hash_text(text)
        if metadata.get("prompt_hash") != prompt_hash:
            raise RuntimeError(
                f"Static prompt cache hash mismatch: {self.metadata_path}"
            )
        return StaticPromptResult(
            text=text,
            prompt_hash=prompt_hash,
            modules=list(metadata.get("modules") or []),
            created=False,
        )


class PromptInjectionPolicy:
    def decide(
        self,
        *,
        request_context: PromptRequestContext,
        dynamic_modules: list[PromptModule],
    ) -> PromptInjectionDecision:
        # Production only reaches this policy at the before-model-request
        # stage of an agent turn, and the request-system selection is always
        # the full registry set, so the decision is fixed.
        return PromptInjectionDecision(
            inject_dynamic=True,
            module_ids=[module.id for module in dynamic_modules],
            reason="before_model_request",
            request_stage=request_context.request_stage,
        )


class AgentContextManager:
    def __init__(
        self,
        *,
        session_dir: Path,
        workspace_root: Path,
        store: EventStore | None = None,
        prompt_registry: PromptRegistry | None = None,
        prompt_composer: PromptComposer | None = None,
        system_prompt: str = "",
        user_lara_root: Path | None = None,
        project_lara_root: Path | None = None,
    ) -> None:
        self.session_dir = session_dir
        self.workspace_root = workspace_root
        # The composition root's configured static system prompt; the prompt
        # registry ships empty and only carries injected modules on top.
        self.system_prompt = system_prompt
        self.user_lara_root = (
            (user_lara_root or Path.home() / ".lara").expanduser().resolve()
        )
        self.project_lara_root = (
            (
                project_lara_root
                or resolve_project_lara_root(workspace_root, self.user_lara_root)
            )
            .expanduser()
            .resolve()
        )
        self.project_skills_dir = (
            (self.project_lara_root / "skills").expanduser().resolve()
        )
        self.user_skills_dir = (self.user_lara_root / "skills").expanduser().resolve()
        self.store = store
        self.prompt_composer = prompt_composer or PromptComposer(prompt_registry)
        self.static_prompt_cache = StaticPromptSessionCache(
            session_dir, self.prompt_composer
        )
        self.injection_policy = PromptInjectionPolicy()

    def projection(
        self, history: list[dict[str, Any]], limit: int = 8
    ) -> dict[str, Any]:
        recent_messages = [
            {
                "role": message.get("role"),
                "content": str(message.get("content", ""))[:1200],
            }
            for message in history[-limit:]
        ]
        return {"recent_messages": recent_messages}

    def build_model_request_prompt(
        self,
        *,
        context: LaraContext,
        tool_names: list[str],
        request_type: Literal[
            "agent_turn", "case_run", "summary", "evaluation"
        ] = "agent_turn",
    ) -> ModelRequestPrompt:
        projection = self.projection(context.history)
        if self.store is not None:
            self.store.append(
                "context.projection_created",
                actor="system",
                payload={
                    "message_count": len(context.history),
                    "projection": projection,
                },
                turn_id=context.turn_id,
            )
        request_id = f"{context.session_id}:{context.turn_id or 'unknown'}:{len(context.history)}"
        dynamic_inputs = {
            "workspace_root": str(self.workspace_root),
            "session_id": context.session_id,
            "turn_id": context.turn_id,
            "tool_names": tool_names,
            "projection": projection,
            "request_type": request_type,
        }
        render_ctx = PromptRenderContext(
            session_id=context.session_id,
            workspace_root=self.workspace_root,
            session_dir=self.session_dir,
            turn_id=context.turn_id,
            projection=projection,
            tool_names=tool_names,
            request_id=request_id,
            request_type=request_type,
            user_lara_root=self.user_lara_root,
            project_lara_root=self.project_lara_root,
            user_skills_dir=self.user_skills_dir,
            project_skills_dir=self.project_skills_dir,
        )
        static_prompt = self.static_prompt_cache.get_or_create(render_ctx)
        request_system_modules = self.prompt_composer.resolve_request_system_modules(
            render_ctx
        )
        request_context = PromptRequestContext(
            request_stage="before_model_request",
            request_type=request_type,
        )
        decision = self.injection_policy.decide(
            request_context=request_context,
            dynamic_modules=request_system_modules,
        )

        request_system_text: str | None = None
        request_system_prompt_hash: str | None = None
        request_system_rendered_modules: list[dict[str, Any]] = []
        if decision.inject_dynamic:
            request_system_text, request_system_rendered_modules = (
                self.prompt_composer.compose_request_system(
                    render_ctx,
                    module_ids=decision.module_ids,
                )
            )
            request_system_prompt_hash = (
                _hash_text(request_system_text) if request_system_text else None
            )

        prompt_parts = [self.system_prompt, static_prompt.text]
        if request_system_text:
            prompt_parts.append(request_system_text)
        prompt = "\n\n".join(part for part in prompt_parts if part)
        if not prompt.strip():
            raise ValueError(
                "system prompt is empty: configure RunConfig.system_prompt "
                "or inject prompt modules via the composition root"
            )
        prompt_hash = _hash_text(prompt)
        modules = [*static_prompt.modules, *request_system_rendered_modules]
        model_prompt = ModelRequestPrompt(text=prompt)
        if self.store is not None:
            self.store.append(
                "prompt.rendered",
                actor="system",
                payload={
                    "prompt": prompt,
                    "module_ids": [module["id"] for module in modules],
                    "static_module_ids": [
                        module["id"] for module in static_prompt.modules
                    ],
                    "dynamic_module_ids": [],
                    "request_system_module_ids": [
                        module["id"] for module in request_system_rendered_modules
                    ],
                    "modules": modules,
                    "prompt_hash": prompt_hash,
                    "static_prompt_hash": static_prompt.prompt_hash,
                    "dynamic_prompt_hash": None,
                    "request_system_prompt_hash": request_system_prompt_hash,
                    "static_prompt_created": static_prompt.created,
                    "injection_decision": decision.to_dict(),
                    "dynamic_inputs": dynamic_inputs,
                },
                turn_id=context.turn_id,
            )
        return model_prompt
