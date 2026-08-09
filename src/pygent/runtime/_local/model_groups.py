"""Runtime-owned dynamic model-group control-plane facades."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from typing import Any

from pygent.llm import (
    FallbackPolicy,
    ModelGroupConfig,
    ModelGroupConfigurationError,
    ModelProfileSnapshot,
    ModelResourceBundle,
    ModelResourceOwnership,
    ModelResourceRef,
    ModelRoute,
)

from ..model_deployment import build_snapshot


class ModelGroupHandle:
    __slots__ = ("_runtime", "_scope_id", "requirement")

    def __init__(self, runtime: Any, scope_id: str, requirement: ModelGroupConfig) -> None:
        if not requirement.is_deferred:
            raise ValueError("ModelGroupHandle requires a deferred ModelGroupConfig")
        self._runtime = runtime
        self._scope_id = scope_id
        self.requirement = requirement

    @property
    def deployment_scope_id(self) -> str:
        return self._scope_id

    async def ensure_profile(
        self,
        *,
        profile: str,
        routes: tuple[ModelRoute, ...],
        fallback: FallbackPolicy,
        invoker: Any | None = None,
        resource_ref: ModelResourceRef | None = None,
        resource_bundle: ModelResourceBundle | None = None,
        ownership: ModelResourceOwnership = ModelResourceOwnership.BORROWED,
        make_default: bool = False,
        deadline: float,
    ) -> ModelProfileSnapshot:
        if self._runtime._closed:
            raise ModelGroupConfigurationError("Runtime is closed")
        if not isinstance(ownership, ModelResourceOwnership):
            raise TypeError("ownership must be a ModelResourceOwnership")
        if invoker is None and ownership is ModelResourceOwnership.OWNED:
            raise ValueError("OWNED ownership requires a resident invoker")
        if resource_ref is not None and resource_bundle is not None:
            raise ValueError("provide resource_ref or resource_bundle, not both")
        resources = resource_bundle
        if resource_ref is not None:
            resources = ModelResourceBundle.shared(tuple(routes), resource_ref)
        if invoker is None and resources is None:
            raise ModelGroupConfigurationError(
                "a dynamic profile requires an invoker or reconstructable resources"
            )
        snapshot = build_snapshot(
            scope_id=self._scope_id,
            requirement=self.requirement,
            profile=profile,
            routes=tuple(routes),
            fallback=fallback,
            resources=resources,
        )
        if isinstance(deadline, bool) or not isinstance(deadline, (int, float)):
            raise TypeError("deadline must be an absolute monotonic timestamp")
        key = (self._scope_id, self.requirement.name, profile, snapshot.digest)
        task = self._runtime._profile_publications.get(key)
        if task is None:
            async def publish_once() -> ModelProfileSnapshot:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("model profile configuration deadline exceeded")
                async with asyncio.timeout(remaining):
                    await self._runtime._ensure_model_store_open()
                    if resources is not None:
                        resolver = self._runtime._model_resource_resolvers.get(
                            resources.resolver_id
                        )
                        if resolver is None:
                            raise ModelGroupConfigurationError(
                                f"no model resource resolver {resources.resolver_id!r} is registered"
                            )
                        validate = getattr(resolver, "validate", None)
                        if callable(validate):
                            await validate(snapshot.model_group, resources)
                    result = await self._runtime.model_deployment_store.ensure_profile(
                        snapshot, make_default=make_default
                    )
                    if invoker is not None:
                        self._runtime._resident_model_invokers.setdefault(
                            result.snapshot_id, (invoker, ownership)
                        )
                    return result

            task = asyncio.create_task(
                publish_once(),
                name=f"pygent-profile-{self.requirement.name}-{profile}",
            )
            self._runtime._profile_publications[key] = task
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("model profile configuration deadline exceeded")
        try:
            async with asyncio.timeout(remaining):
                return await asyncio.shield(task)
        finally:
            if task.done() and self._runtime._profile_publications.get(key) is task:
                del self._runtime._profile_publications[key]

    async def set_default(self, profile: str, *, deadline: float) -> None:
        await self._runtime._await_until(
            deadline,
            self._set_default(profile),
        )

    async def _set_default(self, profile: str) -> None:
        await self._runtime._ensure_model_store_open()
        await self._runtime.model_deployment_store.set_default(
            self._scope_id, self.requirement.name, profile
        )

    async def current(self, profile: str | None = None) -> ModelProfileSnapshot:
        await self._runtime._ensure_model_store_open()
        if profile is None:
            profile = await self._runtime.model_deployment_store.default_profile(
                self._scope_id, self.requirement.name
            )
        return await self._runtime.model_deployment_store.current(
            self._scope_id, self.requirement.name, profile
        )

    async def list_profiles(self) -> tuple[str, ...]:
        await self._runtime._ensure_model_store_open()
        return await self._runtime.model_deployment_store.list_profiles(
            self._scope_id, self.requirement.name
        )

    async def retire(
        self, profile: str, *, replacement_default: str | None = None
    ) -> None:
        await self._runtime._ensure_model_store_open()
        await self._runtime.model_deployment_store.retire(
            self._scope_id,
            self.requirement.name,
            profile,
            replacement_default,
        )

    async def available_models(
        self,
        *,
        resource_ref: ModelResourceRef | None = None,
        profile: str | None = None,
    ) -> tuple[Any, ...]:
        if resource_ref is None:
            snapshot = await self.current(profile)
            if snapshot.resources is None:
                raise ModelGroupConfigurationError(
                    "profile has no reconstructable resource for catalog discovery"
                )
            resource_ref = snapshot.resources.route_resources[0][1]
        resolver = self._runtime._model_resource_resolvers.get(resource_ref.resolver_id)
        if resolver is None:
            raise ModelGroupConfigurationError(
                f"no model resource resolver {resource_ref.resolver_id!r} is registered"
            )
        list_models = getattr(resolver, "list_models", None)
        if not callable(list_models):
            raise ModelGroupConfigurationError("model resource resolver has no catalog")
        return tuple(await list_models(resource_ref))


class ModelGroupCollection:
    __slots__ = ("_requirements", "_runtime", "_scope_id")

    def __init__(
        self,
        runtime: Any,
        scope_id: str,
        requirements: Mapping[str, ModelGroupConfig] | None = None,
    ) -> None:
        self._runtime = runtime
        self._scope_id = scope_id
        self._requirements = None if requirements is None else dict(requirements)

    def get(self, requirement: ModelGroupConfig) -> ModelGroupHandle:
        if not isinstance(requirement, ModelGroupConfig) or not requirement.is_deferred:
            raise TypeError("get() requires a deferred ModelGroupConfig")
        if self._requirements is not None:
            declared = self._requirements.get(requirement.name)
            if declared is None or declared != requirement:
                raise ModelGroupConfigurationError(
                    "model group requirement is not declared by this bound graph"
                )
        return ModelGroupHandle(self._runtime, self._scope_id, requirement)


__all__ = ["ModelGroupCollection", "ModelGroupHandle"]
