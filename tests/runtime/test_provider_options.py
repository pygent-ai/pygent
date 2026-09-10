from __future__ import annotations

import json
import sqlite3
import time
from contextlib import asynccontextmanager
from typing import cast

import pytest

from pygent import (
    Context,
    GenerationConfig,
    ModelCallLayer,
    ModelEntry,
    ModelGroup,
    RetryPolicy,
    UserMessage,
)
from pygent.core import FrozenJsonObject, freeze_json_object
from pygent.llm import (
    ModelDeploymentUnavailableError,
    ModelGroupConfigurationError,
    ModelResourceRef,
)
from pygent.llm._model_spec_codec import model_entry_value
from pygent.llm.layer import _model_effect_request
from pygent.runtime import (
    LocalRuntime,
    SQLiteModelDeploymentStore,
    compile_execution_plan,
)
from pygent.runtime._worker_protocol import (
    MODEL_PROVIDER_OPTIONS_CAPABILITY,
    WorkerEventSink,
    WorkerInvocation,
    WorkerRemoteError,
)
from pygent.runtime.model_deployment import (
    _snapshot_from_value,
    _snapshot_value,
    build_snapshot,
)
from pygent.runtime.worker_server import HTTPWorkerApp
from pygent.runtime.worker_target import _validate_worker_model_admission
from tests.support.model_specs import model_entry


def _requirement() -> ModelGroup:
    return ModelGroup.deferred(name="assistant")


def _layer(group: ModelGroup) -> ModelCallLayer:
    return ModelCallLayer(
        model_group=group,
        retry_policy=RetryPolicy(),
        generation=GenerationConfig(),
    )


class _ValidatingInvoker:
    def __init__(self) -> None:
        self.supported = True
        self.validations = 0

    def validate_model(self, model: ModelEntry) -> None:
        self.validations += 1
        if not self.supported:
            raise ValueError(f"unsupported model {model.name}")


def test_provider_options_change_definition_and_effect_identity_but_empty_does_not() -> None:
    empty_layer = _layer(
        ModelGroup(
            "assistant",
            (model_entry("primary", "openai", "gpt-5"),),
        )
    )
    configured_layer = _layer(
        ModelGroup(
            "assistant",
            (
                model_entry(
                    "primary",
                    "openai",
                    "gpt-5",
                    provider_options={"vendor_feature": True},
                ),
            ),
        )
    )
    assert (
        compile_execution_plan(empty_layer).modules[0].config_ref
        == "sha256:0a8f35c101079ea63a1e1b27de4889d8e2ab19c14bc30771625adc316cf22098"
    )
    assert (
        compile_execution_plan(empty_layer).modules[0].config_ref
        != compile_execution_plan(configured_layer).modules[0].config_ref
    )

    empty_effect = _model_effect_request(
        empty_layer,
        UserMessage(content="hello"),
        Context(),
        (),
    )
    configured_effect = _model_effect_request(
        configured_layer,
        UserMessage(content="hello"),
        Context(),
        (),
    )
    effect_group = cast(FrozenJsonObject, empty_effect["model_group"])
    effect_models = cast(tuple[object, ...], effect_group["models"])
    empty_model = cast(FrozenJsonObject, effect_models[0])
    empty_spec = cast(FrozenJsonObject, empty_model["spec"])
    assert empty_spec["provider_options"] == freeze_json_object({})
    assert empty_effect["retry"] == freeze_json_object(
        {
            "max_attempts_per_route": 2,
            "retry_on": [
                "timeout",
                "rate_limit",
                "unavailable",
                "incomplete_response",
            ],
            "attempt_idle_timeout_seconds": None,
            "backoff": {
                "initial": 0.0,
                "maximum": 0.0,
                "multiplier": 2.0,
            },
        }
    )
    assert empty_effect != configured_effect


def test_model_entry_codec_emits_the_current_complete_shape() -> None:
    model = model_entry("primary", "openai", "gpt-5")

    assert model_entry_value(model)["spec"]["provider_options"] == {}  # type: ignore[index]


@pytest.mark.asyncio
async def test_dynamic_profile_validates_before_digest_and_key_order_is_stable() -> None:
    requirement = _requirement()
    runtime = LocalRuntime()
    bound = runtime.bind(_layer(requirement))
    handle = bound.model_groups.get(requirement)

    with pytest.raises(ModelGroupConfigurationError):
        await handle.ensure_profile(
            profile="invalid",
            models=(
                model_entry(
                    "main",
                    "custom",
                    "model",
                    provider_options={"feature": True},
                ),
            ),
            invoker=object(),
            deadline=time.monotonic() + 2,
        )

    invoker = _ValidatingInvoker()
    first = await handle.ensure_profile(
        profile="quality",
        models=(
            model_entry(
                "main",
                "custom",
                "model",
                provider_options={"outer": {"a": 1, "b": 2}},
            ),
        ),
        invoker=invoker,
        deadline=time.monotonic() + 2,
    )
    second = await handle.ensure_profile(
        profile="quality",
        models=(
            model_entry(
                "main",
                "custom",
                "model",
                provider_options={"outer": {"b": 2, "a": 1}},
            ),
        ),
        invoker=invoker,
        deadline=time.monotonic() + 2,
    )
    assert first.digest == second.digest
    assert first.snapshot_id == second.snapshot_id
    assert invoker.validations == 2

    invoker.supported = False
    with pytest.raises(ModelDeploymentUnavailableError):
        async with runtime._model_deployment_lease(first):
            pass
    await runtime.close()


@pytest.mark.asyncio
async def test_sqlite_round_trip_and_tampered_provider_options_fail_closed(
    tmp_path,
) -> None:
    path = tmp_path / "models.sqlite3"
    store = await SQLiteModelDeploymentStore(path).open()
    requirement = _requirement()
    snapshot = build_snapshot(
        scope_id="scope",
        requirement=requirement,
        profile="quality",
        models=(
            model_entry(
                "main",
                "deepseek",
                "deepseek-chat",
                provider_options={"thinking": {"type": "disabled"}},
            ),
        ),
        resources=None,
    )
    await store.ensure_profile(snapshot, make_default=True)
    admission = await store.admit("scope", ("assistant",), {})
    await store.release_admission(admission.admission_id, recoverable=True)
    await store.close()

    restored = await SQLiteModelDeploymentStore(path).open()
    current = await restored.current("scope", "assistant", "quality")
    assert current.model_group.models[0].spec.provider_options["thinking"]["type"] == "disabled"  # type: ignore[index]
    assert await restored.get_admission(admission.admission_id) is not None
    await restored.close()

    connection = sqlite3.connect(path)
    payload = json.loads(
        connection.execute(
            "SELECT snapshot_json FROM pygent_model_profiles"
        ).fetchone()[0]
    )
    payload["model_group"]["models"][0]["spec"]["provider_options"]["thinking"][
        "type"
    ] = "enabled"
    connection.execute(
        "UPDATE pygent_model_profiles SET snapshot_json=?",
        (json.dumps(payload, separators=(",", ":"), sort_keys=True),),
    )
    connection.commit()
    connection.close()

    with pytest.raises(ValueError, match="digest"):
        await SQLiteModelDeploymentStore(path).open()


def test_legacy_profile_shape_is_rejected_without_compatibility_parsing() -> None:
    snapshot = build_snapshot(
        scope_id="scope",
        requirement=_requirement(),
        profile="legacy",
        models=(model_entry("main", "deepseek", "deepseek-v4-flash"),),
        resources=None,
    )
    value = _snapshot_value(snapshot)
    value["model_group"] = {
        "name": "assistant",
        "routes": [],
        "fallback": [],
        "resolution": "concrete",
    }

    with pytest.raises(ValueError, match="current schema"):
        _snapshot_from_value(value)


class _Resolver:
    resolver_id = "resolver"
    coordinator_domain = "domain"

    async def validate(self, model_group: object, resources: object) -> None:
        del model_group, resources

    @asynccontextmanager
    async def acquire(self, model_group: object, resources: object):
        del model_group, resources
        yield _ValidatingInvoker()


@pytest.mark.asyncio
async def test_worker_requires_provider_options_capability_at_admission() -> None:
    requirement = _requirement()
    runtime = LocalRuntime()
    resolver = _Resolver()
    runtime.register_model_resource_resolver(resolver)
    bound = runtime.bind(_layer(requirement))
    handle = bound.model_groups.get(requirement)
    ref = ModelResourceRef(
        resolver_id="resolver",
        resource_id="deepseek",
        revision="v1",
        capacity_owner_id="deepseek",
        coordinator_domain="domain",
    )
    await handle.ensure_profile(
        profile="quality",
        models=(
            model_entry(
                "main",
                "deepseek",
                "deepseek-chat",
                provider_options={"thinking": {"type": "disabled"}},
            ),
        ),
        resource_ref=ref,
        make_default=True,
        deadline=time.monotonic() + 2,
    )
    request_id = "provider-options-request"
    request = WorkerInvocation(
        binding_ref="assistant",
        request_id=request_id,
        input=freeze_json_object({}),
        plan_id=bound.plan.plan_id,
        graph_hash=bound.plan.graph_hash,
        deadline=time.monotonic() + 2,
        required_capabilities=("model.deferred.exact-pin.v1",),
        model_admission_ref=request_id,
        model_store_namespace=runtime.model_deployment_store.namespace_id,
    )
    with pytest.raises(WorkerRemoteError, match="provider options"):
        await _validate_worker_model_admission(bound, request)

    async def handler(
        invocation: WorkerInvocation, event_sink: WorkerEventSink
    ) -> FrozenJsonObject:
        del invocation, event_sink
        return freeze_json_object({})

    worker = HTTPWorkerApp(handler, model_store_namespace="store:test")
    assert MODEL_PROVIDER_OPTIONS_CAPABILITY in worker.capabilities
    await worker.close()
    await runtime.close()
