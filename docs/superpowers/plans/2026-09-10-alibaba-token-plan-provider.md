# Alibaba Cloud Token Plan Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the `aliyun_token_plan` Provider, its complete 18-model capability catalog, multimodal capability values, strict Alibaba OpenAI options, and protocol-driven reasoning continuation, then verify every catalogued model.

**Architecture:** Keep Provider connection presets, `(provider, model_id, protocol)` capability records, open protocol identifiers, and executable Adapters independent. Generalize the existing capability value objects for output modalities and nullable token limits, reuse the existing OpenAI Chat Completions and Anthropic Messages execution paths, and leave dedicated image/video/audio protocols catalog-only. A separate explicit live verifier exercises every official Token Plan model without presenting those dedicated protocols as Pygent Adapters.

**Tech Stack:** Python 3.11 immutable dataclasses, JSON package resources, asyncio, httpx, WebSocket/HTTP provider probes, pytest, Ruff, mypy, uv, Maturin, Twine.

**Design:** `docs/llm/ALIYUN_TOKEN_PLAN_PROVIDER_PROPOSAL.md`

**Baseline:** `987 passed`; Ruff, mypy, build, Twine, and the existing DeepSeek OpenAI/Anthropic live smoke were green at commit `2c5288e`.

---

## File structure

- Modify `src/pygent/llm/configuration.py`: closed modality values, output-modality streaming, nullable token limits, strict mapping and serialization.
- Modify `src/pygent/llm/catalogs.py`: parse the new capability representation while keeping text presets explicit.
- Modify `src/pygent/llm/invoker.py`: select text streaming by output modality and skip unknown output-limit comparisons.
- Modify `src/pygent/llm/openai_compatible.py`: Alibaba option validation and response-driven `reasoning_content` continuation.
- Modify `src/pygent/llm/data/providers.json`: Token Plan OpenAI and Anthropic connection presets.
- Modify `src/pygent/llm/data/model_capabilities.json`: migrate existing records and add the 27 Token Plan records.
- Modify `src/pygent/llm/data/capability_presets.json`: migrate the four existing text presets to `streaming.output`.
- Modify focused LLM/runtime/integration tests and shared model fixtures for the born-as-new capability projection.
- Create `tests/live/aliyun_token_plan_probe.py`: explicit, non-pytest live verification for all 18 model IDs.
- Create `tests/live/test_aliyun_token_plan_probe.py`: offline tests for probe inventory, request projection, redaction, and result accounting.
- Modify `docs/llm/FEATURES.md`, `docs/llm/README.md`, `docs/llm/SDK.md`, and affected model architecture documents/examples.

### Task 1: Generalize the immutable capability values

**Files:**
- Modify: `tests/llm/test_model_config.py`
- Modify: `src/pygent/llm/configuration.py`

- [ ] **Step 1: Write failing tests for closed modalities and output streaming**

Add tests using the intended public constructors and Mapping shape:

```python
def test_streaming_output_is_an_immutable_output_modality_subset() -> None:
    capabilities = ModelCapabilities.from_mapping(
        {
            "modalities": {"input": ["text", "image"], "output": ["text", "audio"]},
            "streaming": {"output": ["text", "audio"]},
            "tools": {"call": False, "choice": [], "parallel": False},
            "structured_output": {"json_object": False, "json_schema": False},
            "reasoning": {"supported": False, "controllable": False},
            "limits": {"context_tokens": None, "max_output_tokens": None},
        }
    )
    assert capabilities.streaming.output == ("text", "audio")
    assert capabilities.to_mapping()["streaming"] == {"output": ["text", "audio"]}


@pytest.mark.parametrize("field", ["input", "output"])
def test_modalities_reject_unknown_values(field: str) -> None:
    value = _capabilities()
    value["modalities"][field] = ["text", "embedding"]
    with pytest.raises(ValueError, match="unsupported modalities"):
        ModelCapabilities.from_mapping(value)


def test_streaming_output_must_be_output_modality_subset() -> None:
    value = _capabilities()
    value["modalities"]["output"] = ["text"]
    value["streaming"] = {"output": ["audio"]}
    with pytest.raises(ValueError, match="streaming.output"):
        ModelCapabilities.from_mapping(value)


def test_old_streaming_text_shape_is_rejected() -> None:
    value = _capabilities()
    value["streaming"] = {"text": True}
    with pytest.raises(ValueError, match="streaming"):
        ModelCapabilities.from_mapping(value)
```

- [ ] **Step 2: Run the tests and confirm RED**

Run: `uv run pytest -q tests/llm/test_model_config.py`

Expected: FAIL because `ModelStreamingCapabilities` still accepts `text: bool`, modalities are open strings, and limits reject `None`.

- [ ] **Step 3: Implement the minimal born-as-new values**

In `configuration.py`, define and use:

```python
_MODEL_MODALITIES = frozenset({"text", "image", "audio", "video"})


def _optional_positive_int(value: object, label: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, label)


@dataclass(frozen=True, slots=True)
class ModelStreamingCapabilities:
    output: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "output", _modalities(self.output, "streaming.output"))


@dataclass(frozen=True, slots=True)
class ModelLimits:
    context_tokens: int | None
    max_output_tokens: int | None
```

Validate both modality collections against `_MODEL_MODALITIES`; after constructing `ModelCapabilities`, reject `set(streaming.output) - set(modalities.output)`. Serialize only `{"output": [...]}` and preserve explicit null limits.

- [ ] **Step 4: Run focused tests and confirm GREEN**

Run: `uv run pytest -q tests/llm/test_model_config.py`

Expected: PASS.

- [ ] **Step 5: Commit the capability contract**

```text
git add src/pygent/llm/configuration.py tests/llm/test_model_config.py
git commit -m "refactor(llm): model streaming as output modalities"
```

### Task 2: Migrate consumers, codecs, fixtures, and presets

**Files:**
- Modify: `tests/llm/test_model_spec_execution.py`
- Modify: `tests/llm/test_builtin_catalogs.py`
- Modify: `tests/support/model_specs.py`
- Modify: `tests/llm/test_model_config.py`
- Modify: `benchmarks/models.py`
- Modify: `examples/live_agent/agent.py`
- Modify: `src/pygent/llm/invoker.py`
- Modify: `src/pygent/llm/catalogs.py`
- Modify: `src/pygent/llm/data/capability_presets.json`
- Modify: `src/pygent/llm/data/model_capabilities.json`

- [ ] **Step 1: Write failing Invoker tests for text streaming and nullable limits**

Extend the existing `_entry()` helper with `streaming_output: tuple[str, ...] = ()` and `max_output_tokens: int | None = 4096`. Materialize with a temporary positive limit, then replace the immutable limits when the requested value is `None`. Add focused assertions through the existing `FakeClient` and public event stream:

```python
async def test_invoker_streams_only_when_text_is_a_streaming_output() -> None:
    client = FakeClient([_completion("non-streamed")])
    entry = _entry(
        "primary", "model", streaming_output=(), max_output_tokens=None
    )
    invoker = DefaultModelInvoker(
        adapters={"openai_chat_completions": OpenAICompatibleAdapter()},
        clients={"primary": client},
    )
    execution = invoker.execute(
        model_group=ModelGroup("assistant", (entry,)),
        retry_policy=RetryPolicy(),
        generation=GenerationConfig(max_output_tokens=128),
        message=UserMessage(content="hello"),
        context=Context(),
    )
    assert (await execution.result()).message.content == "non-streamed"


async def test_unknown_model_output_limit_does_not_warn() -> None:
    client = FakeClient([_completion()])
    entry = _entry("primary", "model", max_output_tokens=None)
    invoker = DefaultModelInvoker(
        adapters={"openai_chat_completions": OpenAICompatibleAdapter()},
        clients={"primary": client},
    )
    execution = invoker.execute(
        model_group=ModelGroup("assistant", (entry,)),
        retry_policy=RetryPolicy(),
        generation=GenerationConfig(max_output_tokens=128),
        message=UserMessage(content="hello"),
        context=Context(),
    )
    await execution.result()
    async with execution.subscribe() as events:
        warnings = [event async for event in events if event.kind == "model.capability.warning"]
    assert all("limits.max_output_tokens" not in event.data["missing_capabilities"] for event in warnings)
```

- [ ] **Step 2: Run focused consumers and confirm RED**

Run: `uv run pytest -q tests/llm/test_model_spec_execution.py tests/llm/test_builtin_catalogs.py`

Expected: FAIL on `.streaming.text`, `None` comparison, and old JSON shapes.

- [ ] **Step 3: Migrate implementation and all internal constructors**

Use exactly:

```python
if "text" in model.capabilities.streaming.output:
    ...

if (
    generation.max_output_tokens is not None
    and capabilities.limits.max_output_tokens is not None
    and generation.max_output_tokens > capabilities.limits.max_output_tokens
):
    missing.append("limits.max_output_tokens")
```

Change test/support constructors to `ModelStreamingCapabilities(output=("text",))` or `output=()`. Migrate every package JSON record from `{"text": true}` to `{"output": ["text"]}` and from false to an empty list. Keep the four capability presets text-only and keep their two positive limit arguments required.

- [ ] **Step 4: Scan out the old projection and run GREEN tests**

Run: `rg -n 'streaming[.]text|"streaming"\s*:\s*\{"text"' src tests examples benchmarks docs .github`

Expected: no code, Mapping, or serialization use remains; prose that explicitly explains the rejected old form is allowed.

Run: `uv run pytest -q tests/llm/test_model_config.py tests/llm/test_model_spec_execution.py tests/llm/test_builtin_catalogs.py tests/runtime`

Expected: PASS.

- [ ] **Step 5: Commit the mechanical migration**

```text
git add src tests examples benchmarks
git commit -m "refactor(llm): migrate capability consumers"
```

### Task 3: Add the Token Plan Provider and all model records

**Files:**
- Modify: `tests/llm/test_builtin_catalogs.py`
- Modify: `src/pygent/llm/data/providers.json`
- Modify: `src/pygent/llm/data/model_capabilities.json`

- [ ] **Step 1: Write failing Provider and complete-inventory tests**

Define the exact expected IDs and assert both protocol records for text models:

```python
ALIYUN_TEXT_MODELS = {
    "qwen3.8-max", "qwen3.8-flash", "qwen3.7-max", "qwen3.7-plus",
    "qwen3.6-flash", "deepseek-v4-pro", "deepseek-v4-pro-0813",
    "deepseek-v4-flash-0731", "glm-5.2",
}
ALIYUN_SPECIALIZED_MODELS = {
    "qwen-image-3.0-pro", "wan2.7-image", "wan2.7-image-pro",
    "happyhorse-1.1-i2v", "happyhorse-1.1-t2v", "happyhorse-1.1-r2v",
    "qwen-audio-3.0-tts-plus", "qwen-audio-3.0-realtime-plus",
    "qwen-audio-3.0-asr-flash",
}


def test_builtin_token_plan_provider_has_two_executable_protocol_presets() -> None:
    preset = ProviderCatalog.builtin().providers["aliyun_token_plan"]
    assert preset.default_protocol == "openai_chat_completions"
    assert set(preset.protocols) == {"openai_chat_completions", "anthropic_messages"}
    assert preset.protocols["openai_chat_completions"].api_key_env == (
        "ALIYUN_TOKEN_PLAN_OPENAI_API_KEY"
    )
    assert preset.protocols["anthropic_messages"].api_key_env == (
        "ALIYUN_TOKEN_PLAN_ANTHROPIC_API_KEY"
    )


def test_builtin_token_plan_catalog_has_all_18_models_and_27_records() -> None:
    records = {
        key: value for key, value in ModelCapabilityCatalog.builtin().models.items()
        if key[0] == "aliyun_token_plan"
    }
    assert {key[1] for key in records} == ALIYUN_TEXT_MODELS | ALIYUN_SPECIALIZED_MODELS
    assert len(records) == 27
    for model_id in ALIYUN_TEXT_MODELS:
        assert ("aliyun_token_plan", model_id, "openai_chat_completions") in records
        assert ("aliyun_token_plan", model_id, "anthropic_messages") in records
```

- [ ] **Step 2: Run catalog tests and confirm RED**

Run: `uv run pytest -q tests/llm/test_builtin_catalogs.py`

Expected: FAIL because `aliyun_token_plan` and its model records do not exist.

- [ ] **Step 3: Add exact Provider and model JSON records**

Add the two approved connection presets. Add each model/protocol capability as a complete record rather than a template reference. Use these dedicated protocol identifiers exactly:

```text
dashscope_multimodal_generation
dashscope_video_generation
dashscope_speech_synthesis
dashscope_realtime
dashscope_speech_recognition
```

Encode input/output/streaming/limits exactly as frozen in the proposal. For all nine text models on both text protocols, set `tools.call=true`, `choice=[none, auto, required, named]`, and `parallel=true`; the two wire APIs expose all four selection forms and parallel calls. Set structured output exactly as follows:

```text
OpenAI qwen3.8/qwen3.7:  json_object=true, json_schema=true
OpenAI qwen3.6:          json_object=true, json_schema=false
OpenAI deepseek/glm:     json_object=true, json_schema=false
Anthropic qwen3.8/qwen3.7/deepseek/glm: json_object=true, json_schema=true
Anthropic qwen3.6:       json_object=true, json_schema=false
```

Set all nine text models to `reasoning.supported=true`, `reasoning.controllable=true`, context 1,000,000 and the proposal's input modalities. Never infer JSON Schema from JSON Object support.

- [ ] **Step 4: Add per-model capability assertions and run GREEN**

For every one of the 27 keys, assert the complete `to_mapping()` value in a parametrized test. Include dedicated checks that Qwen vision inputs contain image, image models output only image, each HappyHorse variant has its correct input set, realtime streams text and audio, and ASR maps audio to text.

Run: `uv run pytest -q tests/llm/test_builtin_catalogs.py`

Expected: PASS with all 27 records exercised.

- [ ] **Step 5: Commit the catalog**

```text
git add src/pygent/llm/data/providers.json src/pygent/llm/data/model_capabilities.json tests/llm/test_builtin_catalogs.py
git commit -m "feat(llm): catalog Alibaba Token Plan models"
```

### Task 4: Validate Alibaba OpenAI options strictly

**Files:**
- Modify: `tests/llm/test_provider_options.py`
- Modify: `tests/llm/test_builtin_catalogs.py`
- Modify: `src/pygent/llm/openai_compatible.py`
- Modify: `src/pygent/llm/data/providers.json`

- [ ] **Step 1: Write failing option projection and rejection tests**

Cover each approved option and representative invalid values:

```python
@pytest.mark.parametrize(
    "options",
    [
        {"enable_thinking": True},
        {"preserve_thinking": True},
        {"reasoning_effort": "high"},
        {"thinking": {"type": "adaptive"}},
        {"thinking_budget": 0},
        {"tool_stream": True},
    ],
)
def test_aliyun_token_plan_options_are_projected(options: dict[str, object]) -> None:
    entry = model_entry(
        "main", "aliyun_token_plan", "qwen3.8-max", provider_options=options
    )
    payload = OpenAICompatibleAdapter().build_request(
        _request(entry)
    )
    for key, value in options.items():
        assert payload[key] == value


@pytest.mark.parametrize(
    "options",
    [
        {"enable_thinking": "yes"},
        {"preserve_thinking": 1},
        {"reasoning_effort": "extreme"},
        {"thinking": {"type": "enabled"}},
        {"thinking_budget": -1},
        {"tool_stream": "true"},
        {"unknown_aliyun_option": True},
    ],
)
def test_aliyun_token_plan_options_fail_closed(options: dict[str, object]) -> None:
    entry = model_entry(
        "main", "aliyun_token_plan", "qwen3.8-max", provider_options=options
    )
    with pytest.raises((TypeError, ValueError)):
        OpenAICompatibleAdapter().build_request(_request(entry))
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `uv run pytest -q tests/llm/test_provider_options.py tests/llm/test_builtin_catalogs.py`

Expected: FAIL because Alibaba currently falls through the generic option validator and its catalog schema is absent.

- [ ] **Step 3: Implement one Provider-specific validator without a registry**

Keep reserved-field and credential portability checks shared. Dispatch only the stable Alibaba extension schema from `_validate_openai_provider_options`. Accept booleans for `enable_thinking`, `preserve_thinking`, and `tool_stream`; a non-boolean integer greater than or equal to zero for `thinking_budget`; `reasoning_effort` in `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, or `max`; and exactly `{"type": "adaptive"}` or `{"type": "disabled"}` for `thinking`. Reject unknown fields and all other shapes before I/O. Project valid options unchanged into the request body. Do not hardcode per-model option compatibility matrices.

Mirror the same constraints in `providers.json` `provider_options_schema` so UI validation and Adapter validation agree.

- [ ] **Step 4: Run GREEN and regression tests**

Run: `uv run pytest -q tests/llm/test_provider_options.py tests/llm/test_openai_compatible.py tests/llm/test_builtin_catalogs.py`

Expected: PASS, including existing DeepSeek strict option tests.

- [ ] **Step 5: Commit option support**

```text
git add src/pygent/llm/openai_compatible.py src/pygent/llm/data/providers.json tests/llm/test_provider_options.py tests/llm/test_builtin_catalogs.py
git commit -m "feat(llm): validate Alibaba OpenAI options"
```

### Task 5: Make OpenAI reasoning continuation response-driven

**Files:**
- Modify: `tests/llm/test_openai_compatible.py`
- Modify: `tests/llm/test_invoker.py`
- Modify: `src/pygent/llm/openai_compatible.py`

- [ ] **Step 1: Write failing provider-neutral response tests**

Use a custom Provider name to prove there is no whitelist:

```python
@pytest.mark.parametrize("provider", ["aliyun_token_plan", "custom_gateway"])
def test_openai_reasoning_content_creates_provider_scoped_continuation(provider: str) -> None:
    entry = model_entry("main", provider, "reasoning-model")
    response = OpenAICompatibleAdapter().parse_response(
        provider_request(
            entry=entry,
            message=UserMessage(content="question"),
            context=Context(),
            generation=GenerationConfig(),
        ),
        freeze_json_object({"choices": [{"message": {"role": "assistant", "content": "ok", "reasoning_content": "r"}, "finish_reason": "stop"}]}),
    )
    assert response.message.continuation == ModelContinuation(
        provider=provider,
        protocol="openai_chat_completions",
        data={"version": 1, "reasoning_content": "r"},
    )


def test_openai_reasoning_content_rejects_non_string_when_present() -> None:
    entry = model_entry("main", "custom_gateway", "reasoning-model")
    with pytest.raises(ModelProviderError, match="reasoning_content"):
        OpenAICompatibleAdapter().parse_response(
            provider_request(entry=entry, message=UserMessage(content="question"), context=Context(), generation=GenerationConfig()),
            freeze_json_object({"choices": [{"message": {"content": "ok", "reasoning_content": {}}, "finish_reason": "stop"}]}),
        )


def test_openai_response_without_reasoning_content_has_no_continuation() -> None:
    response = OpenAICompatibleAdapter().parse_response(
        _request(),
        freeze_json_object({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}),
    )
    assert response.message.continuation is None
```

Add equivalent streamed-delta and tool-message replay tests for `aliyun_token_plan`.

- [ ] **Step 2: Run tests and confirm RED**

Run: `uv run pytest -q tests/llm/test_openai_compatible.py tests/llm/test_invoker.py`

Expected: FAIL because continuation creation, validation, and replay are gated on `provider == "deepseek"`.

- [ ] **Step 3: Remove only the Provider whitelist checks**

When a non-streamed message or streamed delta contains `reasoning_content`, validate and accumulate it regardless of Provider. Construct continuation with `request.model.provider`. In message encoding, replay a matching version-1 `reasoning_content` continuation for any Provider, while preserving the existing provider/protocol equality check and malformed-data rejection.

- [ ] **Step 4: Scan and run GREEN tests**

Run: `rg -n 'provider == "deepseek"|provider != "deepseek"' src/pygent/llm/openai_compatible.py`

Expected: no Provider condition remains around reasoning continuation; DeepSeek-only option validation may remain.

Run: `uv run pytest -q tests/llm/test_openai_compatible.py tests/llm/test_invoker.py tests/runtime/test_provider_options.py`

Expected: PASS.

- [ ] **Step 5: Commit the protocol rule**

```text
git add src/pygent/llm/openai_compatible.py tests/llm/test_openai_compatible.py tests/llm/test_invoker.py
git commit -m "refactor(llm): preserve OpenAI reasoning by response"
```

### Task 6: Synchronize public documentation and durable projections

**Files:**
- Modify: `docs/llm/FEATURES.md`
- Modify: `docs/llm/README.md`
- Modify: `docs/llm/SDK.md`
- Modify: `docs/llm/MODEL_ACCESS_ARCHITECTURE_PROPOSAL.md`
- Modify: `docs/llm/ANTHROPIC_MESSAGES_PROPOSAL.md`
- Modify: `docs/llm/PROVIDER_OPTIONS_SPEC.md`
- Modify: `examples/live_agent/agent.py`
- Modify: `benchmarks/models.py`
- Modify: `tests/runtime/test_provider_options.py`
- Modify: `tests/support/model_specs.py`

- [ ] **Step 1: Write failing round-trip assertions before touching codecs or fixtures**

Extend `test_sqlite_round_trip_and_tampered_provider_options_fail_closed` by replacing the fixture entry's immutable capabilities before `build_snapshot`:

```python
entry = model_entry("main", "custom", "image-model")
capabilities = replace(
    entry.spec.capabilities,
    modalities=ModelModalities(input=("text", "image"), output=("image",)),
    streaming=ModelStreamingCapabilities(output=()),
    limits=ModelLimits(context_tokens=None, max_output_tokens=None),
)
entry = replace(entry, spec=replace(entry.spec, capabilities=capabilities))
snapshot = build_snapshot(
    scope_id="scope",
    requirement=requirement,
    profile="quality",
    models=(entry,),
    resources=None,
)
await store.ensure_profile(snapshot, make_default=True)
current = await store.current("scope", "assistant", "quality")
assert current.model_group.models[0].spec.capabilities == capabilities
```

- [ ] **Step 2: Run the focused durable tests and confirm RED where fixtures remain old**

Run: `uv run pytest -q tests/runtime tests/llm/test_request_snapshot.py tests/integration/test_public_api.py`

Expected: any remaining old capability fixture fails strict parsing or serialization; the new round-trip must not pass until all projections are migrated.

- [ ] **Step 3: Apply mechanical projection changes only**

Update serialized fixtures and docs from:

```yaml
streaming: {text: true}
```

to:

```yaml
streaming: {output: [text]}
```

Document `int | null` limits, the `aliyun_token_plan` configuration example, and that dedicated protocols are catalogued but have no built-in Adapter. Do not change Runtime, Worker, Binding, fallback, capacity, or message APIs.

- [ ] **Step 4: Run docs scan and focused tests**

Run: `rg -n 'streaming[.]text|streaming:\s*\{text:' docs src tests examples benchmarks`

Expected: matches only describe the intentionally rejected legacy shape.

Run: `uv run pytest -q tests/runtime tests/llm tests/integration/test_public_api.py`

Expected: PASS.

- [ ] **Step 5: Commit public projection updates**

```text
git add docs src tests examples benchmarks
git commit -m "docs(llm): publish multimodal capability projection"
```

### Task 7: Verify all 18 Token Plan models and complete release checks

**Files:**
- Create: `tests/live/aliyun_token_plan_probe.py`
- Create: `tests/live/test_aliyun_token_plan_probe.py`
- Modify: `.github/workflows/python-publish.yml` only if the existing catalog artifact check does not already cover changed files

- [ ] **Step 1: Write failing offline tests for the explicit live inventory**

The probe must expose a frozen inventory with exactly 18 unique Model IDs and account for every result:

```python
def test_probe_inventory_matches_builtin_token_plan_catalog() -> None:
    inventory = probe_inventory()
    assert len(inventory) == 18
    assert len({probe.model_id for probe in inventory}) == 18
    assert {probe.model_id for probe in inventory} == token_plan_model_ids_from_catalog()


def test_probe_summary_never_contains_secret_or_response_content() -> None:
    summary = summarize_result(
        ProbeResult(
            model_id="qwen3.8-max",
            protocol="openai_chat_completions",
            status="passed",
            error_kind=None,
            private_detail="secret output",
        )
    )
    assert "secret output" not in summary
    assert set(summary) == {"model_id", "protocol", "status", "error_kind"}
```

Use injected transports in offline tests to verify request paths and minimal payloads for text, image, video submit/poll, TTS, realtime, and ASR without network access.

- [ ] **Step 2: Run tests and confirm RED**

Run: `uv run pytest -q tests/live/test_aliyun_token_plan_probe.py`

Expected: FAIL because the probe module does not exist.

- [ ] **Step 3: Implement the explicit verifier**

The script must:

```text
- load E:/Projects/pygent/.env explicitly without printing values;
- validate that all required credential variables exist;
- call all nine text Model IDs through OpenAI Chat Completions with a one-token-style prompt;
- call at least one text model through Anthropic Messages, then call the remaining text IDs there as protocol-record verification;
- generate one minimal image with each of the three image Model IDs;
- submit and poll one minimum-duration, minimum-resolution job for each HappyHorse Model ID;
- synthesize a short fixed phrase with qwen-audio-3.0-tts-plus;
- feed generated audio to qwen-audio-3.0-asr-flash;
- open one minimal qwen-audio-3.0-realtime-plus session and close after the first valid server response;
- emit only JSON records containing model_id, protocol, passed/failed status, and normalized error_kind;
- exit nonzero unless every one of the 18 unique Model IDs has at least one successful real invocation and both built-in text protocols have succeeded.
```

Keep this verifier outside normal pytest collection. It verifies Provider availability for catalog-only protocols; it must not import or pretend that Pygent supplies the five dedicated Adapters.

- [ ] **Step 4: Run offline probe tests and the full static suite**

Run: `uv run pytest -q tests/live/test_aliyun_token_plan_probe.py`

Expected: PASS.

Run: `uv run ruff check src tests examples benchmarks`

Expected: PASS.

Run: `uv run mypy src benchmarks`

Expected: PASS.

- [ ] **Step 5: Run every live model and retain the sanitized result table**

Run: `uv run --with websockets python tests/live/aliyun_token_plan_probe.py --env-file E:/Projects/pygent/.env`

Expected: 18 unique model IDs reported `passed`, both `openai_chat_completions` and `anthropic_messages` reported successful text calls, no credential or response content printed, and exit code 0. If a Provider rejects a model or endpoint, preserve the sanitized failure result, diagnose against current official documentation, write a failing regression test for any code defect, and do not mark the goal complete while a catalog record is unverified.

- [ ] **Step 6: Run full release verification**

Run: `uv run pytest -q`

Expected: all tests PASS with no regression from the 987-test baseline.

Run: `uv build`

Expected: wheel and sdist build successfully.

Run: `uvx twine check dist/*`

Expected: all distributions PASS.

Inspect the wheel and sdist and hash their three catalog JSON files; expected packaged bytes equal the source files exactly.

- [ ] **Step 7: Commit the verifier and any release-only adjustment**

```text
git add tests/live .github/workflows/python-publish.yml
git commit -m "test(llm): verify every Token Plan model"
```

### Final audit

- [ ] Confirm all seven task commits contain only planned files and changes.
- [ ] Confirm `git status --short` is empty.
- [ ] Confirm the 18-model sanitized live matrix has no missing model and no secret/response data.
- [ ] Confirm no new Adapter, registry, routing, Runtime, capacity, or multimodal Message API was introduced.
- [ ] Run `rg -n 'streaming[.]text|"streaming"\s*:\s*\{"text"' src tests examples benchmarks docs .github` and allow only explicit prose about rejection.
- [ ] Run the complete pytest, Ruff, mypy, build, Twine, and package-content checks once more after the final commit.
