# AZ 211 Official Provider Conformance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不把 AZ 加入 Pygent Provider 目录的前提下，完善官方 Provider、模型能力和必要协议实现，并让冻结的 211 个 AZ route 的所有必需场景真实通过。

**Architecture:** 生产侧只保存官方 Provider preset、官方 Model ID、完整 capabilities 和可复用协议 Adapter；测试侧保存独立的 AZ route manifest、官方来源索引、连接覆盖和可恢复结果 ledger。现有 Pygent 请求/响应契约能表达的能力必须通过正式 Adapter，异步媒体和搜索等尚不属于该契约的能力由隔离 raw probe 验证。

**Tech Stack:** Python 3.11、`dataclasses`、`httpx`、`httpx-sse`、`websockets`（仅 live extra）、`jsonschema`、pytest、Ruff、mypy、Maturin、Twine。

---

## File structure

新增测试侧文件：

```text
tests/live/az_conformance/
  __init__.py
  route_ids.txt            # 冻结的 211 个排序后 route ID
  manifest.json
  sources.json
  schemas.py
  inventory.py
  results.py
  runner.py
  openai_probes.py
  anthropic_probes.py
  gemini_probes.py
  media_probes.py
  search_probes.py
  cli.py
tests/live/test_az_conformance_schemas.py
tests/live/test_az_conformance_inventory.py
tests/live/test_az_conformance_results.py
tests/live/test_az_conformance_runner.py
tests/live/test_az_conformance_probes.py
```

可能新增的生产侧协议文件按独立 contract 拆分：

```text
src/pygent/llm/openai_responses.py
src/pygent/llm/gemini_generate_content.py
```

以下现有文件仅按任务所需修改：

```text
src/pygent/llm/protocols.py
src/pygent/llm/__init__.py
src/pygent/llm/catalogs.py
src/pygent/llm/configuration.py
src/pygent/llm/invoker.py
src/pygent/llm/openai_compatible.py
src/pygent/llm/anthropic_messages.py
src/pygent/llm/data/providers.json
src/pygent/llm/data/model_capabilities.json
tests/llm/test_builtin_catalogs.py
tests/llm/test_provider_options.py
tests/llm/test_model_spec_execution.py
docs/llm/README.md
docs/llm/SDK.md
docs/llm/PROVIDER_OPTIONS_SPEC.md
```

不新增 AZ 生产模块，不把 `tests/live/az_conformance` 打进 wheel/sdist，不扩张现有文本
`ModelProviderResponse` 来承载视频文件或搜索结果。

### Task 1: Re-establish the baseline and freeze the 211 route IDs

**Files:**
- Create: `tests/live/az_conformance/__init__.py`
- Create: `tests/live/az_conformance/route_ids.txt`
- Test: `tests/live/test_az_conformance_schemas.py`

- [ ] **Step 1: Verify the starting revision and baseline**

Run:

```powershell
git status --short
git rev-parse --short HEAD
uv run pytest -q
```

Expected: clean status, HEAD descends from `256eccb`, and the existing suite reports 1056 passing tests.

- [ ] **Step 2: Write the failing snapshot test**

```python
from hashlib import sha256
from importlib.resources import files


def test_frozen_route_id_snapshot_is_exact() -> None:
    text = (
        files("tests.live.az_conformance")
        .joinpath("route_ids.txt")
        .read_text(encoding="utf-8")
    )
    ids = text.splitlines()
    payload = ("\n".join(ids) + "\n").encode()
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids)) == 211
    assert sha256(payload).hexdigest() == (
        "0052cd3543beb555439a8e0c32bb415f8df34a8992f358ed23f971ba4f79c949"
    )
```

- [ ] **Step 3: Run the snapshot test and confirm RED**

Run:

```powershell
uv run pytest tests/live/test_az_conformance_schemas.py::test_frozen_route_id_snapshot_is_exact -q
```

Expected: FAIL because the package and manifest do not exist.

- [ ] **Step 4: Add the exact frozen route ID file**

Create `route_ids.txt` with exactly the approved 211 IDs in ordinal order, one UTF-8 ID per line and one final
newline. Do not add classification or scenario data in this task; `manifest.json` is created atomically with full
classification in Task 8, so no committed manifest is ever temporarily invalid.

- [ ] **Step 5: Run the snapshot test and confirm GREEN**

Run:

```powershell
uv run pytest tests/live/test_az_conformance_schemas.py::test_frozen_route_id_snapshot_is_exact -q
```

Expected: PASS.

- [ ] **Step 6: Commit the frozen identity**

```powershell
git add tests/live/az_conformance/route_ids.txt tests/live/az_conformance/__init__.py tests/live/test_az_conformance_schemas.py
git commit -m "test(llm): freeze AZ 211 model routes"
```

### Task 2: Implement strict immutable manifest and source schemas

**Files:**
- Create: `tests/live/az_conformance/schemas.py`
- Create: `tests/live/az_conformance/sources.json`
- Modify: `tests/live/test_az_conformance_schemas.py`

- [ ] **Step 1: Write failing schema tests**

Cover exact fields, schema version, sorted unique routes, closed route kinds, non-empty protocols and scenarios,
canonical requirements, alias exclusion from catalog, external-service exclusion, unknown references, immutable
collections, source URL HTTPS validation, and source-key uniqueness.

```python
def test_gateway_alias_requires_canonical_model_and_is_not_catalog_eligible() -> None:
    route = route_from_mapping(
        {
            "route_id": "gpt-5.4-urg",
            "kind": "gateway_alias",
            "canonical_provider": "openai",
            "canonical_model_id": "gpt-5.4",
            "protocols": ["openai_chat_completions"],
            "required_scenarios": ["text"],
            "catalog_eligible": False,
        }
    )
    assert route.canonical_model_id == "gpt-5.4"


@pytest.mark.parametrize("kind", ["gateway_alias", "external_service"])
def test_nonofficial_route_cannot_enter_catalog(kind: str) -> None:
    value = valid_route(kind=kind)
    value["catalog_eligible"] = True
    with pytest.raises(ValueError, match="cannot be catalog eligible"):
        route_from_mapping(value)
```

- [ ] **Step 2: Run the schema tests and confirm RED**

Run: `uv run pytest tests/live/test_az_conformance_schemas.py -q`

Expected: FAIL on missing schema types and validation.

- [ ] **Step 3: Implement closed value objects**

Implement these exact public test-side values:

```python
class RouteKind(StrEnum):
    OFFICIAL_MODEL = "official_model"
    GATEWAY_ALIAS = "gateway_alias"
    EXTERNAL_SERVICE = "external_service"


class Scenario(StrEnum):
    TEXT = "text"
    TEXT_STREAM = "text_stream"
    TOOLS = "tools"
    TOOL_CHOICE = "tool_choice"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"
    REASONING = "reasoning"
    IMAGE_INPUT = "image_input"
    IMAGE_OUTPUT = "image_output"
    IMAGE_EDIT = "image_edit"
    VIDEO_OUTPUT = "video_output"
    AUDIO_OUTPUT = "audio_output"
    AUDIO_INPUT = "audio_input"
    REALTIME = "realtime"
    EMBEDDING = "embedding"
    SEARCH = "search"


@dataclass(frozen=True, slots=True)
class AzRoute:
    route_id: str
    kind: RouteKind
    canonical_provider: str | None
    canonical_model_id: str | None
    protocols: tuple[str, ...]
    required_scenarios: tuple[Scenario, ...]
    catalog_eligible: bool


@dataclass(frozen=True, slots=True)
class SourceRecord:
    provider: str
    model_id: str
    protocol: str
    url: str
    checked_at: date
```

`load_manifest()` and `load_sources()` must reject unknown fields and return tuples/mapping proxies.

- [ ] **Step 4: Add the source-index root and strict parser**

```json
{
  "schema_version": 1,
  "sources": []
}
```

Sources are added during provider classification tasks, never read by production runtime.

- [ ] **Step 5: Run tests and type/lint checks**

```powershell
uv run pytest tests/live/test_az_conformance_schemas.py -q
uv run ruff check tests/live/az_conformance tests/live/test_az_conformance_schemas.py
uv run mypy tests/live/az_conformance
```

Expected: all pass.

- [ ] **Step 6: Commit**

```powershell
git add tests/live/az_conformance tests/live/test_az_conformance_schemas.py
git commit -m "test(llm): define AZ conformance manifest schema"
```

### Task 3: Detect live inventory drift before paid requests

**Files:**
- Create: `tests/live/az_conformance/inventory.py`
- Test: `tests/live/test_az_conformance_inventory.py`

- [ ] **Step 1: Write failing inventory tests**

```python
@pytest.mark.asyncio
async def test_inventory_mismatch_stops_before_probe_dispatch() -> None:
    client = fake_models_client(["chatgpt-4o-latest", "new-route"])
    result = await compare_inventory(client, manifest_with(["chatgpt-4o-latest"]))
    assert result.added == ("new-route",)
    assert result.removed == ()
    assert result.matches is False


def test_snapshot_digest_uses_sorted_ids_and_trailing_newline() -> None:
    assert inventory_digest(("b", "a")) == sha256(b"a\nb\n").hexdigest()
```

Also assert that authentication headers never appear in comparison objects or repr.

- [ ] **Step 2: Run and confirm RED**

Run: `uv run pytest tests/live/test_az_conformance_inventory.py -q`

- [ ] **Step 3: Implement inventory comparison**

```python
@dataclass(frozen=True, slots=True)
class InventoryDiff:
    expected_count: int
    actual_count: int
    expected_sha256: str
    actual_sha256: str
    added: tuple[str, ...]
    removed: tuple[str, ...]

    @property
    def matches(self) -> bool:
        return not self.added and not self.removed
```

The fetch function accepts an injected `httpx.AsyncClient`, normalizes a base ending in either host or `/v1`,
calls exactly one `/models` endpoint, and returns IDs only.

- [ ] **Step 4: Prove drift prevents dispatch**

The top-level runner introduced later must call `require_matching_inventory()` before constructing its probe
queue. Add a fake dispatch callback and assert it remains untouched on mismatch.

- [ ] **Step 5: Run tests and commit**

```powershell
uv run pytest tests/live/test_az_conformance_inventory.py -q
git add tests/live/az_conformance/inventory.py tests/live/test_az_conformance_inventory.py
git commit -m "test(llm): reject AZ inventory drift"
```

### Task 4: Implement sanitized, resumable result records

**Files:**
- Create: `tests/live/az_conformance/results.py`
- Test: `tests/live/test_az_conformance_results.py`

- [ ] **Step 1: Write failing result-contract tests**

```python
def test_result_key_includes_snapshot_revision_route_protocol_and_scenario() -> None:
    result = passed_result()
    assert result.key == (
        SNAPSHOT_SHA,
        "abc1234",
        "gpt-5.4",
        "openai_chat_completions",
        Scenario.TEXT,
    )


def test_public_json_excludes_private_detail_and_secrets() -> None:
    value = failed_result(private_detail="Bearer secret response body").to_public_mapping()
    serialized = json.dumps(value)
    assert "secret" not in serialized
    assert "response body" not in serialized
```

Cover closed error kinds, attempts >= 1, atomic checkpoint replacement, malformed checkpoint rejection, reuse
only for exact passed keys, and superseding an active failure with a newer pass.

- [ ] **Step 2: Run and confirm RED**

Run: `uv run pytest tests/live/test_az_conformance_results.py -q`

- [ ] **Step 3: Implement the result schema**

```python
class ErrorKind(StrEnum):
    CONFIGURATION = "configuration"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    RATE_LIMIT = "rate_limit"
    INVALID_REQUEST = "invalid_request"
    PROTOCOL_MISMATCH = "protocol_mismatch"
    CAPABILITY_MISMATCH = "capability_mismatch"
    INVALID_RESPONSE = "invalid_response"
    TIMEOUT = "timeout"
    GATEWAY_UNAVAILABLE = "gateway_unavailable"
    UPSTREAM_UNAVAILABLE = "upstream_unavailable"
    CONTENT_REJECTED = "content_rejected"
    DEPENDENCY_FAILED = "dependency_failed"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ProbeResult:
    snapshot_sha256: str
    source_revision: str
    route_id: str
    canonical_provider: str | None
    canonical_model_id: str | None
    protocol: str
    scenario: Scenario
    status: Literal["passed", "failed"]
    attempts: int
    error_kind: ErrorKind | None
    private_detail: object = field(default=None, repr=False, compare=False)
```

Write JSON Lines checkpoints with one public mapping per key and replace the index atomically through a sibling
temporary file. The caller supplies the output directory; repository paths are not defaults.

- [ ] **Step 4: Run tests, lint, and commit**

```powershell
uv run pytest tests/live/test_az_conformance_results.py -q
uv run ruff check tests/live/az_conformance/results.py tests/live/test_az_conformance_results.py
git add tests/live/az_conformance/results.py tests/live/test_az_conformance_results.py
git commit -m "test(llm): add resumable AZ probe ledger"
```

### Task 5: Implement the protocol/scenario probe registry and scheduler

**Files:**
- Create: `tests/live/az_conformance/runner.py`
- Test: `tests/live/test_az_conformance_runner.py`

- [ ] **Step 1: Write failing scheduler tests**

Cover exact probe lookup by `(protocol, scenario)`, no fallback lookup, deterministic queue order, checkpoint reuse,
bounded retries only for timeout/429/5xx, `Retry-After`, media serialization, low text concurrency, dependency
ordering for TTS-to-ASR fixtures, cancellation, and final exact-set aggregation.

```python
def test_completion_requires_every_required_key_to_pass() -> None:
    report = build_report(manifest=two_route_manifest(), results=[one_passed_result()])
    assert report.passed_routes == ()
    assert report.missing_keys
    assert report.complete is False


def test_skipped_cannot_satisfy_a_required_scenario() -> None:
    with pytest.raises(ValueError, match="status"):
        result_from_mapping({**passed_mapping(), "status": "skipped"})
```

- [ ] **Step 2: Run and confirm RED**

Run: `uv run pytest tests/live/test_az_conformance_runner.py -q`

- [ ] **Step 3: Implement exact probe registration**

```python
Probe = Callable[[ProbeContext, AzRoute], Awaitable[ProbeResult]]
ProbeKey = tuple[str, Scenario]


@dataclass(frozen=True, slots=True)
class ProbeRegistry:
    probes: Mapping[ProbeKey, Probe]

    def require(self, protocol: str, scenario: Scenario) -> Probe:
        try:
            return self.probes[(protocol, scenario)]
        except KeyError as exc:
            raise ValueError(f"no probe for {protocol}/{scenario}") from exc
```

- [ ] **Step 4: Implement queue and retry policy**

Use a semaphore for cheap request/response probes and a separate semaphore fixed to one for image, video, audio,
and realtime. Retry only `RATE_LIMIT`, `TIMEOUT`, `GATEWAY_UNAVAILABLE`, and `UPSTREAM_UNAVAILABLE`, with a
maximum of three attempts and injectable clock/backoff for tests.

- [ ] **Step 5: Implement exact completion report**

`ConformanceReport.complete` is true only when its passed route set equals the manifest route set and there are
no missing or active failed required keys.

- [ ] **Step 6: Run tests and commit**

```powershell
uv run pytest tests/live/test_az_conformance_runner.py -q
git add tests/live/az_conformance/runner.py tests/live/test_az_conformance_runner.py
git commit -m "test(llm): schedule exact AZ capability probes"
```

### Task 6: Add reusable OpenAI-compatible and Anthropic probes

**Files:**
- Create: `tests/live/az_conformance/openai_probes.py`
- Create: `tests/live/az_conformance/anthropic_probes.py`
- Test: `tests/live/test_az_conformance_probes.py`
- Modify: `src/pygent/llm/openai_compatible.py`
- Modify: `src/pygent/llm/anthropic_messages.py`
- Modify: `tests/llm/test_openai_compatible.py`
- Modify: `tests/llm/test_anthropic_messages.py`

- [ ] **Step 1: Write failing probe-contract tests**

For each existing Adapter, inject a fake client and verify text, stream termination, tools round trip, forced tool
choice, JSON object/schema, reasoning enabled/disabled, and image input. Assert minimal output limits and no raw
content in `ProbeResult`.

```python
@pytest.mark.asyncio
async def test_tool_probe_executes_tool_result_continuation() -> None:
    result = await openai_tool_probe(context_with_scripted_client(), route("gpt-5.4"))
    assert result.status == "passed"
    assert scripted_requests()[0].tools
    assert scripted_requests()[1].messages[-1].role == "tool"
```

- [ ] **Step 2: Run focused tests and confirm RED**

```powershell
uv run pytest tests/live/test_az_conformance_probes.py tests/llm/test_openai_compatible.py tests/llm/test_anthropic_messages.py -q
```

- [ ] **Step 3: Implement probes through official Adapters**

Construct real `ModelSpec` values with canonical provider/model capabilities, but bind clients to AZ connections.
Use route ID only in the transport model field. Reuse `OpenAICompatibleAdapter` and `AnthropicMessagesAdapter`
request/stream decoding; do not duplicate their parsers in live code.

- [ ] **Step 4: Fix only evidence-backed Adapter incompatibilities**

When unit or live evidence shows a provider-specific field difference, add validation/serialization keyed by
canonical provider and exact protocol. Do not add AZ branches to production code. Every fix starts with a focused
failing Adapter test.

- [ ] **Step 5: Run tests and commit**

```powershell
uv run pytest tests/live/test_az_conformance_probes.py tests/llm -q
uv run ruff check src/pygent/llm tests/live/az_conformance tests/live/test_az_conformance_probes.py tests/llm
git add src/pygent/llm tests/live/az_conformance tests/live/test_az_conformance_probes.py tests/llm
git commit -m "test(llm): probe OpenAI and Anthropic capability contracts"
```

### Task 7: Add protocol implementations required by official contracts

**Files:**
- Create: `src/pygent/llm/openai_responses.py`
- Create: `src/pygent/llm/gemini_generate_content.py`
- Create: `tests/llm/test_openai_responses.py`
- Create: `tests/llm/test_gemini_generate_content.py`
- Modify: `src/pygent/llm/protocols.py`
- Modify: `src/pygent/llm/__init__.py`
- Modify: `src/pygent/llm/invoker.py`
- Create/Modify: `tests/live/az_conformance/gemini_probes.py`

- [ ] **Step 1: Lock the documented protocol requirements**

Add tests for the two protocol contracts already documented by the AZ service: `POST /v1/responses` and Gemini
`/v1beta/models/{model}:generateContent`. Also add an offline coverage test that later derives protocol use from
`manifest.json` and fails if a protocol lacks either a production Adapter or an explicitly raw test-only probe.

- [ ] **Step 2: Write RED contract tests for OpenAI Responses**

Cover input item serialization, streaming text deltas, function calls/results, structured output, reasoning
metadata, usage, error mapping, cancellation, and continuation.

```python
def test_builtin_protocol_has_precise_responses_identifier() -> None:
    assert BuiltinModelProtocol.OPENAI_RESPONSES == "openai_responses"
```

- [ ] **Step 3: Implement `OpenAIResponsesAdapter` minimally**

Follow the existing adapter contracts; keep its serializer and stream decoder in `openai_responses.py`, register by
`openai_responses`, and share only neutral helpers. Do not make Chat Completions accept Responses events.

- [ ] **Step 4: Write RED contract tests for Gemini generateContent**

Cover role/parts mapping, system instruction, image input, function calls/results, JSON schema, thinking parts,
SSE streaming, usage, safety/error mapping, cancellation, and API-key authentication without placing a secret in
a URL repr or event.

- [ ] **Step 5: Implement `GeminiGenerateContentAdapter` minimally**

Use the exact identifier `gemini_generate_content`. Extend connection/client construction with a secret-safe API
key header/query strategy only if the official and AZ endpoints require it; add strict tests to prevent credential
serialization.

- [ ] **Step 6: Run protocol suites and commit independently**

```powershell
uv run pytest tests/llm/test_openai_responses.py tests/llm/test_gemini_generate_content.py tests/live/test_az_conformance_probes.py -q
uv run mypy src/pygent/llm
git add src/pygent/llm tests/llm tests/live/az_conformance tests/live/test_az_conformance_probes.py
git commit -m "feat(llm): add required official model protocols"
```

### Task 8: Classify all 211 routes and build the official source index

**Files:**
- Modify: `tests/live/az_conformance/manifest.json`
- Modify: `tests/live/az_conformance/sources.json`
- Modify: `tests/live/test_az_conformance_schemas.py`

- [ ] **Step 1: Add a failing all-routes-classified test**

```python
def test_every_route_is_fully_classified() -> None:
    manifest = load_builtin_manifest()
    assert len(manifest.routes) == 211
    for route in manifest.routes:
        assert route.protocols
        assert route.required_scenarios
        if route.kind is not RouteKind.EXTERNAL_SERVICE:
            assert route.canonical_provider
            assert route.canonical_model_id
```

Also assert every catalog-eligible `(provider, model_id, protocol)` has at least one source record and every alias
canonical target resolves to an official source record.

- [ ] **Step 2: Classify provider families in deterministic batches**

Update manifest and sources in this order, running the all-routes test after each batch:

1. OpenAI and Codex IDs;
2. Anthropic Claude IDs and route-quality aliases;
3. Google Gemini IDs and thinking/urg aliases;
4. Alibaba Qwen/QwQ/QVQ IDs;
5. DeepSeek IDs and `-ds` aliases;
6. Zhipu GLM IDs and urg aliases;
7. Moonshot Kimi IDs;
8. MiniMax IDs and speed/multimodal variants;
9. ByteDance Doubao/Seed/Seedream/Seedance IDs;
10. xAI Grok IDs;
11. embeddings, KAT Coder, and seven SerpAPI services.

For every batch, use exact official URLs and `checked_at: 2026-09-11` or the actual later review date. Names not
confirmed by an official source remain test-only aliases; do not infer catalog eligibility from prefix.

- [ ] **Step 3: Assign protocols from actual AZ endpoint declarations**

All 211 routes include their OpenAI-compatible entry. The 46 routes advertised with Anthropic, 28 with Gemini,
and seven with SerpAPI also include those precise protocols. Protocol arrays are sorted and unique.

- [ ] **Step 4: Assign required scenarios from official capabilities**

Each route receives every applicable scenario from Section 6.2 of the proposal. Gateway aliases inherit the
canonical profile, then add or remove only behavior proven by the alias contract. Every route has at least one
scenario; no required scenario is represented as skipped.

- [ ] **Step 5: Verify exact coverage and commit**

```powershell
uv run pytest tests/live/test_az_conformance_schemas.py tests/live/test_az_conformance_inventory.py -q
uv run python -m tests.live.az_conformance.cli validate --manifest tests/live/az_conformance/manifest.json
git add tests/live/az_conformance/manifest.json tests/live/az_conformance/sources.json tests/live/test_az_conformance_schemas.py
git commit -m "test(llm): classify all 211 AZ routes"
```

Expected CLI summary:

```text
routes=211 classified=211 unclassified=0 missing_sources=0 missing_probes=0
snapshot_sha256=0052cd3543beb555439a8e0c32bb415f8df34a8992f358ed23f971ba4f79c949
```

### Task 9: Add official Provider presets and model capabilities

**Files:**
- Modify: `src/pygent/llm/data/providers.json`
- Modify: `src/pygent/llm/data/model_capabilities.json`
- Modify: `src/pygent/llm/catalogs.py`
- Modify: `tests/llm/test_builtin_catalogs.py`
- Modify: `tests/llm/test_provider_options.py`

- [ ] **Step 1: Write failing catalog projection tests**

Derive the expected production triples from catalog-eligible manifest routes and assert exact equality with the
corresponding official-provider subset in `ModelCapabilityCatalog`. Explicitly assert no route ending in service
suffixes such as `-urg`, `-az`, or `-ds`, and no `serpapi-*`, enters the official catalog unless the exact ID has an
official source proving it is canonical.

```python
def test_az_gateway_aliases_never_enter_builtin_catalog() -> None:
    catalog = ModelCapabilityCatalog.builtin()
    forbidden = {route.route_id for route in manifest.routes if not route.catalog_eligible}
    assert forbidden.isdisjoint(model_id for _, model_id, _ in catalog.models)
```

- [ ] **Step 2: Run and confirm RED**

Run: `uv run pytest tests/llm/test_builtin_catalogs.py tests/llm/test_provider_options.py -q`

- [ ] **Step 3: Add official Provider presets**

Use official endpoints and credential references:

| Provider | Default official base | Credential env |
|---|---|---|
| `openai` | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| `anthropic` | existing Messages base | `ANTHROPIC_API_KEY` |
| `google` | `https://generativelanguage.googleapis.com/v1beta` | `GEMINI_API_KEY` |
| `alibaba_cloud` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `DASHSCOPE_API_KEY` |
| `deepseek` | existing official base | `DEEPSEEK_API_KEY` |
| `zhipu` | `https://open.bigmodel.cn/api/paas/v4` | `ZHIPU_API_KEY` |
| `moonshot` | `https://api.moonshot.cn/v1` | `MOONSHOT_API_KEY` |
| `minimax` | `https://api.minimax.io/v1` | `MINIMAX_API_KEY` |
| `volcengine` | `https://ark.cn-beijing.volces.com/api/v3` | `ARK_API_KEY` |
| `xai` | `https://api.x.ai/v1` | `XAI_API_KEY` |

Add each official extra protocol as a separate preset entry with its own exact base. Do not add AZ endpoints or
AZ credential names.

- [ ] **Step 4: Materialize complete capabilities**

For every catalog-eligible manifest triple, write a complete `ModelCapabilities` mapping. Use `null` for limits
that official evidence does not confirm; do not copy limits from a related model or gateway price record. Ensure
`streaming.output` is a subset of output modalities.

- [ ] **Step 5: Validate provider-private options**

Add only options documented by the official Provider and exact protocol. Start every validator change with a
failing test for accepted values, rejected unknown/invalid values, and immutable storage. Do not introduce gateway
route suffix handling into production validators.

- [ ] **Step 6: Run catalog suites and commit per provider batch**

Commit each provider family separately with these literal messages:

```powershell
uv run pytest tests/llm/test_builtin_catalogs.py tests/llm/test_provider_options.py tests/llm/test_model_config.py -q
git add src/pygent/llm/data src/pygent/llm/catalogs.py tests/llm
git commit -m "feat(llm): catalog openai models"
git commit -m "feat(llm): catalog google models"
git commit -m "feat(llm): catalog alibaba_cloud models"
git commit -m "feat(llm): catalog zhipu models"
git commit -m "feat(llm): catalog moonshot models"
git commit -m "feat(llm): catalog minimax models"
git commit -m "feat(llm): catalog volcengine models"
git commit -m "feat(llm): catalog xai models"
```

Run `git add` and exactly one applicable `git commit` line after each passing provider batch; existing `anthropic`
and `deepseek` records are committed only when official evidence requires a correction. Do not combine unrelated
providers in one commit.

### Task 10: Implement raw probes for capabilities outside the current Agent model contract

**Files:**
- Create/Modify: `tests/live/az_conformance/media_probes.py`
- Create/Modify: `tests/live/az_conformance/search_probes.py`
- Modify: `tests/live/az_conformance/openai_probes.py`
- Test: `tests/live/test_az_conformance_probes.py`

- [ ] **Step 1: Write failing endpoint/payload tests**

Using `httpx.MockTransport` and an injected WebSocket connector, cover:

- embeddings vector validation and stable dimension;
- image generation URL/base64 validation and decode;
- image editing multipart payload;
- TTS bytes/URL validation;
- transcription from fixed generated audio;
- realtime handshake and one legal client/server event exchange;
- async video create/poll/success/failure/timeout;
- SerpAPI result structure;
- dependency failure when a generated fixture is unavailable.

- [ ] **Step 2: Run and confirm RED**

Run: `uv run pytest tests/live/test_az_conformance_probes.py -q`

- [ ] **Step 3: Implement minimal protocol-specific probes**

Each function accepts `ProbeContext` and `AzRoute`, emits only `ProbeResult`, and keeps response bodies/media in
local scope. Downloaded media is validated in memory with strict byte limits and never persisted by default.

- [ ] **Step 4: Register exact scenario keys**

Every protocol/scenario pair used by the manifest must resolve exactly once. Add a test comparing the manifest's
required keys against `ProbeRegistry.probes`.

- [ ] **Step 5: Run tests and commit**

```powershell
uv run pytest tests/live/test_az_conformance_probes.py tests/live/test_az_conformance_runner.py -q
uv run ruff check tests/live/az_conformance tests/live/test_az_conformance_probes.py
git add tests/live/az_conformance tests/live/test_az_conformance_probes.py
git commit -m "test(llm): cover AZ multimodal and search protocols"
```

### Task 11: Add the only live CLI and dry-run gates

**Files:**
- Create: `tests/live/az_conformance/cli.py`
- Modify: `pyproject.toml`
- Test: `tests/live/test_az_conformance_runner.py`

- [ ] **Step 1: Write failing CLI tests**

Test `validate`, `inventory`, `run`, and `report` commands through an injected runner. Assert credentials are read
only from environment, base URL is HTTPS and contains no credentials, output paths are explicit, drift stops before
dispatch, and stdout/stderr contain only sanitized summaries.

- [ ] **Step 2: Add a dedicated marker and optional live dependency**

Add `az_conformance` to pytest markers and add a development-only `websockets` dependency path used by the CLI.
Do not add WebSockets to Pygent runtime dependencies unless a production realtime Adapter is implemented.

- [ ] **Step 3: Implement CLI commands**

```text
python -m tests.live.az_conformance.cli validate --manifest PATH
python -m tests.live.az_conformance.cli inventory --manifest PATH
python -m tests.live.az_conformance.cli run --manifest PATH --output-dir PATH [--route ID] [--scenario NAME]
python -m tests.live.az_conformance.cli report --manifest PATH --output-dir PATH
```

`run` prints the exact pending scenario count before dispatch and exits nonzero for drift, missing probes, active
failures, or incomplete results. `report` exits zero only for exact 211/211 completion.

- [ ] **Step 4: Run dry gates and commit**

```powershell
uv run pytest tests/live/test_az_conformance_runner.py tests/live/test_az_conformance_inventory.py -q
uv run --with websockets python -m tests.live.az_conformance.cli validate --manifest tests/live/az_conformance/manifest.json
git add pyproject.toml tests/live/az_conformance/cli.py tests/live/test_az_conformance_runner.py
git commit -m "test(llm): expose AZ conformance CLI"
```

### Task 12: Execute all live scenarios and repair evidence-backed failures

**Files:**
- Modify only when a failing test proves the need: production Adapter/catalog files or `tests/live/az_conformance/*`
- Write runtime evidence outside Git: user-selected conformance output directory

- [ ] **Step 1: Record source revision and verify inventory**

```powershell
$azResultDir = 'C:\Users\Administrator\.codex\artifacts\pygent-az-211-0052cd35'
New-Item -ItemType Directory -Path $azResultDir -Force | Out-Null
git status --short
git rev-parse HEAD
uv run --with websockets python -m tests.live.az_conformance.cli inventory --manifest tests/live/az_conformance/manifest.json
```

Expected: clean tree and exact 211/count/digest match before any paid request.

- [ ] **Step 2: Run capability waves with checkpoint reuse**

Use the same explicit output directory and source revision for every command:

```powershell
$azResultDir = 'C:\Users\Administrator\.codex\artifacts\pygent-az-211-0052cd35'
uv run --with websockets python -m tests.live.az_conformance.cli run --manifest tests/live/az_conformance/manifest.json --output-dir $azResultDir --scenario text
uv run --with websockets python -m tests.live.az_conformance.cli run --manifest tests/live/az_conformance/manifest.json --output-dir $azResultDir --scenario text_stream
uv run --with websockets python -m tests.live.az_conformance.cli run --manifest tests/live/az_conformance/manifest.json --output-dir $azResultDir --scenario tools
uv run --with websockets python -m tests.live.az_conformance.cli run --manifest tests/live/az_conformance/manifest.json --output-dir $azResultDir --scenario tool_choice
uv run --with websockets python -m tests.live.az_conformance.cli run --manifest tests/live/az_conformance/manifest.json --output-dir $azResultDir --scenario json_object
uv run --with websockets python -m tests.live.az_conformance.cli run --manifest tests/live/az_conformance/manifest.json --output-dir $azResultDir --scenario json_schema
uv run --with websockets python -m tests.live.az_conformance.cli run --manifest tests/live/az_conformance/manifest.json --output-dir $azResultDir --scenario reasoning
uv run --with websockets python -m tests.live.az_conformance.cli run --manifest tests/live/az_conformance/manifest.json --output-dir $azResultDir --scenario image_input
uv run --with websockets python -m tests.live.az_conformance.cli run --manifest tests/live/az_conformance/manifest.json --output-dir $azResultDir --scenario image_output
uv run --with websockets python -m tests.live.az_conformance.cli run --manifest tests/live/az_conformance/manifest.json --output-dir $azResultDir --scenario image_edit
uv run --with websockets python -m tests.live.az_conformance.cli run --manifest tests/live/az_conformance/manifest.json --output-dir $azResultDir --scenario video_output
uv run --with websockets python -m tests.live.az_conformance.cli run --manifest tests/live/az_conformance/manifest.json --output-dir $azResultDir --scenario audio_output
uv run --with websockets python -m tests.live.az_conformance.cli run --manifest tests/live/az_conformance/manifest.json --output-dir $azResultDir --scenario audio_input
uv run --with websockets python -m tests.live.az_conformance.cli run --manifest tests/live/az_conformance/manifest.json --output-dir $azResultDir --scenario realtime
uv run --with websockets python -m tests.live.az_conformance.cli run --manifest tests/live/az_conformance/manifest.json --output-dir $azResultDir --scenario embedding
uv run --with websockets python -m tests.live.az_conformance.cli run --manifest tests/live/az_conformance/manifest.json --output-dir $azResultDir --scenario search
```

The same literal external path must be used for all waves.

- [ ] **Step 3: Classify every failure before changing code**

For each failure, reproduce the exact route/protocol/scenario once and assign one of the closed error kinds. A code
change is allowed only for a Pygent/runner defect demonstrated by a new failing offline test. Configuration errors
are fixed in environment or manifest; gateway/upstream failures are retried only under the bounded policy and remain
active until a later pass for the exact key supersedes them.

- [ ] **Step 4: Commit each verified repair separately**

Run the focused RED/GREEN test, the affected provider suite, and then commit only that repair. Never delete a route,
remove a required scenario, or loosen response validation to turn a live failure green.

- [ ] **Step 5: Produce the exact report**

```powershell
$azResultDir = 'C:\Users\Administrator\.codex\artifacts\pygent-az-211-0052cd35'
uv run --with websockets python -m tests.live.az_conformance.cli report --manifest tests/live/az_conformance/manifest.json --output-dir $azResultDir
```

Expected:

```text
snapshot_routes=211
passed_routes=211
failed_routes=0
missing_routes=0
active_failed_scenarios=0
complete=true
```

### Task 13: Update public documentation and examples

**Files:**
- Modify: `docs/llm/README.md`
- Modify: `docs/llm/SDK.md`
- Modify: `docs/llm/PROVIDER_OPTIONS_SPEC.md`
- Modify: `docs/llm/FEATURES.md`
- Modify: `examples/live_agent/agent.py` only if a new production Adapter changes its setup

- [ ] **Step 1: Write documentation assertions where stable**

Extend existing documentation tests or catalog tests to assert every built-in Provider/protocol shown in SDK
examples exists and every sample Model ID resolves in `ModelCapabilityCatalog`.

- [ ] **Step 2: Document official Provider use**

Show `ModelConfig.from_mapping()`, an official Provider preset, a user-overridden connection, direct and managed
`ModelCallLayer`, and exact protocol selection. AZ must appear only in the opt-in conformance section, never as a
built-in Provider example.

- [ ] **Step 3: Document live verification safety**

Explain the frozen digest, external result directory, credential names, possible cost, drift failure, resume keys,
sanitized output, and exact 211/211 completion rule.

- [ ] **Step 4: Scan old and forbidden language**

```powershell
rg -n "gptplus5|AZ_BASE_URL|AZ_API_KEY" src examples docs/llm tests --glob '!tests/live/az_conformance/**' --glob '!docs/llm/AZ_211_*'
rg -n "streaming\.text|ModelProviderRegistry\.standard" src tests examples docs
```

Expected: no AZ production references and no resurrected legacy model API.

- [ ] **Step 5: Run docs-related tests and commit**

```powershell
uv run pytest tests/llm/test_builtin_catalogs.py tests/llm/test_model_config.py -q
git add docs/llm examples/live_agent/agent.py tests/llm
git commit -m "docs(llm): document official provider conformance"
```

### Task 14: Run the completion audit and release gates

**Files:**
- Modify only if verification exposes a defect
- Verify: source tree, live ledger, wheel, and sdist

- [ ] **Step 1: Audit every proposal requirement against evidence**

Create a local checklist mapping Sections 1–11 of the proposal to manifest tests, Adapter tests, live result keys,
catalog contents, docs, and release artifacts. Missing or indirect evidence means the goal remains incomplete.

- [ ] **Step 2: Re-run the exact 211 report on clean HEAD**

The ledger revision must equal current `git rev-parse HEAD`. If documentation-only commits changed HEAD after live
runs, either define and test a content-addressed execution revision that excludes docs or rerun required probes on
the final revision; do not silently accept stale keys.

- [ ] **Step 3: Run the full repository gates**

```powershell
uv run pytest -q
uv run ruff check src tests examples benchmarks
uv run mypy src benchmarks
uv build
uvx twine check dist/*
```

Expected: all commands exit zero.

- [ ] **Step 4: Compare packaged catalog bytes**

Compute SHA-256 for `providers.json`, `model_capabilities.json`, and `capability_presets.json` in source, wheel,
and sdist. For each filename, all three hashes must be identical.

- [ ] **Step 5: Run final security and scope scans**

Confirm:

- no secret value from `.env` appears in tracked files or command logs;
- no AZ base/key reference appears in production preset data;
- no gateway alias or external service appears in official model catalog;
- no live result or downloaded media is tracked;
- all 211 route IDs remain present with the frozen digest;
- `git status --short` is empty.

- [ ] **Step 6: Commit any final evidence-only documentation and report**

Do not commit the external result ledger. Commit only stable, sanitized aggregate evidence if the repository already
has an established location for it; otherwise report the live result in the handoff with its exact revision and
snapshot digest.

---

## Execution checkpoints

Stop for review after Tasks 5, 9, and 11. These checkpoints respectively prove:

1. the test foundation cannot falsely report 211/211;
2. official catalog expansion is separate from gateway aliases;
3. all required protocol/scenario pairs are executable before paid full runs.

Task 12 may take multiple sessions because media jobs, rate limits, and evidence-backed repairs are expected. The
goal remains active until Task 14 proves every proposal requirement and the exact 211/211 live result.
