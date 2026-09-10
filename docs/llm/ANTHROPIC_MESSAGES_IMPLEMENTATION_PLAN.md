# Anthropic Messages and Multi-Protocol Providers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add native Anthropic Messages support, DeepSeek OpenAI/Anthropic protocol selection, and lossless model continuation while preserving existing OpenAI Chat Completions behavior.

**Architecture:** Provider identity, wire protocol, semantic model configuration, and connection resources remain separate. A private JSON/SSE transport is shared by composition, while each protocol owns a per-call stateful stream decoder. `AIMessage` carries a provider/protocol-scoped opaque continuation through the existing portable-message and durable-execution paths.

**Tech Stack:** Python 3.11 dataclasses and protocols, asyncio, httpx, native HTTP extension, JSON/SSE, jsonschema, pytest, Ruff, mypy, Maturin.

**Design:** `docs/llm/ANTHROPIC_MESSAGES_PROPOSAL.md`

**Baseline:** The pre-change full suite produced `931 passed, 1 failed`; the only failure was `http-worker-invoke` observing a rounded `model_trace_ms == 0.0`, and its immediate isolated rerun passed. Do not modify that performance assertion as part of this feature.

---

## File structure

- Create `src/pygent/llm/protocols.py`: canonical built-in protocol identifiers.
- Create `src/pygent/llm/_json_sse_transport.py`: private JSON request and raw SSE transport, lifecycle, admission, TLS, and bounded errors.
- Create `src/pygent/llm/anthropic_messages.py`: Anthropic client, model catalog, request/response codec, stream decoder, option validation, and error normalization.
- Modify `src/pygent/core/values.py`: immutable `ModelContinuation` and `AIMessage.continuation`.
- Modify `src/pygent/runtime/codec.py`, `src/pygent/runtime/context_codec.py`, and `src/pygent/llm/layer.py`: continuation wire/effect persistence.
- Modify `src/pygent/llm/_adapter_contracts.py`, `src/pygent/llm/invoker.py`, and `src/pygent/llm/_stream_accumulator.py`: stateful decoder SPI and private continuation reduction.
- Modify `src/pygent/llm/openai_compatible.py`: use the shared transport, register the precise protocol name, and preserve DeepSeek reasoning continuation.
- Modify `src/pygent/llm/catalogs.py` and `src/pygent/llm/data/*.json`: per-protocol Provider presets and new built-in model entries.
- Modify public `__init__.py` files, SDK/docs/examples/tests, and release catalog checks to use the new public values.

### Task 1: Canonical protocol identifiers

**Files:**
- Create: `src/pygent/llm/protocols.py`
- Modify: `src/pygent/llm/configuration.py`
- Modify: `src/pygent/llm/__init__.py`
- Modify: `src/pygent/__init__.py`
- Modify: all tracked source, tests, examples, benchmarks, and docs containing `openai_compatible`
- Test: `tests/llm/test_model_config.py`
- Test: `tests/integration/test_public_api.py`

- [ ] **Step 1: Write failing protocol tests**

Add assertions that enum values are canonical, `ModelSpec` normalizes enum input to plain `str`, and the public API exposes the enum:

```python
def test_builtin_protocols_are_precise_and_model_spec_stores_plain_string():
    assert BuiltinModelProtocol.OPENAI_CHAT_COMPLETIONS.value == "openai_chat_completions"
    assert BuiltinModelProtocol.ANTHROPIC_MESSAGES.value == "anthropic_messages"
    spec = ModelSpec(
        provider="deepseek",
        model_id="deepseek-v4-pro",
        protocol=BuiltinModelProtocol.OPENAI_CHAT_COMPLETIONS,
        capabilities=text_tools_capabilities(),
    )
    assert spec.protocol == "openai_chat_completions"
    assert type(spec.protocol) is str
```

- [ ] **Step 2: Run the focused tests and confirm failure**

Run: `uv run pytest -q tests/llm/test_model_config.py tests/integration/test_public_api.py`

Expected: FAIL because `BuiltinModelProtocol` does not exist.

- [ ] **Step 3: Add the open built-in protocol helper**

Implement:

```python
class BuiltinModelProtocol(StrEnum):
    OPENAI_CHAT_COMPLETIONS = "openai_chat_completions"
    ANTHROPIC_MESSAGES = "anthropic_messages"
```

Keep `ModelSpec.protocol` open to strings. In `ModelSpec.__post_init__`, store `protocol.value` for the enum and otherwise validate/store the supplied string. Export the enum from `pygent.llm` and top-level `pygent`.

- [ ] **Step 4: Mechanically migrate the existing protocol key**

Replace the exact protocol value `openai_compatible` with `openai_chat_completions` in source registrations, model fixtures, examples, benchmarks, docs, JSON catalogs, Worker/durable fixtures, and assertions. Keep `OpenAICompatibleClient`, `OpenAICompatibleAdapter`, and `openai_compatible_adapters()` names unchanged. Do not register both protocol strings and do not add a codec fallback.

Run: `rg -n 'openai_compatible' src tests examples benchmarks docs .github`

Expected: remaining matches refer only to Python module/class/function names or prose describing compatibility; no configuration, catalog, adapter key, or serialized fixture uses the old value.

- [ ] **Step 5: Run protocol and existing OpenAI tests**

Run: `uv run pytest -q tests/llm/test_model_config.py tests/llm/test_model_spec_execution.py tests/llm/test_openai_compatible.py tests/integration/test_public_api.py`

Expected: PASS.

- [ ] **Step 6: Commit the protocol migration**

```text
git add src tests examples benchmarks docs .github
git commit -m "refactor(llm): name model wire protocols precisely"
```

### Task 2: Per-protocol Provider catalog presets

**Files:**
- Modify: `src/pygent/llm/catalogs.py`
- Modify: `src/pygent/llm/data/providers.json`
- Modify: `src/pygent/llm/__init__.py`
- Modify: `src/pygent/__init__.py`
- Test: `tests/llm/test_builtin_catalogs.py`

- [ ] **Step 1: Write failing catalog shape tests**

Assert that `deepseek.protocols` is an immutable mapping with independent endpoints and that Anthropic is present:

```python
def test_builtin_provider_catalog_projects_protocol_specific_connections():
    catalog = ProviderCatalog.builtin()
    deepseek = catalog.providers["deepseek"]
    assert deepseek.default_protocol == "openai_chat_completions"
    assert deepseek.protocols["openai_chat_completions"].base_url == "https://api.deepseek.com"
    assert deepseek.protocols["anthropic_messages"].base_url == "https://api.deepseek.com/anthropic"
    assert deepseek.protocols["anthropic_messages"].api_key_env == "DEEPSEEK_API_KEY"
    anthropic = catalog.providers["anthropic"]
    assert anthropic.protocols["anthropic_messages"].api_key_env == "ANTHROPIC_API_KEY"
```

Also test rejection of an array-valued `protocols`, unknown per-protocol fields, a mismatched `protocol` field/key, and a `default_protocol` missing from the mapping.

- [ ] **Step 2: Run the catalog tests and confirm failure**

Run: `uv run pytest -q tests/llm/test_builtin_catalogs.py`

Expected: FAIL because `ProviderPreset.protocols` is currently a tuple and connection defaults live at Provider level.

- [ ] **Step 3: Implement the strict new catalog values**

Add and export:

```python
@dataclass(frozen=True, slots=True)
class ProviderProtocolPreset:
    protocol: str
    base_url: str
    authentication: str
    api_key_env: str | None
    provider_options_schema: FrozenJsonObject

@dataclass(frozen=True, slots=True)
class ProviderPreset:
    provider: str
    display_name: str
    protocols: Mapping[str, ProviderProtocolPreset]
    default_protocol: str
```

Copy `protocols` into `MappingProxyType`. Parse only the new object shape and retain the current URL, authentication, credential, exact-field, and JSON-freezing checks at protocol level.

- [ ] **Step 4: Replace the built-in Provider JSON**

Write schema-version-1 data with `deepseek` and `anthropic`, each containing a protocol-keyed object. DeepSeek contains both endpoints and Anthropic contains only `anthropic_messages`. Keep secret values out of the file.

- [ ] **Step 5: Run focused tests**

Run: `uv run pytest -q tests/llm/test_builtin_catalogs.py tests/llm/test_model_config.py`

Expected: PASS.

- [ ] **Step 6: Commit the catalog refactor**

```text
git add src/pygent/llm/catalogs.py src/pygent/llm/data/providers.json src/pygent/llm/__init__.py src/pygent/__init__.py tests/llm/test_builtin_catalogs.py
git commit -m "refactor(llm): scope provider presets by protocol"
```

### Task 3: Portable and redacted model continuation

**Files:**
- Modify: `src/pygent/core/values.py`
- Modify: `src/pygent/core/__init__.py`
- Modify: `src/pygent/__init__.py`
- Modify: `src/pygent/runtime/codec.py`
- Modify: `src/pygent/llm/layer.py`
- Modify: `src/pygent/llm/_request_snapshot.py`
- Test: `tests/integration/test_module_context_contract.py`
- Test: `tests/runtime/test_wire_codec.py`
- Test: `tests/llm/test_request_snapshot.py`
- Test: `tests/integration/test_durable_effect_replay.py`

- [ ] **Step 1: Write failing immutable-value and redaction tests**

Cover defensive freezing, equality, repr redaction, wire round-trip, effect round-trip, and snapshot digest-only behavior:

```python
continuation = ModelContinuation(
    provider="deepseek",
    protocol="openai_chat_completions",
    data={"version": 1, "reasoning_content": "private-reasoning"},
)
message = AIMessage(content="answer", continuation=continuation)
assert "private-reasoning" not in repr(continuation)
assert "private-reasoning" not in repr(message)
assert message_from_dict(message_to_dict(message)) == message
event = prepared_request_event(request_with_history(message), attempt=1)
encoded = json.dumps(event)
assert "private-reasoning" not in encoded
assert event["request"]["messages"][0]["continuation_digest"].startswith("sha256:")
```

Require assistant wire values to contain `continuation`, including explicit `null`; old missing-field assistant values are rejected instead of silently upgraded.

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `uv run pytest -q tests/integration/test_module_context_contract.py tests/runtime/test_wire_codec.py tests/llm/test_request_snapshot.py tests/integration/test_durable_effect_replay.py`

Expected: FAIL because continuation is not defined or encoded.

- [ ] **Step 3: Implement the core value**

Add:

```python
@dataclass(frozen=True, slots=True)
class ModelContinuation:
    provider: str
    protocol: str
    data: JsonObjectInput = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        _require_non_empty_string(self.provider, "continuation provider")
        _require_non_empty_string(self.protocol, "continuation protocol")
        object.__setattr__(self, "data", freeze_json_object(self.data))
```

Add `continuation: ModelContinuation | None = field(default=None, repr=False)` to `AIMessage`, validate its exact type, and export it from core and top-level APIs.

- [ ] **Step 4: Encode continuation in portable and effect codecs**

Use one strict shape everywhere:

```python
{
    "provider": continuation.provider,
    "protocol": continuation.protocol,
    "data": thaw_json(continuation.data),
}
```

Always write `continuation` on assistant messages. Update runtime decode, context projection, model effect request, and effect replay to require either `null` or the exact three-field object.

- [ ] **Step 5: Add digest-only request snapshots**

Canonicalize `{provider, protocol, data}` with sorted compact JSON and SHA-256. Add only `continuation_digest` to AI-message projections; never include raw data.

- [ ] **Step 6: Run focused tests**

Run: `uv run pytest -q tests/integration/test_module_context_contract.py tests/runtime/test_wire_codec.py tests/llm/test_request_snapshot.py tests/integration/test_durable_effect_replay.py`

Expected: PASS.

- [ ] **Step 7: Commit the portable continuation contract**

```text
git add src/pygent/core src/pygent/runtime/codec.py src/pygent/llm/layer.py src/pygent/llm/_request_snapshot.py src/pygent/__init__.py tests
git commit -m "feat(core): persist opaque model continuation"
```

### Task 4: Stateful stream decoder and continuation reduction

**Files:**
- Modify: `src/pygent/llm/_adapter_contracts.py`
- Modify: `src/pygent/llm/invoker.py`
- Modify: `src/pygent/llm/_stream_accumulator.py`
- Modify: `src/pygent/llm/openai_compatible.py`
- Modify: `src/pygent/llm/__init__.py`
- Test: `tests/llm/test_invoker.py`
- Test: `tests/llm/test_openai_compatible.py`

- [ ] **Step 1: Write failing decoder lifecycle tests**

Define test decoders that record `feed()` calls and reject premature EOF. Verify a new decoder is created for every attempt, continuation parts are not public events, reset clears continuation, and non-streaming responses pass through the same accumulator result.

```python
class RecordingDecoder:
    def feed(self, payload):
        return (ModelProviderStreamPart("text", {"text": payload["text"]}),)

    def finish(self):
        return (ModelProviderStreamPart("finish", {"finish_reason": "stop"}),)
```

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `uv run pytest -q tests/llm/test_invoker.py tests/llm/test_openai_compatible.py`

Expected: FAIL because adapters expose stateless `parse_stream_events()`.

- [ ] **Step 3: Replace the stream Adapter SPI**

Add:

```python
class ModelProviderStreamDecoder(Protocol):
    def feed(self, payload: FrozenJsonObject) -> tuple[ModelProviderStreamPart, ...]: ...
    def finish(self) -> tuple[ModelProviderStreamPart, ...]: ...

class ModelProviderAdapter(Protocol):
    protocol: str
    def create_stream_decoder(self, request: ModelProviderRequest) -> ModelProviderStreamDecoder: ...
```

Remove `parse_stream_events()` from the SPI. Add `CONTINUATION = "continuation"` to `ModelProviderStreamKind`. A continuation part contains exactly `provider`, `protocol`, and `data`; it is an internal reduction item, not a `ModelEventKind`.

- [ ] **Step 4: Drive one decoder per attempt in the invoker**

Create the decoder after request construction, feed each client payload, call `finish()` after normal iterator exhaustion, and run every returned part through the existing attempt identity and finish validation. For non-stream calls, emit usage, continuation, text, tool calls, and finish in that order before reduction.

- [ ] **Step 5: Retain continuation only for the successful attempt**

Add `continuation: ModelContinuation | None` to `ModelStreamAccumulator`. Construct it from strict continuation parts, clear it on RESET, attach it to the final `AIMessage`, and never emit an event for it. Do not mark a continuation part as public output for retry/fallback gating.

- [ ] **Step 6: Convert OpenAI parsing to a per-call decoder without changing output**

Move the current chunk logic into a decoder instance. `feed()` preserves current text/reasoning/tool/usage/finish parts; `finish()` rejects a stream that never received the existing completion marker. Do not add continuation behavior in this step.

- [ ] **Step 7: Run focused stream and invoker tests**

Run: `uv run pytest -q tests/llm/test_invoker.py tests/llm/test_openai_compatible.py tests/llm/test_model_layer.py`

Expected: PASS with unchanged public OpenAI events.

- [ ] **Step 8: Commit the stateful decoder contract**

```text
git add src/pygent/llm tests/llm
git commit -m "refactor(llm): decode provider streams per call"
```

### Task 5: Shared private JSON/SSE transport

**Files:**
- Create: `src/pygent/llm/_json_sse_transport.py`
- Modify: `src/pygent/llm/openai_compatible.py`
- Test: `tests/llm/test_json_sse_transport.py`
- Test: `tests/llm/test_openai_compatible.py`

- [ ] **Step 1: Add characterization and lifecycle tests**

Test JSON success/invalid body, bounded HTTP error body, raw SSE frames, comments, event/data framing, injected-client ownership, native-client ownership, admission release, cancellation, idempotent close, and close while active work drains. Retain all existing OpenAI transport tests as characterization coverage.

- [ ] **Step 2: Run transport tests and confirm failure**

Run: `uv run pytest -q tests/llm/test_json_sse_transport.py tests/llm/test_openai_compatible.py`

Expected: FAIL because `_JsonSSETransport` does not exist.

- [ ] **Step 3: Implement the private transport boundary**

Implement these private contracts:

```python
@dataclass(frozen=True, slots=True)
class _SSEFrame:
    event: str | None
    data: str

@dataclass(frozen=True, slots=True)
class _HTTPResponseError(Exception):
    status: int
    body: bytes

class _JsonSSETransport:
    async def request_json(self, method: str, url: str, payload: Mapping[str, object] | None, *, timeout: float | None = None) -> FrozenJsonObject: ...
    async def stream_sse(self, url: str, payload: Mapping[str, object]) -> AsyncIterator[_SSEFrame]: ...
    async def aclose(self) -> None: ...
```

The transport owns HTTP mechanics only. It does not recognize `[DONE]`, Provider error schemas, model catalogs, or completion payloads. Preserve the existing native/httpx selection, TLS rules, proxy bypass, pool admission, drain grace, bounded error size, and cancellation ordering.

- [ ] **Step 4: Refactor OpenAI client by composition**

Keep the public constructor, properties, context-manager behavior, and exceptions stable. Let the client translate `_HTTPResponseError` with the existing OpenAI error mapper, parse `[DONE]`, and parse each JSON SSE frame before yielding `FrozenJsonObject` to the invoker.

- [ ] **Step 5: Run transport and all OpenAI tests**

Run: `uv run pytest -q tests/llm/test_json_sse_transport.py tests/llm/test_openai_compatible.py tests/llm/test_invoker.py`

Expected: PASS.

- [ ] **Step 6: Commit the transport extraction**

```text
git add src/pygent/llm/_json_sse_transport.py src/pygent/llm/openai_compatible.py tests/llm
git commit -m "refactor(llm): share private JSON SSE transport"
```

### Task 6: DeepSeek reasoning continuation on Chat Completions

**Files:**
- Modify: `src/pygent/llm/openai_compatible.py`
- Modify: `src/pygent/llm/_stream_accumulator.py`
- Test: `tests/llm/test_openai_compatible.py`
- Test: `tests/llm/test_invoker.py`

- [ ] **Step 1: Write failing non-stream, stream, and tool-loop tests**

Use `provider="deepseek"` responses containing `reasoning_content`. Verify the returned continuation is:

```python
ModelContinuation(
    provider="deepseek",
    protocol="openai_chat_completions",
    data={"version": 1, "reasoning_content": "reasoning"},
)
```

Verify a subsequent assistant tool-call message sends the exact `reasoning_content`; malformed matching continuation fails before client I/O; a continuation with another provider or protocol is omitted; official OpenAI models never fabricate this continuation.

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `uv run pytest -q tests/llm/test_openai_compatible.py tests/llm/test_invoker.py`

Expected: FAIL because reasoning is emitted but discarded.

- [ ] **Step 3: Implement non-stream continuation**

When `request.model.provider == "deepseek"`, validate a string `message.reasoning_content`, attach the versioned continuation, and include it when encoding a matching historical `AIMessage`. Do not apply this path to other Providers sharing Chat Completions.

- [ ] **Step 4: Implement stream continuation**

Accumulate non-empty DeepSeek reasoning deltas in the per-call decoder. Immediately before its successful finish part, emit one canonical continuation part. Preserve existing reasoning delta events.

- [ ] **Step 5: Verify retry and fallback isolation**

Add a test where an attempt emits reasoning and fails before completion. Confirm reset removes it and the succeeding attempt's `AIMessage` contains only its own continuation.

- [ ] **Step 6: Run focused tests**

Run: `uv run pytest -q tests/llm/test_openai_compatible.py tests/llm/test_invoker.py tests/agent/test_react.py`

Expected: PASS.

- [ ] **Step 7: Commit DeepSeek continuation**

```text
git add src/pygent/llm tests/llm tests/agent/test_react.py
git commit -m "feat(llm): preserve DeepSeek reasoning continuation"
```

### Task 7: Anthropic client, request codec, response codec, and errors

**Files:**
- Create: `src/pygent/llm/anthropic_messages.py`
- Modify: `src/pygent/llm/__init__.py`
- Test: `tests/llm/test_anthropic_messages.py`
- Test: `tests/llm/test_provider_options.py`

- [ ] **Step 1: Write failing client and non-stream Adapter tests**

Cover endpoint joining for Anthropic and DeepSeek base URLs, headers, injected/native client ownership, `/v1/models`, required `max_output_tokens`, system/user/assistant/tool-result blocks, tool schemas, all tool choices, response schema, usage, request ID, thinking/redacted-thinking layout, and strict completion shapes.

Assert representative request fields:

```python
assert payload["model"] == "claude-opus-5"
assert payload["max_tokens"] == 4096
assert payload["system"] == "system"
assert payload["tools"][0]["input_schema"]["type"] == "object"
assert payload["tool_choice"] == {"type": "any"}
assert payload["output_config"]["format"] == {
    "type": "json_schema",
    "schema": response_schema,
}
```

- [ ] **Step 2: Write failing Provider-option tests**

Test the three exact thinking unions, optional display, `budget_tokens >= 1024`, budget below max tokens, allowed effort/service tier/stop sequences, response-schema ownership of `output_config.format`, temperature constraints, recursive secret rejection, and unknown-field rejection before I/O.

- [ ] **Step 3: Run focused tests and confirm failure**

Run: `uv run pytest -q tests/llm/test_anthropic_messages.py tests/llm/test_provider_options.py`

Expected: FAIL because the module and protocol Adapter do not exist.

- [ ] **Step 4: Implement `AnthropicMessagesClient`**

Compose `_JsonSSETransport`; append `/v1/messages` and `/v1/models` to `base_url.rstrip("/")`; set `x-api-key`, `anthropic-version: 2023-06-01`, and caller headers without logging values. Implement `models`, `invoke`, `stream`, `aclose`, and async context-manager behavior under the existing `ModelProviderClient` contract. Project Anthropic's RFC 3339 model creation time as `ModelInfo.created=None` and set `owned_by="anthropic"`.

- [ ] **Step 5: Implement strict request construction**

Build Anthropic content blocks from Pygent messages and tools. Encode each `ToolResult` into one `tool_result` with `is_error = result.status != "succeeded"`. Merge generic JSON schema into `output_config.format` and validated Provider effort into the same `output_config` object. Reject missing max tokens and invalid tool choice before I/O.

- [ ] **Step 6: Implement non-stream response and continuation parsing**

Flatten text blocks into `AIMessage.content`, decode `tool_use` into ordered `ToolCall` values, preserve thinking/redacted blocks plus text ranges and tool indices in continuation layout version 1, normalize usage, and map stop reasons exactly as specified. Validate that text ranges cover the flattened text without overlap and every tool reference is in range.

- [ ] **Step 7: Implement Anthropic error normalization**

Map status and `error.type` onto existing `ModelErrorKind` and `ModelFailureReason`; retain HTTP status and bounded diagnostics, never the credential or full body. Treat 402/spend-limit errors as quota exhausted, 529 as unavailable, 504 as timeout, and malformed errors as the status-derived fallback.

- [ ] **Step 8: Run focused tests**

Run: `uv run pytest -q tests/llm/test_anthropic_messages.py tests/llm/test_provider_options.py tests/llm/test_model_spec_execution.py`

Expected: PASS.

- [ ] **Step 9: Commit the Anthropic non-stream path**

```text
git add src/pygent/llm/anthropic_messages.py src/pygent/llm/__init__.py tests/llm
git commit -m "feat(llm): add Anthropic Messages adapter"
```

### Task 8: Anthropic SSE and thinking/tool continuation

**Files:**
- Modify: `src/pygent/llm/anthropic_messages.py`
- Test: `tests/llm/test_anthropic_messages.py`
- Test: `tests/llm/test_invoker.py`
- Test: `tests/agent/test_react.py`

- [ ] **Step 1: Write failing stream-sequence tests**

Cover `message_start`, indexed `content_block_start`, `text_delta`, `thinking_delta`, `signature_delta`, `input_json_delta`, `content_block_stop`, `message_delta`, `message_stop`, ping, and streamed error. Verify multiple text blocks and tool blocks retain exact layout and one continuation is emitted only on successful completion.

- [ ] **Step 2: Write failing invalid-stream tests**

Reject duplicate/open block indices, deltas for unopened blocks, malformed partial JSON, missing signature, block type changes, premature EOF, `pause_turn`, and unknown content/delta/stop semantics. Verify `model_context_window_exceeded` maps to `CONTEXT_LENGTH_EXCEEDED`. Ignore only safe non-content events such as ping and unknown top-level events that carry no output or terminal semantics.

- [ ] **Step 3: Run focused tests and confirm failure**

Run: `uv run pytest -q tests/llm/test_anthropic_messages.py tests/llm/test_invoker.py`

Expected: FAIL because the Anthropic stream decoder is incomplete.

- [ ] **Step 4: Implement the per-call Anthropic decoder**

Maintain indexed block state inside the decoder, emit public reasoning/text/tool deltas as they arrive, assemble signatures and tool JSON privately, canonicalize usage on `message_delta`, and emit continuation immediately before the successful finish. `finish()` must reject EOF unless `message_stop` completed the message.

- [ ] **Step 5: Implement continuation replay**

For matching provider/protocol only, validate version 1 and rebuild the prior assistant blocks by slicing `AIMessage.content`, indexing `AIMessage.tool_calls`, and inserting stored thinking/redacted blocks unchanged. Reject modified, missing, overlapping, duplicate, or out-of-range references before I/O.

- [ ] **Step 6: Verify agent tool loops**

Run one synthetic ReAct loop for Anthropic official and one for DeepSeek over `anthropic_messages`. Assert the second request contains the first response's exact thinking/signature blocks followed by tool results.

- [ ] **Step 7: Run focused tests**

Run: `uv run pytest -q tests/llm/test_anthropic_messages.py tests/llm/test_invoker.py tests/agent/test_react.py`

Expected: PASS.

- [ ] **Step 8: Commit Anthropic streaming**

```text
git add src/pygent/llm/anthropic_messages.py tests/llm tests/agent/test_react.py
git commit -m "feat(llm): stream Anthropic messages with continuation"
```

### Task 9: Durable, Worker, and managed-runtime propagation

**Files:**
- Modify: `tests/runtime/test_wire_codec.py`
- Modify: `tests/runtime/test_http_worker.py`
- Modify: `tests/runtime/test_durable_runtime.py`
- Modify: `tests/runtime/test_sqlite_history.py`
- Modify: `tests/integration/test_durable_effect_replay.py`
- Modify: `tests/llm/test_model_layer.py`
- Test: `tests/runtime/test_wire_codec.py`
- Test: `tests/runtime/test_http_worker.py`
- Test: `tests/runtime/test_durable_runtime.py`
- Test: `tests/runtime/test_sqlite_history.py`
- Test: `tests/integration/test_durable_effect_replay.py`
- Test: `tests/llm/test_model_layer.py`

- [ ] **Step 1: Add end-to-end continuation round-trip tests**

Construct an `AIMessage` with continuation in Context, pass it through local managed execution, HTTP Worker encoding, SQLite-backed checkpoint/history, and committed model-effect replay, and assert exact equality after every boundary.

- [ ] **Step 2: Add direct/managed/fallback tests**

Verify direct and managed calls produce identical continuation, fallback clears a failed attempt's continuation, and switching to another protocol omits the mismatched continuation without altering the visible AI text/tool calls.

- [ ] **Step 3: Run the integration tests and confirm any missing projections**

Run: `uv run pytest -q tests/runtime/test_wire_codec.py tests/runtime/test_http_worker.py tests/runtime/test_durable_runtime.py tests/runtime/test_sqlite_history.py tests/integration/test_durable_effect_replay.py tests/llm/test_model_layer.py`

Expected before fixture/projection completion: FAIL at each remaining strict assistant-message boundary; no raw continuation should appear in model events.

- [ ] **Step 4: Complete strict current-schema fixtures and projections**

Update current Worker/effect/context fixtures to carry explicit continuation values. Do not add version branching or missing-field defaults. Ensure execution definition/profile serialization changes only where the renamed protocol or strict Message shape already requires it.

- [ ] **Step 5: Run the integration tests**

Run the command from Step 3.

Expected: PASS.

- [ ] **Step 6: Commit runtime propagation**

```text
git add src tests/runtime tests/integration tests/llm/test_model_layer.py
git commit -m "test(runtime): preserve model continuation across execution"
```

### Task 10: Built-in model catalogs, docs, examples, and release assets

**Files:**
- Modify: `src/pygent/llm/data/model_capabilities.json`
- Modify: `src/pygent/llm/data/providers.json`
- Modify: `docs/llm/README.md`
- Modify: `docs/llm/SDK.md`
- Modify: `docs/llm/PROVIDER_OPTIONS_SPEC.md`
- Modify: `docs/llm/FEATURES.md`
- Modify: `docs/llm/DYNAMIC_MODEL_GROUP_SPEC.md`
- Modify: `docs/llm/DYNAMIC_MODEL_GROUP_IMPLEMENTATION.md`
- Modify: `docs/llm/MODEL_ACCESS_ARCHITECTURE_PROPOSAL.md`
- Modify: `docs/runtime/README.md`
- Modify: `docs/runtime/DURABILITY.md`
- Modify: `examples/tutorial/providers.py`
- Modify: `examples/tutorial/agent.py`
- Modify: `examples/live_agent/agent.py`
- Modify: `examples/service/models.py`
- Modify: `benchmarks/models.py`
- Test: `tests/llm/test_builtin_catalogs.py`
- Test: `tests/examples/test_tutorial.py`

- [ ] **Step 1: Write failing built-in catalog assertions**

Assert the exact `(provider, model_id, protocol)` set for DeepSeek's two protocols and the four Anthropic models. Check 1M/128K limits for Fable 5.1, Opus 5, and Sonnet 5; 200K/64K for Haiku 4.5; and text/image input, text output, streaming, tools, structured output, and reasoning capabilities.

- [ ] **Step 2: Update immutable catalog snapshots**

Add DeepSeek Anthropic-protocol capability entries and Anthropic entries for:

```text
claude-fable-5-1
claude-opus-5
claude-sonnet-5
claude-haiku-4-5-20251001
```

Keep the capability key as `(provider, model_id, protocol)`. Do not add retired aliases, experimental vision models, or an OpenAI Responses catalog entry.

- [ ] **Step 3: Update public documentation and examples**

Document `BuiltinModelProtocol`, per-protocol Provider presets, Mapping loading, Anthropic required max tokens, single-model direct/managed use, DeepSeek protocol selection, continuation behavior, strict options, and custom Provider strings. Use `model_layer` consistently and remove configuration uses of `openai_compatible`.

- [ ] **Step 4: Verify package/release catalog inclusion**

Confirm wheel and sdist contain all three JSON files unchanged. Inspect `.github/workflows/python-publish.yml` and confirm it still attaches the same filenames plus `SHA256SUMS`; no workflow edit is required because filenames remain unchanged.

- [ ] **Step 5: Run catalog, example, and release tests**

Run: `uv run pytest -q tests/llm/test_builtin_catalogs.py tests/examples`

Expected: PASS.

- [ ] **Step 6: Scan stale contracts**

Run: `rg -n 'openai_compatible' src tests examples benchmarks docs .github`

Expected: no old protocol values. Class/module/function identifiers may remain only where preserving the existing public OpenAI-compatible client API is intentional.

- [ ] **Step 7: Commit catalogs and documentation**

```text
git add src/pygent/llm/data docs examples tests .github
git commit -m "docs(llm): publish Anthropic and DeepSeek protocol presets"
```

### Task 11: Full verification and live smoke checks

**Files:**
- Modify: only files required to fix failures caused by this proposal
- Test: complete repository

- [ ] **Step 1: Run the complete unit/integration suite**

Run: `uv run pytest -q`

Expected: all tests pass. If the known `model_trace_ms == 0.0` performance timing failure recurs, rerun that exact parameter once and report both results without changing its assertion in this feature.

- [ ] **Step 2: Run static verification**

```text
uv run ruff check src tests examples benchmarks
uv run mypy src benchmarks
```

Expected: both commands exit 0.

- [ ] **Step 3: Build and inspect distributions**

```text
uv build
uvx twine check dist/*
```

Expected: build succeeds, Twine reports every artifact `PASSED`, and wheel/sdist inspection finds `providers.json`, `model_capabilities.json`, and `capability_presets.json`.

- [ ] **Step 4: Run opt-in real-provider smoke checks**

When `ANTHROPIC_API_KEY` is available, execute one minimal `claude-haiku-4-5-20251001` Messages call with a small explicit output limit. When `DEEPSEEK_API_KEY` is available, execute `deepseek-v4-flash` once through each protocol. Assert only success shape, tool-loop continuation where affordable, and non-empty usage; do not print credentials, response text, or thinking.

- [ ] **Step 5: Review security and lifecycle invariants**

Confirm no secret/continuation content appears in repr, events, error messages, snapshots, or test output; transport closure is idempotent; cancellation waits for active stream cleanup; adapters hold no cross-call mutable state; fallback releases each client/admission resource exactly once.

- [ ] **Step 6: Record verification and final status**

The isolated worktree has no `.doc_project_maintainer/` artifact, so report expanded maintenance with `task_slice_sync_status: unavailable`; do not create an artifact or integrity key. Include the pre-change performance timing result and final applicable verification evidence in the handoff.
