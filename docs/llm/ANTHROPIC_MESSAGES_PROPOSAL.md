# Anthropic Messages 与模型续传 Proposal

## 目标

Pygent 增加原生 `anthropic_messages` 协议支持，并让同一个 Provider 可以通过不同协议访问模型。实现保持 Provider、协议、模型语义和连接资源彼此分离：

- `ModelSpec.provider` 表示模型服务来源；
- `ModelSpec.protocol` 决定请求格式和 Adapter；
- `ModelConnection` 决定端点、凭据引用和 TLS；
- client 继续按 `ModelEntry.name` 绑定；
- `ModelCallLayer`、Invoker、Runtime、fallback、Binding 和资源生命周期继续使用现有语义。

第一版直接支持 Anthropic 官方 Messages API，并让 DeepSeek 官方 Provider 可以选择 `openai_chat_completions` 或 `anthropic_messages`。实现不引入 Anthropic Python SDK，不增加 Provider registry、自动协议探测、能力路由或新的容量系统。

## Provider 与协议

第一版内置以下连接 preset：

| Provider | Protocol | Base URL | Credential environment |
| --- | --- | --- | --- |
| `anthropic` | `anthropic_messages` | `https://api.anthropic.com` | `ANTHROPIC_API_KEY` |
| `deepseek` | `openai_chat_completions` | `https://api.deepseek.com` | `DEEPSEEK_API_KEY` |
| `deepseek` | `anthropic_messages` | `https://api.deepseek.com/anthropic` | `DEEPSEEK_API_KEY` |

协议不根据 URL、模型名或 Provider 自动猜测。`ModelConfig` 为同一个配置键产生语义侧的完整 `ModelEntry` 和资源侧的 `ModelConnection`；其中 protocol 与 connection 必须成对配置。同一模型通过不同协议访问时是两个独立条目，可以进入同一个 `ModelGroup`，顺序仍然表示 fallback 顺序。

`ModelContinuation` 不跨协议转换。进行中的 thinking/tool loop 应继续使用产生该 continuation 的 Provider 和协议；协议不同的后续调用不会携带该 continuation。

DeepSeek 官方同时公开 OpenAI 与 Anthropic 格式端点，具体兼容字段以 [DeepSeek Anthropic API 文档](https://api-docs.deepseek.com/guides/anthropic_api/) 为准。

Protocol 使用具体 wire API 的名称，不使用 Provider 名称代替协议：

- OpenAI Chat Completions：`openai_chat_completions`；
- Anthropic Messages：`anthropic_messages`；
- 未来实现 OpenAI Responses 时使用 `openai_responses`。

ModelScope、DeepSeek 和其他模型平台属于 Provider；如果它们实现相同 wire API，就复用相同 protocol 和 Adapter。只有 wire contract 不同才增加新的 protocol。

Pygent 为已经内置 Adapter 的协议提供便利枚举：

```python
class BuiltinModelProtocol(StrEnum):
    OPENAI_CHAT_COMPLETIONS = "openai_chat_completions"
    ANTHROPIC_MESSAGES = "anthropic_messages"
```

`ModelSpec.protocol` 仍然是开放字符串语义，也接受 `BuiltinModelProtocol`，构造时统一保存为普通字符串。第三方 Adapter 可以使用自定义 protocol，不需要修改该枚举。Provider catalog、Mapping 和持久化格式始终保存规范字符串。未来增加内置 Responses Adapter 时，再同时增加 `BuiltinModelProtocol.OPENAI_RESPONSES`。

现有 `openai_compatible` protocol 标识统一迁移为 `openai_chat_completions`。当前代码、目录、文档、测试和持久化 schema 一次性使用新名称；不保留字符串别名、旧 snapshot reader 或双协议注册。

Provider 目录按协议保存连接默认值，因为同一个 Provider 的不同 wire API 可以使用不同 base URL：

```python
ProviderPreset(
    provider: str,
    display_name: str,
    protocols: Mapping[str, ProviderProtocolPreset],
    default_protocol: str,
)

ProviderProtocolPreset(
    protocol: str,
    base_url: str,
    authentication: str,
    api_key_env: str | None,
    provider_options_schema: FrozenJsonObject,
)
```

旧目录中 Provider 级的 `base_url`、`authentication`、`api_key_env` 和 `provider_options_schema` 删除，目录解析不接受两种结构。`default_protocol` 必须引用 `protocols` 中的一个键；每个 `ProviderProtocolPreset.protocol` 与所在 Mapping 键相同。

Provider catalog 是 Pygent 维护的离线基础数据。普通 Agent 开发者和用户不需要编写它；构建模型配置 UI 或加载外部目录的应用开发者通过 `ProviderCatalog.builtin()` 或 `ProviderCatalog.from_mapping()` 使用它。UI 根据用户选择的 Provider、protocol 和 model 展开默认值，最终保存的仍是现有扁平 `ModelConfig`。Runtime、Agent 和 `ModelCallLayer` 不读取 Provider catalog。

## Client 与 Transport

抽取私有、协议无关的 `_JsonSSETransport`，统一负责：

- JSON 请求与响应；
- SSE 传输；
- 原生 HTTP 与注入 httpx client 的选择；
- TLS、连接池和 admission；
- deadline、idle timeout、取消和关闭；
- 有界错误响应体。

`OpenAICompatibleClient` 与新增的 `AnthropicMessagesClient` 通过组合使用该 transport，不通过继承共享 Provider 行为。transport 不解释 Provider payload。

`OpenAICompatibleClient` 的公开构造方式和现有行为保持不变。`AnthropicMessagesClient` 使用相同风格的 `base_url`、`api_key`、`client` 和 `verify_ssl` 参数，调用 `/v1/messages`，使用 `x-api-key` 认证并固定发送 `anthropic-version: 2023-06-01`。API version 是实现拥有的 wire 版本，不进入 `ModelSpec.provider_options`。

Anthropic 的 `/v1/models` 结果继续投影为现有 `ModelCatalog`。RFC 3339 `created_at` 不强行转换成整数，`ModelInfo.created` 使用 `None`，`owned_by` 使用 `anthropic`。

## Anthropic Messages Adapter

新增 `AnthropicMessagesAdapter`，并以 `protocol == "anthropic_messages"` 注册。Adapter 负责以下映射：

- Context system prompt 映射为顶层 `system`；
- UserMessage 映射为 user text block；
- AIMessage 文本与工具调用映射为 assistant `text` 和 `tool_use` block；
- ToolMessage 映射为 user `tool_result` block；
- ToolDefinition schema 映射为 `input_schema`；
- `tool_choice` 的 `auto`、`required`、`none` 和具名工具分别映射为 `auto`、`any`、`none` 和 `tool`；
- `GenerationConfig.response_schema` 映射为 `output_config.format` JSON schema；
- `GenerationConfig.temperature` 映射为 `temperature`；
- `GenerationConfig.max_output_tokens` 映射为必填的 `max_tokens`。

Anthropic 调用缺少 `GenerationConfig.max_output_tokens` 时，在 Provider I/O 前失败。框架不偷偷使用模型能力上限，也不设置隐藏默认值。

第一版 Adapter 只发送当前 Pygent Message 和 Tool 契约能够无损表达的输入。目录可以如实声明模型具有图像能力，但这不改变当前 Message 的输入类型。

## Provider Options

Anthropic 协议接受以下严格 `provider_options`：

```python
provider_options={
    "thinking": {
        "type": "adaptive",
        "display": "omitted",
    },
    "output_config": {
        "effort": "high",
    },
    "service_tier": "auto",
    "stop_sequences": ["<END>"],
}
```

`thinking` 只接受三种结构：

- `{"type": "adaptive", "display": "summarized" | "omitted"}`；
- `{"type": "disabled"}`；
- `{"type": "enabled", "budget_tokens": N, "display": "summarized" | "omitted"}`。

`display` 可省略。手动 thinking 的 `budget_tokens` 必须至少为 `1024`，并小于本次调用的 `max_output_tokens`。thinking 开启时，temperature 只能省略或为 `1`。

`output_config` 在 provider options 中只允许 `effort`，取值为 `low`、`medium`、`high`、`xhigh` 或 `max`。`output_config.format` 由通用 `response_schema` 独占。`service_tier` 只接受 `auto` 或 `standard_only`；`stop_sequences` 必须是非空字符串列表。未知字段和协议层可以确定的非法组合在 I/O 前拒绝。

Adapter 不硬编码具体 Claude 型号对 adaptive、manual thinking、effort 或 temperature 的支持矩阵。用户依据模型资料配置，型号相关限制最终由 Provider 校验；Adapter 只维护稳定的协议结构和可可靠判断的交叉字段约束。

## ModelContinuation

新增不可变公开值：

```python
ModelContinuation(
    provider: str,
    protocol: str,
    data: Mapping[str, JsonValue],
)
```

并为 `AIMessage` 增加：

```python
continuation: ModelContinuation | None = None
```

Continuation 是一次模型输出的后续调用状态，不属于 `ModelSpec`、模型目录或部署资源。构造时对 `data` 防御性复制并递归冻结。它必须通过 Message/Context codec、Worker、checkpoint、SQLite 持久化和 durable replay 保持不变。

Continuation 不进入 repr、普通模型事件、异常文本或日志。请求快照只记录 continuation digest，不记录原始 data。

Adapter 只消费 `provider` 与 `protocol` 同时匹配的 continuation。匹配但结构或版本非法时必须在 I/O 前拒绝；不匹配时不发送，不尝试协议转换。

### Anthropic continuation

Anthropic continuation 使用版本化的紧凑 block layout：

```python
ModelContinuation(
    provider="anthropic",
    protocol="anthropic_messages",
    data={
        "version": 1,
        "blocks": (...),
    },
)
```

`blocks` 原样保存必须回传的 `thinking`、`redacted_thinking` 和签名；文本 block 使用 `AIMessage.content` 的区间引用，工具 block 使用 `AIMessage.tool_calls` 的索引引用。该结构保存重建 assistant content 所需的顺序，但不复制整份 Provider 响应。

DeepSeek 通过 `anthropic_messages` 返回的 thinking block 使用相同布局，但 continuation 的 `provider` 为 `deepseek`。DeepSeek 文档明确不支持 `redacted_thinking`，Adapter 按实际返回结构严格处理。

### OpenAI-compatible continuation

DeepSeek 的 OpenAI-compatible thinking 使用：

```python
ModelContinuation(
    provider="deepseek",
    protocol="openai_chat_completions",
    data={
        "version": 1,
        "reasoning_content": "...",
    },
)
```

后续工具调用请求将 `reasoning_content` 放回对应 assistant message。官方 OpenAI Chat Completions 响应不生成该 continuation。

## 有状态流解码

Adapter 的流解析改为每次模型调用创建独立的流解码器。解码器持有单次调用的可变解析状态；Adapter 本身不保存调用状态，因此仍可并发复用。

- OpenAI-compatible 解码器保持现有文本、工具、usage 和结束事件行为，并为 DeepSeek 累积 `reasoning_content`；
- Anthropic 解码器组合 content block、thinking delta、signature delta、tool input JSON delta、usage 和终止事件；
- 解码器在 block 或消息结构不完整时返回现有 invalid-response 错误。

Provider-neutral stream vocabulary 增加 continuation part。它只由内部 accumulator 消费，不产生公开事件，也不计为已向调用方输出内容。retry 或 fallback reset 会清除失败 attempt 的 continuation。成功完成后，accumulator 将它写入最终 `AIMessage.continuation`。

非流式 Provider response 同样转换为 continuation part 后进入 accumulator，使流式与非流式只在 transport 上不同，不形成两套结果语义。

Anthropic 可展示的 thinking summary 继续产生 reasoning delta 事件；签名、redacted thinking 和不可展示的 continuation 数据不进入事件。

## 结束原因与错误

Anthropic stop reason 映射如下：

| Anthropic | Pygent |
| --- | --- |
| `end_turn`, `stop_sequence` | `stop` |
| `tool_use` | `tool_calls` |
| `max_tokens` | `length` / `OUTPUT_LIMIT_REACHED` |
| `refusal` | `content_filter` / `CONTENT_POLICY_REJECTED` |
| `model_context_window_exceeded` | `context_length` / `CONTEXT_LENGTH_EXCEEDED` |

`pause_turn` 第一版明确拒绝，因为当前范围不包含 Provider 管理的 server-tool continuation。未知 stop reason 也明确拒绝，不能被当成普通成功。

HTTP 与流内错误复用现有错误体系：

- `401` 映射 authentication；
- `402` 或明确消费额度错误映射 quota exhausted；
- `403` 映射 permission denied；
- `404` 映射 model/resource not found；
- `429` 映射 rate limited；
- `500`、`529` 映射 unavailable；
- `504` 映射 timeout；
- 其他请求问题映射 invalid request；
- 非法响应和非法 SSE 映射 invalid response。

Provider request ID 保留在现有响应与完成事件字段中。未知的非内容 SSE 事件可以忽略；未知 content block、delta 或终止事件必须拒绝，防止静默丢失输出。

## 内置目录

Anthropic Provider preset 使用 `anthropic_messages`、官方 base URL、`ANTHROPIC_API_KEY` 和固定 API version。第一版能力目录收录当前官方模型：

- `claude-fable-5-1`；
- `claude-opus-5`；
- `claude-sonnet-5`；
- `claude-haiku-4-5-20251001`。

目录如实记录文本输出、文本与图像输入、工具、structured output、reasoning 以及官方 context/output limits。能力目录主键继续使用 `(provider, model_id, protocol)`。

DeepSeek 的既有模型能力保持不变，并为相同 model ID 增加 `anthropic_messages` 目录项。Provider 目录增加 DeepSeek Anthropic endpoint preset。PyPI 包内 JSON 快照和 GitHub Release 附件使用相同内容与 checksum；框架运行时不联网更新目录。

模型和限制数据以 [Anthropic Models Overview](https://platform.claude.com/docs/en/models/overview)、[Messages API](https://platform.claude.com/docs/en/api/messages/create/) 与 [DeepSeek Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing/) 为依据。

## 测试与交付

实现遵循测试先行，并覆盖：

- transport 抽取前后的 OpenAI-compatible 请求、SSE、TLS、超时、关闭和错误行为一致；
- Anthropic 非流式与流式文本、system、tools、tool choice、tool result、structured output、thinking、usage、错误和不完整流；
- Anthropic `max_output_tokens` 的必填约束和 provider options 严格校验；
- DeepSeek OpenAI-compatible `reasoning_content` 的非流式、流式和工具循环回传；
- DeepSeek Anthropic endpoint 的文本、工具、thinking 和 continuation；
- continuation 的不可变性、codec、持久化、digest、脱敏、retry/reset、direct/managed/fallback 一致性；
- Provider/model 目录 JSON 与打包产物。

完整验证执行：

```text
uv run pytest -q
uv run ruff check src tests examples benchmarks
uv run mypy src benchmarks
uv build
uvx twine check dist/*
```

存在 `ANTHROPIC_API_KEY` 或 `DEEPSEEK_API_KEY` 时执行对应的真实冒烟测试，不输出密钥、thinking 内容或响应正文。

## 实施边界

本次修改只增加 `anthropic_messages`、通用 continuation、DeepSeek 双协议入口及其所必需的 transport/stream 机械重构。实现不引入 Anthropic SDK、公开 Client 基类、Provider registry、远程目录更新、协议自动探测、跨协议 continuation 转换、server tools、prompt caching、beta header 管理、图像 Message、能力路由或 Provider 容量协调器。

新增契约按 born-as-new 方式实现：不增加临时兼容别名、双格式 codec、隐藏默认值或重复执行路径。与此同时，OpenAI-compatible 的公开 client 构造方式、现有模型调用语义、Runtime 行为和非模型功能保持不变。
